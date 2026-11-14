import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  CheckCircle2,
  Hand,
  Loader2,
  MoreVertical,
  Play,
  Square,
  Video,
  VideoOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import Logo from "@/components/Logo";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  InferenceStatus,
  captureInferenceSnapshot,
  endInferenceRound,
  getInferenceStatus,
  sendInferenceTakeover,
  stopInference,
} from "@/lib/inferenceApi";

const POLL_MS = 1000;

const MODE_LABELS: Record<string, string> = {
  base: "纯执行（不录数据）",
  dagger: "纠正采集（主臂接管）",
  highlight: "抢救片段（含最近 15 秒）",
  sentry: "全自动录制",
};

const PHASE_LABELS: Record<string, string> = {
  autonomous: "策略执行中",
  paused: "已暂停",
  correcting: "人工纠正中",
  recording: "录制中",
  idle: "待命",
};

// The capture session advances through a single button, like the recording page:
// 修整 → 接管 → 开始 → 结束. Each step owns a colour so the state is readable
// from across the bench without looking at the label.
type CaptureStep = "ready" | "takeover" | "begin" | "correct";

function captureStepOf(sessionPhase: string, phase: string): CaptureStep {
  if (sessionPhase === "ready") return "ready";
  if (phase === "correcting") return "correct";
  if (phase === "paused") return "begin";
  return "takeover";
}

const CAPTURE_STEPS: Record<CaptureStep, { label: string; className: string }> = {
  ready: { label: "Next Episode", className: "bg-blue-600 hover:bg-blue-700" },
  takeover: { label: "Take Over", className: "bg-green-600 hover:bg-green-700" },
  begin: { label: "Begin", className: "bg-amber-500 hover:bg-amber-600" },
  correct: { label: "Finish", className: "bg-red-500 hover:bg-red-600" },
};

// Stretches into whatever box the layout hands over: the camera area is this
// page's flexible region, so the feeds must not pin their own aspect ratio.
const CameraPreview: React.FC<{ name: string; src: string; pending?: boolean }> = ({
  name,
  src,
  pending,
}) => {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return (
    <div className="relative bg-sky-50 rounded-lg border border-sky-300 overflow-hidden w-full aspect-[4/3]">
      {pending ? (
        <div className="w-full h-full flex flex-col items-center justify-center text-slate-600">
          <Loader2 className="w-6 h-6 mb-2 animate-spin" />
          <span className="text-xs">等待画面</span>
        </div>
      ) : failed ? (
        <div className="w-full h-full flex flex-col items-center justify-center text-slate-600">
          <VideoOff className="w-6 h-6 mb-2" />
          <span className="text-xs">No preview</span>
        </div>
      ) : (
        <img
          src={src}
          alt={`${name} live feed`}
          className="w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      )}
      <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-sky-100 text-sm text-slate-800">
        {name}
      </span>
    </div>
  );
};

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

const Inference: React.FC = () => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [camerasOn, setCamerasOn] = useState(true);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [endingRound, setEndingRound] = useState(false);
  const navigatedAwayRef = useRef(false);
  const stopRequestedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await stopInference(baseUrl, fetchWithHeaders);
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        // Auto-bounce home once the run is done; the Deploy card is where the
        // next session (capture mode, dataset, ...) gets configured.
        if (!next.inference_active && !navigatedAwayRef.current) {
          navigatedAwayRef.current = true;
          if (next.exited) {
            const totalRuns = next.total_runs ?? 1;
            const runInfo = totalRuns > 1 ? ` (${totalRuns} runs completed)` : "";
            toast({
              title: "Inference finished",
              description:
                next.exit_code === 0
                  ? `Run completed${runInfo}.`
                  : `Exit code ${next.exit_code}. See ${next.log_path}.`,
              variant: next.exit_code === 0 ? "default" : "destructive",
            });
          }
          navigate("/");
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        // A dagger session is not wall-clock bounded at all (the operator owns
        // the rest screen), so leave it to the per-episode ceiling instead.
        const budgetUsed =
          next.capture_mode && next.capture_mode !== "base"
            ? (next.takeover?.policy_s ?? next.rollout_elapsed_s)
            : next.rollout_elapsed_s;
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.capture_mode !== "dagger" &&
          budgetUsed > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              budgetUsed - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Lost connection to backend",
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, navigate, toast]);

  const handleStop = async () => {
    setShowStopConfirm(false);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and navigate home.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleSnapshot = async () => {
    setSnapshotLoading(true);
    try {
      const result = await captureInferenceSnapshot(baseUrl, fetchWithHeaders);
      toast({
        title: "Snapshot captured",
        description: `Saved ${result.n_ticks} frames to ${result.snapshot_dir}`,
      });
    } catch (e) {
      toast({
        title: "Snapshot failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setSnapshotLoading(false);
    }
  };

  const runTakeover = async (
    action: "pause" | "correct" | "save" | "stop" | "success" | "fail" | "start" | "takeover",
  ) => {
    setPendingAction(action);
    try {
      await sendInferenceTakeover(baseUrl, fetchWithHeaders, action);
    } catch (e) {
      toast({
        title: "接管指令没送达",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setPendingAction(null);
    }
  };

  const handleFail = async () => {
    await handleSnapshot();
    // Dagger: same boundary as 本轮成功, minus the data — the take is dropped
    // and only the failure counter moves.
    if ((status?.capture_mode ?? "base") === "dagger") {
      try {
        await sendInferenceTakeover(baseUrl, fetchWithHeaders, "fail");
        toast({ title: "本集不保存，失败 +1", description: "已回到修整页，摆好场景再开始下一集" });
      } catch (e) {
        toast({
          title: "标记失败",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      }
    }
  };

  const handleSuccess = async () => {
    // Dagger: a take ends here — commit the episode, walk home, next one starts.
    if ((status?.capture_mode ?? "base") === "dagger") {
      setEndingRound(true);
      try {
        await sendInferenceTakeover(baseUrl, fetchWithHeaders, "success");
        toast({ title: "本集已保存", description: "已回到修整页，摆好场景再开始下一集" });
      } catch (e) {
        toast({
          title: "标记失败",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      } finally {
        setEndingRound(false);
      }
      return;
    }
    setEndingRound(true);
    try {
      const res = await endInferenceRound(baseUrl, fetchWithHeaders, true);
      toast({ title: res.message, description: res.advanced ? "正在开下一轮" : "轮数已用完" });
      navigatedAwayRef.current = false;
      stopRequestedRef.current = false;
    } catch (e) {
      toast({
        title: "结束本轮失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setEndingRound(false);
    }
  };

  // Recording-page shortcuts: space = the episode's primary action, ← = mark
  // failed and go back, Esc = close the capture session.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!status?.inference_active) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const mode = status.capture_mode ?? "base";
      if (mode === "dagger") {
        const session = status.takeover;
        if (!session?.session_phase) return;
        if (e.code === "Space") {
          e.preventDefault();
          const step = captureStepOf(session.session_phase ?? "", session.phase ?? "");
          if (step === "ready") void runTakeover("start");
          else if (step === "takeover") void runTakeover("takeover");
          else if (step === "begin") void runTakeover("correct");
          else void handleSuccess();
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          void handleFail();
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          void handleSuccess();
        } else if (e.key === "Escape") {
          e.preventDefault();
          void runTakeover("stop");
        }
        return;
      }
      if (mode === "base") {
        if (e.code === "Space") {
          e.preventDefault();
          void handleSuccess();
        } else if (e.key === "ArrowLeft") {
          e.preventDefault();
          void handleFail();
        } else if (e.key === "Escape") {
          e.preventDefault();
          setShowStopConfirm(true);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [status, runTakeover, handleSuccess, handleFail]);

  if (!status) {
    return (
      <div className="min-h-screen bg-sky-50 text-slate-900 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to inference…
      </div>
    );
  }
  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isSettingUp = status.inference_active && status.rollout_started_at == null;
  const isRunning = status.inference_active && status.rollout_started_at != null;
  const captureMode = status.capture_mode ?? "base";
  const takeover = status.takeover ?? {};
  const phase = takeover.phase ?? "";
  const capturing = status.inference_active && captureMode !== "base";
  // LeForge's capture session phase: 'ready' is the rest screen between episodes
  // (arm parked at its start pose, policy silent), 'running' an open episode.
  const sessionPhase = takeover.session_phase ?? "";
  const correcting = phase === "correcting";
  const capturingDagger = capturing && captureMode === "dagger";
  const step = capturingDagger ? captureStepOf(sessionPhase, phase) : null;
  const handleStep = () => {
    if (step === "ready") void runTakeover("start");
    else if (step === "takeover") void runTakeover("takeover");
    else if (step === "begin") void runTakeover("correct");
    else if (step === "correct") void handleSuccess();
  };
  // RA-BC weights a frame by the progress gained over one policy chunk, so an
  // episode shorter than chunk_size carries no per-frame signal.
  const shortEpisode =
    !!takeover.rabc_chunk_size &&
    (takeover.last_episode_frames ?? Number.MAX_SAFE_INTEGER) < takeover.rabc_chunk_size;
  // A dagger session is bounded per episode (弹窗的"时长"= 每集上限) and by
  // 段数上限, not by wall clock; the other modes budget policy driving time.
  const budgetElapsed = capturingDagger
    ? (takeover.episode_elapsed_s ?? 0)
    : capturing
      ? (takeover.policy_s ?? rolloutElapsed)
      : rolloutElapsed;
  const pct =
    isRunning && duration > 0 ? Math.min(100, (budgetElapsed / duration) * 100) : 0;
  const pillLabel = isSettingUp
    ? "SETTING UP"
    : step === "ready"
      ? "READY — 等待开始"
      : step === "takeover"
        ? "本集进行中"
        : step === "begin"
          ? "已接管 · 按开始"
          : step === "correct"
            ? "纠正中"
            : isRunning
              ? "RUNNING"
              : "FINISHED";
  const pillTone = isSettingUp
    ? "bg-amber-100 text-amber-700"
    : step === "ready"
      ? "bg-blue-100 text-blue-700"
      : step === "begin"
        ? "bg-amber-100 text-amber-700"
        : step === "correct"
          ? "bg-red-100 text-red-700"
          : "bg-green-100 text-green-700";
  const timerSeconds = isRunning ? budgetElapsed : setupElapsed;

  // The backend serves exactly the cameras this run was started with (keys
  // match the preview files the rollout subprocess writes).
  const cameraFeeds = (status.camera_names ?? []).map((name) => ({
    key: name,
    name,
  }));

  const targetRepo = takeover.repo_id ?? status.dataset_repo_id;
  const episodesSaved = takeover.episodes ?? 0;
  const modeLabel = MODE_LABELS[captureMode] ?? captureMode;
  const gracefulStop = captureMode === "dagger" || captureMode === "highlight";
  const failures = takeover.failures ?? 0;
  const takeStats = status.take_stats;

  return (
    <div className="h-screen bg-sky-50 text-slate-900 p-6 flex flex-col overflow-hidden">
      <div className="max-w-7xl w-full mx-auto flex-1 min-h-0 flex flex-col">
        <div className="flex items-center gap-3 mb-3 flex-shrink-0">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/")}
            className="text-slate-600 hover:bg-sky-200 hover:text-slate-900 rounded-lg"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <Logo />
          <h1 className="font-bold text-slate-900 text-xl">Inference</h1>
          <div className="ml-auto flex items-center gap-4 text-sm text-slate-600">
            <span className="truncate max-w-[320px]" title={status.policy_ref ?? ""}>
              {status.policy_ref ?? "(unknown)"}
            </span>
            <span className="font-mono text-slate-900 font-semibold">
              {formatTime(timerSeconds)}
              <span className="text-slate-600"> / {formatTime(duration)}</span>
            </span>
            {status.inference_active && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    aria-label="会话操作"
                    className="h-8 w-8 text-slate-600 hover:bg-sky-200 hover:text-slate-900"
                  >
                    <MoreVertical className="w-5 h-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent
                  align="end"
                  onCloseAutoFocus={(e) => e.preventDefault()}
                  className="bg-sky-100 border-sky-300 text-slate-900"
                >
                  {gracefulStop && (
                    <DropdownMenuItem
                      onClick={() => runTakeover("stop")}
                      className="focus:bg-sky-50 focus:text-slate-900"
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      End Capture & Save
                      <span className="ml-2 font-mono text-xs text-slate-600">Esc</span>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuItem
                    onClick={() => setShowStopConfirm(true)}
                    className="text-red-700 focus:bg-sky-50 focus:text-red-700"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Stop
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {cameraFeeds.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCamerasOn((v) => !v)}
                aria-label={camerasOn ? "Hide previews" : "Show previews"}
                className="h-8 w-8 text-slate-600 hover:text-slate-900 hover:bg-sky-200"
              >
                {camerasOn ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
              </Button>
            )}
          </div>
        </div>

        <div className="bg-sky-100 rounded-lg border border-sky-300 p-4 flex-1 min-h-0 flex flex-col">
          <div className="flex items-center gap-2 mb-3 flex-shrink-0 text-xs">
            <span
              className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full font-bold tracking-widest ${pillTone}`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isSettingUp ? "bg-amber-500" : "bg-current"
                } animate-pulse`}
              />
              {pillLabel}
            </span>
            {capturing && phase && (
              <span
                className={`px-2.5 py-1 rounded-full font-bold ${
                  phase === "correcting" || phase === "recording"
                    ? "bg-blue-100 text-blue-700"
                    : "bg-sky-200 text-slate-700"
                }`}
              >
                {PHASE_LABELS[phase] ?? phase}
              </span>
            )}
            {capturing && (
              <span className="text-slate-600 break-all">
                {modeLabel} · {targetRepo ?? "(未命名)"} · 接管{" "}
                {takeStats ? `${takeStats.corrected_takes}/${takeStats.takes}` : "0/0"}
                {takeStats?.takeover_rate != null
                  ? `（${Math.round(takeStats.takeover_rate * 100)}%）`
                  : ""}{" "}
                · 失败 {failures}
              </span>
            )}
            {capturing && (status.episode_target ?? 0) > 0 ? (
              <span className="ml-auto text-slate-600">
                第 {episodesSaved} / {status.episode_target} 集
              </span>
            ) : null}
            {capturing && shortEpisode ? (
              <span className="text-amber-700">
                本集 {takeover.last_episode_frames} 帧 &lt; RA-BC 需要 {takeover.rabc_chunk_size}
              </span>
            ) : null}
            {!capturing && (status.total_runs ?? 0) > 1 && (
              <span className="ml-auto text-slate-600">
                Run {status.current_run ?? 1} / {status.total_runs}
              </span>
            )}
          </div>

          {/* Cameras are the point of this page, so they take every pixel the
              chrome leaves. Sessions are launched from the Deploy card. */}
          {cameraFeeds.length > 0 && camerasOn ? (
            <div
              className="flex-1 min-h-0 grid content-center gap-3"
              style={{
                gridTemplateColumns: `repeat(${Math.min(cameraFeeds.length, 3)}, minmax(0, 1fr))`,
              }}
            >
              {cameraFeeds.map((feed) => (
                <CameraPreview
                  key={feed.key}
                  name={feed.name}
                  pending={isSettingUp}
                  src={`${baseUrl}/inference-camera-feed/${encodeURIComponent(feed.name)}`}
                />
              ))}
            </div>
          ) : (
            <div className="flex-1 min-h-0 flex items-center justify-center text-sm text-slate-600">
              {!status.inference_active
                ? "没有进行中的推理"
                : camerasOn
                  ? "等待摄像头画面…"
                  : "摄像头预览已关闭"}
            </div>
          )}

          <div className="w-full bg-sky-50 rounded-full h-1 my-3 flex-shrink-0">
            <div
              className={`h-1 rounded-full transition-all duration-500 ${
                isSettingUp ? "bg-amber-100 animate-pulse w-full" : "bg-green-500"
              }`}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            {step ? (
              <div className="flex w-full items-center gap-3">
                <Button
                  onClick={handleFail}
                  disabled={!isRunning || step === "ready" || snapshotLoading}
                  variant="outline"
                  className="w-44 h-12 border-red-500/50 text-red-700 hover:bg-red-100 hover:text-red-700 disabled:opacity-40"
                  title="丢弃本集，重录这一集（←）"
                >
                  <ArrowLeft className="w-5 h-5 mr-2" />
                  Defeat
                </Button>
                <Button
                  onClick={handleStep}
                  disabled={!isRunning || !!pendingAction}
                  className={`flex-1 h-12 text-base font-semibold text-slate-900 disabled:opacity-50 ${
                    CAPTURE_STEPS[step].className
                  }`}
                >
                  {pendingAction ? (
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  ) : step === "takeover" ? (
                    <Hand className="w-5 h-5 mr-2" />
                  ) : step === "correct" ? (
                    <CheckCircle2 className="w-5 h-5 mr-2" />
                  ) : (
                    <Play className="w-5 h-5 mr-2" />
                  )}
                  {CAPTURE_STEPS[step].label}
                  <span className="ml-3 rounded bg-sky-100 px-2 py-0.5 font-mono text-xs">
                    Space
                  </span>
                </Button>
                <Button
                  onClick={handleSuccess}
                  disabled={!isRunning || step === "ready" || !!endingRound}
                  variant="outline"
                  className="w-44 h-12 border-green-500/50 text-green-700 hover:bg-green-100 hover:text-green-700 disabled:opacity-40"
                  title="保存本集，准备下一集（→）"
                >
                  Success
                  <ArrowRight className="w-5 h-5 ml-2" />
                </Button>
              </div>
            ) : captureMode === "base" ? (
              <>
                <Button
                  onClick={handleFail}
                  disabled={!isRunning || snapshotLoading}
                  variant="outline"
                  className="w-44 h-12 border-red-500/50 text-red-700 hover:bg-red-100 hover:text-red-700 disabled:opacity-40"
                  title="存档失败现场（最近 30 帧观测与预测动作），不结束本轮（←）"
                >
                  <ArrowLeft className="w-5 h-5 mr-2" />
                  Defeat
                </Button>
                <Button
                  onClick={handleSuccess}
                  disabled={!isRunning || endingRound || !!status.switching_round}
                  className="flex-1 h-12 bg-green-600 hover:bg-green-700 text-base font-semibold text-white disabled:opacity-50"
                >
                  {endingRound || status.switching_round ? (
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle2 className="w-5 h-5 mr-2" />
                  )}
                  {status.switching_round ? "Starting Next Round…" : "Run Succeeded"}
                  <span className="ml-3 rounded bg-sky-100 px-2 py-0.5 font-mono text-xs">
                    Space
                  </span>
                </Button>
              </>
            ) : (
              <>
                <Button
                  onClick={() => setShowStopConfirm(true)}
                  disabled={!status.inference_active}
                  className="w-32 bg-red-500 hover:bg-red-600 text-white font-semibold disabled:opacity-50"
                >
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
                <Button
                  onClick={handleFail}
                  disabled={!isRunning || snapshotLoading}
                  variant="outline"
                  className="w-36 border-amber-500/50 text-amber-700 hover:bg-amber-100 hover:text-amber-700 disabled:opacity-40"
                  title="存档失败现场（最近 30 帧观测与预测动作）"
                >
                  {snapshotLoading ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Camera className="w-4 h-4 mr-2" />
                  )}
                  Mark Failed
                </Button>
                <Button
                  onClick={handleSuccess}
                  disabled={!isRunning || endingRound || !!status.switching_round}
                  className="w-40 bg-green-600 hover:bg-green-700 text-white font-semibold disabled:opacity-50"
                >
                  {endingRound || status.switching_round ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 mr-2" />
                  )}
                  {status.switching_round ? "Starting Next Round…" : "Run Succeeded"}
                </Button>
                {capturing && captureMode === "highlight" && (
                  <Button
                    onClick={() => runTakeover("save")}
                    disabled={!!pendingAction || !isRunning}
                    className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-semibold disabled:opacity-50"
                  >
                    {phase === "recording" ? "Stop & Save" : "Save Segment"}
                  </Button>
                )}
                {capturing && captureMode === "sentry" && (
                  <span className="text-xs text-slate-600">Stop 会结束会话并保存已录的集。</span>
                )}
              </>
            )}

          </div>
        </div>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent className="bg-sky-100 border-sky-300 text-slate-900">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop inference?</AlertDialogTitle>
            <AlertDialogDescription className="text-slate-600">
              The follower will hold its current pose. You can launch another
              run from the job tile.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-sky-50 border-sky-300 text-slate-900 hover:bg-sky-200">
              Keep running
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleStop}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              Stop
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Inference;
