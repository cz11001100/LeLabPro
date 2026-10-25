import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Camera, Hand, Loader2, Square, Video, VideoOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
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

// Stretches into whatever box the layout hands over: the camera area is this
// page's flexible region, so the feeds must not pin their own aspect ratio.
const CameraPreview: React.FC<{ name: string; src: string }> = ({ name, src }) => {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return (
    <div className="relative bg-gray-800 rounded-lg border border-gray-700 overflow-hidden w-full aspect-[4/3]">
      {failed ? (
        <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
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
      <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-sm text-gray-200">
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
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
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

  const runTakeover = async (action: "pause" | "correct" | "save" | "stop") => {
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

  if (!status) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to inference…
      </div>
    );
  }
  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isSettingUp = status.inference_active && status.rollout_started_at == null;
  const isRunning = status.inference_active && status.rollout_started_at != null;
  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillLabel = isSettingUp ? "SETTING UP" : isRunning ? "RUNNING" : "FINISHED";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  // The backend serves exactly the cameras this run was started with (keys
  // match the preview files the rollout subprocess writes).
  const cameraFeeds = (status.camera_names ?? []).map((name) => ({
    key: name,
    name,
  }));

  const captureMode = status.capture_mode ?? "base";
  const takeover = status.takeover ?? {};
  const phase = takeover.phase ?? "";
  const capturing = status.inference_active && captureMode !== "base";
  const targetRepo = takeover.repo_id ?? status.dataset_repo_id;
  const episodesSaved = takeover.episodes ?? 0;
  const modeLabel = MODE_LABELS[captureMode] ?? captureMode;
  const gracefulStop = captureMode === "dagger" || captureMode === "highlight";

  return (
    <div className="h-screen bg-black text-white p-6 flex flex-col overflow-hidden">
      <div className="max-w-7xl w-full mx-auto flex-1 min-h-0 flex flex-col">
        <div className="flex items-center gap-3 mb-3 flex-shrink-0">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/")}
            className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <Logo />
          <h1 className="font-bold text-white text-xl">Inference</h1>
          <div className="ml-auto flex items-center gap-4 text-sm text-gray-400">
            <span className="truncate max-w-[320px]" title={status.policy_ref ?? ""}>
              {status.policy_ref ?? "(unknown)"}
            </span>
            <span className="font-mono text-white font-semibold">
              {formatTime(timerSeconds)}
              <span className="text-gray-500"> / {formatTime(duration)}</span>
            </span>
            {cameraFeeds.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCamerasOn((v) => !v)}
                aria-label={camerasOn ? "Hide previews" : "Show previews"}
                className="h-8 w-8 text-gray-400 hover:text-white hover:bg-gray-800"
              >
                {camerasOn ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
              </Button>
            )}
          </div>
        </div>

        <div className="bg-gray-900 rounded-lg border border-gray-700 p-4 flex-1 min-h-0 flex flex-col">
          <div className="flex items-center gap-2 mb-3 flex-shrink-0 text-xs">
            <span
              className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full font-bold tracking-widest ${
                isSettingUp ? "bg-amber-500/15 text-amber-300" : "bg-green-500/15 text-green-300"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isSettingUp ? "bg-amber-500" : "bg-green-500"
                } animate-pulse`}
              />
              {pillLabel}
            </span>
            {capturing && phase && (
              <span
                className={`px-2.5 py-1 rounded-full font-bold ${
                  phase === "correcting" || phase === "recording"
                    ? "bg-blue-500/20 text-blue-300"
                    : "bg-gray-700 text-gray-300"
                }`}
              >
                {PHASE_LABELS[phase] ?? phase}
              </span>
            )}
            {capturing && (
              <span className="text-gray-400 break-all">
                {modeLabel} · {targetRepo ?? "(未命名)"} · 数据集共 {episodesSaved} 集
              </span>
            )}
            {(status.total_runs ?? 0) > 1 && (
              <span className="ml-auto text-gray-400">
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
                  src={`${baseUrl}/inference-camera-feed/${encodeURIComponent(feed.name)}`}
                />
              ))}
            </div>
          ) : (
            <div className="flex-1 min-h-0 flex items-center justify-center text-sm text-gray-500">
              {!status.inference_active
                ? "没有进行中的推理"
                : camerasOn
                  ? "等待摄像头画面…"
                  : "摄像头预览已关闭"}
            </div>
          )}

          <div className="w-full bg-gray-800 rounded-full h-1 my-3 flex-shrink-0">
            <div
              className={`h-1 rounded-full transition-all duration-500 ${
                isSettingUp ? "bg-amber-500/40 animate-pulse w-full" : "bg-green-500"
              }`}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            <Button
              onClick={() => setShowStopConfirm(true)}
              disabled={!status.inference_active}
              className="w-32 bg-red-500 hover:bg-red-600 text-white font-semibold disabled:opacity-50"
            >
              <Square className="w-4 h-4 mr-2" />
              Stop
            </Button>
            <Button
              onClick={handleSnapshot}
              disabled={!isRunning || snapshotLoading}
              variant="outline"
              className="w-36 border-amber-500/50 text-amber-400 hover:bg-amber-500/10 hover:text-amber-300 disabled:opacity-40"
            >
              {snapshotLoading ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Camera className="w-4 h-4 mr-2" />
              )}
              标记失败
            </Button>
            {capturing && captureMode === "dagger" && (
              <>
                <Button
                  onClick={() => runTakeover(phase === "autonomous" ? "pause" : "correct")}
                  disabled={!!pendingAction || !isRunning}
                  className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-semibold disabled:opacity-50"
                >
                  {pendingAction ? (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Hand className="w-4 h-4 mr-2" />
                  )}
                  {phase === "correcting"
                    ? "结束纠正并保存"
                    : phase === "paused"
                      ? "开始纠正（主臂录入）"
                      : "接管（暂停策略）"}
                </Button>
                {phase === "paused" && (
                  <Button
                    onClick={() => runTakeover("pause")}
                    disabled={!!pendingAction}
                    variant="outline"
                    className="w-28 border-gray-600 text-gray-300 hover:bg-gray-800"
                  >
                    交还策略
                  </Button>
                )}
              </>
            )}
            {capturing && captureMode === "highlight" && (
              <Button
                onClick={() => runTakeover("save")}
                disabled={!!pendingAction || !isRunning}
                className="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-semibold disabled:opacity-50"
              >
                {phase === "recording" ? "停止保存" : "保存这一段"}
              </Button>
            )}
            {capturing && gracefulStop && (
              <Button
                onClick={() => runTakeover("stop")}
                disabled={!!pendingAction}
                variant="outline"
                className="w-40 border-gray-600 text-gray-300 hover:bg-gray-800"
              >
                结束采集并保存
              </Button>
            )}
            {capturing && captureMode === "sentry" && (
              <span className="text-xs text-gray-500">Stop 会结束会话并保存已录的集。</span>
            )}
          </div>
        </div>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop inference?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              The follower will hold its current pose. You can launch another
              run from the job tile.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
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
