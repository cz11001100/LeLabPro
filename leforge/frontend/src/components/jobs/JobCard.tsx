import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { JobRecord, ValidationState, pushModelToHub, getPushStatus } from "@/lib/jobsApi";
import { modelStepOf, modelBaseName } from "@/lib/modelGroups";
import {
  Square,
  X,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  XCircle,
  ExternalLink,
  Play,
  Download,
  RotateCcw,
  ShieldCheck,
  ShieldAlert,
  FlaskConical,
  CloudUpload,
  Boxes,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { useToast } from "@/hooks/use-toast";
import {
  JobCheckpoint,
  listJobCheckpoints,
  getExportProgress,
  ExportProgress,
  cancelJobExport,
} from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";

// The registry lists an imported model twice once it is also on the Hub: the
// local file and the Hub repo, both at the same step. The dropdown must show
// one row per step, preferring the local copy.
const dedupeCheckpoints = (cks: JobCheckpoint[]): JobCheckpoint[] => {
  const byStep = new Map<number, JobCheckpoint>();
  for (const c of cks) {
    const prev = byStep.get(c.step);
    if (!prev || (prev.source !== "local" && c.source === "local")) {
      byStep.set(c.step, c);
    }
  }
  return Array.from(byStep.values()).sort((a, b) => a.step - b.step);
};

interface Props {
  job: JobRecord;
  // Imported models downloaded from the same training run are grouped into one
  // card; `group` holds every member (including `job`, the representative).
  // Selecting a checkpoint swaps the active member for validation/inference.
  group?: JobRecord[];
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
  // The flask opens the evaluation dialog; starting and stopping a run lives
  // there, so the card stays a pure entry point.
  onEvaluate: (job: JobRecord) => void;
  // Same dialog pattern as evaluation: the button opens the sim-eval dialog,
  // which owns starting and stopping the run.
  onSimEval: (job: JobRecord) => void;
  onExport?: (job: JobRecord, step: number) => void | Promise<boolean>;
  onResume?: (job: JobRecord) => void;
}

function formatBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.round(n / 1e3)} KB`;
}

function relativeTime(epochSec: number): string {
  const diff = Math.max(0, Date.now() / 1000 - epochSec);
  if (diff < 60) return `${Math.floor(diff)} 秒前`;
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  return `${Math.floor(diff / 86400)} 天前`;
}

const statePresentation: Record<
  JobRecord["state"],
  { label: string; color: string; Icon: React.ComponentType<{ className?: string }> }
> = {
  running: { label: "运行中", color: "text-green-700", Icon: Loader2 },
  done: { label: "已完成", color: "text-slate-600", Icon: CheckCircle2 },
  failed: { label: "失败", color: "text-red-700", Icon: XCircle },
  interrupted: { label: "已中断", color: "text-amber-700", Icon: AlertTriangle },
};

const JobCard: React.FC<Props> = ({ job: jobProp, group, onStop, onDelete, onPlay, onEvaluate, onSimEval, onExport, onResume }) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { toast } = useToast();
  // When grouped, the card represents one training run and `job` is the member
  // whose checkpoint is currently selected; otherwise it is the single job.
  const members = group && group.length > 1 ? group : null;
  const [activeId, setActiveId] = useState<string>(jobProp.id);
  const job = members
    ? members.find((m) => m.id === activeId) ?? members[0]
    : jobProp;
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  const isStarting = isRunning && job.metrics.total_steps === 0;
  const progressPct =
    job.metrics.total_steps > 0
      ? Math.min(100, (job.metrics.current_step / job.metrics.total_steps) * 100)
      : 0;

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  useEffect(() => {
    if (job.checkpoint_count <= 0) {
      setCheckpoints([]);
      setSelectedStep(null);
      return;
    }
    let cancelled = false;
    listJobCheckpoints(baseUrl, fetchWithHeaders, job.id)
      .then((raw) => {
        if (cancelled) return;
        const cks = dedupeCheckpoints(raw);
        setCheckpoints(cks);
        if (cks.length > 0) {
          const latest = cks[cks.length - 1].step;
          setSelectedStep((prev) =>
            prev != null && cks.some((c) => c.step === prev) ? prev : latest,
          );
        } else {
          setSelectedStep(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setCheckpoints([]);
          setSelectedStep(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, job.id, job.checkpoint_count]);

  const [armAction, setArmAction] = useState(false);
  const armTimer = useRef<number | null>(null);
  const disarm = () => {
    if (armTimer.current != null) window.clearTimeout(armTimer.current);
    armTimer.current = null;
    setArmAction(false);
  };

  const handleAction = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!armAction) {
      // Inline two-click confirm (no browser popup): first click arms,
      // second click within 3s performs the destructive action.
      setArmAction(true);
      if (armTimer.current != null) window.clearTimeout(armTimer.current);
      armTimer.current = window.setTimeout(() => setArmAction(false), 3000);
      return;
    }
    disarm();
    if (isRunning) {
      onStop(job.id);
    } else {
      onDelete(job.id);
    }
  };

  const [pushing, setPushing] = useState(false);
  const [pushElapsed, setPushElapsed] = useState(0);
  const [pushInterrupted, setPushInterrupted] = useState(false);
  const [tokenDialogOpen, setTokenDialogOpen] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [pushedRepo, setPushedRepo] = useState<string | null>(null);
  const [pushError, setPushError] = useState<string | null>(null);
  // 上传成功后直接把状态标签显示为“已上传”，与导入后的“已导入”用同一套样式
  const stateLabel = isImported ? "已导入" : pushedRepo ? "已上传" : present.label;

  // The push runs in a server thread, so a reloaded page re-attaches to it
  // instead of showing an idle button. Only the card that owns the job reacts.
  useEffect(() => {
    let cancelled = false;
    getPushStatus(baseUrl, fetchWithHeaders)
      .then((st) => {
        if (cancelled || st.job_id !== job.id) return;
        if (st.running) {
          setPushing(true);
          const startedAt = st.elapsed_s ? Date.now() - st.elapsed_s * 1000 : Date.now();
          pollPush(startedAt, false);
        } else if (st.interrupted) {
          setPushInterrupted(true);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id]);

  const pollPush = (startedAt: number, notify: boolean) => {
    const timer = window.setInterval(async () => {
      setPushElapsed(Math.round((Date.now() - startedAt) / 1000));
      try {
        const st = await getPushStatus(baseUrl, fetchWithHeaders);
        if (st.running) return;
        window.clearInterval(timer);
        setPushing(false);
        if (st.done) {
          setPushedRepo(st.repo_id ?? null);
          setPushError(null);
        } else {
          setPushError(st.error || "服务端没有给出原因，看 ~/leforge-server.log");
        }
        if (notify) {
          if (st.done) toast({ title: `已上传到 ${st.repo_id}` });
          else
            toast({
              title: "上传失败",
              description: st.error || "服务端没有给出原因，看 ~/leforge-server.log",
              variant: "destructive",
            });
        }
      } catch {
        // keep polling
      }
    }, 3000);
  };

  // Show an upload that finished before this card mounted (reload, other tab),
  // or re-attach when the upload still belongs to this job.
  useEffect(() => {
    let cancelled = false;
    getPushStatus(baseUrl, fetchWithHeaders)
      .then((st) => {
        if (cancelled) return;
        const repo = st.by_job?.[job.id];
        if (repo) setPushedRepo(repo);
        if (st.running && st.job_id === job.id) {
          setPushing(true);
          pollPush(Date.now() - (st.elapsed_s ?? 0) * 1000, true);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // pollPush is stable in behaviour; re-running on id change is wrong.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, baseUrl, fetchWithHeaders]);

  // Pushing to the Hub is what lets a pod *pull* the checkpoint through the
  // mirror instead of receiving a 1.2 GB upload before every remote run.
  // SSH jobs upload straight from the pod — no local download needed.
  const doPush = async (token?: string) => {
    if (pushing || selectedStep == null) return;
    setPushing(true);
    setPushElapsed(0);
    setPushInterrupted(false);
    setPushError(null);
    const startedAt = Date.now();
    const slug = job.name
      .replace(/[^A-Za-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase();
    const repoId =
      auth.status === "authenticated" && slug ? `${auth.username}/${slug}` : "";
    try {
      const res = await pushModelToHub(baseUrl, fetchWithHeaders, {
        job_id: job.id,
        step: selectedStep,
        repo_id: repoId,
        token,
      });
      toast({ title: res.message, description: res.repo_id });
      setTokenDialogOpen(false);
      setHfToken("");
      pollPush(startedAt, true);
    } catch (err) {
      setPushing(false);
      const message = err instanceof Error ? err.message : String(err);
      // No HF session on this machine: ask for a write token instead of
      // failing with a raw 401.
      if (/401|令牌|token/i.test(message) && !token) {
        setTokenDialogOpen(true);
        return;
      }
      toast({ title: "上传没起来", description: message, variant: "destructive" });
    }
  };

  const handlePushHub = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (pushing || selectedStep == null) return;
    doPush();
  };

  const handleTokenSubmit = () => {
    if (!hfToken.trim()) return;
    doPush(hfToken.trim());
  };

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(job, selectedStep);
  };

  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);

  // Re-attach to an export that is already running server-side (page reload,
  // second tab, or the other tab's POST still in flight). Without this the
  // card looks idle and a second click is rejected as a duplicate.
  useEffect(() => {
    if (isRunning || selectedStep == null) return;
    let cancelled = false;
    getExportProgress(baseUrl, fetchWithHeaders, job.id, selectedStep)
      .then((p) => {
        if (!cancelled && (p.state === "preparing" || p.state === "downloading")) {
          setExporting(true);
          setExportProgress(p);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [job.id, selectedStep, baseUrl, fetchWithHeaders, isRunning]);

  // The export POST blocks until the SFTP transfer completes, so bytes
  // downloaded have to come from a side channel while that request is live.
  useEffect(() => {
    if (!exporting || selectedStep == null) return;
    const step = selectedStep;
    const timer = window.setInterval(() => {
      getExportProgress(baseUrl, fetchWithHeaders, job.id, step)
        .then((p) => {
          if (p.state !== "none") setExportProgress(p);
        })
        .catch(() => {});
    }, 600);
    return () => window.clearInterval(timer);
  }, [exporting, selectedStep, job.id, baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!exportProgress || exporting) return;
    const ms = exportProgress.state === "failed" ? 12000 : 5000;
    const timer = window.setTimeout(() => setExportProgress(null), ms);
    return () => window.clearTimeout(timer);
  }, [exportProgress, exporting]);

  const handleExport = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null || !onExport || exporting) return;
    const step = selectedStep;
    setExporting(true);
    setExportProgress({ state: "preparing", transferred_bytes: 0, total_bytes: 0, current_file: "" });
    // Another tab may already own this export; the server rejects duplicates,
    // so attach to the running transfer instead of surfacing an error.
    const running = await getExportProgress(baseUrl, fetchWithHeaders, job.id, step).catch(() => null);
    if (running && (running.state === "preparing" || running.state === "downloading")) {
      setExportProgress(running);
      return;
    }
    try {
      const ok = await onExport(job, step);
      // The server flips to "done" just before the POST returns, so one final
      // read can still be lagging; fall back to the request outcome.
      const final = await getExportProgress(baseUrl, fetchWithHeaders, job.id, step).catch(() => null);
      if (final && (final.state === "done" || final.state === "failed")) {
        setExportProgress(final);
      } else if (ok !== false) {
        setExportProgress({
          state: "done",
          transferred_bytes: final?.total_bytes ?? 0,
          total_bytes: final?.total_bytes ?? 0,
          current_file: "",
        });
      } else {
        setExportProgress({
          state: "failed",
          transferred_bytes: 0,
          total_bytes: 0,
          current_file: "",
          error: "导出失败",
        });
      }
    } finally {
      setExporting(false);
    }
  };

  // Validation is only offered on imported models, not on training jobs
  // (local/ssh/cloud) — per user request, jobs don't get a validate button.
  const canValidate = !isRunning && job.checkpoint_count > 0 && job.runner === "imported";
  const vState: ValidationState = job.validation_state ?? "none";
  const vResults = job.validation_results;

  const handleCancelExport = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    try {
      await cancelJobExport(baseUrl, fetchWithHeaders, job.id, selectedStep);
      toast({ title: "正在取消下载…" });
    } catch (err) {
      toast({
        title: "取消失败",
        description: err instanceof Error ? err.message : String(err),
        variant: "destructive",
      });
    }
  };

  const showProgressBar = isRunning;
  const showInferenceRow = checkpoints.length > 0 && selectedStep != null;
  const isSSH = job.runner === "ssh";
  // Persistent "downloaded to local" state for the selected checkpoint, read
  // from the source job record so it survives reloads and server restarts.
  const isDownloaded =
    isSSH && selectedStep != null && (job.exported_steps ?? []).includes(selectedStep);
  // Grouped imported run: collapse the " (step N)" suffix into a dropdown that
  // switches the active member. Each member is a separate imported job.
  const displayName = members ? modelBaseName(job.name) : job.name;
  const groupCheckpoints = members
    ? members.map((m) => ({ step: modelStepOf(m.name), source: "local" as const, ref: m.id }))
    : [];
  const groupSelectedStep = members ? modelStepOf(job.name) : null;
  const selectGroupStep = (step: number) => {
    const target = members?.find((m) => modelStepOf(m.name) === step);
    if (target) setActiveId(target.id);
  };
  const exportPct =
    !exportProgress || exportProgress.state === "failed"
      ? 0
      : exportProgress.state === "done"
      ? 100
      : exportProgress.total_bytes > 0
      ? Math.min(100, (exportProgress.transferred_bytes / exportProgress.total_bytes) * 100)
      : 0;
  const exportHeadline = !exportProgress
    ? ""
    : exportProgress.state === "preparing"
    ? "正在连接远端…"
    : exportProgress.state === "downloading"
    ? exportProgress.current_file
      ? `下载 ${exportProgress.current_file}`
      : "正在下载模型…"
    : exportProgress.state === "done"
    ? "模型已下载到本地"
    : exportProgress.error || "下载失败";
  const canResume = !isRunning && !isImported && job.runner !== "hf_cloud" && checkpoints.length > 0 && onResume;
  const runnerChip = isSSH
    ? { label: "SSH 云端", cls: "bg-blue-100 text-blue-700 border-blue-300" }
    : job.runner === "hf_cloud"
    ? { label: "HF 云端", cls: "bg-amber-100 text-amber-700 border-amber-300" }
    : { label: "本机", cls: "bg-sky-200/60 text-slate-700 border-sky-300" };

  return (
    <Card
      onClick={() => navigate(`/training/${job.id}`)}
      className={`bg-sky-50/70 border-sky-300 rounded-xl transition-colors cursor-pointer hover:border-sky-400`}
    >
      <CardContent className="p-4 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <div className={`flex items-center gap-1.5 text-xs font-semibold ${present.color}`}>
              <Icon className={`w-3.5 h-3.5 ${isRunning ? "animate-spin" : ""}`} />
              {stateLabel}
              {!isImported ? (
                <span className={`text-[10px] font-normal px-1.5 py-0.5 rounded border ${runnerChip.cls}`}>
                  {runnerChip.label}
                </span>
              ) : null}
            </div>
            {canValidate && vState === "done" && vResults ? (
              <div className="flex items-center gap-1 text-xs text-purple-700">
                <ShieldCheck className="w-3 h-3" />
                <span className="font-medium">验证通过</span>
              </div>
            ) : canValidate && vState === "failed" ? (
              <div className="flex items-center gap-1 text-xs text-red-700">
                <ShieldAlert className="w-3 h-3" />
                <span className="font-medium">验证失败</span>
              </div>
            ) : null}
          </div>
          {job.runner === "hf_cloud" && job.hf_job_url ? (
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-7 w-7 text-slate-600 hover:text-slate-900"
              aria-label="Open Hub job page"
            >
              <a
                href={job.hf_job_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </Button>
          ) : armAction ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleAction}
              className="h-7 px-2 text-red-700 hover:text-red-700"
            >
              {isRunning ? "Confirm Stop?" : "Confirm Delete?"}
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleAction}
              className="h-7 w-7 text-slate-600 hover:text-slate-900"
              aria-label={isRunning ? "Stop task" : "Delete task"}
              title={isRunning ? "Stop task" : "Delete task"}
            >
              {isRunning ? <Square className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
            </Button>
          )}
        </div>
        <div>
          <div className="text-slate-900 font-semibold truncate" title={displayName}>
            {displayName}
          </div>
        </div>
        {showProgressBar ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-sky-100 border border-sky-300">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-sky-400 transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-slate-900 tabular-nums drop-shadow">
              {isStarting ? "训练启动中…" : `${progressPct.toFixed(1)}%`}
            </div>
          </div>
        ) : null}
        {showInferenceRow || canValidate || members || isSSH ? (
          <div className="flex items-center gap-2">
            {members ? (
              <CheckpointDropdown
                checkpoints={groupCheckpoints}
                selectedStep={groupSelectedStep}
                onChange={selectGroupStep}
              />
            ) : showInferenceRow || isSSH ? (
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
                placeholder={isSSH && checkpoints.length === 0 ? "no SSH..." : undefined}
              />
            ) : null}
            {canResume || isSSH ? (
              <Button
                size="icon"
                onClick={canResume ? (e) => { e.stopPropagation(); onResume!(job); } : undefined}
                disabled={!canResume}
                className={`h-8 w-8 ${
                  canResume
                    ? "bg-amber-500 hover:bg-amber-600 text-white"
                    : "bg-sky-200 text-slate-600 cursor-not-allowed"
                }`}
                aria-label="Resume training from latest checkpoint"
                title={canResume ? "Resume from latest checkpoint" : "No remote checkpoint detected"}
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            ) : null}
            {isSSH && onExport ? (
              <Button
                size="icon"
                onClick={exporting ? handleCancelExport : handleExport}
                disabled={!exporting && !showInferenceRow}
                className={`h-8 w-8 text-slate-900 ${
                  exporting
                    ? "bg-red-500 hover:bg-red-600"
                    : !showInferenceRow
                    ? "bg-sky-200 text-slate-600 cursor-not-allowed"
                    : isDownloaded
                    ? "bg-green-600 hover:bg-green-500"
                    : "bg-blue-500 hover:bg-blue-600"
                }`}
                aria-label={exporting ? "Cancel download" : isDownloaded ? "Re-download to local" : "Download checkpoint to local"}
                title={
                  exporting
                    ? "Downloading — click to cancel"
                    : !showInferenceRow
                    ? "No remote checkpoint detected"
                    : isDownloaded
                    ? "Downloaded to local (click to re-download)"
                    : "Download to local"
                }
              >
                {exporting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : isDownloaded ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
              </Button>
            ) : null}
            {isImported && showInferenceRow ? (
              <Button
                size="icon"
                onClick={handlePlay}
                className="h-8 w-8 bg-green-600 hover:bg-green-700 text-white"
                aria-label="Run inference with this checkpoint"
                title="Run on robot"
              >
                <Play className="w-4 h-4" />
              </Button>
            ) : null}
            {canValidate ? (
              <Button
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  onEvaluate(job);
                }}
                className={`h-8 w-8 text-slate-900 ${
                  vState === "running"
                    ? "bg-red-500 hover:bg-red-600"
                    : "bg-purple-500 hover:bg-purple-600"
                }`}
                aria-label="Open evaluation"
                title={
                  vState === "running"
                    ? "评测进行中 — 点击查看"
                    : vState === "done"
                      ? "View evaluation results"
                      : "Evaluate model"
                }
              >
                {vState === "running" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <FlaskConical className="w-4 h-4" />
                )}
              </Button>
            ) : null}
            {canValidate ? (
              <Button
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  onSimEval(job);
                }}
                className={`h-8 w-8 text-slate-900 ${
                  job.sim_eval_state === "running"
                    ? "bg-amber-600 hover:bg-amber-700"
                    : "bg-teal-600 hover:bg-teal-700"
                }`}
                aria-label="Open simulation evaluation"
                title={
                  job.sim_eval_state === "running"
                    ? "仿真评测进行中 — 点击查看"
                    : job.sim_eval_state === "done"
                      ? "查看仿真评测结果"
                      : "Evaluate in simulation"
                }
              >
                {job.sim_eval_state === "running" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Boxes className="w-4 h-4" />
                )}
              </Button>
            ) : null}
            {(isImported || isSSH) && showInferenceRow ? (
              <Button
                size="icon"
                onClick={handlePushHub}
                disabled={pushing || selectedStep == null}
                className={`h-8 w-8 text-slate-900 ${
                  pushInterrupted && !pushing
                    ? "bg-amber-600 hover:bg-amber-700"
                    : "bg-sky-600 hover:bg-sky-700"
                }`}
                aria-label="Upload this checkpoint to the Hub"
                title={
                  pushing
                    ? `上传中 ${pushElapsed}s`
                    : isSSH
                      ? "从云端直接上传到 Hub（不经过本地）"
                      : pushInterrupted
                        ? "上次上传被中断（服务重启），可重试"
                        : "Upload to Hub"
                }
              >
                {pushing ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <CloudUpload className="w-4 h-4" />
                )}
              </Button>
            ) : null}
          </div>
        ) : null}

        {exportProgress ? (
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-xs text-slate-700">
              <span className="flex items-center gap-1.5 min-w-0">
                {exportProgress.state === "done" ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-green-700 shrink-0" />
                ) : exportProgress.state === "failed" ? (
                  <XCircle className="w-3.5 h-3.5 text-red-700 shrink-0" />
                ) : (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-700 shrink-0" />
                )}
                <span className="truncate" title={exportProgress.current_file || undefined}>
                  {exportHeadline}
                </span>
              </span>
              {exportProgress.total_bytes > 0 ? (
                <span className="tabular-nums shrink-0">
                  {formatBytes(exportProgress.transferred_bytes)} /{" "}
                  {formatBytes(exportProgress.total_bytes)}
                </span>
              ) : null}
            </div>
            <div className="relative h-5 w-full overflow-hidden rounded-md bg-sky-100 border border-sky-300">
              <div
                className={`h-full transition-[width] duration-300 ${
                  exportProgress.state === "failed"
                    ? "bg-red-100"
                    : "bg-gradient-to-r from-blue-500 to-sky-400"
                }`}
                style={{ width: `${exportPct}%` }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-slate-900 tabular-nums drop-shadow">
                {exportProgress.state === "done"
                  ? "下载完成"
                  : exportProgress.state === "failed"
                  ? "下载失败"
                  : exportProgress.total_bytes > 0
                  ? `${exportPct.toFixed(1)}%`
                  : "准备中…"}
              </div>
            </div>
          </div>
        ) : null}

        {pushError ? (
          <div className="text-xs text-red-700 break-all">上传失败：{pushError}</div>
        ) : null}
      </CardContent>

      <Dialog
        open={tokenDialogOpen}
        onOpenChange={(open) => {
          if (!pushing) setTokenDialogOpen(open);
        }}
      >
        <DialogContent
          className="bg-sky-100 border-sky-300 text-slate-900"
          onClick={(e) => e.stopPropagation()}
        >
          <DialogHeader>
            <DialogTitle>需要 Hugging Face 写入令牌</DialogTitle>
            <DialogDescription className="text-slate-600">
              本机还没有登录 Hugging Face。粘贴一个带 write 权限的令牌，只用于本次上传。
            </DialogDescription>
          </DialogHeader>
          <Input
            type="password"
            value={hfToken}
            onChange={(e) => setHfToken(e.target.value)}
            placeholder="hf_..."
            autoFocus
            className="bg-sky-50 border-sky-300 text-slate-900"
            onKeyDown={(e) => {
              if (e.key === "Enter") handleTokenSubmit();
            }}
          />
          <p className="text-xs text-slate-600">
            获取地址 huggingface.co/settings/tokens（Write 权限）
          </p>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setTokenDialogOpen(false)}
              disabled={pushing}
              className="border-sky-300 bg-transparent text-slate-800 hover:bg-sky-200"
            >
              取消
            </Button>
            <Button
              onClick={handleTokenSubmit}
              disabled={pushing || !hfToken.trim()}
              className="bg-sky-600 hover:bg-sky-700 text-white"
            >
              {pushing ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : null}
              上传到 Hub
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
};

export default JobCard;
