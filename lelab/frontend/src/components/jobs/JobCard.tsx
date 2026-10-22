import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { JobRecord, startJobValidation, ValidationState } from "@/lib/jobsApi";
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
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  JobCheckpoint,
  listJobCheckpoints,
  getExportProgress,
  ExportProgress,
} from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";

interface Props {
  job: JobRecord;
  onStop: (id: string) => void;
  onDelete: (id: string) => void;
  onPlay: (job: JobRecord, step: number) => void;
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
  running: { label: "运行中", color: "text-green-400", Icon: Loader2 },
  done: { label: "已完成", color: "text-slate-400", Icon: CheckCircle2 },
  failed: { label: "失败", color: "text-red-400", Icon: XCircle },
  interrupted: { label: "已中断", color: "text-amber-400", Icon: AlertTriangle },
};

const JobCard: React.FC<Props> = ({ job, onStop, onDelete, onPlay, onExport, onResume }) => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const present = statePresentation[job.state];
  const Icon = present.Icon;
  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  const importedSource = job.hf_repo_id || job.output_dir;
  const stateLabel = isImported ? "已导入" : present.label;
  const isStarting = isRunning && job.metrics.total_steps === 0;
  const progressPct =
    job.metrics.total_steps > 0
      ? Math.min(100, (job.metrics.current_step / job.metrics.total_steps) * 100)
      : 0;

  const subtitle = isImported
    ? importedSource
    : isStarting
    ? "启动中…"
    : isRunning
    ? `开始于${relativeTime(job.started_at)}`
    : job.ended_at != null
    ? `结束于${relativeTime(job.ended_at)}`
    : present.label.toLowerCase();

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
      .then((cks) => {
        if (cancelled) return;
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

  const handlePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (selectedStep == null) return;
    onPlay(job, selectedStep);
  };

  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);

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

  const [validating, setValidating] = useState(false);
  // Validation is only offered on imported models, not on training jobs
  // (local/ssh/cloud) — per user request, jobs don't get a validate button.
  const canValidate = !isRunning && job.checkpoint_count > 0 && job.runner === "imported";
  const vState: ValidationState = job.validation_state ?? "none";
  const vResults = job.validation_results;

  const handleValidate = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setValidating(true);
    try {
      await startJobValidation(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "验证已启动" });
    } catch (err) {
      toast({
        title: "验证失败",
        description: err instanceof Error ? err.message : String(err),
        variant: "destructive",
      });
    } finally {
      setValidating(false);
    }
  };

  const showProgressBar = isRunning;
  const showInferenceRow = checkpoints.length > 0 && selectedStep != null;
  const isSSH = job.runner === "ssh";
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
    ? { label: "SSH 云端", cls: "bg-blue-900/40 text-blue-200 border-blue-700" }
    : job.runner === "hf_cloud"
    ? { label: "HF 云端", cls: "bg-amber-900/40 text-amber-200 border-amber-700" }
    : { label: "本机", cls: "bg-slate-700/60 text-slate-300 border-slate-600" };

  return (
    <Card
      onClick={() => navigate(`/training/${job.id}`)}
      className={`bg-slate-800/50 border-slate-700 rounded-xl transition-colors cursor-pointer hover:border-slate-500`}
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
              <div className="flex items-center gap-1 text-xs text-purple-400">
                <ShieldCheck className="w-3 h-3" />
                <span className="font-medium">验证通过</span>
              </div>
            ) : canValidate && vState === "failed" ? (
              <div className="flex items-center gap-1 text-xs text-red-400">
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
              className="h-7 w-7 text-slate-400 hover:text-white"
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
              className="h-7 px-2 text-red-400 hover:text-red-300"
            >
              {isRunning ? "确认停止?" : "确认删除?"}
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleAction}
              className="h-7 w-7 text-slate-400 hover:text-white"
              aria-label={isRunning ? "停止任务" : "删除任务"}
              title={isRunning ? "停止任务" : "删除任务"}
            >
              {isRunning ? <Square className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
            </Button>
          )}
        </div>
        <div>
          <div className="text-white font-semibold truncate" title={job.name}>
            {job.name}
          </div>
          {!isImported ? (
            <div
              className="text-xs text-slate-400 truncate"
              title={subtitle}
            >
              {subtitle}
            </div>
          ) : null}
        </div>
        {showProgressBar ? (
          <div className="relative h-5 w-full overflow-hidden rounded-md bg-slate-900 border border-slate-700">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-sky-400 transition-[width] duration-500"
              style={{ width: `${progressPct}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
              {isStarting ? "训练启动中…" : `${progressPct.toFixed(1)}%`}
            </div>
          </div>
        ) : null}
        {showInferenceRow || canValidate ? (
          <div className="flex items-center gap-2">
            {showInferenceRow && !isImported ? (
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
              />
            ) : null}
            {canResume ? (
              <Button
                size="icon"
                onClick={(e) => { e.stopPropagation(); onResume!(job); }}
                className="h-8 w-8 bg-amber-500 hover:bg-amber-600 text-white"
                aria-label="Resume training from latest checkpoint"
              >
                <RotateCcw className="w-4 h-4" />
              </Button>
            ) : null}
            {isSSH && onExport ? (
              <Button
                size="icon"
                onClick={handleExport}
                disabled={exporting}
                className="h-8 w-8 bg-blue-500 hover:bg-blue-600 text-white"
                aria-label="Export checkpoint to local"
              >
                {exporting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
              </Button>
            ) : null}
            {showInferenceRow ? (
              <Button
                size="icon"
                onClick={handlePlay}
                className="h-8 w-8 bg-green-500 hover:bg-green-600 text-white"
                aria-label="Run inference with this checkpoint"
              >
                <Play className="w-4 h-4" />
              </Button>
            ) : null}
            {canValidate ? (
              <Button
                size="icon"
                onClick={handleValidate}
                disabled={validating || vState === "running"}
                className="h-8 w-8 bg-purple-500 hover:bg-purple-600 text-white"
                aria-label="Validate model"
              >
                {validating || vState === "running" ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <FlaskConical className="w-4 h-4" />
                )}
              </Button>
            ) : null}
          </div>
        ) : null}

        {exportProgress ? (
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-2 text-xs text-slate-300">
              <span className="flex items-center gap-1.5 min-w-0">
                {exportProgress.state === "done" ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-green-400 shrink-0" />
                ) : exportProgress.state === "failed" ? (
                  <XCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                ) : (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400 shrink-0" />
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
            <div className="relative h-5 w-full overflow-hidden rounded-md bg-slate-900 border border-slate-700">
              <div
                className={`h-full transition-[width] duration-300 ${
                  exportProgress.state === "failed"
                    ? "bg-red-500/70"
                    : "bg-gradient-to-r from-blue-500 to-sky-400"
                }`}
                style={{ width: `${exportPct}%` }}
              />
              <div className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white tabular-nums drop-shadow">
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
      </CardContent>
    </Card>
  );
};

export default JobCard;
