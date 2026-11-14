import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useJobsChangedSignal } from "@/hooks/useJobsChangedSignal";
import {
  HubJob,
  HubModel,
  JobProgressSnapshot,
  JobRecord,
  SSHStatus,
  deleteJob,
  discoverRemoteJobs,
  getSSHStatus,
  listHubJobs,
  listJobs,
  listSSHConfigs,
  stopJob,
} from "@/lib/jobsApi";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import HubModelCard from "./HubModelCard";
import InferenceModal from "@/components/landing/InferenceModal";
import ImportModelModal from "./ImportModelModal";
import { useRobots } from "@/hooks/useRobots";
import { useDatasets } from "@/hooks/useDatasets";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  CalendarDays,
  ChevronRight,
  CloudDownload,
  Download,
  ExternalLink,
  Gauge,
  HardDrive,
  HardDriveUpload,
  RefreshCw,
  Search,
  Wifi,
  X,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { DatasetItem } from "@/lib/replayApi";
import { exportCheckpoint, resumeJob } from "@/lib/checkpointsApi";
import { groupImportedByRun } from "@/lib/modelGroups";
import { randomGreeting } from "@/lib/greetings";
import { useHfAuth } from "@/contexts/HfAuthContext";
import ScoringModal from "./ScoringModal";
import EvaluationModal from "./EvaluationModal";
import SimEvalModal from "./SimEvalModal";
import { CARD_BOX_W } from "./CheckpointDropdown";

// The registry holds every job ever run; the untracked section collapses
// finished ones, so a small cap hid whole months of history (26 records were
// on disk, only the newest 10 — all failed smoke tests — ever reached the UI).
const LIMIT = 200;

// Hub stages still doing work. Anything outside this set (COMPLETED, FAILED,
// CANCELED, …) gets demoted to UNTRACKED.
const HUB_ACTIVE_STAGES = new Set(["RUNNING", "QUEUED", "SCHEDULING"]);

const isJobActive = (j: JobRecord) =>
  j.state === "running" || j.checkpoint_count > 0;

const isHubJobActive = (h: HubJob) =>
  HUB_ACTIVE_STAGES.has((h.status?.stage ?? "").toUpperCase());

// Module-level cache keyed by API base: back-navigation to Landing seeds the
// jobs list from the last fetch so the cards render without a reload flash;
// the fetch still revalidates in the background.
const jobsCache = new Map<string, JobRecord[]>();

const sshAgeText = (s?: number | null) => {
  if (s == null) return "";
  if (s < 60) return `${s}s 前验证`;
  if (s < 3600) return `${Math.round(s / 60)} 分钟前验证`;
  return `${Math.round(s / 3600)} 小时前验证`;
};

const DatasetCard: React.FC<{
  dataset: DatasetItem;
  hub?: boolean;
  onOpen: (repoId: string) => void;
  onScore: (repoId: string) => void;
  onDelete: (repoId: string) => void;
}> = ({ dataset: d, hub = false, onOpen, onScore, onDelete }) => {
  // Two-step inline confirm with the same 3s auto-disarm as the job cards:
  // deleting wipes the local dataset directory, so one click is never enough.
  const [armed, setArmed] = useState(false);
  const armTimer = useRef<number | null>(null);
  const hasLocalCopy = d.source !== "hub";

  const disarm = () => {
    if (armTimer.current != null) {
      window.clearTimeout(armTimer.current);
      armTimer.current = null;
    }
    setArmed(false);
  };

  useEffect(() => {
    return () => {
      if (armTimer.current != null) window.clearTimeout(armTimer.current);
    };
  }, []);

  return (
    <Card
      onClick={() => onOpen(d.repo_id)}
      className="bg-slate-800/50 border-slate-700 rounded-xl transition-colors cursor-pointer hover:border-slate-500"
    >
      <CardContent className="p-4 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div
            className={`flex items-center gap-1.5 text-xs font-semibold ${
              hub ? "text-cyan-400" : "text-slate-400"
            }`}
          >
            {hub ? (
              <HardDriveUpload className="w-3.5 h-3.5" />
            ) : (
              <HardDrive className="w-3.5 h-3.5" />
            )}
            {hub ? "On Hub" : "Local"}
            {d.private ? (
              <span className="text-[10px] font-normal px-1.5 py-0.5 rounded border bg-slate-700/60 text-slate-300 border-slate-600">
                Private
              </span>
            ) : null}
          </div>
          {hasLocalCopy ? (
            armed ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  disarm();
                  onDelete(d.repo_id);
                }}
                className="h-7 px-2 text-red-400 hover:text-red-300"
              >
                Confirm Delete?
              </Button>
            ) : (
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setArmed(true);
                  if (armTimer.current != null) window.clearTimeout(armTimer.current);
                  armTimer.current = window.setTimeout(() => setArmed(false), 3000);
                }}
                className="h-7 w-7 text-slate-400 hover:text-white"
                aria-label="Delete local dataset"
                title="Delete local dataset"
              >
                <X className="w-3.5 h-3.5" />
              </Button>
            )
          ) : null}
        </div>
        <div className="text-white font-semibold truncate" title={d.repo_id}>
          {d.repo_id}
        </div>
        <div className="flex items-center gap-2">
          <div className={`flex h-8 ${CARD_BOX_W} shrink-0 items-center gap-1.5 rounded-md border border-slate-700 bg-slate-800 px-2 text-xs text-white`}>
            <CalendarDays className="w-3.5 h-3.5 shrink-0 text-slate-400" />
            <span className="truncate tabular-nums">
              {d.last_modified ? d.last_modified.slice(0, 10) : "—"}
            </span>
          </div>
          <Button
            size="icon"
            onClick={(e) => {
              e.stopPropagation();
              onScore(d.repo_id);
            }}
            className="h-8 w-8 bg-indigo-500 text-white hover:bg-indigo-600"
            aria-label="Score this dataset"
            title="Score this dataset"
          >
            <Gauge className="w-4 h-4" />
          </Button>
          {hub ? (
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="h-8 w-8 bg-cyan-600 text-white hover:bg-cyan-700"
              aria-label="Open Hub dataset page"
              title="Open Hub dataset page"
            >
              <a
                href={`https://huggingface.co/datasets/${d.repo_id}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            </Button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
};

const JobsSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { auth } = useHfAuth();

  // Picked once per mount, so navigating home re-rolls the line; waiting for
  // the HF status lets the user's handle appear instead of a nameless variant.
  const [greeting, setGreeting] = useState<string | null>(null);
  useEffect(() => {
    if (greeting !== null || auth.status === "loading") return;
    setGreeting(
      randomGreeting(auth.status === "authenticated" ? auth.username : ""),
    );
  }, [auth, greeting]);

  const [jobs, setJobs] = useState<JobRecord[]>(() => jobsCache.get(baseUrl) ?? []);
  const [hubJobs, setHubJobs] = useState<HubJob[]>([]);
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [hubAuthenticated, setHubAuthenticated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [scanning, setScanning] = useState(false);

  const { selectedRecord } = useRobots();
  // Datasets for the DATASETS section: Hub-uploaded ones shown, local-only
  // ones folded into a nested collapsed group.
  const { datasets, loading: datasetsLoading, refresh: refreshDatasets } = useDatasets();
  const hubDatasets = useMemo(
    () => datasets.filter((d) => d.source === "hub" || d.source === "both"),
    [datasets],
  );
  const localOnlyDatasets = useMemo(
    () => datasets.filter((d) => d.source === "local"),
    [datasets],
  );
  const [inferenceModalOpen, setInferenceModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [scoringModalOpen, setScoringModalOpen] = useState(false);
  const [scoringRepoId, setScoringRepoId] = useState("");
  const [evaluationModalOpen, setEvaluationModalOpen] = useState(false);
  const [evaluationJob, setEvaluationJob] = useState<JobRecord | null>(null);
  const [evaluationGroup, setEvaluationGroup] = useState<JobRecord[] | null>(null);
  const [simEvalModalOpen, setSimEvalModalOpen] = useState(false);
  const [simEvalJob, setSimEvalJob] = useState<JobRecord | null>(null);
  const [simEvalGroup, setSimEvalGroup] = useState<JobRecord[] | null>(null);
  const [inferenceJob, setInferenceJob] = useState<JobRecord | null>(null);
  const [inferenceStep, setInferenceStep] = useState<number | null>(null);
  const [sshStatus, setSshStatus] = useState<SSHStatus | null>(null);

  // The badge shows the last connection that actually worked — probing SSH on
  // load would stall the page for a minute when a pod is powered off.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      getSSHStatus(baseUrl, fetchWithHeaders)
        .then((res) => {
          if (!cancelled) setSshStatus(res);
        })
        .catch(() => {});
    load();
    window.addEventListener("focus", load);
    return () => {
      cancelled = true;
      window.removeEventListener("focus", load);
    };
  }, [baseUrl, fetchWithHeaders]);

  // From a dataset card, a dialog over the list beats navigating away; the full
  // section still lives on the Dataset page.
  const openScoring = useCallback((repoId: string) => {
    setScoringRepoId(repoId);
    setScoringModalOpen(true);
  }, []);

  // The dataset detail page loads everything else itself from /dataset-info.
  const openDataset = useCallback(
    (repoId: string) => {
      navigate("/upload", { state: { datasetInfo: { dataset_repo_id: repoId } } });
    },
    [navigate],
  );

  // Local disk only — the Hub copy is left alone by design.
  const deleteLocalDataset = useCallback(
    async (repoId: string) => {
      try {
        const response = await fetchWithHeaders(`${baseUrl}/delete-dataset`, {
          method: "POST",
          body: JSON.stringify({ dataset_repo_id: repoId }),
        });
        const data = await response.json();
        if (response.ok && data.success) {
          toast({
            title: "Dataset deleted",
            description: `${repoId} removed from local disk.`,
          });
          refreshDatasets();
        } else {
          toast({
            title: "Delete failed",
            description: data.message ?? "Unable to delete the local dataset.",
            variant: "destructive",
          });
        }
      } catch {
        toast({
          title: "Connection error",
          description: "Cannot reach the backend.",
          variant: "destructive",
        });
      }
    },
    [baseUrl, fetchWithHeaders, refreshDatasets, toast],
  );

  const refresh = useCallback(async () => {
    try {
      const [next, hub] = await Promise.all([
        listJobs(baseUrl, fetchWithHeaders, LIMIT),
        listHubJobs(baseUrl, fetchWithHeaders),
      ]);
      jobsCache.set(baseUrl, next);
      setJobs(next);
      setHubJobs(hub.jobs);
      setHubModels(hub.models);
      setHubAuthenticated(hub.authenticated);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [baseUrl, fetchWithHeaders]);

  // Initial fetch on mount + refetch when the tab regains focus. Backend
  // pushes a `jobs_changed` WS event on every registry mutation, which
  // covers any change originating on this machine. The focus refresh
  // catches changes originating elsewhere (e.g. a job submitted from
  // another tab or the HF dashboard) without burning the rate limit.
  useEffect(() => {
    refresh();
    const onVisible = () => {
      if (document.visibilityState === "visible") refresh();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", refresh);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", refresh);
    };
  }, [refresh]);

  const applyProgress = useCallback((snapshots: JobProgressSnapshot[]) => {
    if (snapshots.length === 0) return;
    setJobs((prev) => {
      if (prev.length === 0) return prev;
      const byId = new Map(snapshots.map((s) => [s.id, s]));
      let mutated = false;
      const next = prev.map((j) => {
        const s = byId.get(j.id);
        if (!s) return j;
        mutated = true;
        return {
          ...j,
          state: s.state,
          metrics: s.metrics,
          trackio_run_url: s.trackio_run_url,
          checkpoint_count: s.checkpoint_count,
        };
      });
      return mutated ? next : prev;
    });
  }, []);

  useJobsChangedSignal(refresh, applyProgress);

  // A pod can hold runs launched from another computer; those have no local
  // job.json, so scan every saved SSH target and register what it finds.
  const handleScanRemote = async () => {
    setScanning(true);
    try {
      const configs = await listSSHConfigs(baseUrl, fetchWithHeaders);
      let discovered = 0;
      const failures: string[] = [];
      for (const cfg of configs) {
        try {
          const res = await discoverRemoteJobs(baseUrl, fetchWithHeaders, cfg.id);
          discovered += res.discovered;
        } catch (e) {
          failures.push(`${cfg.name}: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
      if (discovered > 0) {
        toast({ title: `发现 ${discovered} 个云端任务`, description: "已加入任务列表，可下载其 checkpoint" });
        refresh();
      } else {
        toast({ title: "没有发现新的云端任务" });
      }
      if (failures.length > 0) {
        toast({ title: "部分连接扫描失败", description: failures.join("；"), variant: "destructive" });
      }
    } catch (e) {
      toast({
        title: "扫描失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setScanning(false);
    }
  };

  const handleStop = async (id: string) => {
    try {
      await stopJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job stopping" });
      refresh();
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handlePlay = (job: JobRecord, step: number) => {
    setInferenceJob(job);
    setInferenceStep(step);
    setInferenceModalOpen(true);
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteJob(baseUrl, fetchWithHeaders, id);
      toast({ title: "Job removed" });
      refresh();
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleExport = async (job: JobRecord, step: number): Promise<boolean> => {
    try {
      await exportCheckpoint(baseUrl, fetchWithHeaders, job.id, step);
      toast({ title: "模型已导出", description: "现在可以在本地用于推理" });
      refresh();
      return true;
    } catch (e) {
      toast({
        title: "导出失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
      return false;
    }
  };

  const handleResume = async (job: JobRecord) => {
    try {
      toast({ title: "Resuming job...", description: `Continuing training for ${job.name}` });
      await resumeJob(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "Job resumed", description: "New training run started from latest checkpoint" });
      refresh();
    } catch (e) {
      toast({
        title: "Resume failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const query = search.trim().toLowerCase();
  const matchesQuery = useCallback(
    (text: string | null | undefined) =>
      !query || (text ?? "").toLowerCase().includes(query),
    [query],
  );

  const filteredJobs = useMemo(
    () => jobs.filter((j) => matchesQuery(j.name)),
    [jobs, matchesQuery],
  );
  const filteredHubJobs = useMemo(
    () =>
      hubJobs.filter((h) =>
        matchesQuery(h.docker_image ?? h.space_id ?? h.id),
      ),
    [hubJobs, matchesQuery],
  );
  const filteredHubModels = useMemo(
    () => hubModels.filter((m) => matchesQuery(m.repo_id)),
    [hubModels, matchesQuery],
  );

  const localJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "local"),
    [filteredJobs],
  );
  // SSH trainings run on remote (typically rented cloud) machines, so they
  // are grouped with the cloud section rather than with local jobs.
  const sshJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "ssh"),
    [filteredJobs],
  );
  const trackedCloudJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "hf_cloud"),
    [filteredJobs],
  );
  const importedJobs = useMemo(
    () => filteredJobs.filter((j) => j.runner === "imported"),
    [filteredJobs],
  );
  // Collapse per-step imported jobs into one card per training run (shared with
  // the Landing Deploy/Eval pickers via modelGroups).
  const importedGroups = useMemo(() => groupImportedByRun(importedJobs), [importedJobs]);

  // The flask on a model card opens the evaluation dialog; the run itself is
  // started and stopped inside it. Siblings of the same training run come
  // along so the dialog can switch checkpoint.
  const groupMembersOf = useCallback(
    (jobId: string) =>
      importedGroups.find((g) => g.members.some((m) => m.id === jobId))?.members ??
      null,
    [importedGroups],
  );

  const handleEvaluate = useCallback(
    (job: JobRecord) => {
      setEvaluationJob(job);
      setEvaluationGroup(groupMembersOf(job.id));
      setEvaluationModalOpen(true);
    },
    [groupMembersOf],
  );

  const handleSimEval = useCallback(
    (job: JobRecord) => {
      setSimEvalJob(job);
      setSimEvalGroup(groupMembersOf(job.id));
      setSimEvalModalOpen(true);
    },
    [groupMembersOf],
  );
  // Hub jobs already mirrored by a local JobRecord get their richer card via
  // trackedCloudJobs; everything else from the hub gets a plain HubJobCard.
  const trackedHfJobIds = useMemo(
    () =>
      new Set(
        trackedCloudJobs
          .map((j) => j.hf_job_id)
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs],
  );
  const untrackedHubJobs = useMemo(
    () => filteredHubJobs.filter((h) => !trackedHfJobIds.has(h.id)),
    [filteredHubJobs, trackedHfJobIds],
  );
  // Hide model repos that map 1-to-1 to a tracked cloud job (those already
  // appear via JobCard); the remainder are past trainings the registry no
  // longer remembers.
  const trackedRepoIds = useMemo(
    () =>
      new Set(
        trackedCloudJobs
          .map((j) => j.hf_repo_id)
          .filter((id): id is string => !!id),
      ),
    [trackedCloudJobs],
  );
  const untrackedHubModels = useMemo(
    () => filteredHubModels.filter((m) => !trackedRepoIds.has(m.repo_id)),
    [filteredHubModels, trackedRepoIds],
  );

  // Active = running or has runnable checkpoints. Everything else collapses
  // under UNTRACKED so the eye lands on what's still relevant.
  const localActive = useMemo(() => localJobs.filter(isJobActive), [localJobs]);
  const localUntracked = useMemo(
    () => localJobs.filter((j) => !isJobActive(j)),
    [localJobs],
  );
  const sshActive = useMemo(() => sshJobs.filter(isJobActive), [sshJobs]);
  const sshUntracked = useMemo(
    () => sshJobs.filter((j) => !isJobActive(j)),
    [sshJobs],
  );
  const trackedCloudActive = useMemo(
    () => trackedCloudJobs.filter(isJobActive),
    [trackedCloudJobs],
  );
  const trackedCloudUntracked = useMemo(
    () => trackedCloudJobs.filter((j) => !isJobActive(j)),
    [trackedCloudJobs],
  );
  const untrackedHubActive = useMemo(
    () => untrackedHubJobs.filter(isHubJobActive),
    [untrackedHubJobs],
  );
  const untrackedHubInactive = useMemo(
    () => untrackedHubJobs.filter((h) => !isHubJobActive(h)),
    [untrackedHubJobs],
  );

  const untrackedCount =
    localUntracked.length +
    sshUntracked.length +
    trackedCloudUntracked.length +
    untrackedHubInactive.length;

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-white truncate">{greeting ?? ""}</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search jobs…"
              className="h-8 w-48 sm:w-60 pl-8 bg-slate-800/50 border-slate-700 text-sm text-white placeholder:text-slate-500"
              aria-label="Search jobs"
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate("/training", { state: { sshRunner: true } })}
            title={
              sshStatus?.connected
                ? `${sshStatus.last_ok?.name || "SSH"} 已连通 · ${sshAgeText(sshStatus.age_s)}`
                : "未连接 · 点进去添加连接或用 Test 验证"
            }
            className={`h-8 border-slate-700 bg-slate-800/50 ${
              sshStatus?.connected
                ? "text-blue-300 hover:text-blue-200"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <Wifi className="w-3.5 h-3.5 mr-1.5" />
            SSH
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportModalOpen(true)}
            className="h-8 border-slate-700 bg-slate-800/50 text-slate-200 hover:text-white"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" />
            Import Model
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleScanRemote}
            disabled={scanning}
            className="h-8 border-slate-700 bg-slate-800/50 text-slate-200 hover:text-white"
            title="扫描已保存的 SSH 目标，把别的电脑发起的训练任务登记到本机"
          >
            {scanning ? (
              <RefreshCw className="w-3.5 h-3.5 mr-1.5 animate-spin" />
            ) : (
              <CloudDownload className="w-3.5 h-3.5 mr-1.5" />
            )}
            扫描云端任务
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={refresh}
            className="h-7 w-7 text-slate-400 hover:text-white"
            aria-label="Refresh jobs"
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {error ? <p className="text-sm text-red-300">任务加载失败：{error}</p> : null}

      <Collapsible defaultOpen>
        <CollapsibleTrigger className="group flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
          <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
          任务 Jobs (
          {localActive.length +
            sshActive.length +
            trackedCloudActive.length +
            untrackedHubActive.length +
            untrackedHubModels.length}
          )
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3">
          {localActive.length === 0 &&
          sshActive.length === 0 &&
          trackedCloudActive.length === 0 &&
          untrackedHubActive.length === 0 &&
          untrackedHubModels.length === 0 ? (
            <p className="text-sm text-slate-500">
              {query
                ? "没有匹配的任务。"
                : "暂无任务；可在训练页发起。"}
            </p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {localActive.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {sshActive.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {trackedCloudActive.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {untrackedHubActive.map((job) => (
                <HubJobCard key={job.id} job={job} />
              ))}
              {untrackedHubModels.map((model) => (
                <HubModelCard key={model.repo_id} model={model} />
              ))}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      {importedJobs.length > 0 ? (
        <>
          <div className="border-t border-slate-700" />
          <Collapsible defaultOpen>
            <CollapsibleTrigger className="group flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
              <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
              模型 Models ({importedJobs.length})
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-3">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {importedGroups.map((g) => (
                  <JobCard
                    key={g.members[0].id}
                    job={g.members[0]}
                    group={g.members.length > 1 ? g.members : undefined}
                    onStop={handleStop}
                    onDelete={handleDelete}
                    onPlay={handlePlay}
                    onEvaluate={handleEvaluate}
                    onSimEval={handleSimEval}
                    onExport={handleExport}
                    onResume={handleResume}
                  />
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      ) : null}

      <div className="border-t border-slate-700" />

      <Collapsible defaultOpen>
        <CollapsibleTrigger className="group flex items-center gap-1.5 text-sm font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
          <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
          数据集 DATASETS
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3 space-y-4">
          {datasetsLoading ? (
            <p className="text-sm text-slate-500">Loading datasets…</p>
          ) : hubDatasets.length === 0 && localOnlyDatasets.length === 0 ? (
            <p className="text-sm text-slate-500">No datasets yet.</p>
          ) : (
            <div className="space-y-4">
              {hubDatasets.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                  {hubDatasets.map((d) => (
                    <DatasetCard
                      key={d.repo_id}
                      dataset={d}
                      hub
                      onOpen={openDataset}
                      onScore={openScoring}
                      onDelete={deleteLocalDataset}
                    />
                  ))}
                </div>
              ) : null}
              {localOnlyDatasets.length > 0 ? (
                <Collapsible defaultOpen={false}>
                  <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500 hover:text-slate-300 transition-colors">
                    <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
                    其他数据集 Local only ({localOnlyDatasets.length})
                  </CollapsibleTrigger>
                  <CollapsibleContent className="pt-3">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                      {localOnlyDatasets.map((d) => (
                        <DatasetCard
                          key={d.repo_id}
                          dataset={d}
                          onOpen={openDataset}
                          onScore={openScoring}
                          onDelete={deleteLocalDataset}
                        />
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              ) : null}
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>

      {untrackedCount > 0 ? (
        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-white transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            未跟踪 Untracked ({untrackedCount})
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-3">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {localUntracked.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {sshUntracked.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {trackedCloudUntracked.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  onStop={handleStop}
                  onDelete={handleDelete}
                  onPlay={handlePlay}
                  onEvaluate={handleEvaluate}
                  onSimEval={handleSimEval}
                  onExport={handleExport}
                  onResume={handleResume}
                />
              ))}
              {untrackedHubInactive.map((job) => (
                <HubJobCard key={job.id} job={job} />
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      ) : null}

      {inferenceJob ? (
        <InferenceModal
          open={inferenceModalOpen}
          onOpenChange={setInferenceModalOpen}
          robot={selectedRecord}
          jobId={inferenceJob.id}
          initialStep={inferenceStep}
        />
      ) : null}

      <ImportModelModal
        open={importModalOpen}
        onOpenChange={setImportModalOpen}
        onImported={refresh}
      />

      <EvaluationModal
        open={evaluationModalOpen}
        onOpenChange={setEvaluationModalOpen}
        job={evaluationJob}
        group={evaluationGroup ?? undefined}
      />

      <SimEvalModal
        open={simEvalModalOpen}
        onOpenChange={setSimEvalModalOpen}
        job={simEvalJob}
        group={simEvalGroup ?? undefined}
      />

      <ScoringModal
        open={scoringModalOpen}
        onOpenChange={setScoringModalOpen}
        datasetRepoId={scoringRepoId || undefined}
      />
    </section>
  );
};

export default JobsSection;
