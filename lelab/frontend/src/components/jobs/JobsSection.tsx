import React, { useCallback, useEffect, useMemo, useState } from "react";
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
  deleteJob,
  listHubJobs,
  listJobs,
  stopJob,
} from "@/lib/jobsApi";
import JobCard from "./JobCard";
import HubJobCard from "./HubJobCard";
import HubModelCard from "./HubModelCard";
import InferenceModal from "@/components/landing/InferenceModal";
import ImportModelModal from "./ImportModelModal";
import { useRobots } from "@/hooks/useRobots";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight, Download, RefreshCw, Search } from "lucide-react";
import { exportCheckpoint, resumeJob } from "@/lib/checkpointsApi";

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

const JobsSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const [jobs, setJobs] = useState<JobRecord[]>(() => jobsCache.get(baseUrl) ?? []);
  const [hubJobs, setHubJobs] = useState<HubJob[]>([]);
  const [hubModels, setHubModels] = useState<HubModel[]>([]);
  const [hubAuthenticated, setHubAuthenticated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const { selectedRecord } = useRobots();
  const [inferenceModalOpen, setInferenceModalOpen] = useState(false);
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [inferenceJob, setInferenceJob] = useState<JobRecord | null>(null);
  const [inferenceStep, setInferenceStep] = useState<number | null>(null);

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
          wandb_run_url: s.wandb_run_url,
          checkpoint_count: s.checkpoint_count,
        };
      });
      return mutated ? next : prev;
    });
  }, []);

  useJobsChangedSignal(refresh, applyProgress);

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
  // Group imported models that came from the SAME training run. Each downloaded
  // checkpoint is registered as its own imported job named "<run> (step N)"; we
  // collapse those into one card whose checkpoint dropdown switches between the
  // steps. Jobs without the " (step N)" suffix (manual imports) stay alone.
  const importedGroups = useMemo(() => {
    const stepRe = /^(.*) \(step (\d+)\)$/;
    const groups: { base: string; members: JobRecord[] }[] = [];
    const byBase = new Map<string, JobRecord[]>();
    for (const job of importedJobs) {
      const m = stepRe.exec(job.name);
      if (!m) {
        groups.push({ base: job.name, members: [job] });
        continue;
      }
      const base = m[1];
      const bucket = byBase.get(base);
      if (bucket) bucket.push(job);
      else {
        byBase.set(base, [job]);
        groups.push({ base, members: byBase.get(base)! });
      }
    }
    // Latest step first within each group.
    for (const g of groups) {
      if (g.members.length > 1) {
        g.members.sort(
          (a, b) => Number(stepRe.exec(b.name)![2]) - Number(stepRe.exec(a.name)![2]),
        );
      }
    }
    return groups;
  }, [importedJobs]);
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
        <h2 className="text-lg font-semibold text-white">训练任务 Jobs</h2>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400 pointer-events-none" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索任务…"
              className="h-8 w-48 sm:w-60 pl-8 bg-slate-800/50 border-slate-700 text-sm text-white placeholder:text-slate-500"
              aria-label="搜索任务"
            />
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setImportModalOpen(true)}
            className="h-8 border-slate-700 bg-slate-800/50 text-slate-200 hover:text-white"
          >
            <Download className="w-3.5 h-3.5 mr-1.5" />
            导入模型
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
                    onExport={handleExport}
                    onResume={handleResume}
                  />
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </>
      ) : null}

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
    </section>
  );
};

export default JobsSection;
