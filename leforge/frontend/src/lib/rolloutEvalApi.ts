import { Fetcher, apiRequest } from "./apiClient";

export interface RolloutDatasetInfo {
  repo_id: string;
  episodes: number;
  frames: number;
  fps?: number | null;
  scored: boolean;
  /** Capture datasets sort first; merged training sets are scoreable too. */
  is_rollout?: boolean;
  has_intervention?: boolean;
  backends?: string[];
  modified: string;
}

export interface RolloutEpisodeDetail {
  episode_index: number;
  frames: number | null;
  progress_start: number;
  progress_end: number;
  progress_peak: number;
  stalled_at: number | null;
  success: boolean;
}

export interface RolloutAlignment {
  ok: boolean;
  reason?: string;
  message?: string;
  rows?: number;
  frames?: number;
  episodes_without_progress?: number;
}

export interface RolloutSummary {
  available: boolean;
  error?: string;
  no_progress?: boolean;
  misaligned?: boolean;
  hint?: string;
  repo_id?: string;
  column?: string;
  episodes?: number;
  frames?: number;
  successes?: number;
  success_rate?: number | null;
  threshold?: number;
  has_intervention?: boolean;
  alignment?: RolloutAlignment;
  details?: RolloutEpisodeDetail[];
}

export interface RolloutScoreState {
  running: boolean;
  repo_id?: string;
  log_path?: string;
  command?: string;
  elapsed_s?: number;
  exit_code?: number | null;
  /** Parsed from the scorer's tqdm bar while it runs. */
  episodes_done?: number;
  episodes_total?: number;
  percent?: number;
  eta_s?: number | null;
}

export interface RolloutScoreRequest {
  dataset_repo_id: string;
  backend?: "robometer" | "sarm";
  reward_model_path?: string;
  image_key?: string;
  device?: string;
  /** sarm only: score every Nth frame, interpolate the rest. */
  stride?: number;
  /** robometer only: frames each per-frame prediction window looks at. */
  num_subsampled_frames?: number;
  /** robometer only: prediction windows per forward pass. */
  batch_size?: number;
  head_mode?: string;
  episodes?: number[] | null;
  push_to_hub?: boolean;
}

export async function listRolloutDatasets(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RolloutDatasetInfo[]> {
  const res = await apiRequest<{ datasets: RolloutDatasetInfo[] }>(
    baseUrl,
    fetcher,
    "/rollout-datasets",
    { action: "List rollout datasets" },
  );
  return res.datasets ?? [];
}

export async function getRolloutSummary(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  opts?: { successProgress?: number; headMode?: string },
): Promise<RolloutSummary> {
  const params = new URLSearchParams({ repo_id: repoId });
  if (opts?.successProgress != null) params.set("success_progress", String(opts.successProgress));
  if (opts?.headMode) params.set("head_mode", opts.headMode);
  return apiRequest<RolloutSummary>(baseUrl, fetcher, `/rollout-summary?${params}`, {
    action: "Get rollout summary",
  });
}

export interface RolloutEpisodeSeries {
  available: boolean;
  error?: string;
  hint?: string;
  episode_index?: number;
  fps?: number;
  length?: number;
  /** [frame, progress] pairs, downsampled for drawing. */
  points?: [number, number | null][];
  correction_windows?: { start: number; end: number }[];
}

export async function getRolloutEpisodeSeries(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  episodeIndex: number,
  headMode?: string,
): Promise<RolloutEpisodeSeries> {
  const params = new URLSearchParams({
    repo_id: repoId,
    episode_index: String(episodeIndex),
  });
  if (headMode) params.set("head_mode", headMode);
  return apiRequest<RolloutEpisodeSeries>(baseUrl, fetcher, `/rollout-episode-series?${params}`, {
    action: "Get episode progress curve",
  });
}

export interface RolloutTakeStats {
  takes: number;
  corrected_takes: number;
  failures: number;
  takeover_rate: number | null;
  failure_rate: number | null;
  updated?: string;
}

export async function getRolloutTakeStats(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<RolloutTakeStats> {
  return apiRequest<RolloutTakeStats>(
    baseUrl,
    fetcher,
    `/rollout-take-stats?repo_id=${encodeURIComponent(repoId)}`,
    { action: "Get capture take counters" },
  );
}

export async function startRolloutScore(
  baseUrl: string,
  fetcher: Fetcher,
  request: RolloutScoreRequest,
): Promise<{ message: string; log_path: string; command: string }> {
  return apiRequest(baseUrl, fetcher, "/rollout-score", {
    method: "POST",
    body: request,
    action: "Score rollout dataset",
  });
}

export async function getRolloutScoreStatus(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RolloutScoreState> {
  return apiRequest<RolloutScoreState>(baseUrl, fetcher, "/rollout-score-status", {
    action: "Get rollout scoring status",
  });
}

/**
 * RA-BC zeroes frames whose progress did not advance — which is exactly what a
 * recovery looks like when it starts by backing off. This lifts the recorded
 * correction windows in the progress file so those frames keep their weight.
 */
export async function exemptInterventions(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<{ message: string; frames_lifted: number }> {
  return apiRequest(baseUrl, fetcher, "/rollout-exempt-interventions", {
    method: "POST",
    body: { repo_id: repoId },
    action: "Exempt correction frames",
  });
}

export interface RolloutComparison {
  available: boolean;
  backends?: string[];
  hint?: string;
  frames?: number;
  pearson?: number | null;
  mean_abs_diff?: number;
  success_rate?: { robometer: number | null; sarm: number | null };
  agreement?: number | null;
  disagreements?: {
    episode_index: number;
    robometer_end: number;
    sarm_end: number;
    robometer_success: boolean;
    sarm_success: boolean;
    mean_abs_diff: number;
  }[];
}

export async function compareRolloutProgress(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
): Promise<RolloutComparison> {
  return apiRequest<RolloutComparison>(
    baseUrl,
    fetcher,
    `/rollout-compare?repo_id=${encodeURIComponent(repoId)}`,
    { action: "Compare scoring backends" },
  );
}

export async function promoteRolloutProgress(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  backend: "robometer" | "sarm",
): Promise<{ message: string }> {
  return apiRequest(baseUrl, fetcher, "/rollout-promote-progress", {
    method: "POST",
    body: { repo_id: repoId, backend },
    action: "Promote progress file",
  });
}

export interface ScoreLogs {
  available: boolean;
  source?: "local" | "ssh";
  log_path?: string;
  text: string;
}

/** Live stdout of the running scorer, for the terminal panel. */
export async function getScoreLogs(
  baseUrl: string,
  fetcher: Fetcher,
  maxChars = 20000,
): Promise<ScoreLogs> {
  return apiRequest<ScoreLogs>(baseUrl, fetcher, `/rollout-score-logs?max_chars=${maxChars}`, {
    action: "Get scoring logs",
  });
}

export interface ScoreJobRecord {
  runner: "local" | "ssh";
  dataset_repo_id: string;
  backend: string;
  config_id?: string;
  log_path?: string;
  started_at?: number;
  episodes_total?: number;
  pid?: number;
}

export interface ScoreJobInfo {
  active: boolean;
  job?: ScoreJobRecord;
  status?: RolloutScoreState & RolloutRemoteStatus;
}

/** The scoring job this page should re-attach to after a reload / LeForge restart. */
export async function getScoreJob(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<ScoreJobInfo> {
  return apiRequest<ScoreJobInfo>(baseUrl, fetcher, "/rollout-score-job", {
    action: "Get running scoring job",
  });
}

export interface RolloutPreflight {
  ok: boolean;
  checks: { name: string; ok: boolean; detail?: string }[];
  command?: string;
}

/** "Can this scoring job actually run here?" — per-runner, item by item. */
export async function preflightRolloutScore(
  baseUrl: string,
  fetcher: Fetcher,
  request: RolloutScoreRequest & { runner: "local" | "ssh"; config_id?: string },
): Promise<RolloutPreflight> {
  return apiRequest<RolloutPreflight>(baseUrl, fetcher, "/rollout-score-preflight", {
    method: "POST",
    body: request,
    action: "Check scoring environment",
  });
}

export interface RolloutRealWorld extends RolloutTakeStats {
  repo_id?: string;
  scored?: boolean;
}

/** What the policy last did on the rig, for the deploy page's model card. */
export async function getLatestRealWorld(
  baseUrl: string,
  fetcher: Fetcher,
  policyRef?: string,
): Promise<RolloutRealWorld | null> {
  const params = policyRef ? `?policy_ref=${encodeURIComponent(policyRef)}` : "";
  const res = await apiRequest<Partial<RolloutRealWorld>>(
    baseUrl,
    fetcher,
    `/rollout-latest-real-world${params}`,
    { action: "Get latest on-robot numbers" },
  );
  return res?.takes ? ({ ...res, takes: res.takes ?? 0 } as RolloutRealWorld) : null;
}

export async function startRolloutScoreRemote(
  baseUrl: string,
  fetcher: Fetcher,
  request: RolloutScoreRequest & { config_id: string; use_cn_mirror?: boolean },
): Promise<{ message: string; log_path: string; command: string }> {
  return apiRequest(baseUrl, fetcher, "/rollout-score-remote", {
    method: "POST",
    body: request,
    action: "Score rollout dataset on pod",
  });
}

export interface RolloutRemoteStatus {
  success: boolean;
  message?: string;
  running?: boolean;
  finished?: boolean;
  finished_ok?: boolean;
  log_tail?: string;
  /** "0 2026-11-03 04:52:58" — bytes + mtime, or "missing". */
  log_stat?: string;
  /** ps: pid etime time pcpu, one line per matching process. */
  proc_stats?: string;
  episodes_done?: number;
  episodes_total?: number;
  percent?: number;
  eta_s?: number | null;
  pulled?: { success: boolean; message: string; hub_file?: string };
  alignment?: RolloutAlignment;
}

export async function getRolloutScoreRemoteStatus(
  baseUrl: string,
  fetcher: Fetcher,
  configId: string,
  repoId: string,
  backend: string,
): Promise<RolloutRemoteStatus> {
  const params = new URLSearchParams({
    config_id: configId,
    dataset_repo_id: repoId,
    backend,
  });
  return apiRequest<RolloutRemoteStatus>(
    baseUrl,
    fetcher,
    `/rollout-score-remote-status?${params}`,
    { action: "Pod scoring status" },
  );
}

export async function startSarmTraining(
  baseUrl: string,
  fetcher: Fetcher,
  body: {
    config_id: string;
    dataset_repo_id: string;
    image_key?: string;
    steps?: number;
    batch_size?: number;
    push_repo_id?: string;
  },
): Promise<{ message: string; log_path: string }> {
  return apiRequest(baseUrl, fetcher, "/ssh-configs/sarm-train", {
    method: "POST",
    body,
    action: "Train SARM scorer",
  });
}

export async function getSarmTrainingStatus(
  baseUrl: string,
  fetcher: Fetcher,
  configId: string,
): Promise<{ success: boolean; training: boolean; finished_ok: boolean; log_tail: string }> {
  return apiRequest(baseUrl, fetcher, "/ssh-configs/sarm-train-status", {
    method: "POST",
    body: { config_id: configId },
    action: "SARM training status",
  });
}
