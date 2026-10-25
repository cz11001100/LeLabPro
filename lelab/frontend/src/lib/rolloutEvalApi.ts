import { Fetcher, apiRequest } from "./apiClient";

export interface RolloutDatasetInfo {
  repo_id: string;
  episodes: number;
  frames: number;
  scored: boolean;
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

export interface RolloutSummary {
  available: boolean;
  error?: string;
  no_progress?: boolean;
  hint?: string;
  repo_id?: string;
  episodes?: number;
  frames?: number;
  successes?: number;
  success_rate?: number | null;
  threshold?: number;
  details?: RolloutEpisodeDetail[];
}

export interface RolloutScoreState {
  running: boolean;
  repo_id?: string;
  log_path?: string;
  command?: string;
  elapsed_s?: number;
  exit_code?: number | null;
}

export interface RolloutScoreRequest {
  dataset_repo_id: string;
  backend?: "robometer" | "sarm";
  reward_model_path?: string;
  image_key?: string;
  device?: string;
  stride?: number;
  head_mode?: string;
  episodes?: number[] | null;
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
): Promise<RolloutSummary> {
  return apiRequest<RolloutSummary>(
    baseUrl,
    fetcher,
    `/rollout-summary?repo_id=${encodeURIComponent(repoId)}`,
    { action: "Get rollout summary" },
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
