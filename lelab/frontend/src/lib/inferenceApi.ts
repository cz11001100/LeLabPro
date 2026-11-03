import { Fetcher, apiRequest } from "./apiClient";

export type CaptureMode = "base" | "dagger" | "highlight" | "sentry";

export interface StartInferenceRequest {
  follower_port: string;
  follower_config: string;
  policy_ref: string;
  task: string;
  cameras: Record<string, {
    type: string;
    camera_index?: number;
    url?: string;
    width: number;
    height: number;
    fps?: number;
  }>;
  duration_s: number;
  n_runs: number;
  chunk_smooth?: boolean;
  chunk_smooth_strength?: number;
  rtc?: boolean;
  capture_mode?: CaptureMode;
  dataset_repo_id?: string;
  dataset_fps?: number;
  num_episodes?: number;
  record_autonomous?: boolean;
  leader_port?: string;
  leader_config?: string;
}

export interface TakeStats {
  takes: number;
  corrected_takes: number;
  failures: number;
  takeover_rate: number | null;
  failure_rate: number | null;
  updated?: string;
  policy_ref?: string;
}

export interface TakeoverState {
  mode?: string;
  phase?: "autonomous" | "paused" | "correcting" | string;
  episodes?: number;
  repo_id?: string;
  // Takes the operator marked as failures during this session (not stored in
  // the dataset; success rate comes from the evaluation page).
  failures?: number;
  // Attempts started, and how many of them needed a human, this session.
  takes?: number;
  corrected_takes?: number;
  // LeLab's capture session phase: 'ready' = rest screen waiting for 开始下一集,
  // 'running' = an episode is in progress. Drives the capture page layout.
  session_phase?: "ready" | "running" | string;
  episode_elapsed_s?: number;
  // Frames in the episode just saved, vs the policy's chunk horizon: RA-BC
  // needs the episode to be at least chunk_size frames long.
  last_episode_frames?: number;
  rabc_chunk_size?: number;
  // Seconds the policy has spent driving — the budget `duration` is measured
  // against, so it stops while the operator is paused or correcting.
  policy_s?: number;
}

export interface InferenceStatus {
  inference_active: boolean;
  started_at: number | null;
  rollout_started_at: number | null;
  elapsed_s: number;
  rollout_elapsed_s: number;
  duration_s: number | null;
  policy_ref: string | null;
  log_path: string | null;
  exited?: boolean;
  exit_code?: number | null;
  current_run?: number;
  total_runs?: number;
  camera_names?: string[];
  capture_mode?: CaptureMode;
  dataset_repo_id?: string | null;
  // Dataset-wide episode target (段数上限): a resumed dataset counts toward this
  // total, so the page shows 第 N / M 集 instead of process restarts.
  episode_target?: number;
  takeover?: TakeoverState;
  // Lifetime counters for the capture dataset being written, folded in from the
  // rollout subprocess on every poll.
  take_stats?: TakeStats;
  switching_round?: boolean;
}

export async function startInference(
  baseUrl: string,
  fetcher: Fetcher,
  request: StartInferenceRequest,
): Promise<{ message: string; log_path: string }> {
  return apiRequest<{ message: string; log_path: string }>(
    baseUrl,
    fetcher,
    "/start-inference",
    { method: "POST", body: request, action: "Start inference" },
  );
}

export async function stopInference(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ message: string }> {
  return apiRequest<{ message: string }>(baseUrl, fetcher, "/stop-inference", {
    method: "POST",
    action: "Stop inference",
  });
}

export async function getInferenceStatus(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<InferenceStatus> {
  return apiRequest<InferenceStatus>(baseUrl, fetcher, "/inference-status", {
    signal,
    action: "Get inference status",
  });
}

export async function captureInferenceSnapshot(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<{ snapshot_dir: string; n_ticks: number; summary: { manifest?: { n_ticks: number; ticks: unknown[] } } }> {
  return apiRequest(baseUrl, fetcher, "/inference-snapshot", {
    method: "POST",
    action: "Capture inference snapshot",
  });
}

export async function sendInferenceTakeover(
  baseUrl: string,
  fetcher: Fetcher,
  action: "pause" | "correct" | "save" | "stop" | "success" | "fail",
): Promise<{ message: string; key: string }> {
  return apiRequest(baseUrl, fetcher, "/inference-takeover", {
    method: "POST",
    body: { action },
    action: "Take over",
  });
}

/**
 * End the current trial now: the strategy finalises its dataset and walks the
 * arm home, then (by default) the next trial starts. Takes up to ~90s when the
 * last episode still needs encoding.
 */
export async function endInferenceRound(
  baseUrl: string,
  fetcher: Fetcher,
  advance = true,
): Promise<{ message: string; advanced?: boolean; current_run?: number }> {
  return apiRequest(baseUrl, fetcher, "/inference-end-round", {
    method: "POST",
    body: { advance },
    action: "End round",
  });
}
