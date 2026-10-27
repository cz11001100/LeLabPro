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
  // dagger: end the round as soon as one correction is saved, then continue.
  auto_advance?: boolean;
  leader_port?: string;
  leader_config?: string;
}

export interface TakeoverState {
  mode?: string;
  phase?: "autonomous" | "paused" | "correcting" | string;
  episodes?: number;
  repo_id?: string;
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
  takeover?: TakeoverState;
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
  action: "pause" | "correct" | "save" | "stop",
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
