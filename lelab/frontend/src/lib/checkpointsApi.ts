import { Fetcher, apiRequest } from "./apiClient";
import { JobRecord } from "./jobsApi";

export interface JobCheckpoint {
  step: number;
  source: "local" | "hub";
  ref: string;
}

export interface PolicyConfigSummary {
  policy_type: string | null;
  image_features: Record<string, { height: number; width: number }>;
  requires_task: boolean;
  tasks?: string[];
}

export interface ModelTrainingInfo {
  batch_size: number | null;
  steps: number | null;
  seed: number | null;
  num_workers: number | null;
  optimizer_type: string | null;
  lr: number | null;
  weight_decay: number | null;
  grad_clip_norm: number | null;
  scheduler_type: string | null;
  warmup_steps: number | null;
  decay_steps: number | null;
  peak_lr: number | null;
  decay_lr: number | null;
}

export interface ModelInfo {
  step: number;
  policy_type: string | null;
  chunk_size: number | null;
  n_action_steps: number | null;
  max_action_dim: number | null;
  num_vlm_layers: number | null;
  num_expert_layers: number | null;
  vlm_model_name: string | null;
  normalization_mapping: Record<string, string> | null;
  dataset_repo_id: string | null;
  training: ModelTrainingInfo | null;
  size_bytes: number | null;
  saved_at: number | null;
}

export async function getCheckpointModelInfo(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<ModelInfo> {
  return apiRequest<ModelInfo>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/checkpoints/${step}/model-info`,
    { signal, action: "Load model info" },
  );
}

export async function listJobCheckpoints(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  signal?: AbortSignal,
): Promise<JobCheckpoint[]> {
  const body = await apiRequest<{ checkpoints: JobCheckpoint[] }>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/checkpoints`,
    { signal, action: "List checkpoints" },
  );
  return body.checkpoints;
}

export async function getCheckpointPolicyConfig(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<PolicyConfigSummary> {
  return apiRequest<PolicyConfigSummary>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/checkpoints/${step}/policy-config`,
    { signal, action: "Load policy config" },
  );
}

export async function exportCheckpoint(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/export-checkpoint/${step}`,
    { signal, method: "POST", action: "Export checkpoint" },
  );
}

export type ExportState = "none" | "preparing" | "downloading" | "done" | "failed";

export interface ExportProgress {
  state: ExportState;
  transferred_bytes: number;
  total_bytes: number;
  current_file: string;
  error?: string | null;
}

export async function getExportProgress(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
  signal?: AbortSignal,
): Promise<ExportProgress> {
  return apiRequest<ExportProgress>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/export-progress/${step}`,
    { signal, action: "Load export progress" },
  );
}

export async function cancelJobExport(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  step: number,
): Promise<{ cancelled: boolean; message: string }> {
  return apiRequest<{ cancelled: boolean; message: string }>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/export-cancel/${step}`,
    { method: "POST", action: "Cancel export" },
  );
}

export async function resumeJob(
  baseUrl: string,
  fetcher: Fetcher,
  jobId: string,
  signal?: AbortSignal,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(
    baseUrl,
    fetcher,
    `/jobs/${jobId}/resume`,
    { signal, method: "POST", action: "Resume job" },
  );
}
