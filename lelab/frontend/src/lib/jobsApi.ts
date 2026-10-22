import { ApiError, Fetcher, apiRequest } from "./apiClient";

export type JobState = "running" | "done" | "failed" | "interrupted";

export interface TrainingMetrics {
  current_step: number;
  total_steps: number;
  current_loss: number | null;
  current_lr: number | null;
  grad_norm: number | null;
  eta_seconds: number | null;
}

export interface LogLine {
  timestamp: number;
  message: string;
}

export type MetricsHistoryPoint = {
  step: number;
  loss: number | null;
  lr: number | null;
  grad_norm: number | null;
};

// Mirror of the backend TrainingRequest. The frontend doesn't send all of
// these; defaults on the server fill in the rest.
export interface TrainingRequest {
  dataset_repo_id: string;
  dataset_root?: string;
  job_name?: string;
  policy_type: string;
  steps: number;
  batch_size: number;
  seed?: number;
  num_workers: number;
  log_freq: number;
  save_freq: number;
  save_checkpoint: boolean;
  resume: boolean;
  resume_from_job_id?: string;
  wandb_enable: boolean;
  wandb_project?: string;
  wandb_entity?: string;
  wandb_notes?: string;
  wandb_mode?: string;
  wandb_disable_artifact: boolean;
  policy_device?: string;
  policy_use_amp: boolean;
  policy_pretrained_path?: string;
  optimizer_type?: string;
  optimizer_lr?: number;
  optimizer_weight_decay?: number;
  optimizer_grad_clip_norm?: number;
  scheduler_warmup_steps?: number;
  scheduler_decay_steps?: number;
  scheduler_decay_lr?: number;
  use_policy_training_preset: boolean;
  // Data augmentation
  augmentation_enable?: boolean;
  augmentation_max_num_transforms?: number;
  augmentation_random_order?: boolean;
  augmentation_brightness?: boolean;
  augmentation_contrast?: boolean;
  augmentation_saturation?: boolean;
  augmentation_hue?: boolean;
  augmentation_sharpness?: boolean;
  augmentation_affine?: boolean;
  // GR00T-specific (only sent when policy_type === "groot"); backend ignores
  // the policy_* ones for other policies.
  dataset_image_transforms_enable?: boolean;
  policy_base_model_path?: string;
  policy_embodiment_tag?: string;
  policy_chunk_size?: number;
  policy_n_action_steps?: number;
  policy_use_relative_actions?: boolean;
  policy_relative_exclude_joints?: string[];
  policy_use_bf16?: boolean;
  // Optional target for runner dispatch; omitted ⇒ local.
  target?: { runner: "local" | "hf_cloud" | "ssh"; flavor?: string; ssh_config_id?: string };
}

export type ValidationState = "none" | "pending" | "running" | "done" | "failed";

export interface ProbeItem {
  name: string;
  label: string;
  delta_deg: number;
  relative_delta: number;
}

export interface ProbeResult {
  items: ProbeItem[];
  shift_sign_agreement?: number | null;
  vision_sensitivity: number | null;
  language_sensitivity: number | null;
  verdict: string;
}

export interface ValidationOptions {
  step?: number | null;
  num_batches?: number;
  batch_size?: number;
  seed?: number;
  holdout_episodes?: number;
  frames_per_episode?: number;
  episodes?: number[] | null;
  dataset_repo_id?: string | null;
  probe?: boolean;
  saliency_frames?: number;
  aim_frames?: number;
  device?: string;
}

export interface ValidationResults {
  success: boolean;
  checkpoint_path?: string;
  dataset_repo_id?: string;
  batches_evaluated?: number;
  avg_loss?: number;
  avg_inference_time_ms?: number;
  action_mse?: number | null;
  action_mae?: number | null;
  error?: string;
  shape_mismatch?: string;
  // Offline evaluator: open-loop action error + perception probe.
  device?: string;
  seed?: number;
  batch_size?: number;
  num_samples?: number;
  batches_skipped?: number;
  episodes_evaluated?: number[];
  holdout_episodes?: number;
  chunk_size?: number;
  fps?: number;
  joint_names?: string[];
  per_joint_mae_deg?: number[];
  per_joint_rmse_deg?: number[];
  per_joint_action_std_deg?: number[];
  per_joint_naive_mae_deg?: number[];
  per_joint_mae_vs_naive?: (number | null)[];
  mean_mae_vs_naive?: number;
  horizon_mae_deg?: number[];
  drift_horizon_steps?: number | null;
  recommended_n_action_steps?: number;
  recommended_replan_hz?: number;
  per_episode?: { episode: number; n: number; mae_deg: number; worst_joint: string }[];
  worst_episodes?: number[];
  observation_conflict_deg?: number | null;
  error_floor_deg?: number | null;
  probe?: ProbeResult | null;
  roi_grounding?: {
    by_camera: Record<string, { mean_ratio: number; median_ratio: number; n: number; grounded: boolean }>;
    rows?: { frame: number; camera: string; patch_px: number; delta_target_deg: number; delta_random_deg: number; ratio: number }[];
    verdict: string;
    threshold: number;
  } | null;
  aim?: {
    mean_aim_error_model_px: number | null;
    mean_aim_error_human_px: number | null;
    calibration_r2_xy?: number[];
    mean_calibration_r2?: number;
    n_fit?: number;
    verdict: string;
  } | null;
  localization?: {
    rows: { frame: number; camera: string; peak_to_ball_px: number | null; saliency_on_ball: number | null; chart: string }[];
    mean_peak_to_ball_px: number | null;
    mean_saliency_on_ball: number | null;
    chance_level: number;
  } | null;
  charts?: Record<string, string>;
}

export interface JobRecord {
  id: string;
  name: string;
  state: JobState;
  config: TrainingRequest;
  output_dir: string;
  started_at: number;
  ended_at: number | null;
  exit_code: number | null;
  error_message: string | null;
  metrics: TrainingMetrics;
  runner: "local" | "hf_cloud" | "ssh" | "imported";
  hf_job_id: string | null;
  hf_flavor: string | null;
  hf_repo_id: string | null;
  hf_job_url: string | null;
  wandb_run_url: string | null;
  checkpoint_count: number;
  validation_state: ValidationState;
  validation_results: ValidationResults | null;
}

// Per-running-job snapshot pushed by the watchdog over WS at ~1Hz. Subset
// of JobRecord — just the fields that change during a running tick.
export interface JobProgressSnapshot {
  id: string;
  state: JobState;
  metrics: TrainingMetrics;
  wandb_run_url: string | null;
  checkpoint_count: number;
}

export async function listJobs(
  baseUrl: string,
  fetcher: Fetcher,
  limit = 10,
  signal?: AbortSignal,
): Promise<JobRecord[]> {
  const body = await apiRequest<{ jobs: JobRecord[] }>(
    baseUrl,
    fetcher,
    `/jobs?limit=${limit}`,
    { signal, action: "List jobs" },
  );
  return body.jobs;
}

export async function getJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, `/jobs/${id}`, {
    signal,
    action: "Get job",
  });
}

export async function getJobLogs(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const body = await apiRequest<{ logs: LogLine[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/logs`,
    { signal, action: "Get job logs" },
  );
  return body.logs;
}

export async function getJobLogFile(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<LogLine[]> {
  const body = await apiRequest<{ logs: LogLine[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/log-file`,
    { signal, action: "Get job log file" },
  );
  return body.logs;
}

export async function getJobMetricsHistory(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<MetricsHistoryPoint[]> {
  const body = await apiRequest<{ points: MetricsHistoryPoint[] }>(
    baseUrl,
    fetcher,
    `/jobs/${id}/metrics-history`,
    { signal, action: "Get job metrics history" },
  );
  return body.points;
}

export async function startTrainingJob(
  baseUrl: string,
  fetcher: Fetcher,
  request: TrainingRequest,
): Promise<JobRecord> {
  const { target, ...config } = request;
  const body = target ? { config, target } : config;
  try {
    return await apiRequest<JobRecord>(baseUrl, fetcher, "/jobs/training", {
      method: "POST",
      body,
      action: "Start training",
    });
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      throw new Error("Another training is already running. Stop it first.");
    }
    throw e;
  }
}

export async function importModel(
  baseUrl: string,
  fetcher: Fetcher,
  source: string,
  name?: string,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, "/jobs/import", {
    method: "POST",
    body: name ? { source, name } : { source },
    action: "Import model",
  });
}

export async function stopJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<JobRecord> {
  return apiRequest<JobRecord>(baseUrl, fetcher, `/jobs/${id}/stop`, {
    method: "POST",
    action: "Stop job",
  });
}

export async function deleteJob(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<void> {
  await apiRequest<void>(baseUrl, fetcher, `/jobs/${id}`, {
    method: "DELETE",
    action: "Delete job",
  });
}

export interface RunnerFlavor {
  name: string;
  pretty_name: string;
  cpu: string;
  ram: string;
  accelerator: string | null;
  unit_cost_usd: number;
  unit_label: string;
}

export interface RunnerHardwareResponse {
  authenticated: boolean;
  username: string | null;
  flavors: RunnerFlavor[];
}

const EMPTY_HARDWARE: RunnerHardwareResponse = {
  authenticated: false,
  username: null,
  flavors: [],
};

export async function listRunnerHardware(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<RunnerHardwareResponse> {
  // Backend returns 401/403 for unauthenticated users; surface as "no flavors"
  // rather than throwing so the UI can render the "log in to use cloud" hint.
  try {
    return await apiRequest<RunnerHardwareResponse>(
      baseUrl,
      fetcher,
      "/jobs/runners/hardware",
      { signal, action: "List runner hardware" },
    );
  } catch (e) {
    if (e instanceof ApiError) return EMPTY_HARDWARE;
    throw e;
  }
}

export interface HubJob {
  id: string;
  created_at: string | null;
  docker_image: string | null;
  space_id: string | null;
  flavor: string | null;
  status: { stage: string; message: string | null } | null;
  owner: string | null;
  url: string;
}

export interface HubModel {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
}

export interface HubJobsResponse {
  authenticated: boolean;
  jobs: HubJob[];
  models: HubModel[];
}

const EMPTY_HUB: HubJobsResponse = {
  authenticated: false,
  jobs: [],
  models: [],
};

export async function listHubJobs(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<HubJobsResponse> {
  // Same graceful degradation as listRunnerHardware.
  try {
    return await apiRequest<HubJobsResponse>(baseUrl, fetcher, "/jobs/hub", {
      signal,
      action: "List hub jobs",
    });
  } catch (e) {
    if (e instanceof ApiError) return EMPTY_HUB;
    throw e;
  }
}

// ---------------------------------------------------------------------------
// SSH configs
// ---------------------------------------------------------------------------

export interface SSHConfig {
  id: string;
  name: string;
  host: string;
  port: number;
  username: string;
  has_password: boolean;
  key_path: string | null;
  remote_python: string;
  remote_workdir: string | null;
  has_wandb_api_key: boolean;
  use_cn_mirror: boolean;
}

export interface SSHConfigInput {
  name: string;
  host: string;
  port: number;
  username: string;
  password?: string;
  key_path?: string;
  remote_python: string;
  remote_workdir?: string;
  wandb_api_key?: string;
  use_cn_mirror?: boolean;
}

export async function listSSHConfigs(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<SSHConfig[]> {
  const body = await apiRequest<{ configs: SSHConfig[] }>(
    baseUrl,
    fetcher,
    "/ssh-configs",
    { signal, action: "List SSH configs" },
  );
  return body.configs;
}

export async function createSSHConfig(
  baseUrl: string,
  fetcher: Fetcher,
  input: SSHConfigInput,
): Promise<SSHConfig> {
  return apiRequest<SSHConfig>(baseUrl, fetcher, "/ssh-configs", {
    method: "POST",
    body: input,
    action: "Create SSH config",
  });
}

export async function updateSSHConfig(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  input: SSHConfigInput,
): Promise<SSHConfig> {
  return apiRequest<SSHConfig>(baseUrl, fetcher, `/ssh-configs/${id}`, {
    method: "PUT",
    body: input,
    action: "Update SSH config",
  });
}

export async function deleteSSHConfig(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
): Promise<void> {
  await apiRequest<void>(baseUrl, fetcher, `/ssh-configs/${id}`, {
    method: "DELETE",
    action: "Delete SSH config",
  });
}

export async function testSSHConfig(
  baseUrl: string,
  fetcher: Fetcher,
  configId: string,
): Promise<{ success: boolean; message: string }> {
  return apiRequest<{ success: boolean; message: string }>(
    baseUrl,
    fetcher,
    "/ssh-configs/test",
    {
      method: "POST",
      body: { config_id: configId },
      action: "Test SSH connection",
    },
  );
}

export interface RemoteEnvIssue {
  code: string;
  severity: "blocking" | "warning" | "info";
  message: string;
  fix: string;
  auto_fixable: boolean;
}

export interface RemoteEnvInfo {
  success: boolean;
  message: string;
  python_version?: string;
  lerobot_installed?: boolean;
  lerobot_version?: string | null;
  cuda_available?: boolean;
  cuda_version?: string | null;
  gpu_name?: string | null;
  disk_free_gb?: number | null;
  policies_ready?: string[];
  policies_missing?: string[];
  // From the unified diagnosis (backend diagnose_env): what is wrong, why,
  // and whether 「安装环境」 can fix it.
  ready?: boolean;
  issues?: RemoteEnvIssue[];
  blocking?: string[];
  warnings?: string[];
  fixable?: string[];
  required_python?: string;
  local_lerobot_version?: string | null;
  remote_python_used?: string;
  torch_version?: string | null;
  ffmpeg?: boolean;
  torchcodec_error?: string | null;
}

export async function checkSSHRemoteEnv(
  baseUrl: string,
  fetcher: Fetcher,
  configId: string,
): Promise<RemoteEnvInfo> {
  return apiRequest<RemoteEnvInfo>(
    baseUrl,
    fetcher,
    "/ssh-configs/check-env",
    {
      method: "POST",
      body: { config_id: configId },
      action: "Check remote environment",
    },
  );
}

export interface InstallProgress {
  step: number;
  total_steps: number;
  label: string;
  status: "running" | "done" | "failed";
  logs: string[];
  message: string;
  // Cause + next step when a step fails (backend step_result(hint=...)).
  hint?: string;
}

export async function installLerobotRemote(
  baseUrl: string,
  fetcher: Fetcher,
  configId: string,
  extra: string = "",
  useCnMirror: boolean = false,
  onProgress?: (update: InstallProgress) => void,
  policies: string[] = [],
): Promise<{ success: boolean; message: string; lines: string[] }> {
  const url = `${baseUrl}/ssh-configs/install-lerobot`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const token = typeof fetcher === "function" ? undefined : (fetcher as { token?: string }).token;
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const resp = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      config_id: configId,
      extra,
      use_cn_mirror: useCnMirror,
      policies,
    }),
  });
  if (!resp.ok || !resp.body) {
    const text = await resp.text().catch(() => "");
    throw new Error(`Install request failed (${resp.status}): ${text}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let last: InstallProgress | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.replace(/^data:\s*/, "").trim();
      if (!line) continue;
      try {
        const update = JSON.parse(line) as InstallProgress;
        last = update;
        onProgress?.(update);
      } catch { /* skip malformed */ }
    }
  }

  if (last) {
    if (last.status === "failed") {
      return { success: false, message: last.message || last.label, lines: last.logs };
    }
    // Only the final step reporting "done" means success. A stream that ends
    // on a running/intermediate event means the connection dropped mid-install.
    if (last.status === "done" && last.step === last.total_steps) {
      return { success: true, message: last.message || last.label, lines: last.logs };
    }
    return {
      success: false,
      message: `Connection lost during install (last: ${last.label})`,
      lines: last.logs,
    };
  }
  return { success: false, message: "No response from server", lines: [] };
}

export interface ActiveTask {
  task_id: string;
  task_type: "ssh_install" | "recording" | "training";
  status: "running" | "done" | "failed";
  label: string;
  message: string;
  started_at: number;
  updated_at: number;
  metadata: Record<string, unknown>;
}

export async function getActiveTasks(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<ActiveTask[]> {
  const res = await apiRequest<{ tasks: ActiveTask[] }>(
    baseUrl,
    fetcher,
    "/tasks/active",
    { action: "Get active tasks" },
  );
  return res.tasks;
}

export async function startJobValidation(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  options?: ValidationOptions,
): Promise<{ validation_state: ValidationState; message: string }> {
  return apiRequest(baseUrl, fetcher, `/jobs/${id}/validate`, {
    method: "POST",
    body: options,
    action: "Start validation",
  });
}

export function evaluationChartUrl(baseUrl: string, jobId: string, relPath: string): string {
  return `${baseUrl}/jobs/${jobId}/evaluation/artifact?path=${encodeURIComponent(relPath)}`;
}

export async function getJobValidation(
  baseUrl: string,
  fetcher: Fetcher,
  id: string,
  signal?: AbortSignal,
): Promise<{ validation_state: ValidationState; validation_results: ValidationResults | null }> {
  return apiRequest(baseUrl, fetcher, `/jobs/${id}/validation`, {
    signal,
    action: "Get validation",
  });
}

// ---------------------------------------------------------------------------
// Three-dimension evaluation scoring
// ---------------------------------------------------------------------------

export type DimensionGrade = "excellent" | "good" | "fair" | "poor" | "bad";

export interface DimensionScore {
  score: number; // 0-100
  grade: DimensionGrade;
  issues: string[]; // 不达标指标列表
}

export interface EvaluationDimensions {
  semantic: DimensionScore;
  visual: DimensionScore;
  action: DimensionScore;
}

const GRADE_THRESHOLDS: { min: number; grade: DimensionGrade }[] = [
  { min: 90, grade: "excellent" },
  { min: 75, grade: "good" },
  { min: 60, grade: "fair" },
  { min: 40, grade: "poor" },
  { min: 0, grade: "bad" },
];

export function scoreToGrade(score: number): DimensionGrade {
  for (const t of GRADE_THRESHOLDS) {
    if (score >= t.min) return t.grade;
  }
  return "bad";
}

export const GRADE_COLORS: Record<DimensionGrade, string> = {
  excellent: "bg-green-500",
  good: "bg-cyan-500",
  fair: "bg-yellow-500",
  poor: "bg-orange-500",
  bad: "bg-red-500",
};

export const GRADE_LABELS: Record<DimensionGrade, string> = {
  excellent: "优",
  good: "良",
  fair: "中",
  poor: "差",
  bad: "拉",
};

/**
 * 计算三维度评估分数。
 * 权重：视觉 50%、动作 40%、语义 10%
 */
export function computeDimensionScores(results: ValidationResults): EvaluationDimensions {
  const semantic = computeSemanticScore(results);
  const visual = computeVisualScore(results);
  const action = computeActionScore(results);
  return { semantic, visual, action };
}

/**
 * 计算综合总分（加权平均）。
 * 权重：视觉 50%、动作 40%、语义 10%
 */
export function computeOverallScore(dimensions: EvaluationDimensions): number {
  return Math.round(
    dimensions.visual.score * 0.5 +
    dimensions.action.score * 0.4 +
    dimensions.semantic.score * 0.1
  );
}

function computeSemanticScore(results: ValidationResults): DimensionScore {
  const issues: string[] = [];
  let totalWeight = 0;
  let weightedSum = 0;

  // 语言敏感度 (100% - 语义维度只有这一个可用指标)
  const langSens = results.probe?.language_sensitivity;
  if (langSens != null) {
    const score = Math.min(100, Math.max(0, langSens * 500));
    weightedSum += score * 1.0;
    totalWeight += 1.0;
    if (score < 60) {
      issues.push(`语言敏感度 ${langSens.toFixed(3)} 偏低，模型对指令变化不敏感`);
    }
  }

  const finalScore = totalWeight > 0 ? weightedSum / totalWeight : 0;
  return {
    score: Math.round(finalScore),
    grade: scoreToGrade(finalScore),
    issues,
  };
}

function computeVisualScore(results: ValidationResults): DimensionScore {
  const issues: string[] = [];
  let totalWeight = 0;
  let weightedSum = 0;

  // 目标定位精度 (35%)
  const peakToBall = results.localization?.mean_peak_to_ball_px;
  if (peakToBall != null) {
    const score = Math.max(0, 100 - peakToBall / 2);
    weightedSum += score * 0.35;
    totalWeight += 0.35;
    if (peakToBall > 120) {
      issues.push(`峰值离球 ${peakToBall.toFixed(0)}px > 120px，定位不准`);
    }
  }

  // ROI 利用率 (25%)
  const roi = results.roi_grounding;
  if (roi) {
    const ratios = Object.values(roi.by_camera).map((v) => v.median_ratio);
    const avgRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
    const score = Math.min(100, Math.max(0, avgRatio * 50));
    weightedSum += score * 0.25;
    totalWeight += 0.25;
    if (avgRatio < roi.threshold) {
      issues.push(`ROI 利用率 ${avgRatio.toFixed(2)}× < 阈值 ${roi.threshold}×，模型未关注目标区域`);
    }
  }

  // 指向反演误差 (20%)
  const aimError = results.aim?.mean_aim_error_model_px;
  if (aimError != null) {
    const score = Math.max(0, 100 - aimError / 2);
    weightedSum += score * 0.2;
    totalWeight += 0.2;
    if (aimError > 100) {
      issues.push(`指向反演误差 ${aimError.toFixed(0)}px，动作与视觉不一致`);
    }
  }

  // 视觉敏感度 (20%)
  const visSens = results.probe?.vision_sensitivity;
  if (visSens != null) {
    const score = Math.min(100, Math.max(0, visSens * 200));
    weightedSum += score * 0.2;
    totalWeight += 0.2;
    if (visSens < 0.1) {
      issues.push(`视觉敏感度 ${visSens.toFixed(3)} 过低`);
    }
  }

  const finalScore = totalWeight > 0 ? weightedSum / totalWeight : 0;
  return {
    score: Math.round(finalScore),
    grade: scoreToGrade(finalScore),
    issues,
  };
}

function computeActionScore(results: ValidationResults): DimensionScore {
  const issues: string[] = [];
  let totalWeight = 0;
  let weightedSum = 0;

  // 相对基线 (30%)
  const relNaive = results.mean_mae_vs_naive;
  if (relNaive != null) {
    const score = Math.max(0, 100 - relNaive * 50);
    weightedSum += score * 0.3;
    totalWeight += 0.3;
    if (relNaive >= 1) {
      issues.push(`相对基线 ${relNaive.toFixed(2)}× ≥ 1，模型不如"瞎猜"`);
    }
  }

  // 开环误差 (25%)
  const mae = results.action_mae;
  if (mae != null) {
    const score = Math.max(0, 100 - mae * 5);
    weightedSum += score * 0.25;
    totalWeight += 0.25;
    if (mae > 10) {
      issues.push(`平均动作误差 ${mae.toFixed(2)}° 过大`);
    }
  }

  // 安全步数 (25%)
  const driftSteps = results.drift_horizon_steps;
  if (driftSteps != null) {
    const score = Math.min(100, driftSteps * 10);
    weightedSum += score * 0.25;
    totalWeight += 0.25;
    if (driftSteps <= 5) {
      issues.push(`安全开环仅 ${driftSteps} 步，需频繁重推理`);
    }
  }

  // 方向一致性 (20%)
  const shiftAgreement = results.probe?.shift_sign_agreement;
  if (shiftAgreement != null) {
    const score = shiftAgreement * 100;
    weightedSum += score * 0.2;
    totalWeight += 0.2;
    if (shiftAgreement < 0.6) {
      issues.push(`方向一致性 ${(shiftAgreement * 100).toFixed(0)}% < 60%`);
    }
  }

  const finalScore = totalWeight > 0 ? weightedSum / totalWeight : 0;
  return {
    score: Math.round(finalScore),
    grade: scoreToGrade(finalScore),
    issues,
  };
}
