import { Fetcher, apiRequest } from "./apiClient";

export type DatasetSource = "local" | "hub" | "both";

export interface DatasetItem {
  repo_id: string;
  last_modified: string | null;
  private: boolean;
  source: DatasetSource;
}

export interface DatasetHealth {
  healthy: boolean;
  total_episodes: number;
  corrupted_episodes: number;
  episodes_healthy: boolean;
  episodes_error: string | null;
  corrupted_data_files: string[];
  total_data_files: number;
  corrupted_video_files: string[];
  total_video_files: number;
  needs_repair: boolean;
  error: string | null;
}

export interface RepairProgress {
  active: boolean;
  phase: string;
  percent: number;
  current: number;
  total: number;
  dataset_repo_id: string;
  error: string | null;
  completed: boolean;
}

export interface DatasetStatistics {
  episode_count: number;
  episode_length: {
    mean: number;
    std: number;
    min: number;
    max: number;
    median: number;
  };
  action_range: Record<string, { min: number; max: number; range: number }>;
  camera_fps: Record<string, number>;
  total_frames: number;
}

export interface DatasetAnomaly {
  episode: number;
  type: string;
  severity: "low" | "medium" | "high";
  message: string;
  joint?: string;
}

export interface DatasetAnalysis {
  success: boolean;
  statistics?: DatasetStatistics;
  anomalies?: DatasetAnomaly[];
  error?: string;
}

export async function listDatasets(
  baseUrl: string,
  fetcher: Fetcher,
  signal?: AbortSignal,
): Promise<DatasetItem[]> {
  return apiRequest<DatasetItem[]>(baseUrl, fetcher, "/datasets", {
    signal,
    action: "List datasets",
  });
}

export async function checkDatasetHealth(
  baseUrl: string,
  fetcher: Fetcher,
  datasetRepoId: string,
): Promise<DatasetHealth & { success: boolean }> {
  return apiRequest<DatasetHealth & { success: boolean }>(
    baseUrl,
    fetcher,
    "/check-dataset-health",
    {
      method: "POST",
      body: { dataset_repo_id: datasetRepoId },
      action: "Check dataset health",
    },
  );
}

export async function startDatasetRepair(
  baseUrl: string,
  fetcher: Fetcher,
  datasetRepoId: string,
  force = false,
): Promise<{ success: boolean; message: string }> {
  return apiRequest<{ success: boolean; message: string }>(
    baseUrl,
    fetcher,
    "/repair-dataset",
    {
      method: "POST",
      body: { dataset_repo_id: datasetRepoId, force },
      action: "Start dataset repair",
    },
  );
}

export async function getRepairProgress(
  baseUrl: string,
  fetcher: Fetcher,
): Promise<RepairProgress | null> {
  return apiRequest<RepairProgress | null>(baseUrl, fetcher, "/repair-progress/current", {
    action: "Get repair progress",
  });
}

export async function getDatasetStatistics(
  baseUrl: string,
  fetcher: Fetcher,
  datasetRepoId: string,
): Promise<DatasetAnalysis> {
  return apiRequest<DatasetAnalysis>(
    baseUrl,
    fetcher,
    "/dataset-statistics",
    {
      method: "POST",
      body: { dataset_repo_id: datasetRepoId },
      action: "Get dataset statistics",
    },
  );
}

// ---------------------------------------------------------------------------
// Local dataset viewer
// ---------------------------------------------------------------------------

export interface ViewerEpisode {
  episode_index: number;
  length: number;
  tasks: string[];
}

export interface DatasetEpisodesInfo {
  episodes: ViewerEpisode[];
  cameras: string[];
  fps: number;
}

export interface EpisodeSeries {
  joint_names: string[];
  frames: number[];
  action: number[][];
  state: number[][];
}

export async function getDatasetEpisodes(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  signal?: AbortSignal,
): Promise<DatasetEpisodesInfo> {
  return apiRequest<DatasetEpisodesInfo>(
    baseUrl,
    fetcher,
    `/dataset-episodes?repo_id=${encodeURIComponent(repoId)}`,
    { signal, action: "Get dataset episodes" },
  );
}

export function datasetVideoUrl(baseUrl: string, repoId: string, episode: number, camera: string): string {
  return `${baseUrl}/dataset-video?repo_id=${encodeURIComponent(repoId)}&episode=${episode}&camera=${encodeURIComponent(camera)}`;
}

export async function getDatasetVideoWindow(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  episode: number,
  camera: string,
  signal?: AbortSignal,
): Promise<{ from_timestamp: number; to_timestamp: number }> {
  return apiRequest(
    baseUrl,
    fetcher,
    `/dataset-video-window?repo_id=${encodeURIComponent(repoId)}&episode=${episode}&camera=${encodeURIComponent(camera)}`,
    { signal, action: "Get video window" },
  );
}

export async function getDatasetEpisodeSeries(
  baseUrl: string,
  fetcher: Fetcher,
  repoId: string,
  episode: number,
  signal?: AbortSignal,
): Promise<EpisodeSeries> {
  return apiRequest<EpisodeSeries>(
    baseUrl,
    fetcher,
    `/dataset-episode-series?repo_id=${encodeURIComponent(repoId)}&episode=${episode}`,
    { signal, action: "Get episode series" },
  );
}

// ---------------------------------------------------------------------------
// Augmentation preview
// ---------------------------------------------------------------------------

export interface AugmentPreviewConfig {
  repo_id: string;
  n_frames?: number;
  augmentation_max_num_transforms?: number;
  augmentation_random_order?: boolean;
  augmentation_brightness?: boolean;
  augmentation_contrast?: boolean;
  augmentation_saturation?: boolean;
  augmentation_hue?: boolean;
  augmentation_sharpness?: boolean;
  augmentation_affine?: boolean;
}

export interface AugmentPreviewResult {
  camera: string;
  original: string[];
  augmented: string[];
  transforms: string[];
  per_transform: { name: string; image: string }[];
}

export async function getAugmentPreview(
  baseUrl: string,
  fetcher: Fetcher,
  config: AugmentPreviewConfig,
): Promise<AugmentPreviewResult> {
  return apiRequest<AugmentPreviewResult>(baseUrl, fetcher, "/augment-preview", {
    method: "POST",
    body: config,
    action: "Preview augmentations",
  });
}
