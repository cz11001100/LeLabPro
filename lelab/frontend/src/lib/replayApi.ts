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
