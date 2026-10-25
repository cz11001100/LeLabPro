import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ArrowLeft,
  Upload as UploadIcon,
  Database,
  Tag,
  Eye,
  EyeOff,
  ExternalLink,
  CheckCircle,
  AlertCircle,
  Loader2,
  Trash2,
  Wrench,
  AlertTriangle,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { DatasetSource, checkDatasetHealth, startDatasetRepair } from "@/lib/replayApi";
import type { DatasetHealth, RepairProgress } from "@/lib/replayApi";
import DatasetHealthWarning from "@/components/DatasetHealthWarning";
import DatasetSummaryCard from "@/components/DatasetSummaryCard";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface DatasetInfo {
  dataset_repo_id: string;
  single_task: string;
  num_episodes: number;
  saved_episodes?: number;
  session_elapsed_seconds?: number;
  fps?: number;
  total_frames?: number;
  robot_type?: string;
  source?: DatasetSource;
  episodes?: { index: number; length: number }[];
}

interface UploadConfig {
  tags: string[];
  private: boolean;
}

import MergeSection from "@/components/dataset/MergeSection";

const Upload = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  // Get initial dataset info from navigation state
  const initialDatasetInfo = location.state?.datasetInfo as DatasetInfo;

  // State for actual dataset info (will be loaded from backend)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isLoadingDatasetInfo, setIsLoadingDatasetInfo] = useState(true);

  // Upload configuration state
  const [uploadConfig, setUploadConfig] = useState<UploadConfig>({
    tags: ["robotics", "lerobot"],
    private: false,
  });

  const [tagsInput, setTagsInput] = useState(uploadConfig.tags.join(", "));
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadProgress, setUploadProgress] = useState({
    phase: "",
    percent: 0,
    current: 0,
    total: 0,
  });
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedEpisodes, setSelectedEpisodes] = useState<Set<number>>(new Set());
  const [showDeleteEpisodesConfirm, setShowDeleteEpisodesConfirm] = useState(false);
  const [isDeletingEpisode, setIsDeletingEpisode] = useState(false);

  // Dataset health check state
  const [health, setHealth] = useState<DatasetHealth | null>(null);
  const [checkingHealth, setCheckingHealth] = useState(false);
  const [isRepairing, setIsRepairing] = useState(false);
  const [repairProgress, setRepairProgress] = useState<RepairProgress | null>(null);

  const loadDatasetInfo = React.useCallback(async () => {
    if (!initialDatasetInfo?.dataset_repo_id) {
      // Arrived without a dataset in hand (deep link, or just here to merge
      // datasets): render the page with the dataset-independent sections.
      setIsLoadingDatasetInfo(false);
      return;
    }

    try {
      const response = await fetchWithHeaders(`${baseUrl}/dataset-info`, {
        method: "POST",
        body: JSON.stringify({
          dataset_repo_id: initialDatasetInfo.dataset_repo_id,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        setDatasetInfo({
          ...data,
          saved_episodes: data.num_episodes,
          session_elapsed_seconds:
            initialDatasetInfo.session_elapsed_seconds || 0,
        });
      } else {
        setDatasetInfo(initialDatasetInfo);
      }
    } catch (error) {
      console.error("Error loading dataset info:", error);
      setDatasetInfo(initialDatasetInfo);
    } finally {
      setIsLoadingDatasetInfo(false);
    }
  }, [initialDatasetInfo, navigate, toast, baseUrl, fetchWithHeaders]);

  // Load actual dataset information from backend
  React.useEffect(() => {
    loadDatasetInfo();
  }, [loadDatasetInfo]);

  // Check dataset health after loading info
  React.useEffect(() => {
    if (!datasetInfo?.dataset_repo_id) return;
    setCheckingHealth(true);
    checkDatasetHealth(baseUrl, fetchWithHeaders, datasetInfo.dataset_repo_id)
      .then((result) => {
        if (result.success) {
          setHealth(result);
        }
      })
      .catch((err) => console.error("Health check failed:", err))
      .finally(() => setCheckingHealth(false));
  }, [datasetInfo?.dataset_repo_id, baseUrl, fetchWithHeaders]);

  const openInHubViewer = (repoId: string) => {
    const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(`/${repoId}`)}`;
    // The user owns/manages the dataset (it appears under their hub
    // listing), so login-redirect always works whether public or
    // private. Avoids passing `private` through navigation state.
    const target = `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`;
    window.open(target, "_blank", "noopener,noreferrer");
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const handleUploadToHub = async () => {
    if (!datasetInfo) return;

    setIsUploading(true);
    setUploadProgress({ phase: "Starting upload...", percent: 0, current: 0, total: 0 });

    const eventSource = new EventSource(`${baseUrl}/upload-progress`);
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setUploadProgress({
        phase: data.phase || "",
        percent: data.percent || 0,
        current: data.current || 0,
        total: data.total || 0,
      });

      if (data.completed || data.error) {
        eventSource.close();
      }
    };
    eventSource.onerror = () => {
      eventSource.close();
    };

    try {
      // Parse tags from input
      const tags = tagsInput
        .split(",")
        .map((tag) => tag.trim())
        .filter((tag) => tag.length > 0);

      const response = await fetchWithHeaders(`${baseUrl}/upload-dataset`, {
        method: "POST",
        body: JSON.stringify({
          dataset_repo_id: datasetInfo.dataset_repo_id,
          tags,
          private: uploadConfig.private,
        }),
      });

      const data = await response.json();
      eventSource.close();

      if (response.ok && data.success) {
        setUploadSuccess(true);
        setUploadProgress({ phase: "Upload complete!", percent: 100, current: 0, total: 0 });
        toast({
          title: "Upload Successful!",
          description: `Dataset ${datasetInfo.dataset_repo_id} has been uploaded to HuggingFace Hub.`,
        });
      } else {
        const fallback = "Failed to upload dataset to HuggingFace Hub.";
        setUploadProgress({ phase: "Upload failed", percent: 0, current: 0, total: 0 });
        toast({
          title: "Upload Failed",
          description: data.docs_url ? (
            <span>
              {data.message || fallback}{" "}
              <a
                href={data.docs_url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline font-medium"
              >
                Open setup guide
              </a>
            </span>
          ) : (
            data.message || fallback
          ),
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error uploading dataset:", error);
      setUploadProgress({ phase: "Upload failed", percent: 0, current: 0, total: 0 });
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleSkipUpload = () => {
    toast({
      title: "Upload Skipped",
      description: "Dataset saved locally. You can upload it manually later.",
    });
    navigate("/");
  };

  const handleDeleteDataset = async () => {
    if (!datasetInfo) return;
    setIsDeleting(true);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/delete-dataset`, {
        method: "POST",
        body: JSON.stringify({ dataset_repo_id: datasetInfo.dataset_repo_id }),
      });
      const data = await response.json();
      if (response.ok && data.success) {
        toast({
          title: "Dataset Deleted",
          description: `${datasetInfo.dataset_repo_id} has been removed from disk.`,
        });
        navigate("/");
      } else {
        toast({
          title: "Delete Failed",
          description: data.message || "Could not delete the dataset.",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleDeleteSelectedEpisodes = async () => {
    if (!datasetInfo || selectedEpisodes.size === 0) return;
    const repoId = datasetInfo.dataset_repo_id;
    const indices = [...selectedEpisodes].sort((a, b) => a - b);
    setIsDeletingEpisode(true);
    setShowDeleteEpisodesConfirm(false);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/delete-episode`, {
        method: "POST",
        body: JSON.stringify({
          dataset_repo_id: repoId,
          episode_indices: indices,
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        toast({
          title: "Delete Failed",
          description: data.message || "Could not delete the episodes.",
          variant: "destructive",
        });
        setIsDeletingEpisode(false);
        return;
      }

      if (!data.async) {
        // Legacy v2 layout finishes inside the request, so there is nothing to stream.
        toast({ title: "Episodes Deleted", description: data.message });
        setSelectedEpisodes(new Set());
        await loadDatasetInfo();
        setIsDeletingEpisode(false);
        return;
      }

      // The rebuild runs on a worker thread and the POST returns immediately, so
      // progress arrives over the same SSE stream the repair flow uses.
      setRepairProgress({
        active: true,
        phase: "Starting...",
        percent: 0,
        current: 0,
        total: 0,
        dataset_repo_id: repoId,
        error: null,
        completed: false,
      });
      const eventSource = new EventSource(`${baseUrl}/repair-progress`);
      eventSource.onmessage = (event) => {
        const progress: RepairProgress = JSON.parse(event.data);
        setRepairProgress(progress);
        if (progress.completed || progress.error) {
          eventSource.close();
          setIsDeletingEpisode(false);
          if (progress.error) {
            toast({ title: "Delete Failed", description: progress.error, variant: "destructive" });
          } else {
            toast({ title: "Episodes Deleted", description: progress.phase });
          }
          setSelectedEpisodes(new Set());
          loadDatasetInfo();
          checkDatasetHealth(baseUrl, fetchWithHeaders, repoId).then(setHealth);
        }
      };
      eventSource.onerror = () => {
        eventSource.close();
        setIsDeletingEpisode(false);
        toast({
          title: "Progress Updates Lost",
          description: "The rebuild may still be running. Reload the dataset to check.",
          variant: "destructive",
        });
        loadDatasetInfo();
      };
    } catch (error) {
      toast({
        title: "Connection Error",
        description: "Could not connect to the backend server.",
        variant: "destructive",
      });
      setIsDeletingEpisode(false);
    }
  };

  const toggleEpisode = (index: number) => {
    setSelectedEpisodes((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const episodeCount = datasetInfo
    ? datasetInfo.saved_episodes || datasetInfo.num_episodes
    : 0;

  const fps = datasetInfo?.fps || 0;
  const episodeRows =
    datasetInfo?.episodes && datasetInfo.episodes.length > 0
      ? datasetInfo.episodes
      : Array.from({ length: episodeCount }, (_, index) => ({ index, length: 0 }));
  const sortedLengths = episodeRows
    .map((e) => e.length)
    .filter((n) => n > 0)
    .sort((a, b) => a - b);
  const medianLength = sortedLengths.length ? sortedLengths[Math.floor(sortedLengths.length / 2)] : 0;

  const allEpisodesSelected = episodeCount > 0 && selectedEpisodes.size === episodeCount;

  const toggleAllEpisodes = () => {
    setSelectedEpisodes(
      allEpisodesSelected
        ? new Set()
        : new Set(Array.from({ length: episodeCount }, (_, i) => i))
    );
  };

  const handleRepairDataset = async () => {
    if (!datasetInfo) return;
    setIsRepairing(true);
    setRepairProgress({ active: true, phase: "Starting...", percent: 0, current: 0, total: 0, dataset_repo_id: datasetInfo.dataset_repo_id, error: null, completed: false });

    try {
      const response = await fetchWithHeaders(`${baseUrl}/repair-dataset`, {
        method: "POST",
        body: JSON.stringify({ dataset_repo_id: datasetInfo.dataset_repo_id }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        toast({ title: "Repair Failed", description: data.message || "Could not start repair", variant: "destructive" });
        setIsRepairing(false);
        return;
      }

      // Poll repair progress via SSE
      const eventSource = new EventSource(`${baseUrl}/repair-progress`);
      eventSource.onmessage = (event) => {
        const progress: RepairProgress = JSON.parse(event.data);
        setRepairProgress(progress);
        if (progress.completed || progress.error) {
          eventSource.close();
          setIsRepairing(false);
          if (progress.error) {
            toast({ title: "Repair Failed", description: progress.error, variant: "destructive" });
          } else {
            toast({ title: "Repair Complete", description: "Dataset has been repaired successfully." });
            // Re-check health and reload info
            loadDatasetInfo();
            checkDatasetHealth(baseUrl, fetchWithHeaders, datasetInfo.dataset_repo_id).then(setHealth);
          }
        }
      };
      eventSource.onerror = () => {
        eventSource.close();
        setIsRepairing(false);
      };
    } catch (error) {
      toast({ title: "Connection Error", description: "Could not connect to the backend server.", variant: "destructive" });
      setIsRepairing(false);
    }
  };

  // Show loading state while fetching dataset info
  if (isLoadingDatasetInfo) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-lg">Loading dataset information...</p>
        </div>
      </div>
    );
  }

  // No dataset picked (deep link, or arrived just to merge): merging works on a
  // selection of its own, so don't hold it behind a dataset detail view.
  if (!datasetInfo) {
    return (
      <div className="min-h-screen bg-black text-white p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-4 mb-6">
            <Button
              onClick={() => navigate("/")}
              variant="outline"
              className="border-gray-500 text-gray-300 hover:text-white"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
            <h1 className="text-3xl font-bold">Dataset</h1>
          </div>
          <MergeSection />
        </div>
      </div>
    );
  }

  const isAlreadyOnHub = datasetInfo.source === "both";

  return (
    <div className="min-h-screen bg-black text-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <Button
              onClick={() => navigate("/")}
              variant="outline"
              className="border-gray-500 hover:border-gray-200 text-gray-300 hover:text-white"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
            <Button
              onClick={() => setShowDeleteConfirm(true)}
              variant="outline"
              size="icon"
              disabled={isDeleting}
              aria-label="Delete dataset from disk"
              className="border-red-500/40 text-red-400 hover:border-red-400 hover:text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-3">
            {uploadSuccess ? (
              <CheckCircle className="w-8 h-8 text-green-500" />
            ) : (
              <Database className="w-8 h-8 text-blue-500" />
            )}
            <h1 className="text-3xl font-bold">
              {uploadSuccess ? "Upload Complete" : "Dataset"}
            </h1>
          </div>
        </div>

        {/* Success State */}
        {uploadSuccess && (
          <div className="bg-green-900/20 border border-green-600 rounded-lg p-6 mb-8">
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle className="w-6 h-6 text-green-500" />
              <h2 className="text-xl font-semibold text-green-400">
                Successfully Uploaded!
              </h2>
            </div>
            <p className="text-gray-300 mb-4">
              Your dataset has been uploaded to HuggingFace Hub and is now
              available for training and sharing.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button
                onClick={() => {
                  const spacePath = `/spaces/lerobot/visualize_dataset?path=${encodeURIComponent(
                    `/${datasetInfo.dataset_repo_id}`
                  )}`;
                  const target = uploadConfig.private
                    ? `https://huggingface.co/login?next=${encodeURIComponent(spacePath)}`
                    : `https://huggingface.co${spacePath}`;
                  window.open(target, "_blank", "noopener,noreferrer");
                }}
                className="bg-blue-500 hover:bg-blue-600 text-white"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View on HuggingFace Hub
              </Button>
              <Button
                onClick={() =>
                  navigate("/training", {
                    state: { datasetRepoId: datasetInfo.dataset_repo_id },
                  })
                }
                className="bg-purple-500 hover:bg-purple-600 text-white"
              >
                Start Training
              </Button>
            </div>
          </div>
        )}

        {/* Upload Form */}
        {!uploadSuccess && (
          <>
            {/* Dataset Summary */}
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-8">
              <h2 className="text-xl font-semibold text-white mb-4">
                Dataset Summary
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div>
                    <span className="text-gray-400">Repository ID:</span>
                    <p className="text-white font-mono text-lg">
                      {datasetInfo.dataset_repo_id}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-400">Task:</span>
                    <p className="text-white">{datasetInfo.single_task}</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div>
                    <span className="text-gray-400">Episodes Recorded:</span>
                    <p className="text-white text-2xl font-bold text-green-400">
                      {datasetInfo.saved_episodes || datasetInfo.num_episodes}
                    </p>
                    {datasetInfo.total_frames && (
                      <p className="text-gray-400 text-sm">
                        {datasetInfo.total_frames} total frames
                      </p>
                    )}
                  </div>
                  <div>
                    <span className="text-gray-400">Session Duration:</span>
                    <p className="text-white">
                      {formatDuration(datasetInfo.session_elapsed_seconds || 0)}
                    </p>
                    {datasetInfo.fps && (
                      <p className="text-gray-400 text-sm">
                        {datasetInfo.fps} FPS
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Dataset Quality Analysis */}
            {datasetInfo.dataset_repo_id && (
              <div className="mb-8">
                <DatasetSummaryCard datasetRepoId={datasetInfo.dataset_repo_id} />
              </div>
            )}

            {/* Dataset Health / Corruption Warning */}
            {checkingHealth && (
              <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 mb-8 flex items-center gap-3">
                <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
                <span className="text-gray-300">Checking dataset health...</span>
              </div>
            )}

            {health && !health.healthy && (
              <DatasetHealthWarning
                health={health}
                isRepairing={isRepairing}
                repairProgress={repairProgress}
                onRepair={handleRepairDataset}
                onDelete={() => setShowDeleteConfirm(true)}
              />
            )}

            {health && health.healthy && (
              <div className="bg-green-900/20 border border-green-600/50 rounded-lg p-4 mb-8 flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span className="text-green-300 text-sm">
                  Dataset is healthy - {health.total_episodes} episodes, all files readable
                </span>
              </div>
            )}

            {/* Episodes List */}
            {episodeCount > 0 && (
              <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-8">
                <div className="flex items-center justify-between mb-4 gap-4">
                  <div className="flex items-center gap-3">
                    <h2 className="text-xl font-semibold text-white">Episodes</h2>
                    <span className="text-sm text-gray-400">
                      {episodeCount} episode{episodeCount !== 1 ? "s" : ""}
                      {selectedEpisodes.size > 0 && (
                        <span className="text-red-400"> · {selectedEpisodes.size} selected</span>
                      )}
                    </span>
                  </div>
                  <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                      <Checkbox
                        checked={allEpisodesSelected}
                        onCheckedChange={toggleAllEpisodes}
                        disabled={isDeletingEpisode}
                        aria-label="Select all episodes"
                      />
                      Select all
                    </label>
                    <Button
                      size="sm"
                      variant="destructive"
                      disabled={isDeletingEpisode || selectedEpisodes.size === 0}
                      onClick={() => setShowDeleteEpisodesConfirm(true)}
                    >
                      {isDeletingEpisode ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4 mr-2" />
                      )}
                      Delete selected
                    </Button>
                  </div>
                </div>
                {isDeletingEpisode && repairProgress && (
                  <div className="mb-4 p-3 bg-gray-900/60 rounded-lg border border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                        <span className="text-sm text-gray-300">{repairProgress.phase}</span>
                      </div>
                      <span className="text-sm text-blue-400 font-mono">
                        {repairProgress.percent}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                      <div
                        className="bg-blue-500 h-2.5 rounded-full transition-all duration-500 ease-out"
                        style={{ width: `${repairProgress.percent}%` }}
                      />
                    </div>
                    {repairProgress.total > 0 && (
                      <p className="text-xs text-gray-500 mt-1">
                        {repairProgress.current} / {repairProgress.total}
                      </p>
                    )}
                  </div>
                )}
                <div className="space-y-2 max-h-96 overflow-y-auto">
                  {episodeRows.map(({ index, length }) => {
                    const isShort = length > 0 && medianLength > 0 && length < medianLength * 0.5;
                    return (
                      <div
                        key={index}
                        className={`flex items-center gap-4 bg-gray-800 rounded-lg px-4 py-3 border transition-colors ${
                          selectedEpisodes.has(index)
                            ? "border-red-500/60"
                            : "border-gray-700 hover:border-gray-600"
                        }`}
                      >
                        <Checkbox
                          checked={selectedEpisodes.has(index)}
                          onCheckedChange={() => toggleEpisode(index)}
                          disabled={isDeletingEpisode}
                          aria-label={`Select episode ${index}`}
                        />
                        <span className="text-green-400 font-mono text-sm">
                          #{index.toString().padStart(3, "0")}
                        </span>
                        <span className="text-gray-300 text-sm">Episode {index}</span>
                        {length > 0 && (
                          <span className={`text-xs ${isShort ? "text-amber-400" : "text-gray-500"}`}>
                            {length} frames
                            {fps ? ` · ${(length / fps).toFixed(1)}s` : ""}
                            {isShort ? " · under half the median" : ""}
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Upload Configuration */}
            {!isAlreadyOnHub && (
              <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-8">
                <h2 className="text-xl font-semibold text-white mb-6">
                  Upload Configuration
                </h2>

                <div className="space-y-6">
                  {/* Tags */}
                  <div>
                    <Label htmlFor="tags" className="text-gray-300 mb-2 block">
                      Tags (comma-separated)
                    </Label>
                    <Input
                      id="tags"
                      value={tagsInput}
                      onChange={(e) => setTagsInput(e.target.value)}
                      placeholder="robotics, lerobot, manipulation"
                      className="bg-gray-800 border-gray-600 text-white"
                    />
                    <p className="text-sm text-gray-500 mt-1">
                      Tags help others discover your dataset on HuggingFace Hub
                    </p>
                  </div>

                  {/* Privacy Setting */}
                  <div className="flex items-center space-x-3">
                    <Checkbox
                      id="private"
                      checked={uploadConfig.private}
                      onCheckedChange={(checked) =>
                        setUploadConfig({
                          ...uploadConfig,
                          private: checked as boolean,
                        })
                      }
                    />
                    <div className="flex items-center gap-2">
                      {uploadConfig.private ? (
                        <EyeOff className="w-4 h-4 text-gray-400" />
                      ) : (
                        <Eye className="w-4 h-4 text-gray-400" />
                      )}
                      <Label htmlFor="private" className="text-gray-300">
                        Make dataset private
                      </Label>
                    </div>
                  </div>
                  <p className="text-sm text-gray-500 ml-6">
                    {uploadConfig.private
                      ? "Only you will be able to access this dataset"
                      : "Dataset will be publicly accessible on HuggingFace Hub"}
                  </p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              {isAlreadyOnHub ? (
                <>
                  <div className="w-full flex items-center justify-center gap-2 text-green-400 mb-2">
                    <CheckCircle className="w-5 h-5" />
                    <span className="text-lg">数据集已同步到 HuggingFace Hub</span>
                  </div>
                  <Button
                    onClick={() => openInHubViewer(datasetInfo.dataset_repo_id)}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-4 px-8 text-lg"
                  >
                    <ExternalLink className="w-5 h-5 mr-2" />
                    View on Hugging Face Hub
                  </Button>
                  <Button
                    onClick={handleUploadToHub}
                    disabled={isUploading}
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-800 py-4 px-8 text-lg"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Syncing...
                      </>
                    ) : (
                      <>
                        <UploadIcon className="w-5 h-5 mr-2" />
                        Re-upload Changes
                      </>
                    )}
                  </Button>
                </>
              ) : (
                <>
                  <Button
                    onClick={handleUploadToHub}
                    disabled={isUploading}
                    className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-4 px-8 text-lg"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                        Uploading to Hub...
                      </>
                    ) : (
                      <>
                        <UploadIcon className="w-5 h-5 mr-2" />
                        Upload to HuggingFace Hub
                      </>
                    )}
                  </Button>

                  <Button
                    onClick={handleSkipUpload}
                    disabled={isUploading}
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white py-4 px-8 text-lg"
                  >
                    Skip Upload
                  </Button>
                </>
              )}
            </div>

            {/* Upload Progress Bar */}
            {isUploading && uploadProgress && (
              <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-300">{uploadProgress.phase || "Preparing..."}</span>
                  <span className="text-sm text-blue-400 font-mono">{uploadProgress.percent}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className="bg-blue-500 h-3 rounded-full transition-all duration-500 ease-out"
                    style={{ width: `${uploadProgress.percent}%` }}
                  />
                </div>
                {uploadProgress.total > 0 && (
                  <p className="text-xs text-gray-500 mt-2">
                    {uploadProgress.current} / {uploadProgress.total} files
                  </p>
                )}
              </div>
            )}

            {/* Info Box */}
            {!isAlreadyOnHub && (
              <div className="mt-8 p-4 bg-blue-900/20 border border-blue-600 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-blue-400 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-blue-400 mb-2">
                      About HuggingFace Hub Upload
                    </h3>
                    <ul className="text-sm text-gray-300 space-y-1">
                      <li>
                        • Your dataset will be uploaded to HuggingFace Hub for
                        sharing and collaboration
                      </li>
                      <li>
                        • You need to be logged in to HuggingFace CLI on the
                        server
                      </li>
                      <li>
                        • Uploaded datasets can be used for training models and
                        sharing with the community
                      </li>
                      <li>
                        • You can always upload manually later using the
                        HuggingFace CLI
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </>
        )}

        <MergeSection />
      </div>

      <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete dataset from disk?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              This permanently removes <span className="font-mono text-white">{datasetInfo.dataset_repo_id}</span> from your local cache. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              Keep dataset
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteDataset}
              disabled={isDeleting}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              {isDeleting ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={showDeleteEpisodesConfirm}
        onOpenChange={(open) => !open && setShowDeleteEpisodesConfirm(false)}
      >
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete {selectedEpisodes.size} episode{selectedEpisodes.size !== 1 ? "s" : ""}?
            </AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              Removing {[...selectedEpisodes]
                .sort((a, b) => a - b)
                .slice(0, 12)
                .join(", ")}
              {selectedEpisodes.size > 12 ? ` and ${selectedEpisodes.size - 12} more` : ""}. Files
              holding only surviving episodes are copied as-is; just the files that mix kept and
              deleted episodes are re-encoded. The original dataset is kept in a sibling folder
              {" "}
              <code className="text-gray-300">
                {(datasetInfo?.dataset_repo_id.split("/").pop() || "dataset")}__bak_del*
              </code>
              , so this is reversible until you delete that backup.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteSelectedEpisodes}
              disabled={isDeletingEpisode}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              {isDeletingEpisode ? "Rebuilding…" : `Delete ${selectedEpisodes.size}`}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Upload;
