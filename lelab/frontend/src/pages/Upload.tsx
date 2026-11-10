import React, { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Checkbox } from "@/components/ui/checkbox";
import {
  ArrowLeft,
  Upload as UploadIcon,
  Database,
  ExternalLink,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  ListVideo,
  Loader2,
  Trash2,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import {
  DatasetSource,
  checkDatasetHealth,
  startDatasetRepair,
} from "@/lib/replayApi";
import type { DatasetHealth, RepairProgress } from "@/lib/replayApi";
import DatasetHealthWarning from "@/components/DatasetHealthWarning";
import DatasetSummaryCard from "@/components/DatasetSummaryCard";
import { compareRolloutProgress } from "@/lib/rolloutEvalApi";
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
  tasks: { task: string; episodes: number }[];
  num_episodes: number;
  saved_episodes?: number;
  fps?: number;
  robot_type?: string;
  source?: DatasetSource;
  episodes?: { index: number; length: number }[];
}

import MergeSection from "@/components/dataset/MergeSection";
import RolloutScoreCard from "@/components/jobs/RolloutScoreCard";
import SarmTrainCard from "@/components/jobs/SarmTrainCard";

const Upload = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  // Get initial dataset info from navigation state
  const initialDatasetInfo = location.state?.datasetInfo as DatasetInfo;
  // A "打分" jump from a dataset card preselects it in the scoring section.
  const scoreRepoId = (location.state?.scoreRepoId as string | undefined) ?? "";
  const [episodesOpen, setEpisodesOpen] = useState(false);

  // Arriving from a dataset card's 打分 button: land on the scoring section.
  React.useEffect(() => {
    if (!scoreRepoId) return;
    const timer = window.setTimeout(
      () =>
        document
          .getElementById("dataset-scoring")
          ?.scrollIntoView({ behavior: "smooth", block: "start" }),
      400,
    );
    return () => window.clearTimeout(timer);
  }, [scoreRepoId]);

  // State for actual dataset info (will be loaded from backend)
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [isLoadingDatasetInfo, setIsLoadingDatasetInfo] = useState(true);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState({
    phase: "",
    percent: 0,
    current: 0,
    total: 0,
  });
  const progressStream = React.useRef<EventSource | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [selectedEpisodes, setSelectedEpisodes] = useState<Set<number>>(
    new Set(),
  );
  const [showDeleteEpisodesConfirm, setShowDeleteEpisodesConfirm] =
    useState(false);
  const [isDeletingEpisode, setIsDeletingEpisode] = useState(false);

  // Dataset health check state
  const [health, setHealth] = useState<DatasetHealth | null>(null);  const [checkingHealth, setCheckingHealth] = useState(false);
  const [isRepairing, setIsRepairing] = useState(false);
  const [repairProgress, setRepairProgress] = useState<RepairProgress | null>(
    null,
  );

  // Which scorers already have a progress file for this dataset.
  const [scoredBackends, setScoredBackends] = useState<string[]>([]);

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
        setDatasetInfo({ ...data, saved_episodes: data.num_episodes });
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

  React.useEffect(() => {
    const repoId = datasetInfo?.dataset_repo_id;
    if (!repoId) {
      setScoredBackends([]);
      return;
    }
    let cancelled = false;
    compareRolloutProgress(baseUrl, fetchWithHeaders, repoId)
      .then((c) => {
        if (cancelled) return;
        // `available` means both parquets exist; otherwise the payload lists
        // whichever single backend has scored.
        setScoredBackends(c.available ? ["robometer", "sarm"] : c.backends ?? []);
      })
      .catch(() => {
        if (!cancelled) setScoredBackends([]);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetInfo?.dataset_repo_id, baseUrl, fetchWithHeaders]);

  const followUploadProgress = React.useCallback(() => {
    if (progressStream.current) return;
    const stream = new EventSource(`${baseUrl}/upload-progress`);
    stream.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setUploadProgress({
        phase: data.phase || "",
        percent: data.percent || 0,
        current: data.current || 0,
        total: data.total || 0,
      });
      if (data.completed) {
        setUploadSuccess(true);
        setUploadError(null);
        setIsUploading(false);
      } else if (data.error) {
        setUploadError(String(data.error));
        setIsUploading(false);
      }
      if (data.completed || data.error) {
        stream.close();
        progressStream.current = null;
      }
    };
    stream.onerror = () => {
      stream.close();
      progressStream.current = null;
    };
    progressStream.current = stream;
  }, [baseUrl]);

  React.useEffect(() => () => progressStream.current?.close(), []);

  // The upload runs on a server thread, so a refresh or a trip back home must not
  // pretend it stopped — re-attach to one that is still going.
  React.useEffect(() => {
    const repoId = initialDatasetInfo?.dataset_repo_id;
    if (!repoId) return;
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/upload-progress/current`)
      .then((r) => r.json())
      .then((p) => {
        if (cancelled || !p?.active || p.dataset_repo_id !== repoId) return;
        setIsUploading(true);
        setUploadProgress({
          phase: p.phase || "",
          percent: p.percent || 0,
          current: p.current || 0,
          total: p.total || 0,
        });
        followUploadProgress();
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [
    initialDatasetInfo?.dataset_repo_id,
    baseUrl,
    fetchWithHeaders,
    followUploadProgress,
  ]);

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

  const handleUploadToHub = async () => {
    if (!datasetInfo) return;

    setIsUploading(true);
    setUploadError(null);
    setUploadSuccess(false);
    setUploadProgress({ phase: "准备上传…", percent: 0, current: 0, total: 0 });
    followUploadProgress();

    const stopFollowing = () => {
      progressStream.current?.close();
      progressStream.current = null;
    };

    try {
      const response = await fetchWithHeaders(`${baseUrl}/upload-dataset`, {
        method: "POST",
        body: JSON.stringify({
          dataset_repo_id: datasetInfo.dataset_repo_id,
          tags: ["robotics", "lerobot"],
          private: false,
        }),
      });

      const data = await response.json();
      if (response.ok && data.success) {
        // The stream's completed event normally gets here first; this covers the
        // case where the subscription never came up.
        stopFollowing();
        setUploadSuccess(true);
        setUploadProgress({
          phase: "上传完成",
          percent: 100,
          current: 0,
          total: 0,
        });
        toast({
          title: "上传成功",
          description: `${datasetInfo.dataset_repo_id} 已上传到 HuggingFace Hub。`,
        });
        loadDatasetInfo();
      } else {
        stopFollowing();
        setUploadError(data.message || "上传失败");
        toast({
          title: "上传失败",
          description: data.docs_url ? (
            <span>
              {data.message || "上传失败"}{" "}
              <a
                href={data.docs_url}
                target="_blank"
                rel="noopener noreferrer"
                className="underline font-medium"
              >
                查看配置说明
              </a>
            </span>
          ) : (
            data.message || "上传失败"
          ),
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Error uploading dataset:", error);
      stopFollowing();
      setUploadError("无法连接后端服务");
      toast({
        title: "连接错误",
        description: "无法连接后端服务。",
        variant: "destructive",
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleSkipUpload = () => {
    toast({
      title: "已跳过上传",
      description: "数据集已保存在本地，之后仍可手动上传。",
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
          title: "数据集已删除",
          description: `${datasetInfo.dataset_repo_id} 已从磁盘删除。`,
        });
        navigate("/");
      } else {
        toast({
          title: "删除失败",
          description: data.message || "无法删除数据集。",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "连接错误",
        description: "无法连接后端服务。",
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
          title: "删除失败",
          description: data.message || "无法删除所选分集。",
          variant: "destructive",
        });
        setIsDeletingEpisode(false);
        return;
      }

      if (!data.async) {
        // Legacy v2 layout finishes inside the request, so there is nothing to stream.
        toast({ title: "已删除分集", description: data.message });
        setSelectedEpisodes(new Set());
        await loadDatasetInfo();
        setIsDeletingEpisode(false);
        return;
      }

      // The rebuild runs on a worker thread and the POST returns immediately, so
      // progress arrives over the same SSE stream the repair flow uses.
      setRepairProgress({
        active: true,
        phase: "开始…",
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
            toast({
              title: "删除失败",
              description: progress.error,
              variant: "destructive",
            });
          } else {
            toast({ title: "已删除分集", description: progress.phase });
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
          title: "进度中断",
          description: "重建可能仍在进行，刷新数据集查看结果。",
          variant: "destructive",
        });
        loadDatasetInfo();
      };
    } catch (error) {
      toast({
        title: "连接错误",
        description: "无法连接后端服务。",
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
      : Array.from({ length: episodeCount }, (_, index) => ({
          index,
          length: 0,
        }));
  const sortedLengths = episodeRows
    .map((e) => e.length)
    .filter((n) => n > 0)
    .sort((a, b) => a - b);
  const medianLength = sortedLengths.length
    ? sortedLengths[Math.floor(sortedLengths.length / 2)]
    : 0;
  const meanLength = sortedLengths.length
    ? Math.round(sortedLengths.reduce((a, b) => a + b, 0) / sortedLengths.length)
    : 0;
  const lengthStd =
    sortedLengths.length && meanLength
      ? Math.round(
          Math.sqrt(
            sortedLengths.reduce((s, v) => s + (v - meanLength) ** 2, 0) /
              sortedLengths.length
          )
        )
      : 0;

  const allEpisodesSelected =
    episodeCount > 0 && selectedEpisodes.size === episodeCount;

  const toggleAllEpisodes = () => {
    setSelectedEpisodes(
      allEpisodesSelected
        ? new Set()
        : new Set(Array.from({ length: episodeCount }, (_, i) => i)),
    );
  };

  const handleRepairDataset = async () => {
    if (!datasetInfo) return;
    setIsRepairing(true);
    setRepairProgress({
      active: true,
      phase: "开始…",
      percent: 0,
      current: 0,
      total: 0,
      dataset_repo_id: datasetInfo.dataset_repo_id,
      error: null,
      completed: false,
    });

    try {
      const response = await fetchWithHeaders(`${baseUrl}/repair-dataset`, {
        method: "POST",
        body: JSON.stringify({ dataset_repo_id: datasetInfo.dataset_repo_id }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        toast({
          title: "修复失败",
          description: data.message || "无法启动修复",
          variant: "destructive",
        });
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
            toast({
              title: "修复失败",
              description: progress.error,
              variant: "destructive",
            });
          } else {
            toast({ title: "修复完成", description: "数据集已修复。" });
            // Re-check health and reload info
            loadDatasetInfo();
            checkDatasetHealth(
              baseUrl,
              fetchWithHeaders,
              datasetInfo.dataset_repo_id,
            ).then(setHealth);
          }
        }
      };
      eventSource.onerror = () => {
        eventSource.close();
        setIsRepairing(false);
      };
    } catch (error) {
      toast({
        title: "连接错误",
        description: "无法连接后端服务。",
        variant: "destructive",
      });
      setIsRepairing(false);
    }
  };

  // Show loading state while fetching dataset info
  if (isLoadingDatasetInfo) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-lg">正在加载数据集信息…</p>
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
              返回首页
            </Button>
            <h1 className="text-3xl font-bold">数据集</h1>
          </div>
          <MergeSection />
          <div id="dataset-scoring" className="mt-8">
            <RolloutScoreCard initialRepoId={scoreRepoId} />
          </div>
          <div className="mt-8">
            <SarmTrainCard initialRepoId={scoreRepoId} />
          </div>
        </div>
      </div>
    );
  }

  const isAlreadyOnHub = datasetInfo.source === "both";
  const onHub = uploadSuccess || isAlreadyOnHub;

  // One status, derived from the server's upload progress rather than local
  // guesses, so a refresh or a second browser tab shows the truth.
  const uploadStatus = isUploading
    ? {
        text: `上传中 ${uploadProgress.percent}%`,
        cls: "border-blue-600/50 bg-blue-900/20 text-blue-400",
      }
    : uploadError
      ? {
          text: "上传失败",
          cls: "border-red-600/50 bg-red-900/20 text-red-400",
        }
      : onHub
        ? {
            text: "已上传到 Hub",
            cls: "border-green-600/50 bg-green-900/20 text-green-400",
          }
        : {
            text: "仅保存在本地",
            cls: "border-gray-600 bg-gray-800 text-gray-400",
          };

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
              返回首页
            </Button>
            <Button
              onClick={() => setShowDeleteConfirm(true)}
              variant="outline"
              size="icon"
              disabled={isDeleting}
              aria-label="从磁盘删除数据集"
              className="border-red-500/40 text-red-400 hover:border-red-400 hover:text-red-300 hover:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>

          <div className="flex items-center gap-3">
            {onHub ? (
              <CheckCircle className="w-8 h-8 text-green-500" />
            ) : (
              <Database className="w-8 h-8 text-blue-500" />
            )}
            <h1 className="text-3xl font-bold">数据集</h1>
          </div>
        </div>

        {/* 数据集概览：身份、内容统计与质量分析合并为一张卡 */}
        <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-8">
          <div className="flex items-center justify-between gap-4 mb-5">
            <h2 className="text-xl font-semibold text-white">数据集概览 Overview</h2>
            <div className="flex items-center gap-2">
              <span
                className={`rounded-full border px-3 py-1 text-xs ${uploadStatus.cls}`}
              >
                {uploadStatus.text}
              </span>
              {scoredBackends.includes("robometer") ? (
                <span className="rounded-full border border-purple-600/50 bg-purple-900/20 px-3 py-1 text-xs text-purple-400">
                  Robometer 已评分
                </span>
              ) : null}
              {scoredBackends.includes("sarm") ? (
                <span className="rounded-full border border-cyan-600/50 bg-cyan-900/20 px-3 py-1 text-xs text-cyan-400">
                  SARM 已评分
                </span>
              ) : null}
            </div>
          </div>

          {isUploading && (
            <div className="mb-5">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300">
                  {uploadProgress.phase || "准备中…"}
                </span>
                <span className="text-sm text-blue-400 font-mono">
                  {uploadProgress.percent}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-blue-500 h-2.5 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${uploadProgress.percent}%` }}
                />
              </div>
              {uploadProgress.total > 0 && (
                <p className="text-xs text-gray-500 mt-1">
                  {uploadProgress.current} / {uploadProgress.total} 个文件
                </p>
              )}
            </div>
          )}

          {uploadError && !isUploading && (
            <div className="mb-5 rounded-lg border border-red-600/50 bg-red-900/20 p-3 text-sm text-red-300">
              上传失败：{uploadError}
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Tile label="仓库 repo_id" className="col-span-2">
              <span className="font-mono text-sm font-semibold">
                {datasetInfo.dataset_repo_id}
              </span>
            </Tile>
            <Tile label="集数 episodes" value={String(episodeCount)} />
            <Tile label="帧率 fps" value={`${fps} FPS`} />
            <Tile
              label="机器人 robot_type"
              value={datasetInfo.robot_type || "-"}
            />
            <Tile
              label="平均长度 avg_length"
              value={meanLength ? `${meanLength} 帧` : "-"}
              sub={meanLength ? `±${lengthStd}` : undefined}
            />
            <Tile label="任务 tasks" className="col-span-2">
              {datasetInfo.tasks?.length ? (
                <ul className="space-y-0.5">
                  {datasetInfo.tasks.map((t) => (
                    <li key={t.task} className="flex items-center gap-2 font-semibold">
                      <span>{t.task}</span>
                      {t.episodes > 0 && (
                        <span className="text-xs font-normal text-gray-400">
                          {t.episodes} 集
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              ) : (
                "-"
              )}
            </Tile>
          </div>

          {datasetInfo.dataset_repo_id && (
            <div className="mt-6 border-t border-gray-700 pt-5">
              <DatasetSummaryCard datasetRepoId={datasetInfo.dataset_repo_id} />
            </div>
          )}

          {/* 操作 */}
          <div className="mt-6 border-t border-gray-700 pt-5 flex flex-wrap gap-3">
            <Button
              onClick={handleUploadToHub}
              disabled={isUploading}
              className="bg-blue-500 hover:bg-blue-600 text-white"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <UploadIcon className="w-4 h-4 mr-2" />
                  {isAlreadyOnHub ? "Re-sync to Hub" : "Upload to HuggingFace Hub"}
                </>
              )}
            </Button>

            {isAlreadyOnHub ? (
              <Button
                onClick={() => openInHubViewer(datasetInfo.dataset_repo_id)}
                variant="outline"
                className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View on Hub
              </Button>
            ) : (
              <Button
                onClick={handleSkipUpload}
                disabled={isUploading}
                variant="outline"
                className="border-gray-600 text-gray-300 hover:bg-gray-800 hover:text-white"
              >
                Not Now
              </Button>
            )}
          </div>
        </div>

        {/* Dataset Health / Corruption Warning */}
        {checkingHealth && (
          <div className="bg-gray-900 rounded-lg p-4 border border-gray-700 mb-8 flex items-center gap-3">
            <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
            <span className="text-gray-300">正在检查数据集完整性…</span>
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
              数据集完整，{health.total_episodes} 集全部文件可读
            </span>
          </div>
        )}

        {/* Episodes List */}
        {episodeCount > 0 && (
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-700 mb-8">
            <div className="flex items-center justify-between mb-4 gap-4">
              <div
                role="button"
                tabIndex={0}
                aria-expanded={episodesOpen}
                onClick={() => setEpisodesOpen((v) => !v)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    setEpisodesOpen((v) => !v);
                  }
                }}
                className="flex items-center gap-2 cursor-pointer select-none"
              >
                <ListVideo className="w-5 h-5 text-blue-400" />
                <h2 className="text-xl font-semibold text-white">分集列表 Episodes</h2>
                <span className="text-sm text-gray-400">
                  共 {episodeCount} 集
                  {selectedEpisodes.size > 0 && (
                    <span className="text-red-400">
                      {" "}
                      · 已选 {selectedEpisodes.size} 集
                    </span>
                  )}
                </span>
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                  <Checkbox
                    checked={allEpisodesSelected}
                    onCheckedChange={toggleAllEpisodes}
                    disabled={isDeletingEpisode}
                    aria-label="全选所有分集"
                  />
                  全选
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
                  Delete Selected
                </Button>
                <button
                  type="button"
                  aria-label={episodesOpen ? "收起分集列表" : "展开分集列表"}
                  onClick={() => setEpisodesOpen((v) => !v)}
                  className="text-gray-500 hover:text-white"
                >
                  {episodesOpen ? (
                    <ChevronDown className="w-4 h-4" />
                  ) : (
                    <ChevronRight className="w-4 h-4" />
                  )}
                </button>
              </div>
            </div>
            {isDeletingEpisode && repairProgress && (
              <div className="mb-4 p-3 bg-gray-900/60 rounded-lg border border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                    <span className="text-sm text-gray-300">
                      {repairProgress.phase}
                    </span>
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
            {episodesOpen && (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {episodeRows.map(({ index, length }) => {
                const isShort =
                  length > 0 && medianLength > 0 && length < medianLength * 0.5;
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
                      aria-label={`选择第 ${index} 集`}
                    />
                    <span className="text-green-400 font-mono text-sm">
                      #{index.toString().padStart(3, "0")}
                    </span>
                    <span className="text-gray-300 text-sm">第 {index} 集</span>
                    {length > 0 && (
                      <span
                        className={`text-xs ${isShort ? "text-amber-400" : "text-gray-500"}`}
                      >
                        {length} 帧
                        {fps ? ` · ${(length / fps).toFixed(1)} 秒` : ""}
                        {isShort ? " · 不足中位数一半" : ""}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            )}
          </div>
        )}

        <MergeSection />

        <div id="dataset-scoring" className="mt-8">
          <RolloutScoreCard initialRepoId={scoreRepoId} />
        </div>

        <div className="mt-8">
          <SarmTrainCard initialRepoId={scoreRepoId} />
        </div>
      </div>

      <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>从磁盘删除数据集？</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              将从本地缓存中永久删除{" "}
              <span className="font-mono text-white">
                {datasetInfo.dataset_repo_id}
              </span>
              ，此操作无法撤销。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              保留数据集
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteDataset}
              disabled={isDeleting}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              {isDeleting ? "删除中…" : "删除"}
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
              删除选中的 {selectedEpisodes.size} 集？
            </AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              将删除第{" "}
              {[...selectedEpisodes]
                .sort((a, b) => a - b)
                .slice(0, 12)
                .join(", ")}
              {selectedEpisodes.size > 12
                ? ` 等 ${selectedEpisodes.size} 集`
                : ""}
              。原数据集备份在同级目录{" "}
              <code className="text-gray-300">
                {datasetInfo?.dataset_repo_id.split("/").pop() || "dataset"}
                __bak_del*
              </code>
              ，删掉备份前都可恢复。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              取消
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteSelectedEpisodes}
              disabled={isDeletingEpisode}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              {isDeletingEpisode
                ? "重建中…"
                : `删除 ${selectedEpisodes.size} 集`}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

const Tile: React.FC<{
  label: string;
  value?: string;
  sub?: string;
  className?: string;
  children?: React.ReactNode;
}> = ({ label, value, sub, className = "", children }) => (
  <div className={`rounded-lg bg-gray-800/60 px-4 py-3 ${className}`}>
    <p className="mb-1 text-xs text-gray-400">{label}</p>
    {children ?? (
      <p className="font-semibold text-white">
        {value}
        {sub && (
          <span className="ml-1 text-xs font-normal text-gray-500">{sub}</span>
        )}
      </p>
    )}
  </div>
);

export default Upload;
