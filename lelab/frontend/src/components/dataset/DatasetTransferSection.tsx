import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowLeftRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  FolderSearch,
  Loader2,
  Play,
  Upload,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasets } from "@/hooks/useDatasets";

interface FormatInfo {
  key: string;
  label: string;
  kind: string;
  directions: string[];
  description: string;
  source_hint: string;
  output_hint: string;
  defaults: { fps?: number; task?: string };
}

interface SourceSummary {
  episode_count?: number;
  episode_lengths?: number[];
  total_frames?: number | null;
  fps?: number | null;
  task?: string | null;
  cameras?: { source: string; name: string; shape: number[] | null }[];
  state_dim?: number | null;
  action_dim?: number | null;
  warnings?: string[];
}

interface InspectResult {
  path: string;
  preset: string;
  label: string;
  kind: string;
  detected: boolean;
  summary: SourceSummary;
}

interface QcCheck {
  name: string;
  ok: boolean;
  detail: string;
}

interface QcReport {
  ok: boolean;
  checks: QcCheck[];
  stats?: Record<string, unknown>;
  reminders?: string[];
}

interface IoResult {
  repo_id?: string;
  path?: string;
  root?: string;
  episodes?: number;
  frames?: number;
  fps?: number;
  cameras?: string[];
  qc?: QcReport;
}

interface IoProgress {
  active: boolean;
  kind: "import" | "export" | null;
  phase: string;
  percent: number;
  current: number;
  total: number;
  error: string | null;
  completed: boolean;
  result: IoResult | null;
}

const emptyProgress: IoProgress = {
  active: false,
  kind: null,
  phase: "",
  percent: 0,
  current: 0,
  total: 0,
  error: null,
  completed: false,
  result: null,
};

const formatDefaultFps = (format: FormatInfo | undefined) =>
  format?.defaults?.fps ? String(format.defaults.fps) : "";

/**
 * Import an external HDF5/Zarr dataset into LeRobotDataset v3, or export a
 * local LeRobot dataset back out. Conversion runs server-side as a background
 * job; the QC report is the part that matters after it finishes.
 */
const DatasetTransferSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { datasets, refresh } = useDatasets();
  const [open, setOpen] = useState(false);
  const [tab, setTab] = useState<"import" | "export">("import");
  const [formats, setFormats] = useState<FormatInfo[]>([]);
  const [progress, setProgress] = useState<IoProgress>(emptyProgress);
  const streamRef = useRef<EventSource | null>(null);

  const localDatasets = useMemo(
    () =>
      datasets.filter(
        (d) =>
          (d.source === "local" || d.source === "both") &&
          !d.repo_id.includes("__bak_"),
      ),
    [datasets],
  );

  const stopFollowing = useCallback(() => {
    streamRef.current?.close();
    streamRef.current = null;
  }, []);

  const followProgress = useCallback(() => {
    if (streamRef.current) return;
    const stream = new EventSource(`${baseUrl}/dataset-io/progress`);
    stream.onmessage = (event) => {
      const data = JSON.parse(event.data) as IoProgress;
      setProgress(data);
      if (data.completed || data.error) {
        stream.close();
        streamRef.current = null;
        if (data.completed) {
          refresh();
          toast({
            title: data.kind === "import" ? "导入完成" : "导出完成",
            description: data.result?.qc?.ok
              ? "自动质检通过。"
              : "转换已完成，但质检报告里有需要关注的问题。",
          });
        } else {
          toast({
            title: data.kind === "import" ? "导入失败" : "导出失败",
            description: data.error || "未知错误",
            variant: "destructive",
          });
        }
      }
    };
    stream.onerror = () => {
      stream.close();
      streamRef.current = null;
    };
    streamRef.current = stream;
  }, [baseUrl, refresh, toast]);

  useEffect(() => () => stopFollowing(), [stopFollowing]);

  // A refresh or a tab switch must not hide a conversion that is still running.
  useEffect(() => {
    fetchWithHeaders(`${baseUrl}/dataset-io/progress/current`)
      .then((r) => r.json())
      .then((data: IoProgress) => {
        if (!data?.active) return;
        setProgress(data);
        followProgress();
      })
      .catch(() => undefined);
  }, [baseUrl, fetchWithHeaders, followProgress]);

  useEffect(() => {
    if (!open || formats.length) return;
    fetchWithHeaders(`${baseUrl}/dataset-io/formats`)
      .then((r) => r.json())
      .then((data) => setFormats(data.formats || []))
      .catch(() => undefined);
  }, [open, formats.length, baseUrl, fetchWithHeaders]);

  const importFormats = formats.filter((f) => f.directions.includes("import"));
  const exportFormats = formats.filter((f) => f.directions.includes("export"));

  // ── import form ───────────────────────────────────────────────────
  const [sourcePath, setSourcePath] = useState("");
  const [importPreset, setImportPreset] = useState("");
  const [inspectResult, setInspectResult] = useState<InspectResult | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [targetRepo, setTargetRepo] = useState("");
  const [fps, setFps] = useState("");
  const [task, setTask] = useState("");
  const [stateKey, setStateKey] = useState("");
  const [actionKey, setActionKey] = useState("");
  const [jointNames, setJointNames] = useState("");
  const [imageKeys, setImageKeys] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const selectedImportFormat = importFormats.find((f) => f.key === importPreset);

  const inspectSource = async () => {
    if (!sourcePath.trim()) return;
    setInspecting(true);
    try {
      const response = await fetchWithHeaders(`${baseUrl}/dataset-io/inspect`, {
        method: "POST",
        body: JSON.stringify({
          path: sourcePath.trim(),
          preset: importPreset || null,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "检测失败");
      const result = data as InspectResult;
      setInspectResult(result);
      if (!importPreset && result.preset) setImportPreset(result.preset);
      const format = formats.find((f) => f.key === result.preset);
      if (!fps) setFps(formatDefaultFps(format) || String(result.summary.fps || ""));
      if (!task && result.summary.task) setTask(result.summary.task);
      if (!targetRepo && sourcePath.trim()) {
        const author =
          localDatasets[0]?.repo_id.split("/")[0] ||
          "你的用户名";
        const leaf = sourcePath
          .trim()
          .replace(/\/+$/, "")
          .split("/")
          .pop()
          ?.replace(/\.(hdf5|h5)$/i, "")
          ?.replace(/\.zarr$/i, "");
        if (leaf) setTargetRepo(`${author}/${leaf}_converted`);
      }
    } catch (error) {
      toast({
        title: "检测失败",
        description: error instanceof Error ? error.message : String(error),
        variant: "destructive",
      });
    } finally {
      setInspecting(false);
    }
  };

  const startImport = async () => {
    if (!importPreset || !sourcePath.trim() || !targetRepo.trim()) {
      toast({
        title: "缺少必填项",
        description: "请填写源路径、目标数据集标识，并选择源格式。",
        variant: "destructive",
      });
      return;
    }
    let parsedImageKeys: Record<string, string> | null = null;
    if (imageKeys.trim()) {
      try {
        parsedImageKeys = JSON.parse(imageKeys);
      } catch {
        toast({
          title: "相机字段不是合法 JSON",
          description: '示例：{"cam_high": "up", "cam_left": "wrist"}',
          variant: "destructive",
        });
        return;
      }
    }
    setProgress({ ...emptyProgress, active: true, kind: "import", phase: "启动…" });
    followProgress();
    try {
      const response = await fetchWithHeaders(`${baseUrl}/dataset-io/import`, {
        method: "POST",
        body: JSON.stringify({
          preset: importPreset,
          source_path: sourcePath.trim(),
          target_repo_id: targetRepo.trim(),
          fps: fps ? Number(fps) : null,
          task: task.trim() || null,
          state_key: stateKey.trim() || null,
          action_key: actionKey.trim() || null,
          joint_names: jointNames.trim()
            ? jointNames.split(",").map((s) => s.trim()).filter(Boolean)
            : null,
          image_keys: parsedImageKeys,
          video: true,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "无法启动导入");
    } catch (error) {
      stopFollowing();
      setProgress({ ...emptyProgress, error: String(error) });
      toast({
        title: "导入失败",
        description: error instanceof Error ? error.message : String(error),
        variant: "destructive",
      });
    }
  };

  // ── export form ───────────────────────────────────────────────────
  const [exportRepo, setExportRepo] = useState("");
  const [exportPreset, setExportPreset] = useState("");
  const [outputPath, setOutputPath] = useState("");
  const [quality, setQuality] = useState("95");

  const selectedExportFormat = exportFormats.find((f) => f.key === exportPreset);

  useEffect(() => {
    if (!exportRepo && localDatasets.length) setExportRepo(localDatasets[0].repo_id);
  }, [localDatasets, exportRepo]);

  const startExport = async () => {
    if (!exportPreset || !exportRepo || !outputPath.trim()) {
      toast({
        title: "缺少必填项",
        description: "请选择本地数据集、导出格式并填写输出路径。",
        variant: "destructive",
      });
      return;
    }
    setProgress({ ...emptyProgress, active: true, kind: "export", phase: "启动…" });
    followProgress();
    try {
      const response = await fetchWithHeaders(`${baseUrl}/dataset-io/export`, {
        method: "POST",
        body: JSON.stringify({
          preset: exportPreset,
          repo_id: exportRepo,
          output_path: outputPath.trim(),
          quality: Number(quality) || 95,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "无法启动导出");
    } catch (error) {
      stopFollowing();
      setProgress({ ...emptyProgress, error: String(error) });
      toast({
        title: "导出失败",
        description: error instanceof Error ? error.message : String(error),
        variant: "destructive",
      });
    }
  };

  const running = progress.active;
  const result = progress.result;

  return (
    <div className="mt-8">
      <div className="bg-gray-900 rounded-lg border border-gray-700">
        <button
          type="button"
          className="w-full flex items-center justify-between px-6 py-4 text-left"
          onClick={() => setOpen((v) => !v)}
        >
          <div className="flex items-center gap-3">
            <ArrowLeftRight className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-xl font-semibold text-white">
                格式转换与导出
              </h2>
              <p className="text-sm text-gray-400">
                在 LeRobotDataset 与 HDF5 / Zarr 之间转换，转换后自动质检
              </p>
            </div>
          </div>
          {open ? (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          )}
        </button>

        {open && (
          <div className="px-6 pb-6 space-y-5">
            <div className="flex gap-2">
              <Button
                variant={tab === "import" ? "default" : "outline"}
                className={
                  tab === "import"
                    ? "bg-cyan-600 hover:bg-cyan-500 text-white"
                    : "border-gray-600 text-gray-300"
                }
                onClick={() => setTab("import")}
                disabled={running}
              >
                <Upload className="w-4 h-4 mr-2" />
                导入转换
              </Button>
              <Button
                variant={tab === "export" ? "default" : "outline"}
                className={
                  tab === "export"
                    ? "bg-cyan-600 hover:bg-cyan-500 text-white"
                    : "border-gray-600 text-gray-300"
                }
                onClick={() => setTab("export")}
                disabled={running}
              >
                <Download className="w-4 h-4 mr-2" />
                导出为外部格式
              </Button>
            </div>

            {running && (
              <div className="rounded-md border border-cyan-700/50 bg-cyan-900/20 p-4">
                <div className="flex items-center justify-between text-sm text-cyan-200">
                  <span className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    {progress.phase || "处理中…"}
                  </span>
                  <span>{progress.percent}%</span>
                </div>
                <div className="mt-2 h-2 bg-gray-800 rounded">
                  <div
                    className="h-2 rounded bg-cyan-500 transition-all"
                    style={{ width: `${Math.max(2, progress.percent)}%` }}
                  />
                </div>
                {progress.total > 0 && (
                  <p className="mt-1 text-xs text-cyan-300/70">
                    {progress.current} / {progress.total}
                  </p>
                )}
              </div>
            )}

            {!running && progress.error && (
              <div className="rounded-md border border-red-700/50 bg-red-900/20 p-4 text-sm text-red-300">
                {progress.error}
              </div>
            )}

            {!running && result && <ResultCard result={result} />}

            {tab === "import" ? (
              <div className="space-y-4">
                <div className="grid gap-3 md:grid-cols-[1fr_auto]">
                  <div>
                    <Label htmlFor="io-source">源路径</Label>
                    <Input
                      id="io-source"
                      value={sourcePath}
                      onChange={(e) => {
                        setSourcePath(e.target.value);
                        setInspectResult(null);
                      }}
                      placeholder="/home/cz/data/aloha_dataset 或 /data/pusht.zarr"
                      className="bg-gray-800 border-gray-700 text-white font-mono"
                      disabled={running}
                    />
                  </div>
                  <div className="flex items-end gap-2">
                    <Button
                      variant="outline"
                      className="border-cyan-700/60 text-cyan-300 hover:text-cyan-200"
                      onClick={inspectSource}
                      disabled={running || inspecting || !sourcePath.trim()}
                    >
                      {inspecting ? (
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <FolderSearch className="w-4 h-4 mr-2" />
                      )}
                      检测
                    </Button>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div>
                    <Label htmlFor="io-preset">源格式</Label>
                    <select
                      id="io-preset"
                      className="mt-1 w-full h-10 rounded-md bg-gray-800 border border-gray-700 px-3 text-sm text-white"
                      value={importPreset}
                      onChange={(e) => {
                        setImportPreset(e.target.value);
                        setInspectResult(null);
                        const fmt = importFormats.find((f) => f.key === e.target.value);
                        if (!fps) setFps(formatDefaultFps(fmt));
                      }}
                      disabled={running}
                    >
                      <option value="">自动检测 / 选择格式</option>
                      {importFormats.map((fmt) => (
                        <option key={fmt.key} value={fmt.key}>
                          {fmt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <Label htmlFor="io-target">目标数据集标识</Label>
                    <Input
                      id="io-target"
                      value={targetRepo}
                      onChange={(e) => setTargetRepo(e.target.value)}
                      placeholder="作者名/数据集名"
                      className="mt-1 bg-gray-800 border-gray-700 text-white font-mono"
                      disabled={running}
                    />
                  </div>
                </div>

                {selectedImportFormat && (
                  <p className="text-xs text-gray-400">
                    {selectedImportFormat.description}
                    <br />
                    源：{selectedImportFormat.source_hint}
                  </p>
                )}

                {inspectResult && (
                  <InspectSummary result={inspectResult} />
                )}

                <div className="grid gap-3 md:grid-cols-3">
                  <div>
                    <Label htmlFor="io-fps">FPS</Label>
                    <Input
                      id="io-fps"
                      value={fps}
                      onChange={(e) => setFps(e.target.value)}
                      inputMode="numeric"
                      className="mt-1 bg-gray-800 border-gray-700 text-white"
                      disabled={running}
                    />
                  </div>
                  <div className="md:col-span-2">
                    <Label htmlFor="io-task">任务描述（task）</Label>
                    <Input
                      id="io-task"
                      value={task}
                      onChange={(e) => setTask(e.target.value)}
                      placeholder="例如 grasp the red ball"
                      className="mt-1 bg-gray-800 border-gray-700 text-white"
                      disabled={running}
                    />
                  </div>
                </div>

                <button
                  type="button"
                  className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-200"
                  onClick={() => setAdvancedOpen((v) => !v)}
                >
                  {advancedOpen ? (
                    <ChevronDown className="w-3 h-3" />
                  ) : (
                    <ChevronRight className="w-3 h-3" />
                  )}
                  字段映射（高级）
                </button>
                {advancedOpen && (
                  <div className="grid gap-3 md:grid-cols-2">
                    <div>
                      <Label htmlFor="io-state-key">state 字段</Label>
                      <Input
                        id="io-state-key"
                        value={stateKey}
                        onChange={(e) => setStateKey(e.target.value)}
                        placeholder="自动（ALOHA: observations/qpos；robomimic: 低维 obs 拼接）"
                        className="mt-1 bg-gray-800 border-gray-700 text-white font-mono text-xs"
                        disabled={running}
                      />
                    </div>
                    <div>
                      <Label htmlFor="io-action-key">action 字段</Label>
                      <Input
                        id="io-action-key"
                        value={actionKey}
                        onChange={(e) => setActionKey(e.target.value)}
                        placeholder="自动（action / actions）"
                        className="mt-1 bg-gray-800 border-gray-700 text-white font-mono text-xs"
                        disabled={running}
                      />
                    </div>
                    <div>
                      <Label htmlFor="io-joint-names">关节名（逗号分隔）</Label>
                      <Input
                        id="io-joint-names"
                        value={jointNames}
                        onChange={(e) => setJointNames(e.target.value)}
                        placeholder="留空则用 joint_0, joint_1 …"
                        className="mt-1 bg-gray-800 border-gray-700 text-white font-mono text-xs"
                        disabled={running}
                      />
                    </div>
                    <div>
                      <Label htmlFor="io-image-keys">相机字段映射（JSON）</Label>
                      <Input
                        id="io-image-keys"
                        value={imageKeys}
                        onChange={(e) => setImageKeys(e.target.value)}
                        placeholder={'{"cam_high": "up", "cam_left": "wrist"}'}
                        className="mt-1 bg-gray-800 border-gray-700 text-white font-mono text-xs"
                        disabled={running}
                      />
                    </div>
                  </div>
                )}

                <Button
                  className="bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={startImport}
                  disabled={running}
                >
                  <Play className="w-4 h-4 mr-2" />
                  开始导入
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-3 md:grid-cols-2">
                  <div>
                    <Label htmlFor="io-export-repo">本地数据集</Label>
                    <select
                      id="io-export-repo"
                      className="mt-1 w-full h-10 rounded-md bg-gray-800 border border-gray-700 px-3 text-sm text-white"
                      value={exportRepo}
                      onChange={(e) => setExportRepo(e.target.value)}
                      disabled={running}
                    >
                      <option value="">选择本地数据集</option>
                      {localDatasets.map((d) => (
                        <option key={d.repo_id} value={d.repo_id}>
                          {d.repo_id}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <Label htmlFor="io-export-preset">导出格式</Label>
                    <select
                      id="io-export-preset"
                      className="mt-1 w-full h-10 rounded-md bg-gray-800 border border-gray-700 px-3 text-sm text-white"
                      value={exportPreset}
                      onChange={(e) => setExportPreset(e.target.value)}
                      disabled={running}
                    >
                      <option value="">选择导出格式</option>
                      {exportFormats.map((fmt) => (
                        <option key={fmt.key} value={fmt.key}>
                          {fmt.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {selectedExportFormat && (
                  <p className="text-xs text-gray-400">
                    {selectedExportFormat.description}
                    <br />
                    输出：{selectedExportFormat.output_hint}
                  </p>
                )}

                <div className="grid gap-3 md:grid-cols-[1fr_140px]">
                  <div>
                    <Label htmlFor="io-output">输出路径</Label>
                    <Input
                      id="io-output"
                      value={outputPath}
                      onChange={(e) => setOutputPath(e.target.value)}
                      placeholder={
                        exportPreset === "zarr_diffusion_policy"
                          ? "/home/cz/export/dataset.zarr"
                          : exportPreset === "robomimic_hdf5"
                            ? "/home/cz/export/dataset.hdf5"
                            : "/home/cz/export/aloha_dataset"
                      }
                      className="mt-1 bg-gray-800 border-gray-700 text-white font-mono"
                      disabled={running}
                    />
                  </div>
                  <div>
                    <Label htmlFor="io-quality">JPEG 质量</Label>
                    <Input
                      id="io-quality"
                      value={quality}
                      onChange={(e) => setQuality(e.target.value)}
                      inputMode="numeric"
                      className="mt-1 bg-gray-800 border-gray-700 text-white"
                      disabled={running || exportPreset !== "aloha_hdf5"}
                    />
                  </div>
                </div>

                <Button
                  className="bg-cyan-600 hover:bg-cyan-500 text-white"
                  onClick={startExport}
                  disabled={running}
                >
                  <Play className="w-4 h-4 mr-2" />
                  开始导出
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const InspectSummary: React.FC<{ result: InspectResult }> = ({ result }) => {
  const s = result.summary;
  return (
    <div className="rounded-md border border-gray-700 bg-gray-800/60 p-4 text-sm">
      <div className="flex items-center gap-2 text-gray-200">
        <CheckCircle2 className="w-4 h-4 text-green-400" />
        {result.detected ? "自动识别为" : "按选择格式读取"}：{result.label}
      </div>
      <div className="mt-2 grid gap-x-6 gap-y-1 text-gray-400 sm:grid-cols-2">
        <span>episode：{s.episode_count ?? "-"}</span>
        <span>帧数：{s.total_frames ?? "-"}</span>
        <span>FPS：{s.fps ?? "未标注"}</span>
        <span>
          state / action 维度：{s.state_dim ?? "-"} / {s.action_dim ?? "-"}
        </span>
      </div>
      {s.cameras?.length ? (
        <p className="mt-2 text-gray-400">
          相机：{s.cameras.map((c) => `${c.name}${c.shape ? ` ${c.shape.join("×")}` : ""}`).join("，")}
        </p>
      ) : null}
      {s.warnings?.length ? (
        <p className="mt-2 text-amber-400">{s.warnings.join("；")}</p>
      ) : null}
    </div>
  );
};

const ResultCard: React.FC<{ result: IoResult }> = ({ result }) => {
  const qc = result.qc;
  return (
    <div className="rounded-md border border-gray-700 bg-gray-800/60 p-4 space-y-2">
      <div className="flex items-center gap-2">
        {qc?.ok ? (
          <CheckCircle2 className="w-4 h-4 text-green-400" />
        ) : (
          <AlertTriangle className="w-4 h-4 text-amber-400" />
        )}
        <span className="text-sm font-medium text-gray-100">
          {qc?.ok ? "转换完成，质检通过" : "转换完成，质检有警告"}
        </span>
      </div>
      <p className="text-xs text-gray-400 font-mono break-all">
        {result.path || result.root}
      </p>
      <p className="text-xs text-gray-400">
        {result.repo_id ? `数据集：${result.repo_id} · ` : ""}
        {result.episodes ?? "-"} 集 · {result.frames ?? "-"} 帧
        {result.fps ? ` · ${result.fps} fps` : ""}
        {result.cameras?.length ? ` · 相机：${result.cameras.join(", ")}` : ""}
      </p>
      {qc?.checks?.length ? (
        <ul className="text-xs space-y-1">
          {qc.checks.slice(0, 12).map((check, index) => (
            <li key={`${check.name}-${index}`} className="flex gap-2">
              <span className={check.ok ? "text-green-400" : "text-red-400"}>
                {check.ok ? "✓" : "✗"}
              </span>
              <span className="text-gray-300">{check.name}</span>
              <span className="text-gray-500">{check.detail}</span>
            </li>
          ))}
        </ul>
      ) : null}
      {qc?.reminders?.length ? (
        <p className="text-xs text-amber-400/90">
          语义提示：{qc.reminders.join("；")}
        </p>
      ) : null}
    </div>
  );
};

export default DatasetTransferSection;
