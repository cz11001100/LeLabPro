import React, { useCallback, useEffect, useRef, useState } from "react";
import { Activity, BarChart3, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { listSSHConfigs, SSHConfig } from "@/lib/jobsApi";
import {
  RolloutComparison,
  RolloutDatasetInfo,
  RolloutScoreState,
  RolloutSummary,
  compareRolloutProgress,
  exemptInterventions,
  getRolloutScoreStatus,
  getRolloutSummary,
  getSarmTrainingStatus,
  listRolloutDatasets,
  promoteRolloutProgress,
  startRolloutScore,
  startSarmTraining,
} from "@/lib/rolloutEvalApi";

const POLL_MS = 2000;

const BACKENDS = [
  { value: "robometer", label: "Robometer（零样本通用）" },
  { value: "sarm", label: "SARM（本任务自训）" },
];

const CAMERA_KEYS = [
  { value: "observation.images.up", label: "up（俯视）" },
  { value: "observation.images.side", label: "side（侧视）" },
  { value: "observation.images.wrist", label: "wrist（腕部）" },
];

const pctText = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

const rateTone = (rate: number | null | undefined) => {
  if (rate == null) return "text-gray-400";
  if (rate >= 0.8) return "text-green-400";
  if (rate >= 0.5) return "text-lime-400";
  if (rate >= 0.25) return "text-amber-400";
  return "text-red-400";
};

/**
 * Closed-loop readout for the policy: score the episodes it actually drove
 * (recorded by the capture modes on the inference page) with a progress model,
 * and report the success rate the open-loop metrics cannot show.
 */
const RolloutScoreCard: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<RolloutDatasetInfo[]>([]);
  const [repoId, setRepoId] = useState("");
  const [summary, setSummary] = useState<RolloutSummary | null>(null);
  const [scoreState, setScoreState] = useState<RolloutScoreState | null>(null);
  const [imageKey, setImageKey] = useState(CAMERA_KEYS[0].value);
  const [stride, setStride] = useState(5);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [exemptMsg, setExemptMsg] = useState("");
  const [backend, setBackend] = useState<"robometer" | "sarm">("robometer");
  const [sarmPath, setSarmPath] = useState("");
  const [comparison, setComparison] = useState<RolloutComparison | null>(null);
  const [sshConfigs, setSshConfigs] = useState<SSHConfig[]>([]);
  const [sshConfigId, setSshConfigId] = useState("");
  const [sarmBusy, setSarmBusy] = useState(false);
  const [sarmStatus, setSarmStatus] = useState<{
    training: boolean;
    finished_ok: boolean;
    log_tail: string;
  } | null>(null);
  const [promoting, setPromoting] = useState("");
  const wasRunningRef = useRef(false);

  const refreshDatasets = useCallback(async () => {
    try {
      const list = await listRolloutDatasets(baseUrl, fetchWithHeaders);
      setDatasets(list);
      setRepoId((prev) => (prev && list.some((d) => d.repo_id === prev) ? prev : list[0]?.repo_id ?? ""));
    } catch (e) {
      toast({
        title: "读不到 rollout 数据集",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  const loadSummary = useCallback(
    async (id: string) => {
      if (!id) return;
      setLoading(true);
      try {
        setSummary(await getRolloutSummary(baseUrl, fetchWithHeaders, id));
      } catch (e) {
        setSummary(null);
        toast({
          title: "汇总失败",
          description: e instanceof Error ? e.message : String(e),
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    },
    [baseUrl, fetchWithHeaders, toast],
  );

  // Follow the pod-side SARM run once it has been started from here.
  useEffect(() => {
    if (!sarmStatus?.training || !sshConfigId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const st = await getSarmTrainingStatus(baseUrl, fetchWithHeaders, sshConfigId);
        if (cancelled) return;
        setSarmStatus({
          training: st.training,
          finished_ok: st.finished_ok,
          log_tail: st.log_tail,
        });
        if (!st.training) refreshDatasets();
      } catch {
        // Pod unreachable mid-run; keep polling.
      }
    };
    const id = setInterval(tick, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sarmStatus?.training, sshConfigId, baseUrl, fetchWithHeaders, refreshDatasets]);

  const loadComparison = useCallback(
    async (id: string) => {
      if (!id) {
        setComparison(null);
        return;
      }
      try {
        setComparison(await compareRolloutProgress(baseUrl, fetchWithHeaders, id));
      } catch {
        setComparison(null);
      }
    },
    [baseUrl, fetchWithHeaders],
  );

  useEffect(() => {
    loadSummary(repoId);
    loadComparison(repoId);
  }, [repoId, loadSummary, loadComparison]);

  // The pod-side SARM trainer needs a machine to run on.
  useEffect(() => {
    let cancelled = false;
    listSSHConfigs(baseUrl, fetchWithHeaders)
      .then((list) => {
        if (cancelled) return;
        setSshConfigs(list);
        setSshConfigId((prev) => prev || list[0]?.id || "");
      })
      .catch(() => setSshConfigs([]));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  // Poll the scoring job; refresh the readout the moment it lands.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const st = await getRolloutScoreStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setScoreState(st);
        if (st.running) {
          wasRunningRef.current = true;
        } else if (wasRunningRef.current) {
          wasRunningRef.current = false;
          refreshDatasets();
          loadSummary(repoId);
          loadComparison(repoId);
          toast({
            title: st.exit_code === 0 ? "打分完成" : "打分结束但可能没跑完",
            description:
              st.exit_code === 0
                ? "成功率与进度曲线已刷新"
                : `退出码 ${st.exit_code}，日志见 ${st.log_path}`,
            variant: st.exit_code === 0 ? "default" : "destructive",
          });
        }
      } catch {
        // Transient backend hiccups are fine; the next tick retries.
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, repoId, refreshDatasets, loadSummary, toast]);

  const handleScore = async () => {
    if (!repoId) return;
    setStarting(true);
    try {
      const res = await startRolloutScore(baseUrl, fetchWithHeaders, {
        dataset_repo_id: repoId,
        backend,
        reward_model_path: backend === "sarm" ? sarmPath : "",
        image_key: imageKey,
        stride,
      });
      setScoreState({ running: true, command: res.command, log_path: res.log_path });
    } catch (e) {
      // A 400 carries the command to run on a bigger GPU instead.
      const detail = e instanceof Error ? e.message : String(e);
      toast({ title: "打分没跑起来", description: detail, variant: "destructive" });
    } finally {
      setStarting(false);
    }
  };

  const handleExempt = async () => {
    if (!repoId) return;
    setExemptMsg("");
    try {
      const res = await exemptInterventions(baseUrl, fetchWithHeaders, repoId);
      setExemptMsg(res.message);
      loadSummary(repoId);
    } catch (e) {
      toast({
        title: "纠正段豁免失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handlePromote = async (which: "robometer" | "sarm") => {
    if (!repoId) return;
    setPromoting(which);
    try {
      await promoteRolloutProgress(baseUrl, fetchWithHeaders, repoId, which);
      await Promise.all([loadSummary(repoId), loadComparison(repoId)]);
    } catch (e) {
      toast({
        title: "设为准没成功",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setPromoting("");
    }
  };

  const handleTrainSarm = async () => {
    if (!repoId || !sshConfigId) return;
    setSarmBusy(true);
    try {
      const res = await startSarmTraining(baseUrl, fetchWithHeaders, {
        config_id: sshConfigId,
        dataset_repo_id: repoId,
        image_key: imageKey,
      });
      toast({ title: res.message, description: `日志 ${res.log_path}` });
      setSarmStatus({ training: true, finished_ok: false, log_tail: "" });
    } catch (e) {
      toast({
        title: "SARM 训练没起来",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setSarmBusy(false);
    }
  };

  const selected = datasets.find((d) => d.repo_id === repoId);
  const backendsPresent = selected?.backends ?? [];
  const running = !!scoreState?.running;
  const details = summary?.details ?? [];

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/40 p-4 space-y-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-slate-400" />
        <h3 className="text-sm font-semibold text-white">真机成功率（rollout 集）</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={refreshDatasets}
          aria-label="刷新数据集列表"
          className="ml-auto h-7 w-7 text-slate-400 hover:text-white"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>

      {datasets.length === 0 ? (
        <p className="text-xs text-slate-400">
          还没有 rollout 数据集。在部署里把采集模式设为纠正采集 / 抢救片段 / 全自动录制，跑完就会出现在这里。
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">数据集</Label>
              <Select value={repoId} onValueChange={setRepoId}>
                <SelectTrigger className="bg-slate-900 border-slate-700 text-white text-xs">
                  <SelectValue placeholder="选择数据集" />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
                  {datasets.map((d) => (
                    <SelectItem key={d.repo_id} value={d.repo_id}>
                      {d.repo_id} · {d.episodes} 集{d.scored ? " · 已打分" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">打分用相机</Label>
              <Select value={imageKey} onValueChange={setImageKey}>
                <SelectTrigger className="bg-slate-900 border-slate-700 text-white text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
                  {CAMERA_KEYS.map((c) => (
                    <SelectItem key={c.value} value={c.value}>
                      {c.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs text-slate-400">帧间隔（stride）</Label>
              <NumberInput
                min={1}
                value={stride}
                onChange={(v) => {
                  if (v !== undefined) setStride(v);
                }}
                className="bg-slate-900 border-slate-700 text-white text-xs h-9"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Select value={backend} onValueChange={(v) => setBackend(v as "robometer" | "sarm")}>
              <SelectTrigger className="w-52 bg-slate-900 border-slate-700 text-white text-xs h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
                {BACKENDS.map((b) => (
                  <SelectItem key={b.value} value={b.value}>
                    {b.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {backend === "sarm" && (
              <Input
                value={sarmPath}
                onChange={(e) => setSarmPath(e.target.value)}
                placeholder="SARM 模型 repo 或路径"
                className="w-56 bg-slate-900 border-slate-700 text-white text-xs h-9"
              />
            )}
            <Button
              onClick={handleScore}
              disabled={!repoId || running || starting || (backend === "sarm" && !sarmPath.trim())}
              className="bg-blue-600 hover:bg-blue-700 text-white text-sm"
            >
              {running || starting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Activity className="w-4 h-4 mr-2" />
              )}
              {running ? "打分中…" : "开始打分"}
            </Button>
            {selected?.scored && (
              <Button
                onClick={handleExempt}
                disabled={running}
                variant="outline"
                className="border-slate-600 text-slate-200 hover:bg-slate-700 text-sm"
              >
                纠正段豁免加权
              </Button>
            )}
            {selected && (
              <span className="text-xs text-slate-400">
                {selected.episodes} 集 · {selected.frames} 帧 · {selected.modified}
                {exemptMsg ? ` · ${exemptMsg}` : ""}
              </span>
            )}
          </div>

          {running && scoreState?.log_path && (
            <p className="text-xs text-slate-400 break-all">
              {Math.round(scoreState.elapsed_s ?? 0)}s · 日志 {scoreState.log_path}
            </p>
          )}

          {summary?.no_progress && (
            <p className="text-xs text-slate-400">{summary.hint}</p>
          )}

          {summary?.error && (
            <Alert className="bg-red-900/30 border-red-800 text-red-200">
              <AlertDescription className="text-xs">{summary.error}</AlertDescription>
            </Alert>
          )}

          {summary?.available && (
            <>
              <div className="flex items-end gap-4">
                <div>
                  <div className={`text-4xl font-mono font-bold ${rateTone(summary.success_rate)}`}>
                    {summary.success_rate == null
                      ? "—"
                      : `${Math.round(summary.success_rate * 100)}%`}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">
                    {summary.successes} / {summary.episodes} 集 · 进度阈值 {summary.threshold}
                  </div>
                </div>
                {loading && <Loader2 className="w-4 h-4 animate-spin text-slate-500" />}
              </div>

              <div className="max-h-64 overflow-y-auto rounded border border-slate-700">
                <table className="w-full text-xs">
                  <thead className="bg-slate-900/60 text-slate-400 sticky top-0">
                    <tr>
                      <th className="text-left px-2 py-1 font-medium">集</th>
                      <th className="text-right px-2 py-1 font-medium">帧</th>
                      <th className="text-right px-2 py-1 font-medium">起</th>
                      <th className="text-right px-2 py-1 font-medium">止</th>
                      <th className="text-right px-2 py-1 font-medium">峰</th>
                      <th className="text-right px-2 py-1 font-medium">卡滞帧</th>
                      <th className="text-right px-2 py-1 font-medium">结果</th>
                    </tr>
                  </thead>
                  <tbody>
                    {details.map((e) => (
                      <tr key={e.episode_index} className="border-t border-slate-700/60">
                        <td className="px-2 py-1 text-slate-200">#{e.episode_index}</td>
                        <td className="px-2 py-1 text-right text-slate-400">{e.frames ?? "—"}</td>
                        <td className="px-2 py-1 text-right">{e.progress_start.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.progress_end.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.progress_peak.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.stalled_at ?? "—"}</td>
                        <td className="px-2 py-1 text-right">
                          <span
                            className={`px-1.5 py-0.5 rounded font-semibold ${
                              e.success
                                ? "bg-green-500/15 text-green-300"
                                : "bg-red-500/15 text-red-300"
                            }`}
                          >
                            {e.success ? "成功" : "失败"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {/* Two scorers on the same episodes: Robometer never saw this rig,
              SARM was trained on it. Where they disagree is what to go watch. */}
          {comparison?.available ? (
            <div className="rounded border border-slate-700 bg-slate-900/40 p-3 space-y-2">
              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-300">
                <span className="font-semibold text-white">两后端对照</span>
                <span>{comparison.frames} 帧</span>
                <span>相关 r={comparison.pearson ?? "—"}</span>
                <span>平均差 {comparison.mean_abs_diff ?? "—"}</span>
                <span>成败一致 {pctText(comparison.agreement)}</span>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <span className="text-slate-300">
                  Robometer 成功率 {pctText(comparison.success_rate?.robometer)}
                </span>
                <span className="text-slate-300">
                  SARM 成功率 {pctText(comparison.success_rate?.sarm)}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handlePromote("robometer")}
                  disabled={!!promoting}
                  className="border-slate-600 text-slate-200 hover:bg-slate-700 text-xs"
                >
                  {promoting === "robometer" && (
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  )}
                  以 Robometer 为准
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handlePromote("sarm")}
                  disabled={!!promoting}
                  className="border-slate-600 text-slate-200 hover:bg-slate-700 text-xs"
                >
                  {promoting === "sarm" && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                  以 SARM 为准
                </Button>
              </div>
              {(comparison.disagreements?.length ?? 0) > 0 && (
                <div className="text-xs text-amber-300">
                  分歧集：
                  {comparison.disagreements!
                    .map(
                      (d) =>
                        `#${d.episode_index}（R ${d.robometer_end} / S ${d.sarm_end}）`,
                    )
                    .join("、")}
                </div>
              )}
            </div>
          ) : backendsPresent.length > 0 ? (
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
              <span>目前只有 {backendsPresent.join(" / ")} 打过这个集。</span>
              {sshConfigs.length > 0 && (
                <>
                  <Select value={sshConfigId} onValueChange={setSshConfigId}>
                    <SelectTrigger className="w-40 bg-slate-900 border-slate-700 text-white text-xs h-8">
                      <SelectValue placeholder="选 pod" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
                      {sshConfigs.map((c) => (
                        <SelectItem key={c.id} value={c.id}>
                          {c.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleTrainSarm}
                    disabled={sarmBusy || !sshConfigId}
                    className="border-slate-600 text-slate-200 hover:bg-slate-700 text-xs h-8"
                  >
                    {sarmBusy && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                    在 pod 上训 SARM 做对照
                  </Button>
                </>
              )}
              {sarmStatus && (
                <span className="text-slate-400">
                  {sarmStatus.training
                    ? "SARM 训练中…"
                    : sarmStatus.finished_ok
                      ? "SARM 训练完成，可选它作打分后端"
                      : "SARM 训练结束（未确认成功）"}
                  {sarmStatus.log_tail ? ` · ${sarmStatus.log_tail.split("\n").pop()}` : ""}
                </span>
              )}
            </div>
          ) : null}
        </>
      )}
    </div>
  );
};

export default RolloutScoreCard;
