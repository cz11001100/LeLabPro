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
import ProgressCurve from "@/components/jobs/ProgressCurve";
import {
  RolloutComparison,
  RolloutDatasetInfo,
  RolloutScoreState,
  RolloutSummary,
  RolloutPreflight,
  RolloutTakeStats,
  compareRolloutProgress,
  exemptInterventions,
  getRolloutScoreRemoteStatus,
  getRolloutScoreStatus,
  getRolloutSummary,
  getRolloutTakeStats,
  getScoreJob,
  getScoreLogs,
  ScoreLogs,
  listRolloutDatasets,
  preflightRolloutScore,
  promoteRolloutProgress,
  startRolloutScore,
  startRolloutScoreRemote,
} from "@/lib/rolloutEvalApi";

const POLL_MS = 2000;

const THRESHOLD_KEY = "leforge.rolloutScore.threshold";

const initialThreshold = () => {
  const saved = Number(window.localStorage.getItem(THRESHOLD_KEY));
  return Number.isFinite(saved) && saved > 0 ? saved : 0.7;
};

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

/** "pid etime time pcpu" rows -> "进程 04:12 · CPU 67.4%" for the busiest one. */
const procSummary = (raw?: string) => {
  if (!raw) return "";
  const rows = raw
    .trim()
    .split("\n")
    .map((line) => line.trim().split(/\s+/))
    .filter((r) => r.length >= 4);
  if (!rows.length) return "";
  const busiest = rows.sort((a, b) => parseFloat(b[3]) - parseFloat(a[3]))[0];
  return `进程 ${busiest[1]} · CPU ${busiest[3]}%`;
};

/** 125 -> "2:05" */
const mmss = (s: number | null | undefined) => {
  if (s == null) return "—";
  const total = Math.max(0, Math.round(s));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
};

/** 05:58, or "11-05 05:58" once the estimate crosses midnight. */
const clockTime = (d: Date) => {
  const now = new Date();
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  return sameDay
    ? hm
    : `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${hm}`;
};

/** "0-9,12" -> [0..9,12]; empty or unparseable -> null (score everything). */
const parseEpisodeList = (text: string): number[] | null => {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const out: number[] = [];
  for (const part of trimmed.split(",")) {
    const range = part.trim().match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      const [lo, hi] = [Number(range[1]), Number(range[2])];
      for (let i = Math.min(lo, hi); i <= Math.max(lo, hi); i += 1) out.push(i);
    } else if (/^\d+$/.test(part.trim())) {
      out.push(Number(part.trim()));
    } else {
      return null;
    }
  }
  return out.length ? Array.from(new Set(out)).sort((a, b) => a - b) : null;
};

const rateTone = (rate: number | null | undefined) => {
  if (rate == null) return "text-slate-600";
  if (rate >= 0.8) return "text-green-700";
  if (rate >= 0.5) return "text-lime-700";
  if (rate >= 0.25) return "text-amber-700";
  return "text-red-700";
};

/**
 * Closed-loop readout for the policy: score the episodes it actually drove
 * (recorded by the capture modes on the inference page) with a progress model,
 * and report the success rate the open-loop metrics cannot show.
 */
const RolloutScoreCard: React.FC<{ initialRepoId?: string }> = ({
  initialRepoId = "",
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<RolloutDatasetInfo[]>([]);
  const [repoId, setRepoId] = useState(initialRepoId);
  const [summary, setSummary] = useState<RolloutSummary | null>(null);
  const [scoreState, setScoreState] = useState<RolloutScoreState | null>(null);
  const [imageKey, setImageKey] = useState(CAMERA_KEYS[0].value);
  const [stride, setStride] = useState(5);
  const [numSubsampled, setNumSubsampled] = useState(8);
  const [batchSize, setBatchSize] = useState(8);
  const [headMode, setHeadMode] = useState("sparse");
  const [threshold, setThreshold] = useState(initialThreshold);
  const [episodeFilter, setEpisodeFilter] = useState("");
  const [curveEp, setCurveEp] = useState<number | null>(null);
  const [takeStats, setTakeStats] = useState<RolloutTakeStats | null>(null);
  const [remoteTail, setRemoteTail] = useState("");
  const [remoteLogStat, setRemoteLogStat] = useState("");
  const [remoteProc, setRemoteProc] = useState("");
  const [remoteProgress, setRemoteProgress] = useState<{
    done: number;
    total: number;
    percent: number;
    eta_s: number | null;
    elapsed_s: number | null;
  } | null>(null);
  const [runner, setRunner] = useState<"local" | "ssh">("local");
  const [testing, setTesting] = useState(false);
  const [preflight, setPreflight] = useState<RolloutPreflight | null>(null);
  const [logs, setLogs] = useState<ScoreLogs | null>(null);
  const logBoxRef = useRef<HTMLPreElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [exemptMsg, setExemptMsg] = useState("");
  const [backend, setBackend] = useState<"robometer" | "sarm">("robometer");
  const [sarmPath, setSarmPath] = useState("");
  const [comparison, setComparison] = useState<RolloutComparison | null>(null);
  const [sshConfigs, setSshConfigs] = useState<SSHConfig[]>([]);
  const [sshConfigId, setSshConfigId] = useState("");
  const [promoting, setPromoting] = useState("");
  const wasRunningRef = useRef(false);

  const refreshDatasets = useCallback(async () => {
    try {
      const list = await listRolloutDatasets(baseUrl, fetchWithHeaders);
      setDatasets(list);
      setRepoId((prev) => {
        const wanted = prev || initialRepoId;
        return wanted && list.some((d) => d.repo_id === wanted)
          ? wanted
          : list[0]?.repo_id ?? "";
      });
    } catch (e) {
      toast({
        title: "读不到 rollout 数据集",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  }, [baseUrl, fetchWithHeaders, toast, initialRepoId]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Re-attach to a job that outlived this page: the pod job runs detached, and
  // the local child survives a LeForge restart, so the card must not pretend
  // nothing is happening just because it was closed.
  useEffect(() => {
    let cancelled = false;
    getScoreJob(baseUrl, fetchWithHeaders)
      .then((res) => {
        if (cancelled || !res?.active || !res.job) return;
        const job = res.job;
        setRepoId(job.dataset_repo_id || "");
        setBackend(job.backend === "sarm" ? "sarm" : "robometer");
        setRunner(job.runner === "ssh" ? "ssh" : "local");
        if (job.runner === "ssh") {
          setSshConfigId(job.config_id || "");
          setRemoteProgress({
            done: res.status?.episodes_done ?? 0,
            total: res.status?.episodes_total ?? job.episodes_total ?? 0,
            percent: res.status?.percent ?? 0,
            eta_s: res.status?.eta_s ?? null,
            elapsed_s: res.status?.elapsed_s ?? null,
          });
          setRemoteTail("已接回上次的打分任务…");
        } else if (res.status) {
          setScoreState({ ...res.status, running: true });
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (initialRepoId) setRepoId(initialRepoId);
  }, [initialRepoId]);

  const loadSummary = useCallback(
    async (id: string) => {
      if (!id) return;
      setLoading(true);
      try {
        setSummary(
          await getRolloutSummary(baseUrl, fetchWithHeaders, id, {
            successProgress: threshold,
            headMode,
          }),
        );
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
    [baseUrl, fetchWithHeaders, toast, threshold, headMode],
  );

  const loadTakeStats = useCallback(
    async (id: string) => {
      if (!id) {
        setTakeStats(null);
        return;
      }
      try {
        setTakeStats(await getRolloutTakeStats(baseUrl, fetchWithHeaders, id));
      } catch {
        setTakeStats(null);
      }
    },
    [baseUrl, fetchWithHeaders],
  );

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
    setCurveEp(null);
    loadSummary(repoId);
    loadComparison(repoId);
    loadTakeStats(repoId);
  }, [repoId, loadSummary, loadComparison, loadTakeStats]);

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

  // Terminal output: poll while a job is live, and fetch once more when it ends
  // so a crash stays readable instead of vanishing with the process.
  const jobActive = !!scoreState?.running || !!remoteProgress;
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      getScoreLogs(baseUrl, fetchWithHeaders)
        .then((res) => {
          if (!cancelled) setLogs(res);
        })
        .catch(() => {});
    load();
    if (!jobActive) {
      return () => {
        cancelled = true;
      };
    }
    const id = setInterval(load, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobActive]);

  useEffect(() => {
    const box = logBoxRef.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [logs?.text]);

  // Each upstream scorer accepts a different flag set; send only what it has.
  const scoreBody = () => ({
    dataset_repo_id: repoId,
    backend,
    reward_model_path: backend === "sarm" ? sarmPath : "",
    head_mode: headMode,
    ...(backend === "sarm"
      ? { stride }
      : {
          image_key: imageKey,
          num_subsampled_frames: numSubsampled,
          batch_size: batchSize,
          episodes: parseEpisodeList(episodeFilter),
        }),
  });

  const handleTest = async () => {
    if (!repoId) return;
    setTesting(true);
    try {
      setPreflight(
        await preflightRolloutScore(baseUrl, fetchWithHeaders, {
          ...scoreBody(),
          runner,
          config_id: runner === "ssh" ? sshConfigId : "",
        }),
      );
    } catch (e) {
      toast({
        title: "环境检查失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setTesting(false);
    }
  };

  const handleScore = async () => {
    if (!repoId) return;
    setStarting(true);
    try {
      if (runner === "ssh") {
        const res = await startRolloutScoreRemote(baseUrl, fetchWithHeaders, {
          ...scoreBody(),
          config_id: sshConfigId,
          push_to_hub: true,
        });
        setRemoteTail(`已提交：${res.log_path}`);
        setRemoteProgress({ done: 0, total: selected?.episodes ?? 0, percent: 0, eta_s: null, elapsed_s: null });
        toast({ title: "已提交到 pod" });
      } else {
        const res = await startRolloutScore(baseUrl, fetchWithHeaders, scoreBody());
        setScoreState({
          running: true,
          command: res.command,
          log_path: res.log_path,
          episodes_total: selected?.episodes ?? 0,
        });
      }
    } catch (e) {
      // A 400 carries the command to run on a bigger GPU instead.
      const detail = e instanceof Error ? e.message : String(e);
      toast({ title: "打分没跑起来", description: detail, variant: "destructive" });
    } finally {
      setStarting(false);
    }
  };

  // Follow the pod job; the backend pulls and verifies the parquet on finish.
  useEffect(() => {
    if (!remoteTail || !sshConfigId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const st = await getRolloutScoreRemoteStatus(baseUrl, fetchWithHeaders, sshConfigId, repoId, backend);
        if (cancelled) return;
        if (st.running) {
          // Keep the bar visible from second zero: the scorer prints nothing for
          // the first minute (model load), and an empty bar beats no feedback.
          setRemoteProgress((prev) => {
            const total = st.episodes_total || prev?.total || 0;
            return {
              done: st.episodes_done ?? 0,
              total,
              percent: st.percent ?? 0,
              eta_s: st.eta_s ?? null,
              elapsed_s: st.elapsed_s ?? null,
            };
          });
          setRemoteTail(`pod 上跑着… ${st.log_tail ?? ""}`);
          setRemoteLogStat(st.log_stat ?? "");
          setRemoteProc(procSummary(st.proc_stats));
          return;
        }
        if (st.finished) {
          setRemoteTail("");
          setRemoteProgress(null);
          const pulled = st.pulled;
          toast({
            title: st.finished_ok ? "pod 打分结束" : "pod 打分没跑完",
            description: pulled?.success ? pulled.message : pulled?.message || st.log_tail?.slice(-160),
            variant: pulled?.success || !pulled ? "default" : "destructive",
          });
          if (pulled?.success) {
            await Promise.all([refreshDatasets(), loadSummary(repoId), loadComparison(repoId)]);
          }
        }
      } catch {
        // Pod unreachable mid-run; keep polling.
      }
    };
    const id = setInterval(tick, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [remoteTail, sshConfigId, repoId, backend, baseUrl, fetchWithHeaders, refreshDatasets, loadSummary, loadComparison, toast]);

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

  const selected = datasets.find((d) => d.repo_id === repoId);
  const backendsPresent = selected?.backends ?? [];
  const running = !!scoreState?.running;
  const details = summary?.details ?? [];

  return (
    <div className="rounded-lg border border-sky-300 bg-sky-50/60 p-4 space-y-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-blue-700" />
        <h2 className="text-xl font-semibold text-slate-900">数据集打分 Dataset Scoring（进度 · 成功率 · 接管率）</h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={refreshDatasets}
          aria-label="刷新数据集列表"
          className="ml-auto h-7 w-7 text-slate-600 hover:text-slate-900"
        >
          <RefreshCw className="w-4 h-4" />
        </Button>
      </div>

      {datasets.length === 0 ? (
        <p className="text-xs text-slate-600">
          还没有 rollout 数据集。在部署里把采集模式设为纠正采集 / 抢救片段 / 全自动录制，跑完就会出现在这里。
        </p>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-slate-600">数据集</Label>
              <Select value={repoId} onValueChange={setRepoId}>
                <SelectTrigger className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                  <SelectValue placeholder="选择数据集" />
                </SelectTrigger>
                <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                  {datasets.map((d) => (
                    <SelectItem key={d.repo_id} value={d.repo_id}>
                      {d.repo_id} · {d.is_rollout ? "采集" : "训练"}
                      {d.scored ? " · 已打分" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {backend === "robometer" ? (
              <div className="space-y-1">
                <Label className="text-xs text-slate-600">打分用相机</Label>
                <Select value={imageKey} onValueChange={setImageKey}>
                  <SelectTrigger className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                    {CAMERA_KEYS.map((c) => (
                      <SelectItem key={c.value} value={c.value}>
                        {c.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            ) : (
              <div className="space-y-1">
                <Label className="text-xs text-slate-600">进度头（head_mode）</Label>
                <Select value={headMode} onValueChange={setHeadMode}>
                  <SelectTrigger className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                    <SelectItem value="sparse">sparse（阶段进度）</SelectItem>
                    <SelectItem value="dense">dense（逐帧进度）</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
            {backend === "robometer" ? (
              <div className="space-y-1">
                <Label className="text-xs text-slate-600">每帧看多少帧</Label>
                <NumberInput
                  min={2}
                  value={numSubsampled}
                  onChange={(v) => {
                    if (v !== undefined) setNumSubsampled(v);
                  }}
                  className="bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
                />
              </div>
            ) : (
              <div className="space-y-1">
                <Label className="text-xs text-slate-600">帧间隔（stride）</Label>
                <NumberInput
                  min={1}
                  value={stride}
                  onChange={(v) => {
                    if (v !== undefined) setStride(v);
                  }}
                  className="bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
                />
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-1">
              <Label className="text-xs text-slate-600">成功阈值（集末进度）</Label>
              <NumberInput
                step={0.05}
                min={0}
                max={1}
                value={threshold}
                onChange={(v) => {
                  if (v !== undefined) {
                    setThreshold(v);
                    window.localStorage.setItem(THRESHOLD_KEY, String(v));
                  }
                }}
                className="bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
              />
            </div>
            {backend === "robometer" && (
              <div className="space-y-1">
                <Label className="text-xs text-slate-600">episode 子集（如 0-9,12）</Label>
                <Input
                  value={episodeFilter}
                  onChange={(e) => setEpisodeFilter(e.target.value)}
                  placeholder="留空=全部"
                  className="bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
                />
              </div>
            )}
            <div className="space-y-1">
              <Label className="text-xs text-slate-600">批量（forward 内窗口数）</Label>
              <NumberInput
                min={1}
                value={batchSize}
                disabled={backend !== "robometer"}
                onChange={(v) => {
                  if (v !== undefined) setBatchSize(v);
                }}
                className="bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Select value={backend} onValueChange={(v) => setBackend(v as "robometer" | "sarm")}>
              <SelectTrigger className="w-52 bg-sky-100 border-sky-300 text-slate-900 text-xs h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                {BACKENDS.map((b) => (
                  <SelectItem key={b.value} value={b.value}>
                    {b.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={runner} onValueChange={(v) => setRunner(v as "local" | "ssh")}>
              <SelectTrigger className="w-44 bg-sky-100 border-sky-300 text-slate-900 text-xs h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                <SelectItem value="local">本地 GPU</SelectItem>
                <SelectItem value="ssh">SSH — 远程机器</SelectItem>
              </SelectContent>
            </Select>
            {runner === "ssh" && (
              <Select value={sshConfigId} onValueChange={setSshConfigId}>
                <SelectTrigger className="w-52 bg-sky-100 border-sky-300 text-slate-900 text-xs h-9">
                  <SelectValue placeholder="选远程机器" />
                </SelectTrigger>
                <SelectContent className="bg-sky-100 border-sky-300 text-slate-900 text-xs">
                  {sshConfigs.map((c) => (
                    <SelectItem key={c.id} value={c.id}>
                      {c.name} · {c.host}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {backend === "sarm" && (
              <Input
                value={sarmPath}
                onChange={(e) => setSarmPath(e.target.value)}
                placeholder="SARM 模型 repo 或路径"
                className="w-56 bg-sky-100 border-sky-300 text-slate-900 text-xs h-9"
              />
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              onClick={handleTest}
              disabled={!repoId || testing}
              variant="outline"
              className="w-56 border-sky-300 text-black hover:bg-sky-200 hover:text-black text-sm"
            >
              {testing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              {testing ? "测试中…" : "Test"}
            </Button>
            <Button
              onClick={handleScore}
              disabled={
                !repoId ||
                running ||
                starting ||
                (backend === "sarm" && !sarmPath.trim()) ||
                (runner === "ssh" && !sshConfigId)
              }
              className="w-56 bg-blue-600 hover:bg-blue-700 text-white text-sm"
            >
              {running || starting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Activity className="w-4 h-4 mr-2" />
              )}
              {running ? "Scoring…" : "Start Scoring"}
            </Button>
            {selected?.scored && (
              <Button
                onClick={handleExempt}
                disabled={running}
                variant="outline"
                className="w-56 border-sky-300 text-black hover:bg-sky-200 hover:text-black text-sm"
              >
                Exempt Correction Segments
              </Button>
            )}
          </div>

          {selected && (
            <p className="text-xs text-slate-600">
              {selected.episodes} 集 · {selected.frames} 帧 · {selected.modified}
              {exemptMsg ? ` · ${exemptMsg}` : ""}
            </p>
          )}

          {preflight && (
            <div className="space-y-1 rounded border border-sky-300 bg-sky-100/60 p-3">
              {preflight.ok ? (
                <p className="text-xs text-green-700">校验成功</p>
              ) : (
                <>
                  {preflight.checks
                    .filter((c) => !c.ok)
                    .map((c) => (
                      <div key={c.name} className="flex items-start gap-2 text-xs">
                        <span className="text-red-700">✗</span>
                        <span className="text-slate-800">{c.name}</span>
                        {c.detail && (
                          <span className="text-slate-600 break-all">— {c.detail}</span>
                        )}
                      </div>
                    ))}
                  <p className="text-xs text-amber-700 pt-1">上面有未通过项，先解决再打分</p>
                </>
              )}
            </div>
          )}

          {(running || remoteProgress) &&
            (() => {
              const p =
                runner === "ssh"
                  ? remoteProgress
                  : {
                      done: scoreState?.episodes_done ?? 0,
                      total: scoreState?.episodes_total ?? 0,
                      percent: scoreState?.percent ?? 0,
                      eta_s: scoreState?.eta_s ?? null,
                      elapsed_s: scoreState?.elapsed_s ?? null,
                    };
              if (!p) return null;
              const pct = p.percent || (p.total ? Math.round((p.done / p.total) * 100) : 0);
              const finishAt = p.eta_s != null ? new Date(Date.now() + p.eta_s * 1000) : null;
              return (
                <div className="space-y-1">
                  <div className="h-1.5 w-full overflow-hidden rounded bg-sky-200">
                    <div
                      className="h-full bg-blue-500 transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="flex flex-wrap gap-3 text-xs text-slate-600">
                    <span>{p.total ? `第 ${p.done} / ${p.total} 集` : "准备中…"}</span>
                    <span>已用 {mmss(p.elapsed_s)}</span>
                    {p.eta_s != null && <span>预计还剩 {mmss(p.eta_s)}</span>}
                    {finishAt && <span>预计 {clockTime(finishAt)} 完成</span>}
                  </div>
                </div>
              );
            })()}

          {remoteTail && (
            <p className="text-xs text-slate-600 break-all">
              {remoteTail}
              {runner === "ssh" && remoteLogStat
                ? remoteLogStat.startsWith("missing")
                  ? " · 日志文件未生成"
                  : ` · 日志 ${remoteLogStat.split(" ")[0]} 字节`
                : ""}
              {remoteProc ? ` · ${remoteProc}` : ""}
            </p>
          )}

          {logs?.available && logs.text && (
            <div className="space-y-1">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>终端输出{logs.source === "ssh" ? "（pod）" : ""}</span>
                <span className="break-all text-slate-600">{logs.log_path}</span>
              </div>
              <pre
                ref={logBoxRef}
                className="max-h-64 overflow-auto whitespace-pre-wrap break-all rounded border border-sky-300 bg-sky-100 p-2 text-[11px] leading-relaxed text-slate-700"
              >
                {logs.text}
              </pre>
            </div>
          )}

          {summary?.misaligned && (
            <Alert className="bg-amber-100 border-amber-300 text-amber-700">
              <AlertDescription className="text-xs">{summary.error}</AlertDescription>
            </Alert>
          )}

          {summary?.error && (
            <Alert className="bg-red-100 border-red-300 text-red-700">
              <AlertDescription className="text-xs">{summary.error}</AlertDescription>
            </Alert>
          )}

          {summary?.available && (
            <>
              <div className="flex items-end gap-6">
                <div>
                  <div className={`text-4xl font-mono font-bold ${rateTone(summary.success_rate)}`}>
                    {summary.success_rate == null
                      ? "—"
                      : `${Math.round(summary.success_rate * 100)}%`}
                  </div>
                  <div className="text-xs text-slate-600 mt-1">
                    {summary.successes} / {summary.episodes} 集 · 进度阈值 {summary.threshold}
                  </div>
                </div>
                {takeStats && takeStats.takes > 0 && (
                  <div>
                    <div className={`text-4xl font-mono font-bold ${rateTone(1 - (takeStats.takeover_rate ?? 0))}`}>
                      {pctText(takeStats.takeover_rate)}
                    </div>
                    <div className="text-xs text-slate-600 mt-1">
                      接管率 · {takeStats.corrected_takes} / {takeStats.takes} 段
                      {takeStats.failures ? ` · 判失败 ${takeStats.failures}` : ""}
                    </div>
                  </div>
                )}
                {loading && <Loader2 className="w-4 h-4 animate-spin text-slate-600" />}
              </div>

              {summary.has_intervention && (
                <p className="text-xs text-slate-600">
                  这个集含人工纠正段，上面的成功率数的是"存下来的集"，不是模型自己做成的比例——看接管率。
                </p>
              )}

              <div className="max-h-64 overflow-y-auto rounded border border-sky-300">
                <table className="w-full text-xs">
                  <thead className="bg-sky-100/80 text-slate-600 sticky top-0">
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
                      <tr
                        key={e.episode_index}
                        onClick={() => setCurveEp((prev) => (prev === e.episode_index ? null : e.episode_index))}
                        className={`border-t border-sky-300/60 cursor-pointer hover:bg-sky-200/30 ${
                          curveEp === e.episode_index ? "bg-sky-200/40" : ""
                        }`}
                      >
                        <td className="px-2 py-1 text-slate-800">#{e.episode_index}</td>
                        <td className="px-2 py-1 text-right text-slate-600">{e.frames ?? "—"}</td>
                        <td className="px-2 py-1 text-right">{e.progress_start.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.progress_end.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.progress_peak.toFixed(2)}</td>
                        <td className="px-2 py-1 text-right">{e.stalled_at ?? "—"}</td>
                        <td className="px-2 py-1 text-right">
                          <span
                            className={`px-1.5 py-0.5 rounded font-semibold ${
                              e.success
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
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

              {curveEp != null && (
                <ProgressCurve
                  baseUrl={baseUrl}
                  fetchWithHeaders={fetchWithHeaders}
                  repoId={repoId}
                  episodeIndex={curveEp}
                  headMode={summary.column?.replace("progress_", "") || headMode}
                  threshold={summary.threshold ?? threshold}
                />
              )}
            </>
          )}

          {/* Two scorers on the same episodes: Robometer never saw this rig,
              SARM was trained on it. Where they disagree is what to go watch. */}
          {comparison?.available ? (
            <div className="rounded border border-sky-300 bg-sky-100/60 p-3 space-y-2">
              <div className="flex flex-wrap items-center gap-4 text-xs text-slate-700">
                <span className="font-semibold text-slate-900">两后端对照</span>
                <span>{comparison.frames} 帧</span>
                <span>相关 r={comparison.pearson ?? "—"}</span>
                <span>平均差 {comparison.mean_abs_diff ?? "—"}</span>
                <span>成败一致 {pctText(comparison.agreement)}</span>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <span className="text-slate-700">
                  Robometer 成功率 {pctText(comparison.success_rate?.robometer)}
                </span>
                <span className="text-slate-700">
                  SARM 成功率 {pctText(comparison.success_rate?.sarm)}
                </span>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handlePromote("robometer")}
                  disabled={!!promoting}
                  className="border-sky-300 text-black hover:bg-sky-200 hover:text-black text-xs"
                >
                  {promoting === "robometer" && (
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  )}
                  Use Robometer
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handlePromote("sarm")}
                  disabled={!!promoting}
                  className="border-sky-300 text-black hover:bg-sky-200 hover:text-black text-xs"
                >
                  {promoting === "sarm" && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                  Use SARM
                </Button>
              </div>
              {(comparison.disagreements?.length ?? 0) > 0 && (
                <div className="text-xs text-amber-700">
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
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-slate-600">以谁为准</span>
                {backendsPresent
                  .filter((b): b is "robometer" | "sarm" => b === "robometer" || b === "sarm")
                  .map((b) => (
                    <Button
                      key={b}
                      size="sm"
                      variant="outline"
                      onClick={() => handlePromote(b)}
                      disabled={!!promoting}
                      className="border-sky-300 text-black hover:bg-sky-200 hover:text-black text-xs h-8"
                    >
                      {promoting === b && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
                      Use {b === "robometer" ? "Robometer" : "SARM"}
                    </Button>
                  ))}
              </div>
            </div>
          ) : null}
        </>
      )}
    </div>
  );
};

export default RolloutScoreCard;
