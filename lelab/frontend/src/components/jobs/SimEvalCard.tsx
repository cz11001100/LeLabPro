import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Box, Loader2, Play, Square } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  getJobSimEval,
  getJobSimEvalLogs,
  simEvalArtifactUrl,
  startJobSimEval,
  stopJobSimEval,
  JobRecord,
  SimEvalDiagnostics,
  SimEvalHistoryEntry,
  SimEvalResults,
  SimEvalState,
} from "@/lib/jobsApi";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { ModelVariant } from "@/lib/modelGroups";

const SIM_EVAL_CONFIG_KEY = "lelab.simeval.config";
// Empty string = the scene vendored in the repo (sim/so101-tabletop-mujoco),
// resolved by the backend; no Hub download needed.
const DEFAULT_ENV_HUB = "";
const LEGACY_ENV_HUB = "lerobot/so101-tabletop-mujoco";

interface SimEvalConfig {
  nEpisodes: number;
  batchSize: number;
  envHub: string;
  smooth: string;
  ensemble: string;
}

const DEFAULT_CONFIG: SimEvalConfig = {
  nEpisodes: 10,
  batchSize: 1,
  envHub: DEFAULT_ENV_HUB,
  smooth: "0",
  ensemble: "",
};

const loadConfig = (): SimEvalConfig => {
  try {
    const raw = localStorage.getItem(SIM_EVAL_CONFIG_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const p = JSON.parse(raw);
    const savedHub = typeof p.envHub === "string" ? p.envHub : DEFAULT_CONFIG.envHub;
    return {
      nEpisodes: typeof p.nEpisodes === "number" ? p.nEpisodes : DEFAULT_CONFIG.nEpisodes,
      batchSize: typeof p.batchSize === "number" ? p.batchSize : DEFAULT_CONFIG.batchSize,
      // Older saves pointed at the Hub copy of the same scene; prefer the builtin one.
      envHub: savedHub === LEGACY_ENV_HUB ? DEFAULT_CONFIG.envHub : savedHub,
      smooth: typeof p.smooth === "string" ? p.smooth : DEFAULT_CONFIG.smooth,
      ensemble: typeof p.ensemble === "string" ? p.ensemble : DEFAULT_CONFIG.ensemble,
    };
  } catch {
    return DEFAULT_CONFIG;
  }
};

const saveConfig = (c: SimEvalConfig) => {
  try {
    localStorage.setItem(SIM_EVAL_CONFIG_KEY, JSON.stringify(c));
  } catch {
    // Storage may be unavailable — ignore
  }
};

const fmt = (n: number | null | undefined, digits = 1, unit = ""): string =>
  n == null || Number.isNaN(n) ? "—" : `${n.toFixed(digits)}${unit}`;

const Metric: React.FC<{ label: string; value: string; hint?: string }> = ({
  label,
  value,
  hint,
}) => (
  <div className="bg-slate-900/50 rounded-lg p-3">
    <div className="text-xs text-slate-500">{label}</div>
    <div className="text-lg font-semibold text-white">{value}</div>
    {hint ? <div className="text-[11px] text-slate-500 mt-0.5">{hint}</div> : null}
  </div>
);

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <Label className="text-slate-400 text-xs">{label}</Label>
    {children}
  </div>
);

const LogBox: React.FC<{ text: string }> = ({ text }) => {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text]);
  return (
    <pre
      ref={ref}
      className="bg-slate-950 border border-slate-700 rounded-lg p-3 font-mono text-xs text-slate-300 max-h-64 overflow-y-auto whitespace-pre-wrap break-all"
    >
      {text || "…"}
    </pre>
  );
};

const DiagnosticsPanel: React.FC<{ d: SimEvalDiagnostics }> = ({ d }) => {
  if (d.error)
    return <div className="text-xs text-slate-500">诊断不可用：{d.error}</div>;
  const spread = d.traj_spread_norm;
  const tone =
    spread == null
      ? "bg-slate-900/50 text-slate-300"
      : spread < 0.05
        ? "bg-red-900/30 text-red-300 border border-red-800/50"
        : spread < 0.2
          ? "bg-amber-900/30 text-amber-300 border border-amber-800/50"
          : "bg-emerald-900/30 text-emerald-300 border border-emerald-800/50";
  const px = (v: [number, number] | null | undefined) => (v ? `${v[0]}, ${v[1]}` : "—");
  return (
    <div className="space-y-3 rounded-lg border border-slate-700 bg-slate-900/40 p-3">
      <div className="text-sm font-medium text-slate-300">评测诊断</div>
      {d.verdict && <div className={`text-sm rounded-lg p-2 ${tone}`}>{d.verdict}</div>}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <Metric
          label="跨集轨迹差异"
          value={fmt(spread, 3)}
          hint="0 = 各集完全相同；越大越随画面变化"
        />
        <Metric label="轨迹差异（关节度）" value={fmt(d.traj_spread_deg, 2, "°")} />
        <Metric
          label="集内步间变化"
          value={fmt(d.within_motion_norm, 3)}
          hint="机械臂自身动作幅度，作对照"
        />
      </div>
      {(d.episodes ?? []).length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-700">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-slate-900/70 text-slate-400">
                <th className="px-3 py-2 text-left">集</th>
                <th className="px-3 py-2 text-left">红球（up 像素）</th>
                <th className="px-3 py-2 text-left">蓝球（up 像素）</th>
                <th className="px-3 py-2 text-right">步数</th>
              </tr>
            </thead>
            <tbody>
              {(d.episodes ?? []).map((e) => (
                <tr key={e.episode} className="border-t border-slate-700/60">
                  <td className="px-3 py-2 text-white font-mono">{e.episode}</td>
                  <td className="px-3 py-2 text-slate-200">{px(e.balls_up_px?.red)}</td>
                  <td className="px-3 py-2 text-slate-200">{px(e.balls_up_px?.blue)}</td>
                  <td className="px-3 py-2 text-right text-slate-400">{e.steps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {d.domain_gap && Object.keys(d.domain_gap).length > 0 && (
        <div className="text-xs text-slate-400 space-y-1">
          <div className="text-slate-300">
            仿真画面 vs 训练画面（均值差 0 = 相同；std 比 1 = 相同）
          </div>
          {Object.entries(d.domain_gap).map(([cam, g]) => (
            <div key={cam}>
              <span className="font-mono text-slate-200">{cam}</span>：均值差 {g.mean_abs_shift} ·
              std 比 {g.std_ratio}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const HistoryTable: React.FC<{ history: SimEvalHistoryEntry[]; currentRunDir?: string | null }> = ({
  history,
  currentRunDir,
}) => {
  if (history.length === 0) return null;
  const ranked = [...history].sort((a, b) => (b.pc_success ?? -1) - (a.pc_success ?? -1));
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-slate-300">Checkpoint 排序</div>
      <div className="overflow-x-auto rounded-lg border border-slate-700">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-slate-900/70 text-slate-400">
              <th className="px-3 py-2 text-left">#</th>
              <th className="px-3 py-2 text-left">Step</th>
              <th className="px-3 py-2 text-right">成功率</th>
              <th className="px-3 py-2 text-right">平均最大奖励</th>
              <th className="px-3 py-2 text-right">成功集数</th>
              <th className="px-3 py-2 text-left">时间</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((h, i) => (
              <tr
                key={`${h.ts}-${h.step}`}
                className={`border-t border-slate-700/60 ${
                  h.run_dir === currentRunDir ? "bg-cyan-900/20" : ""
                }`}
              >
                <td className="px-3 py-2 text-slate-400">{i + 1}</td>
                <td className="px-3 py-2 text-white font-mono">{h.step}</td>
                <td className="px-3 py-2 text-right text-cyan-300">{fmt(h.pc_success, 1, "%")}</td>
                <td className="px-3 py-2 text-right text-slate-200">{fmt(h.avg_max_reward, 3)}</td>
                <td className="px-3 py-2 text-right text-slate-200">
                  {h.n_success ?? "—"}/{h.n_episodes ?? "—"}
                </td>
                <td className="px-3 py-2 text-slate-400">
                  {h.ts ? new Date(h.ts * 1000).toLocaleString() : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const SimEvalCard: React.FC<{
  job: JobRecord;
  variants?: ModelVariant[];
  onSelectJob?: (jobId: string) => void;
}> = ({ job, variants, onSelectJob }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const saved = loadConfig();
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [step, setStep] = useState<string>("");
  const [nEpisodes, setNEpisodes] = useState(saved.nEpisodes);
  const [batchSize, setBatchSize] = useState(saved.batchSize);
  const [envHub, setEnvHub] = useState(saved.envHub);
  const [smooth, setSmooth] = useState(saved.smooth);
  const [ensemble, setEnsemble] = useState(saved.ensemble);
  const [state, setState] = useState<SimEvalState>(job.sim_eval_state ?? "none");
  const [results, setResults] = useState<SimEvalResults | null>(job.sim_eval_results ?? null);
  const [history, setHistory] = useState<SimEvalHistoryEntry[]>([]);
  const [progress, setProgress] = useState<{ batches_done: number; batches_total: number } | null>(
    null,
  );
  const [log, setLog] = useState("");
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);

  const localOnly = job.runner === "local" || job.runner === "imported";
  const running = state === "running";

  useEffect(() => {
    const controller = new AbortController();
    listJobCheckpoints(baseUrl, fetchWithHeaders, job.id, controller.signal)
      .then((cs) => {
        const local = cs.filter((c) => c.source === "local");
        setCheckpoints(local);
        if (local.length) setStep(String(local[local.length - 1].step));
      })
      .catch(() => {});
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, job.id]);

  useEffect(() => {
    setState(job.sim_eval_state ?? "none");
    setResults(job.sim_eval_results ?? null);
  }, [job.id, job.sim_eval_state, job.sim_eval_results]);

  // Restore status + history on mount (the job record may predate the last run).
  useEffect(() => {
    const controller = new AbortController();
    getJobSimEval(baseUrl, fetchWithHeaders, job.id, controller.signal)
      .then((r) => {
        setState(r.sim_eval_state);
        setResults(r.sim_eval_results);
        setHistory(r.history);
        setProgress(r.progress);
      })
      .catch(() => {});
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, job.id]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      getJobSimEval(baseUrl, fetchWithHeaders, job.id)
        .then((r) => {
          setState(r.sim_eval_state);
          setResults(r.sim_eval_results);
          setHistory(r.history);
          setProgress(r.progress);
        })
        .catch(() => {});
      getJobSimEvalLogs(baseUrl, fetchWithHeaders, job.id, 200)
        .then((r) => setLog(r.log))
        .catch(() => {});
    }, 2000);
    return () => window.clearInterval(timer);
  }, [running, baseUrl, fetchWithHeaders, job.id]);

  // Show the log of a finished/failed run too, once.
  useEffect(() => {
    if (running || state === "none") return;
    getJobSimEvalLogs(baseUrl, fetchWithHeaders, job.id, 200)
      .then((r) => setLog(r.log))
      .catch(() => {});
  }, [running, state, baseUrl, fetchWithHeaders, job.id]);

  const run = async () => {
    saveConfig({ nEpisodes, batchSize, envHub, smooth, ensemble });
    setStarting(true);
    setState("running");
    setResults(null);
    setLog("");
    setProgress(null);
    try {
      await startJobSimEval(baseUrl, fetchWithHeaders, job.id, {
        step: step ? Number(step) : undefined,
        n_episodes: nEpisodes,
        batch_size: batchSize,
        env_hub_path: envHub.trim(),
        action_smooth: Math.min(0.95, Math.max(0, Number(smooth) || 0)),
        temporal_ensemble: ensemble.trim() ? Number(ensemble) : undefined,
      });
      toast({ title: "仿真评测已启动" });
    } catch (e) {
      setState("failed");
      toast({
        title: "仿真评测启动失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStarting(false);
    }
  };

  const stop = async () => {
    if (stopping) return;
    setStopping(true);
    try {
      await stopJobSimEval(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "正在停止仿真评测…" });
    } catch (e) {
      toast({
        title: "停止失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setStopping(false);
    }
  };

  const overall = results?.overall;
  const pct =
    progress && progress.batches_total > 0
      ? Math.min(100, Math.round((progress.batches_done / progress.batches_total) * 100))
      : null;

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader className="pb-3">
        <CardTitle className="text-white flex items-center gap-2 text-base">
          <Box className="h-5 w-5 text-cyan-400" />
          仿真评测
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {variants && variants.length > 0 ? (
            <Field label="Checkpoint">
              <CheckpointDropdown
                checkpoints={variants.map((v) => ({
                  step: v.step,
                  source: "local" as const,
                  ref: v.jobId,
                }))}
                selectedStep={variants.find((v) => v.jobId === job.id)?.step ?? null}
                onChange={(s) => {
                  const v = variants.find((x) => x.step === s);
                  if (v) onSelectJob?.(v.jobId);
                }}
                triggerClassName="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
              />
            </Field>
          ) : (
            <Field label="Checkpoint">
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={step ? Number(step) : null}
                onChange={(s) => setStep(String(s))}
                triggerClassName="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
              />
            </Field>
          )}
          <Field label="集数">
            <NumberInput
              value={nEpisodes}
              min={1}
              onChange={(v) => v !== undefined && setNEpisodes(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="并行环境">
            <NumberInput
              value={batchSize}
              min={1}
              onChange={(v) => v !== undefined && setBatchSize(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="环境仓库">
            <Input
              value={envHub}
              onChange={(e) => setEnvHub(e.target.value)}
              placeholder="留空 = 内置本地场景"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 font-mono text-xs"
            />
            <div className="text-[11px] text-slate-500 mt-1">
              留空用仓库内置 sim/so101-tabletop-mujoco（免下载）；可填 Hub 仓库或本地路径
            </div>
          </Field>
          <Field label="动作平滑（0=关）">
            <Input
              value={smooth}
              onChange={(e) => setSmooth(e.target.value)}
              placeholder="0"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 font-mono text-xs"
            />
            <div className="text-[11px] text-slate-500 mt-1">
              关节目标低通滤波，越大越稳但越滞后；建议 0–0.5
            </div>
          </Field>
          <Field label="时序集成（留空=关）">
            <Input
              value={ensemble}
              onChange={(e) => setEnsemble(e.target.value)}
              placeholder="0.01"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 font-mono text-xs"
            />
            <div className="text-[11px] text-slate-500 mt-1">
              ACT 每步重算动作块并加权平均，更平滑也更跟手；建议 0.01
            </div>
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          {running ? (
            <Button
              onClick={stop}
              disabled={stopping}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              {stopping ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Square className="h-4 w-4 mr-2" />
              )}
              停止
            </Button>
          ) : (
            <Button
              onClick={run}
              disabled={!localOnly || starting || checkpoints.length === 0}
              className="bg-cyan-600 hover:bg-cyan-700 text-white disabled:opacity-40"
            >
              {starting ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              开始仿真评测
            </Button>
          )}
          {!localOnly && (
            <span className="text-xs text-slate-500">需要本地 checkpoint（先导出）</span>
          )}
          {running && pct != null && (
            <div className="flex items-center gap-2 flex-1 min-w-[160px]">
              <div className="h-2 flex-1 rounded-full bg-slate-700 overflow-hidden">
                <div
                  className="h-full rounded-full bg-cyan-500 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-slate-400 whitespace-nowrap">
                {progress?.batches_done}/{progress?.batches_total} 批 — {pct}%
              </span>
            </div>
          )}
        </div>

        {(running || state === "failed") && <LogBox text={log} />}

        {state === "failed" && results?.error && (
          <div className="text-sm text-red-300 bg-red-900/20 border border-red-800/50 rounded-lg p-3">
            {results.error}
          </div>
        )}

        {state === "done" && overall && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <Metric
                label="成功率"
                value={fmt(overall.pc_success, 1, "%")}
                hint={
                  overall.pc_success_ci95
                    ? `95% 区间 ${fmt(overall.pc_success_ci95[0], 1)}–${fmt(overall.pc_success_ci95[1], 1)}%`
                    : undefined
                }
              />
              <Metric label="平均最大奖励" value={fmt(overall.avg_max_reward, 3)} />
              <Metric
                label="成功集数"
                value={`${overall.n_success}/${overall.n_episodes}`}
              />
              <Metric label="用时" value={fmt(results?.eval_s ?? overall.eval_s, 0, "s")} />
            </div>
            {(results?.videos ?? []).length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {(results?.videos ?? []).map((v) => (
                  <video
                    key={v}
                    controls
                    preload="metadata"
                    src={simEvalArtifactUrl(baseUrl, job.id, v)}
                    className="w-full rounded-lg border border-slate-700 bg-black"
                  />
                ))}
              </div>
            )}
            {results?.diagnostics && <DiagnosticsPanel d={results.diagnostics} />}
          </>
        )}

        <HistoryTable history={history} currentRunDir={results?.run_dir} />
      </CardContent>
    </Card>
  );
};

export default SimEvalCard;