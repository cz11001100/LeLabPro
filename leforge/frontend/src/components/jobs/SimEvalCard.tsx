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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  createSimEnvironment,
  getJobSimEval,
  getJobSimEvalLogs,
  listSimEnvironments,
  openSimEnvironment,
  simEvalArtifactUrl,
  startJobSimEval,
  stopJobSimEval,
  JobRecord,
  SimEnvOption,
  SimEvalDiagnostics,
  SimEvalHistoryEntry,
  SimEvalResults,
  SimEvalState,
} from "@/lib/jobsApi";
import { getCheckpointPolicyConfig, JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { ModelVariant } from "@/lib/modelGroups";

const SIM_EVAL_CONFIG_KEY = "leforge.simeval.config";
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
  <div className="bg-sky-100/70 rounded-lg p-3">
    <div className="text-xs text-slate-600">{label}</div>
    <div className="text-lg font-semibold text-slate-900">{value}</div>
    {hint ? <div className="text-[11px] text-slate-600 mt-0.5">{hint}</div> : null}
  </div>
);

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <Label className="text-slate-600 text-xs">{label}</Label>
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
      className="bg-sky-50 border border-sky-300 rounded-lg p-3 font-mono text-xs text-slate-700 max-h-64 overflow-y-auto whitespace-pre-wrap break-all"
    >
      {text || "…"}
    </pre>
  );
};

const DiagnosticsPanel: React.FC<{ d: SimEvalDiagnostics }> = ({ d }) => {
  if (d.error)
    return <div className="text-xs text-slate-600">诊断不可用：{d.error}</div>;
  const spread = d.traj_spread_norm;
  const tone =
    spread == null
      ? "bg-sky-100/70 text-slate-700"
      : spread < 0.05
        ? "bg-red-100 text-red-700 border border-red-300/50"
        : spread < 0.2
          ? "bg-amber-100 text-amber-700 border border-amber-300/50"
          : "bg-emerald-100 text-emerald-700 border border-emerald-300/50";
  const px = (v: [number, number] | null | undefined) => (v ? `${v[0]}, ${v[1]}` : "—");
  return (
    <div className="space-y-3 rounded-lg border border-sky-300 bg-sky-100/60 p-3">
      <div className="text-sm font-medium text-slate-700">评测诊断</div>
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
        <div className="overflow-x-auto rounded-lg border border-sky-300">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-sky-100/90 text-slate-600">
                <th className="px-3 py-2 text-left">集</th>
                <th className="px-3 py-2 text-left">红球（up 像素）</th>
                <th className="px-3 py-2 text-left">蓝球（up 像素）</th>
                <th className="px-3 py-2 text-right">步数</th>
              </tr>
            </thead>
            <tbody>
              {(d.episodes ?? []).map((e) => (
                <tr key={e.episode} className="border-t border-sky-300/60">
                  <td className="px-3 py-2 text-slate-900 font-mono">{e.episode}</td>
                  <td className="px-3 py-2 text-slate-800">{px(e.balls_up_px?.red)}</td>
                  <td className="px-3 py-2 text-slate-800">{px(e.balls_up_px?.blue)}</td>
                  <td className="px-3 py-2 text-right text-slate-600">{e.steps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {d.domain_gap && Object.keys(d.domain_gap).length > 0 && (
        <div className="text-xs text-slate-600 space-y-1">
          <div className="text-slate-700">
            仿真画面 vs 训练画面（均值差 0 = 相同；std 比 1 = 相同）
          </div>
          {Object.entries(d.domain_gap).map(([cam, g]) => (
            <div key={cam}>
              <span className="font-mono text-slate-800">{cam}</span>：均值差 {g.mean_abs_shift} ·
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
      <div className="text-sm font-medium text-slate-700">Checkpoint 排序</div>
      <div className="overflow-x-auto rounded-lg border border-sky-300">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-sky-100/90 text-slate-600">
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
                className={`border-t border-sky-300/60 ${
                  h.run_dir === currentRunDir ? "bg-cyan-100" : ""
                }`}
              >
                <td className="px-3 py-2 text-slate-600">{i + 1}</td>
                <td className="px-3 py-2 text-slate-900 font-mono">{h.step}</td>
                <td className="px-3 py-2 text-right text-cyan-700">{fmt(h.pc_success, 1, "%")}</td>
                <td className="px-3 py-2 text-right text-slate-800">{fmt(h.avg_max_reward, 3)}</td>
                <td className="px-3 py-2 text-right text-slate-800">
                  {h.n_success ?? "—"}/{h.n_episodes ?? "—"}
                </td>
                <td className="px-3 py-2 text-slate-600">
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
  const [envOptions, setEnvOptions] = useState<SimEnvOption[]>([]);
  const [showCustomEnv, setShowCustomEnv] = useState(false);
  const [ensembleSupported, setEnsembleSupported] = useState<boolean | null>(null);
  const [showNewEnv, setShowNewEnv] = useState(false);
  const [newEnvName, setNewEnvName] = useState("");
  const [creatingEnv, setCreatingEnv] = useState(false);
  const [openingViewer, setOpeningViewer] = useState(false);
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
  const knownEnv = envHub === "" || envOptions.some((o) => o.value === envHub);
  const customEnv = showCustomEnv || !knownEnv;
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
    const controller = new AbortController();
    listSimEnvironments(baseUrl, fetchWithHeaders, controller.signal)
      .then(setEnvOptions)
      .catch(() => {});
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders]);

  // Temporal ensembling exists only on some policies (ACT today): show the
  // knob once the selected checkpoint reports it, hide it for the rest.
  useEffect(() => {
    setEnsembleSupported(null);
    if (checkpoints.length === 0) return;
    const target = step ? Number(step) : checkpoints[checkpoints.length - 1]?.step;
    if (target == null || Number.isNaN(target)) return;
    const controller = new AbortController();
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, job.id, target, controller.signal)
      .then((s) => setEnsembleSupported(!!s.temporal_ensemble))
      .catch(() => setEnsembleSupported(null));
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, job.id, step, checkpoints]);

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

  const handleCreateEnv = async () => {
    const name = newEnvName.trim();
    if (!name || creatingEnv) return;
    setCreatingEnv(true);
    try {
      const entry = await createSimEnvironment(baseUrl, fetchWithHeaders, name);
      try {
        const list = await listSimEnvironments(baseUrl, fetchWithHeaders);
        setEnvOptions(list);
      } catch {
        // Keep the current list; the new scene is still selected below.
      }
      setEnvHub(entry.value);
      setShowCustomEnv(false);
      setShowNewEnv(false);
      setNewEnvName("");
      toast({ title: `已创建本地场景 ${name}` });
    } catch (e) {
      toast({
        title: "创建场景失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setCreatingEnv(false);
    }
  };

  const handleOpenViewer = async () => {
    if (openingViewer) return;
    setOpeningViewer(true);
    try {
      await openSimEnvironment(baseUrl, fetchWithHeaders, envHub);
      toast({ title: "已在 MuJoCo 窗口打开场景" });
    } catch (e) {
      toast({
        title: "打开 MuJoCo 失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setOpeningViewer(false);
    }
  };

  const selectedEnvKind =
    envHub === "" ? "builtin" : envOptions.find((o) => o.value === envHub)?.kind;
  const canOpenViewer = selectedEnvKind === "builtin" || selectedEnvKind === "local";

  const overall = results?.overall;
  const pct =
    progress && progress.batches_total > 0
      ? Math.min(100, Math.round((progress.batches_done / progress.batches_total) * 100))
      : null;

  return (
    <Card className="bg-sky-50/70 border-sky-300 rounded-xl">
      <CardHeader className="pb-3">
        <CardTitle className="text-slate-900 flex items-center gap-2 text-base">
          <Box className="h-5 w-5 text-cyan-700" />
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
                triggerClassName="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1"
              />
            </Field>
          ) : (
            <Field label="Checkpoint">
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={step ? Number(step) : null}
                onChange={(s) => setStep(String(s))}
                triggerClassName="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1"
              />
            </Field>
          )}
          <Field label="集数">
            <NumberInput
              value={nEpisodes}
              min={1}
              onChange={(v) => v !== undefined && setNEpisodes(v)}
              className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1"
            />
          </Field>
          <Field label="并行环境">
            <NumberInput
              value={batchSize}
              min={1}
              onChange={(v) => v !== undefined && setBatchSize(v)}
              className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1"
            />
          </Field>
          <Field label="环境仓库">
            <Select
              value={customEnv ? "__custom__" : envHub || "builtin"}
              onValueChange={(v) => {
                if (v === "__custom__") {
                  setShowCustomEnv(true);
                  return;
                }
                setShowCustomEnv(false);
                setEnvHub(v === "builtin" ? "" : v);
              }}
            >
              <SelectTrigger className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1">
                <SelectValue placeholder="选择环境" />
              </SelectTrigger>
              <SelectContent>
                {envOptions.map((o) => (
                  <SelectItem key={o.value || "builtin"} value={o.value || "builtin"}>
                    {o.label}
                  </SelectItem>
                ))}
                <SelectItem value="__custom__">自定义（Hub 仓库或本地路径）</SelectItem>
              </SelectContent>
            </Select>
            {(showCustomEnv || !knownEnv) && (
              <Input
                value={envHub}
                onChange={(e) => setEnvHub(e.target.value)}
                placeholder="例如 username/repo 或 /path/to/env"
                className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1 font-mono text-xs"
              />
            )}
            <div className="text-[11px] text-slate-600 mt-1">
              默认内置场景；可选本机 sim/ 下的其它场景，或任意 Hub 仓库
            </div>
            <div className="flex flex-wrap items-center gap-2 mt-2">
              <Button
                type="button"
                onClick={() => setShowNewEnv((v) => !v)}
                className="h-7 px-2 text-xs bg-sky-200 hover:bg-sky-200 text-slate-900"
              >
                新建本地场景
              </Button>
              <Button
                type="button"
                onClick={handleOpenViewer}
                disabled={!canOpenViewer || openingViewer}
                className="h-7 px-2 text-xs bg-sky-200 hover:bg-sky-200 text-slate-900 disabled:opacity-40"
              >
                {openingViewer ? "启动中…" : "在 MuJoCo 打开"}
              </Button>
            </div>
            {showNewEnv && (
              <div className="flex items-center gap-2 mt-2">
                <Input
                  value={newEnvName}
                  onChange={(e) => setNewEnvName(e.target.value)}
                  placeholder="新场景名，如 my-tabletop"
                  className="h-8 bg-sky-100 border-sky-300 text-slate-900 rounded-lg font-mono text-xs"
                />
                <Button
                  type="button"
                  onClick={handleCreateEnv}
                  disabled={creatingEnv || !newEnvName.trim()}
                  className="h-8 px-3 text-xs bg-cyan-600 hover:bg-cyan-700 text-white disabled:opacity-40"
                >
                  {creatingEnv ? "创建中…" : "创建"}
                </Button>
              </div>
            )}
          </Field>
          <Field label="动作平滑">
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={Number(smooth) || 0}
              onChange={(e) => setSmooth(e.target.value)}
              className="w-full mt-2 accent-cyan-500"
            />
            <div className="text-[11px] text-slate-600 mt-1">
              当前 {Number(smooth) || 0}（0=关；越大越稳但越滞后）
            </div>
          </Field>
          {ensembleSupported !== false && (
            <Field label="时序集成（留空=关）">
              <Input
                value={ensemble}
                onChange={(e) => setEnsemble(e.target.value)}
                placeholder="0.01"
                className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg mt-1 font-mono text-xs"
              />
              <div className="text-[11px] text-slate-600 mt-1">
                当前模型支持该特性（如 ACT）时启用；建议 0.01
              </div>
            </Field>
          )}
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
            <span className="text-xs text-slate-600">需要本地 checkpoint（先导出）</span>
          )}
          {running && pct != null && (
            <div className="flex items-center gap-2 flex-1 min-w-[160px]">
              <div className="h-2 flex-1 rounded-full bg-sky-200 overflow-hidden">
                <div
                  className="h-full rounded-full bg-cyan-500 transition-all duration-500"
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="text-xs text-slate-600 whitespace-nowrap">
                {progress?.batches_done}/{progress?.batches_total} 批 — {pct}%
              </span>
            </div>
          )}
        </div>

        {(running || state === "failed") && <LogBox text={log} />}

        {state === "failed" && results?.error && (
          <div className="text-sm text-red-700 bg-red-100 border border-red-300/50 rounded-lg p-3">
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
                    className="w-full rounded-lg border border-sky-300 bg-sky-50"
                  />
                ))}
              </div>
            )}
            {results?.notes && (
              <div className="text-xs text-amber-700 bg-amber-100 border border-amber-300/40 rounded-lg p-2">
                {results.notes}
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