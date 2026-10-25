import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FlaskConical, Loader2, ShieldAlert, ShieldCheck, Square } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  evaluationChartUrl,
  getJobValidation,
  startJobValidation,
  stopJobValidation,
  JobRecord,
  ValidationResults,
  ValidationState,
  computeDimensionScores,
  GRADE_COLORS,
  GRADE_LABELS,
} from "@/lib/jobsApi";
import { JobCheckpoint, listJobCheckpoints } from "@/lib/checkpointsApi";
import { DatasetItem, listDatasets } from "@/lib/replayApi";

const CHART_CAPTIONS: Record<string, string> = {
  per_joint: "每个关节的开环误差（对比「永远预测均值」的基线）",
  horizon: "误差随 chunk 步数的增长 — 决定多久该重新推理一次",
  episodes: "逐集 × 逐关节误差（越亮越离谱）— 用来揪出坏数据",
  overlay: "预测动作 vs 录制动作（0 号关节）",
  probe: "感知敏感度：扰动输入后动作变化了多少",
};

function chartCaption(key: string): string {
  if (key.startsWith("aim")) {
    return "指向反演：绿框=球真实位置，橙框=人的动作反推出的位置，蓝框=模型动作反推出的位置";
  }
  if (key.startsWith("saliency")) {
    return "定位热区图：白框 #1–#3 = 模型动作最依赖的画面区域，绿框 = 球的真实位置（OpenCV 颜色掩码）";
  }
  return CHART_CAPTIONS[key] ?? key;
}

function fmt(n: number | null | undefined, digits = 2, unit = ""): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${n.toFixed(digits)}${unit}`;
}

const Metric: React.FC<{ label: string; value: string; hint?: string; tone?: "ok" | "bad" }> = ({
  label,
  value,
  hint,
  tone,
}) => (
  <div className="bg-slate-900/50 rounded-lg p-3">
    <div className="text-xs text-slate-500">{label}</div>
    <div
      className={`text-lg font-semibold ${
        tone === "bad" ? "text-red-300" : tone === "ok" ? "text-green-300" : "text-white"
      }`}
    >
      {value}
    </div>
    {hint ? <div className="text-[11px] text-slate-500 mt-0.5">{hint}</div> : null}
  </div>
);

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({ label, children }) => (
  <div>
    <Label className="text-slate-400 text-xs">{label}</Label>
    {children}
  </div>
);

const EVAL_CONFIG_KEY = "lelab.eval.config";

interface EvalConfig {
  numBatches: number;
  batchSize: number;
  seed: number;
  holdout: number;
  framesPerEpisode: number;
  probe: boolean;
  saliencyFrames: number;
  aimFrames: number;
  device: string;
}

const DEFAULT_EVAL_CONFIG: EvalConfig = {
  numBatches: 20,
  batchSize: 8,
  seed: 0,
  holdout: 20,
  framesPerEpisode: 8,
  probe: true,
  saliencyFrames: 2,
  aimFrames: 4,
  device: "cuda",
};

const loadEvalConfig = (): EvalConfig => {
  try {
    const raw = localStorage.getItem(EVAL_CONFIG_KEY);
    if (!raw) return DEFAULT_EVAL_CONFIG;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_EVAL_CONFIG, ...parsed };
  } catch {
    return DEFAULT_EVAL_CONFIG;
  }
};

const saveEvalConfig = (config: Partial<EvalConfig>) => {
  try {
    const existing = loadEvalConfig();
    localStorage.setItem(EVAL_CONFIG_KEY, JSON.stringify({ ...existing, ...config }));
  } catch {
    // Storage may be unavailable — ignore
  }
};

const EvaluationCard: React.FC<{ job: JobRecord }> = ({ job }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [step, setStep] = useState<string>("");
  
  // Load saved config from localStorage
  const savedConfig = loadEvalConfig();
  const [numBatches, setNumBatches] = useState(savedConfig.numBatches);
  const [batchSize, setBatchSize] = useState(savedConfig.batchSize);
  const [seed, setSeed] = useState(savedConfig.seed);
  const [holdout, setHoldout] = useState(savedConfig.holdout);
  const [framesPerEpisode, setFramesPerEpisode] = useState(savedConfig.framesPerEpisode);
  const [probe, setProbe] = useState(savedConfig.probe);
  const [saliencyFrames, setSaliencyFrames] = useState(savedConfig.saliencyFrames);
  const [aimFrames, setAimFrames] = useState(savedConfig.aimFrames);
  const [device, setDevice] = useState(savedConfig.device);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [valDataset, setValDataset] = useState("");
  const [state, setState] = useState<ValidationState>(job.validation_state ?? "none");
  const [results, setResults] = useState<ValidationResults | null>(job.validation_results ?? null);
  const [expandedDim, setExpandedDim] = useState<string | null>(null);

  // 计算三维度评分
  const dimensions = results?.success ? computeDimensionScores(results) : null;

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
    listDatasets(baseUrl, fetchWithHeaders, controller.signal)
      .then((d) => setDatasets(d.filter((x) => x.source !== "hub")))
      .catch(() => {});
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    setState(job.validation_state ?? "none");
    setResults(job.validation_results ?? null);
  }, [job.id, job.validation_state, job.validation_results]);

  // The parent page refetches the job on its own cadence; poll directly so the
  // card flips out of "running" as soon as the evaluator thread finishes.
  useEffect(() => {
    if (state !== "running") return;
    const timer = window.setInterval(() => {
      getJobValidation(baseUrl, fetchWithHeaders, job.id)
        .then((r) => {
          setState(r.validation_state);
          setResults(r.validation_results);
        })
        .catch(() => {});
    }, 3000);
    return () => window.clearInterval(timer);
  }, [state, baseUrl, fetchWithHeaders, job.id]);

  const run = async () => {
    // Save current config to localStorage before running
    saveEvalConfig({
      numBatches,
      batchSize,
      seed,
      holdout,
      framesPerEpisode,
      probe,
      saliencyFrames,
      aimFrames,
      device,
    });

    setStarting(true);
    setState("running");
    setResults(null);
    try {
      await startJobValidation(baseUrl, fetchWithHeaders, job.id, {
        step: step ? Number(step) : undefined,
        num_batches: numBatches,
        batch_size: batchSize,
        seed,
        holdout_episodes: valDataset ? 0 : holdout,
        dataset_repo_id: valDataset || undefined,
        frames_per_episode: framesPerEpisode,
        probe,
        saliency_frames: saliencyFrames,
        aim_frames: aimFrames,
        device,
      });
      toast({ title: "评测已启动" });
    } catch (e) {
      setState("failed");
      toast({
        title: "评测启动失败",
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
      await stopJobValidation(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "正在停止评测…" });
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

  const jointNames = results?.joint_names ?? [];
  const perJoint = results?.per_joint_mae_deg ?? [];
  const naive = results?.per_joint_naive_mae_deg ?? [];
  const relative = results?.per_joint_mae_vs_naive ?? [];
  const vision = results?.probe?.vision_sensitivity ?? null;
  const running = state === "running";

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader className="pb-3">
        <CardTitle className="text-white flex items-center gap-2 text-base">
          <FlaskConical className="h-5 w-5 text-purple-400" />
          模型评测
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <Field label="批次数">
            <NumberInput
              value={numBatches}
              onChange={(v) => v !== undefined && setNumBatches(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="每批样本">
            <NumberInput
              value={batchSize}
              onChange={(v) => v !== undefined && setBatchSize(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="随机种子">
            <NumberInput
              value={seed}
              onChange={(v) => v !== undefined && setSeed(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="留出最后几集">
            <NumberInput
              value={holdout}
              onChange={(v) => v !== undefined && setHoldout(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="每集取几帧">
            <NumberInput
              value={framesPerEpisode}
              onChange={(v) => v !== undefined && setFramesPerEpisode(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="定位热区图">
            <NumberInput
              value={saliencyFrames}
              onChange={(v) => v !== undefined && setSaliencyFrames(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="指向反演帧数">
            <NumberInput
              value={aimFrames}
              onChange={(v) => v !== undefined && setAimFrames(v)}
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
            />
          </Field>
          <Field label="验证数据集">
            <Select
              value={valDataset || "__train__"}
              onValueChange={(v) => setValDataset(v === "__train__" ? "" : v)}
            >
              <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                <SelectValue placeholder="用训练数据集" />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white">
                <SelectItem value="__train__">用训练数据集</SelectItem>
                {datasets.map((d) => (
                  <SelectItem key={d.repo_id} value={d.repo_id}>
                    {d.repo_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="设备">
            <Select value={device} onValueChange={setDevice}>
              <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white">
                <SelectItem value="cuda">CUDA</SelectItem>
                <SelectItem value="cpu">CPU</SelectItem>
              </SelectContent>
            </Select>
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-3">
            <Switch id="eval_probe" checked={probe} onCheckedChange={setProbe} />
            <Label htmlFor="eval_probe" className="text-slate-300 text-sm">
              感知探针
            </Label>
          </div>
          {running ? (
            <Button
              onClick={stop}
              disabled={stopping}
              className="bg-red-500 hover:bg-red-600 text-white ml-auto"
            >
              {stopping ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Square className="w-4 h-4 mr-2" />
              )}
              {stopping ? "停止中…" : "停止评测"}
            </Button>
          ) : (
            <Button
              onClick={run}
              disabled={starting}
              className="bg-purple-500 hover:bg-purple-600 text-white ml-auto"
            >
              {starting ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <FlaskConical className="w-4 h-4 mr-2" />
              )}
              {results ? "重新评测" : "开始评测"}
            </Button>
          )}
        </div>

        {running ? (
          <div className="flex items-center gap-2 text-sm text-purple-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            正在评测…（加载模型约 1–2 分钟，之后每批 1–3 秒）
          </div>
        ) : state === "failed" ? (
          <div className="flex items-start gap-2 text-sm text-red-400">
            <ShieldAlert className="w-4 h-4 mt-0.5 shrink-0" />
            <span>评测失败{results?.error ? `：${results.error}` : ""}</span>
          </div>
        ) : results?.success ? (
          <div className="space-y-4">
            {/* 基础指标 */}
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {results.avg_loss != null && (
                <Metric label="损失" value={results.avg_loss.toFixed(4)} />
              )}
              {results.action_mse != null && (
                <Metric label="MSE" value={results.action_mse.toFixed(4)} />
              )}
              {results.action_mae != null && (
                <Metric label="MAE" value={`${results.action_mae.toFixed(4)}°`} />
              )}
              {results.avg_inference_time_ms != null && (
                <Metric label="推理" value={`${results.avg_inference_time_ms.toFixed(1)}ms`} />
              )}
              {results.batches_evaluated != null && (
                <Metric label="批次" value={String(results.batches_evaluated)} />
              )}
            </div>

            {results.probe ? (
              <div className="space-y-2">
                <div className="text-sm font-semibold text-slate-300">评估结论</div>
                <div
                  className={`rounded-lg border p-3 text-sm flex items-start gap-2 ${
                    vision != null && vision < 0.02
                      ? "border-red-700 bg-red-900/20 text-red-200"
                      : vision != null && vision < 0.1
                      ? "border-amber-700 bg-amber-900/20 text-amber-200"
                      : "border-green-700 bg-green-900/20 text-green-200"
                  }`}
                >
                  <ShieldCheck className="w-4 h-4 mt-0.5 shrink-0" />
                  <div className="font-semibold">{results.probe.verdict}</div>
                </div>
              </div>
            ) : null}

            {dimensions && (
              <div className="space-y-4">
                {/* 三维度色块 */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {(["semantic", "visual", "action"] as const).map((dim) => {
                    const d = dimensions[dim];
                    const dimLabels: Record<string, string> = {
                      semantic: "语义",
                      visual: "视觉",
                      action: "动作",
                    };
                    const isExpanded = expandedDim === dim;
                    return (
                      <div
                        key={dim}
                        className={`bg-slate-900/50 rounded-lg p-4 cursor-pointer transition-colors ${
                          isExpanded ? "bg-slate-900/80 ring-2 ring-purple-500/50" : "hover:bg-slate-900/80"
                        }`}
                        onClick={() => setExpandedDim(isExpanded ? null : dim)}
                      >
                        <div className="flex items-center justify-between mb-3">
                          <div className="text-sm font-medium text-slate-300">{dimLabels[dim]}</div>
                          <div className={`px-2 py-0.5 rounded text-xs font-semibold text-white ${GRADE_COLORS[d.grade]}`}>
                            {GRADE_LABELS[d.grade]}
                          </div>
                        </div>
                        <div className="text-3xl font-bold text-white mb-2">{d.score}</div>
                        <div className="text-xs text-slate-500 mb-3">综合评分</div>
                        {d.issues.length > 0 ? (
                          <div className="space-y-1">
                            <div className="text-xs text-slate-400 font-medium">问题：</div>
                            {d.issues.map((issue, i) => (
                              <div key={i} className="text-xs text-red-300 pl-2 border-l-2 border-red-500/50">
                                {issue}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-xs text-green-400">所有指标达标 ✓</div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* 展开的详情 */}
                {expandedDim && dimensions[expandedDim as keyof typeof dimensions] && (
                  <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
                    <DimensionDetail
                      dimension={expandedDim}
                      results={results}
                      onClose={() => setExpandedDim(null)}
                    />
                  </div>
                )}
              </div>
            )}

            {results.roi_grounding ? (
              <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3 text-sm text-slate-200">
                {results.roi_grounding.verdict}
                <div className="text-xs text-slate-500 mt-1">
                  判据：遮掉目标 vs 遮掉等面积随机区域的动作变化比值，≥
                  {results.roi_grounding.threshold} 才算用到目标区域。
                  {Object.entries(results.roi_grounding.by_camera)
                    .map(([cam, v]) => ` ${cam} ${v.median_ratio}×(n=${v.n})`)
                    .join(" · ")}
                </div>
              </div>
            ) : null}

            {results.aim ? (
              <div className="rounded-lg border border-slate-700 bg-slate-900/40 p-3 text-sm text-slate-200">
                {results.aim.verdict}
                <div className="text-xs text-slate-500 mt-1">
                  模型 {fmt(results.aim.mean_aim_error_model_px, 0, "px")} · 人 {fmt(results.aim.mean_aim_error_human_px, 0, "px")} · 标定 R² {fmt(results.aim.mean_calibration_r2, 2)}（n={results.aim.n_fit ?? "—"}）
                </div>
              </div>
            ) : null}

            {results.worst_episodes?.length ? (
              <div className="text-xs text-slate-400">
                误差最大的几集：
                {(results.per_episode ?? [])
                  .slice(0, 6)
                  .map((r) => `#${r.episode} ${r.mae_deg}°(${r.worst_joint})`)
                  .join("  ·  ")}
              </div>
            ) : null}

            {perJoint.length ? (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead className="text-slate-400">
                    <tr>
                      <th className="text-left py-1 pr-3">关节</th>
                      <th className="text-right py-1 pr-3">MAE(°)</th>
                      <th className="text-right py-1 pr-3">RMSE(°)</th>
                      <th className="text-right py-1 pr-3">基线(°)</th>
                      <th className="text-right py-1">相对基线</th>
                    </tr>
                  </thead>
                  <tbody className="text-slate-200">
                    {perJoint.map((m, i) => (
                      <tr key={i} className="border-t border-slate-700/60">
                        <td className="py-1 pr-3">{jointNames[i] ?? `joint_${i}`}</td>
                        <td className="py-1 pr-3 text-right tabular-nums">{m.toFixed(2)}</td>
                        <td className="py-1 pr-3 text-right tabular-nums">
                          {fmt(results.per_joint_rmse_deg?.[i], 2)}
                        </td>
                        <td className="py-1 pr-3 text-right tabular-nums">{fmt(naive[i], 2)}</td>
                        <td
                          className={`py-1 text-right tabular-nums ${
                            (relative[i] ?? 0) >= 1 ? "text-red-300" : "text-green-300"
                          }`}
                        >
                          {fmt(relative[i], 2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}

            {results.charts ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {Object.entries(results.charts).map(([key, rel]) => (
                  <div key={key} className="bg-slate-900/50 rounded-lg p-2">
                    <img
                      src={evaluationChartUrl(baseUrl, job.id, rel)}
                      alt={chartCaption(key)}
                      className="w-full rounded"
                      loading="lazy"
                    />
                    <div className="text-[11px] text-slate-500 mt-1">{CHART_CAPTIONS[key] ?? key}</div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default EvaluationCard;

// ---------------------------------------------------------------------------
// Dimension Detail Component
// ---------------------------------------------------------------------------

interface MetricDetail {
  name: string;
  value: string;
  threshold?: string;
  status: "pass" | "fail" | "na";
  description: string;
}

function getSemanticMetrics(results: ValidationResults): MetricDetail[] {
  const metrics: MetricDetail[] = [];

  const langSens = results.probe?.language_sensitivity;
  if (langSens != null) {
    metrics.push({
      name: "语言敏感度",
      value: langSens.toFixed(3),
      threshold: "> 0.1",
      status: langSens >= 0.1 ? "pass" : "fail",
      description: "换任务指令后动作变化量 ÷ 数据集动作散布",
    });
  }

  return metrics;
}

function getVisualMetrics(results: ValidationResults): MetricDetail[] {
  const metrics: MetricDetail[] = [];

  const peakToBall = results.localization?.mean_peak_to_ball_px;
  if (peakToBall != null) {
    metrics.push({
      name: "峰值离球距离",
      value: `${peakToBall.toFixed(0)}px`,
      threshold: "< 120px",
      status: peakToBall <= 120 ? "pass" : "fail",
      description: "热区图峰值到球真实位置的距离",
    });
  }

  const saliencyOnBall = results.localization?.mean_saliency_on_ball;
  if (saliencyOnBall != null) {
    const chanceLevel = results.localization?.chance_level ?? 0.25;
    metrics.push({
      name: "球上显著性",
      value: `${(saliencyOnBall * 100).toFixed(0)}%`,
      threshold: `> ${(chanceLevel * 100).toFixed(0)}%（随机）`,
      status: saliencyOnBall > chanceLevel ? "pass" : "fail",
      description: "热区图落在球上的概率",
    });
  }

  const roi = results.roi_grounding;
  if (roi) {
    const ratios = Object.entries(roi.by_camera).map(([cam, v]) => ({
      cam,
      ratio: v.median_ratio,
      n: v.n,
    }));
    for (const { cam, ratio, n } of ratios) {
      metrics.push({
        name: `ROI 利用率 (${cam})`,
        value: `${ratio.toFixed(2)}×`,
        threshold: `≥ ${roi.threshold}×`,
        status: ratio >= roi.threshold ? "pass" : "fail",
        description: `遮掉目标 vs 随机区域的动作变化比值 (n=${n})`,
      });
    }
  }

  const aimError = results.aim?.mean_aim_error_model_px;
  if (aimError != null) {
    metrics.push({
      name: "指向反演误差（模型）",
      value: `${aimError.toFixed(0)}px`,
      threshold: "< 100px",
      status: aimError <= 100 ? "pass" : "fail",
      description: "从模型动作反推的指向与球真实位置的偏差",
    });
  }

  const aimHumanError = results.aim?.mean_aim_error_human_px;
  if (aimHumanError != null) {
    metrics.push({
      name: "指向反演误差（人）",
      value: `${aimHumanError.toFixed(0)}px`,
      threshold: "参考",
      status: "na",
      description: "从录制动作反推的指向与球真实位置的偏差",
    });
  }

  const visSens = results.probe?.vision_sensitivity;
  if (visSens != null) {
    metrics.push({
      name: "视觉敏感度",
      value: visSens.toFixed(3),
      threshold: "> 0.1",
      status: visSens >= 0.1 ? "pass" : "fail",
      description: "图像扰动后动作变化量 ÷ 数据集动作散布",
    });
  }

  return metrics;
}

function getActionMetrics(results: ValidationResults): MetricDetail[] {
  const metrics: MetricDetail[] = [];

  const relNaive = results.mean_mae_vs_naive;
  if (relNaive != null) {
    metrics.push({
      name: "相对基线",
      value: `${relNaive.toFixed(2)}×`,
      threshold: "< 1",
      status: relNaive < 1 ? "pass" : "fail",
      description: "模型误差 ÷ 预测均值误差（<1 才比「瞎猜」强）",
    });
  }

  const mae = results.action_mae;
  if (mae != null) {
    metrics.push({
      name: "平均动作误差",
      value: `${mae.toFixed(2)}°`,
      threshold: "< 10°",
      status: mae <= 10 ? "pass" : "fail",
      description: "预测动作与录制动作的平均偏差",
    });
  }

  const driftSteps = results.drift_horizon_steps;
  if (driftSteps != null) {
    metrics.push({
      name: "安全开环步数",
      value: `${driftSteps} 步`,
      threshold: "> 5 步",
      status: driftSteps > 5 ? "pass" : "fail",
      description: "不重新推理能稳定执行多少步",
    });
  }

  const replanHz = results.recommended_replan_hz;
  if (replanHz != null) {
    metrics.push({
      name: "建议重推理频率",
      value: `${replanHz.toFixed(1)}Hz`,
      status: "na",
      description: "基于安全步数推算的最低推理频率",
    });
  }

  const shiftAgreement = results.probe?.shift_sign_agreement;
  if (shiftAgreement != null) {
    metrics.push({
      name: "方向一致性",
      value: `${(shiftAgreement * 100).toFixed(0)}%`,
      threshold: "≥ 60%",
      status: shiftAgreement >= 0.6 ? "pass" : "fail",
      description: "图像左移/右移时动作是否朝相反方向改",
    });
  }

  const errorFloor = results.error_floor_deg;
  if (errorFloor != null) {
    metrics.push({
      name: "数据一致性下限",
      value: `${errorFloor.toFixed(2)}°`,
      status: "na",
      description: "几乎相同画面下录制动作本身的分歧 ÷2",
    });
  }

  return metrics;
}

function getMetricsForDimension(dimension: string, results: ValidationResults): MetricDetail[] {
  switch (dimension) {
    case "semantic":
      return getSemanticMetrics(results);
    case "visual":
      return getVisualMetrics(results);
    case "action":
      return getActionMetrics(results);
    default:
      return [];
  }
}

const DIMENSION_LABELS: Record<string, string> = {
  semantic: "语义",
  visual: "视觉",
  action: "动作",
};

const DIMENSION_DESCRIPTIONS: Record<string, string> = {
  semantic: "评估模型对任务指令的理解能力",
  visual: "评估模型的视觉感知能力，包括目标定位、ROI 利用和指向反演",
  action: "评估模型的动作预测质量，包括误差、基线对比和安全步数",
};

const DimensionDetail: React.FC<{
  dimension: string;
  results: ValidationResults;
  onClose: () => void;
}> = ({ dimension, results, onClose }) => {
  const metrics = getMetricsForDimension(dimension, results);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold text-white">{DIMENSION_LABELS[dimension]}维度详情</div>
          <div className="text-xs text-slate-400 mt-1">{DIMENSION_DESCRIPTIONS[dimension]}</div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white text-sm px-3 py-1 rounded hover:bg-slate-800"
        >
          收起
        </button>
      </div>

      {metrics.length === 0 ? (
        <div className="text-slate-400 text-sm">该维度暂无可用指标</div>
      ) : (
        <div className="space-y-3">
          {metrics.map((m, i) => (
            <div
              key={i}
              className={`p-4 rounded-lg border ${
                m.status === "fail"
                  ? "bg-red-900/20 border-red-700/50"
                  : m.status === "pass"
                  ? "bg-green-900/20 border-green-700/50"
                  : "bg-slate-800/50 border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="font-medium text-white">{m.name}</div>
                <div className="flex items-center gap-2">
                  {m.threshold && m.status !== "na" && (
                    <span className="text-xs text-slate-400">阈值：{m.threshold}</span>
                  )}
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      m.status === "fail"
                        ? "bg-red-500/20 text-red-300"
                        : m.status === "pass"
                        ? "bg-green-500/20 text-green-300"
                        : "bg-slate-700 text-slate-400"
                    }`}
                  >
                    {m.status === "fail" ? "不达标" : m.status === "pass" ? "达标" : "参考"}
                  </span>
                </div>
              </div>
              <div className="text-2xl font-bold text-white mb-1">{m.value}</div>
              <div className="text-xs text-slate-400">{m.description}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
