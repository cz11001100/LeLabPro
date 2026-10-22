import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useApi } from "@/contexts/ApiContext";
import {
  JobRecord,
  ValidationResults,
  computeDimensionScores,
  GRADE_COLORS,
  GRADE_LABELS,
  DimensionGrade,
} from "@/lib/jobsApi";
import { getJob } from "@/lib/jobsApi";

const DIMENSION_LABELS: Record<string, string> = {
  semantic: "语义",
  visual: "视觉",
  action: "动作",
};

const DIMENSION_DESCRIPTIONS: Record<string, string> = {
  semantic: "评估模型对任务指令的理解能力，包括语言敏感度和空间关系理解",
  visual: "评估模型的视觉感知能力，包括目标定位、ROI 利用和指向反演",
  action: "评估模型的动作预测质量，包括误差、基线对比和安全步数",
};

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
      description: "换任务指令后动作变化量  数据集动作散布",
    });
  }

  const shiftAgreement = results.probe?.shift_sign_agreement;
  if (shiftAgreement != null) {
    metrics.push({
      name: "左右移方向一致性",
      value: `${(shiftAgreement * 100).toFixed(0)}%`,
      threshold: "≥ 60%",
      status: shiftAgreement >= 0.6 ? "pass" : "fail",
      description: "图像左移/右移时动作是否朝相反方向改",
    });
  }

  const visSens = results.probe?.vision_sensitivity;
  if (visSens != null) {
    metrics.push({
      name: "视觉敏感度（语义代理）",
      value: visSens.toFixed(3),
      threshold: "> 0.1",
      status: visSens >= 0.1 ? "pass" : "fail",
      description: "扰动输入后动作变化量 ÷ 数据集动作散布",
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
      description: "模型误差  预测均值误差（<1 才比"瞎猜"强）",
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

const EvaluationDimensionPage: React.FC = () => {
  const { dimension, jobId } = useParams<{ dimension: string; jobId: string }>();
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!jobId) return;
    const controller = new AbortController();
    getJob(baseUrl, fetchWithHeaders, jobId, controller.signal)
      .then((j) => {
        setJob(j);
        setLoading(false);
      })
      .catch(() => {
        setLoading(false);
      });
    return () => controller.abort();
  }, [baseUrl, fetchWithHeaders, jobId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin mr-3" /> 加载中…
      </div>
    );
  }

  if (!job || !job.validation_results?.success) {
    return (
      <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
        <div className="flex items-center gap-4 mb-8">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(-1)}
            className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1 className="font-bold text-white text-2xl">评估详情</h1>
        </div>
        <div className="text-center text-slate-400">无有效评估数据</div>
      </div>
    );
  }

  const results = job.validation_results;
  const dimensions = computeDimensionScores(results);
  const dimKey = dimension as keyof typeof dimensions;
  const dimData = dimensions[dimKey];
  const metrics = getMetricsForDimension(dimension ?? "", results);

  if (!dimData) {
    return (
      <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
        <div className="flex items-center gap-4 mb-8">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate(-1)}
            className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <h1 className="font-bold text-white text-2xl">评估详情</h1>
        </div>
        <div className="text-center text-slate-400">未知维度</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
      <div className="flex items-center gap-4 mb-8">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate(-1)}
          className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <div>
          <h1 className="font-bold text-white text-2xl">
            {DIMENSION_LABELS[dimension ?? ""] ?? dimension}评估详情
          </h1>
          <p className="text-sm text-slate-400 mt-1">{DIMENSION_DESCRIPTIONS[dimension ?? ""]}</p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto w-full space-y-6">
        {/* 综合评分卡片 */}
        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
          <div className="flex items-center justify-between mb-4">
            <div className="text-lg font-semibold text-white">综合评分</div>
            <div className={`px-3 py-1 rounded-lg text-sm font-bold text-white ${GRADE_COLORS[dimData.grade]}`}>
              {GRADE_LABELS[dimData.grade]}
            </div>
          </div>
          <div className="text-5xl font-bold text-white mb-2">{dimData.score}</div>
          <div className="text-sm text-slate-400">满分 100 分</div>
        </div>

        {/* 指标详情 */}
        <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-700">
          <div className="text-lg font-semibold text-white mb-4">指标详情</div>
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

        {/* 问题总结 */}
        {dimData.issues.length > 0 && (
          <div className="bg-red-900/20 rounded-xl p-6 border border-red-700/50">
            <div className="text-lg font-semibold text-red-300 mb-3">存在的问题</div>
            <ul className="space-y-2">
              {dimData.issues.map((issue, i) => (
                <li key={i} className="text-sm text-red-200 flex items-start gap-2">
                  <span className="text-red-400 mt-0.5">•</span>
                  <span>{issue}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default EvaluationDimensionPage;
