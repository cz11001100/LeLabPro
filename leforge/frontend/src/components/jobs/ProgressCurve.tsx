import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Fetcher } from "@/lib/apiClient";
import { RolloutEpisodeSeries, getRolloutEpisodeSeries } from "@/lib/rolloutEvalApi";

const W = 620;
const H = 150;
const PAD = { left: 30, right: 10, top: 10, bottom: 18 };

/**
 * One episode's progress curve. The number table says whether the scorer judged
 * an episode a success; only the shape shows whether it can be trusted — a curve
 * that never dips when the ball drops out of the gripper is the failure mode that
 * makes a zero-shot scorer useless. Human-driven windows are shaded so the two
 * stories can be read against each other.
 */
const ProgressCurve: React.FC<{
  baseUrl: string;
  fetchWithHeaders: Fetcher;
  repoId: string;
  episodeIndex: number;
  headMode?: string;
  threshold?: number;
}> = ({ baseUrl, fetchWithHeaders, repoId, episodeIndex, headMode = "sparse", threshold = 0.9 }) => {
  const [series, setSeries] = useState<RolloutEpisodeSeries | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!repoId) return;
    let cancelled = false;
    setLoading(true);
    getRolloutEpisodeSeries(baseUrl, fetchWithHeaders, repoId, episodeIndex, headMode)
      .then((res) => {
        if (!cancelled) setSeries(res);
      })
      .catch(() => {
        if (!cancelled) setSeries({ available: false, error: "读进度曲线失败" });
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, repoId, episodeIndex, headMode]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-slate-600 py-6">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        读进度曲线…
      </div>
    );
  }
  if (!series?.available || !series.points?.length) {
    return (
      <p className="text-xs text-slate-600 py-4">{series?.error || series?.hint || "没有可画的进度"}</p>
    );
  }

  const length = Math.max(1, series.length ?? 1);
  const fps = series.fps || 30;
  const x = (frame: number) =>
    PAD.left + (frame / length) * (W - PAD.left - PAD.right);
  const y = (progress: number) => PAD.top + (1 - progress) * (H - PAD.top - PAD.bottom);
  const path = series.points
    .filter((p): p is [number, number] => p[1] != null)
    .map(([f, p], i) => `${i === 0 ? "M" : "L"}${x(f).toFixed(1)},${y(p).toFixed(1)}`)
    .join(" ");

  return (
    <div className="space-y-1">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto bg-sky-100/80 rounded">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(g)}
              y2={y(g)}
              stroke="#bae6fd"
              strokeWidth={g === 0 ? 1 : 0.5}
            />
            <text x={4} y={y(g) + 3} fill="#475569" fontSize={9}>
              {g.toFixed(2)}
            </text>
          </g>
        ))}

        {(series.correction_windows ?? []).map((w, i) => (
          <rect
            key={i}
            x={x(w.start)}
            y={PAD.top}
            width={Math.max(1, x(w.end) - x(w.start))}
            height={H - PAD.top - PAD.bottom}
            fill="#f59e0b"
            opacity={0.14}
          />
        ))}

        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={y(threshold)}
          y2={y(threshold)}
          stroke="#22c55e"
          strokeDasharray="4 3"
          strokeWidth={1}
        />
        <text x={W - PAD.right - 2} y={y(threshold) - 3} fill="#22c55e" fontSize={9} textAnchor="end">
          成功阈值 {threshold}
        </text>

        <path d={path} fill="none" stroke="#38bdf8" strokeWidth={1.6} />

        <text x={PAD.left} y={H - 5} fill="#475569" fontSize={9}>
          0s
        </text>
        <text x={W - PAD.right} y={H - 5} fill="#475569" fontSize={9} textAnchor="end">
          {(length / fps).toFixed(0)}s · {length} 帧
        </text>
      </svg>
      <p className="text-xs text-slate-600">
        进度 · 琥珀底=人工纠正段（{series.correction_windows?.length ?? 0} 段）· 绿虚线=成功阈值
      </p>
    </div>
  );
};

export default ProgressCurve;
