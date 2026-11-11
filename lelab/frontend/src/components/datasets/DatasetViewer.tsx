import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2 } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useApi } from "@/contexts/ApiContext";
import {
  datasetVideoUrl,
  getDatasetEpisodes,
  getDatasetEpisodeSeries,
  getDatasetVideoWindow,
  DatasetEpisodesInfo,
  EpisodeSeries,
} from "@/lib/replayApi";

const JOINT_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#facc15", "#4ade80", "#fb923c"];

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  repoId: string;
}

const DatasetViewer: React.FC<Props> = ({ open, onOpenChange, repoId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [info, setInfo] = useState<DatasetEpisodesInfo | null>(null);
  const [episode, setEpisode] = useState<number | null>(null);
  const [camera, setCamera] = useState<string>("");
  const [series, setSeries] = useState<EpisodeSeries | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const windowRef = useRef<{ from: number; to: number }>({ from: 0, to: 0 });

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    setInfo(null);
    setSeries(null);
    getDatasetEpisodes(baseUrl, fetchWithHeaders, repoId, controller.signal)
      .then((r) => {
        setInfo(r);
        if (r.episodes.length) setEpisode(r.episodes[0].episode_index);
        if (r.cameras.length) setCamera(r.cameras[0]);
      })
      .catch((e) => !controller.signal.aborted && setError(String(e?.message ?? e)))
      .finally(() => !controller.signal.aborted && setLoading(false));
    return () => controller.abort();
  }, [open, repoId, baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!open || episode == null) return;
    const controller = new AbortController();
    getDatasetEpisodeSeries(baseUrl, fetchWithHeaders, repoId, episode, controller.signal)
      .then(setSeries)
      .catch(() => {});
    return () => controller.abort();
  }, [open, repoId, episode, baseUrl, fetchWithHeaders]);

  // v3 packs many episodes into one mp4: seek to the episode's window on
  // metadata load, and loop back to the window start when it ends.
  useEffect(() => {
    if (!open || episode == null || !camera) return;
    const controller = new AbortController();
    getDatasetVideoWindow(baseUrl, fetchWithHeaders, repoId, episode, camera, controller.signal)
      .then((w) => {
        windowRef.current = { from: w.from_timestamp, to: w.to_timestamp };
        const v = videoRef.current;
        if (v && w.to_timestamp > w.from_timestamp) {
          const seek = () => {
            v.currentTime = w.from_timestamp;
            v.play().catch(() => {});
          };
          if (v.readyState >= 1) seek();
          else v.addEventListener("loadedmetadata", seek, { once: true });
        }
      })
      .catch(() => {});
    return () => controller.abort();
  }, [open, repoId, episode, camera, baseUrl, fetchWithHeaders]);

  const onTimeUpdate = () => {
    const v = videoRef.current;
    const { from, to } = windowRef.current;
    if (v && to > from && v.currentTime >= to) {
      v.currentTime = from;
    }
  };

  const chartData = useMemo(() => {
    if (!series) return [];
    return series.frames.map((f, i) => {
      const row: Record<string, number> = { frame: f };
      series.joint_names.forEach((name, j) => {
        row[`a_${name}`] = series.action[i]?.[j] ?? 0;
        if (series.state.length) row[`s_${name}`] = series.state[i]?.[j] ?? 0;
      });
      return row;
    });
  }, [series]);

  const shortCam = (c: string) => c.replace("observation.images.", "");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[960px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-white text-center text-xl font-bold">
            数据集可视化 · {repoId}
          </DialogTitle>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center gap-2 py-10 text-gray-400">
            <Loader2 className="w-5 h-5 animate-spin" /> 加载中…
          </div>
        )}
        {error && <div className="text-red-300 text-sm py-4">{error}</div>}

        {info && episode != null && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <Select
                value={String(episode)}
                onValueChange={(v) => setEpisode(Number(v))}
              >
                <SelectTrigger className="w-[220px] bg-gray-800 border-gray-700 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-800 border-gray-700 text-white max-h-72">
                  {info.episodes.map((e) => (
                    <SelectItem key={e.episode_index} value={String(e.episode_index)}>
                      第 {e.episode_index} 集 · {e.length} 帧
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <div className="flex gap-2">
                {info.cameras.map((c) => (
                  <Button
                    key={c}
                    size="sm"
                    variant={camera === c ? "default" : "outline"}
                    onClick={() => setCamera(c)}
                    className={
                      camera === c
                        ? "bg-cyan-600 hover:bg-cyan-700 text-white"
                        : "border-gray-600 text-gray-300 hover:bg-gray-800"
                    }
                  >
                    {shortCam(c)}
                  </Button>
                ))}
              </div>
              {info.episodes.find((e) => e.episode_index === episode)?.tasks[0] && (
                <span className="text-xs text-gray-400 truncate">
                  {info.episodes.find((e) => e.episode_index === episode)?.tasks[0]}
                </span>
              )}
            </div>

            {camera && (
              <video
                ref={videoRef}
                key={`${episode}-${camera}`}
                controls
                muted
                playsInline
                preload="auto"
                src={datasetVideoUrl(baseUrl, repoId, episode, camera)}
                onTimeUpdate={onTimeUpdate}
                className="w-full max-h-[420px] rounded-lg border border-gray-700 bg-black"
              />
            )}

            {series && chartData.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm text-gray-300">关节曲线（实线=动作，虚线=状态）</div>
                <div className="bg-gray-950/60 border border-gray-700 rounded-lg p-3">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                      <XAxis dataKey="frame" stroke="#64748b" fontSize={11} />
                      <YAxis stroke="#64748b" fontSize={11} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          fontSize: 12,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {series.joint_names.map((name, j) => (
                        <Line
                          key={`a_${name}`}
                          type="monotone"
                          dataKey={`a_${name}`}
                          name={name}
                          stroke={JOINT_COLORS[j % JOINT_COLORS.length]}
                          dot={false}
                          strokeWidth={1.5}
                        />
                      ))}
                      {series.state.length > 0 &&
                        series.joint_names.map((name, j) => (
                          <Line
                            key={`s_${name}`}
                            type="monotone"
                            dataKey={`s_${name}`}
                            name={`${name} (state)`}
                            stroke={JOINT_COLORS[j % JOINT_COLORS.length]}
                            strokeDasharray="4 3"
                            dot={false}
                            strokeWidth={1}
                            legendType="none"
                          />
                        ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default DatasetViewer;