import React, { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, TrendingUp, Activity, RefreshCw, Wrench, Loader2 } from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import {
  getDatasetStatistics,
  startDatasetRepair,
  getRepairProgress,
  DatasetAnalysis,
  DatasetAnomaly,
  RepairProgress,
} from "@/lib/replayApi";

interface DatasetSummaryCardProps {
  datasetRepoId: string;
}

// Anomalies that a forced episode-table rebuild can fix.
const REBUILDABLE = new Set([
  "index_range_mismatch",
  "duplicate_episode_meta",
  "frame_count_mismatch",
]);

const DatasetSummaryCard: React.FC<DatasetSummaryCardProps> = ({ datasetRepoId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [repairing, setRepairing] = useState(false);
  const [repairProgress, setRepairProgress] = useState<RepairProgress | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStats = useCallback(async () => {
    if (!datasetRepoId) return;
    setLoading(true);
    setError(null);
    try {
      const result = await getDatasetStatistics(baseUrl, fetchWithHeaders, datasetRepoId);
      setAnalysis(result);
    } catch (e) {
      // A transient failure (e.g. the backend just restarted) shouldn't leave the
      // card stuck on "Failed" forever — surface the reason and offer a retry.
      setAnalysis(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetRepoId, baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const handleRebuild = useCallback(async () => {
    setRepairing(true);
    setRepairProgress(null);
    try {
      await startDatasetRepair(baseUrl, fetchWithHeaders, datasetRepoId, true);
    } catch (e) {
      setRepairing(false);
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    let polls = 0;
    pollRef.current = setInterval(async () => {
      polls += 1;
      try {
        const progress = await getRepairProgress(baseUrl, fetchWithHeaders);
        const isOurs = !!progress && progress.dataset_repo_id === datasetRepoId;
        if (!isOurs) {
          // Repair thread hasn't registered (or never started). Give it 10s.
          if (polls >= 5) {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setRepairing(false);
          }
          return;
        }
        if (!progress.active) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setRepairing(false);
          setRepairProgress(progress);
          fetchStats();
        } else {
          setRepairProgress(progress);
        }
      } catch {
        // transient poll failure — keep polling
      }
    }, 2000);
  }, [baseUrl, fetchWithHeaders, datasetRepoId, fetchStats]);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex items-center gap-2 text-gray-400">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-gray-600 border-t-blue-500" />
          Analyzing dataset...
        </div>
      </div>
    );
  }

  if (!analysis || !analysis.success || !analysis.statistics) {
    return (
      <div className="bg-gray-800 rounded-lg border border-red-900/50 p-4">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-red-400 min-w-0">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span className="truncate">
              Failed to analyze dataset{error ? `: ${error}` : ""}
            </span>
          </div>
          <button
            onClick={fetchStats}
            className="flex items-center gap-1 shrink-0 rounded border border-gray-600 px-3 py-1 text-sm text-gray-200 hover:bg-gray-700"
          >
            <RefreshCw className="h-4 w-4" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const { statistics, anomalies = [] } = analysis;
  const highSeverityCount = anomalies.filter((a) => a.severity === "high").length;
  const mediumSeverityCount = anomalies.filter((a) => a.severity === "medium").length;

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-lg flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-purple-400" />
          Dataset Quality Analysis
        </h3>
        {anomalies.length === 0 ? (
          <span className="text-xs text-green-400 flex items-center gap-1">
            <CheckCircle2 className="h-4 w-4" />
            No issues detected
          </span>
        ) : (
          <span className="text-xs text-amber-400 flex items-center gap-1">
            <AlertCircle className="h-4 w-4" />
            {anomalies.length} issue{anomalies.length > 1 ? "s" : ""} found
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Episodes"
          value={statistics.episode_count.toString()}
          icon={<Activity className="h-4 w-4" />}
        />
        <StatCard
          label="Total Frames"
          value={statistics.total_frames.toLocaleString()}
          icon={<Activity className="h-4 w-4" />}
        />
        <StatCard
          label="Avg Length"
          value={`${Math.round(statistics.episode_length.mean)} frames`}
          subtitle={`±${Math.round(statistics.episode_length.std)}`}
        />
        <StatCard
          label="Length Range"
          value={`${statistics.episode_length.min} - ${statistics.episode_length.max}`}
        />
      </div>

      {Object.keys(statistics.action_range).length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-2">Action Range (Joints)</h4>
          <div className="space-y-1">
            {Object.entries(statistics.action_range).map(([joint, range]) => (
              <div key={joint} className="flex items-center justify-between text-xs">
                <span className="text-gray-400">{joint}</span>
                <span className="text-gray-300 font-mono">
                  [{range.min.toFixed(2)}, {range.max.toFixed(2)}]
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {anomalies.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-300 mb-2">
            Detected Issues ({highSeverityCount} high, {mediumSeverityCount} medium)
          </h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {anomalies.map((anomaly, idx) => (
              <AnomalyItem key={idx} anomaly={anomaly} />
            ))}
          </div>
          {anomalies.some((a) => REBUILDABLE.has(a.type)) && (
            <div className="mt-3 space-y-2">
              <button
                onClick={handleRebuild}
                disabled={repairing}
                className="flex items-center gap-2 rounded border border-blue-700 bg-blue-600/20 px-3 py-1.5 text-sm text-blue-300 hover:bg-blue-600/30 disabled:opacity-50"
              >
                {repairing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Wrench className="h-4 w-4" />
                )}
                {repairing ? "Rebuilding..." : "Rebuild episode table"}
              </button>
              {repairing && repairProgress && (
                <div className="text-xs text-gray-400">
                  <div className="flex items-center justify-between mb-1">
                    <span>{repairProgress.phase}</span>
                    <span className="font-mono text-blue-400">{repairProgress.percent}%</span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-700">
                    <div
                      className="h-1.5 rounded-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${repairProgress.percent}%` }}
                    />
                  </div>
                </div>
              )}
              {!repairing && repairProgress?.error && (
                <div className="text-xs text-red-400">Repair failed: {repairProgress.error}</div>
              )}
              {!repairing && repairProgress?.completed && !repairProgress.error && (
                <div className="flex items-center gap-1 text-xs text-green-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Rebuild complete — analysis refreshed.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const StatCard: React.FC<{
  label: string;
  value: string;
  subtitle?: string;
  icon?: React.ReactNode;
}> = ({ label, value, subtitle, icon }) => (
  <div className="bg-gray-900/50 rounded p-3">
    <div className="flex items-center gap-1 text-xs text-gray-400 mb-1">
      {icon}
      {label}
    </div>
    <div className="text-lg font-semibold text-white">{value}</div>
    {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
  </div>
);

const AnomalyItem: React.FC<{ anomaly: DatasetAnomaly }> = ({ anomaly }) => {
  const severityColors = {
    high: "border-red-900/50 bg-red-950/20",
    medium: "border-amber-900/50 bg-amber-950/20",
    low: "border-gray-700 bg-gray-900/20",
  };

  const severityTextColors = {
    high: "text-red-400",
    medium: "text-amber-400",
    low: "text-gray-400",
  };

  return (
    <div className={`border rounded p-2 text-xs ${severityColors[anomaly.severity]}`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`font-medium ${severityTextColors[anomaly.severity]}`}>
          Episode {anomaly.episode}
        </span>
        <span className="text-gray-500">{anomaly.type}</span>
      </div>
      <div className="text-gray-300">{anomaly.message}</div>
    </div>
  );
};

export default DatasetSummaryCard;
