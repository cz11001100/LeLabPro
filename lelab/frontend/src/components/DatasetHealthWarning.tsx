import React from "react";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  Wrench,
  Trash2,
  Loader2,
  CheckCircle,
  FileWarning,
  Film,
} from "lucide-react";
import type { DatasetHealth, RepairProgress } from "@/lib/replayApi";

interface DatasetHealthWarningProps {
  health: DatasetHealth;
  isRepairing: boolean;
  repairProgress: RepairProgress | null;
  onRepair: () => void;
  onDelete: () => void;
}

const DatasetHealthWarning: React.FC<DatasetHealthWarningProps> = ({
  health,
  isRepairing,
  repairProgress,
  onRepair,
  onDelete,
}) => (
  <div className="bg-red-900/20 border border-red-600/50 rounded-lg p-5 mb-8">
    <div className="flex items-center gap-3 mb-4">
      <AlertTriangle className="w-6 h-6 text-red-400 shrink-0" />
      <div>
        <h3 className="text-lg font-semibold text-red-300">Dataset Corruption Detected</h3>
        <p className="text-sm text-red-400/80">Some files are unreadable. The dataset may not load correctly.</p>
      </div>
    </div>
    <div className="space-y-2 mb-4 text-sm">
      {!health.episodes_healthy && (
        <div className="flex items-center gap-2 text-red-300">
          <FileWarning className="w-4 h-4 shrink-0" />
          <span>Episode index corrupted</span>
        </div>
      )}
      {health.corrupted_data_files.length > 0 && (
        <div className="flex items-center gap-2 text-red-300">
          <FileWarning className="w-4 h-4 shrink-0" />
          <span>{health.corrupted_data_files.length} / {health.total_data_files} data files corrupted</span>
        </div>
      )}
      {health.corrupted_video_files.length > 0 && (
        <div className="flex items-center gap-2 text-red-300">
          <Film className="w-4 h-4 shrink-0" />
          <span>{health.corrupted_video_files.length} / {health.total_video_files} video files corrupted</span>
        </div>
      )}
      {health.corrupted_episodes > 0 && (
        <div className="flex items-center gap-2 text-amber-300">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>~{health.corrupted_episodes} / {health.total_episodes} episodes affected</span>
        </div>
      )}
    </div>
    {isRepairing && repairProgress && (
      <div className="mb-4 p-3 bg-gray-900/60 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
            <span className="text-sm text-gray-300">{repairProgress.phase}</span>
          </div>
          <span className="text-sm text-blue-400 font-mono">{repairProgress.percent}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
          <div className="bg-blue-500 h-2.5 rounded-full transition-all duration-500 ease-out" style={{ width: `${repairProgress.percent}%` }} />
        </div>
        {repairProgress.total > 0 && (
          <p className="text-xs text-gray-500 mt-1">{repairProgress.current} / {repairProgress.total}</p>
        )}
      </div>
    )}
    {repairProgress?.completed && !repairProgress.error && (
      <div className="mb-4 flex items-center gap-2 text-green-400 text-sm">
        <CheckCircle className="w-4 h-4" />
        <span>Repair complete!</span>
      </div>
    )}
    {repairProgress?.error && (
      <div className="mb-4 p-3 bg-red-900/30 rounded-lg border border-red-700 text-sm text-red-300">
        Repair failed: {repairProgress.error}
      </div>
    )}
    <div className="flex flex-wrap gap-3">
      <Button onClick={onRepair} disabled={isRepairing} className="bg-blue-600 hover:bg-blue-700 text-white">
        {isRepairing ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin" />Repairing...</>) : (<><Wrench className="w-4 h-4 mr-2" />Repair Dataset</>)}
      </Button>
      <Button onClick={onDelete} disabled={isRepairing} variant="outline" className="border-red-500/40 text-red-400 hover:border-red-400 hover:text-red-300 hover:bg-red-500/10">
        <Trash2 className="w-4 h-4 mr-2" />Delete Dataset
      </Button>
    </div>
    <p className="text-xs text-gray-500 mt-3">Repair recovers readable episodes. Corrupted files will be removed.</p>
  </div>
);

export default DatasetHealthWarning;
