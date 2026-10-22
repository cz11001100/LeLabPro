import { useNavigate, useLocation } from "react-router-dom";
import { useActiveTasks } from "@/hooks/useActiveTasks";
import { X, Loader2, Radio } from "lucide-react";
import { useState } from "react";

export default function ActiveTasksBanner() {
  const { recordingTask, sshInstallTask, hasActiveRecording, hasActiveSshInstall } = useActiveTasks();
  const navigate = useNavigate();
  const location = useLocation();
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  if ((!hasActiveRecording && !hasActiveSshInstall) || dismissed.size >= 2) {
    return null;
  }

  const items: { id: string; label: string; path: string; icon: React.ReactNode }[] = [];

  if (hasActiveRecording && recordingTask && !dismissed.has("recording")) {
    const onRecordingPage = location.pathname === "/recording";
    items.push({
      id: "recording",
      label: recordingTask.label || "Recording in progress",
      path: "/recording",
      icon: <Radio className="w-3.5 h-3.5 text-red-400 animate-pulse" />,
    });
    if (onRecordingPage) {
      // Don't show banner when already on the page
      items.pop();
    }
  }

  if (hasActiveSshInstall && sshInstallTask && !dismissed.has("ssh_install")) {
    const onTrainingPage = location.pathname === "/training";
    items.push({
      id: "ssh_install",
      label: sshInstallTask.label || "Installing lerobot...",
      path: "/training",
      icon: <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />,
    });
    if (onTrainingPage) {
      items.pop();
    }
  }

  if (items.length === 0) return null;

  return (
    <div className="fixed top-0 left-0 right-0 z-50 bg-slate-900/95 border-b border-slate-700 px-4 py-2 flex items-center gap-3 backdrop-blur-sm">
      <span className="text-xs text-slate-400 shrink-0">Background tasks:</span>
      <div className="flex items-center gap-3 flex-1 overflow-x-auto">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => navigate(item.path)}
            className="flex items-center gap-1.5 text-xs text-slate-200 hover:text-white bg-slate-800 hover:bg-slate-700 rounded px-2 py-1 transition-colors shrink-0"
          >
            {item.icon}
            <span className="truncate max-w-[200px]">{item.label}</span>
            <span className="text-slate-500 text-[10px]">Go to →</span>
          </button>
        ))}
      </div>
      <button
        onClick={() => {
          const allIds = items.map((i) => i.id);
          setDismissed((prev) => new Set([...prev, ...allIds]));
        }}
        className="text-slate-500 hover:text-white shrink-0"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
