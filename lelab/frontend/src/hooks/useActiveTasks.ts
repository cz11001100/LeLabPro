import { useEffect, useState, useCallback } from "react";
import { useApi } from "@/contexts/ApiContext";
import { getActiveTasks, ActiveTask } from "@/lib/jobsApi";
import { CaptureMode, InferenceStatus } from "@/lib/inferenceApi";

const CAPTURE_LABELS: Record<CaptureMode, string> = {
  base: "推理",
  dagger: "纠正采集",
  highlight: "抢救采集",
  sentry: "自动采集",
};

export function useActiveTasks(pollInterval: number = 3000) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [tasks, setTasks] = useState<ActiveTask[]>([]);
  const [loading, setLoading] = useState(false);
  // Rollout/inference sessions live outside the task tracker (they are
  // subprocesses the deploy page owns), so poll their status here too — that is
  // what lets the home page offer a way back into a running capture session.
  const [inference, setInference] = useState<InferenceStatus | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const active = await getActiveTasks(baseUrl, fetchWithHeaders);
      setTasks(active);
    } catch {
      // ignore
    }
    try {
      const response = await fetchWithHeaders(`${baseUrl}/inference-status`);
      if (!response.ok) throw new Error("status unavailable");
      const status: InferenceStatus = await response.json();
      setInference(status.inference_active ? status : null);
    } catch {
      setInference(null);
    } finally {
      setLoading(false);
    }
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, pollInterval);
    return () => clearInterval(interval);
  }, [refresh, pollInterval]);

  const recordingTask = tasks.find((t) => t.task_type === "recording" && t.status === "running");
  const sshInstallTask = tasks.find((t) => t.task_type === "ssh_install" && t.status === "running");
  const trainingTask = tasks.find((t) => t.task_type === "training" && t.status === "running");

  const target = inference?.episode_target ?? 0;
  const saved = inference?.takeover?.episodes ?? 0;
  const modeLabel = CAPTURE_LABELS[inference?.capture_mode ?? "base"] ?? "推理";
  const inferenceLabel = inference
    ? target > 0
      ? `${modeLabel} 第 ${saved}/${target} 集`
      : `${modeLabel} ${Math.round(inference.rollout_elapsed_s ?? 0)}s`
    : "";

  return {
    tasks,
    loading,
    refresh,
    recordingTask,
    sshInstallTask,
    trainingTask,
    inference,
    inferenceLabel,
    hasActiveRecording: !!recordingTask,
    hasActiveSshInstall: !!sshInstallTask,
    hasActiveTraining: !!trainingTask,
    hasActiveInference: !!inference,
  };
}
