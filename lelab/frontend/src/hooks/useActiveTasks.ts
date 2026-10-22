import { useEffect, useState, useCallback } from "react";
import { useApi } from "@/contexts/ApiContext";
import { getActiveTasks, ActiveTask } from "@/lib/jobsApi";

export function useActiveTasks(pollInterval: number = 3000) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [tasks, setTasks] = useState<ActiveTask[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const active = await getActiveTasks(baseUrl, fetchWithHeaders);
      setTasks(active);
    } catch {
      // ignore
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

  return {
    tasks,
    loading,
    refresh,
    recordingTask,
    sshInstallTask,
    trainingTask,
    hasActiveRecording: !!recordingTask,
    hasActiveSshInstall: !!sshInstallTask,
    hasActiveTraining: !!trainingTask,
  };
}
