import React, { useCallback, useEffect, useState } from "react";
import { Activity, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { listSSHConfigs, SSHConfig } from "@/lib/jobsApi";
import {
  RolloutDatasetInfo,
  getSarmTrainingStatus,
  listRolloutDatasets,
  startSarmTraining,
} from "@/lib/rolloutEvalApi";

const CAMERA_KEYS = [
  { value: "observation.images.up", label: "up（俯视）" },
  { value: "observation.images.side", label: "side（侧视）" },
  { value: "observation.images.wrist", label: "wrist（腕部）" },
];

/** Train a task-specific SARM progress head on a pod, for the scoring card to use. */
const SarmTrainCard: React.FC<{ initialRepoId?: string }> = ({ initialRepoId = "" }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [datasets, setDatasets] = useState<RolloutDatasetInfo[]>([]);
  const [repoId, setRepoId] = useState(initialRepoId);
  const [imageKey, setImageKey] = useState(CAMERA_KEYS[0].value);
  const [sshConfigs, setSshConfigs] = useState<SSHConfig[]>([]);
  const [sshConfigId, setSshConfigId] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{
    training: boolean;
    finished_ok: boolean;
    log_tail: string;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    listRolloutDatasets(baseUrl, fetchWithHeaders)
      .then((list) => {
        if (cancelled) return;
        setDatasets(list);
        setRepoId((prev) => prev || list[0]?.repo_id || "");
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    let cancelled = false;
    listSSHConfigs(baseUrl, fetchWithHeaders)
      .then((list) => {
        if (cancelled) return;
        setSshConfigs(list);
        setSshConfigId((prev) => prev || list[0]?.id || "");
      })
      .catch(() => setSshConfigs([]));
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders]);

  const loadStatus = useCallback(async () => {
    if (!sshConfigId) return;
    try {
      const st = await getSarmTrainingStatus(baseUrl, fetchWithHeaders, sshConfigId);
      setStatus({
        training: st.training,
        finished_ok: st.finished_ok,
        log_tail: st.log_tail,
      });
    } catch {
      // Pod unreachable; the next poll retries.
    }
  }, [baseUrl, fetchWithHeaders, sshConfigId]);

  useEffect(() => {
    if (!status?.training || !sshConfigId) return;
    const id = setInterval(loadStatus, 5000);
    return () => clearInterval(id);
  }, [status?.training, sshConfigId, loadStatus]);

  const start = async () => {
    if (!repoId || !sshConfigId) return;
    setBusy(true);
    try {
      const res = await startSarmTraining(baseUrl, fetchWithHeaders, {
        config_id: sshConfigId,
        dataset_repo_id: repoId,
        image_key: imageKey,
      });
      toast({ title: res.message, description: `日志 ${res.log_path}` });
      setStatus({ training: true, finished_ok: false, log_tail: "" });
    } catch (e) {
      toast({
        title: "SARM 训练没起来",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/40 p-4 space-y-4">
      <div className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-blue-400" />
        <h2 className="text-xl font-semibold text-white">SARM 训练</h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="space-y-1">
          <Label className="text-xs text-slate-400">数据集</Label>
          <Select value={repoId} onValueChange={setRepoId}>
            <SelectTrigger className="bg-slate-900 border-slate-700 text-white text-xs h-9">
              <SelectValue placeholder="选择数据集" />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
              {datasets.map((d) => (
                <SelectItem key={d.repo_id} value={d.repo_id}>
                  {d.repo_id} · {d.is_rollout ? "采集" : "训练"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-slate-400">相机</Label>
          <Select value={imageKey} onValueChange={setImageKey}>
            <SelectTrigger className="bg-slate-900 border-slate-700 text-white text-xs h-9">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
              {CAMERA_KEYS.map((c) => (
                <SelectItem key={c.value} value={c.value}>
                  {c.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label className="text-xs text-slate-400">pod</Label>
          <Select value={sshConfigId} onValueChange={setSshConfigId}>
            <SelectTrigger className="bg-slate-900 border-slate-700 text-white text-xs h-9">
              <SelectValue placeholder="选 pod" />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-slate-700 text-white text-xs">
              {sshConfigs.map((c) => (
                <SelectItem key={c.id} value={c.id}>
                  {c.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <Button
        onClick={start}
        disabled={busy || !repoId || !sshConfigId}
        variant="outline"
        className="border-slate-600 text-black hover:bg-slate-200 hover:text-black text-sm"
      >
        {busy && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
        Train SARM on Pod
      </Button>

      {status && (
        <div className="text-xs text-slate-400">
          {status.training
            ? "训练中…"
            : status.finished_ok
              ? "训练完成，可在上面的打分卡片里选「SARM（本任务自训）」"
              : "训练结束（未确认成功）"}
          {status.log_tail ? ` · ${status.log_tail.split("\n").pop()}` : ""}
        </div>
      )}
    </div>
  );
};

export default SarmTrainCard;
