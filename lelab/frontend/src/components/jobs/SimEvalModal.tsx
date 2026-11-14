import React, { useCallback, useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Film,
  Gamepad2,
  Loader2,
  ShieldAlert,
  Square,
  Trophy,
} from "lucide-react";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  JobRecord,
  SimEnvOption,
  SimEvalStatus,
  getJobSimEval,
  listSimEnvironments,
  startJobSimEval,
  stopJobSimEval,
} from "@/lib/jobsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { ModelVariant, modelBaseName, modelStepOf } from "@/lib/modelGroups";

const POLL_MS = 2000;

// Radix Select refuses an empty-string value, but the builtin scene is
// addressed by "" (backend resolves it). Keep the sentinel in the picker and
// translate back when starting the run.
const ENV_BUILTIN = "__builtin__";

const Field: React.FC<{ label: string; children: React.ReactNode }> = ({
  label,
  children,
}) => (
  <div className="flex flex-col">
    <Label className="text-xs text-slate-400">{label}</Label>
    {children}
  </div>
);

const Metric: React.FC<{ label: string; value: string; hint?: string }> = ({
  label,
  value,
  hint,
}) => (
  <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-3">
    <div className="text-xs text-slate-400">{label}</div>
    <div className="text-lg font-semibold text-white tabular-nums">{value}</div>
    {hint ? <div className="text-[11px] text-slate-500">{hint}</div> : null}
  </div>
);

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  job: JobRecord | null;
  // Sibling checkpoints of one training run; picking one switches the job the
  // simulation is run against.
  group?: JobRecord[];
}

const SimEvalModal: React.FC<Props> = ({ open, onOpenChange, job, group }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();

  const members = group && group.length > 0 ? group : null;
  const [activeId, setActiveId] = useState<string | null>(job?.id ?? null);
  useEffect(() => {
    setActiveId(job?.id ?? null);
  }, [job?.id]);
  const activeJob = members
    ? members.find((m) => m.id === activeId) ?? members[0]
    : job;

  const variants: ModelVariant[] | undefined = members
    ? members.map((m) => ({ step: modelStepOf(m.name), jobId: m.id }))
    : undefined;

  const [status, setStatus] = useState<SimEvalStatus | null>(null);
  const [environments, setEnvironments] = useState<SimEnvOption[]>([]);
  const [env, setEnv] = useState("");
  const [episodes, setEpisodes] = useState(20);
  const [maxSteps, setMaxSteps] = useState<number | undefined>(undefined);
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    if (!activeJob) return;
    try {
      setStatus(await getJobSimEval(baseUrl, fetchWithHeaders, activeJob.id));
    } catch {
      // A missing record simply leaves the previous readout in place.
    }
  }, [activeJob, baseUrl, fetchWithHeaders]);

  useEffect(() => {
    if (!open || !activeJob) return;
    setStatus(null);
    setActiveId(activeJob.id);
    refresh();
    listSimEnvironments(baseUrl, fetchWithHeaders)
      .then((list) => {
        setEnvironments(list);
        setEnv((prev) =>
          prev || list.some((e) => (e.value || ENV_BUILTIN) === prev)
            ? prev
            : list[0]
              ? list[0].value || ENV_BUILTIN
              : "",
        );
      })
      .catch(() => setEnvironments([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, activeJob?.id]);

  const running = status?.sim_eval_state === "running";
  useEffect(() => {
    if (!open || !running) return;
    const id = window.setInterval(refresh, POLL_MS);
    return () => window.clearInterval(id);
  }, [open, running, refresh]);

  const run = async () => {
    if (!activeJob) return;
    setBusy(true);
    try {
      await startJobSimEval(baseUrl, fetchWithHeaders, activeJob.id, {
        n_episodes: episodes,
        env_hub_path: env === ENV_BUILTIN ? "" : env,
        ...(maxSteps != null ? { max_episode_steps: maxSteps } : {}),
        ...(seed != null ? { seed } : {}),
      });
      toast({ title: "仿真评测已启动" });
    } catch (e) {
      toast({
        title: "仿真评测启动失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const stop = async () => {
    if (!activeJob) return;
    setBusy(true);
    try {
      await stopJobSimEval(baseUrl, fetchWithHeaders, activeJob.id);
      toast({ title: "正在停止仿真评测…" });
    } catch (e) {
      toast({
        title: "停止失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const results = status?.sim_eval_results ?? null;
  const overall = results?.overall;
  const failed = status?.sim_eval_state === "failed" || results?.success === false;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[860px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            仿真评测
            {activeJob
              ? ` · ${members ? modelBaseName(activeJob.name) : activeJob.name}`
              : ""}
          </DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {variants && variants.length > 0 ? (
            <Field label="Checkpoint">
              <CheckpointDropdown
                checkpoints={variants.map((v) => ({
                  step: v.step,
                  source: "local" as const,
                  ref: v.jobId,
                }))}
                selectedStep={
                  variants.find((v) => v.jobId === activeJob?.id)?.step ?? null
                }
                onChange={(step) => {
                  const v = variants.find((x) => x.step === step);
                  if (v) setActiveId(v.jobId);
                }}
                triggerClassName="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
              />
            </Field>
          ) : null}
          <Field label="环境">
            <Select value={env} onValueChange={setEnv} disabled={environments.length === 0}>
              <SelectTrigger className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                <SelectValue placeholder="无可用环境" />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white">
                {environments.map((e) => (
                  <SelectItem
                    key={e.value || ENV_BUILTIN}
                    value={e.value || ENV_BUILTIN}
                  >
                    {e.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
          <Field label="集数">
            <NumberInput
              value={episodes}
              onChange={(v) => setEpisodes(v ?? 1)}
              min="1"
              max="500"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 h-9"
            />
          </Field>
          <Field label="每集最大步数">
            <NumberInput
              value={maxSteps}
              onChange={setMaxSteps}
              min="1"
              placeholder="默认"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 h-9"
            />
          </Field>
          <Field label="随机种子">
            <NumberInput
              value={seed}
              onChange={setSeed}
              min="0"
              placeholder="默认"
              className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1 h-9"
            />
          </Field>
        </div>

        <div className="flex items-center gap-3">
          {running ? (
            <Button
              onClick={stop}
              disabled={busy}
              className="ml-auto bg-red-500 hover:bg-red-600 text-white"
            >
              {busy ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Square className="w-4 h-4 mr-2" />
              )}
              停止仿真评测
            </Button>
          ) : (
            <Button
              onClick={run}
              disabled={busy || !activeJob || environments.length === 0}
              className="ml-auto bg-teal-600 hover:bg-teal-700 text-white"
            >
              {busy ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Gamepad2 className="w-4 h-4 mr-2" />
              )}
              {results ? "重新仿真评测" : "开始仿真评测"}
            </Button>
          )}
        </div>

        {running ? (
          <div className="flex items-center gap-2 text-sm text-teal-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            仿真评测运行中…
            {status?.progress
              ? ` 批次 ${status.progress.batches_done}/${status.progress.batches_total}`
              : ""}
          </div>
        ) : null}

        {failed ? (
          <div className="flex items-start gap-2 text-sm text-red-400">
            <ShieldAlert className="w-4 h-4 mt-0.5 shrink-0" />
            <span className="break-all">
              仿真评测失败{results?.error ? `：${results.error}` : ""}
            </span>
          </div>
        ) : overall ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Metric
                label="成功率"
                // lerobot-eval already reports these as percentages (0–100).
                value={`${overall.pc_success.toFixed(1)}%`}
                hint={
                  overall.pc_success_ci95
                    ? `95% 区间 ${overall.pc_success_ci95[0].toFixed(
                        1,
                      )}–${overall.pc_success_ci95[1].toFixed(1)}%`
                    : undefined
                }
              />
              <Metric
                label="成功集"
                value={`${overall.n_success}/${overall.n_episodes}`}
              />
              <Metric
                label="平均最大回报"
                value={overall.avg_max_reward.toFixed(3)}
                hint={`平均总回报 ${overall.avg_sum_reward.toFixed(3)}`}
              />
              <Metric label="用时" value={`${Math.round(overall.eval_s)}s`} />
            </div>
            {results?.diagnostics?.verdict ? (
              <div className="rounded-lg border border-slate-700 bg-slate-900/60 p-3 text-sm text-slate-200">
                {results.diagnostics.verdict}
              </div>
            ) : null}
            {(overall.video_paths?.length ?? 0) > 0 || status?.sim_eval_run_dir ? (
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <Film className="w-3.5 h-3.5" />
                {overall.video_paths?.length ?? 0} 段录像
                {status?.sim_eval_run_dir ? ` · ${status.sim_eval_run_dir}` : ""}
              </div>
            ) : null}
          </div>
        ) : null}

        {(status?.history?.length ?? 0) > 0 ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-slate-300">
              <Trophy className="w-4 h-4 text-amber-400" />
              历史
            </div>
            <div className="overflow-hidden rounded-lg border border-slate-700">
              <table className="w-full text-xs">
                <thead className="bg-slate-800 text-slate-400">
                  <tr>
                    <th className="px-3 py-2 text-left">时间</th>
                    <th className="px-3 py-2 text-left">Checkpoint</th>
                    <th className="px-3 py-2 text-left">环境</th>
                    <th className="px-3 py-2 text-right">集数</th>
                    <th className="px-3 py-2 text-right">成功率</th>
                  </tr>
                </thead>
                <tbody>
                  {status!.history.slice(0, 8).map((h) => (
                    <tr key={`${h.ts}-${h.run_dir}`} className="border-t border-slate-800">
                      <td className="px-3 py-2 text-slate-300">
                        {new Date(h.ts * 1000).toLocaleString()}
                      </td>
                      <td className="px-3 py-2 text-slate-300">
                        {h.step === 0 ? "latest" : `step ${h.step}`}
                      </td>
                      <td className="px-3 py-2 text-slate-400 truncate max-w-[180px]">
                        {h.env}
                      </td>
                      <td className="px-3 py-2 text-right text-slate-300 tabular-nums">
                        {h.n_episodes ?? "—"}
                      </td>
                      <td className="px-3 py-2 text-right text-white tabular-nums">
                        {h.pc_success == null
                          ? "—"
                          : `${h.pc_success.toFixed(1)}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  );
};

export default SimEvalModal;
