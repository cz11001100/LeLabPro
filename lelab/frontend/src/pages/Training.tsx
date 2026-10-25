import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useHfAuth } from "@/contexts/HfAuthContext";

import { TrainingConfig, TrainingStatus, LogEntry } from "@/components/training/types";
import TrainingHeader from "@/components/training/TrainingHeader";
import ConfigurationTab from "@/components/training/ConfigurationTab";
import MonitoringStats from "@/components/training/monitoring/MonitoringStats";
import TrainingHealthCard from "@/components/training/monitoring/TrainingHealthCard";
import TrainingLogs from "@/components/training/monitoring/TrainingLogs";
import TrainingExtraGate from "@/components/training/TrainingExtraGate";
import PolicyExtraDialog from "@/components/training/PolicyExtraDialog";
import HfAuthBanner from "@/components/landing/HfAuthBanner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Loader2, Play, Square, Trash2, ArrowLeft, Cpu, ChevronDown } from "lucide-react";
import EvaluationCard from "@/components/jobs/EvaluationCard";

import { DatasetItem, listDatasets } from "@/lib/replayApi";
import {
  JobRecord,
  TrainingRequest,
  getJob,
  getJobLogs,
  getJobLogFile,
  listJobs,
  startTrainingJob,
  stopJob,
  deleteJob,
  listRunnerHardware,
  RunnerFlavor,
  SSHConfig,
  listSSHConfigs,
} from "@/lib/jobsApi";
import { JobCheckpoint, ModelInfo, getCheckpointModelInfo, listJobCheckpoints } from "@/lib/checkpointsApi";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import InferenceModal from "@/components/landing/InferenceModal";
import { useRobots } from "@/hooks/useRobots";

const POLL_INTERVAL_MS = 1000;
const MAX_LOG_LINES = 5000;

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-900/50 rounded-lg p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-white truncate" title={value}>
        {value}
      </div>
    </div>
  );
}

function formatBytes(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(0)} KB`;
}

function formatEpoch(sec: number | null): string {
  if (sec == null) return "—";
  return new Date(sec * 1000).toLocaleString();
}

function formatSci(n: number | null): string {
  if (n == null) return "—";
  return n === 0 ? "0" : n.toExponential(1);
}

function jobToStatus(job: JobRecord | null, isStarting: boolean): TrainingStatus {
  // Adapter so MonitoringStats can keep its current prop shape.
  if (!job) {
    return {
      training_active: isStarting,
      current_step: 0,
      total_steps: 0,
      available_controls: { stop_training: false, pause_training: false, resume_training: false },
    };
  }
  return {
    training_active: job.state === "running",
    current_step: job.metrics.current_step,
    total_steps: job.metrics.total_steps,
    current_loss: job.metrics.current_loss ?? undefined,
    current_lr: job.metrics.current_lr ?? undefined,
    grad_norm: job.metrics.grad_norm ?? undefined,
    eta_seconds: job.metrics.eta_seconds ?? undefined,
    available_controls: {
      stop_training: job.state === "running",
      pause_training: false,
      resume_training: false,
    },
  };
}

function configToRequest(c: TrainingConfig): TrainingRequest {
  // The backend's TrainingRequest has more optional fields; the form covers
  // the user-meaningful subset.
  return {
    target: c.target,
    dataset_repo_id: c.dataset_repo_id,
    dataset_root: c.dataset_root,
    job_name: c.job_name?.trim() ? c.job_name.trim() : undefined,
    policy_type: c.policy_type,
    steps: c.steps,
    batch_size: c.batch_size,
    seed: c.seed,
    num_workers: c.num_workers,
    log_freq: c.log_freq,
    save_freq: c.save_freq,
    save_checkpoint: c.save_checkpoint,
    resume: c.resume,
    resume_from_job_id: c.resume_from_job_id,
    wandb_enable: c.wandb_enable,
    wandb_project: c.wandb_project,
    wandb_entity: c.wandb_entity,
    wandb_notes: c.wandb_notes,
    wandb_mode: c.wandb_mode,
    wandb_disable_artifact: c.wandb_disable_artifact,
    policy_device: c.policy_device,
    policy_use_amp: c.policy_use_amp,
    policy_pretrained_path: c.policy_pretrained_path,
    optimizer_type: c.optimizer_type,
    optimizer_lr: c.optimizer_lr,
    optimizer_weight_decay: c.optimizer_weight_decay,
    optimizer_grad_clip_norm: c.optimizer_grad_clip_norm,
    scheduler_warmup_steps: c.scheduler_warmup_steps,
    scheduler_decay_steps: c.scheduler_decay_steps,
    scheduler_decay_lr: c.scheduler_decay_lr,
    use_policy_training_preset: c.use_policy_training_preset,
    // Data augmentation
    augmentation_enable: c.augmentation_enable,
    augmentation_max_num_transforms: c.augmentation_max_num_transforms,
    augmentation_random_order: c.augmentation_random_order,
    augmentation_brightness: c.augmentation_brightness,
    augmentation_contrast: c.augmentation_contrast,
    augmentation_saturation: c.augmentation_saturation,
    augmentation_hue: c.augmentation_hue,
    augmentation_sharpness: c.augmentation_sharpness,
    augmentation_affine: c.augmentation_affine,
    // RA-BC sample weighting
    sample_weighting_enable: c.sample_weighting_enable,
    sample_weighting_type: "rabc",
    sample_weighting_progress_path: c.sample_weighting_progress_path || undefined,
    sample_weighting_kappa: c.sample_weighting_kappa,
    sample_weighting_head_mode: c.sample_weighting_head_mode,
    // GR00T-specific. Only forward the policy_* fields for groot so other
    // policies never receive flags their config doesn't define.
    dataset_image_transforms_enable: c.dataset_image_transforms_enable,
    ...(c.policy_type === "groot"
      ? {
          policy_base_model_path: c.policy_base_model_path,
          policy_embodiment_tag: c.policy_embodiment_tag,
          policy_chunk_size: c.policy_chunk_size,
          policy_n_action_steps: c.policy_n_action_steps,
          policy_use_relative_actions: c.policy_use_relative_actions,
          policy_relative_exclude_joints: c.policy_relative_exclude_joints,
          policy_use_bf16: c.policy_use_bf16,
        }
      : {}),
  };
}

const ConfigurationMode: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { auth } = useHfAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const location = useLocation();
  const prefilledDatasetRepoId =
    (location.state as { datasetRepoId?: string } | null)?.datasetRepoId ?? "";
  const preselectSSH =
    (location.state as { sshRunner?: boolean } | null)?.sshRunner ?? false;

  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    target: preselectSSH ? { runner: "ssh" } : { runner: "local" },
    dataset_repo_id: prefilledDatasetRepoId,
    job_name: undefined,
    policy_type: "act",
    steps: 10000,
    batch_size: 8,
    seed: 1000,
    num_workers: 4,
    log_freq: 250,
    save_freq: 1000,
    save_checkpoint: true,
    resume: false,
    resume_from_job_id: undefined,
    wandb_enable: false,
    wandb_mode: "online",
    wandb_disable_artifact: false,
    policy_device: "cuda",
    policy_use_amp: false,
    use_policy_training_preset: true,
    augmentation_enable: false,
    augmentation_max_num_transforms: 3,
    augmentation_random_order: false,
    augmentation_brightness: true,
    augmentation_contrast: true,
    augmentation_saturation: true,
    augmentation_hue: true,
    augmentation_sharpness: true,
    augmentation_affine: true,
    sample_weighting_enable: false,
    sample_weighting_head_mode: "sparse",
  });

  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [trainingExtraAvailable, setTrainingExtraAvailable] = useState<boolean | null>(null);
  const [trainingExtraInstallHint, setTrainingExtraInstallHint] = useState<string>("pip install accelerate");
  const [localJobRunning, setLocalJobRunning] = useState<boolean>(false);
  const [isStarting, setIsStarting] = useState(false);
  const [policyExtra, setPolicyExtra] = useState<{
    policyType: string;
    packageName: string;
    installTarget: string;
    installHint: string;
  } | null>(null);
  const [authenticated, setAuthenticated] = useState<boolean>(false);
  const [flavors, setFlavors] = useState<RunnerFlavor[]>([]);
  const [hardwareLoading, setHardwareLoading] = useState(true);
  const [sshConfigs, setSSHConfigs] = useState<SSHConfig[]>([]);
  const [sshConfigsLoading, setSSHConfigsLoading] = useState(true);

  useEffect(() => {
    setDatasetsLoading(true);
    listDatasets(baseUrl, fetchWithHeaders)
      .then(setDatasets)
      .catch(() => setDatasets([]))
      .finally(() => setDatasetsLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    fetchWithHeaders(`${baseUrl}/system/training-extra`)
      .then((r) => r.json())
      .then((data: { available: boolean; install_hint: string }) => {
        setTrainingExtraAvailable(data.available);
        setTrainingExtraInstallHint(data.install_hint);
      })
      .catch(() => setTrainingExtraAvailable(true));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Only the local lock matters for the Start button; cloud jobs can stack.
    // Pull a generous slice so a running local isn't masked by newer cloud
    // jobs in the started_at-desc ordering.
    listJobs(baseUrl, fetchWithHeaders, 200)
      .then((j) =>
        setLocalJobRunning(
          j.some((r) => r.runner === "local" && r.state === "running"),
        ),
      )
      .catch(() => setLocalJobRunning(false));
  }, [baseUrl, fetchWithHeaders]);

  useEffect(() => {
    // Re-fetches when auth status flips (e.g. user pastes a token in
    // HfAuthBanner) so flavors unlock without a page reload.
    setHardwareLoading(true);
    listRunnerHardware(baseUrl, fetchWithHeaders)
      .then((data) => {
        setAuthenticated(data.authenticated);
        setFlavors(data.flavors);
      })
      .catch(() => {
        setAuthenticated(false);
        setFlavors([]);
      })
      .finally(() => setHardwareLoading(false));
  }, [baseUrl, fetchWithHeaders, auth.status]);

  useEffect(() => {
    listSSHConfigs(baseUrl, fetchWithHeaders)
      .then(setSSHConfigs)
      .catch(() => setSSHConfigs([]))
      .finally(() => setSSHConfigsLoading(false));
  }, [baseUrl, fetchWithHeaders]);

  const updateConfig = <T extends keyof TrainingConfig>(key: T, value: TrainingConfig[T]) => {
    setTrainingConfig((prev) => ({ ...prev, [key]: value }));
  };

  const handleStart = async () => {
    if (!trainingConfig.dataset_repo_id.trim()) {
      toast({ title: "Error", description: "Dataset repository ID is required", variant: "destructive" });
      return;
    }

    // Pre-flight: smolvla/pi0/diffusion need an optional package installed
    // locally. Catch it here with a one-click installer instead of a buried
    // ImportError after the job has already started. Cloud jobs run in their
    // own environment, so the local package is irrelevant — skip the check.
    if (trainingConfig.target.runner === "local") {
      try {
        const r = await fetchWithHeaders(
          `${baseUrl}/system/policy-extra/${trainingConfig.policy_type}`,
        );
        if (r.ok) {
          const extra = await r.json();
          if (extra.needs_extra && !extra.available) {
            setPolicyExtra({
              policyType: trainingConfig.policy_type,
              packageName: extra.package,
              installTarget: extra.install_target,
              installHint: extra.install_hint,
            });
            return;
          }
        }
      } catch {
        // Check failed (offline / older backend) — fall through and let the
        // job report any problem itself.
      }
    }

    setIsStarting(true);
    try {
      const job = await startTrainingJob(baseUrl, fetchWithHeaders, configToRequest(trainingConfig));
      toast({ title: "Training Started", description: job.name });
      navigate(`/training/${job.id}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast({ title: "Error", description: msg, variant: "destructive" });
      // If the failure was the 409 case, refresh our running-job knowledge.
      listJobs(baseUrl, fetchWithHeaders, 200)
        .then((j) =>
          setLocalJobRunning(
            j.some((r) => r.runner === "local" && r.state === "running"),
          ),
        )
        .catch(() => {});
    } finally {
      setIsStarting(false);
    }
  };

  if (trainingExtraAvailable === null) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto">
          <TrainingHeader />
          <div className="flex items-center justify-center py-24 text-slate-400">
            <Loader2 className="w-6 h-6 animate-spin mr-3" />
            Checking training environment…
          </div>
        </div>
      </div>
    );
  }

  if (trainingExtraAvailable === false) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto">
          <TrainingHeader />
          <TrainingExtraGate installHint={trainingExtraInstallHint} />
        </div>
      </div>
    );
  }

  const targetRequiresAuth = trainingConfig.target.runner === "hf_cloud";
  const targetMissingFlavor =
    trainingConfig.target.runner === "hf_cloud" && !trainingConfig.target.flavor;
  const targetMissingSSHConfig =
    trainingConfig.target.runner === "ssh" && !trainingConfig.target.ssh_config_id;
  const localBlocked =
    trainingConfig.target.runner === "local" && localJobRunning;
  const startDisabled =
    isStarting ||
    !trainingConfig.dataset_repo_id.trim() ||
    localBlocked ||
    (targetRequiresAuth && !authenticated) ||
    targetMissingFlavor ||
    targetMissingSSHConfig;
  const startTooltip = localBlocked
    ? "Another local training is already running"
    : targetRequiresAuth && !authenticated
    ? "Log in to Hugging Face to use cloud compute"
    : targetMissingFlavor
    ? "Select a hardware flavor"
    : targetMissingSSHConfig
    ? "Select an SSH connection"
    : undefined;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        <TrainingHeader />
        <HfAuthBanner />
        <ConfigurationTab
          config={trainingConfig}
          updateConfig={updateConfig}
          datasets={datasets}
          datasetsLoading={datasetsLoading}
          authenticated={authenticated}
          flavors={flavors}
          hardwareLoading={hardwareLoading}
          sshConfigs={sshConfigs}
          sshConfigsLoading={sshConfigsLoading}
        />
        <div className="max-w-3xl mx-auto mt-6 flex justify-end">
          {(() => {
            const startButton = (
              <Button
                onClick={handleStart}
                disabled={startDisabled}
                size="lg"
                className="bg-green-500 hover:bg-green-600 text-white font-semibold px-6"
              >
                {isStarting ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" /> Starting…
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5 mr-2" /> 开始训练 Start Training
                  </>
                )}
              </Button>
            );
            // Native `title` doesn't fire reliably on disabled buttons across
            // browsers — and since Radix's tooltip relies on pointer events
            // that a disabled button swallows, wrap in a span so the trigger
            // still receives hover/focus.
            if (!startTooltip) return startButton;
            return (
              <Tooltip>
                <TooltipTrigger asChild>
                  <span tabIndex={0}>{startButton}</span>
                </TooltipTrigger>
                <TooltipContent>{startTooltip}</TooltipContent>
              </Tooltip>
            );
          })()}
        </div>
      </div>

      {policyExtra && (
        <PolicyExtraDialog
          open={!!policyExtra}
          onOpenChange={(o) => !o && setPolicyExtra(null)}
          policyType={policyExtra.policyType}
          packageName={policyExtra.packageName}
          installTarget={policyExtra.installTarget}
          installHint={policyExtra.installHint}
        />
      )}
    </div>
  );
};

const MonitoringMode: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const { selectedRecord } = useRobots();
  const [job, setJob] = useState<JobRecord | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [inferenceModalOpen, setInferenceModalOpen] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Seed logs from the persistent on-disk file once on mount, so navigating
  // away and back (or coming in fresh on a finished/interrupted job) shows
  // the full log history. Polling /logs continues from this point — the
  // backend drains the live queue in the same /log-file call so we don't
  // double-display lines that were buffered when we landed.
  useEffect(() => {
    let cancelled = false;
    getJobLogFile(baseUrl, fetchWithHeaders, jobId)
      .then((seeded) => {
        if (!cancelled && seeded.length > 0) setLogs(seeded);
      })
      .catch(() => {
        // 404 or transient — fall through; live polling will fill in.
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Read latest job state from a ref so the polling intervals below stay
  // stable instead of tearing down/rebuilding on every state transition.
  const jobStateRef = useRef(job?.state);
  jobStateRef.current = job?.state;

  // Poll checkpoints — every 5s while the job is running.
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
        .then((cks) => {
          if (cancelled) return;
          setCheckpoints(cks);
          if (cks.length > 0) {
            const latest = cks[cks.length - 1].step;
            setSelectedStep((prev) =>
              prev != null && cks.some((c) => c.step === prev) ? prev : latest,
            );
          } else {
            setSelectedStep(null);
          }
        })
        .catch(() => {
          if (!cancelled) {
            setCheckpoints([]);
            setSelectedStep(null);
          }
        });
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Static architecture/training metadata. Imported models have no live
  // training logs, so this is their only "about this model" surface.
  useEffect(() => {
    if (job?.runner !== "imported" || selectedStep == null) {
      setModelInfo(null);
      return;
    }
    let cancelled = false;
    getCheckpointModelInfo(baseUrl, fetchWithHeaders, jobId, selectedStep)
      .then((info) => {
        if (!cancelled) setModelInfo(info);
      })
      .catch(() => {
        if (!cancelled) setModelInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, jobId, job?.runner, selectedStep]);

  // Poll the job + its logs while running. Caps log lines to avoid unbounded growth.
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await getJob(baseUrl, fetchWithHeaders, jobId);
        if (cancelled) return;
        setJob(next);
        if (next.state === "running") {
          const newLogs = await getJobLogs(baseUrl, fetchWithHeaders, jobId);
          if (!cancelled && newLogs.length > 0) {
            setLogs((prev) => {
              const merged = [...prev, ...newLogs];
              return merged.length > MAX_LOG_LINES
                ? merged.slice(merged.length - MAX_LOG_LINES)
                : merged;
            });
          }
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };
    tick();
    const id = setInterval(() => {
      if (cancelled) return;
      if (jobStateRef.current && jobStateRef.current !== "running") return;
      tick();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, jobId]);

  // Auto-scroll the log panel as new lines arrive.
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getProgressPercentage = () => {
    if (!job || job.metrics.total_steps === 0) return 0;
    return (job.metrics.current_step / job.metrics.total_steps) * 100;
  };

  const handleStop = async () => {
    if (!job) return;
    if (!window.confirm("Stop this run?")) return;
    try {
      const next = await stopJob(baseUrl, fetchWithHeaders, job.id);
      setJob(next);
      toast({ title: "Stopping…" });
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleDelete = async () => {
    if (!job) return;
    if (!window.confirm("Delete this run? This wipes the output directory.")) return;
    try {
      await deleteJob(baseUrl, fetchWithHeaders, job.id);
      toast({ title: "Job removed" });
      navigate("/");
    } catch (e) {
      toast({
        title: "Delete failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  if (error && !job) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto space-y-4">
          <Button variant="ghost" onClick={() => navigate("/")} className="text-slate-400">
            <ArrowLeft className="w-4 h-4 mr-2" /> Back to Jobs
          </Button>
          <p className="text-red-300">Couldn't load job {jobId}: {error}</p>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen bg-slate-900 text-white p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-center py-24 text-slate-400">
          <Loader2 className="w-6 h-6 animate-spin mr-3" /> Loading job…
        </div>
      </div>
    );
  }

  const isRunning = job.state === "running";
  const isImported = job.runner === "imported";
  const canValidate = !isRunning && job.checkpoint_count > 0;

  return (
    <div className="min-h-screen bg-slate-900 text-white p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate("/")} className="text-slate-400 hover:text-white">
              <ArrowLeft className="w-4 h-4 mr-2" /> Jobs
            </Button>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-semibold text-white">{job.name}</h1>
                {isImported ? (
                  <span className="text-xs px-2 py-0.5 rounded bg-purple-900/40 text-purple-200 border border-purple-700">
                    Imported
                  </span>
                ) : job.runner === "hf_cloud" ? (
                  <span className="text-xs px-2 py-0.5 rounded bg-amber-900/40 text-amber-200 border border-amber-700">
                    HF · {job.hf_flavor ?? "cloud"}
                  </span>
                ) : job.runner === "ssh" ? (
                  <span className="text-xs px-2 py-0.5 rounded bg-blue-900/40 text-blue-200 border border-blue-700">
                    SSH
                  </span>
                ) : (
                  <span className="text-xs px-2 py-0.5 rounded bg-slate-700 text-slate-200 border border-slate-600">
                    Local
                  </span>
                )}
                {job.runner === "hf_cloud" && job.hf_repo_id && job.state === "done" && (
                  <a
                    href={`https://huggingface.co/${job.hf_repo_id}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-amber-300 hover:text-amber-200 underline"
                  >
                    View on Hub ↗
                  </a>
                )}
                {job.wandb_run_url && (
                  <a
                    href={job.wandb_run_url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-yellow-300 hover:text-yellow-200 underline"
                  >
                    View on W&B ↗
                  </a>
                )}
              </div>
              <p className="text-xs text-slate-400">
                {isImported ? (job.hf_repo_id || job.output_dir) : job.state}
                {job.error_message ? ` — ${job.error_message}` : ""}
              </p>
            </div>
          </div>
          {isRunning ? (
            <Button onClick={handleStop} className="bg-red-500 hover:bg-red-600 text-white">
              <Square className="w-4 h-4 mr-2" /> Stop
            </Button>
          ) : (
            <Button onClick={handleDelete} variant="ghost" className="text-slate-400 hover:text-white">
              <Trash2 className="w-4 h-4 mr-2" /> Delete
            </Button>
          )}
        </div>

        {!isImported && (
          <>
            <MonitoringStats
              jobId={jobId}
              trainingStatus={jobToStatus(job, false)}
              getProgressPercentage={getProgressPercentage}
              formatTime={formatTime}
            />
            <TrainingHealthCard jobId={jobId} refreshInterval={10000} />
          </>
        )}

        {isImported && modelInfo && (
          <Collapsible defaultOpen={false} className="bg-slate-800/50 border border-slate-700 rounded-xl">
            <CollapsibleTrigger className="w-full">
              <CardHeader className="pb-3">
                <CardTitle className="text-white flex items-center gap-2 text-base">
                  <Cpu className="h-5 w-5 text-cyan-400" />
                  模型信息 / 训练配置
                  <ChevronDown className="h-4 w-4 text-slate-400 ml-auto transition-transform data-[state=open]:rotate-180" />
                </CardTitle>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <InfoCell label="策略类型" value={modelInfo.policy_type ?? "—"} />
                  <InfoCell label="Step" value={String(modelInfo.step)} />
                  <InfoCell label="动作块 chunk_size" value={modelInfo.chunk_size != null ? String(modelInfo.chunk_size) : "—"} />
                  <InfoCell label="执行步 n_action_steps" value={modelInfo.n_action_steps != null ? String(modelInfo.n_action_steps) : "—"} />
                  <InfoCell label="VLM 层数" value={modelInfo.num_vlm_layers != null ? String(modelInfo.num_vlm_layers) : "—"} />
                  <InfoCell label="Expert 层数" value={modelInfo.num_expert_layers != null && modelInfo.num_expert_layers > 0 ? String(modelInfo.num_expert_layers) : "默认"} />
                  <InfoCell label="权重大小" value={formatBytes(modelInfo.size_bytes)} />
                  <InfoCell label="保存时间" value={formatEpoch(modelInfo.saved_at)} />
                </div>
                {modelInfo.training && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <InfoCell label="batch_size" value={modelInfo.training.batch_size != null ? String(modelInfo.training.batch_size) : "—"} />
                    <InfoCell label="总步数" value={modelInfo.training.steps != null ? String(modelInfo.training.steps) : "—"} />
                    <InfoCell label="优化器" value={modelInfo.training.optimizer_type ?? "—"} />
                    <InfoCell label="学习率 lr" value={formatSci(modelInfo.training.lr)} />
                    <InfoCell label="weight_decay" value={formatSci(modelInfo.training.weight_decay)} />
                    <InfoCell label="grad_clip" value={modelInfo.training.grad_clip_norm != null ? String(modelInfo.training.grad_clip_norm) : "—"} />
                    <InfoCell label="调度器" value={modelInfo.training.scheduler_type ?? "—"} />
                    <InfoCell label="warmup 步" value={modelInfo.training.warmup_steps != null ? String(modelInfo.training.warmup_steps) : "—"} />
                    <InfoCell label="decay 步" value={modelInfo.training.decay_steps != null ? String(modelInfo.training.decay_steps) : "—"} />
                    <InfoCell label="peak_lr" value={formatSci(modelInfo.training.peak_lr)} />
                    <InfoCell label="decay_lr" value={formatSci(modelInfo.training.decay_lr)} />
                    <InfoCell label="seed" value={modelInfo.training.seed != null ? String(modelInfo.training.seed) : "—"} />
                  </div>
                )}
                {modelInfo.dataset_repo_id && (
                  <p className="text-xs text-slate-400">
                    训练数据集：
                    <span className="text-cyan-300">{modelInfo.dataset_repo_id}</span>
                  </p>
                )}
              </CardContent>
            </CollapsibleContent>
          </Collapsible>
        )}

        {canValidate && <EvaluationCard job={job} />}

        <div className="bg-slate-800/40 border border-slate-700 rounded-lg p-4 flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-300">Run inference</span>
          {checkpoints.length === 0 ? (
            <span className="text-xs text-slate-500">No checkpoints yet — wait for the first save.</span>
          ) : (
            <>
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
              />
              <Button
                onClick={() => setInferenceModalOpen(true)}
                disabled={selectedStep == null}
                className="bg-green-500 hover:bg-green-600 text-white"
              >
                <Play className="w-4 h-4 mr-2" />
                Run on robot
              </Button>
            </>
          )}
        </div>
        <InferenceModal
          open={inferenceModalOpen}
          onOpenChange={setInferenceModalOpen}
          robot={selectedRecord}
          jobId={jobId}
          initialStep={selectedStep}
        />
        {!isImported && <TrainingLogs logs={logs} logContainerRef={logContainerRef} />}
      </div>
    </div>
  );
};

const Training: React.FC = () => {
  const { jobId } = useParams<{ jobId?: string }>();
  return jobId ? <MonitoringMode jobId={jobId} /> : <ConfigurationMode />;
};

export default Training;
