import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { NumberInput } from '@/components/ui/number-input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { ConfigComponentProps } from '../types';
import { useApi } from '@/contexts/ApiContext';
import { listJobs, JobRecord } from '@/lib/jobsApi';
import { listJobCheckpoints } from '@/lib/checkpointsApi';
import AugmentationSection from './AugmentationSection';

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
    {children}
  </h4>
);

const AdvancedCard: React.FC<ConfigComponentProps> = ({ config, updateConfig }) => {
  const [expanded, setExpanded] = useState(false);
  const { baseUrl, fetchWithHeaders } = useApi();
  const [sshJobs, setSshJobs] = useState<JobRecord[]>([]);
  const [ckptOptions, setCkptOptions] = useState<{ value: string; label: string }[]>([]);
  const [ckptLoading, setCkptLoading] = useState(false);

  const isSSH = config.target?.runner === "ssh";

  useEffect(() => {
    if (!expanded || !isSSH) return;
    const controller = new AbortController();
    listJobs(baseUrl, fetchWithHeaders, 100, controller.signal)
      .then((jobs) => setSshJobs(jobs.filter((j) => j.runner === "ssh" && j.checkpoint_count > 0)))
      .catch(() => {});
    return () => controller.abort();
  }, [expanded, isSSH, baseUrl, fetchWithHeaders]);

  // Checkpoints usable as finetune starting weights. Paths must exist on the
  // machine that trains: SSH jobs expose remote paths, local/imported jobs
  // expose host paths — so filter by the selected runner.
  useEffect(() => {
    if (!expanded) return;
    const controller = new AbortController();
    setCkptLoading(true);
    listJobs(baseUrl, fetchWithHeaders, 100, controller.signal)
      .then((jobs) => {
        const relevant = jobs.filter(
          (j) => j.checkpoint_count > 0 && (isSSH ? j.runner === "ssh" : j.runner !== "ssh"),
        );
        return Promise.all(
          relevant.map((job) =>
            listJobCheckpoints(baseUrl, fetchWithHeaders, job.id, controller.signal)
              .then((cks) =>
                cks
                  .filter((c) => c.source === "local")
                  .map((c) => ({
                    value: c.ref,
                    label: `${job.name} · step ${c.step}`,
                  })),
              )
              .catch(() => [] as { value: string; label: string }[]),
          ),
        );
      })
      .then((lists) => setCkptOptions(lists.flat()))
      .catch(() => setCkptOptions([]))
      .finally(() => setCkptLoading(false));
    return () => controller.abort();
  }, [expanded, isSSH, baseUrl, fetchWithHeaders]);

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader
        role="button"
        tabIndex={0}
        aria-expanded={expanded}
        onClick={() => setExpanded((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            setExpanded((v) => !v);
          }
        }}
        className="cursor-pointer select-none flex flex-row items-center justify-between"
      >
        <span className="text-white font-semibold">Advanced</span>
        <span className="flex items-center gap-1 text-slate-400 text-sm">
          {expanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
          {expanded ? 'Hide' : 'Show'}
        </span>
      </CardHeader>

      {expanded && (
        <CardContent className="space-y-8">
          {/* Policy */}
          <section className="space-y-4">
            <SectionHeading>Policy</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="policy_device" className="text-slate-300">
                  Device
                </Label>
                <Select
                  value={config.policy_device || 'cuda'}
                  onValueChange={(value) => updateConfig('policy_device', value)}
                >
                  <SelectTrigger id="policy_device" className="bg-slate-900 border-slate-600 text-white rounded-lg">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600 text-white">
                    <SelectItem value="cuda">CUDA (GPU)</SelectItem>
                    <SelectItem value="cpu">CPU</SelectItem>
                    <SelectItem value="mps">MPS (Apple Silicon)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center space-x-3 pt-6">
                <Switch
                  id="policy_use_amp"
                  checked={config.policy_use_amp}
                  onCheckedChange={(checked) => updateConfig('policy_use_amp', checked)}
                />
                <Label htmlFor="policy_use_amp" className="text-slate-300">
                  Use Automatic Mixed Precision
                </Label>
              </div>
            </div>
            <div>
              <Label htmlFor="policy_pretrained_path" className="text-slate-300">
                Finetune from checkpoint (optional)
              </Label>
              <Select
                value={config.policy_pretrained_path || '__none__'}
                onValueChange={(value) => {
                  if (value === '__none__') {
                    updateConfig('policy_pretrained_path', undefined);
                  } else {
                    updateConfig('policy_pretrained_path', value);
                    // Mutually exclusive with resume: resume restores the old
                    // lr schedule/optimizer, finetune-from-weights starts fresh.
                    updateConfig('resume', false);
                    updateConfig('resume_from_job_id', undefined);
                  }
                }}
              >
                <SelectTrigger id="policy_pretrained_path" className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                  <SelectValue placeholder={ckptLoading ? 'Loading checkpoints…' : 'Train from scratch (base policy)'} />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600 text-white">
                  <SelectItem value="__none__">Train from scratch (base policy)</SelectItem>
                  {ckptOptions.map((o) => (
                    <SelectItem key={o.value} value={o.value}>
                      {o.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-slate-500 mt-1">
                Starts a NEW run from these weights (fresh optimizer/scheduler, new
                dataset allowed) — unlike Resume, which continues the old run.
              </p>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Training */}
          <section className="space-y-4">
            <SectionHeading>Training</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="seed" className="text-slate-300">
                  Random Seed
                </Label>
                <NumberInput
                  id="seed"
                  value={config.seed}
                  onChange={(v) => updateConfig('seed', v)}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="num_workers" className="text-slate-300">
                  Number of Workers
                </Label>
                <NumberInput
                  id="num_workers"
                  value={config.num_workers}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('num_workers', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Optimizer */}
          <section className="space-y-4">
            <SectionHeading>Optimizer</SectionHeading>
            <div>
              <Label htmlFor="optimizer_type" className="text-slate-300">
                Optimizer
              </Label>
              <Select
                value={config.optimizer_type || 'preset'}
                onValueChange={(value) =>
                  updateConfig('optimizer_type', value === 'preset' ? undefined : value)
                }
              >
                <SelectTrigger id="optimizer_type" className="bg-slate-900 border-slate-600 text-white rounded-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600 text-white">
                  <SelectItem value="preset">Policy preset (recommended)</SelectItem>
                  <SelectItem value="adam">Adam</SelectItem>
                  <SelectItem value="adamw">AdamW</SelectItem>
                  <SelectItem value="sgd">SGD</SelectItem>
                  <SelectItem value="multi_adam">Multi Adam</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="optimizer_lr" className="text-slate-300">
                  Learning Rate
                </Label>
                <NumberInput
                  id="optimizer_lr"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_lr}
                  onChange={(v) => updateConfig('optimizer_lr', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="optimizer_weight_decay" className="text-slate-300">
                  Weight Decay
                </Label>
                <NumberInput
                  id="optimizer_weight_decay"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_weight_decay}
                  onChange={(v) => updateConfig('optimizer_weight_decay', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="optimizer_grad_clip_norm" className="text-slate-300">
                  Gradient Clipping
                </Label>
                <NumberInput
                  id="optimizer_grad_clip_norm"
                  integer={false}
                  step="0.0001"
                  value={config.optimizer_grad_clip_norm}
                  onChange={(v) => updateConfig('optimizer_grad_clip_norm', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          {/* Scheduler */}
          <section className="space-y-4">
            <SectionHeading>Scheduler</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <Label htmlFor="scheduler_warmup_steps" className="text-slate-300">
                  Warmup Steps
                </Label>
                <NumberInput
                  id="scheduler_warmup_steps"
                  value={config.scheduler_warmup_steps}
                  onChange={(v) => updateConfig('scheduler_warmup_steps', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="scheduler_decay_steps" className="text-slate-300">
                  Decay Steps
                </Label>
                <NumberInput
                  id="scheduler_decay_steps"
                  value={config.scheduler_decay_steps}
                  onChange={(v) => updateConfig('scheduler_decay_steps', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="scheduler_decay_lr" className="text-slate-300">
                  Final LR
                </Label>
                <NumberInput
                  id="scheduler_decay_lr"
                  integer={false}
                  step="0.000001"
                  value={config.scheduler_decay_lr}
                  onChange={(v) => updateConfig('scheduler_decay_lr', v)}
                  placeholder="Use policy default"
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
            <p className="text-xs text-slate-500">
              Leave empty to keep the policy preset. For a finetune-from-checkpoint
              run, set Decay Steps = Steps and a lower Learning Rate above.
            </p>
          </section>

          <Separator className="bg-slate-700" />

          {/* RA-BC sample weighting */}
          <section className="space-y-4">
            <SectionHeading>RA-BC 样本加权</SectionHeading>
            <div className="flex items-center justify-between gap-4">
              <Label htmlFor="sample_weighting_enable" className="text-slate-300">
                按任务进度加权
              </Label>
              <Switch
                id="sample_weighting_enable"
                checked={!!config.sample_weighting_enable}
                onCheckedChange={(v) => updateConfig('sample_weighting_enable', v)}
              />
            </div>
            {config.sample_weighting_enable && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="sample_weighting_kappa" className="text-slate-300">
                    Kappa（进度增益阈值）
                  </Label>
                  <NumberInput
                    id="sample_weighting_kappa"
                    integer={false}
                    step="0.001"
                    value={config.sample_weighting_kappa}
                    onChange={(v) => updateConfig('sample_weighting_kappa', v)}
                    placeholder="0.01"
                    className="bg-slate-900 border-slate-600 text-white rounded-lg"
                  />
                </div>
                <div>
                  <Label htmlFor="sample_weighting_head_mode" className="text-slate-300">
                    进度头
                  </Label>
                  <Select
                    value={config.sample_weighting_head_mode ?? 'sparse'}
                    onValueChange={(v) => updateConfig('sample_weighting_head_mode', v)}
                  >
                    <SelectTrigger className="bg-slate-900 border-slate-600 text-white">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-900 border-slate-700 text-white">
                      <SelectItem value="sparse">sparse</SelectItem>
                      <SelectItem value="dense">dense</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="md:col-span-2">
                  <Label htmlFor="sample_weighting_progress_path" className="text-slate-300">
                    进度文件
                  </Label>
                  <Input
                    id="sample_weighting_progress_path"
                    value={config.sample_weighting_progress_path ?? ''}
                    onChange={(e) => updateConfig('sample_weighting_progress_path', e.target.value)}
                    placeholder="留空 = 数据集目录下的 sarm_progress.parquet"
                    className="bg-slate-900 border-slate-600 text-white rounded-lg"
                  />
                </div>
                <p className="text-xs text-slate-500 md:col-span-2">
                  需先在评测的「真机成功率」里给该数据集打分。逐样本加权目前只有
                  SmolVLA / pi0 / pi0.5 支持，ACT 与 Diffusion 开了会报错。
                </p>
              </div>
            )}
          </section>

          <Separator className="bg-slate-700" />

          {/* Logging & Checkpointing */}
          <section className="space-y-4">
            <SectionHeading>Logging & Checkpointing</SectionHeading>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="log_freq" className="text-slate-300">
                  Log Frequency
                </Label>
                <NumberInput
                  id="log_freq"
                  value={config.log_freq}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('log_freq', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
              <div>
                <Label htmlFor="save_freq" className="text-slate-300">
                  Save Frequency
                </Label>
                <NumberInput
                  id="save_freq"
                  value={config.save_freq}
                  onChange={(v) => {
                    if (v !== undefined) updateConfig('save_freq', v);
                  }}
                  className="bg-slate-900 border-slate-600 text-white rounded-lg"
                />
              </div>
            </div>
            <div className="flex items-center space-x-3">
              <Switch
                id="save_checkpoint"
                checked={config.save_checkpoint}
                onCheckedChange={(checked) => updateConfig('save_checkpoint', checked)}
              />
              <Label htmlFor="save_checkpoint" className="text-slate-300">
                Save Checkpoints
              </Label>
            </div>
            <div className="flex items-center space-x-3">
              <Switch
                id="resume"
                checked={config.resume}
                onCheckedChange={(checked) => {
                  updateConfig('resume', checked);
                  if (!checked) updateConfig('resume_from_job_id', undefined);
                  if (checked) updateConfig('policy_pretrained_path', undefined);
                }}
              />
              <Label htmlFor="resume" className="text-slate-300">
                Resume from Checkpoint
              </Label>
            </div>
            {config.resume && isSSH && (
              <div>
                <Label htmlFor="resume_from_job" className="text-slate-300">
                  Resume from Job
                </Label>
                <Select
                  value={config.resume_from_job_id || ''}
                  onValueChange={(value) => updateConfig('resume_from_job_id', value || undefined)}
                >
                  <SelectTrigger id="resume_from_job" className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1">
                    <SelectValue placeholder="Select a job to resume..." />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-600 text-white">
                    {sshJobs.map((job) => (
                      <SelectItem key={job.id} value={job.id}>
                        {job.name} ({job.checkpoint_count} ckpt{job.checkpoint_count !== 1 ? 's' : ''})
                      </SelectItem>
                    ))}
                    {sshJobs.length === 0 && (
                      <SelectItem value="__none__" disabled>
                        No SSH jobs with checkpoints
                      </SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-500 mt-1">
                  Training will continue from the latest checkpoint of the selected job.
                </p>
              </div>
            )}
          </section>

          {config.wandb_enable && (
            <>
              <Separator className="bg-slate-700" />
              <section className="space-y-4">
                <SectionHeading>Weights & Biases</SectionHeading>
                <div>
                  <Label htmlFor="wandb_entity" className="text-slate-300">
                    W&B Entity (optional)
                  </Label>
                  <Input
                    id="wandb_entity"
                    value={config.wandb_entity || ''}
                    onChange={(e) =>
                      updateConfig('wandb_entity', e.target.value || undefined)
                    }
                    placeholder="your-username"
                    className="bg-slate-900 border-slate-600 text-white rounded-lg"
                  />
                </div>
                <div>
                  <Label htmlFor="wandb_notes" className="text-slate-300">
                    W&B Notes (optional)
                  </Label>
                  <Input
                    id="wandb_notes"
                    value={config.wandb_notes || ''}
                    onChange={(e) =>
                      updateConfig('wandb_notes', e.target.value || undefined)
                    }
                    placeholder="Training run notes..."
                    className="bg-slate-900 border-slate-600 text-white rounded-lg"
                  />
                </div>
                <div>
                  <Label htmlFor="wandb_mode" className="text-slate-300">
                    W&B Mode
                  </Label>
                  <Select
                    value={config.wandb_mode || 'online'}
                    onValueChange={(value) => updateConfig('wandb_mode', value)}
                  >
                    <SelectTrigger id="wandb_mode" className="bg-slate-900 border-slate-600 text-white rounded-lg">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-600 text-white">
                      <SelectItem value="online">Online</SelectItem>
                      <SelectItem value="offline">Offline</SelectItem>
                      <SelectItem value="disabled">Disabled</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex items-center space-x-3">
                  <Switch
                    id="wandb_disable_artifact"
                    checked={config.wandb_disable_artifact}
                    onCheckedChange={(checked) =>
                      updateConfig('wandb_disable_artifact', checked)
                    }
                  />
                  <Label htmlFor="wandb_disable_artifact" className="text-slate-300">
                    Disable Artifacts
                  </Label>
                </div>
              </section>
            </>
          )}

          {!config.wandb_enable && <Separator className="bg-slate-700" />}

          {/* Misc */}
          <section className="space-y-4">
            <SectionHeading>Misc</SectionHeading>
            <div className="flex items-center space-x-3">
              <Switch
                id="use_policy_training_preset"
                checked={config.use_policy_training_preset}
                onCheckedChange={(checked) =>
                  updateConfig('use_policy_training_preset', checked)
                }
              />
              <Label htmlFor="use_policy_training_preset" className="text-slate-300">
                Use Policy Training Preset
              </Label>
            </div>
          </section>

          <Separator className="bg-slate-700" />

          <AugmentationSection config={config} updateConfig={updateConfig} />
        </CardContent>
      )}
    </Card>
  );
};

export default AdvancedCard;
