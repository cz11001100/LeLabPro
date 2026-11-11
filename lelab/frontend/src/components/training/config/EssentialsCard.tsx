import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { NumberInput } from '@/components/ui/number-input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ConfigComponentProps } from '../types';
import DatasetCombobox from '@/components/replay/DatasetCombobox';
import { DatasetItem } from '@/lib/replayApi';
import TrackioInstallDialog from '../TrackioInstallDialog';
import { useApi } from '@/contexts/ApiContext';

interface EssentialsCardProps extends ConfigComponentProps {
  datasets: DatasetItem[];
  datasetsLoading: boolean;
}

const EssentialsCard: React.FC<EssentialsCardProps> = ({ config, updateConfig, datasets, datasetsLoading }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [trackioDialogOpen, setTrackioDialogOpen] = useState(false);
  const [trackioInstallHint, setTrackioInstallHint] = useState('pip install trackio');

  const handleTrackioToggle = async (checked: boolean) => {
    if (!checked) {
      updateConfig('trackio_enable', false);
      return;
    }
    // Check availability before flipping the switch on. If trackio isn't
    // importable in this lelab process, surface the same install flow used
    // for the training extra (accelerate) instead of letting the user start
    // a run that will fail.
    try {
      const r = await fetchWithHeaders(`${baseUrl}/system/trackio-extra`);
      const data: { available: boolean; install_hint: string } = await r.json();
      if (data.available) {
        updateConfig('trackio_enable', true);
      } else {
        setTrackioInstallHint(data.install_hint);
        setTrackioDialogOpen(true);
      }
    } catch {
      // Backend unreachable — let the user proceed; training start will
      // surface the real error if trackio is genuinely missing.
      updateConfig('trackio_enable', true);
    }
  };

  return (
    <Card className="bg-slate-800/50 border-slate-700 rounded-xl">
      <CardHeader>
        <CardTitle className="text-white">运行配置 Run Configuration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <Label className="text-slate-300">数据集 Dataset *</Label>
          <div className="mt-1">
            <DatasetCombobox
              datasets={datasets}
              loading={datasetsLoading}
              value={config.dataset_repo_id || null}
              onChange={(repoId) => {
                if (repoId) updateConfig('dataset_repo_id', repoId);
              }}
            />
          </div>
        </div>

        <div>
          <Label htmlFor="job_name" className="text-slate-300">
            任务名称 Job Name <span className="text-slate-500 text-xs">(可选)</span>
          </Label>
          <Input
            id="job_name"
            value={config.job_name || ''}
            onChange={(e) => updateConfig('job_name', e.target.value || undefined)}
            placeholder={`留空则默认：${config.policy_type.toUpperCase()} · ${config.dataset_repo_id || 'org/dataset'}`}
            className="bg-slate-900 border-slate-600 text-white rounded-lg mt-1"
          />
          <p className="text-xs text-slate-500 mt-1">
            显示在任务列表与详情页的标题；不影响输出目录名。
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label htmlFor="policy_type" className="text-slate-300">
              策略 Policy
            </Label>
            <Select
              value={config.policy_type}
              onValueChange={(value) => updateConfig('policy_type', value)}
            >
              <SelectTrigger id="policy_type" className="bg-slate-900 border-slate-600 text-white rounded-lg">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-slate-800 border-slate-600 text-white max-h-80">
                <SelectGroup>
                  <SelectLabel className="text-slate-500">常用</SelectLabel>
                  <SelectItem value="act">ACT (Action Chunking Transformer)</SelectItem>
                  <SelectItem value="diffusion">Diffusion Policy</SelectItem>
                  <SelectItem value="pi0">PI0</SelectItem>
                  <SelectItem value="pi05">PI0.5</SelectItem>
                  <SelectItem value="pi0_fast">PI0 Fast</SelectItem>
                  <SelectItem value="smolvla">SmolVLA</SelectItem>
                  <SelectItem value="groot">GR00T N1.7</SelectItem>
                  <SelectItem value="tdmpc">TD-MPC</SelectItem>
                  <SelectItem value="vqbet">VQ-BeT</SelectItem>
                </SelectGroup>
                <SelectGroup>
                  <SelectLabel className="text-slate-500">VLA</SelectLabel>
                  <SelectItem value="dm05">DM05</SelectItem>
                  <SelectItem value="eo1">EO1</SelectItem>
                  <SelectItem value="evo1">EVO1</SelectItem>
                  <SelectItem value="fineart_vla">FineART VLA</SelectItem>
                  <SelectItem value="g05">G05</SelectItem>
                  <SelectItem value="lingbot_vla_v2">LingBot-VLA v2</SelectItem>
                  <SelectItem value="molmoact2">MolmoAct2</SelectItem>
                  <SelectItem value="wall_oss_05">WALL-OSS 0.5</SelectItem>
                  <SelectItem value="wall_x">WALL-X</SelectItem>
                  <SelectItem value="xvla">XVLA</SelectItem>
                </SelectGroup>
                <SelectGroup>
                  <SelectLabel className="text-slate-500">世界模型</SelectLabel>
                  <SelectItem value="cosmos3_nano">Cosmos3 Nano</SelectItem>
                  <SelectItem value="cosmos3_edge">Cosmos3 Edge</SelectItem>
                  <SelectItem value="fastwam">FastWAM</SelectItem>
                  <SelectItem value="flux3">FLUX.3</SelectItem>
                  <SelectItem value="lawam">LaWAM</SelectItem>
                  <SelectItem value="lingbot_va">LingBot-VA</SelectItem>
                  <SelectItem value="longwam">LongWAM</SelectItem>
                  <SelectItem value="vla_jepa">VLA-JEPA</SelectItem>
                </SelectGroup>
                <SelectGroup>
                  <SelectLabel className="text-slate-500">其他</SelectLabel>
                  <SelectItem value="multi_task_dit">Multi-task DiT</SelectItem>
                  <SelectItem value="reward_classifier">Reward Classifier</SelectItem>
                  <SelectItem value="sac" disabled>SAC（在线 RL，暂不支持）</SelectItem>
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="steps" className="text-slate-300">
              训练步数 Training Steps
            </Label>
            <NumberInput
              id="steps"
              value={config.steps}
              onChange={(v) => {
                if (v !== undefined) updateConfig('steps', v);
              }}
              className="bg-slate-900 border-slate-600 text-white rounded-lg"
            />
          </div>

          <div>
            <Label htmlFor="batch_size" className="text-slate-300">
              批大小 Batch Size
            </Label>
            <NumberInput
              id="batch_size"
              value={config.batch_size}
              onChange={(v) => {
                if (v !== undefined) updateConfig('batch_size', v);
              }}
              className="bg-slate-900 border-slate-600 text-white rounded-lg"
            />
          </div>

          <div className="flex items-center space-x-3 pt-6">
            <Switch
              id="trackio_enable"
              checked={config.trackio_enable}
              onCheckedChange={handleTrackioToggle}
              className="data-[state=checked]:bg-green-500"
            />
            <Label htmlFor="trackio_enable" className="text-slate-300">
              启用 Trackio
            </Label>
          </div>
        </div>

        <TrackioInstallDialog
          open={trackioDialogOpen}
          onOpenChange={setTrackioDialogOpen}
          installHint={trackioInstallHint}
        />

        {config.trackio_enable && (
          <div>
            <Label htmlFor="trackio_project" className="text-slate-300">
              Trackio 项目名 Project Name
            </Label>
            <Input
              id="trackio_project"
              value={config.trackio_project || ''}
              onChange={(e) =>
                updateConfig('trackio_project', e.target.value || undefined)
              }
              placeholder="my-robotics-project"
              className="bg-slate-900 border-slate-600 text-white rounded-lg"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default EssentialsCard;
