import React from 'react';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { NumberInput } from '@/components/ui/number-input';
import { ConfigComponentProps } from '../types';

const TOGGLES = [
  ['augmentation_brightness', '亮度抖动'],
  ['augmentation_contrast', '对比度抖动'],
  ['augmentation_saturation', '饱和度抖动'],
  ['augmentation_hue', '色相抖动'],
  ['augmentation_sharpness', '锐度抖动'],
  ['augmentation_affine', '仿射变换 (±5°)'],
] as const;

const AugmentationSection: React.FC<ConfigComponentProps> = ({ config, updateConfig }) => (
  <section className="space-y-4">
    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
      数据增强
    </h4>
    <div className="flex items-center space-x-3">
      <Switch
        id="adv_augmentation_enable"
        checked={config.augmentation_enable}
        onCheckedChange={(checked) => updateConfig('augmentation_enable', checked)}
      />
      <Label htmlFor="adv_augmentation_enable" className="text-slate-300">
        启用图像变换
      </Label>
    </div>
    {config.augmentation_enable && (
      <div className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {TOGGLES.map(([key, label]) => (
            <div key={key} className="flex items-center space-x-2">
              <Switch
                id={`adv_${key}`}
                checked={config[key]}
                onCheckedChange={(checked) => updateConfig(key, checked)}
              />
              <Label htmlFor={`adv_${key}`} className="text-slate-300 text-sm">
                {label}
              </Label>
            </div>
          ))}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <Label htmlFor="adv_augmentation_max_num_transforms" className="text-slate-300">
              每帧最大变换数
            </Label>
            <NumberInput
              id="adv_augmentation_max_num_transforms"
              value={config.augmentation_max_num_transforms}
              onChange={(v) => {
                if (v !== undefined) updateConfig('augmentation_max_num_transforms', v);
              }}
              className="bg-slate-900 border-slate-600 text-white rounded-lg"
            />
          </div>
          <div className="flex items-center space-x-3 pt-6">
            <Switch
              id="adv_augmentation_random_order"
              checked={config.augmentation_random_order}
              onCheckedChange={(checked) => updateConfig('augmentation_random_order', checked)}
            />
            <Label htmlFor="adv_augmentation_random_order" className="text-slate-300">
              随机应用顺序
            </Label>
          </div>
        </div>
      </div>
    )}
  </section>
);

export default AugmentationSection;
