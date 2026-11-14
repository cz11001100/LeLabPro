import React, { useState } from 'react';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { NumberInput } from '@/components/ui/number-input';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Eye, Loader2 } from 'lucide-react';
import { useApi } from '@/contexts/ApiContext';
import { getAugmentPreview, AugmentPreviewResult } from '@/lib/replayApi';
import { ConfigComponentProps } from '../types';

const TOGGLES = [
  ['augmentation_brightness', '亮度抖动'],
  ['augmentation_contrast', '对比度抖动'],
  ['augmentation_saturation', '饱和度抖动'],
  ['augmentation_hue', '色相抖动'],
  ['augmentation_sharpness', '锐度抖动'],
  ['augmentation_affine', '仿射变换 (±5°)'],
] as const;

const TRANSFORM_LABELS: Record<string, string> = {
  brightness: '亮度（最大）',
  contrast: '对比度（最大）',
  saturation: '饱和度（最大）',
  hue: '色相（最大）',
  sharpness: '锐度（最大）',
  affine: '仿射（最大）',
};

const AugmentationSection: React.FC<ConfigComponentProps> = ({ config, updateConfig }) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [preview, setPreview] = useState<AugmentPreviewResult | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const runPreview = async () => {
    setPreviewOpen(true);
    setPreviewLoading(true);
    setPreview(null);
    setPreviewError(null);
    try {
      const result = await getAugmentPreview(baseUrl, fetchWithHeaders, {
        repo_id: config.dataset_repo_id,
        n_frames: 3,
        augmentation_max_num_transforms: config.augmentation_max_num_transforms,
        augmentation_random_order: config.augmentation_random_order,
        augmentation_brightness: config.augmentation_brightness,
        augmentation_contrast: config.augmentation_contrast,
        augmentation_saturation: config.augmentation_saturation,
        augmentation_hue: config.augmentation_hue,
        augmentation_sharpness: config.augmentation_sharpness,
        augmentation_affine: config.augmentation_affine,
      });
      setPreview(result);
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreviewLoading(false);
    }
  };

  return (
    <section className="space-y-4">
      <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
        数据增强
      </h4>
      <div className="flex items-center space-x-3">
        <Switch
          id="adv_augmentation_enable"
          checked={config.augmentation_enable}
          onCheckedChange={(checked) => updateConfig('augmentation_enable', checked)}
        />
        <Label htmlFor="adv_augmentation_enable" className="text-slate-700">
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
                <Label htmlFor={`adv_${key}`} className="text-slate-700 text-sm">
                  {label}
                </Label>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label htmlFor="adv_augmentation_max_num_transforms" className="text-slate-700">
                每帧最大变换数
              </Label>
              <NumberInput
                id="adv_augmentation_max_num_transforms"
                value={config.augmentation_max_num_transforms}
                onChange={(v) => {
                  if (v !== undefined) updateConfig('augmentation_max_num_transforms', v);
                }}
                className="bg-sky-100 border-sky-300 text-slate-900 rounded-lg"
              />
            </div>
            <div className="flex items-center space-x-3 pt-6">
              <Switch
                id="adv_augmentation_random_order"
                checked={config.augmentation_random_order}
                onCheckedChange={(checked) => updateConfig('augmentation_random_order', checked)}
              />
              <Label htmlFor="adv_augmentation_random_order" className="text-slate-700">
                随机应用顺序
              </Label>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={runPreview}
            disabled={!config.dataset_repo_id || previewLoading}
            className="border-cyan-600/60 text-cyan-700 hover:bg-cyan-100 hover:text-cyan-700"
          >
            {previewLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Eye className="w-4 h-4 mr-2" />
            )}
            预览效果
          </Button>
        </div>
      )}

      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="bg-sky-100 border-sky-200 text-slate-900 sm:max-w-[860px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-slate-900 text-center text-xl font-bold">
              增广预览{preview?.camera ? ` · ${preview.camera.replace('observation.images.', '')}` : ''}
            </DialogTitle>
          </DialogHeader>
          {previewLoading && (
            <div className="flex items-center justify-center gap-2 py-10 text-slate-600">
              <Loader2 className="w-5 h-5 animate-spin" /> 正在应用变换…
            </div>
          )}
          {previewError && (
            <div className="text-sm text-red-700 bg-red-100 border border-red-300/50 rounded-lg p-3">
              {previewError}
            </div>
          )}
          {preview && preview.original.length > 0 && (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="text-xs text-slate-600">单项变换（最大强度，同一参考帧）</div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <img
                      src={`data:image/jpeg;base64,${preview.original[0]}`}
                      alt="original"
                      className="w-full rounded-lg border border-sky-300"
                    />
                    <div className="text-[11px] text-slate-600 mt-1">原始帧</div>
                  </div>
                  {preview.per_transform.map((t) => (
                    <div key={t.name}>
                      <img
                        src={`data:image/jpeg;base64,${t.image}`}
                        alt={t.name}
                        className="w-full rounded-lg border border-cyan-300/50"
                      />
                      <div className="text-[11px] text-cyan-700 mt-1">
                        {TRANSFORM_LABELS[t.name] ?? t.name}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs text-slate-600">组合随机（训练时实际采样）</div>
                {preview.original.map((orig, i) => (
                  <div key={i} className="grid grid-cols-2 gap-3">
                    <img
                      src={`data:image/jpeg;base64,${orig}`}
                      alt={`original ${i}`}
                      className="w-full rounded-lg border border-sky-300"
                    />
                    <img
                      src={`data:image/jpeg;base64,${preview.augmented[i]}`}
                      alt={`augmented ${i}`}
                      className="w-full rounded-lg border border-cyan-300/50"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
          {preview && preview.original.length === 0 && (
            <div className="text-slate-600 text-sm py-6 text-center">没有启用任何变换</div>
          )}
        </DialogContent>
      </Dialog>
    </section>
  );
};

export default AugmentationSection;