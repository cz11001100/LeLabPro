import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { JobCheckpoint } from "@/lib/checkpointsApi";

// One width for every first-row control box on the cards — the checkpoint chip
// here and the dataset card's date box — so the columns line up across sections.
export const CARD_BOX_W = "w-[104px]";

interface Props {
  checkpoints: JobCheckpoint[];
  selectedStep: number | null;
  onChange: (step: number) => void;
  disabled?: boolean;
  placeholder?: string;
  // Override the trigger styling (e.g. full-width to match a form grid);
  // defaults to the compact fixed-width chip used on the job/model cards.
  triggerClassName?: string;
}

export const CheckpointDropdown: React.FC<Props> = ({
  checkpoints,
  selectedStep,
  onChange,
  disabled,
  placeholder = "Select checkpoint",
  triggerClassName = `bg-sky-50 border-sky-300 text-slate-900 h-8 text-xs px-2 ${CARD_BOX_W} shrink-0`,
}) => {
  // step 0 is the sentinel for an imported single-model checkpoint (lerobot
  // never saves at step 0), so it has no meaningful step number — show
  // "latest" instead. Real training checkpoints keep their step label.
  const labelFor = (step: number) => (step === 0 ? "latest" : `step ${step}`);
  const value = selectedStep != null ? String(selectedStep) : undefined;
  return (
    <Select
      value={value}
      onValueChange={(v) => onChange(Number(v))}
      disabled={disabled || checkpoints.length === 0}
    >
      <SelectTrigger
        className={triggerClassName}
        onClick={(e) => e.stopPropagation()}
      >
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="bg-sky-100 border-sky-300 text-slate-900">
        {checkpoints.map((c) => (
          <SelectItem
            key={c.step}
            value={String(c.step)}
            onClick={(e) => e.stopPropagation()}
          >
            {labelFor(c.step)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
};

export default CheckpointDropdown;
