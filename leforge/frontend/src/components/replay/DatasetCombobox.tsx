import React from "react";
import { Check, ChevronsUpDown, Pencil } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command";
import { cn } from "@/lib/utils";
import { DatasetItem } from "@/lib/replayApi";

interface Props {
  datasets: DatasetItem[];
  loading: boolean;
  value: string | null;
  onChange: (repoId: string | null) => void;
}

const REPO_ID_RE = /^[\w.\-]+\/[\w.\-]+$/;

const DatasetCombobox: React.FC<Props> = ({ datasets, loading, value, onChange }) => {
  const [open, setOpen] = React.useState(false);
  const [customMode, setCustomMode] = React.useState(false);
  const [customValue, setCustomValue] = React.useState("");

  const submitCustom = () => {
    const v = customValue.trim();
    if (REPO_ID_RE.test(v)) {
      onChange(v);
      setCustomMode(false);
    }
  };

  const localDatasets = datasets.filter((d) => d.source === "local" || d.source === "both");
  const hubDatasets = datasets.filter((d) => d.source === "hub");

  const renderItem = (d: DatasetItem) => (
    <CommandItem
      key={d.repo_id}
      value={d.repo_id}
      onSelect={() => { onChange(d.repo_id); setOpen(false); }}
      className="text-slate-900 aria-selected:bg-sky-200"
    >
      <Check className={cn("mr-2 h-4 w-4", value === d.repo_id ? "opacity-100" : "opacity-0")} />
      <span className="flex-1 truncate">{d.repo_id}</span>
      {d.source === "both" && <span className="text-xs text-slate-600 mr-2">on Hub</span>}
      {d.private && <span className="text-xs text-amber-700">private</span>}
    </CommandItem>
  );

  if (customMode) {
    return (
      <div className="flex gap-2">
        <Input
          autoFocus
          value={customValue}
          onChange={(e) => setCustomValue(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") submitCustom(); }}
          placeholder="org/数据集名，例如 CZ11001100/0921_4"
          className="bg-sky-50 border-sky-300 text-slate-900"
        />
        <Button onClick={submitCustom} disabled={!REPO_ID_RE.test(customValue.trim())}>
          Use
        </Button>
        <Button variant="ghost" onClick={() => setCustomMode(false)}>
          Cancel
        </Button>
      </div>
    );
  }

  return (
    <div>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between bg-sky-50 border-sky-300 text-slate-900 hover:bg-sky-200"
          >
            {value ?? (loading ? "Loading datasets…" : "Select dataset…")}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-sky-50 border-sky-300" align="start">
          <Command className="bg-sky-50 text-slate-900">
            <CommandInput placeholder="搜索数据集…" className="text-slate-900" />
            <CommandList>
              <CommandEmpty>{loading ? "加载…" : "无数据集。"}</CommandEmpty>
              {localDatasets.length > 0 && (
                <CommandGroup heading="本地 Local">
                  {localDatasets.map(renderItem)}
                </CommandGroup>
              )}
              {hubDatasets.length > 0 && (
                <CommandGroup heading="Hugging Face (Hub)">
                  {hubDatasets.map(renderItem)}
                </CommandGroup>
              )}
              <CommandGroup>
                <CommandItem
                  onSelect={() => { setCustomMode(true); setOpen(false); }}
                  className="text-purple-700 aria-selected:bg-sky-200"
                >
                  <Pencil className="mr-2 h-4 w-4" />
                  Enter Hub dataset ID manually…
                </CommandItem>
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      <button
        type="button"
        onClick={() => setCustomMode(true)}
        className="mt-1 text-xs text-purple-700 hover:text-purple-700 underline"
      >
        Enter Hub dataset ID manually (org/repo)…
      </button>
    </div>
  );
};

export default DatasetCombobox;
