import React, { useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { DatasetItem } from "@/lib/replayApi";

interface ResumeDatasetPickerProps {
  datasets: DatasetItem[];
  loading: boolean;
  onSelect: (item: DatasetItem) => void;
  children: React.ReactNode;
}

const ResumeDatasetPicker: React.FC<ResumeDatasetPickerProps> = ({
  datasets,
  loading,
  onSelect,
  children,
}) => {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  // Only show local datasets (can't resume from Hub-only datasets)
  const localDatasets = datasets.filter(
    (d) => d.source === "local" || d.source === "both",
  );

  const filteredDatasets = localDatasets.filter((d) =>
    d.repo_id.toLowerCase().includes(query.toLowerCase()),
  );

  const handleSelect = (item: DatasetItem) => {
    onSelect(item);
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="w-[320px] p-0 bg-gray-800 border-gray-700 text-white"
        align="end"
      >
        <Command className="bg-gray-800">
          <CommandInput
            placeholder="Search datasets to resume…"
            value={query}
            onValueChange={(v) => setQuery(v)}
            className="text-white"
          />
          <CommandList>
            {filteredDatasets.length === 0 && (
              <CommandEmpty className="py-6 text-sm text-gray-400 text-center">
                {loading ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin text-blue-400" />
                    <span>Loading datasets…</span>
                  </div>
                ) : localDatasets.length === 0 ? (
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-medium">No local datasets</span>
                    <span className="text-xs">Record a new dataset first</span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-medium">No matches</span>
                    <span className="text-xs">Try a different search term</span>
                  </div>
                )}
              </CommandEmpty>
            )}
            {filteredDatasets.length > 0 && (
              <CommandGroup heading="Local datasets">
                {filteredDatasets.map((d) => (
                  <CommandItem
                    key={d.repo_id}
                    value={d.repo_id}
                    onSelect={() => handleSelect(d)}
                    className="text-white aria-selected:bg-blue-900/40"
                  >
                    <span className="flex-1 truncate">{d.repo_id}</span>
                    {d.private && (
                      <span className="text-xs text-amber-400">private</span>
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
            )}
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
};

export default ResumeDatasetPicker;
