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

interface ResumeDatasetPickerProps<T> {
  datasets: T[];
  getRepoId: (item: T) => string;
  /** Row subtitle, e.g. episode count. */
  getDetail?: (item: T) => string | undefined;
  /** Hide entries that cannot be resumed from (Hub-only, empty shells). */
  isSelectable?: (item: T) => boolean;
  loading: boolean;
  onSelect: (item: T) => void;
  children: React.ReactNode;
  searchPlaceholder?: string;
  groupHeading?: string;
  emptyHint?: string;
}

/** Searchable dataset list shared by the 续录 entry and the deploy modal's
 * capture picker, so both pages select a dataset the same way. */
function ResumeDatasetPicker<T>({
  datasets,
  getRepoId,
  getDetail,
  isSelectable,
  loading,
  onSelect,
  children,
  searchPlaceholder = "Search datasets to resume…",
  groupHeading = "Local datasets",
  emptyHint = "Record a new dataset first",
}: ResumeDatasetPickerProps<T>) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const selectable = datasets.filter((item) => (isSelectable ? isSelectable(item) : true));
  const filtered = selectable.filter((item) =>
    getRepoId(item).toLowerCase().includes(query.toLowerCase()),
  );

  const handleSelect = (item: T) => {
    onSelect(item);
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        className="w-[320px] p-0 bg-sky-50 border-sky-300 text-slate-900"
        align="end"
      >
        <Command className="bg-sky-50">
          <CommandInput
            placeholder={searchPlaceholder}
            value={query}
            onValueChange={(v) => setQuery(v)}
            className="text-slate-900"
          />
          <CommandList>
            {filtered.length === 0 && (
              <CommandEmpty className="py-6 text-sm text-slate-600 text-center">
                {loading ? (
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-5 w-5 animate-spin text-blue-700" />
                    <span>Loading datasets…</span>
                  </div>
                ) : selectable.length === 0 ? (
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-medium">No datasets</span>
                    <span className="text-xs">{emptyHint}</span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-1">
                    <span className="font-medium">No matches</span>
                    <span className="text-xs">Try a different search term</span>
                  </div>
                )}
              </CommandEmpty>
            )}
            {filtered.length > 0 && (
              <CommandGroup heading={groupHeading}>
                {filtered.map((item) => (
                  <CommandItem
                    key={getRepoId(item)}
                    value={getRepoId(item)}
                    onSelect={() => handleSelect(item)}
                    className="text-slate-900 aria-selected:bg-blue-100"
                  >
                    <span className="flex-1 truncate">{getRepoId(item)}</span>
                    {getDetail?.(item) && (
                      <span className="text-xs text-slate-600">{getDetail(item)}</span>
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
}

export default ResumeDatasetPicker;
