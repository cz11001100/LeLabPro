import React, { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronRight, Loader2, Merge } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { apiRequest } from "@/lib/apiClient";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useDatasets } from "@/hooks/useDatasets";

interface MergeSource {
  repo_id: string;
  episodes: number | null;
  frames: number | null;
  fps: number | null;
  robot_type: string | null;
  optional_columns: string[];
}

interface MergeCheck {
  sources: MergeSource[];
  blockers: string[];
}

interface MergeState {
  running: boolean;
  sources?: string[];
  target?: string;
  log_path?: string;
  elapsed_s?: number;
  exit_code?: number | null;
  target_exists?: boolean;
}

const POLL_MS = 2000;

const stamp = () => {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};

/**
 * Fold several local datasets into one — how correction episodes recorded on
 * the inference page get into a real training run.
 */
const MergeSection: React.FC = () => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const { datasets } = useDatasets();
  const local = useMemo(
    () =>
      datasets.filter(
        (d) =>
          (d.source === "local" || d.source === "both") &&
          // Delete/repair backups sit in the same cache dir; offering them as
          // merge inputs would silently retrain on superseded data.
          !d.repo_id.includes("__bak_"),
      ),
    [datasets],
  );
  const [picked, setPicked] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const [target, setTarget] = useState("");
  const [check, setCheck] = useState<MergeCheck | null>(null);
  const [state, setState] = useState<MergeState | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (target || local.length === 0) return;
    const user = local[0].repo_id.split("/")[0];
    setTarget(`${user}/merged_${stamp()}`);
  }, [local, target]);

  useEffect(() => {
    setCheck(null);
  }, [picked.join("|")]);

  useEffect(() => {
    if (!state?.running) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await apiRequest<MergeState>(baseUrl, fetchWithHeaders, "/dataset-merge-status", {
          action: "Merge status",
        });
        if (cancelled) return;
        setState(next);
        if (!next.running) {
          toast({
            title: next.exit_code === 0 && next.target_exists ? "合并完成" : "合并结束但没产出数据集",
            description:
              next.exit_code === 0 && next.target_exists
                ? `${next.target} · ${next.sources?.length ?? 0} 个来源`
                : `退出码 ${next.exit_code}，日志见 ${next.log_path}`,
            variant: next.exit_code === 0 && next.target_exists ? "default" : "destructive",
          });
        }
      } catch {
        // Transient: the next tick retries.
      }
    };
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [state?.running, baseUrl, fetchWithHeaders, toast]);

  const toggle = (repoId: string) =>
    setPicked((prev) =>
      prev.includes(repoId) ? prev.filter((r) => r !== repoId) : [...prev, repoId],
    );

  const runCheck = async () => {
    setBusy(true);
    try {
      setCheck(
        await apiRequest<MergeCheck>(baseUrl, fetchWithHeaders, "/dataset-merge-check", {
          method: "POST",
          body: { sources: picked },
          action: "Check merge",
        }),
      );
    } catch (e) {
      toast({
        title: "检查失败",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
    }
  };

  const runMerge = async () => {
    setBusy(true);
    try {
      const res = await apiRequest(baseUrl, fetchWithHeaders, "/dataset-merge", {
        method: "POST",
        body: { sources: picked, target_repo_id: target },
        action: "Merge datasets",
      });
      setState({ running: true, target, sources: picked, ...res });
    } catch (e) {
      toast({
        title: "合并没跑起来",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
    }
  };

  const blocked = !!check?.blockers?.length;
  const canMerge = picked.length >= 2 && !!check && !blocked && !!target.trim();

  return (
    <div className="bg-sky-100 border border-sky-300 rounded-lg p-6 mt-8">
      <div
        role="button"
        tabIndex={0}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
        className="flex items-center gap-2 mb-4 cursor-pointer select-none"
      >
        <Merge className="w-5 h-5 text-blue-700" />
        <h2 className="text-xl font-semibold text-slate-900">合并数据集 Merge Datasets</h2>
        {open ? (
          <ChevronDown className="w-4 h-4 text-slate-600 ml-auto" />
        ) : (
          <ChevronRight className="w-4 h-4 text-slate-600 ml-auto" />
        )}
      </div>

      {!open ? null : local.length === 0 ? (
        <p className="text-sm text-slate-600">本地还没有数据集。</p>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-64 overflow-y-auto mb-4">
            {local.map((d) => (
              <label
                key={d.repo_id}
                className="flex items-center gap-2 text-sm text-slate-800 px-2 py-1 rounded hover:bg-sky-200 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={picked.includes(d.repo_id)}
                  onChange={() => toggle(d.repo_id)}
                  className="accent-blue-500"
                />
                <span className="truncate">{d.repo_id}</span>
                <span className="text-slate-600 ml-auto flex-shrink-0">{d.last_modified?.slice(0, 10)}</span>
              </label>
            ))}
          </div>

          <div className="flex flex-wrap items-end gap-3 mb-4">
            <div className="flex-1 min-w-[280px] space-y-1">
              <Label htmlFor="merge-target" className="text-sm text-slate-700">
                合并结果
              </Label>
              <Input
                id="merge-target"
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="bg-sky-50 border-sky-300 text-slate-900"
              />
            </div>
            <Button
              onClick={runCheck}
              disabled={picked.length < 2 || busy}
              variant="outline"
              className="border-sky-300 text-slate-800 hover:bg-sky-200"
            >
              Check
            </Button>
            <Button
              onClick={runMerge}
              disabled={!canMerge || busy || state?.running}
              className="bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-40"
            >
              {state?.running || busy ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
              {state?.running ? `Merging ${Math.round(state.elapsed_s ?? 0)}s` : "Start Merge"}
            </Button>
          </div>

          {check && blocked && (
            <Alert className="bg-red-100 border-red-300 text-red-700 mb-4">
              <AlertDescription className="text-sm">
                {check.blockers.join("；")}
              </AlertDescription>
            </Alert>
          )}

          {check && !blocked && (
            <div className="overflow-x-auto rounded border border-sky-300">
              <table className="w-full text-sm">
                <thead className="bg-sky-50 text-slate-600">
                  <tr>
                    <th className="text-left px-3 py-1 font-medium">数据集</th>
                    <th className="text-right px-3 py-1 font-medium">集数</th>
                    <th className="text-right px-3 py-1 font-medium">帧数</th>
                    <th className="text-right px-3 py-1 font-medium">FPS</th>
                    <th className="text-left px-3 py-1 font-medium">多余列</th>
                  </tr>
                </thead>
                <tbody>
                  {check.sources.map((s) => (
                    <tr key={s.repo_id} className="border-t border-sky-300/60 text-slate-800">
                      <td className="px-3 py-1">{s.repo_id}</td>
                      <td className="px-3 py-1 text-right">{s.episodes ?? "—"}</td>
                      <td className="px-3 py-1 text-right">{s.frames ?? "—"}</td>
                      <td className="px-3 py-1 text-right">{s.fps ?? "—"}</td>
                      <td className="px-3 py-1 text-slate-600">
                        {s.optional_columns.length ? s.optional_columns.join(", ") : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {check.sources.some((s) => s.optional_columns.length > 0) && (
                <p className="text-xs text-slate-600 px-3 py-2">
                  合并前会先从带这些列的数据集上删掉（原数据集仍在，只是删列后重写）。
                </p>
              )}
            </div>
          )}

          {state?.log_path && !state.running && (
            <p className="text-xs text-slate-600 mt-3 break-all">
              日志 {state.log_path}
            </p>
          )}
        </>
      )}
    </div>
  );
};

export default MergeSection;
