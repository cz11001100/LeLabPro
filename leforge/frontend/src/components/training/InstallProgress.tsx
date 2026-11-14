import React from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Loader2,
  XCircle,
} from "lucide-react";
import type { InstallState, LogEntry } from "@/hooks/useInstallExtra";

interface InstallProgressProps {
  state: InstallState;
  error: string | null;
  logs: LogEntry[];
  logBoxRef: React.RefObject<HTMLDivElement>;
  onInstall: () => void;
  onRetry: () => void;

  installHint: string;
  packageName: string;
  idleTitle: string;
  idleDescription: React.ReactNode;
  doneDescription: React.ReactNode;
}

export function installTitle(state: InstallState, idleTitle: string): string {
  switch (state) {
    case "done":
      return "Install Complete";
    case "error":
      return "Install Failed";
    case "installing":
      return "Installing…";
    default:
      return idleTitle;
  }
}

export function InstallTitleIcon({ state }: { state: InstallState }) {
  if (state === "done") return <CheckCircle2 className="w-6 h-6 text-green-700" />;
  if (state === "error") return <XCircle className="w-6 h-6 text-red-700" />;
  if (state === "installing")
    return <Loader2 className="w-6 h-6 text-sky-700 animate-spin" />;
  return <AlertTriangle className="w-6 h-6 text-amber-700" />;
}

export const InstallProgress: React.FC<InstallProgressProps> = ({
  state,
  error,
  logs,
  logBoxRef,
  onInstall,
  onRetry,
  installHint,
  packageName,
  idleDescription,
  doneDescription,
}) => {
  const { toast } = useToast();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(installHint);
      toast({ title: "Copied", description: installHint });
    } catch {
      toast({
        title: "Copy failed",
        description: "Select the command and copy manually.",
        variant: "destructive",
      });
    }
  };

  return (
    <>
      {state === "idle" && (
        <>
          <p className="text-slate-700">{idleDescription}</p>
          <div className="flex items-center gap-2">
            <code className="flex-1 bg-sky-100 border border-sky-300 rounded-lg px-3 py-2 text-sm text-slate-800 font-mono">
              {installHint}
            </code>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleCopy}
              className="text-slate-600 hover:text-slate-900"
              aria-label="Copy install command"
            >
              <Copy className="w-4 h-4" />
            </Button>
          </div>
          <Button
            onClick={onInstall}
            className="bg-green-600 hover:bg-green-700 text-white font-semibold"
          >
            Install Now
          </Button>
        </>
      )}

      {state === "installing" && (
        <p className="text-slate-700">
          Installing{" "}
          <code className="px-1 py-0.5 rounded bg-sky-100 text-sky-700">
            {packageName}
          </code>
          . This usually takes about 10 seconds.
        </p>
      )}

      {state === "done" && (
        <div className="space-y-3 text-slate-700">{doneDescription}</div>
      )}

      {state === "error" && (
        <>
          <p className="text-red-700">{error || "Install failed."}</p>
          <Button
            onClick={onRetry}
            className="bg-sky-200 hover:bg-sky-200 text-slate-900"
          >
            Try again
          </Button>
        </>
      )}

      {state === "error" && logs.length > 0 && (
        <div
          ref={logBoxRef}
          className="bg-sky-100 rounded-lg p-3 h-48 overflow-y-auto font-mono text-xs border border-sky-300 text-slate-700 whitespace-pre-wrap break-words"
        >
          {logs.map((log, idx) => (
            <div key={idx}>{log.message}</div>
          ))}
        </div>
      )}
    </>
  );
};

export const RestartInstructions: React.FC<{ purpose: string }> = ({
  purpose,
}) => (
  <>
    <p>
      Install complete. Restart{" "}
      <code className="px-1 py-0.5 rounded bg-sky-100 text-sky-700">
        leforge
      </code>{" "}
      to enable {purpose}:
    </p>
    <ol className="list-decimal list-inside space-y-2 pl-1">
      <li>
        Press{" "}
        <kbd className="px-1.5 py-0.5 rounded bg-sky-100 border border-sky-300 text-xs font-mono text-slate-800">
          Ctrl+C
        </kbd>{" "}
        in the terminal running{" "}
        <code className="px-1 py-0.5 rounded bg-sky-100 text-sky-700">
          leforge
        </code>
        .
      </li>
      <li>
        Run{" "}
        <code className="px-1 py-0.5 rounded bg-sky-100 text-sky-700">
          leforge
        </code>{" "}
        again.
      </li>
    </ol>
  </>
);
