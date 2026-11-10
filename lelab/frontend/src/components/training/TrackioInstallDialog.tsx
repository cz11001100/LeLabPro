import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useInstallExtra } from "@/hooks/useInstallExtra";
import {
  InstallProgress,
  InstallTitleIcon,
  RestartInstructions,
  installTitle,
} from "./InstallProgress";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  installHint: string;
}

const TrackioInstallDialog: React.FC<Props> = ({ open, onOpenChange, installHint }) => {
  const install = useInstallExtra("system/trackio-extra", open);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-slate-800 border-slate-700 text-white max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3 text-white">
            <InstallTitleIcon state={install.state} />
            {installTitle(install.state, "Trackio Not Installed")}
          </DialogTitle>
          <DialogDescription className="sr-only">
            Install the trackio package to enable run tracking.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <InstallProgress
            state={install.state}
            error={install.error}
            logs={install.logs}
            logBoxRef={install.logBoxRef}
            onInstall={install.handleInstall}
            onRetry={install.handleRetry}
            installHint={installHint}
            packageName="trackio"
            idleTitle="Trackio Not Installed"
            idleDescription={
              <>
                Enabling run tracking requires the{" "}
                <code className="px-1 py-0.5 rounded bg-slate-900 text-sky-300">
                  trackio
                </code>{" "}
                package, which isn't installed in this environment. Install it
                to log this run.
              </>
            }
            doneDescription={<RestartInstructions purpose="run tracking" />}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TrackioInstallDialog;
