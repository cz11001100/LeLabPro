import React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, CheckCircle, ChevronDown, RotateCw } from "lucide-react";
import CameraConfiguration, {
  CameraConfig,
} from "@/components/recording/CameraConfiguration";
import { useHfAuth } from "@/contexts/HfAuthContext";
import { RobotRecord } from "@/hooks/useRobots";
import { getTeleopMode, TELEOP_MODE_LABELS } from "@/lib/teleopMode";

interface RecordingModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  datasetName: string;
  setDatasetName: (value: string) => void;
  singleTask: string;
  setSingleTask: (value: string) => void;
  numEpisodes: number;
  setNumEpisodes: (value: number) => void;
  episodeTimeS: number;
  setEpisodeTimeS: (value: number) => void;
  resetTimeS: number;
  setResetTimeS: (value: number) => void;
  streamingEncoding: boolean;
  setStreamingEncoding: (value: boolean) => void;
  resume: boolean;
  setResume: (value: boolean) => void;
  cameras: CameraConfig[];
  setCameras: (cameras: CameraConfig[]) => void;
  emaEnabled: boolean;
  setEmaEnabled: (value: boolean) => void;
  emaAlpha: number;
  setEmaAlpha: (value: number) => void;
  onStart: () => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>;
}

const RecordingModal: React.FC<RecordingModalProps> = ({
  open,
  onOpenChange,
  robot,
  datasetName,
  setDatasetName,
  singleTask,
  setSingleTask,
  numEpisodes,
  setNumEpisodes,
  episodeTimeS,
  setEpisodeTimeS,
  resetTimeS,
  setResetTimeS,
  streamingEncoding,
  setStreamingEncoding,
  resume,
  setResume,
  cameras,
  setCameras,
  emaEnabled,
  setEmaEnabled,
  emaAlpha,
  setEmaAlpha,
  onStart,
  releaseStreamsRef,
}) => {
  const { auth } = useHfAuth();

  const canStart = !!robot && robot.is_clean;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-sky-100 border-sky-200 text-slate-900 sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
              <span className="text-slate-900 font-bold text-sm">REC</span>
            </div>
          </div>
          <DialogTitle className="text-slate-900 text-center text-2xl font-bold">
            {resume ? "Continue Recording" : "Configure Recording"}
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-6 py-4">
          {resume && (
            <div className="flex items-center gap-2 rounded-lg bg-blue-100 border border-blue-500/30 px-4 py-3">
              <RotateCw className="h-4 w-4 text-blue-700 shrink-0" />
              <p className="text-sm text-blue-700">
                Appending to existing dataset{" "}
                <span className="font-mono font-semibold text-blue-700">
                  {auth.status === "authenticated"
                    ? `${auth.username}/${datasetName}`
                    : datasetName}
                </span>
              </p>
            </div>
          )}
          <DialogDescription className="text-slate-600 text-base leading-relaxed text-center">
            {resume
              ? "New episodes will be added to the existing dataset."
              : "Pick a configured robot and dataset parameters for recording."}
          </DialogDescription>

          <div className="grid grid-cols-1 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 border-b border-sky-300 pb-2">
                Robot Configuration
              </h3>
              {!robot ? (
                <Alert className="bg-amber-100 border-amber-300 text-amber-700">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Select and configure a robot on the Landing page before
                    recording.
                  </AlertDescription>
                </Alert>
              ) : !robot.is_clean ? (
                <Alert className="bg-amber-100 border-amber-300 text-amber-700">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    <strong>{robot.name}</strong> is missing a calibration.
                    Configure it before recording.
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="flex items-center gap-2 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-700" />
                  <span className="text-slate-800">
                    Recording with <strong>{robot.name}</strong>
                  </span>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 border-b border-sky-300 pb-2">
                Dataset Configuration
              </h3>
              <div className="grid grid-cols-1 gap-4">
                <div className="space-y-2">
                  <Label
                    htmlFor="datasetName"
                    className="text-sm font-medium text-slate-700"
                  >
                    Dataset Name *
                  </Label>
                  <Input
                    id="datasetName"
                    value={datasetName}
                    onChange={(e) =>
                      setDatasetName(
                        e.target.value.replace(/[^A-Za-z0-9._-]/g, "_")
                      )
                    }
                    placeholder="my_dataset"
                    readOnly={resume}
                    className={`bg-sky-50 border-sky-300 text-slate-900 ${resume ? "opacity-70 cursor-not-allowed" : ""}`}
                  />
                  <p className="text-xs text-slate-600">
                    Letters, numbers, <code>.</code> <code>_</code>{" "}
                    <code>-</code> only — other characters become{" "}
                    <code>_</code>.
                  </p>
                  {datasetName &&
                    (auth.status === "authenticated" ? (
                      <p className="text-xs text-slate-600">
                        Will be saved as{" "}
                        <span className="text-slate-700 font-mono">
                          {auth.username}/{datasetName}
                        </span>
                      </p>
                    ) : auth.status === "unauthenticated" ? (
                      <p className="text-xs text-amber-700/80">
                        Log in to Hugging Face to set the repository owner.
                      </p>
                    ) : null)}
                </div>
                <div className="space-y-2">
                  <Label
                    htmlFor="singleTask"
                    className="text-sm font-medium text-slate-700"
                  >
                    Task Description *
                  </Label>
                  <Input
                    id="singleTask"
                    value={singleTask}
                    onChange={(e) => setSingleTask(e.target.value)}
                    placeholder="e.g., pick up the red block and place it on the blue square"
                    className="bg-sky-50 border-sky-300 text-slate-900"
                  />
                </div>
                <div className="space-y-2">
                  <Label
                    htmlFor="numEpisodes"
                    className="text-sm font-medium text-slate-700"
                  >
                    Number of Episodes
                  </Label>
                  <NumberInput
                    id="numEpisodes"
                    min="1"
                    max="100"
                    value={numEpisodes}
                    onChange={(v) => {
                      if (v !== undefined) setNumEpisodes(v);
                    }}
                    className="bg-sky-50 border-sky-300 text-slate-900"
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label
                      htmlFor="episodeTimeS"
                      className="text-sm font-medium text-slate-700"
                    >
                      Episode duration (seconds)
                    </Label>
                    <NumberInput
                      id="episodeTimeS"
                      min="1"
                      value={episodeTimeS}
                      onChange={(v) => {
                        if (v !== undefined) setEpisodeTimeS(v);
                      }}
                      className="bg-sky-50 border-sky-300 text-slate-900"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label
                      htmlFor="resetTimeS"
                      className="text-sm font-medium text-slate-700"
                    >
                      Reset duration (seconds)
                    </Label>
                    <NumberInput
                      id="resetTimeS"
                      min="1"
                      value={resetTimeS}
                      onChange={(v) => {
                        if (v !== undefined) setResetTimeS(v);
                      }}
                      className="bg-sky-50 border-sky-300 text-slate-900"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <CameraConfiguration
                cameras={cameras}
                onCamerasChange={setCameras}
                releaseStreamsRef={releaseStreamsRef}
                readOnly
              />
              <p className="text-xs text-slate-600">
                These are the cameras set up for this robot. To add or change a
                camera, configure it on the Calibration page.
              </p>
            </div>

            <Collapsible className="space-y-4 group">
              <CollapsibleTrigger className="flex items-center justify-between w-full text-lg font-semibold text-slate-900 border-b border-sky-300 pb-2">
                <span>Advanced Parameters</span>
                <ChevronDown className="w-4 h-4 transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3">
                <div className="flex items-start gap-3">
                  <Checkbox
                    id="streamingEncoding"
                    checked={streamingEncoding}
                    onCheckedChange={(value) =>
                      setStreamingEncoding(value === true)
                    }
                    className="mt-0.5 border-sky-400 data-[state=checked]:bg-red-500 data-[state=checked]:border-red-500"
                  />
                  <div className="space-y-1">
                    <Label
                      htmlFor="streamingEncoding"
                      className="text-sm font-medium text-slate-800 cursor-pointer"
                    >
                      Streaming video encoding
                    </Label>
                    <p className="text-xs text-slate-600">
                      Encodes frames in real time during capture so each
                      episode saves almost instantly. Uncheck to fall back to
                      the slower PNG-then-encode flow.
                    </p>
                  </div>
                </div>
                <div className="flex items-center justify-between pt-2 border-t border-sky-300">
                  <span className="text-sm text-slate-700">Teleop input</span>
                  <span className="text-sm font-medium text-slate-900">
                    {TELEOP_MODE_LABELS[getTeleopMode()]}
                  </span>
                </div>
                <div className="space-y-3 pt-2 border-t border-sky-300">
                  <div className="flex items-start gap-3">
                    <Checkbox
                      id="emaFilter"
                      checked={emaEnabled}
                      onCheckedChange={(value) =>
                        setEmaEnabled(value === true)
                      }
                      className="mt-0.5 border-sky-400 data-[state=checked]:bg-red-500 data-[state=checked]:border-red-500"
                    />
                    <div className="space-y-1 flex-1">
                      <Label
                        htmlFor="emaFilter"
                        className="text-sm font-medium text-slate-800 cursor-pointer"
                      >
                        Jitter reduction filter (EMA)
                      </Label>
                      <p className="text-xs text-slate-600">
                        Smooths high-frequency jitter in leader-follower control.
                        Lower values = more smoothing but more lag.
                      </p>
                      {emaEnabled && (
                        <div className="space-y-2 pt-2">
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-slate-600">Smooth</span>
                            <span className="text-xs font-mono text-slate-700">
                              {emaAlpha.toFixed(2)}
                            </span>
                            <span className="text-xs text-slate-600">Responsive</span>
                          </div>
                          <input
                            type="range"
                            min={0.1}
                            max={1.0}
                            step={0.05}
                            value={emaAlpha}
                            onChange={(e) => setEmaAlpha(parseFloat(e.target.value))}
                            className="w-full h-2 bg-sky-200 rounded-lg appearance-none cursor-pointer accent-red-500"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={onStart}
              disabled={!canStart}
              className="w-full sm:w-auto bg-red-500 hover:bg-red-600 text-white px-10 py-6 text-lg transition-all shadow-md shadow-red-500/30 hover:shadow-lg hover:shadow-red-500/40 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Start Recording
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto border-sky-400 hover:border-sky-300 px-10 py-6 text-lg text-slate-500 bg-sky-100 hover:bg-sky-200"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default RecordingModal;
