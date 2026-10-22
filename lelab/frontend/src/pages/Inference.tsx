import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Camera, Loader2, Square, Video, VideoOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import Logo from "@/components/Logo";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  InferenceStatus,
  captureInferenceSnapshot,
  getInferenceStatus,
  stopInference,
} from "@/lib/inferenceApi";

const POLL_MS = 1000;

const CameraPreview: React.FC<{ name: string; src: string }> = ({ name, src }) => {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  return (
    <div className="relative bg-gray-800 rounded-lg overflow-hidden aspect-[4/3]">
      {failed ? (
        <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
          <VideoOff className="w-5 h-5 mb-1" />
          <span className="text-[10px]">No preview</span>
        </div>
      ) : (
        <img
          src={src}
          alt={`${name} live feed`}
          className="w-full h-full object-cover"
          onError={() => setFailed(true)}
        />
      )}
      <span className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/60 text-sm text-gray-200">
        {name}
      </span>
    </div>
  );
};

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(s / 60);
  const secs = s % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

const Inference: React.FC = () => {
  const navigate = useNavigate();
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const [status, setStatus] = useState<InferenceStatus | null>(null);
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const [camerasOn, setCamerasOn] = useState(true);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const navigatedAwayRef = useRef(false);
  const stopRequestedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const stopIfHung = async () => {
      try {
        await stopInference(baseUrl, fetchWithHeaders);
      } catch {
        // The next status poll will surface the failure if it persists.
      }
    };
    const tick = async () => {
      try {
        const next = await getInferenceStatus(baseUrl, fetchWithHeaders);
        if (cancelled) return;
        setStatus(next);
        // Auto-bounce home once the run is done.
        if (!next.inference_active && !navigatedAwayRef.current) {
          navigatedAwayRef.current = true;
          if (next.exited) {
            const totalRuns = next.total_runs ?? 1;
            const runInfo = totalRuns > 1 ? ` (${totalRuns} runs completed)` : "";
            toast({
              title: "Inference finished",
              description:
                next.exit_code === 0
                  ? `Run completed${runInfo}.`
                  : `Exit code ${next.exit_code}. See ${next.log_path}.`,
              variant: next.exit_code === 0 ? "default" : "destructive",
            });
          }
          navigate("/");
          return;
        }
        // Safety net: only fire after the rollout *main loop* has actually
        // started (lerobot honours --duration there). Setup time — policy
        // load, snapshot_download, bus connect, camera connect — can take
        // 10–30s and must NOT count against the user's configured duration.
        if (
          next.inference_active &&
          next.rollout_started_at != null &&
          next.duration_s != null &&
          next.duration_s > 0 &&
          next.rollout_elapsed_s > next.duration_s + 10 &&
          !stopRequestedRef.current
        ) {
          stopRequestedRef.current = true;
          toast({
            title: "Inference seems hung",
            description: `Rollout past duration by ${Math.round(
              next.rollout_elapsed_s - next.duration_s,
            )}s. Stopping.`,
            variant: "destructive",
          });
          stopIfHung();
        }
      } catch (e) {
        if (!cancelled) {
          toast({
            title: "Lost connection to backend",
            description: e instanceof Error ? e.message : String(e),
            variant: "destructive",
          });
        }
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseUrl, fetchWithHeaders, navigate, toast]);

  const handleStop = async () => {
    setShowStopConfirm(false);
    try {
      await stopInference(baseUrl, fetchWithHeaders);
      // Status poll will catch the inactive state and navigate home.
    } catch (e) {
      toast({
        title: "Stop failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    }
  };

  const handleSnapshot = async () => {
    setSnapshotLoading(true);
    try {
      const result = await captureInferenceSnapshot(baseUrl, fetchWithHeaders);
      toast({
        title: "Snapshot captured",
        description: `Saved ${result.n_ticks} frames to ${result.snapshot_dir}`,
      });
    } catch (e) {
      toast({
        title: "Snapshot failed",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setSnapshotLoading(false);
    }
  };

  if (!status) {
    return (
      <div className="min-h-screen bg-black text-white flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin mr-3" /> Connecting to inference…
      </div>
    );
  }

  const setupElapsed = status.elapsed_s ?? 0;
  const rolloutElapsed = status.rollout_elapsed_s ?? 0;
  const duration = status.duration_s ?? 0;
  const isSettingUp = status.inference_active && status.rollout_started_at == null;
  const isRunning = status.inference_active && status.rollout_started_at != null;
  // When setting up: progress is uncertain — show a soft pulsing bar.
  // When rolling out: progress is rolloutElapsed / duration.
  const pct =
    isRunning && duration > 0
      ? Math.min(100, (rolloutElapsed / duration) * 100)
      : 0;
  const pillLabel = isSettingUp
    ? "SETTING UP"
    : isRunning
    ? "RUNNING"
    : "FINISHED";
  const timerSeconds = isRunning ? rolloutElapsed : setupElapsed;

  // The backend serves exactly the cameras this run was started with (keys
  // match the preview files the rollout subprocess writes).
  const cameraFeeds = (status.camera_names ?? []).map((name) => ({
    key: name,
    name,
  }));

  return (
    <div className="min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8">
      <div className="flex items-center gap-4 mb-8">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => navigate("/")}
          className="text-slate-400 hover:bg-slate-800 hover:text-white rounded-lg"
        >
          <ArrowLeft className="w-5 h-5" />
        </Button>
        <Logo />
        <h1 className="font-bold text-white text-2xl">Inference</h1>
      </div>

      <div className="flex-1 flex items-start justify-center gap-6">
        <div className="flex-1 max-w-xl">
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-8">
            <div className="text-center mb-6">
            <div
              className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold tracking-widest ${
                isSettingUp
                  ? "bg-amber-500/15 text-amber-300"
                  : "bg-green-500/15 text-green-300"
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isSettingUp ? "bg-amber-500" : "bg-green-500"
                } animate-pulse`}
              />
              {pillLabel}
            </div>
          </div>

          <div className="text-center mb-4">
            <div
              className={`text-7xl font-mono font-bold leading-none ${
                isSettingUp ? "text-amber-400" : "text-green-400"
              }`}
            >
              {formatTime(timerSeconds)}
            </div>
            <div className="text-sm text-gray-500 mt-2">
              {isSettingUp
                ? "Loading policy & connecting hardware…"
                : `/ ${formatTime(duration)}`}
            </div>
          </div>

          <div className="w-full bg-gray-800 rounded-full h-1.5 mb-8">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${
                isSettingUp
                  ? "bg-amber-500/40 animate-pulse w-full"
                  : "bg-green-500"
              }`}
              style={isSettingUp ? undefined : { width: `${pct}%` }}
            />
          </div>

          <div className="text-xs text-slate-500 break-all mb-6">
            policy: {status.policy_ref ?? "(unknown)"}
          </div>

          {(status.total_runs ?? 0) > 1 && (
            <div className="text-center mb-4">
              <span className="text-sm text-gray-400">
                Run {status.current_run ?? 1} / {status.total_runs}
              </span>
            </div>
          )}

          <Button
            onClick={() => setShowStopConfirm(true)}
            disabled={!status.inference_active}
            className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-6 text-lg disabled:opacity-50"
          >
            <Square className="w-5 h-5 mr-2" />
            Stop
          </Button>

          <Button
            onClick={handleSnapshot}
            disabled={!isRunning || snapshotLoading}
            variant="outline"
            className="w-full mt-3 border-amber-500/50 text-amber-400 hover:bg-amber-500/10 hover:text-amber-300 disabled:opacity-50"
          >
            {snapshotLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Camera className="w-4 h-4 mr-2" />
            )}
            标记失败 (Snapshot)
          </Button>
          </div>
        </div>

        {cameraFeeds.length > 0 && (
          <div className="w-72 flex-shrink-0 bg-gray-900 rounded-lg border border-gray-700 p-4 flex flex-col gap-3 sticky top-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Video className="w-4 h-4 text-gray-400" />
                <h2 className="text-sm font-medium text-gray-200">Cameras</h2>
              </div>
              <div className="flex items-center gap-2">
                <Label htmlFor="inference-cam-toggle" className="text-xs text-gray-400 cursor-pointer">
                  {camerasOn ? "On" : "Off"}
                </Label>
                <Switch
                  id="inference-cam-toggle"
                  checked={camerasOn}
                  onCheckedChange={setCamerasOn}
                />
              </div>
            </div>
            {camerasOn ? (
              <div className="flex flex-col gap-2 overflow-y-auto">
                {cameraFeeds.map((feed) => (
                  <CameraPreview
                    key={feed.key}
                    name={feed.name}
                    src={`${baseUrl}/inference-camera-feed/${encodeURIComponent(feed.name)}`}
                  />
                ))}
              </div>
            ) : (
              <p className="text-xs text-gray-500">
                Turn on to preview cameras during inference.
              </p>
            )}
          </div>
        )}
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent className="bg-gray-900 border-gray-700 text-white">
          <AlertDialogHeader>
            <AlertDialogTitle>Stop inference?</AlertDialogTitle>
            <AlertDialogDescription className="text-gray-400">
              The follower will hold its current pose. You can launch another
              run from the job tile.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-gray-800 border-gray-700 text-white hover:bg-gray-700">
              Keep running
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={handleStop}
              className="bg-red-500 hover:bg-red-600 text-white"
            >
              Stop
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default Inference;
