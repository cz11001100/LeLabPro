import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, CheckCircle, ChevronsUpDown, Loader2, Play, VideoOff } from "lucide-react";
import { RobotRecord } from "@/hooks/useRobots";
import { useApi } from "@/contexts/ApiContext";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import {
  JobCheckpoint,
  PolicyConfigSummary,
  getCheckpointPolicyConfig,
  listJobCheckpoints,
} from "@/lib/checkpointsApi";
import { CaptureMode, startInference } from "@/lib/inferenceApi";
import {
  RolloutDatasetInfo,
  RolloutRealWorld,
  getLatestRealWorld,
  listRolloutDatasets,
} from "@/lib/rolloutEvalApi";
import ResumeDatasetPicker from "@/components/landing/ResumeDatasetPicker";
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
import { ModelVariant } from "@/lib/modelGroups";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { useCameraStream } from "@/hooks/useCameraStream";

const CameraThumbnail: React.FC<{ deviceId: string; paused: boolean }> = ({
  deviceId,
  paused,
}) => {
  const { videoRef, hasError } = useCameraStream(deviceId, paused);
  if (paused || hasError || !deviceId) {
    return (
      <div className="w-32 h-24 bg-gray-800 rounded border border-gray-700 flex flex-col items-center justify-center">
        <VideoOff className="w-5 h-5 text-gray-500 mb-1" />
        <span className="text-[10px] text-gray-500">
          {paused ? "Released" : "No preview"}
        </span>
      </div>
    );
  }
  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      className="w-32 h-24 object-cover rounded border border-gray-700 bg-black"
    />
  );
};

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  robot: RobotRecord | null;
  jobId: string;
  initialStep: number | null;
  // Imported run collapsed into one entry: sibling checkpoints, each owned by a
  // different job. When present, the Checkpoint dropdown switches the active job
  // instead of listing one job's internal steps.
  variants?: ModelVariant[];
}

const DEFAULT_FPS = 30;
const CUSTOM_TASK = "__custom__";

// One entry per采集预设. 'mode' + 'merge' is what the backend actually takes,
// so new modes only need a row here.
type CapturePreset = "base" | "rabc" | "dagger" | "highlight" | "sentry";

const CAPTURE_PRESETS: {
  value: CapturePreset;
  label: string;
  mode: CaptureMode;
  merge: boolean;
}[] = [
  { value: "base", label: "纯执行（不录数据）", mode: "base", merge: false },
  { value: "rabc", label: "RA-BC 采集（纠正 + 混合集）", mode: "dagger", merge: true },
  { value: "dagger", label: "纯纠正采集（只存纠正段）", mode: "dagger", merge: false },
  { value: "highlight", label: "抢救片段（含最近 15 秒）", mode: "highlight", merge: false },
  { value: "sentry", label: "全自动录制", mode: "sentry", merge: false },
];

const presetOf = (mode: CaptureMode, merge: boolean): CapturePreset =>
  mode === "dagger" ? (merge ? "rabc" : "dagger") : (mode as CapturePreset);

// Mirrors the backend's day-stable auto name so the placeholder tells the truth.
const autoDatasetTail = () => {
  const d = new Date();
  return `${String(d.getMonth() + 1).padStart(2, "0")}${String(d.getDate()).padStart(2, "0")}`;
};
// Prefix on a camera-binding value that carries a stream URL instead of a cv2 index.
const URL_BINDING = "url:";

const InferenceModal: React.FC<Props> = ({
  open,
  onOpenChange,
  robot,
  jobId,
  initialStep,
  variants,
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const usingVariants = !!variants && variants.length > 0;
  // For an imported run, the active job is the sibling that owns the selected
  // checkpoint; otherwise it is the single jobId handed in.
  const [activeJobId, setActiveJobId] = useState(jobId);
  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [task, setTask] = useState("");
  const [taskIsCustom, setTaskIsCustom] = useState(false);
  const [durationS, setDurationS] = useState(60);
  const [nRuns, setNRuns] = useState(1);
  // Both smoothers are 0=off sliders. RTC guidance (max_guidance_weight) and
  // the chunk low-pass are mutually exclusive.
  const [smoothStrength, setSmoothStrength] = useState(0);
  const [rtcGuidance, setRtcGuidance] = useState(0);
  const chunkSmooth = smoothStrength > 0;
  // Data collection during rollout. 'base' just runs the policy; the other
  // modes are lerobot rollout strategies that write episodes as it drives.
  const [captureMode, setCaptureMode] = useState<CaptureMode>("base");
  const [lastRealWorld, setLastRealWorld] = useState<RolloutRealWorld | null>(null);
  const [datasetName, setDatasetName] = useState("");
  const [maxCorrection, setMaxCorrection] = useState(30);
  const [recordAutonomous, setRecordAutonomous] = useState(false);
  // Existing capture datasets, so the operator can keep appending to one the
  // same way the recording page's 续录 does instead of retyping its name.
  const [captureDatasets, setCaptureDatasets] = useState<RolloutDatasetInfo[]>([]);
  const [captureDatasetsLoading, setCaptureDatasetsLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [policyConfig, setPolicyConfig] = useState<PolicyConfigSummary | null>(null);
  const [policyConfigLoading, setPolicyConfigLoading] = useState(false);
  const [policyConfigError, setPolicyConfigError] = useState<string | null>(null);

  // Per expected camera name → the source to read it from: a cv2 index for a USB
  // camera, or a stream URL for a network one (null = not bound yet).
  const [cameraBindings, setCameraBindings] = useState<
    Record<string, number | string | null>
  >({});
  const { cameras: availableCameras } = useAvailableCameras({ enabled: open });
  // The robot's own network cameras are the only stream sources the modal can
  // offer; USB cameras come from the live cv2 enumeration instead.
  const networkCameras = (robot?.cameras ?? []).filter(
    (c) => c.type === "network" || !!c.url,
  );

  // Re-seed the active job when the incoming jobId changes (a fresh selection).
  useEffect(() => {
    setActiveJobId(jobId);
    setSelectedStep(initialStep);
  }, [jobId, initialStep]);

  // Existing capture datasets, fetched once per modal opening that shows them.
  useEffect(() => {
    if (!open || captureMode === "base" || captureDatasets.length > 0) return;
    let cancelled = false;
    setCaptureDatasetsLoading(true);
    listRolloutDatasets(baseUrl, fetchWithHeaders)
      .then((list) => !cancelled && setCaptureDatasets(list))
      .catch(() => !cancelled && setCaptureDatasets([]))
      .finally(() => !cancelled && setCaptureDatasetsLoading(false));
    return () => {
      cancelled = true;
    };
  }, [open, captureMode, captureDatasets.length, baseUrl, fetchWithHeaders]);

  // Load checkpoints for the active job when the modal opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listJobCheckpoints(baseUrl, fetchWithHeaders, activeJobId)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        // Real jobs pick among their own steps; variant runs keep the display
        // step from the name and only need the member's internal checkpoint.
        if (!usingVariants && cks.length > 0) {
          const latest = cks[cks.length - 1].step;
          setSelectedStep((prev) => (prev != null ? prev : latest));
        }
      })
      .catch(() => {
        if (cancelled) return;
        setCheckpoints([]);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, activeJobId, usingVariants]);


  // The step actually sent to the backend: for a variant run the member job
  // exposes a single checkpoint (its own latest); otherwise the selected step.
  const internalStep = usingVariants
    ? (checkpoints.length > 0 ? checkpoints[checkpoints.length - 1].step : null)
    : selectedStep;

  // Load policy config when step changes.
  useEffect(() => {
    if (!open || internalStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, activeJobId, internalStep)
      .then((cfg) => {
        if (cancelled) return;
        setPolicyConfig(cfg);
        // Language-conditioned policies embed the task string verbatim, so a
        // reworded prompt is out-of-distribution: default to the first exact
        // string the dataset was recorded with.
        const recorded = cfg.tasks ?? [];
        if (recorded.length > 0) {
          setTask((prev) => (recorded.includes(prev) ? prev : recorded[0]));
        }
        // Reset camera bindings to one entry per expected camera name.
        // Preserve any prior selection that's still relevant.
        setCameraBindings((prev) => {
          const next: Record<string, number | string | null> = {};
          for (const name of Object.keys(cfg.image_features)) {
            next[name] = prev[name] ?? null;
          }
          return next;
        });
      })
      .catch((e) => {
        if (cancelled) return;
        setPolicyConfig(null);
        setPolicyConfigError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setPolicyConfigLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, baseUrl, fetchWithHeaders, activeJobId, internalStep]);

  // If the selected robot has cameras whose names match a policy-expected
  // camera, auto-bind them. A saved stream URL binds directly; for a USB camera
  // prefer matching by browser device_id (stable across cv2 index drift) and
  // fall back to the saved camera_index.
  useEffect(() => {
    if (!policyConfig) return;
    const robotCams = robot?.cameras ?? [];
    if (robotCams.length === 0) return;
    setCameraBindings((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const policyName of Object.keys(policyConfig.image_features)) {
        if (next[policyName] != null) continue;
        const robotCam = robotCams.find(
          (c) => c.name.toLowerCase() === policyName.toLowerCase(),
        );
        if (!robotCam) continue;
        if (robotCam.type === "network" || robotCam.url) {
          next[policyName] = robotCam.url!;
          changed = true;
          continue;
        }
        const live =
          (robotCam.device_id &&
            availableCameras.find((c) => c.deviceId === robotCam.device_id)) ||
          availableCameras.find((c) => c.index === robotCam.camera_index);
        if (live) {
          next[policyName] = live.index;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [policyConfig, robot, availableCameras]);

  const selectedRef =
    internalStep != null
      ? checkpoints.find((c) => c.step === internalStep)?.ref ?? null
      : null;

  // What this checkpoint last did on the real arm, so the collection decision is
  // made with that number in front of you instead of from memory.
  useEffect(() => {
    if (selectedRef == null) {
      setLastRealWorld(null);
      return;
    }
    let cancelled = false;
    getLatestRealWorld(baseUrl, fetchWithHeaders, String(selectedRef))
      .then((res) => {
        if (!cancelled) setLastRealWorld(res);
      })
      .catch(() => {
        if (!cancelled) setLastRealWorld(null);
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, selectedRef]);

  const expectedCameraNames = policyConfig
    ? Object.keys(policyConfig.image_features)
    : [];
  const allCamerasBound = expectedCameraNames.every(
    (name) => cameraBindings[name] != null,
  );

  const canStart =
    !!robot &&
    robot.is_clean &&
    selectedRef != null &&
    !!policyConfig &&
    allCamerasBound &&
    (!policyConfig.requires_task || task.trim().length > 0) &&
    !submitting;

  const handleStart = async () => {
    if (!robot || selectedRef == null || !policyConfig) return;
    // Setting submitting=true makes every CameraPreview drop its
    // browser stream — required so the rollout subprocess can open the
    // same camera index via OpenCV without colliding on the device.
    setSubmitting(true);
    await new Promise((r) => setTimeout(r, 300));
    const cameraDict: Record<string, {
      type: string; camera_index?: number; url?: string; width: number; height: number; fps?: number;
      exposure?: number; gain?: number;
    }> = {};
    for (const [name, dims] of Object.entries(policyConfig.image_features)) {
      const binding = cameraBindings[name];
      if (binding == null) continue;
      if (typeof binding === "string") {
        // A stream source sends the size it was configured for; the policy's own
        // transform resizes it, so keep the saved probe values rather than
        // pretending it can serve the checkpoint's resolution.
        const saved = robot.cameras.find(
          (c) => c.name.toLowerCase() === name.toLowerCase(),
        );
        cameraDict[name] = {
          type: "network",
          url: binding,
          width: saved?.width ?? dims.width,
          height: saved?.height ?? dims.height,
          fps: saved?.fps ?? DEFAULT_FPS,
        };
      } else {
        // Exposure/gain are device-level; reuse what the robot record saved so
        // the policy sees the same image pipeline the dataset was recorded with.
        const saved = robot.cameras.find(
          (c) => c.name.toLowerCase() === name.toLowerCase(),
        );
        cameraDict[name] = {
          type: "opencv",
          camera_index: binding,
          width: dims.width,
          height: dims.height,
          fps: DEFAULT_FPS,
          ...(saved?.exposure !== undefined ? { exposure: saved.exposure } : {}),
          ...(saved?.gain !== undefined ? { gain: saved.gain } : {}),
        };
      }
    }
    try {
      await startInference(baseUrl, fetchWithHeaders, {
        follower_port: robot.follower_port,
        follower_config: robot.follower_config,
        policy_ref: selectedRef,
        task,
        cameras: cameraDict,
        duration_s: durationS,
        n_runs: nRuns,
        chunk_smooth: chunkSmooth,
        chunk_smooth_strength: smoothStrength / 100,
        rtc: rtcGuidance > 0,
        rtc_guidance: rtcGuidance,
        capture_mode: captureMode,
        dataset_repo_id: datasetName,
        num_episodes: nRuns,
        max_correction_s: maxCorrection,
        record_autonomous: recordAutonomous,
      });
      onOpenChange(false);
      navigate("/inference");
    } catch (e) {
      toast({
        title: "Couldn't start inference",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
      // Failure: bring the previews back so the user can adjust.
      setSubmitting(false);
    }
  };

  const onCameraBindingChange = (name: string, value: string) => {
    // A binding is either a cv2 index or a stream URL; both share one Select, so
    // the URL options carry a prefix that a numeric index can't collide with.
    const bound = value.startsWith(URL_BINDING)
      ? value.slice(URL_BINDING.length)
      : Number(value);
    setCameraBindings((prev) => ({ ...prev, [name]: bound }));
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-[600px] p-8 max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex justify-center items-center mb-4">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <Play className="w-4 h-4 text-white" />
            </div>
          </div>
          <DialogTitle className="text-white text-center text-2xl font-bold">
            Configure Inference
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          <DialogDescription className="text-gray-400 text-base leading-relaxed text-center">
            Pick a checkpoint and confirm hardware. The selected policy will
            drive the follower autonomously for the configured duration.
          </DialogDescription>

          <div className="grid grid-cols-2 gap-4">
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Robot Configuration
            </h3>
            {!robot ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Select and configure a robot on the Landing page first.
                </AlertDescription>
              </Alert>
            ) : !robot.is_clean ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>{robot.name}</strong> is missing a calibration.
                  Configure it before running inference.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-slate-200">
                  Running on <strong>{robot.name}</strong>
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Checkpoint
            </h3>
            {checkpoints.length === 0 ? (
              <Alert className="bg-amber-900/40 border-amber-700 text-amber-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  No checkpoints available for this job yet.
                </AlertDescription>
              </Alert>
            ) : usingVariants ? (
              <CheckpointDropdown
                checkpoints={variants!.map((v) => ({
                  step: v.step,
                  source: "local" as const,
                  ref: v.jobId,
                }))}
                selectedStep={selectedStep}
                onChange={(step) => {
                  const v = variants!.find((x) => x.step === step);
                  if (v) {
                    setActiveJobId(v.jobId);
                    setSelectedStep(step);
                  }
                }}
              />
            ) : (
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
              />
            )}
          </div>

          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              数据采集
            </h3>
            {lastRealWorld && (
              <p className="text-xs text-gray-400">
                上次真机：接管 {Math.round((lastRealWorld.takeover_rate ?? 0) * 100)}%（
                {lastRealWorld.corrected_takes}/{lastRealWorld.takes} 段）·{" "}
                {lastRealWorld.repo_id}
                {lastRealWorld.scored ? " · 已打分" : " · 未打分"}
              </p>
            )}
            <div className="space-y-2">
              <Label className="text-sm font-medium text-gray-300">模式</Label>
              <Select
                value={presetOf(captureMode, recordAutonomous)}
                onValueChange={(v) => {
                  const preset = CAPTURE_PRESETS.find((p) => p.value === v);
                  if (!preset) return;
                  setCaptureMode(preset.mode);
                  setRecordAutonomous(preset.merge);
                }}
              >
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white disabled:opacity-50">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  {CAPTURE_PRESETS.map((p) => (
                    <SelectItem key={p.value} value={p.value}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {captureMode !== "base" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="capture-dataset" className="text-sm font-medium text-gray-300">
                    纠正数据集
                  </Label>
                  <Input
                    id="capture-dataset"
                    value={datasetName}
                    onChange={(e) => setDatasetName(e.target.value)}
                    placeholder={`留空自动新建 rollout_${autoDatasetTail()}`}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                  <ResumeDatasetPicker
                    datasets={captureDatasets}
                    getRepoId={(d) => d.repo_id}
                    getDetail={(d) => `${d.episodes} 集`}
                    isSelectable={(d) => d.episodes > 0}
                    loading={captureDatasetsLoading}
                    onSelect={(d) => setDatasetName(d.repo_id)}
                    searchPlaceholder="选择要续录的采集数据集…"
                    groupHeading="已有采集数据集"
                    emptyHint="还没有可续录的 rollout_* 数据集"
                  >
                    <Button
                      type="button"
                      variant="outline"
                      className="w-full justify-between bg-gray-800 border-blue-600/50 text-blue-300 hover:bg-blue-900/20 hover:border-blue-500"
                    >
                      <span className="truncate flex items-center gap-2">
                        {captureDatasetsLoading ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Loading datasets…
                          </>
                        ) : (
                          "Select existing dataset to resume…"
                        )}
                      </span>
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </ResumeDatasetPicker>
                  <p className="text-xs text-gray-500">
                    新建 rollout_* 集，不写遥操录的数据集；同一天多轮落进同一个集。
                  </p>
                </div>
                {captureMode === "dagger" && (
                  <>
                    <div className="space-y-2">
                      <Label htmlFor="capture-correction" className="text-sm font-medium text-gray-300">
                        纠正段上限（秒）
                      </Label>
                      <NumberInput
                        id="capture-correction"
                        min={5}
                        value={maxCorrection}
                        onChange={(v) => {
                          if (v !== undefined) setMaxCorrection(v);
                        }}
                        className="bg-gray-800 border-gray-700 text-white"
                      />
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Run parameters
            </h3>
            {policyConfig?.requires_task ? (
              <div className="space-y-2">
                <Label htmlFor="task" className="text-sm font-medium text-gray-300">
                  Task description
                </Label>
                {(policyConfig.tasks ?? []).length > 0 ? (
                  <>
                    <Select
                      value={
                        taskIsCustom || !(policyConfig.tasks ?? []).includes(task)
                          ? CUSTOM_TASK
                          : task
                      }
                      onValueChange={(v) => {
                        if (v === CUSTOM_TASK) {
                          setTaskIsCustom(true);
                        } else {
                          setTaskIsCustom(false);
                          setTask(v);
                        }
                      }}
                    >
                      <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                        <SelectValue placeholder="Select a task" />
                      </SelectTrigger>
                      <SelectContent className="bg-gray-900 border-gray-700 text-white">
                        {(policyConfig.tasks ?? []).map((t) => (
                          <SelectItem key={t} value={t}>
                            {t}
                          </SelectItem>
                        ))}
                        <SelectItem value={CUSTOM_TASK}>Custom…</SelectItem>
                      </SelectContent>
                    </Select>
                    {taskIsCustom && (
                      <Input
                        id="task"
                        value={task}
                        onChange={(e) => setTask(e.target.value)}
                        placeholder="e.g., pick up the red block"
                        className="bg-gray-800 border-gray-700 text-white"
                      />
                    )}
                    <p className="text-xs text-gray-500">
                      {taskIsCustom
                        ? "A custom prompt was never seen in training; prefer a recorded task string."
                        : "Exact task strings from the training dataset."}
                    </p>
                  </>
                ) : (
                  <Input
                    id="task"
                    value={task}
                    onChange={(e) => setTask(e.target.value)}
                    placeholder="e.g., pick up the red block"
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                )}
                <p className="text-xs text-gray-500">
                  This policy is language-conditioned ({policyConfig.policy_type}).
                </p>
              </div>
            ) : null}
            <div className="space-y-2">
              <Label htmlFor="durationS" className="text-sm font-medium text-gray-300">
                {captureMode === "base"
                  ? "Max duration (seconds)"
                  : "Per-episode limit (seconds)"}
              </Label>
              <NumberInput
                id="durationS"
                min={1}
                value={durationS}
                onChange={(v) => {
                  if (v !== undefined) setDurationS(v);
                }}
                className="bg-gray-800 border-gray-700 text-white"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="nRuns" className="text-sm font-medium text-gray-300">
                {captureMode === "base" ? "Number of runs" : "推理次数（= 纠正集段数上限）"}
              </Label>
              <NumberInput
                id="nRuns"
                min={1}
                max={100}
                value={nRuns}
                onChange={(v) => {
                  if (v !== undefined) setNRuns(v);
                }}
                className="bg-gray-800 border-gray-700 text-white"
              />
              {captureMode === "base" && (
                <p className="text-xs text-gray-500">
                  Run inference consecutively this many times. Each run uses the
                  same checkpoint and settings.
                </p>
              )}
            </div>
            <div className="space-y-3 rounded-lg border border-gray-700 bg-gray-800/40 p-4">
              <div className="space-y-1">
                <div className="flex items-baseline justify-between gap-4">
                  <Label className="text-sm font-medium text-gray-200">
                    RTC 平滑执行（Real-Time Chunking）
                  </Label>
                  <span className="text-xs text-gray-400 whitespace-nowrap">
                    当前 {rtcGuidance}（0=关）
                  </span>
                </div>
                <p className="text-xs text-gray-500">
                  官方平滑执行：每执行视野步重规划并在策略内部混合动作块，适合
                  SmolVLA 等慢 VLA；数值为引导强度，越大越跟手。与下方 Chunk
                  间平滑互斥。
                </p>
                <input
                  type="range"
                  min={0}
                  max={30}
                  step={1}
                  value={rtcGuidance}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setRtcGuidance(v);
                    if (v > 0) setSmoothStrength(0);
                  }}
                  className="w-full accent-green-500"
                />
              </div>
              <div className="space-y-1">
                <div className="flex items-baseline justify-between gap-4">
                  <Label className="text-sm font-medium text-gray-200">
                    Chunk smoothing（Chunk 间平滑）
                  </Label>
                  <span className="text-xs text-gray-400 whitespace-nowrap">
                    当前 {smoothStrength}%（0=关）
                  </span>
                </div>
                <p className="text-xs text-gray-500">
                  对执行动作做低通滤波，两块动作切换的跳变变成短斜坡；会多一点滞后，
                  夹爪通道不过滤。与上方 RTC 互斥。
                </p>
                <input
                  type="range"
                  min={0}
                  max={95}
                  step={5}
                  value={smoothStrength}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setSmoothStrength(v);
                    if (v > 0) setRtcGuidance(0);
                  }}
                  className="w-full accent-green-500"
                />
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              Cameras
            </h3>
            {policyConfigLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 className="w-4 h-4 animate-spin" />
                Reading policy config…
              </div>
            ) : policyConfigError ? (
              <Alert className="bg-red-900/40 border-red-700 text-red-100">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  Couldn't load policy config: {policyConfigError}
                </AlertDescription>
              </Alert>
            ) : !policyConfig ? null : expectedCameraNames.length === 0 ? (
              <p className="text-xs text-gray-500">
                This policy doesn't use cameras.
              </p>
            ) : (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  为策略训练时的每个相机名选择来源（USB 摄像头或网络 RTSP/URL）。
                </p>
                {expectedCameraNames.map((name) => {
                  const dims = policyConfig.image_features[name];
                  const value = cameraBindings[name];
                  const bound =
                    typeof value === "number"
                      ? availableCameras.find((c) => c.index === value)
                      : undefined;
                  return (
                    <div key={name} className="flex items-center gap-3">
                      <div className="flex-1">
                        <Label className="text-sm font-medium text-gray-200">
                          {name}
                        </Label>
                        <p className="text-xs text-gray-500">
                          {dims.width}×{dims.height}
                        </p>
                      </div>
                      <Select
                        value={value != null ? String(value) : undefined}
                        onValueChange={(v) => onCameraBindingChange(name, v)}
                      >
                        <SelectTrigger className="bg-gray-800 border-gray-700 text-white w-56">
                          <SelectValue placeholder="Select a camera" />
                        </SelectTrigger>
                        <SelectContent className="bg-gray-900 border-gray-700 text-white">
                          {availableCameras.length === 0 &&
                          networkCameras.length === 0 ? (
                            <div className="px-2 py-1.5 text-xs text-gray-500">
                              No cameras detected
                            </div>
                          ) : (
                            <>
                              {availableCameras.map((cam) => (
                                <SelectItem
                                  key={cam.index}
                                  value={String(cam.index)}
                                >
                                  #{cam.index} — {cam.name}
                                </SelectItem>
                              ))}
                              {networkCameras.map((cam) => (
                                <SelectItem
                                  key={cam.url}
                                  value={`${URL_BINDING}${cam.url}`}
                                >
                                  <div className="flex flex-col">
                                    <span>{cam.name} · Network</span>
                                    <span className="text-xs text-gray-400 break-all">
                                      {cam.url}
                                    </span>
                                  </div>
                                </SelectItem>
                              ))}
                            </>
                          )}
                        </SelectContent>
                      </Select>
                      <CameraThumbnail deviceId={bound?.deviceId ?? ""} paused={submitting} />
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <Button
              onClick={handleStart}
              disabled={!canStart}
              className="w-full sm:w-auto bg-green-500 hover:bg-green-600 text-white px-10 py-6 text-lg disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-5 h-5 mr-2" />
              {submitting
                ? "Starting…"
                : captureMode === "base"
                  ? "Start Inference"
                  : "Start Collecting"}
            </Button>
            <Button
              onClick={() => onOpenChange(false)}
              variant="outline"
              className="w-full sm:w-auto border-gray-500 hover:border-gray-200 px-10 py-6 text-lg text-zinc-500 bg-zinc-900 hover:bg-zinc-800"
            >
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default InferenceModal;
