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
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertTriangle, CheckCircle, Loader2, Play, VideoOff } from "lucide-react";
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
import CheckpointDropdown from "@/components/jobs/CheckpointDropdown";
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
}

const DEFAULT_FPS = 30;
const CUSTOM_TASK = "__custom__";

const CAPTURE_MODES: { value: CaptureMode; label: string }[] = [
  { value: "base", label: "纯执行（不录数据）" },
  { value: "dagger", label: "纠正采集（主臂接管）" },
  { value: "highlight", label: "抢救片段（含最近 15 秒）" },
  { value: "sentry", label: "全自动录制" },
];

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
}) => {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [checkpoints, setCheckpoints] = useState<JobCheckpoint[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(initialStep);
  const [task, setTask] = useState("");
  const [taskIsCustom, setTaskIsCustom] = useState(false);
  const [durationS, setDurationS] = useState(60);
  const [nRuns, setNRuns] = useState(1);
  const [chunkSmooth, setChunkSmooth] = useState(false);
  const [smoothStrength, setSmoothStrength] = useState(50);
  const [rtc, setRtc] = useState(false);
  // Data collection during rollout. 'base' just runs the policy; the other
  // modes are lerobot rollout strategies that write episodes as it drives.
  const [captureMode, setCaptureMode] = useState<CaptureMode>("base");
  const [datasetName, setDatasetName] = useState("");
  const [numEpisodes, setNumEpisodes] = useState(20);
  const [recordAutonomous, setRecordAutonomous] = useState(false);
  const [autoAdvance, setAutoAdvance] = useState(false);
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

  // Load checkpoints when modal opens.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    listJobCheckpoints(baseUrl, fetchWithHeaders, jobId)
      .then((cks) => {
        if (cancelled) return;
        setCheckpoints(cks);
        if (cks.length > 0) {
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
  }, [open, baseUrl, fetchWithHeaders, jobId]);


  // Load policy config when step changes.
  useEffect(() => {
    if (!open || selectedStep == null) {
      setPolicyConfig(null);
      setPolicyConfigError(null);
      return;
    }
    let cancelled = false;
    setPolicyConfigLoading(true);
    setPolicyConfigError(null);
    getCheckpointPolicyConfig(baseUrl, fetchWithHeaders, jobId, selectedStep)
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
  }, [open, baseUrl, fetchWithHeaders, jobId, selectedStep]);

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
    selectedStep != null
      ? checkpoints.find((c) => c.step === selectedStep)?.ref ?? null
      : null;

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
        cameraDict[name] = {
          type: "opencv",
          camera_index: binding,
          width: dims.width,
          height: dims.height,
          fps: DEFAULT_FPS,
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
        rtc,
        capture_mode: captureMode,
        dataset_repo_id: datasetName,
        num_episodes: numEpisodes,
        record_autonomous: recordAutonomous,
        auto_advance: autoAdvance,
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
            ) : (
              <CheckpointDropdown
                checkpoints={checkpoints}
                selectedStep={selectedStep}
                onChange={setSelectedStep}
              />
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
                Max duration (seconds)
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
                Number of runs
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
              <p className="text-xs text-gray-500">
                Run inference consecutively this many times. Each run uses the
                same checkpoint and settings.
              </p>
            </div>
            <div className="space-y-3 rounded-lg border border-gray-700 bg-gray-800/40 p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <Label htmlFor="rtc" className="text-sm font-medium text-gray-200">
                    RTC 平滑执行（Real-Time Chunking）
                  </Label>
                  <p className="text-xs text-gray-500">
                    Official smooth execution for slow VLA models: replans every 10
                    steps and blends chunks inside the policy. Recommended for
                    SmolVLA; mutually exclusive with chunk smoothing below.
                  </p>
                </div>
                <Switch
                  id="rtc"
                  checked={rtc}
                  onCheckedChange={(v) => {
                    setRtc(v);
                    if (v) setChunkSmooth(false);
                  }}
                />
              </div>
              <div className="flex items-center justify-between gap-4">
                <div>
                  <Label htmlFor="chunk-smooth" className="text-sm font-medium text-gray-200">
                    Chunk smoothing（Chunk 间平滑）
                  </Label>
                  <p className="text-xs text-gray-500">
                    Low-pass filters the executed actions so the jump between
                    two action chunks becomes a short ramp. Adds a little lag;
                    the gripper channel is never filtered.
                  </p>
                </div>
                <Switch
                  id="chunk-smooth"
                  checked={chunkSmooth}
                  onCheckedChange={(v) => {
                    setChunkSmooth(v);
                    if (v) setRtc(false);
                  }}
                />
              </div>
              {chunkSmooth && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-gray-400">
                    <span>Strength（强度）</span>
                    <span>{smoothStrength}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={95}
                    step={5}
                    value={smoothStrength}
                    onChange={(e) => setSmoothStrength(Number(e.target.value))}
                    className="w-full accent-green-500"
                  />
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white border-b border-gray-700 pb-2">
              数据采集
            </h3>
            <div className="space-y-2">
              <Label className="text-sm font-medium text-gray-300">模式</Label>
              <Select
                value={captureMode}
                onValueChange={(v) => setCaptureMode(v as CaptureMode)}
              >
                <SelectTrigger className="bg-gray-800 border-gray-700 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-900 border-gray-700 text-white">
                  {CAPTURE_MODES.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
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
                  <p className="text-xs text-gray-500">
                    新建 rollout_* 集，不写遥操录的数据集；同一天多轮落进同一个集。
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="capture-episodes" className="text-sm font-medium text-gray-300">
                    纠正段数上限
                  </Label>
                  <NumberInput
                    id="capture-episodes"
                    min={1}
                    value={numEpisodes}
                    onChange={(v) => {
                      if (v !== undefined) setNumEpisodes(v);
                    }}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                </div>
                {captureMode === "dagger" && (
                  <>
                    <div className="flex items-center justify-between gap-4">
                      <Label htmlFor="hybrid-episode" className="text-sm font-medium text-gray-200">
                        混合集（策略段一起录进同一集）
                      </Label>
                      <Switch
                        id="hybrid-episode"
                        checked={recordAutonomous}
                        onCheckedChange={setRecordAutonomous}
                      />
                    </div>
                    {!recordAutonomous && (
                      <div className="flex items-center justify-between gap-4">
                        <Label htmlFor="auto-advance" className="text-sm font-medium text-gray-200">
                          纠正保存后自动开下一轮
                        </Label>
                        <Switch
                          id="auto-advance"
                          checked={autoAdvance}
                          onCheckedChange={setAutoAdvance}
                        />
                      </div>
                    )}
                    {(autoAdvance || recordAutonomous) && nRuns <= 1 && (
                      <p className="text-xs text-amber-400">
                        轮数需调到 2 以上，一轮结束后才会自动接下一轮。
                      </p>
                    )}
                  </>
                )}
              </>
            )}
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
