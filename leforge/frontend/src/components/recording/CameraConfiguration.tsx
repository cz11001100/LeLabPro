import React, { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Camera, Plus, X, VideoOff, RefreshCw, ChevronRight, RotateCw } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { useAvailableCameras } from "@/hooks/useAvailableCameras";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useNetworkCameraSource } from "@/hooks/useNetworkCameraSource";

// Sentinels distinguish "leave unset" (auto-detect / platform default) from an
// explicit choice. Radix Select disallows an empty-string value, so we map these
// to `undefined` on the CameraConfig.
const FOURCC_AUTO = "__auto__";
const BACKEND_DEFAULT = "__default__";
const NETWORK_SOURCE = "__network__";
const FOURCC_OPTIONS = ["MJPG", "YUYV", "I420", "NV12", "H264", "MP4V"];
// Mirrors lerobot's Cv2Backends enum names.
const BACKEND_OPTIONS = [
  "ANY",
  "V4L2",
  "DSHOW",
  "PVAPI",
  "ANDROID",
  "AVFOUNDATION",
  "MSMF",
];

// Canonical SO100/SO101 camera names. These match the keys SmolVLA checkpoints
// are trained with (lerobot's SmolVLA README uses `up` + `side`; `wrist` is the
// standard wrist-cam name), so picking from this list keeps the recorded
// dataset's camera keys — and therefore the orientation each name implies —
// consistent across recording, training and inference. A Custom escape hatch is
// still available for non-standard setups.
const PRESET_CAMERA_NAMES = ["side", "up", "wrist"];
const CUSTOM_NAME = "__custom__";

export interface CameraConfig {
  id: string;
  name: string;
  type: string;
  camera_index?: number; // cv2 index — what the recorder opens
  url?: string; // Stream source (rtsp/http); replaces camera_index when set
  device_id?: string; // Browser deviceId matched to the cv2 index by AVFoundation localizedName
  width: number;
  height: number;
  fps?: number;
  fourcc?: string; // 4-char OpenCV pixel format (e.g. "MJPG"); undefined = auto-detect
  backend?: string; // Cv2Backends name (e.g. "AVFOUNDATION"); undefined = platform default
  exposure?: number; // V4L2 exposure_time_absolute (100 µs steps); undefined = auto exposure
  gain?: number; // V4L2 gain; undefined = driver-controlled
  rotation?: 0 | 90 | 180 | 270; // Clockwise quarter-turn applied to a network stream before storage
}

interface CameraConfigurationProps {
  cameras: CameraConfig[];
  onCamerasChange: (cameras: CameraConfig[]) => void;
  releaseStreamsRef?: React.MutableRefObject<(() => void) | null>; // Ref to expose stream release function
  // When true, hide the add/remove/edit controls and show the configured
  // cameras as read-only live previews. Used in the recording dialog, where
  // cameras are managed on the Calibration page rather than added ad hoc.
  readOnly?: boolean;
}

const CameraConfiguration: React.FC<CameraConfigurationProps> = ({
  cameras,
  onCamerasChange,
  releaseStreamsRef,
  readOnly = false,
}) => {
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  // Exposure/gain are device-level V4L2 controls: the browser preview never
  // requests them, so typing a value only shows up once the backend writes it
  // to the device. A per-camera debounce keeps typing "100" from firing a
  // v4l2-ctl call per keystroke.
  const [controlErrors, setControlErrors] = useState<Record<string, string>>({});
  const camerasRef = React.useRef(cameras);
  camerasRef.current = cameras;
  const controlTimersRef = React.useRef<
    Record<string, ReturnType<typeof setTimeout>>
  >({});

  const applyControlsSoon = (cameraId: string) => {
    const pending = controlTimersRef.current[cameraId];
    if (pending) clearTimeout(pending);
    controlTimersRef.current[cameraId] = setTimeout(async () => {
      delete controlTimersRef.current[cameraId];
      const cam = camerasRef.current.find((c) => c.id === cameraId);
      if (!cam || cam.type === "network" || cam.camera_index === undefined) {
        return;
      }
      const setError = (message: string | null) =>
        setControlErrors((prev) => {
          if (message === null) {
            if (!(cameraId in prev)) return prev;
            const next = { ...prev };
            delete next[cameraId];
            return next;
          }
          return { ...prev, [cameraId]: message };
        });
      try {
        const res = await fetchWithHeaders(
          `${baseUrl}/cameras/${cam.camera_index}/controls`,
          {
            method: "POST",
            body: JSON.stringify({
              exposure: cam.exposure ?? null,
              gain: cam.gain ?? null,
            }),
          }
        );
        const data = await res.json().catch(() => ({}));
        const failed = data?.failed as Record<string, string> | undefined;
        if (!res.ok || (failed && Object.keys(failed).length > 0)) {
          setError(
            failed && Object.keys(failed).length > 0
              ? Object.values(failed).join("; ")
              : data?.message ?? `Apply failed (${res.status})`
          );
        } else {
          setError(null);
        }
      } catch {
        setError("Backend unreachable");
      }
    }, 400);
  };

  // Drop pending applies when the panel unmounts.
  React.useEffect(() => {
    const timers = controlTimersRef.current;
    return () => {
      Object.values(timers).forEach(clearTimeout);
    };
  }, []);

  const {
    cameras: availableCameras,
    isLoading: isLoadingCameras,
    refresh: refreshCameras,
  } = useAvailableCameras();
  const [selectedCameraIndex, setSelectedCameraIndex] = useState<string>("");
  const [cameraName, setCameraName] = useState("");
  const [cameraUrl, setCameraUrl] = useState("");
  const [pickerRotation, setPickerRotation] = useState<0 | 90 | 180 | 270>(0);
  // Which entry the name dropdown shows: a preset name, CUSTOM_NAME (reveal the
  // free-text input), or "" (nothing picked yet).
  const [nameMode, setNameMode] = useState<string>("");

  const isNetworkSource = selectedCameraIndex === NETWORK_SOURCE;
  // The backend has to open the stream before this camera can be added: an
  // IP camera refuses a resolution it wasn't configured to encode, and the size
  // and frame rate it does send become the dataset's image features.
  const {
    probe: streamProbe,
    displayWidth: streamWidth,
    displayHeight: streamHeight,
    error: streamError,
    isChecking: streamChecking,
    feedUrl: streamFeedUrl,
  } = useNetworkCameraSource(
    isNetworkSource ? cameraUrl : undefined,
    pickerRotation,
  );

  // Names already taken by configured cameras — presets matching these are
  // disabled in the dropdown so the same orientation name can't be reused.
  const usedNames = new Set(cameras.map((c) => c.name.trim().toLowerCase()));

  const handleNameModeChange = (value: string) => {
    setNameMode(value);
    if (value === CUSTOM_NAME) {
      setCameraName("");
    } else {
      setCameraName(value);
    }
  };

  // cv2's AVFoundation order is uniqueID-sorted, so plugging/unplugging a
  // device between sessions shifts indices. The browser device_id stays
  // stable per-origin, so use it to refresh each seeded camera's
  // camera_index — otherwise the recorder opens the wrong physical device
  // and the dropdown's "already added" check guards a stale index.
  // Also backfill device_id for cameras that were saved without one (e.g.
  // when browser permission was denied during initial setup).
  useEffect(() => {
    if (availableCameras.length === 0 || cameras.length === 0) return;
    let changed = false;
    const refreshed = cameras.map((cam) => {
      // If we have a device_id, use it to refresh the camera_index
      if (cam.device_id) {
        const match = availableCameras.find((m) => m.deviceId === cam.device_id);
        if (match && match.index !== cam.camera_index) {
          changed = true;
          return { ...cam, camera_index: match.index };
        }
        return cam;
      }
      // If no device_id but we have a camera_index, backfill the device_id
      if (cam.camera_index !== undefined) {
        const matchByIndex = availableCameras.find((m) => m.index === cam.camera_index);
        if (matchByIndex && matchByIndex.deviceId) {
          changed = true;
          return { ...cam, device_id: matchByIndex.deviceId };
        }
      }
      return cam;
    });
    if (changed) onCamerasChange(refreshed);
    // We deliberately don't depend on `cameras`/`onCamerasChange` to avoid
    // re-running every keystroke in the camera-name input — re-syncing only
    // when the available-cameras list itself changes is sufficient.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableCameras]);

  const trimmedName = () => cameraName.trim();

  const nameTaken = (candidate: string) =>
    cameras.some((cam) => cam.name.trim().toLowerCase() === candidate.toLowerCase());

  const commitCamera = (newCamera: CameraConfig) => {
    onCamerasChange([...cameras, newCamera]);
    setSelectedCameraIndex("");
    setCameraName("");
    setCameraUrl("");
    setPickerRotation(0);
    setNameMode("");
    toast({
      title: "摄像头已添加",
      description: `${newCamera.name} 已加入相机配置。`,
    });
  };

  const addNetworkCamera = () => {
    if (!streamProbe) return;
    const url = cameraUrl.trim();
    if (cameras.some((cam) => (cam.url || "").trim().toLowerCase() === url.toLowerCase())) {
      toast({
        title: "相机重复",
        description: "这个 URL 已经在配置里了。",
        variant: "destructive",
      });
      return;
    }
    commitCamera({
      id: `camera_${Date.now()}`,
      name: trimmedName(),
      type: "network",
      url,
      width: streamProbe.width,
      height: streamProbe.height,
      fps: streamProbe.fps ?? 30,
      rotation: pickerRotation,
    });
  };

  const addCamera = () => {
    if (!selectedCameraIndex || !trimmedName()) {
      toast({
        title: "信息不完整",
        description: "请选择相机来源并填写名称。",
        variant: "destructive",
      });
      return;
    }

    if (nameTaken(trimmedName())) {
      toast({
        title: "名称重复",
        description: "每个相机需要唯一的名称，它会作为数据集与策略的图像键。",
        variant: "destructive",
      });
      return;
    }

    if (isNetworkSource) {
      addNetworkCamera();
      return;
    }

    const cameraIndex = parseInt(selectedCameraIndex);
    const selectedCamera = availableCameras.find(
      (cam) => cam.index === cameraIndex
    );

    if (!selectedCamera) {
      toast({
        title: "相机无效",
        description: "选中的相机当前不可用。",
        variant: "destructive",
      });
      return;
    }

    // Block duplicates by either cv2 index or browser deviceId — a stale
    // camera_index in a seeded camera can otherwise let the same physical
    // device sneak in under a different index.
    const isDuplicate = cameras.some(
      (cam) =>
        cam.camera_index === selectedCamera.index ||
        (selectedCamera.deviceId && cam.device_id === selectedCamera.deviceId),
    );
    if (isDuplicate) {
      toast({
        title: "相机已添加",
        description: "这个相机已在配置中。",
        variant: "destructive",
      });
      return;
    }

    commitCamera({
      id: `camera_${Date.now()}`,
      name: trimmedName(),
      type: "opencv",
      camera_index: selectedCamera.index,
      device_id: selectedCamera.deviceId,
      width: 640,
      height: 480,
      fps: 30,
    });
  };

  const removeCamera = (cameraId: string) => {
    onCamerasChange(cameras.filter((cam) => cam.id !== cameraId));
    toast({
      title: "Camera Removed",
      description: "Camera has been removed from the configuration.",
    });
  };

  const updateCamera = (cameraId: string, updates: Partial<CameraConfig>) => {
    onCamerasChange(
      cameras.map((cam) =>
        cam.id === cameraId ? { ...cam, ...updates } : cam
      )
    );
    // Changing either control should show up in the live preview right away,
    // not only once the next recording starts.
    if ("exposure" in updates || "gain" in updates) {
      applyControlsSoon(cameraId);
    }
  };

  // When the recording session is starting, the parent calls
  // releaseStreamsRef.current() to make every CameraPreview drop its browser
  // stream so cv2.VideoCapture can grab the camera exclusively.
  const [streamsPaused, setStreamsPaused] = useState(false);
  const releaseAllCameraStreams = useCallback(() => {
    setStreamsPaused(true);
  }, []);

  useEffect(() => {
    if (releaseStreamsRef) {
      releaseStreamsRef.current = releaseAllCameraStreams;
    }
  }, [releaseStreamsRef, releaseAllCameraStreams]);


  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-slate-900 border-b border-sky-300 pb-2">
        {readOnly ? "Cameras" : "Camera Configuration"}
      </h3>

      {/* Add Camera Section — hidden in read-only mode (cameras are managed on
          the Calibration page, not added here). */}
      {!readOnly && (
      <div className="bg-sky-50/60 rounded-lg p-4 space-y-4">
        <h4 className="text-md font-medium text-slate-700">Add Camera</h4>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium text-slate-700">
                Available Cameras
              </Label>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => refreshCameras()}
                disabled={isLoadingCameras}
                className="h-6 w-6 text-slate-600 hover:text-slate-900"
                title="Rescan for cameras (e.g. after plugging in a new USB camera)"
                aria-label="Rescan for cameras"
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 ${isLoadingCameras ? "animate-spin" : ""}`}
                />
              </Button>
            </div>
            <Select
              value={selectedCameraIndex}
              onValueChange={setSelectedCameraIndex}
              disabled={isLoadingCameras}
            >
              <SelectTrigger className="bg-sky-50 border-sky-300 text-slate-900">
                <SelectValue
                  placeholder={
                    isLoadingCameras ? "Loading cameras..." : "Select camera"
                  }
                />
              </SelectTrigger>
              <SelectContent className="bg-sky-50 border-sky-300">
                {availableCameras.map((camera) => {
                  const alreadyAdded = cameras.some(
                    (cam) =>
                      cam.camera_index === camera.index ||
                      (camera.deviceId && cam.device_id === camera.deviceId),
                  );
                  return (
                    <SelectItem
                      key={camera.index}
                      value={camera.index.toString()}
                      className="text-slate-900 hover:bg-sky-200"
                      disabled={!camera.available || alreadyAdded}
                    >
                      <div className="flex flex-col">
                        <span className="font-medium">{camera.name}</span>
                        <span className="text-xs text-slate-600">
                          Index {camera.index}
                          {alreadyAdded && " · already added"}
                        </span>
                      </div>
                    </SelectItem>
                  );
                })}
                <SelectItem
                  value={NETWORK_SOURCE}
                  className="text-slate-900 hover:bg-sky-200"
                >
                  <div className="flex flex-col">
                    <span className="font-medium">Network Camera</span>
                    <span className="text-xs text-slate-600">RTSP / URL</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
            {isNetworkSource && (
              <div className="flex gap-2">
                <Input
                  value={cameraUrl}
                  onChange={(e) => setCameraUrl(e.target.value)}
                  placeholder="http://10.55.30.197:8080/video"
                  className="bg-sky-50 border-sky-300 text-slate-900 font-mono text-xs flex-1"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setPickerRotation(((pickerRotation + 90) % 360) as 0 | 90 | 180 | 270)
                  }
                  className="bg-sky-50 border-sky-300 text-slate-900 hover:bg-sky-200 whitespace-nowrap"
                  title="Rotate preview 90° clockwise"
                >
                  <RotateCw className="w-4 h-4 mr-1" />
                  {pickerRotation || 0}°
                </Button>
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Label className="text-sm font-medium text-slate-700">
              Camera Name
            </Label>
            <Select value={nameMode} onValueChange={handleNameModeChange}>
              <SelectTrigger className="bg-sky-50 border-sky-300 text-slate-900">
                <SelectValue placeholder="Select orientation" />
              </SelectTrigger>
              <SelectContent className="bg-sky-50 border-sky-300">
                {PRESET_CAMERA_NAMES.map((preset) => {
                  const inUse = usedNames.has(preset.toLowerCase());
                  return (
                    <SelectItem
                      key={preset}
                      value={preset}
                      className="text-slate-900 hover:bg-sky-200"
                      disabled={inUse}
                    >
                      {preset}
                      {inUse && " · in use"}
                    </SelectItem>
                  );
                })}
                <SelectItem
                  value={CUSTOM_NAME}
                  className="text-slate-900 hover:bg-sky-200"
                >
                  Custom…
                </SelectItem>
              </SelectContent>
            </Select>
            {nameMode === CUSTOM_NAME && (
              <Input
                value={cameraName}
                onChange={(e) => setCameraName(e.target.value)}
                placeholder="e.g., workspace_cam"
                className="bg-sky-50 border-sky-300 text-slate-900"
              />
            )}
          </div>

          <div className="space-y-2 flex flex-col justify-end">
            <Button
              onClick={addCamera}
              className="bg-blue-500 hover:bg-blue-600 text-white"
              disabled={
                !selectedCameraIndex ||
                !cameraName.trim() ||
                (isNetworkSource && !streamProbe)
              }
            >
              <Plus className="w-4 h-4 mr-2" />
              Add Camera
            </Button>
          </div>
        </div>

        {isNetworkSource && cameraUrl.trim() !== "" && (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs">
              {streamChecking && <span className="text-slate-600">连接中…</span>}
              {streamProbe && (
                <span className="text-green-700">
                  {streamWidth}×{streamHeight}
                  {streamProbe.fps ? ` · ${streamProbe.fps} fps` : ""}
                </span>
              )}
              {streamError && !streamChecking && (
                <span className="text-amber-700">{streamError}</span>
              )}
            </div>
            {streamFeedUrl && (
              <img
                src={streamFeedUrl}
                alt={`${cameraName || "Network Camera"} preview`}
                className="w-full max-w-md aspect-[4/3] rounded-lg border border-sky-300 bg-sky-50 object-cover"
              />
            )}
          </div>
        )}
      </div>
      )}

      {/* Configured Cameras */}
      {cameras.length > 0 && (
        <div className="space-y-4">
          {!readOnly && (
            <h4 className="text-md font-medium text-slate-700">
              Configured Cameras ({cameras.length})
            </h4>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-4">
            {cameras.map((camera) => (
              <CameraPreview
                key={camera.id}
                camera={camera}
                paused={streamsPaused}
                readOnly={readOnly}
                controlError={controlErrors[camera.id]}
                onRemove={() => removeCamera(camera.id)}
                onUpdate={(updates) => updateCamera(camera.id, updates)}
              />
            ))}
          </div>
        </div>
      )}

      {cameras.length === 0 && (
        <div className="text-center py-8 text-slate-600">
          <Camera className="w-12 h-12 mx-auto mb-4 text-slate-600" />
          <p>
            {readOnly
              ? "No cameras are configured for this robot."
              : "No cameras configured. Add a camera to get started."}
          </p>
        </div>
      )}
    </div>
  );
};

interface CameraPreviewProps {
  camera: CameraConfig;
  paused: boolean;
  readOnly: boolean;
  controlError?: string;
  onRemove: () => void;
  onUpdate: (updates: Partial<CameraConfig>) => void;
}

const CameraPreview: React.FC<CameraPreviewProps> = ({
  camera,
  paused,
  readOnly,
  controlError,
  onRemove,
  onUpdate,
}) => {
  const isNetwork = camera.type === "network" || !!camera.url;
  // A stream has no browser representation, so its preview comes from the
  // backend's own copy of the feed — RTSP takes a second client session, unlike
  // a UVC device, which the recorder must hold alone.
  const {
    probe: streamProbe,
    displayWidth: streamWidth,
    displayHeight: streamHeight,
    error: streamError2,
    isChecking: streamChecking,
    feedUrl: streamFeedUrl,
  } = useNetworkCameraSource(
    isNetwork && !paused ? camera.url : undefined,
    camera.rotation ?? 0,
  );
  const { videoRef, hasError: streamError } = useCameraStream(
    isNetwork ? undefined : camera.device_id,
    paused
  );
  const showVideo = !paused && !isNetwork && camera.device_id && !streamError;
  const showFeed = !paused && isNetwork && !!streamFeedUrl;
  return (
    <div className="bg-sky-100 rounded-lg border border-sky-300 overflow-hidden">
      <div className="aspect-[4/3] bg-sky-50 relative">
        {showVideo ? (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
        ) : showFeed ? (
          <img
            src={streamFeedUrl}
            alt={`${camera.name} 预览`}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-slate-600 mb-2" />
            <span className="text-slate-600 text-sm">
              {paused
                ? "Preview paused"
                : isNetwork
                ? streamChecking
                  ? "连接中…"
                  : streamError2 || "No preview"
                : camera.device_id
                ? "Preview failed"
                : "No browser match"}
            </span>
          </div>
        )}
      </div>

      {/* Camera Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <h5 className="font-medium text-slate-900 truncate">{camera.name}</h5>
          {!readOnly && (
            <Button
              onClick={onRemove}
              size="sm"
              variant="ghost"
              className="text-red-700 hover:text-red-700 hover:bg-red-100 p-1"
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>

        {readOnly ? (
          <p className="text-xs text-slate-600">
            {camera.width}×{camera.height}
            {camera.fps ? ` · ${camera.fps} fps` : ""}
          </p>
        ) : (
        <Collapsible>
          <CollapsibleTrigger className="group flex items-center gap-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 transition-colors">
            <ChevronRight className="w-3.5 h-3.5 transition-transform group-data-[state=open]:rotate-90" />
            Configuration
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 space-y-2">
            {isNetwork && (
              <div className="space-y-2 text-xs text-slate-600">
                <div className="flex items-center gap-2">
                  <span className="w-16 shrink-0">URL:</span>
                  <Input
                    value={camera.url ?? ""}
                    onChange={(e) =>
                      onUpdate({ url: e.target.value, type: "network" })
                    }
                    className="bg-sky-50 border-sky-300 text-slate-900 text-[11px] font-mono h-6"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-16 shrink-0">Stream:</span>
                  <span className={streamError2 ? "text-amber-700" : "text-slate-700"}>
                    {streamProbe
                      ? `${streamWidth}×${streamHeight}${
                          streamProbe.fps ? ` · ${streamProbe.fps} fps` : ""
                        }`
                      : streamChecking
                      ? "探测中…"
                      : streamError2 || "未连接"}
                  </span>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const next = (((camera.rotation ?? 0) + 90) % 360) as
                        | 0
                        | 90
                        | 180
                        | 270;
                      onUpdate({ rotation: next });
                    }}
                    className="ml-auto h-6 px-2 text-[11px] bg-sky-50 border-sky-300 text-slate-900 hover:bg-sky-200"
                    title="Rotate 90° clockwise"
                  >
                    <RotateCw className="w-3 h-3 mr-1" />
                    {camera.rotation ?? 0}°
                  </Button>
                </div>
              </div>
            )}
            {!isNetwork && (
            <div className="grid grid-cols-1 gap-2 text-xs text-slate-600">
              <div className="flex items-center gap-2">
                <span className="w-16">Resolution:</span>
                <div className="flex items-center gap-1">
                  <NumberInput
                    value={camera.width}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ width: v });
                    }}
                    className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-16"
                    min="320"
                    max="1920"
                  />
                  <span className="flex items-center">×</span>
                  <NumberInput
                    value={camera.height}
                    onChange={(v) => {
                      if (v !== undefined) onUpdate({ height: v });
                    }}
                    className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-16"
                    min="240"
                    max="1080"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FPS:</span>
                <NumberInput
                  value={camera.fps ?? 30}
                  onChange={(v) => {
                    if (v !== undefined) onUpdate({ fps: v });
                  }}
                  className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-16"
                  min="10"
                  max="60"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">FOURCC:</span>
                <Select
                  value={camera.fourcc ?? FOURCC_AUTO}
                  onValueChange={(v) =>
                    onUpdate({ fourcc: v === FOURCC_AUTO ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-sky-50 border-sky-300">
                    <SelectItem
                      value={FOURCC_AUTO}
                      className="text-slate-900 hover:bg-sky-200 text-xs"
                    >
                      Auto
                    </SelectItem>
                    {FOURCC_OPTIONS.map((code) => (
                      <SelectItem
                        key={code}
                        value={code}
                        className="text-slate-900 hover:bg-sky-200 text-xs"
                      >
                        {code}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Backend:</span>
                <Select
                  value={camera.backend ?? BACKEND_DEFAULT}
                  onValueChange={(v) =>
                    onUpdate({ backend: v === BACKEND_DEFAULT ? undefined : v })
                  }
                >
                  <SelectTrigger className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-sky-50 border-sky-300">
                    <SelectItem
                      value={BACKEND_DEFAULT}
                      className="text-slate-900 hover:bg-sky-200 text-xs"
                    >
                      Default
                    </SelectItem>
                    {BACKEND_OPTIONS.map((name) => (
                      <SelectItem
                        key={name}
                        value={name}
                        className="text-slate-900 hover:bg-sky-200 text-xs"
                      >
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Exposure:</span>
                <NumberInput
                  value={camera.exposure}
                  onChange={(v) => onUpdate({ exposure: v })}
                  className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-16"
                  min="0"
                  placeholder="auto"
                  title="V4L2 exposure_time_absolute，100 µs 为一档（100 = 10 ms）；留空 = 自动曝光"
                />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-16">Gain:</span>
                <NumberInput
                  value={camera.gain}
                  onChange={(v) => onUpdate({ gain: v })}
                  className="bg-sky-50 border-sky-300 text-slate-900 text-xs h-6 px-2 w-16"
                  min="0"
                  placeholder="auto"
                  title="V4L2 gain；留空 = 由驱动控制"
                />
              </div>
              {controlError && (
                <p className="text-[10px] text-amber-700 leading-tight">
                  {controlError}
                </p>
              )}
              <p className="text-[10px] text-slate-600 leading-tight">
                Overriding the backend can reorder camera indices on macOS.
              </p>
            </div>
            )}
            <div className="text-xs text-slate-600 break-all">
              {isNetwork
                ? `Type: network | URL: ${camera.url}`
                : `Type: ${camera.type} | Device: ${camera.device_id?.substring(
                    0,
                    10
                  )}...`}
            </div>
          </CollapsibleContent>
        </Collapsible>
        )}
      </div>
    </div>
  );
};

export default CameraConfiguration;
