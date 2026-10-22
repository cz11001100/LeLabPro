import React, { useState, useMemo, useEffect, useRef } from "react";
import { RefreshCw, CameraOff, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useRobots } from "@/hooks/useRobots";
import { useBrowserCameras } from "@/hooks/useBrowserCameras";
import { useCameraStream } from "@/hooks/useCameraStream";
import CameraFeed from "./CameraFeed";

/**
 * Optional live camera panel for the teleoperation page. Off by default so we
 * never call getUserMedia just by landing on the page.
 *
 * Camera assignment uses a visual picker: when a robot camera has no matched
 * browser device (common on Linux where cv2/V4L2 names don't match browser
 * labels), we show a grid of live browser-camera previews. The user clicks a
 * preview to assign it to the robot camera. This bypasses unreliable
 * name-based matching entirely.
 */
const TeleopCameraPanel: React.FC = () => {
  const [enabled, setEnabled] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [permissionState, setPermissionState] = useState<PermissionState | "unknown">("unknown");
  // Manual camera assignments: robot camera id -> browser deviceId.
  const [manualAssignments, setManualAssignments] = useState<Record<string, string>>({});
  // Device IDs currently shown in the visual picker grid (prevents the same
  // browser camera from appearing in two previews at once).
  const [previewingDeviceIds, setPreviewingDeviceIds] = useState<Set<string>>(new Set());

  const { selectedRecord, isLoading: robotsLoading } = useRobots();
  const {
    cameras: browserCameras,
    isLoading: isEnumerating,
    permissionDenied: browserPermissionDenied,
    enumerate: enumerateBrowserCameras,
  } = useBrowserCameras();

  // Enumerate browser cameras when the panel is turned on.
  useEffect(() => {
    if (enabled) enumerateBrowserCameras();
  }, [enabled, enumerateBrowserCameras]);

  // Check the Permissions API state so we can show targeted guidance.
  useEffect(() => {
    if (!enabled || !navigator.permissions) return;
    let mounted = true;
    navigator.permissions.query({ name: "camera" as PermissionName }).then((status) => {
      if (!mounted) return;
      setPermissionState(status.state);
      const onChange = () => { if (mounted) setPermissionState(status.state); };
      status.addEventListener("change", onChange);
      return () => status.removeEventListener("change", onChange);
    });
    return () => { mounted = false; };
  }, [enabled]);

  // Resolve deviceIds for each robot-configured camera. Priority:
  //   1. Stored device_id on the robot config
  //   2. Manual assignment from the visual picker
  const configured = selectedRecord?.cameras ?? [];
  const feeds = useMemo(() => {
    return configured.map((c) => {
      let deviceId = c.device_id;
      if (!deviceId && manualAssignments[c.id]) {
        deviceId = manualAssignments[c.id];
      }
      return { key: c.id, name: c.name, deviceId };
    });
  }, [configured, manualAssignments]);

  const permissionDenied = permissionState === "denied" || browserPermissionDenied;
  const unassignedFeeds = feeds.filter((f) => !f.deviceId);

  // Browser cameras not yet assigned to any robot camera — these are the
  // candidates shown in the visual picker grid.
  const assignedDeviceIds = new Set([
    ...Object.values(manualAssignments),
    ...feeds.filter((f) => f.deviceId).map((f) => f.deviceId!),
  ]);
  const availableForAssignment = useMemo(
    () => browserCameras.filter((bc) => !assignedDeviceIds.has(bc.deviceId)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [browserCameras, manualAssignments, feeds]
  );

  // When the browser device list changes (hotplug, re-enumerate), drop any
  // previewing device that's no longer present.
  useEffect(() => {
    const validIds = new Set(browserCameras.map((c) => c.deviceId));
    setPreviewingDeviceIds((prev) => {
      const next = new Set([...prev].filter((id) => validIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [browserCameras]);

  const handleRetry = async () => {
    if (navigator.permissions) {
      const status = await navigator.permissions.query({ name: "camera" as PermissionName });
      setPermissionState(status.state);
    }
    await enumerateBrowserCameras();
    setReloadKey((k) => k + 1);
  };

  const handleAssign = (robotCameraId: string, deviceId: string) => {
    setManualAssignments((prev) => ({ ...prev, [robotCameraId]: deviceId }));
    setPreviewingDeviceIds((prev) => {
      const next = new Set(prev);
      next.add(deviceId);
      return next;
    });
  };

  const handleUnassign = (robotCameraId: string) => {
    const deviceId = manualAssignments[robotCameraId];
    setManualAssignments((prev) => {
      const next = { ...prev };
      delete next[robotCameraId];
      return next;
    });
    if (deviceId) {
      setPreviewingDeviceIds((prev) => {
        const next = new Set(prev);
        next.delete(deviceId);
        return next;
      });
    }
  };

  const releaseAllPreviews = () => {
    setPreviewingDeviceIds(new Set());
  };

  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-col gap-4 h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-medium text-gray-200">Cameras</h2>
        <div className="flex items-center gap-2">
          {enabled && feeds.length > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={handleRetry}
              className="h-9 w-9 text-gray-400 hover:text-white flex-shrink-0"
              title="Retry camera feeds (e.g. after reconnecting a camera)"
              aria-label="Retry camera feeds"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          )}
          <Label htmlFor="teleop-camera-toggle" className="text-sm text-gray-400">
            {enabled ? "On" : "Off"}
          </Label>
          <Switch
            id="teleop-camera-toggle"
            checked={enabled}
            onCheckedChange={setEnabled}
          />
        </div>
      </div>

      {enabled ? (
        permissionDenied ? (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <CameraOff className="w-10 h-10 text-amber-500" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-amber-400">摄像头权限被拒绝</p>
              <p className="text-xs text-gray-400 max-w-xs">
                请在浏览器设置中允许此网站访问摄像头，然后点击刷新按钮重试。
              </p>
              <p className="text-xs text-gray-500 max-w-xs">
                Chrome: 点击地址栏左侧的锁图标 → 网站设置 → 摄像头 → 允许
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRetry}
              className="border-amber-600 text-amber-400 hover:bg-amber-900/20"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              刷新权限
            </Button>
          </div>
        ) : feeds.length > 0 ? (
          <div className="flex flex-col gap-3 overflow-y-auto">
            {feeds.filter((f) => f.deviceId).map((feed) => (
              <CameraFeed
                key={`${feed.key}:${reloadKey}`}
                deviceId={feed.deviceId!}
                label={feed.name}
              />
            ))}

            {unassignedFeeds.length > 0 && (
              <VisualCameraPicker
                unassignedFeeds={unassignedFeeds}
                availableCameras={availableForAssignment}
                previewingDeviceIds={previewingDeviceIds}
                isEnumerating={isEnumerating}
                onAssign={handleAssign}
                onUnassign={handleUnassign}
                onRefresh={enumerateBrowserCameras}
                onReleaseAll={releaseAllPreviews}
              />
            )}
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            {robotsLoading
              ? "Loading robot..."
              : "No cameras configured for this robot. Add them during calibration to see live feeds here."}
          </p>
        )
      ) : (
        <p className="text-sm text-gray-500">
          Turn on to watch your cameras while you teleoperate.
        </p>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Visual camera picker
// ---------------------------------------------------------------------------

interface VisualCameraPickerProps {
  unassignedFeeds: { key: string; name: string }[];
  availableCameras: { index: number; label: string; deviceId: string }[];
  previewingDeviceIds: Set<string>;
  isEnumerating: boolean;
  onAssign: (robotCameraId: string, deviceId: string) => void;
  onUnassign: (robotCameraId: string) => void;
  onRefresh: () => void;
  onReleaseAll: () => void;
}

const VisualCameraPicker: React.FC<VisualCameraPickerProps> = ({
  unassignedFeeds,
  availableCameras,
  previewingDeviceIds,
  isEnumerating,
  onAssign,
  onUnassign,
  onRefresh,
  onReleaseAll,
}) => {
  const [pendingRobotCameraId, setPendingRobotCameraId] = useState<string | null>(null);

  const releasedRef = useRef(false);
  useEffect(
    () => () => {
      if (!releasedRef.current) {
        releasedRef.current = true;
        onReleaseAll();
      }
    },
    [onReleaseAll]
  );

  const handleSelectForPending = (deviceId: string) => {
    if (!pendingRobotCameraId) return;
    onAssign(pendingRobotCameraId, deviceId);
    setPendingRobotCameraId(null);
  };

  return (
    <div className="bg-gray-800/50 rounded-lg p-3 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-amber-400">
          {unassignedFeeds.length} 个摄像头未自动匹配
        </p>
        <Button
          variant="ghost"
          size="sm"
          onClick={onRefresh}
          disabled={isEnumerating}
          className="h-7 text-xs text-gray-400 hover:text-white"
        >
          <RefreshCw className={`w-3 h-3 mr-1 ${isEnumerating ? "animate-spin" : ""}`} />
          刷新
        </Button>
      </div>

      <p className="text-xs text-gray-400">
        点击下方摄像头名称进行手动分配，然后从实时预览中选择对应的浏览器摄像头。
      </p>

      <div className="space-y-1">
        {unassignedFeeds.map((feed) => (
          <button
            key={feed.key}
            type="button"
            onClick={() =>
              setPendingRobotCameraId((cur) => (cur === feed.key ? null : feed.key))
            }
            className={`w-full text-left px-3 py-2 rounded text-sm transition-colors ${
              pendingRobotCameraId === feed.key
                ? "bg-amber-600/20 text-amber-300 border border-amber-600/50"
                : "bg-gray-700/50 text-gray-300 hover:bg-gray-700 border border-transparent"
            }`}
          >
            {feed.name}
            {pendingRobotCameraId === feed.key && (
              <span className="text-xs ml-2 text-amber-400">← 选择预览中的摄像头...</span>
            )}
          </button>
        ))}
      </div>

      {pendingRobotCameraId && (
        <div className="space-y-2">
          <p className="text-xs text-gray-400">
            点击下方预览将摄像头分配给{" "}
            <span className="text-amber-300">
              {unassignedFeeds.find((f) => f.key === pendingRobotCameraId)?.name}
            </span>
            ：
          </p>
          {isEnumerating && availableCameras.length === 0 ? (
            <p className="text-xs text-gray-500 py-4 text-center">正在检测摄像头...</p>
          ) : availableCameras.length === 0 ? (
            <p className="text-xs text-gray-500 py-4 text-center">
              没有可用的浏览器摄像头。请确认摄像头已连接并授予权限。
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {availableCameras.map((cam) => (
                <CameraPreviewButton
                  key={cam.deviceId}
                  deviceId={cam.deviceId}
                  label={cam.label}
                  isPreviewing={previewingDeviceIds.has(cam.deviceId)}
                  onClick={() => handleSelectForPending(cam.deviceId)}
                />
              ))}
            </div>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPendingRobotCameraId(null)}
            className="text-xs text-gray-500 hover:text-gray-300"
          >
            取消
          </Button>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// Single camera preview tile — owns its own getUserMedia stream.
// ---------------------------------------------------------------------------

interface CameraPreviewButtonProps {
  deviceId: string;
  label: string;
  isPreviewing: boolean;
  onClick: () => void;
}

const CameraPreviewButton: React.FC<CameraPreviewButtonProps> = ({
  deviceId,
  label,
  isPreviewing,
  onClick,
}) => {
  const { videoRef, hasError } = useCameraStream(deviceId, false);

  return (
    <button
      type="button"
      onClick={onClick}
      className="relative rounded overflow-hidden border border-gray-700 hover:border-amber-500 transition-colors group cursor-pointer"
    >
      <div className="aspect-[4/3] bg-gray-800">
        {hasError ? (
          <div className="w-full h-full flex items-center justify-center">
            <CameraOff className="w-6 h-6 text-gray-600" />
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
        )}
      </div>
      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
        <span className="opacity-0 group-hover:opacity-100 text-xs text-white bg-amber-600 px-2 py-1 rounded transition-opacity">
          点击选择
        </span>
      </div>
      {isPreviewing && (
        <div className="absolute top-1 right-1 bg-green-600 rounded-full p-0.5">
          <Check className="w-3 h-3 text-white" />
        </div>
      )}
      <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-2 py-1">
        <span className="text-xs text-gray-300 truncate block">{label}</span>
      </div>
    </button>
  );
};

export default TeleopCameraPanel;
