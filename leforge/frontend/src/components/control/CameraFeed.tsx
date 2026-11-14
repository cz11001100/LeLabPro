import React from "react";
import { VideoOff } from "lucide-react";
import { useCameraStream } from "@/hooks/useCameraStream";
import { useNetworkCameraSource } from "@/hooks/useNetworkCameraSource";

interface CameraFeedProps {
  /** Browser deviceId to stream. Empty string renders the "no camera" state. */
  deviceId: string;
  /** Optional caption shown under the feed. */
  label?: string;
}

/** Live browser-camera feed bound to a deviceId via getUserMedia. */
const CameraFeed: React.FC<CameraFeedProps> = ({ deviceId, label }) => {
  const { videoRef, hasError } = useCameraStream(deviceId, false);
  const showVideo = deviceId && !hasError;

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
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-slate-600 mb-2" />
            <span className="text-slate-600 text-sm">
              {deviceId ? "Preview failed" : "No camera selected"}
            </span>
          </div>
        )}
      </div>
      {label && (
        <div className="p-2 text-sm text-slate-700 truncate border-t border-sky-200">
          {label}
        </div>
      )}
    </div>
  );
};

interface StreamCameraFeedProps {
  /** RTSP/HTTP source to preview through the backend. */
  url: string;
  /** Clockwise quarter-turn applied by the backend before serving the preview. */
  rotation?: number;
  label?: string;
}

/**
 * Live feed for a network camera. The browser can't read RTSP, so the backend
 * opens its own client session and serves MJPEG — a stream takes several
 * concurrent sessions, so this never steals the device from a recording or an
 * inference run the way a USB camera would.
 */
export const StreamCameraFeed: React.FC<StreamCameraFeedProps> = ({ url, rotation = 0, label }) => {
  const { probe, feedUrl, error, isChecking } = useNetworkCameraSource(url, rotation);
  const showFeed = !!feedUrl && !error;

  return (
    <div className="bg-sky-100 rounded-lg border border-sky-300 overflow-hidden">
      <div className="aspect-[4/3] bg-sky-50 relative">
        {showFeed ? (
          <img src={feedUrl} alt={`${label || "Network Camera"} preview`} className="w-full h-full object-cover" />
        ) : (
          <div className="w-full h-full flex flex-col items-center justify-center">
            <VideoOff className="w-8 h-8 text-slate-600 mb-2" />
            <span className="text-slate-600 text-sm px-3 text-center">
              {isChecking ? "连接中…" : error || "等待画面"}
            </span>
          </div>
        )}
        {showFeed && probe && (
          <span className="absolute bottom-1 left-1 text-[10px] text-slate-700 bg-sky-100 px-1 rounded">
            {probe.width}×{probe.height}
            {probe.fps ? ` · ${probe.fps} fps` : ""}
          </span>
        )}
      </div>
      {label && (
        <div className="p-2 text-sm text-slate-700 truncate border-t border-sky-200">
          {label}
        </div>
      )}
    </div>
  );
};

export default CameraFeed;
