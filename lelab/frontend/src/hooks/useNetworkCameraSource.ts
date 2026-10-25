import { useEffect, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

export interface NetworkStreamProbe {
  /** Preview token; the URL itself carries credentials, so it stays server-side. */
  key: string;
  url: string;
  width: number;
  height: number;
  fps: number | null;
}

/**
 * Registers a stream URL (RTSP/HTTP/MJPEG) with the backend and reports what the
 * camera actually sends, plus an MJPEG preview address.
 *
 * The backend has to open the stream before the camera is usable: an IP camera
 * refuses a resolution it wasn't configured to encode, and whatever it does send
 * becomes the dataset's image features. Re-runs whenever the URL changes, so
 * typing a wrong address shows up as an error instead of a black box.
 */
export function useNetworkCameraSource(url: string | undefined, debounceMs = 500) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const [probe, setProbe] = useState<NetworkStreamProbe | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const target = (url || "").trim();

  useEffect(() => {
    if (!target) {
      setProbe(null);
      setError(null);
      setIsChecking(false);
      return;
    }
    let cancelled = false;
    setIsChecking(true);
    setError(null);

    const timer = window.setTimeout(async () => {
      try {
        const r = await fetchWithHeaders(`${baseUrl}/network-camera-source`, {
          method: "POST",
          body: JSON.stringify({ url: target }),
        });
        const data = await r.json().catch(() => ({}));
        if (cancelled) return;
        if (!r.ok) {
          setProbe(null);
          setError(data.detail || `无法连接该地址（HTTP ${r.status}）`);
        } else {
          setProbe({
            key: data.key,
            url: data.url,
            width: data.width,
            height: data.height,
            fps: data.fps ?? null,
          });
        }
      } catch (e) {
        if (cancelled) return;
        setProbe(null);
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setIsChecking(false);
      }
    }, debounceMs);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [target, baseUrl, fetchWithHeaders, debounceMs]);

  const feedUrl = probe ? `${baseUrl}/network-camera-feed?key=${probe.key}` : null;

  return { probe, error, isChecking, feedUrl };
}
