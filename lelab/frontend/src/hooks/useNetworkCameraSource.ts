import { useEffect, useMemo, useState } from "react";
import { useApi } from "@/contexts/ApiContext";

export interface NetworkStreamProbe {
  /** Preview token; the URL itself carries credentials, so it stays server-side. */
  key: string;
  url: string;
  /** Raw stream dimensions as reported by the probe (before any rotation swap). */
  width: number;
  height: number;
  fps: number | null;
}

/**
 * Registers a stream URL (RTSP/HTTP/MJPEG) with the backend and reports what the
 * camera actually sends, plus an MJPEG preview address.
 *
 * Only the URL triggers a re-probe; the backend opens the stream, so the
 * reported size and frame rate match what will be recorded rather than what
 * the form says. Rotation lives outside the probe and rides on the feed URL,
 * so the picker button reconnects in one frame instead of waiting a few
 * seconds for a fresh open.
 */
export function useNetworkCameraSource(
  url: string | undefined,
  rotation: number = 0,
  debounceMs = 400,
) {
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

  const swapped = rotation === 90 || rotation === 270;
  const displayWidth = probe ? (swapped ? probe.height : probe.width) : undefined;
  const displayHeight = probe ? (swapped ? probe.width : probe.height) : undefined;

  const feedUrl = useMemo(() => {
    if (!probe) return null;
    // Rotation and a nonce change the src, which is what forces the browser to
    // drop the old multipart stream and reconnect. Without a nonce the browser
    // may hold the same connection when rotation collapses back to a value it
    // already saw (e.g. 0° → 90° → 180° → 270° → 0°).
    const params = new URLSearchParams({ key: probe.key, rotation: String(rotation) });
    return `${baseUrl}/network-camera-feed?${params.toString()}`;
  }, [probe, rotation, baseUrl]);

  return { probe, displayWidth, displayHeight, error, isChecking, feedUrl };
}
