import { useCallback, useEffect, useState } from "react";

export interface BrowserCamera {
  index: number;
  label: string;
  deviceId: string;
}

/**
 * Enumerate browser-visible cameras via getUserMedia + enumerateDevices.
 * Independent of the backend `/available-cameras` endpoint so the visual
 * picker works even when the server is down.
 *
 * We request camera permission up front because enumerateDevices() returns
 * empty labels/deviceIds until a permission grant exists.
 */
export function useBrowserCameras() {
  const [cameras, setCameras] = useState<BrowserCamera[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const enumerate = useCallback(async (): Promise<BrowserCamera[]> => {
    setIsLoading(true);
    setPermissionDenied(false);
    try {
      let granted = false;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach((t) => t.stop());
        granted = true;
      } catch {
        setPermissionDenied(true);
      }

      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter((d) => d.kind === "videoinput");

      const cams: BrowserCamera[] = devices.map((d, i) => ({
        index: i,
        label: d.label || `Camera ${i}`,
        deviceId: d.deviceId,
      }));

      // Without permission, deviceIds are unusable for getUserMedia anyway.
      if (!granted) {
        setCameras([]);
        return [];
      }

      setCameras(cams);
      return cams;
    } catch {
      setCameras([]);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const handler = () => { enumerate(); };
    navigator.mediaDevices.addEventListener("devicechange", handler);
    return () => navigator.mediaDevices.removeEventListener("devicechange", handler);
  }, [enumerate]);

  return { cameras, isLoading, permissionDenied, enumerate };
}
