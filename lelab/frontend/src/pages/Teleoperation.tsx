import React, { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import VisualizerPanel from "@/components/control/VisualizerPanel";
import TeleopCameraPanel from "@/components/control/TeleopCameraPanel";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useApi } from "@/contexts/ApiContext";
import { getTeleopMode, setTeleopMode, TELEOP_MODE_LABELS } from "@/lib/teleopMode";
import type { TeleopMode } from "@/lib/teleopMode";

const TeleoperationPage = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { baseUrl, fetchWithHeaders } = useApi();

  const [mode, setMode] = useState<TeleopMode>(() => getTeleopMode());
  const [switching, setSwitching] = useState(false);

  // Stop teleoperation exactly once, however the user leaves, so the back
  // button, an in-app link, and the unmount safety net can't double-stop or
  // double-toast.
  const stoppedRef = useRef(false);
  const stopTeleoperation = useCallback(async () => {
    if (stoppedRef.current) return;
    stoppedRef.current = true;
    try {
      const res = await fetchWithHeaders(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
      });
      const data = await res.json();
      if (data?.success) {
        toast({
          title: "Teleoperation stopped",
          description: "The arm was disconnected cleanly.",
        });
      }
    } catch {
      /* best-effort */
    }
  }, [baseUrl, fetchWithHeaders, toast]);

  // Cover every exit path so a session can't keep running and block the next
  // start with "already active":
  //   - the back button awaits stopTeleoperation() then navigates (below);
  //   - any other in-app navigation unmounts this component → stop via cleanup;
  //   - a browser-level leave (URL change, reload, tab close) never runs React
  //     cleanup, so `pagehide` fires a keepalive stop that survives the unload
  //     and stashes a flag the next page reads to confirm the clean disconnect.
  //     It uses a bare fetch (no JSON Content-Type) so the request stays a CORS
  //     "simple request" and isn't dropped to a preflight mid-unload.
  useEffect(() => {
    const handlePageHide = () => {
      try {
        sessionStorage.setItem("lelab:teleop-stopped", "1");
      } catch {
        /* sessionStorage may be unavailable; the stop below still runs */
      }
      fetch(`${baseUrl}/stop-teleoperation`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    };
    window.addEventListener("pagehide", handlePageHide);

    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      stopTeleoperation();
    };
  }, [baseUrl, stopTeleoperation]);

  // The mode is global: persist it, then let the backend restart the running
  // session so teleoperation and recording stay on the same input device.
  const handleModeChange = useCallback(
    async (next: TeleopMode) => {
      if (next === mode || switching) return;
      setTeleopMode(next);
      setMode(next);
      setSwitching(true);
      try {
        const res = await fetchWithHeaders(`${baseUrl}/set-teleop-mode`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: next }),
        });
        const data = await res.json();
        if (res.ok && data?.success) {
          toast({
            title: "Teleop mode",
            description: data.restarted
              ? `${TELEOP_MODE_LABELS[next]} is now active.`
              : data.message,
          });
        } else {
          toast({
            title: "Could not switch teleop mode",
            description: data?.message || "Failed to restart teleoperation.",
            variant: "destructive",
          });
        }
      } catch {
        toast({
          title: "Connection error",
          description: "Could not reach the backend to switch modes.",
          variant: "destructive",
        });
      } finally {
        setSwitching(false);
      }
    },
    [baseUrl, fetchWithHeaders, mode, switching, toast],
  );

  const handleGoBack = async () => {
    await stopTeleoperation();
    navigate("/");
  };

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-2 sm:p-4">
      <div className="w-full h-[95vh] flex">
        <VisualizerPanel
          onGoBack={handleGoBack}
          className="lg:w-full"
          rightSlot={<TeleopCameraPanel />}
          headerExtra={
            <>
              {mode === "keyboard_ee" && (
                <span className="hidden text-xs text-gray-400 md:inline">
                  ↑↓ 前后 · ←→ 左右 · Shift 上下 · Ctrl 开/合
                </span>
              )}
              <Select
                value={mode}
                onValueChange={(value) => handleModeChange(value as TeleopMode)}
                disabled={switching}
              >
                <SelectTrigger className="h-8 w-40 bg-gray-800 border-gray-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-gray-800 border-gray-600 text-white">
                  <SelectItem value="leader">{TELEOP_MODE_LABELS.leader}</SelectItem>
                  <SelectItem value="keyboard_ee">{TELEOP_MODE_LABELS.keyboard_ee}</SelectItem>
                </SelectContent>
              </Select>
            </>
          }
        />
      </div>
    </div>
  );
};

export default TeleoperationPage;
