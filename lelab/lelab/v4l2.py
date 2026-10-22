"""Shared V4L2 camera fixes applied right after connect.

Recording and inference must apply identical controls, otherwise the two see
different image distributions and a policy trained on one misbehaves on the
other.

- auto white balance: kills the green tint these UVC cams power on with.
- 50 Hz power-line frequency + driver-controlled exposure: brightness adapts
  to ambient light (the old fixed 10 ms manual exposure overexposed white
  tabletops whenever the room got brighter), while the power-line constraint
  keeps 50 Hz lighting from flickering.
- fallback: cams that reject driver-controlled exposure get the legacy manual
  10 ms pin (one full 50 Hz period, hence flicker-free).
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# V4L2 auto_exposure menu: 1 = Manual, 3 = Aperture Priority (driver picks the
# exposure; exposure_time_absolute goes inactive). Indices 0/2 are rejected by
# these cams.
_AUTO_EXPOSURE_DRIVER = "3"
_LEGACY_MANUAL_CONTROLS = ("auto_exposure=1", "exposure_time_absolute=100")


def _set_ctrl(device_path: str, ctrl: str) -> bool:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl={ctrl}"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("v4l2-ctl set %s on %s failed: %s", ctrl, device_path, exc)
        return False


def _supports(device_path: str, ctrl_name: str) -> bool:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--list-ctrls"],
            check=False,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0 and ctrl_name in result.stdout.decode()
    except Exception as exc:
        logger.debug("v4l2-ctl list on %s failed: %s", device_path, exc)
        return False


def apply_camera_controls(device_path: str, label: str = "") -> None:
    name = f"{label} ({device_path})" if label else device_path
    _set_ctrl(device_path, "white_balance_automatic=1")
    if not _supports(device_path, "auto_exposure"):
        return
    _set_ctrl(device_path, "power_line_frequency=1")
    if _set_ctrl(device_path, f"auto_exposure={_AUTO_EXPOSURE_DRIVER}"):
        logger.info("Applied auto white balance + flicker-safe auto exposure to %s", name)
    else:
        for ctrl in _LEGACY_MANUAL_CONTROLS:
            _set_ctrl(device_path, ctrl)
        logger.info("Applied auto white balance + manual 10ms exposure to %s", name)
