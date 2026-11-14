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

A camera whose UI entry carries an exposure/gain value is pinned to manual
exposure with exactly that value instead, so recording and inference of the same
rig stay pixel-identical.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

# V4L2 auto_exposure menu: 1 = Manual, 3 = Aperture Priority (driver picks the
# exposure; exposure_time_absolute goes inactive). Indices 0/2 are rejected by
# these cams.
_AUTO_EXPOSURE_DRIVER = "3"
_AUTO_EXPOSURE_MANUAL = "1"
_LEGACY_MANUAL_CONTROLS = ("auto_exposure=1", "exposure_time_absolute=100")

# Manual overrides from the UI, keyed by device path. lerobot's camera configs
# have no room for these, and both paths read them from here after connect.
_manual_controls: dict[str, dict[str, int]] = {}


def configure_manual_controls(cameras: dict[str, dict] | None) -> None:
    """Remember per-device exposure/gain the UI asked for.

    Values are raw V4L2 units (`exposure_time_absolute` counts 100 µs steps, so
    100 = 10 ms). Leaving either out keeps the driver's auto behaviour.
    """
    manual: dict[str, dict[str, int]] = {}
    for cfg in (cameras or {}).values():
        if not isinstance(cfg, dict) or not isinstance(cfg.get("camera_index"), int):
            continue
        wanted = {
            key: cfg[key]
            for key in ("exposure", "gain")
            if isinstance(cfg.get(key), int) and cfg[key] >= 0
        }
        if wanted:
            manual[f"/dev/video{cfg['camera_index']}"] = wanted
    _manual_controls.clear()
    _manual_controls.update(manual)


def _manual_for(device_path: str) -> dict[str, int]:
    return _manual_controls.get(device_path, {})


def _set_ctrl(device_path: str, ctrl: str) -> bool:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl={ctrl}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        logger.warning("v4l2-ctl set %s on %s failed: %s", ctrl, device_path, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "v4l2-ctl set %s on %s failed: %s",
            ctrl,
            device_path,
            (result.stderr or result.stdout).strip() or f"exit {result.returncode}",
        )
    return result.returncode == 0


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
    manual = _manual_for(device_path)
    _set_ctrl(device_path, "white_balance_automatic=1")
    if "gain" in manual:
        _set_ctrl(device_path, f"gain={manual['gain']}")
    if not _supports(device_path, "auto_exposure"):
        return
    _set_ctrl(device_path, "power_line_frequency=1")

    parts = ["auto white balance"]
    if "exposure" in manual:
        # Manual exposure is the only mode where exposure_time_absolute applies.
        _set_ctrl(device_path, f"auto_exposure={_AUTO_EXPOSURE_MANUAL}")
        _set_ctrl(device_path, f"exposure_time_absolute={manual['exposure']}")
        parts.append(f"manual exposure={manual['exposure']}")
    elif _set_ctrl(device_path, f"auto_exposure={_AUTO_EXPOSURE_DRIVER}"):
        parts.append("auto exposure")
    else:
        for ctrl in _LEGACY_MANUAL_CONTROLS:
            _set_ctrl(device_path, ctrl)
        parts.append("manual 10ms exposure")
    if "gain" in manual:
        parts.append(f"gain={manual['gain']}")
    logger.info("Applied %s to %s", " + ".join(parts), name)


def apply_live_controls(
    device_path: str, exposure: int | None, gain: int | None
) -> dict[str, object]:
    """Apply one camera's exposure/gain immediately (configuration-panel path).

    The panel's preview is the browser's getUserMedia stream, which never asks
    the driver for these values, so writing them device-side is the only way a
    change becomes visible before a recording starts. Unlike
    apply_camera_controls this touches only the two requested controls (white
    balance / power line stay as they are), and a cleared exposure hands the
    device back to driver-controlled metering, matching the record-time default.
    """
    applied: list[str] = []
    failed: dict[str, str] = {}

    if gain is not None:
        if not _supports(device_path, "gain"):
            failed["gain"] = "device has no gain control"
        elif _set_ctrl(device_path, f"gain={gain}"):
            applied.append("gain")
        else:
            failed["gain"] = f"gain={gain} rejected by driver"

    if exposure is not None:
        if not _supports(device_path, "auto_exposure"):
            failed["exposure"] = "device has no auto_exposure control"
        elif not _set_ctrl(device_path, f"auto_exposure={_AUTO_EXPOSURE_MANUAL}"):
            failed["exposure"] = "driver refused manual exposure mode"
        elif _set_ctrl(device_path, f"exposure_time_absolute={exposure}"):
            applied.append("exposure")
        else:
            failed["exposure"] = f"exposure={exposure} rejected (out of driver range?)"
    elif _supports(device_path, "auto_exposure"):
        _set_ctrl(device_path, f"auto_exposure={_AUTO_EXPOSURE_DRIVER}")
        applied.append("auto_exposure")

    return {"applied": applied, "failed": failed}
