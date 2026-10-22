# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference mode: drives the SO-101 follower with a trained policy.

Mirrors `app/teleoperating.py` in shape — single global session, mutex
with teleoperation/recording (the follower's serial bus can only be
opened once), `lerobot.scripts.lerobot_rollout` running as a subprocess
for clean cancellation. Hub-checkpoint refs are resolved to a local dir
via huggingface_hub.snapshot_download before we spawn the subprocess.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .utils.config import setup_follower_calibration_file

logger = logging.getLogger(__name__)


class InferenceRequest(BaseModel):
    follower_port: str
    follower_config: str
    policy_ref: str  # opaque ref returned by /jobs/{id}/checkpoints
    task: str = ""
    cameras: dict[str, dict[str, Any]] = {}
    duration_s: int = 60
    n_runs: int = 1  # total number of consecutive inference runs
    # Inter-chunk action smoothing (deployment-time only). strength is 0–0.95;
    # it maps to smooth_rollout's alpha as 1 - strength (alpha 1 = off).
    chunk_smooth: bool = False
    chunk_smooth_strength: float = 0.5
    # Official real-time chunking (lerobot --inference.type=rtc): blends chunks
    # inside the policy instead of post-filtering. Mutually exclusive with
    # chunk_smooth — when rtc is on we force alpha 1.0 (no double smoothing).
    rtc: bool = False


inference_active: bool = False
_inference_proc: subprocess.Popen | None = None
_inference_started_at: float | None = None
_inference_rollout_started_at: float | None = None
_inference_meta: dict[str, Any] = {}
_inference_remaining_runs: int = 0
_inference_current_run: int = 0
_inference_restart_config: dict[str, Any] = {}
# Guards mutations to the globals above; held only for the short critical
# sections in start/stop/status. Reentrant because handle_inference_status
# calls handle_start_inference to auto-restart the next run while it already
# holds the lock.
_state_lock = threading.RLock()
# Inference camera preview: open cameras independently via cv2 and serve
# frames via MJPEG. The inference subprocess owns the primary camera access;
# we attempt a secondary open here. If V4L2 doesn't allow shared access,
# the preview silently fails.
_inference_cams: dict[str, Any] = {}
_inference_cam_frames: dict[str, bytes] = {}
_inference_cam_lock = threading.Lock()
_inference_cam_stop = threading.Event()
_inference_cam_thread: threading.Thread | None = None
# Preferred preview source: lelab.smooth_rollout (the inference subprocess)
# writes its already-captured observation frames here as {cam_key}.jpg every
# 0.2s. Reading files avoids a secondary V4L2 open, which is single-opener on
# this machine and races/kills the subprocess when attempted.
_PREVIEW_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_preview"
_SNAPSHOT_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_snapshots"
_HUB_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")
# lerobot prints this once per run, the moment its main control loop is
# about to take over from the setup phase. We watch stdout for it so the
# UI can present a "rollout time" separate from the multi-second policy
# load + bus connect + camera connect setup overhead.
_ROLLOUT_START_MARKER = "Rollout setup complete"


def _pump_stdout(proc: subprocess.Popen, log_handle) -> None:
    """Tee the subprocess's stdout to the log file and watch for the
    rollout-start marker."""
    global _inference_rollout_started_at
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            try:
                log_handle.write(line)
                log_handle.flush()
            except Exception:
                pass
            if _inference_rollout_started_at is None and _ROLLOUT_START_MARKER in line:
                _inference_rollout_started_at = time.time()
                logger.info(
                    "Inference rollout main loop started after %.1fs of setup",
                    _inference_rollout_started_at - (_inference_started_at or _inference_rollout_started_at),
                )
                # Cameras are connected by now; apply the same v4l2 fixes
                # record.py uses (auto white balance kills the green tint).
                _apply_v4l2_camera_controls(_inference_meta.get("cameras", {}))
    except Exception as exc:
        logger.exception("Inference stdout pump failed: %s", exc)
    finally:
        with contextlib.suppress(Exception):
            log_handle.close()


def _detect_device() -> str:
    """cuda → mps → cpu, picked once at start time."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_policy_path(policy_ref: str) -> str:
    """Turn a checkpoints API ref into a local path that lerobot accepts.

    Local refs are already absolute paths to a pretrained_model dir.
    Hub refs look like 'user/repo@checkpoints/<step_dir>' where
    <step_dir> is lerobot's zero-padded directory name (e.g. 000050) — we
    forward it verbatim into snapshot_download's allow_patterns and the
    resolved local path.
    A 'user/repo@root' ref means the whole repo IS the pretrained_model
    (no checkpoints sub-tree); the full repo is downloaded via
    snapshot_download and its root is returned directly."""
    if Path(policy_ref).is_dir():
        return policy_ref
    from huggingface_hub import snapshot_download

    m = _HUB_REF_RE.match(policy_ref)
    if m:
        repo_id, step_dir = m.group("repo"), m.group("step_dir")
        local_root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"checkpoints/{step_dir}/pretrained_model/*"],
        )
        return str(Path(local_root) / "checkpoints" / step_dir / "pretrained_model")
    m = _HUB_ROOT_REF_RE.match(policy_ref)
    if m:
        return snapshot_download(repo_id=m.group("repo"), repo_type="model")
    raise ValueError(f"Unrecognised policy ref: {policy_ref!r}")


def _read_policy_config(policy_path: str) -> dict[str, Any]:
    """Load pretrained_model/config.json if present."""
    config_path = Path(policy_path) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        with open(config_path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to read policy config at %s: %s", config_path, exc)
        return {}


def _rollout_inference_args(policy_path: str, force_rtc: bool = False) -> list[str]:
    """Return extra lerobot-rollout flags for policies that reject sync inference.

    GR00T and other relative-action policies decode action chunks against the
    observation state at inference time. The default sync backend calls
    ``select_action`` per tick and can re-decode cached relative actions
    against newer states, so lerobot requires the RTC/chunked rollout path
    instead.

    GR00T also needs tuned RTC queue settings: the lerobot default
    ``queue_threshold=30`` makes the background thread replan while a 16-step
    chunk is still playing, which replaces the queue mid-motion and feels
    like one action then a full recalculation.

    ``force_rtc`` opts in from the UI for policies that *can* run sync but
    benefit from RTC (SmolVLA: whole-chunk open-loop execution of 50 steps is
    a known success-rate killer; RTC replans every execution_horizon steps)."""
    cfg = _read_policy_config(policy_path)
    policy_type = cfg.get("type")
    needs_rtc = bool(cfg.get("use_relative_actions")) or policy_type == "groot"
    if force_rtc and not needs_rtc:
        args = ["--inference.type=rtc", "--inference.rtc.execution_horizon=10", "--inference.rtc.max_guidance_weight=10.0"]
        return args
    if not needs_rtc:
        return []

    args = ["--inference.type=rtc"]

    n_action_steps = cfg.get("n_action_steps")
    if isinstance(n_action_steps, int) and n_action_steps > 0:
        args.append(f"--inference.rtc.execution_horizon={n_action_steps}")
    elif policy_type == "groot":
        args.append("--inference.rtc.execution_horizon=16")

    # Wait until the current chunk is consumed before replanning. GROOT docs
    # recommend keeping this at 0 (never > 5) for stable real-robot rollout.
    args.append("--inference.queue_threshold=0")
    return args


def _format_cameras_arg(cameras: dict[str, dict[str, Any]]) -> str:
    """Convert {name: {type, camera_index, width, height, fps}} into
    lerobot's CLI dict syntax. The frontend key `camera_index` is
    remapped to lerobot's `index_or_path`."""
    parts = []
    for name, cfg in cameras.items():
        remapped = {
            ("index_or_path" if k == "camera_index" else k): v for k, v in cfg.items() if v is not None
        }
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


# Exception lines at the tail of a Python traceback look like
# "RuntimeError: ..." or "lerobot.errors.DeviceNotConnectedError: ...".
_EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Timeout|Failure)\b")


def _extract_error_from_log(log_path: str | None) -> str | None:
    """Pull the meaningful error out of a failed rollout's log so the UI can
    show it directly instead of telling the user to open a file in the cache."""
    if not log_path:
        return None
    try:
        # Only the tail matters; avoid materializing a multi-MB verbose log.
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            data = fh.read()
    except OSError:
        return None
    tail = data.decode("utf-8", errors="replace").splitlines()[-50:]
    # Prefer the last exception line + everything after it (the message body).
    exc_idx = next((i for i in range(len(tail) - 1, -1, -1) if _EXC_LINE_RE.match(tail[i])), None)
    if exc_idx is not None:
        snippet = "\n".join(tail[exc_idx:]).strip()
    else:
        non_empty = [ln for ln in tail if ln.strip()]
        snippet = "\n".join(non_empty[-6:]).strip()
    snippet = re.sub(r"\n\s*\n+", "\n", snippet)
    if len(snippet) > 500:
        snippet = snippet[:500].rstrip() + "…"
    return snippet or None


def _friendly_hint(error_text: str | None) -> str | None:
    """A plain-language, actionable headline for the common SO-101 failures."""
    if not error_text:
        return None
    low = error_text.lower()
    if "overload" in low or "torque_enable" in low:
        return (
            "A motor overloaded — usually the gripper holding an object too hard. Release the object / "
            "open the gripper and power-cycle the arm before trying again."
        )
    if "missing motor ids" in low or "motor check failed" in low:
        return (
            "A follower motor isn't responding (often the gripper, id 6). If a skill was holding an object "
            "it likely overloaded — remove it, power-cycle the arm, then try teleoperation first."
        )
    if "could not connect" in low or "failed to connect" in low or "not connected" in low:
        return "Couldn't connect to the arm — make sure it's plugged in, powered on, and on the right port."
    if "frame is too old" in low or "no frame" in low or "frame timeout" in low:
        return (
            "A camera can't keep up — frames are arriving too slowly. Lower its resolution/FPS, "
            "set FOURCC=MJPG, and close other heavy apps, then try again."
        )
    if "failed to set capture_" in low or "actual_width" in low or "actual_height" in low:
        return "A camera doesn't support the configured resolution — open camera settings and click Auto."
    if "permission" in low and ("port" in low or "com" in low):
        return "Couldn't open the serial port — close anything else using it, or run `lelab --stop`."
    if "relative-action" in low or "relative chunk actions" in low:
        return (
            "This policy was trained with relative actions and needs RTC (chunked) inference. "
            "Update lelab and try again — recent versions enable that automatically."
        )
    return None


# Errors that mean the policy actually ran and only shutdown/cleanup tripped —
# e.g. disabling torque on a gripper still holding an object. Connection-loss
# errors are deliberately excluded: a mid-run disconnect is a real failure.
_CLEANUP_MARKERS = ("overload", "torque_enable")


def _classify_outcome(rc: int | None, rollout_started: bool, error_text: str | None) -> str:
    """ok | ran_with_warning | failed.

    A non-zero exit *after* the rollout main loop started, where the error is a
    torque-disable/overload on shutdown, means the skill ran but a motor (usually
    the loaded gripper) complained during cleanup — that's a warning, not a
    failure, so the UI shouldn't call a working run "failed"."""
    if not rc:
        return "ok"
    low = (error_text or "").lower()
    if rollout_started and any(marker in low for marker in _CLEANUP_MARKERS):
        return "ran_with_warning"
    return "failed"


def _apply_v4l2_camera_controls(cameras: dict[str, dict[str, Any]]) -> None:
    """Same fix record.py applies after camera connect (shared lelab.v4l2 so
    train and inference see identical exposure/white balance). Controls are
    device-level, so setting them from here also fixes the frames the policy
    itself sees."""
    from .v4l2 import apply_camera_controls

    for cam_name, cam_cfg in cameras.items():
        idx = cam_cfg.get("camera_index")
        if not isinstance(idx, int):
            continue
        apply_camera_controls(f"/dev/video{idx}", cam_name)


def _inference_camera_capture(cameras: dict[str, dict[str, Any]]) -> None:
    """Background thread: open cameras via cv2 and capture latest frames.

    Runs only while inference is active. Opens each camera independently
    (secondary to the subprocess's primary open). If V4L2 doesn't allow
    shared access, the open fails silently and no preview is available.
    """
    global _inference_cam_thread
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not available; inference camera preview disabled")
        return

    for cam_key, cam_cfg in cameras.items():
        cam_idx = cam_cfg.get("camera_index", 0)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            logger.warning("Failed to open camera %s (index %s) for preview", cam_key, cam_idx)
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        _inference_cams[cam_key] = cap

    if not _inference_cams:
        return

    while not _inference_cam_stop.is_set():
        for cam_key, cap in list(_inference_cams.items()):
            ret, frame = cap.read()
            if not ret:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with _inference_cam_lock:
                    _inference_cam_frames[cam_key] = buf.tobytes()
        time.sleep(0.2)


def inference_camera_frames(cam_key: str, fps: float = 5.0):
    """Yield multipart-MJPEG chunks of inference camera frames.

    The capture thread starts once the current run is past its setup grace
    period (the subprocess must connect to the cameras first). The check lives
    inside the loop so previews also recover after an auto-restart between
    consecutive runs.
    """
    global _inference_cam_thread

    interval = 1.0 / fps
    last_mtime = 0.0
    while inference_active:
        if (
            _inference_cam_thread is None
            and _inference_rollout_started_at is not None
        ):
            # Only try to open the cameras once the inference subprocess has
            # connected them. UVC devices are single-opener: opening earlier
            # makes the preview thread hold the device and the subprocess
            # dies with EBUSY. After the subprocess owns the device our open
            # fails harmlessly and the preview simply stays blank.
            from . import rollout as _self

            _inference_cam_thread = threading.Thread(
                target=_self._inference_camera_capture,
                args=(_inference_meta.get("cameras", {}),),
                name="inference-camera-capture",
                daemon=True,
            )
            _inference_cam_thread.start()
        frame = None
        preview_file = _PREVIEW_DIR / f"{cam_key}.jpg"
        try:
            mtime = preview_file.stat().st_mtime
            if mtime > last_mtime and mtime >= (_inference_started_at or 0.0) - 1.0:
                frame = preview_file.read_bytes()
                last_mtime = mtime
        except OSError:
            frame = None
        if frame is None:
            with _inference_cam_lock:
                frame = _inference_cam_frames.get(cam_key)
        if frame is None:
            time.sleep(interval)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(interval)


def _stop_inference_cameras() -> None:
    """Stop camera capture thread and release all cameras."""
    global _inference_cam_thread
    _inference_cam_stop.set()
    if _inference_cam_thread:
        _inference_cam_thread.join(timeout=2.0)
        _inference_cam_thread = None
    for cap in _inference_cams.values():
        cap.release()
    _inference_cams.clear()
    _inference_cam_frames.clear()
    _inference_cam_stop.clear()


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Start a one-shot rollout subprocess. Returns a dict — the route
    layer turns it into a JSON response or HTTPException as appropriate."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config

    # Mutex with teleop and recording: all three drive the same serial bus.
    from . import record as _record, teleoperate as _teleoperate

    with _state_lock:
        if _teleoperate.teleoperation_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Teleoperation is currently active. Stop it first.",
            }
        if _record.recording_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Recording is currently active. Stop it first.",
            }
        if inference_active:
            return {
                "success": False,
                "status_code": 409,
                "message": "Inference is already active. Stop it first.",
            }
        # Claim the slot now so a concurrent caller losing the race sees us.
        inference_active = True

    # Opened partway through setup; the stdout pump thread takes ownership once
    # it starts. Tracked here so a failure before that hand-off can close it
    # instead of leaking the file handle (which on Windows also keeps the log
    # locked).
    log_handle = None
    try:
        # `setup_follower_calibration_file` returns the basename without the
        # .json extension. We need that stripped form for `--robot.id`,
        # because lerobot appends `.json` itself when constructing
        # `calibration_dir / f"{id}.json"`.
        follower_id = setup_follower_calibration_file(request.follower_config)
        policy_path = _resolve_policy_path(request.policy_ref)

        # Launch through lelab.smooth_rollout rather than lerobot's rollout
        # directly: the wrapper publishes the control loop's observation frames
        # to _PREVIEW_DIR (the web preview's only frame source — V4L2 is
        # single-opener so the server can't open the cameras itself) and applies
        # inter-chunk EMA smoothing when enabled. alpha 1.0 = smoothing off, but
        # the preview publisher runs regardless.
        smooth_alpha = (
            1.0
            if request.rtc
            else (1.0 - request.chunk_smooth_strength if request.chunk_smooth else 1.0)
        )

        cmd = [
            sys.executable,
            "-m",
            "lelab.smooth_rollout",
            f"--lelab.smooth_alpha={smooth_alpha}",
            "--strategy.type=base",
            f"--policy.path={policy_path}",
            f"--policy.device={_detect_device()}",
            "--robot.type=so101_follower",
            f"--robot.port={request.follower_port}",
            f"--robot.id={follower_id}",
            f"--task={request.task}",
            f"--duration={request.duration_s}",
            *_rollout_inference_args(policy_path, force_rtc=request.rtc),
        ]
        if request.cameras:
            cmd.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")

        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{int(time.time())}.log"
        log_handle = log_path.open("w", buffering=1)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Feed a single newline into stdin so SOFollower.calibrate()'s
        # `input("Press ENTER to use the calibration file ...")` returns "" and
        # writes the existing calibration to the motors instead of hanging
        # forever waiting for an interactive operator. Subsequent input()
        # calls in the recalibration path get EOF and raise — which is fine,
        # because we never want to enter that path from the UI.
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as exc:
            logger.warning("Failed to seed stdin for inference subprocess: %s", exc)
        threading.Thread(
            target=_pump_stdout,
            args=(proc, log_handle),
            name="inference-stdout-pump",
            daemon=True,
        ).start()
        log_handle = None  # pump thread owns and closes it from here on
    except Exception as exc:
        logger.exception("Failed to start inference")
        # Startup failed before the pump thread took over (most often Popen
        # itself). Release the slot and close the log file if we opened it before
        # failing, so the handle isn't leaked.
        if log_handle is not None:
            log_handle.close()
        with _state_lock:
            inference_active = False
        return {"success": False, "status_code": 500, "message": f"Failed to start inference: {exc}"}

    with _state_lock:
        _inference_proc = proc
        _inference_started_at = time.time()
        _inference_rollout_started_at = None
        _inference_meta = {
            "policy_ref": request.policy_ref,
            "duration_s": request.duration_s,
            "log_path": str(log_path),
            "cameras": dict(request.cameras),
        }
        _inference_current_run = 1
        _inference_remaining_runs = max(0, request.n_runs - 1)
        _inference_restart_config = {
            "follower_port": request.follower_port,
            "follower_config": request.follower_config,
            "policy_ref": request.policy_ref,
            "task": request.task,
            "cameras": dict(request.cameras),
            "duration_s": request.duration_s,
            "chunk_smooth": request.chunk_smooth,
            "chunk_smooth_strength": request.chunk_smooth_strength,
            "rtc": request.rtc,
        }
    logger.info("Inference started: pid=%s policy=%s runs=%d", proc.pid, policy_path, request.n_runs)
    return {"success": True, "message": "Inference started", "log_path": str(log_path)}


def handle_stop_inference() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config

    with _state_lock:
        if not inference_active or _inference_proc is None:
            return {"success": False, "status_code": 409, "message": "No inference is active"}
        proc = _inference_proc
        # Cancel any pending auto-restarts so a manual stop really stops all.
        _inference_remaining_runs = 0
        _inference_restart_config = {}

    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Inference did not exit in 5s; killing")
            proc.kill()
            proc.wait()
    except Exception as exc:
        logger.exception("Stop inference: %s", exc)

    with _state_lock:
        inference_active = False
        _inference_proc = None
        _inference_started_at = None
        _inference_rollout_started_at = None
        _inference_meta = {}
        _inference_current_run = 0
    _stop_inference_cameras()
    return {"success": True, "message": "Inference stopped"}


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config

    total_runs = _inference_current_run + _inference_remaining_runs

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            logger.info("Inference subprocess exited rc=%s (run %d/%d)",
                        rc, _inference_current_run, total_runs)
            finished_meta = _inference_meta
            finished_started = _inference_started_at
            finished_rollout_started = _inference_rollout_started_at
            _inference_proc = None
            _inference_started_at = None
            _inference_rollout_started_at = None
            _inference_meta = {}
            _stop_inference_cameras()

            # Auto-restart if there are remaining runs and this run succeeded
            # (or at least ran through the main loop).
            error = _extract_error_from_log(finished_meta.get("log_path")) if rc else None
            outcome = _classify_outcome(rc, finished_rollout_started is not None, error)
            if _inference_remaining_runs > 0 and outcome != "failed" and _inference_restart_config:
                _inference_remaining_runs -= 1
                next_current = _inference_current_run + 1
                remaining_after = _inference_remaining_runs
                logger.info("Auto-restarting inference (%d runs remaining)",
                            remaining_after)
                restart_req = InferenceRequest(**_inference_restart_config)
                # The previous subprocess has exited; free the slot so
                # handle_start_inference's active-guard accepts the restart
                # instead of returning 409. RLock makes the nested acquire safe.
                inference_active = False
                result = handle_start_inference(restart_req)
                if result.get("success"):
                    # handle_start_inference reset the run counters for a fresh
                    # session (current=1, remaining from its default n_runs);
                    # restore the continuation values for this multi-run series.
                    _inference_current_run = next_current
                    _inference_remaining_runs = remaining_after
                    return {
                        "inference_active": True,
                        "started_at": _inference_started_at,
                        "rollout_started_at": None,
                        "elapsed_s": 0,
                        "rollout_elapsed_s": 0,
                        "duration_s": restart_req.duration_s,
                        "policy_ref": restart_req.policy_ref,
                        "log_path": result.get("log_path"),
                        "current_run": next_current,
                        "total_runs": total_runs,
                        "camera_names": list(restart_req.cameras.keys()),
                    }
                logger.warning("Auto-restart failed: %s", result.get("message"))

            # All runs done or restart failed — finalise.
            inference_active = False
            _inference_restart_config = {}
            _inference_current_run = 0
            _inference_remaining_runs = 0
            return {
                "inference_active": False,
                "exited": True,
                "exit_code": rc,
                "outcome": outcome,
                "error": error,
                "hint": _friendly_hint(error),
                "policy_ref": finished_meta.get("policy_ref"),
                "duration_s": finished_meta.get("duration_s"),
                "log_path": finished_meta.get("log_path"),
                "started_at": finished_started,
                "rollout_started_at": finished_rollout_started,
                "rollout_elapsed_s": 0,
                "elapsed_s": 0,
                "current_run": 0,
                "total_runs": total_runs,
                "camera_names": list(finished_meta.get("cameras", {}).keys()),
            }
        elapsed = (time.time() - _inference_started_at) if _inference_started_at else 0
        rollout_elapsed = time.time() - _inference_rollout_started_at if _inference_rollout_started_at else 0
        return {
            "inference_active": inference_active,
            "started_at": _inference_started_at,
            "rollout_started_at": _inference_rollout_started_at,
            "elapsed_s": elapsed,
            "rollout_elapsed_s": rollout_elapsed,
            "duration_s": _inference_meta.get("duration_s"),
            "policy_ref": _inference_meta.get("policy_ref"),
            "log_path": _inference_meta.get("log_path"),
            "current_run": _inference_current_run,
            "total_runs": total_runs,
            "camera_names": list(_inference_meta.get("cameras", {}).keys()),
        }


def handle_inference_snapshot() -> dict[str, Any]:
    """Trigger a snapshot dump from the inference subprocess and wait for it.

    Writes a trigger file that the subprocess's snapshot watcher picks up;
    the watcher dumps the last ~30 ticks of (images, actions, states) to a
    timestamped directory. We poll for the .done marker and return the
    snapshot path + manifest summary so the UI can confirm capture."""
    import uuid

    if not inference_active:
        return {"success": False, "status_code": 409, "message": "No inference is active"}

    token = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    trigger_file = _PREVIEW_DIR / "snapshot_request"
    snapshot_dir = _SNAPSHOT_DIR / token

    try:
        _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        trigger_file.write_text(token)
    except OSError as exc:
        return {"success": False, "status_code": 500, "message": f"Failed to write trigger: {exc}"}

    done_marker = snapshot_dir / ".done"
    manifest_path = snapshot_dir / "manifest.json"
    for _ in range(30):
        time.sleep(0.2)
        if done_marker.exists():
            n_ticks = 0
            try:
                n_ticks = int(done_marker.read_text().strip())
            except Exception:
                pass
            summary = {"n_ticks": n_ticks}
            if manifest_path.exists():
                try:
                    summary["manifest"] = json.loads(manifest_path.read_text())
                except Exception:
                    pass
            return {
                "success": True,
                "snapshot_dir": str(snapshot_dir),
                "n_ticks": n_ticks,
                "summary": summary,
            }

    return {"success": False, "status_code": 504, "message": "Snapshot timed out (subprocess may not be running)"}
