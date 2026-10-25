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

from .utils.config import setup_calibration_files, setup_follower_calibration_file

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
    # Data collection during rollout. 'base' runs the policy without recording;
    # the others select a lerobot rollout strategy that writes episodes.
    capture_mode: str = "base"  # base | dagger | highlight | sentry
    dataset_repo_id: str = ""  # 'user/rollout_name'; empty = auto-name
    dataset_fps: int = 30
    num_episodes: int = 20
    # dagger only: record the policy frames too, so one episode holds both the
    # model's attempt and the human correction (hybrid episode).
    record_autonomous: bool = False
    leader_port: str = ""
    leader_config: str = ""


class TakeoverRequest(BaseModel):
    action: str  # dagger: pause | correct | stop;  highlight: save | push | stop


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
# How long the rollout's preview publisher may stay quiet before the server
# risks opening the cameras itself (see inference_camera_frames).
_PREVIEW_STALL_S = 8.0
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


_HUB_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _hub_repo_ids_in(obj: Any) -> set[str]:
    """Collect every ``org/name`` string anywhere in a policy config."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for value in obj.values():
            found |= _hub_repo_ids_in(value)
    elif isinstance(obj, list):
        for value in obj:
            found |= _hub_repo_ids_in(value)
    elif isinstance(obj, str) and _HUB_REPO_ID_RE.match(obj):
        found.add(obj)
    return found


def _hf_offline_env(policy_path: str) -> dict[str, str]:
    """Env vars pinning the rollout subprocess to the local HF cache.

    While building the SmolVLM processor, transformers HEAD-probes
    huggingface.co once per config file it resolves — 31 requests, ~22s of the
    ~50s setup wall, and minutes when the hub is unreachable from the robot's
    network. Offline mode drops the probes entirely.

    Only applied when every repo the policy config names is already cached:
    an uncached base would then be a hard failure rather than a download, so
    we leave those policies online.
    """
    repos = _hub_repo_ids_in(_read_policy_config(policy_path))
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return {}
    for repo in sorted(repos):
        try:
            snapshot_download(repo_id=repo, local_files_only=True)
        except Exception:
            logger.info("HF repo '%s' not cached; inference subprocess stays online", repo)
            return {}
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


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
    """Convert {name: {type, camera_index|url, width, height, fps}} into
    lerobot's CLI dict syntax.

    Two things differ from a plain key copy. The frontend key `camera_index` is
    remapped to lerobot's `index_or_path`, and a stream URL has to go out as
    `type: network`: the `opencv` config types that field `int | Path`, so draccus
    coerces the URL into a Path and silently collapses `rtsp://host/live` to
    `rtsp:/host/live`.
    """
    from .network_camera import camera_source

    parts = []
    for name, cfg in cameras.items():
        kind, source = camera_source(cfg)
        if kind == "network":
            fields = {"type": "network", "index_or_path": f'"{source}"'}
            for key in ("width", "height", "fps"):
                if cfg.get(key) is not None:
                    fields[key] = cfg[key]
            body = ", ".join(f"{k}: {v}" for k, v in fields.items())
            parts.append(f"{name}: {{{body}}}")
            continue

        remapped = {
            ("index_or_path" if k == "camera_index" else k): v
            for k, v in cfg.items()
            if v is not None and k != "url"
        }
        body = ", ".join(f"{k}: {v}" for k, v in remapped.items())
        parts.append(f"{name}: {{{body}}}")
    return "{" + ", ".join(parts) + "}"


# ---------------------------------------------------------------------------
# Rollout data collection (lerobot rollout strategies)
# ---------------------------------------------------------------------------

_CAPTURE_MODES = ("base", "dagger", "highlight", "sentry")
# UI verbs mapped to the canonical key names each strategy's listener expects.
# The keys travel to the subprocess through a trigger file because a Popen'd
# rollout has neither a TTY nor a global keyboard hook.
_MODE_KEYS: dict[str, dict[str, str]] = {
    "dagger": {"pause": "space", "correct": "tab", "stop": "esc"},
    "highlight": {"save": "s", "stop": "esc"},
}
_TAKEOVER_KEY_FILE = "takeover_key"
_TAKEOVER_STATE_FILE = "takeover_state.json"


def _read_takeover_state() -> dict[str, Any]:
    """Return the phase/episode counters the rollout subprocess publishes."""
    try:
        state = json.loads((_PREVIEW_DIR / _TAKEOVER_STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _hf_default_user() -> str | None:
    from .utils.hf_auth import cached_whoami

    info = cached_whoami()
    return info.get("name") if info else None


def _read_dataset_info(root: Path) -> dict[str, Any] | None:
    """Existing dataset's meta/info.json, or None when the dataset is new."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _detect_leader(request: "InferenceRequest") -> tuple[str, str]:
    """Resolve the leader arm for takeover from the saved robot records.

    The inference dialog only picks the follower (the policy drives it), but a
    correction session needs the leader as well. Both ports live in the robot
    record the follower came from, so asking again would only duplicate it.
    """
    if request.leader_port and request.leader_config:
        return request.leader_port, request.leader_config
    from .utils.config import list_robot_records

    try:
        records = list_robot_records()
    except Exception as exc:
        logger.warning("Leader autodetection failed: %s", exc)
        return request.leader_port, request.leader_config
    match = next((r for r in records if r.get("follower_port") == request.follower_port), None)
    if match is None and len(records) == 1:
        match = records[0]
    if match is None:
        return request.leader_port, request.leader_config
    return str(match.get("leader_port") or ""), str(match.get("leader_config") or "")


def _build_capture_args(request: "InferenceRequest") -> tuple[list[str], dict[str, Any], str | None]:
    """Translate the capture request into ``lerobot-rollout`` flags.

    Returns (flags, session meta, error). lerobot's rollout engine only records
    through a strategy, so a capture mode always carries ``--dataset.*``; the
    'base' strategy rejects a dataset, hence base mode stays flag-free.
    """
    mode = (request.capture_mode or "base").strip().lower()
    if mode not in _CAPTURE_MODES:
        return [], {}, f"Unknown capture mode '{mode}'."
    if mode == "base":
        return [], {"capture_mode": "base"}, None

    from lerobot.utils.constants import HF_LEROBOT_HOME

    name = request.dataset_repo_id.strip()
    user = _hf_default_user()
    if not name:
        if not user:
            return [], {}, "No HF account detected, so the dataset can't be auto-named. Type a dataset repo id."
        # Day-stable, so repeated rounds of one collection session land in the
        # same dataset instead of forking a new one per launch.
        name = f"{user}/rollout_{time.strftime('%m%d')}"
    if "/" not in name:
        if not user:
            return [], {}, f"Dataset '{name}' needs a user/org prefix."
        name = f"{user}/{name}"
    org, _, tail = name.partition("/")
    root = Path(HF_LEROBOT_HOME) / name
    existing = _read_dataset_info(root)
    # A dataset already on disk keeps its name — lerobot only enforces the
    # rollout_* prefix when creating (and stamps a timestamp onto new names).
    if existing is None and not tail.startswith("rollout_"):
        name = f"{org}/rollout_{tail}"
        root = Path(HF_LEROBOT_HOME) / name
        existing = _read_dataset_info(root)

    args = [
        f"--dataset.repo_id={name}",
        f"--dataset.root={root}",
        f"--dataset.single_task={request.task}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={request.num_episodes}",
    ]
    resolved_leader = ""
    if existing is not None:
        # Appending to a dataset already on disk. Its fps wins: one fps per
        # dataset, and add_frame derives timestamps from it.
        args.append(f"--dataset.fps={existing.get('fps') or request.dataset_fps}")
        args.append("--resume=true")
        if mode == "dagger" and "intervention" not in (existing.get("features") or {}):
            return [], {}, (
                f"'{name}' was recorded with plain teleop and has no 'intervention' column, "
                "which correction episodes need. Pick a new dataset name (it gets created with "
                "that column) or convert the old one first."
            )
        if mode != "dagger" and "intervention" in (existing.get("features") or {}):
            # The other strategies never write that column, and validate_frame
            # rejects a declared feature missing from a frame.
            return [], {}, (
                f"'{name}' 是纠正采集建的（带 intervention 列），"
                f"{mode} 模式不写这一列，逐帧校验会失败。请换一个新的数据集名。"
            )
    else:
        args.append(f"--dataset.fps={request.dataset_fps}")

    if mode == "dagger":
        resolved_leader, leader_config = _detect_leader(request)
        if not resolved_leader or not leader_config:
            return [], {}, "纠正采集需要主臂：在机器人配置里保存 leader 端口与标定后重试。"
        try:
            leader_id, _ = setup_calibration_files(leader_config, request.follower_config)
        except (FileNotFoundError, OSError) as exc:
            return [], {}, f"Leader calibration unavailable: {exc}"
        args += [
            "--teleop.type=so101_leader",
            f"--teleop.port={resolved_leader}",
            f"--teleop.id={leader_id}",
            "--strategy.type=dagger",
            f"--strategy.num_episodes={request.num_episodes}",
            f"--strategy.record_autonomous={'true' if request.record_autonomous else 'false'}",
        ]
        if request.record_autonomous:
            args += ["--dataset.streaming_encoding=true", "--dataset.encoder_threads=2"]
    elif mode == "highlight":
        args += ["--strategy.type=highlight", "--strategy.ring_buffer_seconds=15"]
    else:
        args += ["--strategy.type=sentry", "--strategy.upload_every_n_episodes=5"]

    return (
        args,
        {
            "capture_mode": mode,
            "dataset_repo_id": name,
            "dataset_root": str(root),
            "appending": existing is not None,
            "num_episodes": request.num_episodes,
            "record_autonomous": bool(request.record_autonomous),
            "leader_port": resolved_leader,
        },
        None,
    )


def handle_inference_takeover(action: str) -> dict[str, Any]:
    """Hand a control action to the running rollout's data-collection strategy.

    'action' is a mode-specific verb (dagger: pause/correct/stop; highlight:
    save/push/stop); it becomes a canonical key name in the trigger file the
    subprocess polls."""
    if not inference_active:
        return {"success": False, "status_code": 409, "message": "No inference is active"}

    mode = str(_inference_meta.get("capture_mode") or "base")
    verbs = _MODE_KEYS.get(mode)
    if not verbs:
        return {
            "success": False,
            "status_code": 409,
            "message": f"Capture mode '{mode}' has no takeover controls.",
        }
    key = verbs.get(action)
    if key is None:
        return {"success": False, "status_code": 400, "message": f"Unknown action '{action}' for '{mode}'."}

    try:
        _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        (_PREVIEW_DIR / _TAKEOVER_KEY_FILE).write_text(key)
    except OSError as exc:
        return {"success": False, "status_code": 500, "message": f"Failed to write takeover trigger: {exc}"}

    # Taking over is precisely the moment the policy went wrong, and the ring
    # buffer still holds what it *saw* and what it *chose* to do — the one piece
    # a correction episode doesn't record. Archive it automatically so the
    # divergence point survives without the operator remembering a second button.
    if action == "pause":
        threading.Thread(target=handle_inference_snapshot, daemon=True).start()

    return {"success": True, "message": f"Sent '{key}' to the {mode} strategy", "key": key}


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
    Stream sources are the exception — RTSP takes several client sessions, so
    their preview always works and reopens itself after a dropped frame.
    """
    global _inference_cam_thread
    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not available; inference camera preview disabled")
        return

    from .network_camera import camera_source, preview_stream, redact

    for cam_key, cam_cfg in cameras.items():
        kind, source = camera_source(cam_cfg)
        if kind == "network":
            _inference_cams[cam_key] = preview_stream(str(source))
            logger.info("🌐 Preview stream opened for %s (%s)", cam_key, redact(source))
            continue

        cam_idx = source if source is not None else 0
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


def _start_inference_camera_capture() -> None:
    """Launch the secondary cv2 capture thread (fallback preview source).

    Only ever called when the rollout subprocess stopped publishing preview
    files: UVC devices here are single-opener, so grabbing the camera away from
    a healthy subprocess makes the rollout itself drop frames.
    """
    global _inference_cam_thread
    if _inference_cam_thread is not None:
        return
    from . import rollout as _self

    _inference_cam_thread = threading.Thread(
        target=_self._inference_camera_capture,
        args=(_inference_meta.get("cameras", {}),),
        name="inference-camera-capture",
        daemon=True,
    )
    _inference_cam_thread.start()


def inference_camera_frames(cam_key: str, fps: float = 5.0):
    """Yield multipart-MJPEG chunks of inference camera frames.

    Primary source is the preview file the rollout subprocess publishes from its
    own observation frames. Once a frame has arrived it keeps being re-sent, so
    a brief pause in the writer shows as a still image instead of a blank tile —
    a silent stream is indistinguishable from a dead one in the browser.

    The secondary cv2 open (which loses on this machine, but is the only option
    when the subprocess never publishes) is deliberately delayed: the cameras
    must stay owned by the rollout while it is producing frames.
    """
    interval = 1.0 / fps
    preview_file = _PREVIEW_DIR / f"{cam_key}.jpg"
    last_mtime = 0.0
    last_frame: bytes | None = None
    last_new_frame_at = 0.0

    while inference_active:
        frame: bytes | None = None
        try:
            mtime = preview_file.stat().st_mtime
            if mtime >= (_inference_started_at or 0.0) - 1.0 and mtime > last_mtime:
                last_frame = preview_file.read_bytes()
                last_mtime = mtime
                last_new_frame_at = time.time()
        except OSError:
            pass
        frame = last_frame

        if frame is None:
            with _inference_cam_lock:
                frame = _inference_cam_frames.get(cam_key)

        if frame is None:
            started = _inference_rollout_started_at or 0.0
            if started and time.time() - started > _PREVIEW_STALL_S:
                _start_inference_camera_capture()
            time.sleep(interval)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(interval)
        # The writer went quiet: we are replaying a still image, so try the
        # secondary open to get a live picture back.
        if (
            _inference_cam_thread is None
            and last_new_frame_at
            and time.time() - last_new_frame_at > _PREVIEW_STALL_S
        ):
            _start_inference_camera_capture()


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

        capture_args, capture_meta, capture_error = _build_capture_args(request)
        if capture_error:
            with _state_lock:
                inference_active = False
            return {"success": False, "status_code": 400, "message": capture_error}

        # Setting up the leader bus and the writer threads collides with the
        # run-restart loop below, so a capture session is always a single run.
        if capture_meta.get("capture_mode", "base") != "base":
            request.n_runs = 1

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
            f"--strategy.type={capture_meta.get('capture_mode', 'base')}",
            f"--policy.path={policy_path}",
            f"--policy.device={_detect_device()}",
            "--robot.type=so101_follower",
            f"--robot.port={request.follower_port}",
            f"--robot.id={follower_id}",
            f"--task={request.task}",
            f"--duration={request.duration_s}",
            *_rollout_inference_args(policy_path, force_rtc=request.rtc),
            *capture_args,
        ]
        if request.cameras:
            cmd.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")

        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{int(time.time())}.log"
        log_handle = log_path.open("w", buffering=1)

        # Stale bridge files from a previous run would make the subprocess act
        # on an old button press or report an old phase.
        with contextlib.suppress(OSError):
            _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for stale in (_TAKEOVER_KEY_FILE, _TAKEOVER_STATE_FILE):
                (_PREVIEW_DIR / stale).unlink(missing_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(_hf_offline_env(policy_path))
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
            **capture_meta,
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
            "capture_mode": request.capture_mode,
            "dataset_repo_id": request.dataset_repo_id,
            "dataset_fps": request.dataset_fps,
            "num_episodes": request.num_episodes,
            "record_autonomous": request.record_autonomous,
            "leader_port": request.leader_port,
            "leader_config": request.leader_config,
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


def _capture_status_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Capture-session fields every status response carries, so the UI can show
    which dataset is being written and which strategy phase we're in."""
    return {
        "capture_mode": meta.get("capture_mode", "base"),
        "dataset_repo_id": meta.get("dataset_repo_id"),
        "takeover": _read_takeover_state(),
    }


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
                        **_capture_status_fields(_inference_meta),
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
                **_capture_status_fields(finished_meta),
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
            **_capture_status_fields(_inference_meta),
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
