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

from .utils.config import (
    LEADER_CONFIG_PATH,
    setup_calibration_files,
    setup_follower_calibration_file,
)

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
    # RTC guidance strength (max_guidance_weight). 10.0 is the tuned default;
    # higher steers the executed chunk harder toward the fresh prediction.
    rtc_guidance: float = 10.0
    # Data collection during rollout. 'base' runs the policy without recording;
    # the others select a lerobot rollout strategy that writes episodes.
    capture_mode: str = "base"  # base | dagger | highlight | sentry
    dataset_repo_id: str = ""  # 'user/rollout_name'; empty = auto-name
    dataset_fps: int = 30
    num_episodes: int = 20
    # dagger only: record the policy frames too, so one episode holds both the
    # model's attempt and the human correction (hybrid episode).
    record_autonomous: bool = False
    # dagger only: a correction left open would hang the rig, since `duration_s`
    # now budgets policy driving time rather than wall clock.
    max_correction_s: int = 30
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
# Counts automatic retries after a run dies during setup (before its main loop).
# Serial/camera bring-up occasionally drops one packet ("no status packet" /
# "failed to open camera"); those are transient and self-heal on a fresh process.
_inference_setup_retries: int = 0
_inference_restart_config: dict[str, Any] = {}
# Set while "this round succeeded, start the next" is in flight: the status poll
# must not race it for the restart, and the page must not bounce home during the
# few seconds the subprocess takes to finalise and respawn.
_inference_end_in_progress: bool = False
# After a run exits, the kernel can still hold the serial port / V4L2 cameras
# and the dying process's host+GPU memory may not be reclaimed for a moment.
# Respawning the next run immediately intermittently kills it during native
# init — before Python prints anything, leaving a 0-byte log with no error.
# A fixed cooldown alone proved insufficient (a 3rd consecutive run still died),
# and overlapping processes can trip the userspace OOM killer, so we also wait
# for host memory to recover.
_RESTART_COOLDOWN_S = 6.0
_RESTART_MEM_WAIT_S = 20.0
_RESTART_MIN_AVAIL_FRAC = 0.5
# When the last rollout exited, so a manual start right after a teardown can
# wait for the cameras/serial port to be released instead of dying in native
# init with a 0-byte log.
_inference_last_exit_at: float = 0.0
# Rest between continuous rounds — long enough to walk home, short enough that
# the operator can reset the scene without the run looking stuck.
_ROUND_REST_S = 3.0
# Estimated time the strategy spends interpolating home between rounds; only
# used to size the reported total duration so the hung detector doesn't fire.
_ROUND_HOME_S = 3.0
# How many times to auto-retry a run that died during setup (transient serial/
# camera bring-up glitches) before reporting failure.
_MAX_SETUP_RETRIES = 3
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
    global _inference_rollout_started_at, _inference_setup_retries
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
                # Setup succeeded — reset the transient-failure retry budget.
                _inference_setup_retries = 0
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


def _rollout_inference_args(
    policy_path: str, force_rtc: bool = False, rtc_guidance: float = 10.0
) -> list[str]:
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
        # RTC guides only the first `execution_horizon` steps of a new chunk, and
        # the queue drops the first `ceil(latency / frame_period)` steps because
        # the robot already spent that long executing the old chunk. On this rig
        # SmolVLA takes ~0.35s per chunk ≈ 11 steps at 30fps, so a horizon of 10
        # was discarded entirely and RTC had nothing left to steer. Keep the
        # horizon comfortably above the measured delay and let the queue run down
        # to it before replanning.
        args = [
            "--inference.type=rtc",
            "--inference.rtc.execution_horizon=20",
            f"--inference.rtc.max_guidance_weight={max(0.5, float(rtc_guidance))}",
            "--inference.queue_threshold=20",
        ]
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
    `rtsp:/host/live`. Exposure and gain are V4L2 device controls, not lerobot
    config fields, so they are dropped here and applied via lelab.v4l2 instead.
    """
    from .network_camera import camera_source

    _v4l2_only = ("url", "exposure", "gain")
    parts = []
    for name, cfg in cameras.items():
        kind, source = camera_source(cfg)
        if kind == "network":
            fields = {"type": "network", "index_or_path": f'"{source}"'}
            for key in ("width", "height", "fps", "rotation"):
                if cfg.get(key) is not None:
                    fields[key] = cfg[key]
            body = ", ".join(f"{k}: {v}" for k, v in fields.items())
            parts.append(f"{name}: {{{body}}}")
            continue

        remapped = {
            ("index_or_path" if k == "camera_index" else k): v
            for k, v in cfg.items()
            if v is not None and k not in _v4l2_only
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
    "dagger": {
        "pause": "space",
        "correct": "tab",
        "stop": "esc",
        # LeLab-only verbs, handled by the subprocess bridge: 'start' opens the
        # next episode from the rest screen; 'success' commits the take and
        # 'fail' drops it (counting one failure). Both park the session back on
        # the rest screen with the arm at its start pose.
        "start": "start",
        "takeover": "takeover",
        "success": "success",
        "fail": "fail",
    },
    "highlight": {"save": "s", "stop": "esc"},
}
_TAKEOVER_KEY_FILE = "takeover_key"
_TAKEOVER_STATE_FILE = "takeover_state.json"
# Dropped by the server to end ONLY the current round of a continuous base
# session; the subprocess returns home, rests and starts the next round in the
# same process (no model reload).
_ROUND_END_FILE = "round_end"


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
    """Existing dataset's meta/info.json, or None when the dataset is new.

    "Existing" means lerobot can load it from disk alone. A directory holding
    only `info.json` is a stub left by a run that died between creating the
    dataset and saving its first episode; the metadata loader rejects it and
    falls back to pulling `meta/` from the Hub, which an offline inference
    subprocess turns into SIGABRT (-6) with nothing but a traceback in the log.
    Treating a stub as new lets the capture recreate it.
    """
    meta = root / "meta"
    if not (meta / "info.json").is_file():
        return None
    tasks = (meta / "tasks.parquet").is_file() or (meta / "tasks.jsonl").is_file()
    stats = (meta / "stats.json").is_file() or (meta / "episodes_stats.jsonl").is_file()
    episodes = (meta / "episodes").is_dir() or (meta / "episodes.jsonl").is_file()
    if not (tasks and stats and episodes):
        logger.info("Dataset %s holds only a stub meta/ — recreating it", root)
        return None
    try:
        info = json.loads((meta / "info.json").read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _clear_dataset_stub(root: Path) -> bool:
    """Move a failed run's leftover dataset directory aside.

    lerobot's create() uses ``exist_ok=False``, so a stub holding only a
    half-written ``meta/`` blocks the name forever — and it cannot be resumed
    either: with no tasks/stats/episodes there is nothing to load. Renaming it
    (rather than deleting) keeps whatever the run did write recoverable.
    Refuses to touch anything that might hold real episodes.
    """
    try:
        entries = {p.name for p in root.iterdir()}
        meta_entries = {p.name for p in (root / "meta").iterdir()} if (root / "meta").is_dir() else set()
    except OSError:
        return False
    if entries - {"meta", ".cache"} or meta_entries - {"info.json"}:
        return False
    aside = root.with_name(f"{root.name}.stub-{time.strftime('%Y%m%d_%H%M%S')}")
    try:
        root.rename(aside)
    except OSError:
        return False
    logger.info("Dataset %s held only a stub meta/ — moved aside to %s", root, aside.name)
    return True


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
    # Capture always runs with push_to_hub=false, so the org segment is only a
    # local directory namespace — fall back to "local" when not signed into HF
    # instead of refusing to start.
    user = _hf_default_user() or "local"
    if not name:
        # Day-stable, so repeated rounds of one collection session land in the
        # same dataset instead of forking a new one per launch.
        name = f"{user}/rollout_{time.strftime('%m%d')}"
    if "/" not in name:
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
    if existing is None:
        # A rollout session stamps a timestamp onto the repo id at create time,
        # so an earlier round of this session may live in a "<name>_<stamp>"
        # directory. Reuse the newest one instead of silently forking a sibling.
        stamped = re.compile(rf"^{re.escape(tail)}_\d{{8}}_\d{{6}}$")
        candidates = []
        with contextlib.suppress(OSError):
            candidates = sorted(
                (p for p in root.parent.glob(f"{tail}_*") if stamped.match(p.name)),
                key=lambda p: p.name,
            )
        for cand in reversed(candidates):
            info = _read_dataset_info(cand)
            if info is not None:
                name, root, existing = f"{org}/{cand.name}", cand, info
                break

    if existing is None and root.exists() and not _clear_dataset_stub(root):
        return [], {}, (
            f"数据集目录 {root} 已存在但元数据不完整（上次启动中断留下的空壳），"
            "lerobot 无法在其中新建。请换一个数据集名字，或手动删掉该目录后重试。"
        )

    # 段数上限 is the dataset's TOTAL, not this process's budget: a resumed
    # dataset only has to record what it is still missing. lerobot counts the
    # episodes it saves itself, so hand it the remainder.
    already = int((existing or {}).get("total_episodes") or 0)
    remaining = max(1, request.num_episodes - already)
    if already:
        logger.info("Resuming %s with %d/%d episodes already saved", name, already, request.num_episodes)

    # lerobot's own loop closes the session once it has saved its target, so
    # `remaining` is the take budget for ONE process: the operator marks each
    # take success/failure and the recorder rolls straight into the next one
    # without reloading the model.
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
        # Pre-flight: the rollout subprocess runs with stdin closed, so it
        # cannot complete lerobot's interactive leader calibration. If the
        # leader has no usable calibration under the id we're about to pass,
        # lerobot falls through to the interactive path and the subprocess
        # aborts (exit -6) against a 0-byte log. Catch it here with a message.
        leader_cal = Path(LEADER_CONFIG_PATH) / f"{leader_id}.json"
        if not leader_cal.is_file():
            return [], {}, (
                f"主臂(id={leader_id})没有标定文件，纠正采集无法在无交互的子进程里完成标定。"
                f"请先在遥操/标定界面标定主臂（生成 {leader_cal.name}），"
                f"或在部署时重新选择已标定的机器人（如 001）后重试。"
            )
        # A degenerate calibration (range_min == range_max, e.g. all-2047 from a
        # prior failed headless attempt) trips the same abort — reject it too.
        try:
            _data = json.loads(leader_cal.read_text())
            _bad = [
                name for name, m in _data.items()
                if isinstance(m, dict)
                and m.get("range_min") is not None
                and m.get("range_min") == m.get("range_max")
            ]
            if _bad:
                return [], {}, (
                    f"主臂标定 {leader_cal.name} 里这些关节 range_min==range_max（{', '.join(_bad)}），"
                    f"像是上次无交互标定写坏的。请在遥操/标定界面重新标定主臂后重试。"
                )
        except (ValueError, OSError):
            pass  # unreadable → let lerobot surface it at connect time
        args += [
            "--teleop.type=so101_leader",
            f"--teleop.port={resolved_leader}",
            f"--teleop.id={leader_id}",
            "--strategy.type=dagger",
            f"--strategy.num_episodes={remaining}",
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
    if action in ("pause", "takeover"):
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


# Failures during bring-up that commonly self-heal on a fresh process: a servo
# drops one packet while torque is enabled, or a camera is still busy from the
# previous run.
_TRANSIENT_SETUP_MARKERS = (
    "no status packet",
    "failed to write",
    # A bus that just re-enumerated (USB reset/replug) can miss one id on the
    # handshake. The servo answers on the next connect; a retry is enough.
    "motor check failed",
    "missing motor ids",
    "failed to open opencvcamera",
    "timed out waiting for frame",
    "read thread alive",
    "connection reset",
    "resource temporarily unavailable",
)


def _error_is_transient(error_text: str | None) -> bool:
    low = (error_text or "").lower()
    return any(marker in low for marker in _TRANSIENT_SETUP_MARKERS)


def _apply_v4l2_camera_controls(cameras: dict[str, dict[str, Any]]) -> None:
    """Same fix record.py applies after camera connect (shared lelab.v4l2 so
    train and inference see identical exposure/white balance). Controls are
    device-level, so setting them from here also fixes the frames the policy
    itself sees."""
    from .v4l2 import apply_camera_controls, configure_manual_controls

    configure_manual_controls(cameras)

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
            _inference_cams[cam_key] = preview_stream(
                str(source), rotation=int(cam_cfg.get("rotation", 0) or 0)
            )
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

    Frames come only from the preview JPEG the rollout subprocess publishes —
    the server never opens the cameras itself, because that races the
    subprocess's exclusive V4L2 open.
    """
    interval = 1.0 / fps
    preview_file = _PREVIEW_DIR / f"{cam_key}.jpg"
    last_mtime = 0.0
    last_frame: bytes | None = None

    while inference_active:
        frame: bytes | None = None
        try:
            mtime = preview_file.stat().st_mtime
            # Seed with the first available frame regardless of age: opening the
            # MJPEG response silently and then staying quiet through the ~15–20s
            # setup phase makes browsers show a blank tile (some drop the
            # connection and never reconnect). After the seed, only fresh frames
            # (written since this run started) replace it.
            if mtime > last_mtime and (
                last_frame is None or mtime >= (_inference_started_at or 0.0) - 1.0
            ):
                last_frame = preview_file.read_bytes()
                last_mtime = mtime
        except OSError:
            pass
        frame = last_frame

        if frame is None:
            with _inference_cam_lock:
                frame = _inference_cam_frames.get(cam_key)

        if frame is None:
            # Do NOT fall back to opening the cameras from the server here: the
            # rollout subprocess owns them exclusively, and a secondary open
            # races its own connect() — that race is what made every
            # auto-restarted run fail with "Failed to open OpenCVCamera(N)"
            # (the browser keeps the preview <img> open, so at run 2 the
            # generator immediately hits the stall timeout and grabs the
            # device first). The subprocess publishes preview JPEGs; during the
            # setup phase just wait for the first one.
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


def _wait_for_restart_ready(prev_proc: subprocess.Popen | None = None) -> None:
    """Pause before respawning so the previous run's resources are free.

    Fixed sleep alone was not enough (a 3rd consecutive run still died with a
    0-byte log). After the floor we poll MemAvailable so an overlapping respawn
    neither fights for the serial port/cameras nor gets SIGKILLed by the
    userspace OOM killer.
    """
    time.sleep(_RESTART_COOLDOWN_S)
    if prev_proc is not None:
        with contextlib.suppress(Exception):
            prev_proc.wait(timeout=5)
    try:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return
        deadline = time.time() + _RESTART_MEM_WAIT_S
        while time.time() < deadline:
            total = available = 0
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1])
            if total and available / total >= _RESTART_MIN_AVAIL_FRAC:
                break
            time.sleep(1.0)
    except Exception:
        pass


def handle_start_inference(request: InferenceRequest) -> dict[str, Any]:
    """Start a one-shot rollout subprocess. Returns a dict — the route
    layer turns it into a JSON response or HTTPException as appropriate."""
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config
    global _inference_setup_retries, _inference_last_exit_at

    # A fresh operator-initiated session (no run in flight) resets the transient
    # setup-retry budget; internal retries/restarts keep their count.
    if _inference_current_run == 0:
        _inference_setup_retries = 0

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

        _mode = capture_meta.get("capture_mode", "base")
        # Base inference runs every requested round inside ONE subprocess so the
        # model loads (and hardware connects) only once. Dagger already loops
        # episodes internally, so only base uses this continuous-rounds path.
        base_rounds = request.n_runs if (_mode == "base" and request.n_runs > 1) else 1

        # Capture sessions may span rounds: a restart reuses the same dataset
        # because _build_capture_args resolves the stamped directory lerobot
        # created on the first round.

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
            # A capture session is bounded by 段数上限 and the per-episode
            # ceiling, not by wall clock: the rest screen waits on the operator.
            f"--duration={0 if _mode == 'dagger' else request.duration_s}",
            *_rollout_inference_args(
                policy_path, force_rtc=request.rtc, rtc_guidance=request.rtc_guidance
            ),
            *capture_args,
        ]
        if request.cameras:
            cmd.append(f"--robot.cameras={_format_cameras_arg(request.cameras)}")
        if base_rounds > 1:
            cmd.extend([f"--lelab.rounds={base_rounds}", f"--lelab.rest_s={_ROUND_REST_S}"])
        if _mode == "dagger":
            cmd.extend(
                [
                    f"--lelab.max_correction_s={request.max_correction_s}",
                    # The modal's 时长 means "per episode" while capturing.
                    f"--lelab.per_episode_s={request.duration_s}",
                ]
            )

        log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{int(time.time())}.log"
        log_handle = log_path.open("w", buffering=1)

        # Stale bridge files from a previous run would make the subprocess act
        # on an old button press or report an old phase.
        with contextlib.suppress(OSError):
            _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
            for stale in (_TAKEOVER_KEY_FILE, _TAKEOVER_STATE_FILE, _ROUND_END_FILE):
                (_PREVIEW_DIR / stale).unlink(missing_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.update(_hf_offline_env(policy_path))
        # A rollout that exited moments ago can still own the cameras and the
        # serial port; starting on top of it dies in native init before Python
        # prints anything, which the page can only report as an empty log.
        since_exit = time.time() - _inference_last_exit_at
        if 0 <= since_exit < _RESTART_COOLDOWN_S:
            wait_s = _RESTART_COOLDOWN_S - since_exit
            logger.info("Previous rollout exited %.1fs ago — waiting %.1fs before start", since_exit, wait_s)
            time.sleep(wait_s)
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
        total_duration = (
            base_rounds * (request.duration_s + _ROUND_REST_S + _ROUND_HOME_S)
            if base_rounds > 1
            else request.duration_s
        )
        _inference_meta = {
            "policy_ref": request.policy_ref,
            "duration_s": total_duration,
            "rounds": base_rounds,
            "round_duration": request.duration_s,
            "rest_s": _ROUND_REST_S,
            "log_path": str(log_path),
            "cameras": dict(request.cameras),
            **capture_meta,
        }
        _inference_current_run = 1
        # A continuous-round session runs every round in its own subprocess, so
        # the server has nothing to restart.
        _inference_remaining_runs = 0 if base_rounds > 1 else max(0, request.n_runs - 1)
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
            "rtc_guidance": request.rtc_guidance,
            "capture_mode": request.capture_mode,
            "dataset_repo_id": request.dataset_repo_id,
            "dataset_fps": request.dataset_fps,
            "num_episodes": request.num_episodes,
            "record_autonomous": request.record_autonomous,
            "max_correction_s": request.max_correction_s,
            "leader_port": request.leader_port,
            "leader_config": request.leader_config,
        }
    logger.info("Inference started: pid=%s policy=%s runs=%d", proc.pid, policy_path, request.n_runs)
    return {"success": True, "message": "Inference started", "log_path": str(log_path)}


def handle_stop_inference() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config
    global _inference_last_exit_at

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
    finally:
        _inference_last_exit_at = time.time()

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
    state = _read_takeover_state()
    repo = meta.get("dataset_repo_id")
    # A state file left by the previous session reports another repo; ignore it.
    if state.get("repo_id") and repo and state["repo_id"] != repo:
        state = {k: v for k, v in state.items() if k not in ("episodes", "repo_id")}
    # Policy load takes ~20s, during which the subprocess has published nothing
    # yet — the page would show "0 集" for a dataset that already holds
    # episodes, so read the count off disk until the writer takes over.
    if "episodes" not in state:
        root = meta.get("dataset_root")
        info = _read_dataset_info(Path(root)) if root else None
        if info is not None:
            state = {**state, "episodes": int(info.get("total_episodes") or 0)}
    stats = _persist_take_stats(state, str(state.get("repo_id") or repo or ""))
    return {
        "capture_mode": meta.get("capture_mode", "base"),
        "dataset_repo_id": repo,
        "episode_target": meta.get("num_episodes"),
        "takeover": state,
        # Lifetime counters for this capture dataset, folded in from the child.
        "take_stats": stats,
    }


def _persist_take_stats(state: dict[str, Any], repo: str) -> dict[str, Any]:
    """Fold the subprocess's per-take counters into the dataset's running totals.

    A capture session is one process that records many takes, so the counters
    live in the child and only the child sees the phase transitions. The server
    polls them and adds the *delta* since the last poll — a fresh session resets
    the child's counters to zero, which shows up as a negative delta and is read
    as "this session started over".
    """
    takes = state.get("takes")
    if takes is None or not repo:
        return {}

    with _state_lock:
        prev = dict(_inference_meta.get("last_take_counts") or {})
        corrected = int(state.get("corrected_takes") or 0)
        failures = int(state.get("failures") or 0)
        delta_takes = int(takes) - int(prev.get("takes") or 0)
        delta_corrected = corrected - int(prev.get("corrected") or 0)
        delta_failures = failures - int(prev.get("failures") or 0)
        if delta_takes < 0:  # new subprocess, counters restarted
            delta_takes, delta_corrected, delta_failures = int(takes), corrected, failures
        _inference_meta["last_take_counts"] = {
            "takes": int(takes),
            "corrected": corrected,
            "failures": failures,
        }

    from .rollout_eval import add_take_stats

    try:
        return add_take_stats(
            repo,
            takes=max(0, delta_takes),
            corrected=max(0, min(delta_corrected, max(0, delta_takes))),
            failures=max(0, delta_failures),
            policy_ref=str(_inference_meta.get("policy_ref") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - stats must never break a session
        logger.info("Take stats not recorded for %s: %s", repo, exc)
        return {}


def handle_inference_end_round(advance: bool = True) -> dict[str, Any]:
    """End this trial now and, unless told otherwise, start the next one.

    "本轮成功" means the operator has seen what this attempt does: the arm should
    return home and a fresh trial begin — the rollout twin of the recording
    page's exit-early. Stopping is graceful: the recording strategies finalise
    the dataset and walk the robot back on their own shutdown path, so ask them
    through the trigger file and only escalate to a signal if they don't exit.
    """
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_end_in_progress
    global _inference_last_exit_at

    with _state_lock:
        if not inference_active or _inference_proc is None:
            return {"success": False, "status_code": 409, "message": "No inference is active"}
        if _inference_end_in_progress:
            return {"success": False, "status_code": 409, "message": "上一轮正在收尾，稍等"}
        proc = _inference_proc
        mode = str(_inference_meta.get("capture_mode") or "base")
        current_run = _inference_current_run or 1
        # Continuous base session: "本轮成功" marks THIS trial done. Signal the
        # subprocess to end only the current round and keep running (home, rest,
        # next round) instead of killing it and reloading the model.
        if mode == "base" and int(_inference_meta.get("rounds", 1) or 1) > 1:
            try:
                _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                (_PREVIEW_DIR / _ROUND_END_FILE).write_text("1")
            except OSError as exc:
                return {"success": False, "status_code": 500, "message": f"标记失败: {exc}"}
            return {"success": True, "message": "已标记成功，进入休整，准备下一轮", "advanced": True}
        # A capture session is ONE continuous recorder: marking a trial as
        # success must not tear it down (that would reload the model and lose
        # the dataset handle). Close only the correction in progress — the
        # strategy saves that episode, walks home and keeps recording.
        if mode in _MODE_KEYS and advance:
            phase = _read_takeover_state().get("phase", "")
            key = None
            if mode == "dagger" and phase == "correcting":
                key = "tab"
            elif mode == "highlight" and phase == "recording":
                key = "s"
            if key is not None:
                with contextlib.suppress(OSError):
                    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                    (_PREVIEW_DIR / _TAKEOVER_KEY_FILE).write_text(key)
            return {"success": True, "message": "已标记成功，继续采集", "advanced": True}
        _inference_end_in_progress = True

    exited = False
    try:
        if mode in _MODE_KEYS:
            with contextlib.suppress(OSError):
                _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
                (_PREVIEW_DIR / _TAKEOVER_KEY_FILE).write_text("esc")
            # Encoding the last episode takes time; give it room before killing.
            for _ in range(90):
                time.sleep(1.0)
                if proc.poll() is not None:
                    exited = True
                    break
        if not exited:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("End round: subprocess ignored SIGTERM; killing")
                proc.kill()
                proc.wait()
    finally:
        with _state_lock:
            _inference_end_in_progress = False
            _inference_last_exit_at = time.time()
            inference_active = False
            _inference_proc = None
            _inference_started_at = None
            _inference_rollout_started_at = None
            _inference_meta = {}
            _inference_current_run = 0
        _stop_inference_cameras()

    # Reaching this point for a capture mode means the operator asked to close
    # the whole session (advance=False): the recorder finalises its dataset on
    # its own shutdown path. Only plain base trials advance to a fresh round.
    if not advance or mode in _MODE_KEYS:
        with _state_lock:
            _inference_remaining_runs = 0
            _inference_restart_config = {}
        return {"success": True, "message": "本轮已结束", "advanced": False}

    with _state_lock:
        if _inference_remaining_runs <= 0 or not _inference_restart_config:
            _inference_remaining_runs = 0
            return {"success": True, "message": "本轮已结束，轮数已用完", "advanced": False}
        _inference_remaining_runs -= 1
        next_current = current_run + 1
        remaining_after = _inference_remaining_runs
        config = dict(_inference_restart_config)

    # Let the serial port, cameras and GPU/host memory of the previous run release.
    _wait_for_restart_ready()
    result = handle_start_inference(InferenceRequest(**config))
    if not result.get("success"):
        with _state_lock:
            _inference_remaining_runs = 0
        return {
            "success": False,
            "status_code": result.get("status_code", 500),
            "message": f"下一轮启动失败: {result.get('message')}",
        }
    with _state_lock:
        _inference_current_run = next_current
        _inference_remaining_runs = remaining_after
    return {
        "success": True,
        "message": f"第 {next_current} 轮已开始",
        "advanced": True,
        "current_run": next_current,
        "remaining_runs": remaining_after,
    }


def handle_inference_status() -> dict[str, Any]:
    global inference_active, _inference_proc, _inference_started_at
    global _inference_rollout_started_at, _inference_meta
    global _inference_remaining_runs, _inference_current_run, _inference_restart_config
    global _inference_setup_retries, _inference_last_exit_at

    total_runs = _inference_current_run + _inference_remaining_runs

    # Finalise state lazily if the subprocess died on its own.
    with _state_lock:
        proc = _inference_proc
        if _inference_end_in_progress:
            # A round is being closed deliberately ("本轮成功"). end_round owns the
            # restart, so report the session as alive — otherwise the page sees a
            # gap between rounds and bounces to home.
            return {
                "inference_active": True,
                "started_at": _inference_started_at,
                "rollout_started_at": None,
                "elapsed_s": 0,
                "rollout_elapsed_s": 0,
                "duration_s": _inference_meta.get("duration_s"),
                "policy_ref": _inference_meta.get("policy_ref"),
                "log_path": _inference_meta.get("log_path"),
                "current_run": _inference_current_run,
                "total_runs": total_runs,
                "camera_names": list(_inference_meta.get("cameras", {}).keys()),
                "switching_round": True,
                **_capture_status_fields(_inference_meta),
            }
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            _inference_last_exit_at = time.time()
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
            # Transient bring-up failure (servo dropped a packet while torque was
            # enabled, or a camera was still busy): retry a fresh subprocess a
            # few times before surfacing it. These self-heal on a clean process.
            if (
                outcome == "failed"
                and finished_rollout_started is None
                and _inference_setup_retries < _MAX_SETUP_RETRIES
                and _inference_restart_config
                and _error_is_transient(error)
            ):
                _inference_setup_retries += 1
                logger.warning(
                    "Transient setup failure — retrying (%d/%d): %s",
                    _inference_setup_retries, _MAX_SETUP_RETRIES, error,
                )
                _wait_for_restart_ready(proc)
                inference_active = False
                result = handle_start_inference(InferenceRequest(**_inference_restart_config))
                if result.get("success"):
                    return {
                        "inference_active": True,
                        "started_at": _inference_started_at,
                        "rollout_started_at": None,
                        "elapsed_s": 0,
                        "rollout_elapsed_s": 0,
                        "duration_s": finished_meta.get("duration_s"),
                        "policy_ref": finished_meta.get("policy_ref"),
                        "log_path": result.get("log_path"),
                        "current_run": _inference_current_run,
                        "total_runs": total_runs,
                        "camera_names": list(finished_meta.get("cameras", {}).keys()),
                    }
                logger.warning("Setup retry could not start: %s", result.get("message"))
            if _inference_remaining_runs > 0 and outcome != "failed" and _inference_restart_config:
                _inference_remaining_runs -= 1
                next_current = _inference_current_run + 1
                remaining_after = _inference_remaining_runs
                logger.info("Auto-restarting inference (%d runs remaining)",
                            remaining_after)
                # Wait for serial/cameras/memory to release before respawning.
                _wait_for_restart_ready(proc)
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
            # A non-zero exit with an empty log means the subprocess died in
            # native init before Python could print — almost always a device
            # or memory race with the previous run. Say so instead of showing a
            # bare exit code that points at a 0-byte log.
            if rc and not error:
                log_path = finished_meta.get("log_path")
                try:
                    empty = not (log_path and Path(log_path).stat().st_size)
                except OSError:
                    empty = True
                if empty:
                    error = "进程在输出任何日志前就退出了（多为串口/摄像头/显存尚未释放）。"
                    outcome = "failed"
            hint = _friendly_hint(error) or (
                "这一轮在启动阶段就退出了。若连续多轮出现，请降低并发或检查摄像头/串口是否被占用。"
                if error else None
            )
            return {
                "inference_active": False,
                "exited": True,
                "exit_code": rc,
                "outcome": outcome,
                "error": error,
                "hint": hint,
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
