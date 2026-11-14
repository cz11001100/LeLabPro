"""lerobot-rollout wrapper that adds inter-chunk action smoothing.

Launched instead of ``lerobot.scripts.lerobot_rollout`` when the user enables
"chunk smoothing" in the inference dialog. Action-chunk policies (SmolVLA,
ACT, ...) execute a cached chunk until it is exhausted, then jump onto a
freshly predicted one; the seam between two chunks shows up as a visible
jerk. This module wraps the rollout's inference engine with an exponential
moving average over the *executed* action stream, which absorbs those seams
(and general tick-to-tick jitter) at the cost of a small lag.

The gripper channel is excluded from filtering: it is effectively binary in
the demonstrations, and smoothing it would delay grasp/release.

LeLab-specific flags are consumed here and stripped from ``sys.argv`` before
lerobot's draccus parser sees the rest:

    --lelab.smooth_alpha=0.6   blend weight of the new raw action (1 = off)

The wrapper also publishes the observation images the control loop already
captured to ``~/.cache/huggingface/lerobot/inference_preview/*.jpg`` so the
LeLab server can serve a live preview without opening the cameras a second
time (V4L2 capture access is exclusive to this process).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_LELAB_PREFIX = "--lelab."
_PREVIEW_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_preview"
_PREVIEW_INTERVAL_S = 0.2
_SNAPSHOT_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "inference_snapshots"
_SNAPSHOT_CAPACITY = 30


class SnapshotBuffer:
    """Ring buffer of recent (observation, action) ticks for failure diagnosis.

    The inference subprocess maintains this in memory; when a trigger file
    appears in _PREVIEW_DIR, the watcher thread dumps the buffer to a
    timestamped directory under _SNAPSHOT_DIR. Each dump contains per-tick
    JPEGs (one per camera) and a manifest.json with actions, states, and
    timestamps — enough to run post-hoc saliency/aim/localization analysis
    on exactly the frames the model saw when it failed."""

    def __init__(self, capacity: int = _SNAPSHOT_CAPACITY):
        self.capacity = capacity
        self._buffer: list[dict] = []
        self._lock = threading.Lock()

    def add(self, tick: dict) -> None:
        with self._lock:
            self._buffer.append(tick)
            if len(self._buffer) > self.capacity:
                self._buffer.pop(0)

    def dump(self, out_dir: Path) -> int:
        import cv2
        import numpy as np

        with self._lock:
            snapshot = list(self._buffer)
        if not snapshot:
            return 0
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"ticks": [], "n_ticks": len(snapshot)}
        for i, tick in enumerate(snapshot):
            tick_dir = out_dir / f"tick_{i:03d}"
            tick_dir.mkdir(exist_ok=True)
            entry = {"timestamp": tick["timestamp"]}
            for cam_name, frame in tick.get("images", {}).items():
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        (tick_dir / f"{cam_name}.jpg").write_bytes(buf.tobytes())
                        entry.setdefault("cameras", []).append(cam_name)
                except Exception:
                    pass
            if tick.get("action") is not None:
                entry["action"] = np.asarray(tick["action"]).tolist()
            if tick.get("state") is not None:
                entry["state"] = np.asarray(tick["state"]).tolist()
            manifest["ticks"].append(entry)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return len(snapshot)


# ---------------------------------------------------------------------------
# Data-collection bridge (dagger / highlight strategies)
# ---------------------------------------------------------------------------

_TAKEOVER_KEY_FILE = _PREVIEW_DIR / "takeover_key"
_TAKEOVER_STATE_FILE = _PREVIEW_DIR / "takeover_state.json"
_TAKEOVER_POLL_S = 0.1
_STATE_WRITE_S = 0.3
# LeLab's own verbs. 'start' opens the next episode; 'success'/'fail' close the
# one in progress — the strategy only has a key for closing a correction, so the
# bridge translates them into that event and arms the flags (discard / failure
# count) that decide what the closing does with the buffered frames.
_START_KEY = "start"
_TAKEOVER_KEY = "takeover"
_SUCCESS_KEY = "success"
_FAIL_KEY = "fail"
_CORRECTION_KEY = "tab"  # toggles the correction on and off
_RESUME_KEY = "space"
# LeLab session phases shown in the UI. 'ready' is the between-episode rest
# screen: the arm is parked at its start pose and the policy sends nothing until
# the operator opens the next episode.
_SOFT_START_TICKS = 10  # ~0.3s at 30fps gliding out of the rest pose
_PHASE_READY = "ready"
_PHASE_RUNNING = "running"
# Set when LeLab itself pauses the strategy to park the session, so the leader's
# smooth handover is skipped — see `_enter_ready`.
_SKIP_HANDOVER = threading.Event()


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _LeLabKeyListener:
    """Drop-in replacement for lerobot's ``create_key_listener``, driven by LeLab.

    The rollout subprocess is spawned by the web server: no TTY, and pynput's
    global hook is unavailable on a Wayland session, so lerobot's own keyboard
    backends both come back ``None`` and a UI-driven strategy would be
    uncontrollable. The server writes canonical key names ('space', 'tab',
    'esc', 's', 'h') into a trigger file; we dispatch them here, which keeps
    every strategy's own key→event table untouched.
    """

    def __init__(self, dispatch):
        self._dispatch = dispatch
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="lelab-takeover-keys", daemon=True).start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(_TAKEOVER_POLL_S)
            try:
                if not _TAKEOVER_KEY_FILE.exists():
                    continue
                name = _TAKEOVER_KEY_FILE.read_text().strip()
                _TAKEOVER_KEY_FILE.unlink(missing_ok=True)
            except OSError:
                continue
            if not name:
                continue
            try:
                self._dispatch(name)
                logger.info("Takeover key dispatched: %s", name)
            except Exception as exc:
                logger.warning("Takeover key '%s' failed: %s", name, exc)

    def stop(self) -> None:
        self._stop.set()


def _install_takeover_input(strategy, ctx) -> None:
    """Point the recording strategies at the file bridge instead of a keyboard.

    lerobot's factory returns an *already started* listener, so the strategies
    never call ``start()`` themselves — the replacement has to do it here.

    The bridge also owns LeLab's session verbs. 'start' opens the next episode
    and 'success'/'fail' close the one in progress. Only a *correction* has a
    boundary inside the strategy, so mid-correction the verb becomes that event
    (with the save armed to commit or throw away); otherwise the request is
    queued for the control loop's next tick, because touching the arm or the
    planner from this listener thread would race the policy for the serial bus.
    """
    from lerobot.rollout.strategies import dagger, highlight

    def listener(dispatch, *, controls_help: str = "") -> _LeLabKeyListener:
        strategy.takeover_dispatch = dispatch

        def bridge_dispatch(name: str) -> None:
            if name == _START_KEY:
                strategy.start_episode = True
                return
            if name == _TAKEOVER_KEY:
                _takeover_now(strategy, dispatch)
                return
            if name in (_SUCCESS_KEY, _FAIL_KEY):
                _mark_take_end(
                    strategy,
                    save=name == _SUCCESS_KEY,
                    count_failure=name == _FAIL_KEY,
                )
                return
            dispatch(name)

        bridge = _LeLabKeyListener(bridge_dispatch)
        bridge.start()
        logger.info("LeLab takeover bridge active%s", f" ({controls_help})" if controls_help else "")
        return bridge

    for module in (dagger, highlight):
        module.create_key_listener = listener


def _take_phase(strategy) -> str:
    phase = getattr(getattr(strategy, "_events", None), "phase", None)
    return getattr(phase, "value", "") or ""


def _pending_frame_count(dataset) -> int:
    """Frames buffered for the episode in progress (0 when nothing is pending)."""
    try:
        buffer = getattr(getattr(dataset, "writer", None), "episode_buffer", None)
        return int(buffer["size"]) if buffer else 0
    except Exception:
        return 0


def _close_take_episode(strategy, ctx, *, save: bool) -> None:
    """Commit or drop the buffered frames of the take that just ended.

    RA-BC weights a frame by `progress[t + chunk_size] - progress[t]` clamped to
    its own episode, so an episode shorter than the policy's chunk gives every
    frame the same clamped future — the per-frame signal collapses to one number
    per episode. Warn loudly instead of quietly filling the dataset with those.
    """
    dataset = getattr(ctx.data, "dataset", None)
    try:
        frames = _pending_frame_count(dataset)
        if dataset is not None and frames:
            if save:
                dataset.save_episode()
                strategy.last_episode_frames = frames
                logger.info("LeLab: take saved (episodes: %d, %d frames)", dataset.num_episodes, frames)
                _stop_at_target(strategy, ctx, dataset)
                chunk = int(getattr(getattr(ctx.policy.policy, "config", None), "chunk_size", 0) or 0)
                if chunk and frames < chunk:
                    logger.warning(
                        "本集只有 %d 帧，短于 RA-BC 的 chunk_size=%d：进度差全被夹到集尾，"
                        "这一集的逐帧权重会退化成一个常数",
                        frames,
                        chunk,
                    )
            else:
                dataset.clear_episode_buffer()
                logger.info("LeLab: take discarded (%d frames), nothing saved", frames)
    except Exception:
        logger.warning("Could not close the episode for this take", exc_info=True)


def _stop_at_target(strategy, ctx, dataset) -> None:
    """End the session once the dataset holds the target number of episodes.

    lerobot only enforces this in corrections-only mode, where its loop counts
    the episodes it saves; hybrid recording rotates on video size instead, so
    without this a resumed dataset would keep growing past the total the
    operator asked for.
    """
    target = int(getattr(getattr(ctx.runtime.cfg, "dataset", None), "num_episodes", 0) or 0)
    if not target or dataset.num_episodes < target:
        return
    stop = getattr(getattr(strategy, "_events", None), "stop_recording", None)
    if stop is not None:
        logger.info("LeLab: dataset reached its %d-episode target — closing the session", target)
        stop.set()


def _finish_take(strategy, ctx, *, save: bool) -> None:
    """Close the take from the control loop, then park the session."""
    _close_take_episode(strategy, ctx, save=save)
    _enter_ready(strategy, ctx)


def _walk_home(strategy, ctx) -> None:
    moving = getattr(strategy, "moving_guard", None)
    if moving is not None:
        moving.set()
    try:
        strategy.return_to_initial_position(ctx.hardware)
    except Exception:
        logger.warning("Could not walk the follower home at the take boundary", exc_info=True)
    finally:
        if moving is not None:
            moving.clear()


def _enter_ready(strategy, ctx, *, home: bool = True) -> None:
    """Park the session between episodes: arm at its start pose, policy silent.

    lerobot has no idle phase, so this borrows PAUSED — whose control branch
    keeps re-sending an action — and pins that action to the initial position.
    The leader's smooth handover is suppressed on the way in: nobody is taking
    over yet, and steering the leader under hands that are not on it is exactly
    what this screen exists to avoid.
    """
    hold = getattr(strategy, "paused_hold", None)
    if hold is not None and ctx.hardware.initial_position:
        hold.clear()
        hold.update(ctx.hardware.initial_position)
    dispatch = getattr(strategy, "takeover_dispatch", None)
    if dispatch is not None and _take_phase(strategy) == "autonomous":
        _SKIP_HANDOVER.set()
        dispatch(_RESUME_KEY)
    if home:
        _walk_home(strategy, ctx)
    # lerobot re-torques the leader when a correction ends. Left like that it
    # sits locked through the rest screen and then drops the moment 开始下一集
    # releases it — that fall is the "violent motion" before inference starts.
    teleop = ctx.hardware.teleop
    if teleop is not None and home:
        try:
            teleop.disable_torque()
        except Exception:
            logger.warning("Could not release the leader at the take boundary", exc_info=True)
    strategy.lelab_phase = _PHASE_READY
    strategy.episode_started_at = None
    strategy.soft_start_ticks = 0


def _begin_episode(strategy, ctx) -> None:
    """Open the next episode: released hold, fresh plan, policy driving."""
    hold = getattr(strategy, "paused_hold", None)
    if hold is not None:
        hold.clear()
    _reset_planner(strategy)
    dispatch = getattr(strategy, "takeover_dispatch", None)
    if dispatch is not None and _take_phase(strategy) == "paused":
        dispatch(_RESUME_KEY)
    strategy.lelab_phase = _PHASE_RUNNING
    strategy.episode_started_at = time.monotonic()
    strategy.soft_start_ticks = _SOFT_START_TICKS
    # One attempt counter per episode opened; `take_corrected` is raised by the
    # phase hook if a human has to intervene during this one.
    strategy.takes = int(getattr(strategy, "takes", 0)) + 1
    strategy.take_corrected = False
    dataset = getattr(ctx.data, "dataset", None)
    logger.info("LeLab: episode %d starting", (dataset.num_episodes if dataset is not None else 0) + 1)


def _reset_planner(strategy) -> None:
    """Drop the cached action chunk so the next take re-plans from scratch.

    Without this the leftover chunk keeps playing out right after the walk-home,
    so the arm drives back toward where the last take ended and the new episode
    reads as a continuation instead of a fresh attempt.
    """
    try:
        engine = getattr(strategy, "_engine", None)
        if engine is not None:
            engine.reset()
        interpolator = getattr(strategy, "_interpolator", None)
        if interpolator is not None:
            interpolator.reset()
        strategy._cached_obs_processed = None
    except Exception:
        logger.warning("Planner reset at the take boundary failed", exc_info=True)


def _install_episode_discard(strategy, ctx) -> None:
    """Let the operator throw away the take in progress and redo it.

    A correction is committed the instant it ends, so 're-record this episode'
    has to intercept that save: drop the buffered frames instead of appending an
    episode, and hand the episode budget back so the discarded take doesn't
    consume a slot. Without it the only way out of a botched take was deleting
    the episode afterwards in the dataset editor.
    """
    dataset = ctx.data.dataset
    if dataset is None:
        return
    original_save = dataset.save_episode

    def save_episode(*args, **kwargs):
        if getattr(strategy, "discard_episode", False):
            strategy.discard_episode = False
            try:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                budget = getattr(strategy.config, "num_episodes", None)
                if isinstance(budget, int):
                    strategy.config.num_episodes = budget + 1
                logger.info("LeLab: take discarded — ready to re-record this episode")
                return
            except Exception:
                logger.warning("Episode discard failed; saving instead", exc_info=True)
        if not args and not kwargs:
            try:
                if not dataset.has_pending_frames():
                    # Teardown's final save after a take that was already closed
                    # would append a zero-length episode.
                    logger.debug("LeLab: no pending frames — skipping empty episode save")
                    return
            except Exception:
                logger.debug("Pending-frame check failed; saving anyway", exc_info=True)
        return original_save(*args, **kwargs)

    dataset.save_episode = save_episode


_TAKEOVER_TORQUE_RETRIES = 4
_TAKEOVER_TORQUE_RETRY_S = 0.12


def _harden_takeover_bus(ctx) -> None:
    """Keep a dropped status packet on the leader bus from ending the session.

    Entering a correction re-enables torque on the leader so it can be slid
    under the follower; those writes go out with no retry, so one transient
    'Incorrect status packet' propagates out of the control loop, parks the
    follower at its start pose and exits with code 1. The torque writes are
    idempotent, so retrying them — and dropping the optional smooth handover
    on final failure — turns a crash into a takeover that simply isn't
    centred under the follower.
    """
    from lerobot.rollout.strategies import dagger

    teleop = ctx.hardware.teleop
    if teleop is None:
        return

    def retry(name):
        original = getattr(teleop, name, None)
        if original is None:
            return

        def wrapped():
            for attempt in range(_TAKEOVER_TORQUE_RETRIES):
                try:
                    original()
                    if attempt:
                        logger.info("Teleop %s recovered after %d retries", name, attempt)
                    return
                except Exception as exc:
                    logger.debug("Teleop %s attempt %d failed: %s", name, attempt + 1, exc)
                    time.sleep(_TAKEOVER_TORQUE_RETRY_S * (attempt + 1))
            logger.warning("Teleop %s still failing after %d tries — continuing anyway",
                           name, _TAKEOVER_TORQUE_RETRIES)

        setattr(teleop, name, wrapped)

    for _name in ("enable_torque", "disable_torque"):
        retry(_name)

    orig_smooth_move = dagger.teleop_smooth_move_to

    def smooth_move(target_teleop, target_pos, **kwargs):
        # LeLab parked the session (no operator on the leader): skip the
        # handover instead of steering the leader on its own.
        if _SKIP_HANDOVER.is_set():
            _SKIP_HANDOVER.clear()
            logger.debug("Leader handover skipped (session parked)")
            return
        try:
            orig_smooth_move(target_teleop, target_pos, **kwargs)
        except Exception as exc:
            logger.warning("Smooth takeover handover skipped: %s", exc)

    dagger.teleop_smooth_move_to = smooth_move
    logger.info("Takeover bus hardening active (torque retries + handover guard)")


def _start_state_writer(strategy, ctx) -> None:
    """Publish the strategy phase and episode counter for the UI to poll."""

    def writer() -> None:
        while True:
            time.sleep(_STATE_WRITE_S)
            config = getattr(strategy, "config", None)
            state: dict = {"mode": getattr(config, "type", "")}
            failed = getattr(strategy, "failed_takes", None)
            if failed is not None:
                state["failures"] = int(failed)
            takes = getattr(strategy, "takes", None)
            if takes is not None:
                state["takes"] = int(takes)
                state["corrected_takes"] = int(getattr(strategy, "corrected_takes", 0) or 0)
            # LeLab's own session phase drives the 修整页 ↔ 本集页 switch; the
            # strategy phase above only says where DAgger's state machine is.
            if getattr(strategy, "lelab_phase", None) is not None:
                state["session_phase"] = strategy.lelab_phase
            started = getattr(strategy, "episode_started_at", None)
            if started is not None:
                state["episode_elapsed_s"] = round(time.monotonic() - started, 1)
            last_frames = getattr(strategy, "last_episode_frames", None)
            if last_frames is not None:
                state["last_episode_frames"] = int(last_frames)
                chunk = getattr(getattr(getattr(ctx, "policy", None), "policy", None), "config", None)
                state["rabc_chunk_size"] = int(getattr(chunk, "chunk_size", 0) or 0)
            policy_s = getattr(strategy, "policy_time_s", None)
            if policy_s is not None:
                state["policy_s"] = round(float(policy_s), 1)
            phase = getattr(getattr(strategy, "_events", None), "phase", None)
            if phase is not None:
                state["phase"] = getattr(phase, "value", str(phase))
            # Highlight has no phase machine: it is either recording or not.
            live = getattr(strategy, "_recording_live", None)
            if live is not None:
                state["phase"] = "recording" if live.is_set() else "idle"
            dataset = getattr(getattr(ctx, "data", None), "dataset", None)
            if dataset is not None:
                try:
                    state["episodes"] = int(dataset.num_episodes)
                    # Creating a dataset stamps a timestamp onto its name, so
                    # this is the only place the real target repo is known.
                    state["repo_id"] = dataset.repo_id
                except Exception:
                    pass
            try:
                tmp = _PREVIEW_DIR / ".takeover_state.tmp"
                tmp.write_text(json.dumps(state))
                tmp.replace(_TAKEOVER_STATE_FILE)
            except OSError as exc:
                logger.debug("Takeover state write failed: %s", exc)

    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=writer, name="lelab-takeover-state", daemon=True).start()


def _start_snapshot_watcher(buffer: SnapshotBuffer) -> None:
    """Watch for trigger files and dump the snapshot buffer when one appears.

    The server creates _PREVIEW_DIR / "snapshot_request" containing a unique
    token; we dump to _SNAPSHOT_DIR / <token> and write a ".done" marker so
    the server knows the dump is complete."""
    trigger_file = _PREVIEW_DIR / "snapshot_request"

    def watcher() -> None:
        while True:
            time.sleep(0.3)
            if not trigger_file.exists():
                continue
            try:
                token = trigger_file.read_text().strip()
                trigger_file.unlink(missing_ok=True)
            except OSError:
                continue
            if not token:
                continue
            out_dir = _SNAPSHOT_DIR / token
            try:
                n = buffer.dump(out_dir)
                (out_dir / ".done").write_text(str(n))
                logger.info("Snapshot dumped: %d ticks → %s", n, out_dir)
            except Exception as exc:
                logger.warning("Snapshot dump failed: %s", exc)

    threading.Thread(target=watcher, name="lelab-snapshot-watcher", daemon=True).start()


def _parse_and_strip_lelab_args() -> dict[str, float]:
    opts: dict[str, float] = {}
    kept: list[str] = []
    for arg in sys.argv[1:]:
        if arg.startswith(_LELAB_PREFIX):
            key, _, value = arg[len(_LELAB_PREFIX) :].partition("=")
            try:
                opts[key] = float(value)
            except ValueError:
                logger.warning("Ignoring malformed lelab arg: %s", arg)
        else:
            kept.append(arg)
    sys.argv[1:] = kept
    return opts


class SmoothingInferenceEngine:
    """Delegate InferenceEngine that low-pass filters the action stream."""

    def __init__(self, inner, alpha: float, passthrough_idx: list[int], snapshot_buffer: SnapshotBuffer | None = None):
        self._inner = inner
        self._alpha = min(max(alpha, 0.05), 1.0)
        self._passthrough_idx = passthrough_idx
        self._last = None
        self._alpha_vec = None
        self._snapshot_buffer = snapshot_buffer

    def __getattr__(self, name):
        # lerobot's engine interface grows over time (latency, pump_query,
        # dispatch gating, task ownership); forward anything not overridden.
        return getattr(self._inner, name)

    def _coeffs(self, action):
        if self._alpha_vec is None or self._alpha_vec.numel() != action.numel():
            import torch

            vec = torch.full(
                (action.numel(),), self._alpha, device=action.device, dtype=action.dtype
            )
            for i in self._passthrough_idx:
                if 0 <= i < vec.numel():
                    vec[i] = 1.0
            self._alpha_vec = vec
        return self._alpha_vec

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()

    def reset(self) -> None:
        self._last = None
        self._inner.reset()

    def get_action(self, obs_frame):
        action = self._inner.get_action(obs_frame)
        if action is None:
            return None
        if self._snapshot_buffer is not None:
            try:
                import numpy as np

                images = {
                    k: v
                    for k, v in obs_frame.items()
                    if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3
                }
                state = obs_frame.get("observation.state")
                if state is not None and hasattr(state, "numpy"):
                    state = state.detach().cpu().numpy()
                action_np = action.detach().cpu().numpy() if hasattr(action, "numpy") else np.asarray(action)
                self._snapshot_buffer.add({
                    "timestamp": time.time(),
                    "images": images,
                    "action": action_np,
                    "state": state,
                })
            except Exception:
                pass
        if self._last is None or self._last.shape != action.shape:
            self._last = action.clone()
            return action
        coeffs = self._coeffs(action)
        smoothed = coeffs * action + (1.0 - coeffs) * self._last
        self._last = smoothed
        return smoothed

    def notify_observation(self, obs: dict) -> None:
        self._inner.notify_observation(obs)

    def pause(self) -> None:
        self._inner.pause()

    def resume(self) -> None:
        self._inner.resume()

    @property
    def ready(self) -> bool:
        return self._inner.ready

    @property
    def failed(self) -> bool:
        return self._inner.failed


def _start_preview_publisher(ctx) -> None:
    """Dump the control loop's observation images to JPEGs for the web preview."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("cv2/numpy unavailable; inference preview disabled")
        return

    robot_wrapper = ctx.hardware.robot_wrapper
    if robot_wrapper is None:
        return
    original_get_observation = robot_wrapper.get_observation

    latest: dict[str, np.ndarray] = {}
    lock = threading.Lock()

    def get_observation_with_preview():
        obs = original_get_observation()
        try:
            images = {
                k: v
                for k, v in obs.items()
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3
            }
            if images:
                with lock:
                    latest.update(images)
        except Exception:
            pass
        return obs

    robot_wrapper.get_observation = get_observation_with_preview

    def writer() -> None:
        _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            time.sleep(_PREVIEW_INTERVAL_S)
            with lock:
                snapshot = dict(latest)
            for key, frame in snapshot.items():
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        continue
                    tmp = _PREVIEW_DIR / f".{key}.tmp"
                    tmp.write_bytes(buf.tobytes())
                    tmp.replace(_PREVIEW_DIR / f"{key}.jpg")
                except Exception:
                    continue

    threading.Thread(target=writer, name="lelab-preview-writer", daemon=True).start()
    logger.info("LeLab inference preview publisher started (dir=%s)", _PREVIEW_DIR)


def _install_take_rounding(strategy, ctx) -> None:
    """Run the recording session as 录制 does: rest screen, episode, rest.

    DAgger's PAUSED branch re-sends the *pre-correction* action each tick, so a
    move at the take boundary would be undone instantly — the hold is overridden
    with the initial pose while the operator resets the scene (the homing move is
    exempt, otherwise it degenerates into a jump). Requests raised outside a
    correction are serviced from the control loop's own tick, because walking the
    arm or resetting the planner from the listener thread would race the policy.
    """
    from lerobot.rollout.strategies.dagger import DAggerPhase

    hardware = ctx.hardware
    robot = hardware.robot_wrapper
    events = getattr(strategy, "_events", None)
    if robot is None or events is None or not hardware.initial_position:
        return

    hold: dict = {}
    moving = threading.Event()
    strategy.paused_hold = hold
    strategy.moving_guard = moving
    original_send_action = robot.send_action

    last_cmd: dict = {}

    def send_action_with_hold(action, *args, **kwargs):
        if hold and not moving.is_set() and events.phase == DAggerPhase.PAUSED:
            action = hold
        else:
            ramp = int(getattr(strategy, "soft_start_ticks", 0) or 0)
            if ramp > 0 and last_cmd:
                # Glide out of the rest pose. The policy's first chunk often
                # points well above home (it drives to its training start), and
                # snapping straight onto it reads as the arm "standing up".
                alpha = 1.0 - ramp / _SOFT_START_TICKS
                blended = {}
                for key, value in action.items():
                    base = _as_float(last_cmd.get(key))
                    target = _as_float(value)
                    blended[key] = (
                        base * (1.0 - alpha) + target * alpha
                        if base is not None and target is not None
                        else value
                    )
                action = blended
                strategy.soft_start_ticks = ramp - 1
        last_cmd.clear()
        for key, value in action.items():
            as_float = _as_float(value)
            if as_float is not None:
                last_cmd[key] = as_float
        return original_send_action(action, *args, **kwargs)

    robot.send_action = send_action_with_hold

    original_apply = strategy._apply_transition

    def apply_transition(old_phase, new_phase, engine, interpolator, ctx_, prev_action):
        hold.clear()
        if new_phase == DAggerPhase.CORRECTING and not getattr(strategy, "take_corrected", False):
            # One takeover per take, however many times they pause and resume it:
            # the number the UI reports is "how many attempts needed a human".
            strategy.take_corrected = True
            strategy.corrected_takes = int(getattr(strategy, "corrected_takes", 0)) + 1
        # 开始 (PAUSED→CORRECTING) needs no extra work: lerobot releases the
        # leader's torque there, which is exactly what "开始" means now that the
        # takeover step already aligned it under the follower.
        original_apply(old_phase, new_phase, engine, interpolator, ctx_, prev_action)
        if new_phase == DAggerPhase.PAUSED:
            parked = getattr(strategy, "pending_park", None)
            strategy.pending_park = None
            if old_phase == DAggerPhase.CORRECTING:
                # Hybrid recording has no per-correction boundary in lerobot (its
                # episodes rotate on video size), so close the episode here. In
                # corrections-only mode the strategy saves it right after this
                # call, and the discard flag is left for that save to consume.
                if getattr(strategy.config, "record_autonomous", False):
                    save = not getattr(strategy, "discard_episode", False)
                    strategy.discard_episode = False
                    _close_take_episode(strategy, ctx, save=save)
            elif parked is not None:
                _close_take_episode(strategy, ctx, save=parked)
            else:
                # The operator's own "take over" pause: hold everything in place
                # and wait for them to start the correction.
                return
            _enter_ready(strategy, ctx)

    strategy._apply_transition = apply_transition

    original_consume = events.consume_transition

    def consume_transition():
        transition = original_consume()
        if transition is not None:
            return transition
        # First tick only: the loop's own `events.reset()` clears the phase
        # queue, so parking before it would simply be undone.
        if getattr(strategy, "lelab_phase", None) is None:
            _enter_ready(strategy, ctx, home=False)
            return None
        if getattr(strategy, "start_episode", False):
            strategy.start_episode = False
            _begin_episode(strategy, ctx)
            return None
        pending = getattr(strategy, "pending_take_end", None)
        if pending is None:
            return None
        if _take_phase(strategy) == "autonomous":
            # Ask the state machine to park and let the boundary run when the
            # pause lands. Finishing here would leave this tick in its AUTONOMOUS
            # branch, which replays the leftover chunk and drags the arm back over
            # the walk-home — the "returns then jumps to where I pressed" bug.
            strategy.pending_take_end = None
            strategy.pending_park = pending
            _SKIP_HANDOVER.set()
            dispatch = getattr(strategy, "takeover_dispatch", None)
            if dispatch is not None:
                dispatch(_RESUME_KEY)
            return None
        strategy.pending_take_end = None
        _finish_take(strategy, ctx, save=pending)
        return None

    events.consume_transition = consume_transition
    logger.info("Session rounding active (修整页 ↔ 本集, save-or-drop per episode)")


def _takeover_now(strategy, dispatch) -> None:
    """Pause the policy and bring the leader onto the follower's pose.

    Taking over is one confirm (then 开始 just releases the leader): the leader
    is smoothly driven to where the follower is holding, so the operator's hand
    lands in the right place. Parking the session between episodes suppresses
    that alignment — nobody is there to take over.
    """
    if _take_phase(strategy) != "autonomous":
        # 开始 was just pressed: the phase flip is consumed on the next tick.
        # Wait it out instead of dropping the click.
        deadline = time.monotonic() + 1.0
        while _take_phase(strategy) != "autonomous" and time.monotonic() < deadline:
            if getattr(strategy, "lelab_phase", None) != _PHASE_RUNNING:
                return
            time.sleep(0.02)
        if _take_phase(strategy) != "autonomous":
            return
    dispatch(_RESUME_KEY)


def _mark_take_end(strategy, *, save: bool, count_failure: bool = False) -> None:
    """Close the take in progress: commit or drop it, optionally count a failure.

    Mid-correction the closing goes through the strategy's own event (it owns the
    episode boundary there); anywhere else the request is queued for the control
    loop, which is the only thread allowed to move the arm or reset the planner.
    """
    if count_failure:
        strategy.failed_takes = int(getattr(strategy, "failed_takes", 0)) + 1
        logger.info("LeLab failure marked (total: %d)", strategy.failed_takes)
    if _take_phase(strategy) == "correcting":
        strategy.discard_episode = not save
        dispatch = getattr(strategy, "takeover_dispatch", None)
        if dispatch is not None:
            dispatch(_CORRECTION_KEY)
    else:
        strategy.pending_take_end = save


def _start_session_watchdog(strategy, ctx, max_correction_s: float, per_episode_s: float) -> None:
    """Cap how long a correction or an unjudged episode may run.

    Neither lerobot's duration nor an operator's click bounds these any more: a
    correction nobody closed, or an episode left running while the operator steps
    away, would hold the rig indefinitely. A correction that hits its cap is kept
    - the human did teach something - while an episode that times out without a
    verdict is treated exactly like the failure button: dropped, and counted.
    """
    if max_correction_s <= 0 and per_episode_s <= 0:
        return

    def loop() -> None:
        correcting_since: float | None = None
        shutdown = ctx.runtime.shutdown_event
        while not shutdown.is_set():
            time.sleep(0.2)
            now = time.monotonic()
            if _take_phase(strategy) == "correcting":
                if correcting_since is None:
                    correcting_since = now
                elif max_correction_s > 0 and now - correcting_since >= max_correction_s:
                    correcting_since = None
                    logger.info("Correction hit its %.0fs cap - saving the take", max_correction_s)
                    _mark_take_end(strategy, save=True)
            else:
                correcting_since = None
            started = getattr(strategy, "episode_started_at", None)
            if (
                per_episode_s > 0
                and started is not None
                and getattr(strategy, "lelab_phase", None) == _PHASE_RUNNING
                and now - started >= per_episode_s
            ):
                strategy.episode_started_at = None
                logger.info("Episode hit its %.0fs limit - marking it failed", per_episode_s)
                _mark_take_end(strategy, save=False, count_failure=True)

    threading.Thread(target=loop, name="lelab-session-watchdog", daemon=True).start()
    logger.info(
        "Session watchdog armed (correction %.0fs / episode %.0fs)", max_correction_s, per_episode_s
    )


def _start_memory_reclaimer(interval_s: float = 10.0) -> None:
    """Periodically hand freed heap back to the OS during multi-episode capture.

    lerobot's dagger saves episodes back-to-back in *this* process. Python frees
    each episode's numpy frames after save_episode(), but glibc keeps them in its
    arena, so RSS climbs (~700MB/episode measured) until the kernel OOM-kills the
    run mid-save — the same failure record.py fixed for recording. The rollout
    subprocess never trimmed, so a long capture session could die after a few
    episodes. Reuse the recording path's gc + libc malloc_trim on a timer.
    """
    try:
        from .record import _release_free_memory
    except Exception:
        logger.warning("Memory reclaimer unavailable", exc_info=True)
        return

    def loop() -> None:
        while True:
            time.sleep(interval_s)
            try:
                _release_free_memory()
            except Exception:
                pass

    threading.Thread(target=loop, name="lelab-mem-reclaim", daemon=True).start()
    logger.info("LeLab memory reclaimer started (every %.0fs)", interval_s)


_DEFAULT_ROUND_REST_S = 3.0
# A correction left open would otherwise hang the rig forever now that the
# session's duration only budgets policy driving time.
_DEFAULT_MAX_CORRECTION_S = 30.0
_ROUND_END_FILE = "round_end"


def _run_one_round(strategy, ctx, round_end: Path) -> None:
    """Run one base round, ending early when the operator marks success.

    Mirrors BaseStrategy.run's control loop but also breaks when the server
    drops a ``round_end`` file: "mark this trial as success" should finish THIS
    round (then home + rest) without tearing down the process, so the model is
    never reloaded. A hard stop still goes through ``shutdown_event``.
    """
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.cycle_timer import CycleTimer

    engine = strategy._engine
    cfg = ctx.runtime.cfg
    robot = ctx.hardware.robot_wrapper
    interpolator = strategy._interpolator
    timer = CycleTimer(
        cfg.fps,
        interpolator.multiplier,
        records_data=False,
        report=ctx.runtime.cadence_report,
    )
    start = time.perf_counter()
    engine.resume()
    logger.info("Base strategy control loop started")
    try:
        while not ctx.runtime.shutdown_event.is_set():
            if round_end.exists():
                try:
                    round_end.unlink()
                except OSError:
                    pass
                logger.info("Round ended early: operator marked success")
                return
            strategy._tick(timer, new_cycle=interpolator.needs_new_action())
            if cfg.duration > 0 and (time.perf_counter() - start) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                return
            with timer.section("observe"):
                obs = robot.get_observation()
            with timer.section("process_obs"):
                obs_processed = strategy._process_observation_and_notify(ctx.processors, obs)
            if strategy._handle_warmup(cfg.use_torch_compile, timer):
                continue
            step = send_next_action(obs_processed, obs, ctx, interpolator, timer)
            with timer.section("telemetry"):
                strategy._log_telemetry(obs_processed, step, ctx.runtime)
            with timer.section("query"):
                engine.pump_query(obs_processed)
            timer.wait()
    finally:
        timer.log_run_summary()


def _settle(ctx, seconds: float) -> None:
    """Rest between rounds, so the operator can reset the scene.

    Gives the arm time to stop at home, the cameras/USB buffers to settle, and
    a moment to put the ball back before the next round starts. Interruptible so
    a stop doesn't wait out the whole settle.
    """
    if seconds <= 0:
        return
    logger.info("LeLab rest between rounds: %.0fs", seconds)
    deadline = time.time() + seconds
    event = getattr(getattr(ctx, "runtime", None), "shutdown_event", None)
    while time.time() < deadline:
        if event is not None and event.is_set():
            return
        time.sleep(min(0.2, max(0.0, deadline - time.time())))


def main() -> None:
    opts = _parse_and_strip_lelab_args()
    alpha = opts.get("smooth_alpha", 0.6)
    rounds = max(1, int(opts.get("rounds", 1) or 1))
    rest_s = max(0.0, float(opts.get("rest_s", _DEFAULT_ROUND_REST_S) or 0.0))
    max_correction_s = float(opts.get("max_correction_s", _DEFAULT_MAX_CORRECTION_S) or 0.0)
    # Per-episode ceiling for the 修整 flow: 0 disables it.
    per_episode_s = float(opts.get("per_episode_s", 0.0) or 0.0)

    import lerobot.scripts.lerobot_rollout as lr

    # LeLab names capture datasets itself (from the dialog input) and manages
    # reuse via --dataset.root + --resume, so lerobot's per-create timestamp
    # stamping only makes the name long and diverge from what the user typed
    # (e.g. "rollout_1028_3" → "rollout_1028_3_20261028_163026"). Disable it.
    try:
        from lerobot.configs.dataset import DatasetRecordConfig

        DatasetRecordConfig.stamp_repo_id = lambda self: None
    except Exception:
        logger.warning("Could not disable dataset repo_id stamping", exc_info=True)

    orig_build = lr.build_rollout_context
    snapshot_buffer = SnapshotBuffer()
    _start_snapshot_watcher(snapshot_buffer)
    _start_memory_reclaimer()
    try:
        (_PREVIEW_DIR / _ROUND_END_FILE).unlink(missing_ok=True)
    except OSError:
        pass
    ctx_holder: dict = {}

    def build_with_smoothing(cfg, shutdown_event):
        ctx = orig_build(cfg, shutdown_event)
        ctx_holder["ctx"] = ctx
        # A slow camera (this rig's icspring stalls past lerobot's 200ms
        # async_read budget) raises TimeoutError mid-loop and kills the whole
        # rollout. Reuse the recording path's escalating-timeout + last-frame
        # fallback so a brief stall doesn't end the session.
        try:
            from .record import _harden_camera_reads

            _harden_camera_reads(ctx.hardware.robot_wrapper._robot)
        except Exception:
            logger.warning("Could not harden camera reads for rollout", exc_info=True)
        try:
            _harden_takeover_bus(ctx)
        except Exception:
            logger.warning("Could not harden the takeover bus", exc_info=True)
        # Say out loud which smoothing path this run actually took: RTC guidance
        # lives inside the policy (rtc_processor), not in the engine, so an RTC
        # engine with a processor-less policy silently behaves like sync.
        engine_name = type(ctx.policy.inference).__name__
        rtc_processor = getattr(ctx.policy.policy, "rtc_processor", None)
        if rtc_processor is not None and engine_name != "RTCInferenceEngine":
            logger.warning("Policy has an RTC processor but engine is %s — guidance unused", engine_name)
        elif engine_name == "RTCInferenceEngine" and rtc_processor is None:
            logger.warning("RTC engine without a policy RTC processor — guidance inactive")
        else:
            rtc_cfg = getattr(getattr(ctx.policy.policy, "config", None), "rtc_config", None)
            detail = ""
            if rtc_cfg is not None:
                detail = (
                    f" (horizon={getattr(rtc_cfg, 'execution_horizon', '?')},"
                    f" guidance={getattr(rtc_cfg, 'max_guidance_weight', '?')})"
                )
            logger.info(
                "Inference engine: %s%s | RTC guidance: %s | LeLab chunk smoothing: %s",
                engine_name,
                detail,
                "on" if rtc_processor is not None else "off",
                "off" if alpha >= 1.0 else f"alpha={alpha:.2f}",
            )
        keys = list(ctx.data.ordered_action_keys)
        passthrough = [i for i, k in enumerate(keys) if "gripper" in k.lower()]
        if alpha < 1.0:
            ctx.policy.inference = SmoothingInferenceEngine(
                ctx.policy.inference, alpha, passthrough, snapshot_buffer
            )
            logger.info(
                "LeLab chunk smoothing enabled (alpha=%.2f, passthrough channels=%s)",
                alpha,
                passthrough,
            )
        else:
            ctx.policy.inference = SmoothingInferenceEngine(
                ctx.policy.inference, 1.0, passthrough, snapshot_buffer
            )
        _start_preview_publisher(ctx)
        return ctx

    orig_create_strategy = lr.create_strategy

    def create_strategy_with_bridge(strategy_cfg):
        strategy = orig_create_strategy(strategy_cfg)
        # Recording strategies take their cues from a keyboard listener, which a
        # server-spawned subprocess has no access to; route them to the UI instead.
        if getattr(strategy_cfg, "type", "base") != "base":
            ctx = ctx_holder.get("ctx")
            _install_takeover_input(strategy, ctx)
            _start_state_writer(strategy, ctx)
            # Sentry/Highlight call `_background_push` on their own without
            # consulting `dataset.push_to_hub`, which would upload every few
            # episodes even though LeLab keeps capture sessions local.
            dataset_cfg = getattr(ctx.runtime.cfg, "dataset", None)
            if dataset_cfg is not None and not dataset_cfg.push_to_hub:
                strategy._background_push = lambda dataset, cfg: None
            # "This take is over" is the operator's round boundary: commit or
            # drop the frames, put the arm back where the attempt started, and
            # let the next one run on its own.
            if getattr(strategy_cfg, "type", "") == "dagger" and ctx is not None:
                _install_take_rounding(strategy, ctx)
                _install_episode_discard(strategy, ctx)
                _start_session_watchdog(strategy, ctx, max_correction_s, per_episode_s)
        # Continuous rounds for plain inference: keep ONE process (one model
        # load, one hardware connect) and loop `duration` rounds, returning home
        # and resting between them instead of tearing down and respawning.
        if getattr(strategy_cfg, "type", "base") == "base" and rounds > 1:
            round_end_path = _PREVIEW_DIR / _ROUND_END_FILE

            def run_rounds(ctx, _strategy=strategy):
                for r in range(rounds):
                    if ctx.runtime.shutdown_event.is_set():
                        break
                    logger.info("LeLab round %d/%d starting", r + 1, rounds)
                    _run_one_round(_strategy, ctx, round_end_path)
                    if r == rounds - 1 or ctx.runtime.shutdown_event.is_set():
                        break
                    try:
                        if _strategy._engine is not None:
                            _strategy._engine.reset()
                        if _strategy._interpolator is not None:
                            _strategy._interpolator.reset()
                        _strategy._cached_obs_processed = None
                    except Exception:
                        logger.warning("Engine reset between rounds failed", exc_info=True)
                    try:
                        _strategy.return_to_initial_position(ctx.hardware)
                    except Exception:
                        logger.warning("Return-to-home between rounds failed", exc_info=True)
                    _settle(ctx, rest_s)
                    logger.info("LeLab round %d/%d done", r + 1, rounds)

            strategy.run = run_rounds
        return strategy

    lr.build_rollout_context = build_with_smoothing
    lr.create_strategy = create_strategy_with_bridge
    # A `type: network` camera needs its config class registered and the camera
    # factory taught to build it before lerobot parses argv below.
    from .network_camera import install_camera_factory_hook

    install_camera_factory_hook()
    lr.main()


if __name__ == "__main__":
    main()
