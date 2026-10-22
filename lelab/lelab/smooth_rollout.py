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


def main() -> None:
    opts = _parse_and_strip_lelab_args()
    alpha = opts.get("smooth_alpha", 0.6)

    import lerobot.scripts.lerobot_rollout as lr

    orig_build = lr.build_rollout_context
    snapshot_buffer = SnapshotBuffer()
    _start_snapshot_watcher(snapshot_buffer)

    def build_with_smoothing(cfg, shutdown_event):
        ctx = orig_build(cfg, shutdown_event)
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

    lr.build_rollout_context = build_with_smoothing
    lr.main()


if __name__ == "__main__":
    main()
