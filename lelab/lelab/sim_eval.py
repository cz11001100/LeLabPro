"""Closed-loop MuJoCo evaluation for the SO-101 tabletop scene.

This is the "sim" mode of the job evaluator. It loads a LeRobot checkpoint,
drives the SO-101 in the local MuJoCo scene, and returns episode-level success
rates plus one three-view video per episode.

The scene is vendored in this repo under ``sim/so101-tabletop-mujoco``. It is
looked up from, in order: an explicit path, ``LELAB_SIM_DIR``, the repo-relative
``sim/so101-tabletop-mujoco``, and ``~/LeLabPro/sim/so101-tabletop-mujoco``. Set
``LELAB_SIM_DIR`` when the scene lives elsewhere.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")

logger = logging.getLogger(__name__)


def _scene_dir(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("LELAB_SIM_DIR"):
        candidates.append(Path(os.environ["LELAB_SIM_DIR"]).expanduser())
    candidates += [
        Path(__file__).resolve().parents[2] / "sim" / "so101-tabletop-mujoco",
        Path.home() / "LeLabPro" / "sim" / "so101-tabletop-mujoco",
    ]
    for candidate in candidates:
        if (candidate / "env.py").is_file() and (candidate / "so101_tabletop.xml").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MuJoCo scene not found. Set LELAB_SIM_DIR to the directory containing env.py "
        f"and so101_tabletop.xml. Tried: {[str(c) for c in candidates]}"
    )


def _load_sim_env(scene_dir: Path):
    """Load the scene's env.py without clobbering LeLab's imported modules."""
    module_name = "lelab_so101_tabletop_env"
    spec = importlib.util.spec_from_file_location(module_name, scene_dir / "env.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load env.py from {scene_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _default_dataset(checkpoint_path: Path) -> str:
    train_cfg = checkpoint_path / "train_config.json"
    if train_cfg.is_file():
        try:
            data = json.loads(train_cfg.read_text())
            repo_id = (data.get("dataset") or {}).get("repo_id")
            if repo_id:
                return str(repo_id)
        except Exception:
            pass
    return ""


def run_sim_evaluation(
    checkpoint_path: str,
    dataset_repo_id: str,
    task: str = "grasp the red ball",
    episodes: int = 5,
    max_steps: int = 300,
    seed: int = 0,
    replan_steps: int = 20,
    device: str = "cuda",
    output_dir: str | None = None,
    scene_dir: str | None = None,
    cancel_event=None,
) -> dict:
    """Run closed-loop episodes and return a JSON-serialisable result dict."""
    started = time.time()
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.is_dir():
        return {"success": False, "mode": "sim", "error": f"checkpoint not found: {ckpt}"}

    dataset_repo_id = dataset_repo_id or _default_dataset(ckpt)
    if not dataset_repo_id:
        return {"success": False, "mode": "sim", "error": "dataset_repo_id is empty"}

    out_dir = Path(output_dir or (Path.cwd() / "evaluation" / datetime.now().strftime("%Y%m%d_%H%M%S")))
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        scene = _scene_dir(scene_dir)
        sim = _load_sim_env(scene)
    except Exception as exc:
        logger.exception("Failed to load MuJoCo scene")
        return {"success": False, "mode": "sim", "error": f"scene load failed: {exc}"}

    if task not in sim.TARGET_BALLS:
        return {
            "success": False,
            "mode": "sim",
            "error": f"task {task!r} is not supported by this scene; choose {sorted(sim.TARGET_BALLS)}",
        }

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_pre_post_processors, make_policy
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device == "cuda" and dev.type == "cpu":
        logger.warning("CUDA requested but unavailable; evaluating on CPU")

    logger.info("Sim eval checkpoint=%s dataset=%s task=%s device=%s", ckpt, dataset_repo_id, task, dev)
    try:
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        cfg.pretrained_path = str(ckpt)
        cfg.device = str(dev)
        cfg.n_action_steps = max(1, min(int(cfg.n_action_steps or replan_steps), int(replan_steps)))

        meta = LeRobotDatasetMetadata(dataset_repo_id)
        policy = make_policy(cfg=cfg, ds_meta=meta)
        policy.eval()
        policy.config.n_action_steps = cfg.n_action_steps

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": str(dev)}},
        )
    except Exception as exc:
        logger.exception("Failed to load policy/dataset for sim evaluation")
        return {
            "success": False,
            "mode": "sim",
            "error": f"policy load failed: {exc}",
            "checkpoint_path": str(ckpt),
            "dataset_repo_id": dataset_repo_id,
        }

    try:
        env = sim.SO101Env(instruction=task)
    except Exception as exc:
        return {"success": False, "mode": "sim", "error": f"env creation failed: {exc}"}

    results: list[dict] = []
    videos: list[str] = []
    cancelled = False
    try:
        for ep in range(int(episodes)):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            ep_seed = int(seed) + ep
            obs, reset_info = env.reset(seed=ep_seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            video_name = f"episode_{ep:03d}_seed{ep_seed}.mp4"
            writer = imageio.get_writer(
                out_dir / video_name, fps=env.fps, codec="libx264", quality=8, macro_block_size=None
            )
            videos.append(video_name)

            max_lift = 0.0
            success = False
            step = 0
            info: dict[str, Any] = {}
            ep_started = time.time()
            try:
                for step in range(int(max_steps)):
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break

                    batch = preprocess_observation(obs)
                    batch["task"] = [task]
                    processed = preprocessor(batch)
                    with torch.inference_mode():
                        policy_output = policy.select_action(processed)
                        action = transition_to_policy_action(policy_output_to_transition(policy_output))
                        action = postprocessor(action)

                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim == 2:
                        action_np = action_np[0]

                    obs, _reward, terminated, truncated, info = env.step(action_np)
                    max_lift = max(max_lift, float(info.get("lift", 0.0)))
                    writer.append_data(
                        np.hstack([obs["pixels"][key] for key in ("up", "side", "wrist")])
                    )
                    if terminated or truncated:
                        break
            finally:
                writer.close()

            if cancelled:
                break

            success = bool(info.get("is_success", False))
            results.append(
                {
                    "episode": ep,
                    "seed": ep_seed,
                    "success": success,
                    "steps": step + 1,
                    "max_lift_cm": round(max_lift * 100.0, 2),
                    "ball_positions_cm": reset_info.get("ball_positions_cm"),
                    "seconds": round(time.time() - ep_started, 2),
                }
            )
            logger.info(
                "sim eval episode %s/%s seed=%s success=%s steps=%s lift=%.1fcm",
                ep + 1, episodes, ep_seed, success, step + 1, max_lift * 100.0,
            )
    except Exception as exc:
        logger.exception("Sim evaluation failed")
        return {"success": False, "mode": "sim", "error": str(exc), "traceback": __import__("traceback").format_exc()}
    finally:
        env.close()

    if cancelled:
        return {"success": False, "mode": "sim", "cancelled": True, "error": "评测已取消"}

    n_success = sum(1 for row in results if row["success"])
    return {
        "success": True,
        "mode": "sim",
        "checkpoint_path": str(ckpt),
        "dataset_repo_id": dataset_repo_id,
        "task": task,
        "device": str(dev),
        "episodes": len(results),
        "successes": n_success,
        "success_rate": round(n_success / max(1, len(results)), 4),
        "start_seed": int(seed),
        "max_steps": int(max_steps),
        "replan_steps": int(cfg.n_action_steps),
        "total_seconds": round(time.time() - started, 2),
        "sim_episode_results": results,
        "videos": videos,
    }
