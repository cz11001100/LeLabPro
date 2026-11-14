#!/usr/bin/env python
"""Closed-loop policy evaluation in the SO-101 tabletop MuJoCo scene.

Run with the LeLab Python environment (it has torch/lerobot):

    cd <仓库根>/sim
    MUJOCO_GL=egl <仓库根>/lerobot-env/bin/python \
        eval_policy.py --checkpoint <.../pretrained_model> --task "grasp the red ball" --episodes 5

Each episode resets the scene with a different seed (seed + episode index), so
the ball positions differ but remain reproducible. Results and videos go to
``out/eval_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")

HERE = Path(__file__).resolve().parent
REPO = HERE / "so101-tabletop-mujoco"
sys.path.insert(0, str(REPO))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env import SO101Env  # noqa: E402


def _default_dataset(ckpt: Path) -> str:
    train_cfg = ckpt / "train_config.json"
    if train_cfg.is_file():
        data = json.loads(train_cfg.read_text())
        repo_id = (data.get("dataset") or {}).get("repo_id")
        if repo_id:
            return str(repo_id)
    return "CZ11001100/1021_8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop SO-101 simulation evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint 目录（含 config.json/model.safetensors）")
    parser.add_argument("--dataset", type=str, default=None, help="训练该策略的数据集 repo_id；默认从 train_config.json 读")
    parser.add_argument("--task", type=str, default="grasp the red ball",
                        choices=["grasp the red ball", "grasp the blue ball"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0, help="起始 seed；第 i 回合使用 seed+i")
    parser.add_argument("--replan-steps", type=int, default=20,
                        help="每执行多少步重新推理一次（ACT 默认 n_action_steps=100）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    dataset_repo_id = args.dataset or _default_dataset(ckpt)

    out_dir = (args.out or (HERE / "out" / f"eval_{datetime.now():%Y%m%d_%H%M%S}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type == "cpu":
        print("[warn] CUDA not available, falling back to CPU")

    print("checkpoint:", ckpt)
    print("dataset:", dataset_repo_id)
    print("task:", args.task)
    print("output:", out_dir)
    print("device:", device)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors, make_policy
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action

    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    cfg.pretrained_path = str(ckpt)
    cfg.device = str(device)
    cfg.n_action_steps = max(1, min(int(cfg.n_action_steps or args.replan_steps), args.replan_steps))
    print("chunk_size:", cfg.chunk_size, "n_action_steps (replan):", cfg.n_action_steps)

    meta = LeRobotDatasetMetadata(dataset_repo_id)
    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.eval()
    policy.config.n_action_steps = cfg.n_action_steps

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    env = SO101Env(instruction=args.task)
    episodes: list[dict] = []
    started = time.time()

    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, reset_info = env.reset(seed=seed)
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            video_path = out_dir / f"episode_{ep:03d}_seed{seed}.mp4"
            writer = None
            if not args.no_video:
                writer = imageio.get_writer(
                    video_path, fps=env.fps, codec="libx264", quality=8, macro_block_size=None
                )

            max_lift = 0.0
            success = False
            step = 0
            ep_started = time.time()
            try:
                for step in range(args.max_steps):
                    batch = preprocess_observation(obs)
                    batch["task"] = [args.task]
                    processed = preprocessor(batch)

                    with torch.inference_mode():
                        policy_output = policy.select_action(processed)
                        action = transition_to_policy_action(policy_output_to_transition(policy_output))
                        action = postprocessor(action)

                    action_np = action.detach().cpu().numpy()
                    if action_np.ndim == 2:
                        action_np = action_np[0]

                    obs, reward, terminated, truncated, info = env.step(action_np)
                    max_lift = max(max_lift, float(info.get("lift", 0.0)))

                    if writer is not None:
                        writer.append_data(
                            np.hstack([obs["pixels"][key] for key in ("up", "side", "wrist")])
                        )
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()

            success = bool(info.get("is_success", False))
            record = {
                "episode": ep,
                "seed": seed,
                "success": success,
                "steps": step + 1,
                "max_lift_cm": round(max_lift * 100.0, 2),
                "ball_positions_cm": reset_info.get("ball_positions_cm"),
                "seconds": round(time.time() - ep_started, 2),
                "video": None if args.no_video else str(video_path.relative_to(out_dir)),
            }
            episodes.append(record)
            print(
                f"[{ep + 1}/{args.episodes}] seed={seed} success={success} "
                f"steps={record['steps']} max_lift={record['max_lift_cm']}cm "
                f"balls={record['ball_positions_cm']} time={record['seconds']}s"
            )
    finally:
        env.close()

    n_success = sum(1 for e in episodes if e["success"])
    summary = {
        "checkpoint": str(ckpt),
        "dataset_repo_id": dataset_repo_id,
        "task": args.task,
        "device": str(device),
        "episodes": args.episodes,
        "successes": n_success,
        "success_rate": round(n_success / max(1, args.episodes), 4),
        "start_seed": args.seed,
        "max_steps": args.max_steps,
        "replan_steps": cfg.n_action_steps,
        "total_seconds": round(time.time() - started, 2),
        "results": episodes,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    print(json.dumps({k: summary[k] for k in ("task", "episodes", "successes", "success_rate", "total_seconds")}, indent=2))
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
