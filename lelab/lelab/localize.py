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
# limitations under the the License.

"""Real-robot localisation probe — does the policy aim at the object?

The offline probe shifts the *image*, which is only a proxy. Here the object
itself moves: put the ball down, capture one observation, move it, capture
again. The arm is never commanded — this script only reads.

For every capture we record where an OpenCV colour mask says the ball is (per
camera) and what the policy's predicted move is (predicted action minus current
joint angles, in degrees). A policy that localises the target produces moves
that track the ball's pixel position, so we report Spearman correlation between
the two plus the raw spread of the predictions.

    python -m lelab.localize --policy <pretrained_model dir> --robot-name 001 \
        --task "grasp the green ball" --trials 6 --interval 20

``--replay`` swaps the live robot for dataset frames so the whole pipeline can
be exercised without hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROBOTS_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "robots"
COLORS = ("green", "orange", "red", "blue")


def _colors_for(task: str) -> list[str]:
    found = [c for c in COLORS if c in (task or "").lower()]
    return found or list(COLORS)


def read_robot_config(name: str) -> dict:
    path = ROBOTS_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"robot config {path} not found")
    return json.loads(path.read_text())


def camera_arg(cameras: list[dict]) -> dict[str, dict]:
    """LeLab's stored camera list → lerobot's ``--robot.cameras`` mapping.

    A stream URL goes out as ``type: network``, whose config types ``index_or_path``
    as a string; the ``opencv`` config would coerce a URL into a Path and drop the
    ``//`` after the scheme.
    """
    from .network_camera import camera_source, install_camera_factory_hook

    out: dict[str, dict] = {}
    for cam in cameras:
        kind, source = camera_source(cam)
        if kind == "network":
            install_camera_factory_hook()
            out[cam["name"]] = {
                "type": "network",
                "index_or_path": str(source),
                "width": cam.get("width", 640),
                "height": cam.get("height", 480),
                "fps": cam.get("fps", 30),
            }
            continue
        out[cam["name"]] = {
            "type": kind,
            "index_or_path": source if source is not None else 0,
            "width": cam.get("width", 640),
            "height": cam.get("height", 480),
            "fps": cam.get("fps", 30),
        }
    return out


class Predictor:
    """Loads a checkpoint and turns an observation dict into an action chunk."""

    def __init__(self, checkpoint: str, device: str, dataset_repo_id: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        # Hub revalidation of the VLM backbone costs ~90s on a slow link.
        import os

        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.device = str(self.device)
        self.cfg = cfg

        # make_policy needs dataset metadata for the feature/shape contract.
        # The checkpoint records which dataset it was trained on, so live mode
        # does not have to be told again.
        repo_id = dataset_repo_id
        if not repo_id:
            train_cfg = Path(checkpoint) / "train_config.json"
            if train_cfg.is_file():
                repo_id = (json.loads(train_cfg.read_text()).get("dataset") or {}).get("repo_id")
        if not repo_id:
            raise ValueError("无法确定数据集：请传 --dataset <repo_id>")
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self.ds_meta = LeRobotDatasetMetadata(repo_id)
        self.policy = make_policy(cfg=cfg, ds_meta=self.ds_meta)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def chunk(self, observation: dict, task: str):
        """Predict one action chunk. ``observation`` holds HWC numpy/CHW tensors."""
        import torch

        batch: dict = {}
        for key, value in observation.items():
            if "image" in key:
                t = value if torch.is_tensor(value) else torch.from_numpy(value)
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC → CHW
                    t = t.permute(2, 0, 1)
                batch[key] = t.unsqueeze(0).float()
            elif key in ("task", "robot_type"):
                batch[key] = [value]
            else:
                t = value if torch.is_tensor(value) else torch.from_numpy(value)
                batch[key] = t.float().reshape(1, -1)

        with torch.inference_mode():
            processed = self.pre(batch)
            obs = {k: v for k, v in processed.items() if k != "action"}
            raw = self.policy.predict_action_chunk(obs)
            chunk = self.post(raw.to(self.device)).float().cpu()[0]
        return chunk  # [T, D] degrees


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, idx in enumerate(order):
        ranks[idx] = float(rank)
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    import math

    a, b = _rank(xs), _rank(ys)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return round(num / (da * db), 3) if da > 0 and db > 0 else None


def summarise(trials: list[dict], joint_names: list[str]) -> dict:
    """Correlate where the ball is with which way the policy wants to move."""
    usable = [t for t in trials if t.get("ball_px")]
    out: dict = {
        "trials": len(trials),
        "trials_with_ball": len(usable),
        "move_spread_deg": {},
        "corr_with_ball_x": {},
        "corr_with_ball_y": {},
    }
    if not trials:
        return out

    for j, name in enumerate(joint_names):
        moves = [t["move_deg"][j] for t in trials]
        out["move_spread_deg"][name] = round(max(moves) - min(moves), 2)

    if len(usable) >= 3:
        xs = [t["ball_px"][0] for t in usable]
        ys = [t["ball_px"][1] for t in usable]
        for j, name in enumerate(joint_names):
            out["corr_with_ball_x"][name] = spearman(xs, [t["move_deg"][j] for t in usable])
            out["corr_with_ball_y"][name] = spearman(ys, [t["move_deg"][j] for t in usable])

    pairs = [(k, v) for k, v in out["corr_with_ball_x"].items() if v is not None]
    pairs += [(k, v) for k, v in out["corr_with_ball_y"].items() if v is not None]
    best = max(pairs, key=lambda kv: abs(kv[1]), default=None)
    spread = max(out["move_spread_deg"].values()) if out["move_spread_deg"] else 0.0

    if len(usable) < 3:
        out["verdict"] = "样本不足（少于 3 次带球位置的采集），无法判定"
    elif spread < 1.5:
        out["verdict"] = f"球换了位置，但预测动作几乎没变（最大幅度 {spread:.1f}°）→ 它不看球在哪"
    elif best and abs(best[1]) >= 0.7:
        out["verdict"] = f"能定位：{best[0]} 的预测随球位置单调变化（ρ={best[1]}）"
    elif best and abs(best[1]) >= 0.4:
        out["verdict"] = f"部分定位：最强相关 {best[0]} ρ={best[1]}，但还不够稳"
    else:
        out["verdict"] = f"预测会变（幅度 {spread:.1f}°）但方向与球位置不相关（最强 ρ={best[1] if best else '—'}）"
    out["best_correlation"] = f"{best[0]}:{best[1]}" if best else None
    return out


def _save_frame(frame, ball, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.imshow(frame)
    if ball:
        ax.add_patch(
            Rectangle((ball["cx"] - ball["w"] / 2, ball["cy"] - ball["h"] / 2), ball["w"], ball["h"], fill=False, ec="#00ff00", lw=2.5)
        )
        ax.text(ball["cx"], ball["cy"] - ball["h"] / 2 - 4, "ball (CV)", color="#00ff00", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run(args) -> dict:
    from .saliency import _to_uint8_hwc, find_ball

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(args.policy, args.device, args.dataset)
    colors = _colors_for(args.task)
    state_feat = (predictor.ds_meta.info.get("features") or {}).get("observation.state") or {}
    joint_names = [str(n).replace(".pos", "") for n in (state_feat.get("names") or [])]

    if args.replay:
        trials = _replay(args, predictor, colors, out_dir, joint_names)
    else:
        trials = _live(args, predictor, colors, out_dir, joint_names)

    default_joints = [f"joint_{i}" for i in range(len(trials[0]["move_deg"]))] if trials else []
    summary = summarise(trials, joint_names or default_joints)
    if args.replay:
        summary["mode"] = "replay"
        summary["caveat"] = (
            "回放模式：这些帧的机械臂状态也各不相同，动作幅度差异主要来自关节当前姿态"
            "而不是球的位置，相关系数只用于验证链路，不能作为定位结论。"
        )
    result = {
        "policy": args.policy,
        "task": args.task,
        "mode": "replay" if args.replay else "robot",
        "trials": trials,
        "summary": summary,
    }
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _record(predictor, observation: dict, args, colors, out_dir: Path, slot: int, joint_names):
    import torch

    from .saliency import _to_uint8_hwc, find_ball

    chunk = predictor.chunk(observation, args.task)
    state = observation["observation.state"]
    state_vec = [float(x) for x in torch.as_tensor(state).reshape(-1).tolist()]
    horizon = max(1, min(args.horizon, chunk.shape[0]))
    move = [round(float(chunk[horizon - 1][j]) - state_vec[j], 2) for j in range(len(state_vec))]
    row = {"index": slot, "move_deg": move, "state_deg": [round(v, 2) for v in state_vec], "horizon": horizon, "ball_by_camera": {}}

    for key in sorted(k for k in observation if "image" in k and not k.endswith("_is_pad")):
        raw = torch.as_tensor(observation[key])
        if raw.ndim == 3 and raw.shape[-1] in (1, 3) and raw.shape[0] not in (1, 3):
            raw = raw.permute(2, 0, 1)
        frame = _to_uint8_hwc(raw)
        ball = find_ball(frame, colors)
        name = key.split(".")[-1]
        _save_frame(
            frame,
            ball,
            out_dir / f"trial_{slot}_{name}.png",
            f"#{slot} {name} · " + (f"ball @ ({ball['cx']:.0f},{ball['cy']:.0f})" if ball else "未检测到目标"),
        )
        row["ball_by_camera"][name] = [round(ball["cx"], 1), round(ball["cy"], 1)] if ball else None
        if ball and not row.get("ball_px"):
            row["ball_px"] = row["ball_by_camera"][name]
            row["ball_camera"] = name
    ref = row["ball_by_camera"].get(args.ref_camera)
    if ref:
        row["ball_px"], row["ball_camera"] = ref, args.ref_camera
    return row


def _live(args, predictor, colors, out_dir: Path, joint_names) -> list[dict]:
    from lerobot.robots import RobotConfig, make_robot_from_config

    cfg_raw = read_robot_config(args.robot_name)
    cams = camera_arg(cfg_raw["cameras"])
    robot_cfg = RobotConfig.from_dict(
        {
            "type": args.robot_type,
            "port": cfg_raw["follower_port"],
            "id": args.robot_name,
            "cameras": cams,
        }
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print(f"已连接 {args.robot_name}（{cfg_raw['follower_port']}），只读取观测，不会下发动作", flush=True)

    rows = []
    try:
        for slot in range(1, args.trials + 1):
            print(f"→ 请把球放到第 {slot} 个位置，{args.interval} 秒后自动采集", flush=True)
            time.sleep(args.interval)
            obs = robot.get_observation()
            row = _record(predictor, obs, args, colors, out_dir, slot, joint_names)
            row["index"] = slot
            rows.append(row)
            print(
                f"  #{slot} 球位置={row.get('ball_px')} 预测移动={row['move_deg']}",
                flush=True,
            )
    finally:
        robot.disconnect()
    return rows


def _replay(args, predictor, colors, out_dir: Path, joint_names) -> list[dict]:
    """Same prediction path, fed from dataset frames — no hardware needed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from .utils.video_backend import video_backend

    dataset = LeRobotDataset(args.dataset, video_backend=video_backend())
    indices = [int(x) for x in str(args.replay_indices).split(",") if x.strip()]
    rows = []
    for slot, index in enumerate(indices, start=1):
        item = dataset[int(index)]
        obs = {k: v for k, v in item.items() if k.startswith("observation")}
        obs["task"] = args.task or str(item.get("task", ""))
        row = _record(predictor, obs, args, colors, out_dir, slot, joint_names)
        row["index"] = slot
        row["dataset_frame"] = int(index)
        rows.append(row)
        print(f"  #{slot} frame={index} 球位置={row.get('ball_px')} 预测移动={row['move_deg']}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Real-robot localisation probe")
    parser.add_argument("--policy", required=True, help="pretrained_model dir")
    parser.add_argument("--robot-name", default="001", help="stored LeLab robot config")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--task", default="grasp the green ball")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--interval", type=int, default=20, help="seconds to reposition the ball")
    parser.add_argument("--horizon", type=int, default=25, help="use the action N steps ahead as the intended move")
    parser.add_argument("--ref-camera", default="up")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="/tmp/lelab_localize")
    parser.add_argument("--replay", action="store_true", help="use dataset frames instead of the robot")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--replay-indices", default="0,60,120,180,240,300")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = run(args)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"RESULT_JSON {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
