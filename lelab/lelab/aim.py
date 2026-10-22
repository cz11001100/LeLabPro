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

"""Where does the policy *think* the ball is?

SmolVLA has no detector, so it never emits a position — it only emits joint
targets. Those targets still encode where the demonstrator was reaching, so the
policy's aim can be read out:

1. fit ``ball_pixel ≈ B @ [demonstrated_target, current_state, 1]`` on the
   dataset and score it on a held-out split;
2. push the *policy's* predicted target through the same map → the pixel the
   policy is implicitly aiming at;
3. compare with where the ball actually is (OpenCV colour mask), pushing the
   human's own target through the same map as the calibration baseline.

A forward regression replaces the earlier inverse one: solving joint-space for
pixel-space is ill-conditioned (the two pixel axes explain nearly the same joint
directions) and produced coordinates hundreds of thousands of pixels off.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _sample_indices(dataset, max_frame_index: int, n: int, seed: int) -> list[int]:
    import random

    from .validate import _frame_columns

    _ep, fr_col = _frame_columns(dataset)
    pool = [i for i, fr in enumerate(fr_col) if fr <= max_frame_index]
    return sorted(random.Random(seed).sample(pool, min(n, len(pool)))) if pool else []


def fit(dataset, camera: str, colors: list[str], horizon: int = 25, n_samples: int = 400,
        max_frame_index: int = 150, seed: int = 7) -> dict | None:
    """Learn "where was the ball" from "where did the arm go" plus the pose."""
    import numpy as np

    from .saliency import _to_uint8_hwc, find_ball

    feats, labels = [], []
    for index in _sample_indices(dataset, max_frame_index, n_samples, seed):
        item = dataset[index]
        ball = find_ball(_to_uint8_hwc(item[camera]), colors)
        if not ball:
            continue
        action = item["action"].float()
        if action.ndim == 1:
            raise ValueError("dataset 需要带 delta_timestamps 才能取到动作 chunk")
        state = item["observation.state"].float().reshape(-1)
        h = max(1, min(horizon, action.shape[0]))
        if float((action[h - 1] - state).abs().max()) < 2.0:
            continue  # the arm has not started reaching yet
        feats.append(action[h - 1].numpy().tolist() + state.tolist())
        labels.append([ball["cx"], ball["cy"]])

    if len(feats) < 40:
        logger.warning("Aim calibration only found %d usable frames", len(feats))
        return None

    x = np.column_stack([np.asarray(feats), np.ones(len(feats))])
    y = np.asarray(labels)

    rng = np.random.default_rng(seed)
    test = rng.permutation(len(x))[: max(10, len(x) // 4)]
    train = np.setdiff1d(np.arange(len(x)), test)
    b, *_ = np.linalg.lstsq(x[train], y[train], rcond=None)
    held = y[test] - x[test] @ b
    ss_res = (held**2).sum(axis=0)
    ss_tot = ((y[test] - y[train].mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-9)

    return {
        "B": b.T,  # maps [target, state, 1] → (x, y)
        "n_state": len(feats[0]) // 2,
        "r2_xy": [round(float(v), 3) for v in r2],
        "n_fit": len(feats),
        "horizon": horizon,
        "camera": camera,
        "mean_r2": round(float(r2.mean()), 3),
    }


def aim_point(fit_result: dict, target, state, size_hw: tuple[int, int] | None = None):
    """The pixel this joint target implies, clamped to the frame."""
    import numpy as np

    target = np.asarray(target, dtype=float).reshape(-1)
    state = np.asarray(state, dtype=float).reshape(-1)
    shared = min(target.shape[0], state.shape[0])
    target, state = target[:shared], state[:shared]
    if shared != fit_result["n_state"]:
        return None

    px = (fit_result["B"] @ np.concatenate([target, state, [1.0]])).tolist()
    if size_hw:
        height, width = size_hw
        px = [min(max(px[0], 0.0), float(width)), min(max(px[1], 0.0), float(height))]
    return round(px[0], 1), round(px[1], 1)


def _render(frame, ball, est_human, est_model, out_path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(6.6, 5))
    ax.imshow(frame)
    size = 64

    def box(centre, colour, label):
        if centre is None:
            return
        ax.add_patch(
            Rectangle((centre[0] - size / 2, centre[1] - size / 2), size, size, fill=False, ec=colour, lw=2.5)
        )
        ax.text(centre[0], centre[1] - size / 2 - 5, label, color=colour, fontsize=9, ha="center")

    box((ball["cx"], ball["cy"]) if ball else None, "#00ff00", "ball (true)")
    box(est_human, "#ff9800", "human aim")
    box(est_model, "#40c4ff", "model aim")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def probe(dataset, predict_item, indices: list[int], fit_result: dict, colors: list[str],
          out_dir: Path) -> dict:
    """Overlay the policy's implied aim point on the true target position.

    ``predict_item(item)`` returns ``([T, D] predicted chunk, state vector)``.
    """
    import numpy as np

    from .saliency import _to_uint8_hwc, find_ball

    out_dir.mkdir(parents=True, exist_ok=True)
    camera = fit_result["camera"]
    horizon = fit_result["horizon"]
    rows = []

    for slot, index in enumerate(indices):
        item = dataset[int(index)]
        frame = _to_uint8_hwc(item[camera])
        ball = find_ball(frame, colors)
        if not ball:
            continue
        chunk, state = predict_item(item)
        action = item["action"].float()
        h = max(1, min(horizon, chunk.shape[0], action.shape[0]))
        est_model = aim_point(fit_result, chunk[h - 1], state, frame.shape[:2])
        est_human = aim_point(fit_result, action[h - 1], state, frame.shape[:2])
        if est_model is None:
            continue

        err_model = float(np.hypot(est_model[0] - ball["cx"], est_model[1] - ball["cy"]))
        err_human = float(np.hypot(est_human[0] - ball["cx"], est_human[1] - ball["cy"])) if est_human else None
        name = f"aim_{slot}.png"
        _render(
            frame,
            ball,
            est_human,
            est_model,
            out_dir / name,
            f"frame {index} · model aim off by {err_model:.0f}px"
            + (f" · human {err_human:.0f}px" if err_human is not None else ""),
        )
        rows.append(
            {
                "frame": int(index),
                "ball_px": [round(ball["cx"], 1), round(ball["cy"], 1)],
                "model_px": list(est_model),
                "human_px": list(est_human) if est_human else None,
                "aim_error_model_px": round(err_model, 1),
                "aim_error_human_px": round(err_human, 1) if err_human is not None else None,
                "chart": name,
            }
        )

    if not rows:
        return {"rows": [], "verdict": "没有帧能同时检测到目标并完成映射"}

    model_err = float(np.mean([r["aim_error_model_px"] for r in rows]))
    human_errs = [r["aim_error_human_px"] for r in rows if r["aim_error_human_px"] is not None]
    human_err = float(np.mean(human_errs)) if human_errs else None
    r2 = fit_result["mean_r2"]

    if r2 < 0.4 or human_err is None or human_err > 150.0:
        verdict = f"映射质量不足（留出集 R²={r2}，人的还原偏差 {human_err:.0f}px），像素误差只作参考"
    elif model_err <= max(60.0, human_err * 1.8):
        verdict = f"模型指向位置平均偏差 {model_err:.0f}px（人的标定偏差 {human_err:.0f}px）→ 它确实对着球"
    else:
        verdict = (
            f"模型指向位置平均偏差 {model_err:.0f}px，而同样的映射作用在人的真实动作上只有 "
            f"{human_err:.0f}px → 模型指向的地方和球无关"
        )

    return {
        "rows": rows,
        "mean_aim_error_model_px": round(model_err, 1),
        "mean_aim_error_human_px": round(human_err, 1) if human_err is not None else None,
        "calibration_r2_xy": fit_result["r2_xy"],
        "mean_calibration_r2": r2,
        "n_fit": fit_result["n_fit"],
        "camera": camera,
        "horizon": horizon,
        "verdict": verdict,
    }
