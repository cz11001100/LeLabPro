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

"""Where does the policy think the object is?

Policies like SmolVLA/ACT have no detection head, so there is no real bounding
box to draw. What we *can* measure is causal saliency: blank out one tile of a
camera at a time, re-predict the action chunk, and record how much the action
moved. Tiles that make the action move a lot are the tiles the policy depends
on — that is the closest honest answer to "what it sees".

The OpenCV HSV colour mask on the same frame gives the ball's true position, so
the overlay shows two boxes: where the policy looks, and where the ball is.

Usage:
    python -m lelab.saliency --checkpoint <pretrained_model dir> \
        --dataset CZ11001100/1011_4 --frames 120,3400 --camera observation.images.up
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

COLOR_RANGES = {
    # HSV upper/lower bounds, OpenCV convention (H 0-179).
    "green": [((35, 60, 60), (85, 255, 255))],
    "orange": [((5, 80, 45), (25, 255, 255)), ((168, 80, 45), (179, 255, 255))],
    # The red ball renders dark (V≈66) under the top light, so the value floor
    # has to stay low or the target silently disappears from the mask.
    "red": [((0, 60, 35), (12, 255, 255)), ((168, 60, 35), (179, 255, 255))],
    "blue": [((100, 80, 80), (130, 255, 255))],
}


def load_for_saliency(checkpoint_path: str, dataset_repo_id: str, dataset_root: str | None, device: str):
    """Load policy + dataset exactly the way the robot-side inference does."""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    # Hub revalidation hangs for minutes on a slow/blocked link even when the
    # weights are cached locally; fail fast and fall back to the cache.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
    cfg.pretrained_path = checkpoint_path
    cfg.device = str(dev)

    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root or None)
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root or None,
        delta_timestamps=resolve_delta_timestamps(cfg, ds_meta),
    )
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint_path,
        preprocessor_overrides={"device_processor": {"device": str(dev)}},
    )
    return cfg, ds_meta, dataset, policy, preprocessor, postprocessor, dev


def find_ball(img_hwc_uint8, colors: list[str]):
    """Locate the ball with an HSV colour mask. Returns (cx, cy, w, h) or None."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(img_hwc_uint8, cv2.COLOR_RGB2HSV)
    best = None
    for color in colors:
        for lo, hi in COLOR_RANGES.get(color, []):
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]:
                area = cv2.contourArea(cnt)
                if area < 150:  # ignore specks
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if best is None or area > best[-1]:
                    best = (x + w / 2, y + h / 2, w, h, area)
    if best is None:
        return None
    frame_area = img_hwc_uint8.shape[0] * img_hwc_uint8.shape[1]
    if best[4] > 0.4 * frame_area:  # a surface filling the view, not an object
        return None
    return {"cx": best[0], "cy": best[1], "w": best[2], "h": best[3], "area": best[4]}


def _to_uint8_hwc(tensor):
    """[C,H,W] float in [0,1] (or uint8) → HWC uint8 numpy."""
    import torch

    t = tensor.detach().float().cpu()
    if t.max() > 1.5:
        arr = t.permute(1, 2, 0).clamp(0, 255).byte().numpy()
    else:
        arr = (t.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return arr


def occlusion_saliency(
    dataset,
    policy,
    preprocessor,
    postprocessor,
    index: int,
    camera_key: str,
    grid: int = 6,
    tile_batch: int = 6,
    device: str = "cuda",
    fill: str = "black",
):
    """Measure |Δaction| for each occluded tile of one camera.

    ``fill`` chooses the replacement: ``black`` is itself a strong out-of-
    distribution cue, so a flat map with it is not conclusive; ``mean`` blends
    the tile into the scene's own colours, which isolates *where* the action
    depends on the content rather than on the black square.

    Returns (heat[grid, grid] in degrees, baseline prediction, frame uint8 HWC).
    """
    import torch
    from torch.utils.data import default_collate

    sample = dataset[index]
    base_batch = default_collate([sample])
    heat = torch.zeros(grid, grid)

    image = base_batch[camera_key][0]  # [C,H,W]
    _, height, width = image.shape
    fill_value = torch.zeros(image.shape[0], device=image.device) if fill == "black" else image.mean(dim=(1, 2))

    def predict(batch):
        from .validate import _predict

        pred, gt, _, _ = _predict(policy, preprocessor, postprocessor, batch, torch.device(device), want_loss=False)
        return pred[0]

    baseline = predict(base_batch)

    tiles = [(r, c) for r in range(grid) for c in range(grid)]
    for start in range(0, len(tiles), tile_batch):
        chunk = tiles[start : start + tile_batch]
        views = []
        for r, c in chunk:
            masked = image.clone()
            y0, y1 = int(r * height / grid), int((r + 1) * height / grid)
            x0, x1 = int(c * width / grid), int((c + 1) * width / grid)
            masked[:, y0:y1, x0:x1] = fill_value[:, None, None]
            views.append(masked)
        batch = dict(base_batch)
        batch[camera_key] = torch.stack(views).to(device)
        # Everything else keeps batch size 1 → broadcast against the tile stack.
        for k, v in batch.items():
            if k != camera_key and torch.is_tensor(v) and v.shape[0] == 1:
                batch[k] = v.expand(len(chunk), *v.shape[1:]).contiguous()
            if k == "task":
                batch[k] = list(v) * len(chunk)
        with torch.inference_mode():
            preds = predict(batch)
        for i, (r, c) in enumerate(chunk):
            heat[r, c] = (preds[i] - baseline).abs().mean().item()

    return heat.numpy(), baseline, _to_uint8_hwc(image)


def _render(frame, heat, ball, peaks, out_path: Path, title: str):
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(frame)

    h, w = heat.shape
    extent = (0, frame.shape[1], frame.shape[0], 0)
    lo, hi = float(heat.min()), float(heat.max())
    norm = (heat - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(heat)
    rgba = plt.cm.jet(norm)
    rgba[..., 3] = norm**2 * 0.6  # flat maps stay transparent, hot tiles stand out
    ax.imshow(rgba, extent=extent, interpolation="nearest")

    if ball:
        bw = max(ball["w"], frame.shape[1] / w) * 1.6
        bh = max(ball["h"], frame.shape[0] / h) * 1.6
        ax.add_patch(
            Rectangle(
                (ball["cx"] - bw / 2, ball["cy"] - bh / 2), bw, bh, fill=False, ec="#00ff00", lw=2.5
            )
        )
        ax.text(ball["cx"], ball["cy"] - bh / 2 - 4, "ball (CV)", color="#00ff00", fontsize=8)

    for i, (r, c) in enumerate(peaks):
        x0 = c * frame.shape[1] / w
        y0 = r * frame.shape[0] / h
        ax.add_patch(
            Rectangle(
                (x0, y0), frame.shape[1] / w, frame.shape[0] / h, fill=False, ec="white", lw=2.0
            )
        )
        ax.text(x0 + 3, y0 + 14, f"#{i + 1}", color="white", fontsize=9, weight="bold")

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def run(
    checkpoint_path: str,
    dataset_repo_id: str,
    dataset_root: str | None,
    frames: list[int],
    cameras: list[str],
    grid: int,
    colors: list[str],
    out_dir: str,
    device: str,
    tile_batch: int,
    fill: str = "black",
) -> dict:
    import numpy as np

    cfg, ds_meta, dataset, policy, pre, post, dev = load_for_saliency(
        checkpoint_path, dataset_repo_id, dataset_root, device
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_keys = [k for k in cfg.input_features if "image" in k]
    cams = cameras or image_keys

    rows = []
    for index in frames:
        for cam in cams:
            if cam not in image_keys:
                logger.warning("Camera %s not in this policy's inputs, skipping", cam)
                continue
            heat, _, frame = occlusion_saliency(
                dataset, policy, pre, post, index, cam, grid=grid, tile_batch=tile_batch, device=str(dev), fill=fill
            )
            ball = find_ball(frame, colors)
            # Subtract the median: when every tile matters equally the map
            # should render flat, not as a solid wash over the frame.
            rel = heat - float(np.median(heat))
            order = np.argsort(rel.flatten())[::-1]
            peaks = [(int(i // grid), int(i % grid)) for i in order[:3]]

            h_px, w_px = frame.shape[0] / grid, frame.shape[1] / grid
            mass = 0.0
            dist = None
            if ball:
                ball_r, ball_c = int(ball["cy"] // h_px), int(ball["cx"] // w_px)
                for r in range(max(0, ball_r - 1), min(grid, ball_r + 2)):
                    for c in range(max(0, ball_c - 1), min(grid, ball_c + 2)):
                        mass += max(0.0, float(rel[r, c]))
                peak_r, peak_c = peaks[0]
                dist = float(
                    np.hypot(peak_c * w_px + w_px / 2 - ball["cx"], peak_r * h_px + h_px / 2 - ball["cy"])
                )
            positive = float(np.clip(rel, 0, None).sum()) or 1.0
            name = f"sal_{index}_{cam.split('.')[-1]}.png"
            _render(
                frame,
                rel,
                ball,
                peaks,
                out / name,
                f"frame {index} · {cam.split('.')[-1]} · peak->ball {dist:.0f}px" if dist is not None else f"frame {index} · {cam}",
            )
            row = {
                "frame": index,
                "camera": cam,
                "peak_tile": list(peaks[0]),
                "top3_tiles": [list(p) for p in peaks],
                "delta_max_deg": round(float(heat.max()), 2),
                "delta_median_deg": round(float(np.median(heat)), 2),
                "contrast_ratio": round(float(heat.max() / (float(np.median(heat)) + 1e-9)), 2),
                "saliency_on_ball": round(mass / positive, 4) if ball else None,
                "peak_to_ball_px": round(dist, 1) if dist is not None else None,
                "ball_px": [round(ball["cx"], 1), round(ball["cy"], 1)] if ball else None,
                "image": name,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    return {"frames": rows, "grid": grid, "cameras": cams, "output_dir": str(out)}


def attach_saliency(
    dataset,
    policy,
    preprocessor,
    postprocessor,
    device,
    indices: list[int],
    camera_key: str,
    out_dir: Path,
    grid: int = 6,
    tile_batch: int = 6,
    fill: str = "mean",
    colors: list[str] | None = None,
) -> dict:
    """Run the occlusion probe on an already-loaded policy.

    The evaluator calls this so the overlay shares one model load. Returns the
    per-frame rows plus two summary numbers the UI can show directly.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = colors or ["green", "orange"]
    rows = []
    for slot, index in enumerate(indices):
        heat, _, frame = occlusion_saliency(
            dataset,
            policy,
            preprocessor,
            postprocessor,
            index,
            camera_key,
            grid=grid,
            tile_batch=tile_batch,
            device=str(device),
            fill=fill,
        )
        ball = find_ball(frame, colors)
        rel = heat - float(np.median(heat))
        order = np.argsort(rel.flatten())[::-1]
        peaks = [(int(i // grid), int(i % grid)) for i in order[:3]]
        tile_h, tile_w = frame.shape[0] / grid, frame.shape[1] / grid

        dist = None
        on_ball = None
        if ball:
            dist = float(
                np.hypot(
                    peaks[0][1] * tile_w + tile_w / 2 - ball["cx"],
                    peaks[0][0] * tile_h + tile_h / 2 - ball["cy"],
                )
            )
            positive = float(np.clip(rel, 0, None).sum()) or 1.0
            br, bc = int(ball["cy"] // tile_h), int(ball["cx"] // tile_w)
            mass = sum(
                max(0.0, float(rel[r, c]))
                for r in range(max(0, br - 1), min(grid, br + 2))
                for c in range(max(0, bc - 1), min(grid, bc + 2))
            )
            on_ball = round(mass / positive, 3)

        name = f"saliency_{slot}.png"
        _render(
            frame,
            rel,
            ball,
            peaks,
            out_dir / name,
            f"frame {index} · {camera_key.split('.')[-1]}"
            + (f" · peak->ball {dist:.0f}px" if dist is not None else " · 未检测到目标"),
        )
        rows.append(
            {
                "frame": int(index),
                "camera": camera_key,
                "peak_to_ball_px": round(dist, 1) if dist is not None else None,
                "saliency_on_ball": on_ball,
                "chart": name,
            }
        )

    found = [r for r in rows if r["peak_to_ball_px"] is not None]
    on_ball_vals = [r["saliency_on_ball"] for r in rows if r["saliency_on_ball"] is not None]
    return {
        "rows": rows,
        "mean_peak_to_ball_px": round(float(np.mean([r["peak_to_ball_px"] for r in found])), 1) if found else None,
        "mean_saliency_on_ball": round(float(np.mean(on_ball_vals)), 3) if on_ball_vals else None,
        "chance_level": round(9 / (grid * grid), 3),
    }


def roi_ablation(
    dataset,
    policy,
    preprocessor,
    postprocessor,
    device,
    index: int,
    camera_key: str,
    colors: list[str],
    n_random: int = 8,
    fill: str = "mean",
    seed: int = 3,
) -> dict | None:
    """Grounding test used in the VLA literature: hide the target itself and
    compare with hiding equal-area random patches.

    A policy that uses the object's position reacts far more to the target
    ablation than to a random one; a policy that merely notices "the image
    changed" reacts the same to both.
    """
    import random

    import torch
    from torch.utils.data import default_collate

    from .validate import _predict

    sample = dataset[index]
    batch = default_collate([sample])
    image = batch[camera_key][0]
    _, height, width = image.shape
    frame = _to_uint8_hwc(image)
    ball = find_ball(frame, colors)
    if not ball:
        return None

    dev = torch.device(device)
    base, _, _, _ = _predict(policy, preprocessor, postprocessor, batch, dev, want_loss=False)
    base = base[0]

    area = max(1, int(ball["w"]) * int(ball["h"]))
    side = max(16, int(area**0.5))

    def patch_delta(x0: int, y0: int, w: int, h: int) -> float:
        masked = image.clone()
        masked[:, y0 : min(height, y0 + h), x0 : min(width, x0 + w)] = (
            torch.zeros(image.shape[0], device=image.device)
            if fill == "black"
            else image.mean(dim=(1, 2)).view(-1, 1, 1)
        )
        view = batch.copy()
        view[camera_key] = masked.unsqueeze(0)
        pred, _, _, _ = _predict(policy, preprocessor, postprocessor, view, dev, want_loss=False)
        return float((pred[0] - base).abs().mean())

    bx = max(0, int(ball["cx"] - side / 2))
    by = max(0, int(ball["cy"] - side / 2))
    delta_target = patch_delta(bx, by, side, side)

    rng = random.Random(seed)
    randoms = []
    for _ in range(n_random):
        x0 = rng.randint(0, max(0, width - side))
        y0 = rng.randint(0, max(0, height - side))
        if abs(x0 - bx) < side and abs(y0 - by) < side:
            continue  # don't let a "random" patch overlap the target
        randoms.append(patch_delta(x0, y0, side, side))
    if not randoms:
        return None

    delta_random = sum(randoms) / len(randoms)
    ratio = delta_target / max(delta_random, 1e-6)
    return {
        "frame": int(index),
        "camera": camera_key,
        "patch_px": side,
        "delta_target_deg": round(delta_target, 2),
        "delta_random_deg": round(delta_random, 2),
        "ratio": round(ratio, 2),
        "n_random": len(randoms),
    }


def roi_grounding_summary(
    dataset,
    policy,
    preprocessor,
    postprocessor,
    device,
    indices: list[int],
    cameras: list[str],
    colors: list[str],
) -> dict:
    """ROI-vs-random ablation per camera, aggregated over a few frames."""
    import statistics

    by_camera: dict[str, dict] = {}
    rows = []
    for cam in cameras:
        ratios = []
        for index in indices:
            row = roi_ablation(
                dataset, policy, preprocessor, postprocessor, device, index, cam, colors
            )
            if row:
                rows.append(row)
                ratios.append(row["ratio"])
        if ratios:
            by_camera[cam.split(".")[-1]] = {
                "mean_ratio": round(statistics.fmean(ratios), 2),
                "median_ratio": round(statistics.median(ratios), 2),
                "n": len(ratios),
                "grounded": statistics.median(ratios) >= 1.5,
            }

    grounded = [c for c, v in by_camera.items() if v["grounded"]]
    weak = [c for c, v in by_camera.items() if not v["grounded"]]
    if not by_camera:
        verdict = "没有帧能检测到颜色目标，无法评估"
    elif not weak:
        verdict = "所有相机都用到目标区域：" + "、".join(f"{c} {by_camera[c]['median_ratio']}×" for c in grounded)
    elif not grounded:
        verdict = "没有相机在用目标区域：遮住球和遮住空桌面，动作变化一样大"
    else:
        verdict = (
            "用到目标区域的相机：" + "、".join(f"{c} {by_camera[c]['median_ratio']}×" for c in grounded)
            + "；没用到的：" + "、".join(f"{c} {by_camera[c]['median_ratio']}×" for c in weak)
        )
    return {"by_camera": by_camera, "rows": rows, "verdict": verdict, "threshold": 1.5}


def main():
    parser = argparse.ArgumentParser(description="Policy saliency overlay")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--frames", default="0", help="Comma-separated dataset frame indices")
    parser.add_argument("--camera", default=None, help="Comma-separated camera keys (default: all)")
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--colors", default="green,orange")
    parser.add_argument("--out", default="/tmp/lelab_saliency")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-batch", type=int, default=6)
    parser.add_argument("--fill", default="black", choices=["black", "mean"], help="mean blends the tile into the scene")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = run(
        checkpoint_path=args.checkpoint,
        dataset_repo_id=args.dataset,
        dataset_root=args.dataset_root,
        frames=[int(x) for x in args.frames.split(",") if x.strip()],
        cameras=[x.strip() for x in args.camera.split(",")] if args.camera else None,
        grid=args.grid,
        colors=[x.strip() for x in args.colors.split(",") if x.strip()],
        out_dir=args.out,
        device=args.device,
        tile_batch=args.tile_batch,
        fill=args.fill,
    )
    print(f"RESULT_JSON {json.dumps(result, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
