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

"""Offline policy evaluation — no robot, no simulator.

Two questions this answers, which a training loss cannot:

1. *Open-loop accuracy*: replay real dataset frames through the same
   pre/post-processor pipeline the robot uses, predict an action chunk and
   compare it with the recorded actions in degrees — per joint and per horizon
   step. The horizon curve says how many steps you can safely execute before
   the plan goes stale.

2. *Does the policy look at the camera at all*: re-predict with the images
   blacked out, shifted, swapped with another sample's views, and with the task
   string swapped. If the action barely moves, the policy is ignoring vision and
   no amount of retraining on the same data will make it find the object.

Can be used as a module (``run_validation``) or as a standalone script so the
exact same code can run on a training server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)

# An open-loop chunk step whose mean error exceeds this many degrees is no
# longer worth executing; the robot should re-plan instead.
DRIFT_THRESHOLD_DEG = 10.0
# A perturbation that changes the action by less than this fraction of the
# dataset's own action spread is treated as "the policy did not notice".
INSENSITIVE_RATIO = 0.02


def _joint_names(ds_meta, n: int) -> list[str]:
    feat = (ds_meta.info.get("features") or {}).get("action") or {}
    names = feat.get("names")
    if isinstance(names, list) and len(names) == n:
        return [str(x).replace(".pos", "") for x in names]
    return [f"joint_{i}" for i in range(n)]


def _frame_columns(dataset):
    """Return (episode_index, frame_index) per dataset frame as plain ints.

    ``hf_dataset`` hands back torch scalars when a format is set, and those are
    not hashable, so read through the underlying Arrow table.
    """
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None:
        raise RuntimeError("Unsupported LeRobotDataset: no hf_dataset to index episodes")
    table = getattr(hf, "data", None)
    if table is not None and hasattr(table, "column"):
        return table.column("episode_index").to_pylist(), table.column("frame_index").to_pylist()
    return [int(x) for x in hf["episode_index"]], [int(x) for x in hf["frame_index"]]


def _sample_indices(
    dataset,
    chunk_len: int,
    n: int,
    seed: int,
    episodes: list[int] | None,
    frames_per_episode: int = 0,
) -> list[int]:
    """Deterministically pick frames whose full ground-truth chunk exists.

    With ``frames_per_episode`` the budget ``n`` is spent as evenly-spaced
    frames inside whole episodes instead of frames scattered across the dataset
    — that is what makes a per-episode error table readable.
    """
    ep_col, fr_col = _frame_columns(dataset)
    ep_len: dict[int, int] = {}
    for ep, fr in zip(ep_col, fr_col):
        ep_len[ep] = max(ep_len.get(ep, 0), fr + 1)

    wanted = set(episodes) if episodes else set(ep_len)
    pools: dict[int, list[int]] = {ep: [] for ep in wanted}
    for i, (ep, fr) in enumerate(zip(ep_col, fr_col)):
        if ep in pools and fr + chunk_len < ep_len[ep]:
            pools[ep].append(i)
    eligible = [i for lst in pools.values() for i in lst]
    if not eligible:
        raise RuntimeError(
            f"No frames have a full {chunk_len}-step ground-truth window in episodes {sorted(wanted)[:5]}…"
        )

    if frames_per_episode > 0:
        eps = sorted(ep for ep, lst in pools.items() if lst)
        budget = max(1, n // frames_per_episode)
        if len(eps) > budget:
            stride = len(eps) / budget
            eps = [eps[int(j * stride)] for j in range(budget)]
        rng = random.Random(seed)
        picked = []
        for ep in eps:
            lst = pools[ep]
            picked.extend(sorted(rng.sample(lst, min(frames_per_episode, len(lst)))))
        return sorted(picked)

    n = min(n, len(eligible))
    return sorted(random.Random(seed).sample(eligible, n))


def _collate(items: list[dict]):
    from torch.utils.data import default_collate

    return default_collate(items)


def _image_keys(batch) -> list[str]:
    """Real image tensors — ``*_is_pad`` flags carry the substring "image" too."""
    return [k for k in batch if "image" in k and not k.endswith("_is_pad")]


def _observation_signature(batch):
    """Cheap fingerprint of what the policy sees: joint state + coarse colour grid.

    Used to find near-duplicate observations, i.e. places where the dataset
    itself is ambiguous and no model can be accurate.
    """
    import torch

    parts = [batch["observation.state"].float().cpu().reshape(batch["action"].shape[0], -1)]
    for key in sorted(_image_keys(batch)):
        pooled = torch.nn.functional.adaptive_avg_pool2d(batch[key].float(), (4, 4))
        parts.append(pooled.reshape(pooled.shape[0], -1).cpu())
    return torch.cat(parts, dim=1).numpy()


def _episode_chart(path, joint_names, sample_eps, err_joint, per_episode):
    """Episodes × joints error matrix — outlier episodes show up as bright rows."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [r["episode"] for r in per_episode]
    order = {ep: i for i, ep in enumerate(eps)}
    mat = np.zeros((len(eps), err_joint.shape[1]))
    counts = np.zeros(len(eps))
    for row, ep in enumerate(sample_eps):
        i = order[int(ep)]
        mat[i] += err_joint[row]
        counts[i] += 1
    mat /= np.maximum(counts, 1)[:, None]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.2 * len(eps) + 1)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(eps)), [str(e) for e in eps], fontsize=7)
    ax.set_xticks(range(len(joint_names)), [j.replace("_", "\n") for j in joint_names], fontsize=7)
    ax.set_ylabel("episode (worst first)")
    fig.colorbar(im, ax=ax, label="MAE (deg)", fraction=0.046)
    fig.tight_layout()
    fig.savefig(path / "episode_heatmap.png", dpi=110)
    plt.close(fig)
    return "episode_heatmap.png"


def _predict(policy, preprocessor, postprocessor, batch, device, want_loss: bool):
    """Run one inference-path forward: normalize → chunk → unnormalize."""
    import torch

    gt = batch["action"]
    autocast = torch.autocast("cuda") if device.type == "cuda" and policy.config.use_amp else nullcontext()
    with torch.inference_mode(), autocast:
        processed = preprocessor(dict(batch))
        obs = {k: v for k, v in processed.items() if k != "action"}
        t0 = time.perf_counter()
        pred = policy.predict_action_chunk(obs)
        elapsed = time.perf_counter() - t0
        pred = postprocessor(pred.to(device))
        loss = None
        if want_loss:
            out = policy(processed)
            loss = out["loss"] if isinstance(out, dict) else out[0]

    pred = pred.float().cpu()
    if pred.shape[-1] > gt.shape[-1]:
        pred = pred[..., : gt.shape[-1]]
    horizon = min(pred.shape[1], gt.shape[1])
    return pred[:, :horizon], gt[:, :horizon].float(), (None if loss is None else float(loss)), elapsed


def _perturb(batch, kind: str, other: dict | None, alt_task: str | None):
    """Return a copy of ``batch`` with one input channel degraded."""
    import torch

    out = dict(batch)
    image_keys = _image_keys(out)
    if kind == "black":
        for k in image_keys:
            out[k] = torch.zeros_like(out[k])
    elif kind in ("shift_right", "shift_left"):
        shift = 40 if kind == "shift_right" else -40
        for k in image_keys:
            out[k] = torch.roll(out[k], shifts=shift, dims=-1).contiguous()
    elif kind == "swap_views":
        if other is None:
            return None
        for k in image_keys:
            out[k] = other[k]
    elif kind == "task_swap":
        if not alt_task:
            return None
        out["task"] = [alt_task] * len(out["task"])
    else:
        raise ValueError(f"unknown perturbation {kind!r}")
    return out


def _charts(path, joint_names, per_joint_mae, per_joint_naive, horizon_mae, drift_step, overlay, probe_rows):
    """Write the four diagnostic PNGs. Returns {key: Path}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6.5, 3))
    x = range(len(joint_names))
    ax.bar([i - 0.2 for i in x], per_joint_mae, width=0.4, label="model")
    ax.bar([i + 0.2 for i in x], per_joint_naive, width=0.4, label="predict-the-mean baseline")
    ax.set_xticks(list(x), joint_names, rotation=30, ha="right")
    ax.set_ylabel("MAE (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path / "per_joint_mae.png", dpi=110)
    plt.close(fig)
    out["per_joint"] = "per_joint_mae.png"

    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(range(len(horizon_mae)), horizon_mae, marker="o", ms=3)
    ax.axhline(DRIFT_THRESHOLD_DEG, color="r", ls="--", lw=1, label=f"drift threshold {DRIFT_THRESHOLD_DEG:.0f}°")
    if drift_step is not None:
        ax.axvline(drift_step, color="g", ls=":", lw=1, label=f"safe horizon ≈ {drift_step} steps")
    ax.set_xlabel("steps into the predicted chunk")
    ax.set_ylabel("mean MAE (deg)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path / "horizon_curve.png", dpi=110)
    plt.close(fig)
    out["horizon"] = "horizon_curve.png"

    n = max(1, len(overlay))
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.1 * n), sharex=True)
    for i, ax in enumerate(axes if n > 1 else [axes]):
        row = overlay[i]
        t = range(len(row["gt"]))
        ax.plot(t, row["gt"], "k", lw=1.6, label="recorded")
        ax.plot(t, row["pred"], "r", lw=1.4, label="predicted")
        ax.set_ylabel("deg", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("steps into the chunk")
    fig.suptitle(f"{joint_names[0] if joint_names else 'joint_0'} — recorded vs predicted", fontsize=10)
    fig.tight_layout()
    fig.savefig(path / "action_overlay.png", dpi=110)
    plt.close(fig)
    out["overlay"] = "action_overlay.png"

    if probe_rows:
        fig, ax = plt.subplots(figsize=(6.5, 3))
        names = [r["name"] for r in probe_rows]
        vals = [r["relative_delta"] for r in probe_rows]
        bars = ax.barh(names, vals, color=["#ef4444" if v < INSENSITIVE_RATIO else "#22c55e" for v in vals])
        ax.axvline(INSENSITIVE_RATIO, color="gray", ls="--", lw=1)
        ax.set_xlabel("action change / action spread")
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
        fig.tight_layout()
        fig.savefig(path / "probe_sensitivity.png", dpi=110)
        plt.close(fig)
        out["probe"] = "probe_sensitivity.png"

    return out


def run_validation(
    checkpoint_path: str,
    dataset_repo_id: str,
    dataset_root: str | None = None,
    num_batches: int = 20,
    device: str = "cuda",
    batch_size: int = 8,
    seed: int = 0,
    episodes: list[int] | None = None,
    holdout_episodes: int = 0,
    frames_per_episode: int = 0,
    probe: bool = True,
    saliency_frames: int = 0,
    aim_frames: int = 0,
    output_dir: str | None = None,
    charts: bool = True,
) -> dict:
    """Evaluate a checkpoint offline.

    Args:
        checkpoint_path: ``pretrained_model/`` directory.
        dataset_repo_id: dataset the policy was trained on.
        dataset_root: optional local dataset root.
        num_batches: kept for the legacy SSH caller — converted to a sample
            count (``num_batches * batch_size`` frames).
        device: torch device.
        batch_size: inference batch size.
        seed: fixes which frames are sampled, so two checkpoints can be compared.
        episodes: restrict evaluation to these episode indices.
        holdout_episodes: exclude the last K episodes (measures generalisation
            instead of memorisation).
        probe: also run the vision/language sensitivity probe.
        saliency_frames: also render occlusion-saliency overlays for this many
            sampled frames (white box = where the action depends on, green box
            = where the ball actually is).
        aim_frames: also invert the demonstrated action/position mapping over
            this many frames to estimate where the policy aims (blue box).
        output_dir: where to write PNG charts; defaults to a temp subdir.
        charts: disable plots (remote runs have no writable dir yet).

    Returns:
        Metric dict; JSON-serialisable.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    # Hub revalidation hangs for minutes on a slow/blocked link even when every
    # weight is already cached locally. Fail fast and fall back to the cache.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")

    ckpt = Path(checkpoint_path)
    if not ckpt.is_dir():
        return {"success": False, "error": f"Checkpoint path not found: {checkpoint_path}"}

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    try:
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        cfg.pretrained_path = str(ckpt)
        cfg.device = str(dev)

        ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root or None)

        # Chunked policies expect [B, chunk, dim] actions; mirror training.
        delta_timestamps = resolve_delta_timestamps(cfg, ds_meta)
        dataset = LeRobotDataset(dataset_repo_id, root=dataset_root or None, delta_timestamps=delta_timestamps)

        policy = make_policy(cfg=cfg, ds_meta=ds_meta)
        policy.eval()

        # Use the checkpoint's SAVED normalizer stats — the distribution the
        # model was actually trained against.
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": str(dev)}},
        )
    except Exception as exc:
        logger.exception("Failed to load policy/dataset for evaluation")
        return {"success": False, "error": f"load failed: {exc}"}

    chunk_len = max(len(v) for v in delta_timestamps.values()) if delta_timestamps else int(cfg.chunk_size)
    total_episodes = int(ds_meta.total_episodes)
    if episodes is None:
        kept = list(range(total_episodes - max(0, holdout_episodes)))
    else:
        kept = list(episodes)
    if not kept:
        return {"success": False, "error": "No episodes left after applying the holdout"}

    n_samples = max(1, num_batches) * batch_size
    try:
        indices = _sample_indices(dataset, chunk_len, n_samples, seed, kept, frames_per_episode)
    except Exception as exc:
        return {"success": False, "error": f"sample selection failed: {exc}"}

    alt_tasks = [str(t) for t in ds_meta.tasks.index]
    joint_names = _joint_names(ds_meta, dataset[indices[0]]["action"].shape[-1])
    ep_col = _frame_columns(dataset)[0]

    results: dict = {
        "success": False,
        "checkpoint_path": checkpoint_path,
        "dataset_repo_id": dataset_repo_id,
        "device": str(dev),
        "seed": seed,
        "batch_size": batch_size,
        "num_samples": len(indices),
        "holdout_episodes": holdout_episodes,
        "episodes_evaluated": sorted({int(ep_col[i]) for i in indices}),
        "chunk_size": chunk_len,
        "fps": ds_meta.fps,
        "joint_names": joint_names,
        "batches_skipped": 0,
    }

    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    gt_all, pred_all, losses, times = [], [], [], []
    overlay_rows: list[dict] = []
    base_preds: dict[int, torch.Tensor] = {}
    sample_eps: list[int] = []
    sigs: list = []

    for batch_indices in batches:
        try:
            batch = _collate([dataset[i] for i in batch_indices])
        except Exception as exc:
            # Video files can hold fewer frames than the parquet rows claim.
            logger.warning("Skipping batch due to load error: %s", exc)
            results["batches_skipped"] = results.get("batches_skipped", 0) + 1
            continue
        try:
            pred, gt, loss, elapsed = _predict(policy, preprocessor, postprocessor, batch, dev, want_loss=True)
        except Exception as exc:
            logger.warning("Skipping batch due to inference error: %s", exc)
            results["batches_skipped"] = results.get("batches_skipped", 0) + 1
            continue

        gt_all.append(gt)
        pred_all.append(pred)
        sample_eps.extend(int(ep_col[i]) for i in batch_indices)
        sigs.append(_observation_signature(batch))
        if loss is not None:
            losses.append(loss)
        times.append(elapsed * 1000)
        if len(overlay_rows) < 3:
            overlay_rows.append({"gt": gt[0, :, 0].tolist(), "pred": pred[0, :, 0].tolist()})
        for row, idx in enumerate(batch_indices):
            base_preds[idx] = pred[row]

    if not gt_all:
        return {"success": False, "error": "No batches could be evaluated", "num_samples": len(indices)}

    import torch

    gt_cat = torch.cat(gt_all, dim=0)
    pred_cat = torch.cat(pred_all, dim=0)
    err = (pred_cat - gt_cat).abs()

    per_joint_mae = err.mean(dim=(0, 1)).tolist()
    per_joint_rmse = ((pred_cat - gt_cat) ** 2).mean(dim=(0, 1)).sqrt().tolist()
    action_std = gt_cat.std(dim=(0, 1)).tolist()
    naive_mae = (gt_cat - gt_cat.mean(dim=(0, 1), keepdim=True)).abs().mean(dim=(0, 1)).tolist()
    horizon_mae = err.mean(dim=(0, 2)).tolist()

    drift_step = next((i for i, v in enumerate(horizon_mae) if v > DRIFT_THRESHOLD_DEG), None)
    safe_steps = drift_step if drift_step is not None else len(horizon_mae)
    relative = [m / n if n > 1e-6 else None for m, n in zip(per_joint_mae, naive_mae)]
    # <1 means the model beats "always predict the dataset's mean action".
    mean_relative = sum(r for r in relative if r is not None) / max(1, len([r for r in relative if r is not None]))

    results.update(
        {
            "success": True,
            "batches_evaluated": len(gt_all),
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "avg_inference_time_ms": sum(times) / len(times) if times else None,
            "action_mae": err.mean().item(),
            "action_mse": ((pred_cat - gt_cat) ** 2).mean().item(),
            "per_joint_mae_deg": per_joint_mae,
            "per_joint_rmse_deg": per_joint_rmse,
            "per_joint_action_std_deg": action_std,
            "per_joint_naive_mae_deg": naive_mae,
            "per_joint_mae_vs_naive": relative,
            "mean_mae_vs_naive": mean_relative,
            "horizon_mae_deg": horizon_mae,
            "drift_horizon_steps": drift_step,
            "recommended_n_action_steps": max(1, min(safe_steps, len(horizon_mae))),
            "recommended_replan_hz": round(ds_meta.fps / max(1, safe_steps), 2),
        }
    )

    # --- per-episode agreement: which episodes are the outliers, not just the average
    import numpy as np

    err_joint = err.mean(dim=1).numpy()
    eps_arr = np.array(sample_eps[: err.shape[0]])
    per_episode = []
    for ep in sorted(set(eps_arr.tolist())):
        m = eps_arr == ep
        per_episode.append(
            {
                "episode": int(ep),
                "n": int(m.sum()),
                "mae_deg": round(float(err_joint[m].mean()), 2),
                "worst_joint": joint_names[int(err_joint[m].mean(axis=0).argmax())],
            }
        )
    per_episode.sort(key=lambda r: -r["mae_deg"])
    results["per_episode"] = per_episode
    results["worst_episodes"] = [r["episode"] for r in per_episode[:5]]

    # --- data consistency: near-identical observations whose recorded actions
    # disagree put a hard floor under any model's error.
    sig = np.concatenate(sigs[: err.shape[0]])
    z = sig / (sig.std(axis=0) + 1e-6)
    dist = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    dist[np.eye(len(dist), dtype=bool) | (eps_arr[:, None] == eps_arr[None, :])] = np.inf
    nearest = dist.argmin(axis=1)
    nearest_d = dist[np.arange(len(dist)), nearest]
    finite = np.isfinite(nearest_d)
    conflict = None
    if finite.sum() >= 4:
        cutoff = float(np.percentile(nearest_d[finite], 25))
        pairs = nearest_d <= cutoff
        if pairs.any():
            gt_np = gt_cat.numpy()
            conflict = float(np.abs(gt_np - gt_np[nearest]).mean(axis=(1, 2))[pairs].mean())
    results["observation_conflict_deg"] = round(conflict, 2) if conflict is not None else None
    # Half the disagreement is roughly the best any policy can do on this data.
    results["error_floor_deg"] = round(conflict / 2, 2) if conflict is not None else None

    if probe:
        probe_rows = []
        signed: dict[str, list] = {}
        for kind in ("black", "shift_right", "shift_left", "swap_views", "task_swap"):
            other_idx = None
            deltas = []
            signed_rows = []
            for batch_indices in batches[: max(1, len(batches) // 2)]:
                try:
                    batch = _collate([dataset[i] for i in batch_indices])
                    other = None
                    if kind == "swap_views":
                        if other_idx is None:
                            other_idx = _sample_indices(dataset, chunk_len, batch_size, seed + 7777, kept)
                        other = _collate([dataset[i] for i in other_idx[: len(batch_indices)]])
                    alt_task = next((t for t in alt_tasks if t != batch["task"][0]), None)
                    pert = _perturb(batch, kind, other, alt_task)
                    if pert is None:
                        continue
                    pred, gt, _, _ = _predict(policy, preprocessor, postprocessor, pert, dev, want_loss=False)
                    for row, idx in enumerate(batch_indices):
                        base = base_preds.get(idx)
                        if base is None or base.shape != pred[row].shape:
                            continue
                        deltas.append((pred[row] - base).abs().mean(dim=-1))
                        signed_rows.append((pred[row] - base).mean(dim=0))
                except Exception as exc:
                    logger.warning("Probe %s skipped a batch: %s", kind, exc)
                    continue
            if not deltas:
                continue
            signed[kind] = torch.stack(signed_rows).mean(dim=0)
            d = torch.stack(deltas).mean().item()
            spread = float(torch.tensor(action_std).mean().item()) or 1.0
            probe_rows.append(
                {
                    "name": kind,
                    "delta_deg": round(d, 4),
                    "relative_delta": round(d / spread, 4),
                    "label": {
                        "black": "图像全黑",
                        "shift_right": "图像右移 40px",
                        "shift_left": "图像左移 40px",
                        "swap_views": "换成别的场景视角",
                        "task_swap": "换任务指令",
                    }[kind],
                }
            )

        sign_agreement = None
        if "shift_right" in signed and "shift_left" in signed:
            vr, vl = signed["shift_right"], signed["shift_left"]
            live = (vr.abs() > 0.5) & (vl.abs() > 0.5)
            if live.any():
                sign_agreement = round(float((torch.sign(vr[live]) == -torch.sign(vl[live])).float().mean()), 3)

        vision = [r["relative_delta"] for r in probe_rows if not r["name"].startswith("shift") and r["name"] != "task_swap"]
        lang = next((r["relative_delta"] for r in probe_rows if r["name"] == "task_swap"), None)
        weakest = min(vision) if vision else None
        if weakest is None:
            verdict = "无法判定（探针未产生结果）"
        elif weakest < INSENSITIVE_RATIO:
            verdict = "模型几乎不看摄像头：扰动图像后动作变化小于动作本身散布的 2%"
        elif weakest < 0.10:
            verdict = "模型对视觉只有弱响应：换画面后动作只变了一点，容易在位置变化时抓空"
        else:
            verdict = "模型确实在用视觉：画面改变会明显改变动作"
        if sign_agreement is not None and sign_agreement < 0.5:
            verdict = (
                f"画面有反应但不定位目标：左右平移时动作方向只有 {sign_agreement * 100:.0f}% 相反"
                "（>60% 才算它算出了球在哪）"
            )
        results["probe"] = {
            "items": probe_rows,
            "shift_sign_agreement": sign_agreement,
            "vision_sensitivity": weakest,
            "language_sensitivity": lang,
            "verdict": verdict,
        }
    else:
        results["probe"] = None

    if charts and output_dir:
        try:
            art = Path(output_dir)
            art.mkdir(parents=True, exist_ok=True)
            results["charts"] = _charts(
                art, joint_names, per_joint_mae, naive_mae, horizon_mae, drift_step, overlay_rows, results["probe"]["items"] if probe else []
            )
            if len(per_episode) >= 3:
                results["charts"]["episodes"] = _episode_chart(art, joint_names, sample_eps, err_joint, per_episode)
            results["chart_dir"] = str(art)
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)

    if saliency_frames and output_dir:
        try:
            from .saliency import attach_saliency

            cam = next((k for k in cfg.input_features if "image" in k), None)
            if cam:
                colours = sorted({c for t in alt_tasks for c in ("green", "orange", "red", "blue") if c in t.lower()})
                loc = attach_saliency(
                    dataset,
                    policy,
                    preprocessor,
                    postprocessor,
                    dev,
                    indices[: int(saliency_frames)],
                    cam,
                    Path(output_dir),
                    colors=colours or None,
                )
                charts = results.setdefault("charts", {})
                for i, row in enumerate(loc["rows"]):
                    charts[f"saliency_{i}"] = row["chart"]
                results["localization"] = loc
            try:
                from .saliency import roi_grounding_summary

                cams = [k for k in cfg.input_features if "image" in k]
                grounding = roi_grounding_summary(
                    dataset,
                    policy,
                    preprocessor,
                    postprocessor,
                    dev,
                    indices[: int(saliency_frames)],
                    cams,
                    sorted({c for t in alt_tasks for c in ("green", "orange", "red", "blue") if c in t.lower()})
                    or ["green", "orange"],
                )
                grounding["rows"] = grounding["rows"][:12]
                results["roi_grounding"] = grounding
            except Exception as exc:
                logger.warning("ROI grounding test failed: %s", exc)
        except Exception as exc:
            logger.warning("Saliency probe failed: %s", exc)
            results["charts_error"] = str(exc)

    if aim_frames and output_dir:
        try:
            from .aim import fit as fit_aim
            from .aim import probe as probe_aim

            cam = next((k for k in cfg.input_features if "image" in k), None)
            colours = sorted({c for t in alt_tasks for c in ("green", "orange", "red", "blue") if c in t.lower()})
            calibration = fit_aim(dataset, cam, colours or ["green", "orange"], horizon=25) if cam else None
            if calibration:

                def _predict_item(item):
                    import torch

                    batch = _collate([item])
                    pred, _, _, _ = _predict(policy, preprocessor, postprocessor, batch, dev, want_loss=False)
                    state = torch.as_tensor(batch["observation.state"]).float().reshape(-1)
                    return pred[0], state[: pred.shape[-1]]

                aim = probe_aim(
                    dataset,
                    _predict_item,
                    indices[: int(aim_frames)],
                    calibration,
                    colours or ["green", "orange"],
                    Path(output_dir),
                )
                charts = results.setdefault("charts", {})
                for i, row in enumerate(aim["rows"]):
                    charts[f"aim_{i}"] = row["chart"]
                aim.pop("rows", None)
                results["aim"] = aim
        except Exception as exc:
            logger.warning("Aim probe failed: %s", exc)

    del policy, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Offline policy evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_model dir")
    parser.add_argument("--dataset", required=True, help="Dataset repo ID")
    parser.add_argument("--dataset-root", default=None, help="Dataset local root")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", default=None, help="Comma-separated episode indices")
    parser.add_argument("--holdout-episodes", type=int, default=0, help="Exclude the last K episodes")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=0,
        help="Sample N evenly-spaced frames per episode instead of scattered frames",
    )
    parser.add_argument("--no-probe", action="store_true", help="Skip the sensitivity probe")
    parser.add_argument("--saliency-frames", type=int, default=0, help="Render this many occlusion overlays")
    parser.add_argument("--aim-frames", type=int, default=0, help="Invert this many frames into an aim position")
    parser.add_argument("--output-dir", default=None, help="Where to write PNG charts")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    results = run_validation(
        checkpoint_path=args.checkpoint,
        dataset_repo_id=args.dataset,
        dataset_root=args.dataset_root,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        seed=args.seed,
        episodes=[int(x) for x in args.episodes.split(",")] if args.episodes else None,
        holdout_episodes=args.holdout_episodes,
        frames_per_episode=args.frames_per_episode,
        probe=not args.no_probe,
        saliency_frames=args.saliency_frames,
        aim_frames=args.aim_frames,
        output_dir=args.output_dir,
        charts=not args.no_charts,
        device=args.device,
    )
    print(json.dumps(results, indent=2))
    # Single-line marker so a remote (SSH) runner can grep the result out of the
    # streamed log without parsing the pretty-printed JSON block above.
    print(f"RESULT_JSON {json.dumps(results)}", flush=True)
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
