"""Closed-loop scoring of rollout episodes: progress per frame -> success rate.

LeLab's existing model evaluation is open-loop (predicted action vs the recorded
action). That number cannot tell whether the policy actually finishes the task —
a policy can score 1° MAE and still fail on the robot every time. The capture
modes on the inference page record the policy's *own* episodes; scoring them
with a progress model turns those into the number that matters, and the same
per-frame progress is what RA-BC training reads as sample weights
(`sarm_progress.parquet` next to the dataset).

The progress model is a 4B VLM (Robometer / SARM heads on Qwen3-VL-4B), so this
runs wherever there is VRAM for it — the 4090 pod, not the laptop GPU. When the
local box can't host it, `handle_rollout_score` hands back the exact command to
run there instead of pretending it worked.
"""

from __future__ import annotations

import contextlib
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROGRESS_FILENAME = "sarm_progress.parquet"
# The file RA-BC auto-detects is the one above. Each scoring backend keeps its
# own parquet next to it so two scorings can be compared instead of overwriting
# each other; "promote" copies one of them over the authoritative file.
PROGRESS_BY_BACKEND = {
    "robometer": "progress_robometer.parquet",
    "sarm": "progress_sarm.parquet",
}
DEFAULT_BACKEND = "robometer"
# A rollout counts as successful when the task progress at the end of the
# episode reaches this. Robometer's own success head is thresholded at 0.5;
# progress is the more informative signal for a pick-and-place style task.
DEFAULT_SUCCESS_PROGRESS = 0.9
# Frames are ~30MB/s of video per camera; scoring every frame is wasteful when
# the progress curve is smooth. stride=5 interpolates the rest.
DEFAULT_STRIDE = 5

_score_state: dict[str, Any] = {"running": False}
# The scoring job outlives the page (and, on a pod, this server): the record on
# disk is what lets the card show "still running, 12/29" after LeLab restarts.
SCORE_JOB_FILENAME = "rollout_score_job.json"
_score_lock = threading.Lock()
# The pod-side job runs on another machine, so its identity (which dataset, which
# backend) is tracked in memory too, for the finish-time reconcile.
_remote_score_state: dict[str, Any] = {}


def _score_job_path() -> Path:
    return _cache_root() / SCORE_JOB_FILENAME


def save_score_job(**fields: Any) -> dict[str, Any]:
    import json

    record = {"started_at": time.time(), **fields}
    try:
        tmp = _score_job_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        tmp.replace(_score_job_path())
    except OSError as exc:
        logger.info("Score job record not written: %s", exc)
    return record


def load_score_job() -> dict[str, Any]:
    import json

    try:
        record = json.loads(_score_job_path().read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def clear_score_job() -> None:
    with contextlib.suppress(OSError):
        _score_job_path().unlink()


def mark_score_job_pulled() -> None:
    """Remember that the pod's parquet is already here.

    Without this marker every card mount re-enters the finish branch and pulls
    the remote file again, silently overwriting a reconciled or promoted copy.
    """
    import json

    record = load_score_job()
    if not record:
        return
    record["pulled_at"] = time.time()
    try:
        tmp = _score_job_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        tmp.replace(_score_job_path())
    except OSError:
        pass


def finish_score_job(*, failed: bool) -> None:
    """Mark the job done but keep its record — the log is what explains a crash."""
    import json

    record = load_score_job()
    if not record:
        return
    record["finished_at"] = time.time()
    record["failed"] = bool(failed)
    try:
        tmp = _score_job_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        tmp.replace(_score_job_path())
    except OSError:
        pass


class RolloutScoreRequest(BaseModel):
    dataset_repo_id: str
    # robometer = pretrained zero-shot scorer; sarm = one you trained yourself
    # on this task (see start_sarm_training_remote). reward_model_path defaults
    # per backend.
    backend: str = DEFAULT_BACKEND
    reward_model_path: str = ""
    # Which of the dataset's cameras plays the role of the reward model's
    # third-person view. Robometer's config ships with `observation.images.top`
    # (ALOHA convention); SO-101 rigs here name the overhead camera `up`.
    # SARM has no such flag — its camera is baked into the checkpoint's config.
    image_key: str = "observation.images.up"
    device: str = "cuda"
    # sarm only: compute progress every Nth frame and interpolate the rest.
    stride: int = DEFAULT_STRIDE
    # robometer only: how many frames each per-frame prediction window looks at,
    # and how many windows go through the VLM in one forward. Together they set
    # the cost — robometer has no stride, it scores every frame. 32x16 put 512
    # images through the vision tower per step and OOM'd a 24 GB 4090; 8x8 needs
    # roughly 6 GB and is the safe default for one consumer card.
    num_subsampled_frames: int = 8
    batch_size: int = 8
    # robometer/topreward accept an episode subset; sarm does not.
    episodes: list[int] | None = None
    # sarm only: which head of the model to read. Robometer writes a single
    # `progress_sparse` column.
    head_mode: str = "sparse"
    # Pod-side scoring pushes the parquet back to the dataset repo; the local
    # copy then pulls it down. Only meaningful together with push_to_hub.
    push_to_hub: bool = False


def _validate_score_request(request: RolloutScoreRequest) -> str | None:
    """The three scorer scripts have different CLI surfaces; catch mismatches here."""
    if request.backend == "robometer" and request.head_mode not in ("sparse", ""):
        return "Robometer 只有一个进度头（progress_sparse），head_mode 选 dense 会读不到列。"
    if request.backend == "sarm" and request.episodes:
        return "SARM 打分不支持按 episode 子集跑（上游脚本没有 --episodes），请整集打分。"
    return None


def _score_flags(request: RolloutScoreRequest) -> list[str]:
    """Backend-specific flags, built from what that upstream script accepts."""
    if request.backend == "robometer":
        flags = [
            f"--image-key={request.image_key}",
            f"--num-subsampled-frames={request.num_subsampled_frames}",
            f"--batch-size={request.batch_size}",
        ]
        if request.episodes:
            flags += ["--episodes"] + [str(e) for e in request.episodes]
        return flags
    return [f"--stride={request.stride}", f"--head-mode={request.head_mode}"]


def _cache_root() -> Path:
    import os

    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _dataset_root(repo_id: str) -> Path:
    return _cache_root() / repo_id


def _episode_lengths(root: Path) -> dict[int, int]:
    """episode_index -> frame count, from the dataset's own metadata."""
    import pandas as pd

    out: dict[int, int] = {}
    for path in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["episode_index", "length"])
        except Exception as exc:
            logger.warning("Failed reading episodes meta %s: %s", path, exc)
            continue
        for ep, length in zip(df["episode_index"], df["length"]):
            out[int(ep)] = int(length)
    return out


def _dataset_info(root: Path) -> dict[str, Any]:
    import json

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


def _frame_index_map(root: Path) -> Any:
    """The dataset's own (episode_index, frame_index) -> global index table."""
    import pandas as pd

    parts = []
    for path in sorted(root.glob("data/chunk-*/file-*.parquet")):
        try:
            parts.append(pd.read_parquet(path, columns=["episode_index", "frame_index", "index"]))
        except Exception as exc:
            logger.warning("Failed reading data shard %s: %s", path, exc)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def check_progress_alignment(repo_id: str, filename: str = PROGRESS_FILENAME) -> dict[str, Any]:
    """Does this progress file actually describe this dataset?

    The scorer keys every row on the *global* frame index, and merging datasets
    renumbers frames — so a parquet computed before a merge still loads fine and
    silently assigns each weight to the wrong frame. Compare the file against the
    dataset's own index map: that catches it before training does.
    """
    import pandas as pd

    root = _dataset_root(repo_id)
    path = root / filename
    if not path.is_file():
        return {"ok": False, "reason": "no_progress", "message": f"还没有 {filename}"}
    try:
        table = pd.read_parquet(path, columns=["index", "episode_index", "frame_index"])
    except Exception as exc:
        return {"ok": False, "reason": "unreadable", "message": f"进度文件读不了：{exc}"}

    truth = _frame_index_map(root)
    if truth is None or truth.empty:
        return {"ok": True, "reason": "unchecked", "message": "读不到数据集分片，跳过对齐校验"}

    joined = table.merge(truth, on=["episode_index", "frame_index"], how="left", suffixes=("_p", "_d"))
    unmatched = int(joined["index_d"].isna().sum())
    shifted = int((joined["index_p"] != joined["index_d"]).sum())
    lengths = _episode_lengths(root)
    covered = set(int(e) for e in table["episode_index"].unique())
    missing_eps = sorted(set(lengths) - covered)

    ok = unmatched == 0 and shifted == 0
    if ok and missing_eps:
        return {
            "ok": True,
            "reason": "partial",
            "rows": len(table),
            "frames": int(len(truth)),
            "episodes_without_progress": len(missing_eps),
            "message": f"{len(missing_eps)} 集没有进度记录，训练时这些帧按默认权重 1.0 处理",
        }
    if ok:
        return {"ok": True, "reason": "aligned", "rows": len(table), "frames": int(len(truth))}
    return {
        "ok": False,
        "reason": "misaligned",
        "rows": len(table),
        "frames": int(len(truth)),
        "unmatched_rows": unmatched,
        "shifted_rows": shifted,
        "episodes_without_progress": len(missing_eps),
        "message": (
            f"进度文件和这个数据集对不上：{shifted + unmatched} 行找不到对应的帧。"
            "多半是这份进度是在合并/删集之前算的——合并会重排全局帧号，"
            "请在这个数据集上重新打分。"
        ),
    }


def _progress_table(root: Path) -> Any:
    """The scoring output as a DataFrame, or None when it hasn't been produced."""
    path = root / PROGRESS_FILENAME
    if not path.is_file():
        return None
    import pandas as pd

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        logger.warning("Unreadable progress file %s: %s", path, exc)
        return None


def list_capture_datasets() -> list[dict[str, Any]]:
    """Local datasets that can be scored, with their scoring status.

    Rollout captures (`rollout_*`) are the primary target, but the merged
    training set is not named that way — and progress *must* be computed on the
    merged dataset, because merging renumbers frames. So every local dataset is
    listed; rollout captures sort first.
    """
    root = _cache_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for org_dir in sorted(root.iterdir()):
        if not org_dir.is_dir():
            continue
        for ds_dir in sorted(org_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name.startswith("__bak_") or "__bak_" in ds_dir.name:
                continue
            if not (ds_dir / "meta" / "info.json").is_file():
                continue
            lengths = _episode_lengths(ds_dir)
            info = _dataset_info(ds_dir)
            out.append(
                {
                    "repo_id": f"{org_dir.name}/{ds_dir.name}",
                    "episodes": len(lengths),
                    "frames": sum(lengths.values()),
                    "fps": info.get("fps"),
                    "scored": (ds_dir / PROGRESS_FILENAME).is_file(),
                    "is_rollout": ds_dir.name.startswith("rollout_"),
                    "has_intervention": "intervention" in (info.get("features") or {}),
                    "backends": [
                        b for b, f in PROGRESS_BY_BACKEND.items() if (ds_dir / f).is_file()
                    ],
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(ds_dir.stat().st_mtime)
                    ),
                }
            )
    # Rollout captures first, newest first inside each group.
    out.sort(key=lambda d: d["modified"] or "", reverse=True)
    out.sort(key=lambda d: not d["is_rollout"])
    return out



def handle_rollout_summary(
    repo_id: str,
    success_progress: float = DEFAULT_SUCCESS_PROGRESS,
    head_mode: str = "sparse",
) -> dict[str, Any]:
    """Per-episode progress summary and the closed-loop success rate."""
    root = _dataset_root(repo_id)
    if not (root / "meta" / "info.json").is_file():
        return {"available": False, "error": f"dataset not found locally: {repo_id}"}

    lengths = _episode_lengths(root)
    table = _progress_table(root)
    if table is None:
        return {
            "available": False,
            "no_progress": True,
            "repo_id": repo_id,
            "episodes": len(lengths),
            "frames": sum(lengths.values()),
            "hint": f"还没有 {PROGRESS_FILENAME}，先跑一次打分。",
        }

    alignment = check_progress_alignment(repo_id)
    if not alignment.get("ok"):
        return {
            "available": False,
            "misaligned": True,
            "repo_id": repo_id,
            "episodes": len(lengths),
            "frames": sum(lengths.values()),
            "error": alignment.get("message"),
        }

    wanted = f"progress_{head_mode}"
    column = wanted if wanted in table.columns else (
        "progress_sparse" if "progress_sparse" in table.columns else "progress"
    )
    if column not in table.columns:
        return {"available": False, "error": f"progress 文件里没有进度列：{list(table.columns)}"}

    import numpy as np

    grouped = table.groupby("episode_index")[column]
    episodes: list[dict[str, Any]] = []
    successes = 0
    for ep, series in grouped:
        values = series.dropna()
        if values.empty:
            continue
        first = float(values.iloc[0])
        last = float(values.iloc[-1])
        peak = float(values.max())
        # Stagnation point, as a frame number inside the episode: the last frame
        # at which progress was still climbing. Everything after it is the
        # stretch worth re-teaching, so it doubles as the clip to jump to.
        gains = values.diff().to_numpy()
        moving = np.flatnonzero(gains > 0.01)
        total = len(values)
        pos = int(moving[-1]) if len(moving) else 0
        frames = lengths.get(int(ep))
        stalled_at = int(round(pos / max(1, total - 1) * (frames - 1))) if frames else None
        ok = last >= success_progress
        successes += int(ok)
        episodes.append(
            {
                "episode_index": int(ep),
                "frames": frames,
                "progress_start": round(first, 3),
                "progress_end": round(last, 3),
                "progress_peak": round(peak, 3),
                "stalled_at": stalled_at,
                "success": bool(ok),
            }
        )
    episodes.sort(key=lambda e: e["episode_index"])
    scored = len(episodes)
    has_intervention = "intervention" in (_dataset_info(root).get("features") or {})
    return {
        "available": True,
        "repo_id": repo_id,
        "column": column,
        "episodes": scored,
        "successes": successes,
        "success_rate": round(successes / scored, 4) if scored else None,
        "threshold": success_progress,
        "has_intervention": has_intervention,
        "alignment": alignment,
        "details": episodes,
    }


def handle_rollout_episode_series(
    repo_id: str, episode_index: int, max_points: int = 400, head_mode: str = "sparse"
) -> dict[str, Any]:
    """One episode's progress curve, downsampled, with the human-driven windows.

    The number table says *whether* the scorer thought an episode succeeded; only
    the shape says whether it can be trusted on this rig — a curve that ignores
    the moment the ball leaves the gripper is the failure mode that makes a
    zero-shot 4B scorer useless. Correction windows come from the dataset's own
    `intervention` column so the two can be read against each other.
    """
    import numpy as np
    import pandas as pd

    root = _dataset_root(repo_id)
    table = _progress_table(root)
    if table is None:
        return {"available": False, "hint": f"还没有 {PROGRESS_FILENAME}，先跑一次打分。"}
    wanted = f"progress_{head_mode}"
    column = wanted if wanted in table.columns else (
        "progress_sparse" if "progress_sparse" in table.columns else None
    )
    if column is None:
        return {"available": False, "error": f"进度文件没有可用进度列：{list(table.columns)}"}

    ep = table[table["episode_index"].astype(int) == int(episode_index)]
    if ep.empty:
        return {"available": False, "error": f"进度文件里没有 episode {episode_index}"}
    ep = ep.sort_values("frame_index")
    frames = ep["frame_index"].to_numpy(dtype=int)
    values = ep[column].to_numpy(dtype=float)

    fps = _dataset_info(root).get("fps") or 30
    if len(frames) > max_points:
        step = int(np.ceil(len(frames) / max_points))
        frames, values = frames[::step], values[::step]

    windows = _intervention_windows(root, int(episode_index))
    return {
        "available": True,
        "repo_id": repo_id,
        "episode_index": int(episode_index),
        "column": column,
        "fps": fps,
        "length": int(ep["frame_index"].max()) + 1,
        "points": [[int(f), None if np.isnan(v) else round(float(v), 4)] for f, v in zip(frames, values)],
        "correction_windows": windows,
    }


def _intervention_windows(root: Path, episode_index: int) -> list[dict[str, Any]]:
    """[start_frame, end_frame] runs where the dataset says a human was driving."""
    import numpy as np
    import pandas as pd

    if "intervention" not in (_dataset_info(root).get("features") or {}):
        return []
    runs: list[dict[str, Any]] = []
    start: int | None = None
    last: int | None = None
    for path in sorted(root.glob("data/chunk-*/file-*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["episode_index", "frame_index", "intervention"])
        except Exception as exc:
            logger.warning("Failed reading intervention shard %s: %s", path, exc)
            continue
        df = df[df["episode_index"].astype(int) == int(episode_index)]
        if df.empty:
            continue
        flags = df["intervention"].map(lambda v: bool(np.asarray(v).reshape(-1)[0])).to_numpy()
        for frame, flag in zip(df["frame_index"].to_numpy(dtype=int), flags):
            if flag and start is None:
                start, last = int(frame), int(frame)
            elif flag:
                last = int(frame)
            elif start is not None:
                runs.append({"start": start, "end": int(last or start)})
                start, last = None, None
    if start is not None:
        runs.append({"start": start, "end": int(last or start)})
    return runs


ROUND_STATS_FILENAME = "rollout_rounds.json"
_EMPTY_TAKE_STATS = {
    "takes": 0,
    "corrected_takes": 0,
    "failures": 0,
    "episodes_saved": 0,
    "takeover_rate": None,
    "failure_rate": None,
}


def read_take_stats(repo_id: str) -> dict[str, Any]:
    """Running take counters for a capture dataset, or an empty record."""
    import json

    stats = dict(_EMPTY_TAKE_STATS)
    try:
        stored = json.loads((_dataset_root(repo_id) / ROUND_STATS_FILENAME).read_text())
    except (OSError, ValueError):
        return stats
    if isinstance(stored, dict):
        stats.update(stored)
    return stats


def add_take_stats(
    repo_id: str,
    *,
    takes: int = 0,
    corrected: int = 0,
    failures: int = 0,
    policy_ref: str = "",
) -> dict[str, Any]:
    """Accumulate one poll's worth of take counters into the dataset's own file.

    Corrections-only capture saves an episode only when a human takes over, so
    that dataset's scored success rate describes the operator. `takeover_rate` =
    corrected / takes is the number that describes the policy, and only the
    rollout subprocess can see which takes were intervened in.
    """
    import json

    if not (takes or corrected or failures):
        return read_take_stats(repo_id)

    stats = read_take_stats(repo_id)
    stats["takes"] = int(stats["takes"]) + int(takes)
    stats["corrected_takes"] = int(stats["corrected_takes"]) + int(corrected)
    stats["failures"] = int(stats["failures"]) + int(failures)
    stats["takeover_rate"] = round(stats["corrected_takes"] / stats["takes"], 4) if stats["takes"] else None
    stats["failure_rate"] = round(stats["failures"] / stats["takes"], 4) if stats["takes"] else None
    if policy_ref:
        stats["policy_ref"] = policy_ref
    stats["updated"] = time.strftime("%Y-%m-%d %H:%M")

    root = _dataset_root(repo_id)
    if not root.is_dir():
        return stats
    try:
        tmp = root / f".{ROUND_STATS_FILENAME}.tmp"
        tmp.write_text(json.dumps(stats, ensure_ascii=False, indent=1))
        tmp.replace(root / ROUND_STATS_FILENAME)
    except OSError as exc:
        logger.warning("Failed to persist take stats for %s: %s", repo_id, exc)
    return stats


def apply_intervention_exemption(repo_id: str, kappa: float = 0.01) -> dict[str, Any]:
    """Make correction windows pass RA-BC's progress gate.

    RA-BC zeroes any frame whose progress does not rise over the policy's chunk
    horizon. A recovery *starts* by backing off — so the behaviour we most want
    to teach is exactly what the gate suppresses. The dataset already knows which
    frames were human-driven (`intervention`), so rewrite progress across each
    correction window as a monotone ramp between the values bracketing it: the
    frames keep their real observations and actions, only their claimed progress
    becomes what the correction was for.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = _dataset_root(repo_id)
    progress_path = root / PROGRESS_FILENAME
    if not progress_path.is_file():
        return {"success": False, "status_code": 409, "message": "先打分，再谈纠正段豁免"}
    alignment = check_progress_alignment(repo_id)
    if not alignment.get("ok"):
        return {"success": False, "status_code": 409, "message": alignment.get("message")}

    frames = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not frames:
        return {"success": False, "status_code": 409, "message": "找不到 data 分片，读不到 intervention 列"}
    marks = []
    for path in frames:
        try:
            df = pd.read_parquet(path, columns=["frame_index", "episode_index", "intervention"])
        except Exception as exc:
            return {"success": False, "status_code": 409, "message": f"读取 intervention 列失败：{exc}"}
        marks.append(df)
    marks = pd.concat(marks)
    # The feature is declared bool/shape (1,), so parquet hands back a one-element
    # array per row rather than a bare bool.
    marks["is_corrected"] = marks["intervention"].map(
        lambda v: bool(np.asarray(v).reshape(-1)[0])
    )
    column = "progress_sparse"
    table = pd.read_parquet(progress_path)
    if column not in table.columns:
        return {"success": False, "status_code": 400, "message": f"进度文件没有 {column} 列"}

    # Map global frame index -> progress row position, then walk each episode's
    # correction runs and lift their progress to a strictly rising ramp.
    lifted = 0
    for ep, window in marks.groupby("episode_index"):
        corrected = window[window["is_corrected"]]["frame_index"].to_numpy()
        if len(corrected) == 0:
            continue
        rows = table[table["episode_index"] == ep].sort_values("index")
        if rows.empty:
            continue
        pos = rows["index"].to_numpy()
        values = rows[column].to_numpy(dtype=float)
        lookup = {int(f): i for i, f in enumerate(pos)}
        start = min(int(corrected[0]), int(corrected[-1]))
        stop = max(int(corrected[-1]) + 1, start + 1)
        before = next((values[i] for i in range(lookup.get(start, 0) - 1, -1, -1) if not np.isnan(values[i])), 0.0)
        after = next((values[i] for i in range(lookup.get(stop, len(values) - 1), len(values)) if not np.isnan(values[i])), 1.0)
        span = max(1, stop - start)
        # End just below the post-correction value so the ramp joins the real
        # curve instead of jumping; a flat tail would zero the last frames again.
        target = max(after, before + kappa * 2)
        for offset, frame in enumerate(range(start, stop)):
            i = lookup.get(frame)
            if i is None:
                continue
            want = before + (target - before) * (offset + 1) / (span + 1)
            if np.isnan(values[i]) or values[i] < want:
                values[i] = want
                lifted += 1
        table.loc[rows.index, column] = values

    table[column] = table[column].clip(upper=1.0)
    metadata = pq.read_metadata(progress_path).schema.to_arrow_schema().metadata
    out = pa.Table.from_pandas(table, preserve_index=False).replace_schema_metadata(metadata)
    tmp = root / f".{PROGRESS_FILENAME}.tmp"
    pq.write_table(out, tmp)
    tmp.replace(progress_path)
    return {
        "success": True,
        "message": f"已抬升 {lifted} 个纠正帧的进度",
        "frames_lifted": lifted,
        "path": str(progress_path),
    }


def _default_model_path(backend: str) -> str:
    return "lerobot/Robometer-4B" if backend == "robometer" else ""


def _progress_series(root: Path, filename: str) -> Any:
    """(index -> progress) series plus episode mapping for one parquet."""
    import pandas as pd

    df = pd.read_parquet(root / filename)
    column = "progress_sparse" if "progress_sparse" in df.columns else "progress"
    return df[["index", "episode_index", column]].rename(columns={column: "progress"})


def handle_compare_progress(
    repo_id: str, success_progress: float = DEFAULT_SUCCESS_PROGRESS
) -> dict[str, Any]:
    """Agreement between the two scoring backends on the same episodes.

    The two models disagree on purpose in interesting places: Robometer is a
    zero-shot generalist that has never seen this rig, SARM is trained on it.
    Episodes where their success labels split are the ones worth watching in
    the camera before trusting either number."""
    import numpy as np

    root = _dataset_root(repo_id)
    tables: dict[str, Any] = {}
    for backend, filename in PROGRESS_BY_BACKEND.items():
        if (root / filename).is_file():
            tables[backend] = _progress_series(root, filename)
    if len(tables) < 2:
        return {
            "available": False,
            "backends": sorted(tables),
            "hint": "两个后端都打过分才能对照。",
        }

    a, b = tables["robometer"], tables["sarm"]
    common = a["index"].isin(b["index"]) & b["index"].isin(a["index"])
    a = a[a["index"].isin(b["index"])].set_index("index").sort_index()
    b = b[b["index"].isin(a.index)].set_index("index").sort_index()
    x, y = a["progress"].to_numpy(float), b["progress"].to_numpy(float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None

    per_episode = []
    disagreements = []
    for ep in sorted(set(a["episode_index"]) & set(b["episode_index"])):
        pa = a[a["episode_index"] == ep]["progress"].to_numpy(float)
        pb = b[b["episode_index"] == ep]["progress"].to_numpy(float)
        sa, sb = bool(pa[-1] >= success_progress), bool(pb[-1] >= success_progress)
        row = {
            "episode_index": int(ep),
            "robometer_end": round(float(pa[-1]), 3),
            "sarm_end": round(float(pb[-1]), 3),
            "robometer_success": sa,
            "sarm_success": sb,
            "mean_abs_diff": round(float(np.mean(np.abs(pa - pb))), 3),
        }
        per_episode.append(row)
        if sa != sb:
            disagreements.append(row)

    n = len(per_episode)
    agree = n - len(disagreements)
    return {
        "available": True,
        "repo_id": repo_id,
        "frames": int(len(x)),
        "pearson": round(pearson, 4) if pearson is not None else None,
        "mean_abs_diff": round(float(np.mean(np.abs(x - y))), 4),
        "success_rate": {
            "robometer": round(sum(r["robometer_success"] for r in per_episode) / n, 4) if n else None,
            "sarm": round(sum(r["sarm_success"] for r in per_episode) / n, 4) if n else None,
        },
        "agreement": round(agree / n, 4) if n else None,
        "disagreements": disagreements,
    }


def promote_progress(repo_id: str, backend: str) -> dict[str, Any]:
    """Make one backend's parquet the file RA-BC and the summary read."""
    import shutil

    if backend not in PROGRESS_BY_BACKEND:
        return {"success": False, "status_code": 400, "message": f"未知后端 {backend}"}
    root = _dataset_root(repo_id)
    src = root / PROGRESS_BY_BACKEND[backend]
    if not src.is_file():
        return {"success": False, "status_code": 404, "message": f"{backend} 还没打过这个集"}
    alignment = check_progress_alignment(repo_id, PROGRESS_BY_BACKEND[backend])
    if not alignment.get("ok"):
        return {"success": False, "status_code": 409, "message": alignment.get("message")}
    tmp = root / f".{PROGRESS_FILENAME}.tmp"
    shutil.copyfile(src, tmp)
    tmp.replace(root / PROGRESS_FILENAME)
    note = alignment.get("message") if alignment.get("reason") == "partial" else ""
    return {"success": True, "message": f"已把 {backend} 的进度设为训练/评估使用的版本。{note}".strip()}


def build_score_command(
    request: RolloutScoreRequest,
    *,
    python: str | None = None,
    output_file: Path | str | None = None,
) -> list[str]:
    """The lerobot progress-scoring invocation for one dataset.

    Each backend writes its own parquet so two scorings coexist for comparison;
    the file RA-BC reads is only touched by promote_progress(). `output_file`
    overrides the destination — the pod path has to be somewhere that exists
    there, and the result comes back over the Hub rather than the local root.
    """
    backend = request.backend if request.backend in PROGRESS_BY_BACKEND else DEFAULT_BACKEND
    model_path = request.reward_model_path or _default_model_path(backend)
    out_file = Path(output_file) if output_file else _dataset_root(request.dataset_repo_id) / PROGRESS_BY_BACKEND[backend]
    cmd = [
        python or sys.executable,
        "-m",
        f"lerobot.rewards.{backend}.compute_rabc_weights",
        f"--dataset-repo-id={request.dataset_repo_id}",
        f"--reward-model-path={model_path}",
        f"--device={request.device}",
        f"--output-path={out_file}",
    ] + _score_flags(request)
    if request.push_to_hub:
        cmd.append("--push-to-hub")
    return cmd


def _score_episode_total(request: RolloutScoreRequest) -> int:
    """How many episodes this run will score — the denominator before tqdm starts."""
    if request.episodes:
        return len(request.episodes)
    return len(_episode_lengths(_dataset_root(request.dataset_repo_id)))


def handle_rollout_score(request: RolloutScoreRequest) -> dict[str, Any]:
    """Score a rollout dataset in the background; the parquet lands in its root.

    Refuses to start a second job while one is running, and returns the command
    verbatim so the same job can be run by hand on a bigger GPU instead."""
    mismatch = _validate_score_request(request)
    if mismatch:
        return {"success": False, "status_code": 400, "message": mismatch}

    with _score_lock:
        if _score_state.get("running"):
            return {
                "success": False,
                "status_code": 409,
                "message": "已有打分任务在跑",
                "state": dict(_score_state),
            }

    cmd = build_score_command(request)
    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "rollout_eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{int(time.time())}.log"

    if not _gpu_big_enough():
        return {
            "success": False,
            "status_code": 400,
            "message": "本地显存放不下 4B 进度模型，可改用「在 pod 上打分」，或手动执行返回的命令",
            "command": " ".join(cmd),
        }

    try:
        handle = log_path.open("w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=str(_cache_root().parent),
        )
    except Exception as exc:
        logger.exception("Failed to start rollout scoring")
        return {"success": False, "status_code": 500, "message": f"启动打分失败: {exc}"}

    with _score_lock:
        _score_state.update(
            {
                "running": True,
                "repo_id": request.dataset_repo_id,
                "pid": proc.pid,
                "log_path": str(log_path),
                "started_at": time.time(),
                "command": " ".join(cmd),
                "exit_code": None,
            }
        )
    save_score_job(
        runner="local",
        dataset_repo_id=request.dataset_repo_id,
        backend=request.backend,
        log_path=str(log_path),
        pid=proc.pid,
        command=" ".join(cmd),
        episodes_total=_score_episode_total(request),
    )

    def pump() -> None:
        code = proc.wait()
        handle.close()
        with _score_lock:
            _score_state.update({"running": False, "exit_code": code, "finished_at": time.time()})
        finish_score_job(failed=code != 0)
        logger.info("Rollout scoring finished (rc=%s): %s", code, request.dataset_repo_id)

    threading.Thread(target=pump, name="rollout-score-watcher", daemon=True).start()
    return {
        "success": True,
        "message": "打分已开始",
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }


def latest_capture_stats(policy_ref: str = "") -> dict[str, Any]:
    """The newest capture dataset's real-world counters, for the deploy page.

    A model card should be able to say what that policy last did on the rig
    without the operator going and reading the scoring page, so look it up by
    walking the capture datasets' own stats files (cheap: one small JSON each).
    """
    root = _cache_root()
    if not root.is_dir():
        return {}
    best: dict[str, Any] = {}
    best_key = ""
    for org_dir in root.iterdir():
        if not org_dir.is_dir():
            continue
        for ds_dir in org_dir.iterdir():
            if not ds_dir.is_dir() or not ds_dir.name.startswith("rollout_"):
                continue
            if not (ds_dir / "meta" / "info.json").is_file():
                continue
            repo = f"{org_dir.name}/{ds_dir.name}"
            stats = read_take_stats(repo)
            if not stats.get("takes"):
                continue
            if policy_ref and stats.get("policy_ref") not in ("", None, policy_ref):
                continue
            key = f"{stats.get('updated') or ''}|{ds_dir.stat().st_mtime}"
            if key > best_key:
                best_key = key
                best = {
                    "repo_id": repo,
                    **stats,
                    "scored": (ds_dir / PROGRESS_FILENAME).is_file(),
                }
    return best


def preflight_rabc_training(dataset_repo_id: str) -> str | None:
    """Block an RA-BC run whose progress file is missing or belongs to another layout.

    Both failure modes are silent: lerobot falls back to weight 1.0 for frames it
    cannot look up, and a stale-but-loadable parquet trains every frame on the
    wrong weight. Only worth checking when the dataset is here — a cloud/pod run
    pulls its own copy, and the Hub file is whatever the last promote left.
    """
    root = _dataset_root(dataset_repo_id)
    if not (root / "meta" / "info.json").is_file():
        return None
    if not (root / PROGRESS_FILENAME).is_file():
        return (
            f"{dataset_repo_id} 开了 RA-BC 加权，但数据集根目录没有 {PROGRESS_FILENAME}。"
            "先在评估页给这个数据集打分，再点「以 X 为准」。"
        )
    alignment = check_progress_alignment(dataset_repo_id)
    return None if alignment.get("ok") else alignment.get("message")


_TQDM_RE = re.compile(
    r"Episodes:\s*(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([0-9:]+)<([0-9:]+)"
)


def _hms(value: str) -> int | None:
    try:
        parts = [int(p) for p in value.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _parse_scoring_progress(text: str) -> dict[str, Any]:
    """Episode counter and ETA from the scorer's outer tqdm bar.

    tqdm redraws the same line with carriage returns, so a plain `tail` hands
    back one long blob; split on \\r and read the last redraw.
    """
    if not text:
        return {}
    for line in reversed(text.replace("\r", "\n").splitlines()[-60:]):
        match = _TQDM_RE.search(line)
        if match:
            return {
                "episodes_done": int(match.group(2)),
                "episodes_total": int(match.group(3)),
                "percent": int(match.group(1)),
                "elapsed_s": _hms(match.group(4)),
                "eta_s": _hms(match.group(5)),
            }
    return {}


def _read_log_tail(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - limit))
            return fh.read().decode("utf-8", "ignore")
    except OSError:
        return ""


def _pid_alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def handle_rollout_score_status() -> dict[str, Any]:
    with _score_lock:
        state = dict(_score_state)
    if state.get("started_at"):
        state["elapsed_s"] = round(time.time() - state["started_at"], 1)
    if state.get("running") and state.get("log_path"):
        state.update(_parse_scoring_progress(_read_log_tail(str(state["log_path"]))))
    state.setdefault("running", False)

    # The child outlives a LeLab restart; the record on disk is what lets the
    # card reattach to it instead of claiming nothing is running.
    if not state.get("running"):
        record = load_score_job()
        if record.get("runner") == "local" and record.get("pid"):
            if _pid_alive(int(record["pid"])):
                state = dict(record)
                state["running"] = True
                state["elapsed_s"] = round(
                    time.time() - float(record.get("started_at") or time.time()), 1
                )
                state.update(_parse_scoring_progress(_read_log_tail(str(record.get("log_path") or ""))))
            else:
                clear_score_job()
    return state


def _prettify_log(text: str, keep_lines: int = 400) -> str:
    """Make tqdm's carriage-return redraws readable in a <pre> box."""
    lines: list[str] = []
    for chunk in text.replace("\r\n", "\n").split("\n"):
        last = chunk.split("\r")[-1].rstrip()
        if last and (not lines or lines[-1] != last):
            lines.append(last)
    return "\n".join(lines[-keep_lines:])


def handle_rollout_score_logs(max_chars: int = 20000) -> dict[str, Any]:
    """Live stdout of the running scorer, for the terminal panel."""
    with _score_lock:
        state = dict(_score_state)
    log_path = str(state.get("log_path") or "")
    if state.get("running") and log_path:
        return {
            "available": True,
            "source": "local",
            "log_path": log_path,
            "text": _prettify_log(_read_log_tail(log_path, max_chars)),
        }

    record = load_score_job()
    if not record:
        return {"available": False, "text": ""}
    log_path = str(record.get("log_path") or "")
    if record.get("runner") == "ssh":
        from .ssh_remote_env import read_remote_file_tail

        raw = read_remote_file_tail(str(record.get("config_id") or ""), log_path, max_chars)
        return {
            "available": bool(raw),
            "source": "ssh",
            "log_path": log_path,
            "text": _prettify_log(raw),
        }
    if not log_path:
        return {"available": False, "text": ""}
    return {
        "available": True,
        "source": "local",
        "log_path": log_path,
        "text": _prettify_log(_read_log_tail(log_path, max_chars)),
    }


def handle_score_job() -> dict[str, Any]:
    """Which scoring job is this card supposed to re-attach to, if any."""
    record = load_score_job()
    if not record:
        return {"active": False}
    if record.get("runner") == "ssh":
        status = handle_rollout_score_remote_status(
            str(record.get("config_id") or ""),
            str(record.get("dataset_repo_id") or ""),
            str(record.get("backend") or ""),
        )
        return {"active": bool(status.get("running")), "job": record, "status": status}
    status = handle_rollout_score_status()
    return {"active": bool(status.get("running")), "job": record, "status": status}


class RemoteScoreRequest(RolloutScoreRequest):
    config_id: str
    use_cn_mirror: bool = True


# What each upstream scorer names its output when it uploads with --push-to-hub.
HUB_PROGRESS_FILENAME = {
    "robometer": "robometer_progress.parquet",
    "sarm": "sarm_progress.parquet",
}


def _hub_dataset_exists(repo_id: str) -> tuple[bool, str]:
    """Whether the dataset is on the Hub — the pod pulls it from there.

    Bounded: `dataset_info` has no client-side timeout and on a slow link it
    blocks for minutes, which showed up as "Start Scoring does nothing". Ten
    seconds is enough to answer, and a timeout falls through to "try it" — the
    pod's own pull is the real enforcement.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    from huggingface_hub import HfApi

    def probe() -> tuple[bool, str]:
        try:
            HfApi().dataset_info(repo_id)
        except Exception as exc:  # noqa: BLE001 - RepositoryNotFoundError among others
            name = type(exc).__name__
            if "RepositoryNotFound" in name or "404" in str(exc):
                return False, name
            logger.info("Hub existence check skipped for %s: %s", repo_id, exc)
        return True, ""

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(probe).result(timeout=10)
    except FutureTimeout:
        logger.info("Hub existence check timed out for %s", repo_id)
        return True, "timeout"
    finally:
        pool.shutdown(wait=False)


def handle_rollout_score_remote(request: RemoteScoreRequest) -> dict[str, Any]:
    """Score on a pod and bring the parquet back over SFTP.

    The dataset still comes *from* the Hub (the pod pulls it), but the result
    does not go back that way: a pod with no write token fails at
    `--push-to-hub` after the work is done, so the file is copied over the SSH
    login we already use.
    """
    mismatch = _validate_score_request(request)
    if mismatch:
        return {"success": False, "status_code": 400, "message": mismatch}

    root = _dataset_root(request.dataset_repo_id)
    if not (root / "meta" / "info.json").is_file():
        return {"success": False, "status_code": 404, "message": f"本地没有数据集 {request.dataset_repo_id}"}

    on_hub, why = _hub_dataset_exists(request.dataset_repo_id)
    if not on_hub:
        return {
            "success": False,
            "status_code": 409,
            "message": f"{request.dataset_repo_id} 还不在 Hub 上（pod 从 Hub 拉数据）。先在 Dataset 页把它上传，再回来打分。",
        }

    # No `--push-to-hub`: a pod without a write token fails there *after* the
    # scoring is done (401 preupload), throwing away the result. The parquet
    # comes back over SFTP instead.
    job = request.model_copy(update={"device": "cuda", "push_to_hub": False})
    cmd = build_score_command(
        job, python="python3", output_file=f"/tmp/{PROGRESS_BY_BACKEND[job.backend]}"
    )
    from .ssh_remote_env import remote_scoring_status, run_remote_scoring

    # Two scorers on one GPU is how both die: the second start truncates the
    # first's log and the pair fight over VRAM. Refuse instead.
    existing = remote_scoring_status(request.config_id)
    if existing.get("running"):
        return {
            "success": False,
            "status_code": 409,
            "message": "pod 上已经有一个打分任务在跑，等它结束（或先停掉）再开新的",
        }

    result = run_remote_scoring(request.config_id, " ".join(cmd), use_cn_mirror=request.use_cn_mirror)
    if result.get("success"):
        with _score_lock:
            _remote_score_state.update(
                {
                    "running": True,
                    "dataset_repo_id": request.dataset_repo_id,
                    "backend": request.backend,
                    "config_id": request.config_id,
                    "started_at": time.time(),
                }
            )
        save_score_job(
            runner="ssh",
            dataset_repo_id=request.dataset_repo_id,
            backend=request.backend,
            config_id=request.config_id,
            log_path=str(result.get("log_path") or ""),
            command=" ".join(cmd),
            episodes_total=_score_episode_total(request),
        )
        result["message"] = "已在 pod 上开始打分，跑完会自动把进度文件拉回本地数据集目录"
    return result


def pull_progress_from_hub(repo_id: str, backend: str) -> dict[str, Any]:
    """Copy the pod-produced parquet from the Hub into the local dataset root."""
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    root = _dataset_root(repo_id)
    try:
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"列 Hub 文件失败：{exc}"}

    wanted = HUB_PROGRESS_FILENAME.get(backend, "")
    candidates = [wanted] if wanted in files else []
    candidates += [f for f in files if "progress" in f and f.endswith(".parquet") and f != PROGRESS_FILENAME]
    if not candidates:
        return {"success": False, "message": "Hub 上还没有进度文件，可能 pod 上的打分还没结束"}

    try:
        downloaded = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=candidates[0])
        tmp = root / f".{PROGRESS_BY_BACKEND[backend]}.tmp"
        shutil.copyfile(downloaded, tmp)
        tmp.replace(root / PROGRESS_BY_BACKEND[backend])
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"拉取进度文件失败：{exc}"}
    return {"success": True, "message": f"已从 Hub 取回 {candidates[0]}", "hub_file": candidates[0]}


def handle_rollout_score_remote_status(
    config_id: str, dataset_repo_id: str = "", backend: str = ""
) -> dict[str, Any]:
    """Poll the pod job; once it exits cleanly, reconcile the local copy."""
    from .ssh_remote_env import remote_scoring_status

    with _score_lock:
        tracked = dict(_remote_score_state)
    dataset_repo_id = dataset_repo_id or str(tracked.get("dataset_repo_id") or "")
    backend = backend or str(tracked.get("backend") or "")

    status = remote_scoring_status(config_id)
    if status.get("running"):
        status.update(_parse_scoring_progress(str(status.get("log_chunk") or "")))
    status.pop("log_chunk", None)
    already_pulled = bool(load_score_job().get("pulled_at"))
    if status.get("finished_ok") and dataset_repo_id and backend and not already_pulled:
        from .ssh_remote_env import download_remote_file

        filename = PROGRESS_BY_BACKEND[backend]
        root = _dataset_root(dataset_repo_id)
        tmp = root / f".{filename}.part"
        got = download_remote_file(config_id, f"/tmp/{filename}", str(tmp))
        if got.get("success"):
            with contextlib.suppress(OSError):
                tmp.replace(root / filename)
            mark_score_job_pulled()
            status["pulled"] = got
        else:
            status["pulled"] = got
        if status["pulled"].get("success"):
            alignment = check_progress_alignment(dataset_repo_id, PROGRESS_BY_BACKEND[backend])
            status["alignment"] = alignment
            if not alignment.get("ok"):
                status["pulled"]["success"] = False
                status["pulled"]["message"] = alignment.get("message")
        with _score_lock:
            _remote_score_state.clear()
        finish_score_job(failed=not status.get("finished_ok"))
    elif status.get("finished") and not status.get("running"):
        with _score_lock:
            _remote_score_state.clear()
        finish_score_job(failed=True)
    return status


class ScorePreflightRequest(RolloutScoreRequest):
    runner: str = "local"  # local | ssh
    config_id: str = ""


def _free_vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:  # noqa: BLE001
        return 0
    return max(values) if values else 0


def handle_score_preflight(request: ScorePreflightRequest) -> dict[str, Any]:
    """Answer "can this scoring job actually run?" before the click is spent.

    The two runners fail in completely different places — laptop VRAM versus a
    pod that is powered off, missing weights, or a dataset that never made it to
    the Hub — so report each finding instead of one opaque error at the end.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    root = _dataset_root(request.dataset_repo_id)
    add("数据集在本地", (root / "meta" / "info.json").is_file(), str(root))

    bad = _validate_score_request(request)
    add("后端与参数匹配", bad is None, bad or f"{request.backend} · {request.head_mode}")
    if request.backend == "sarm" and not request.reward_model_path.strip():
        add("SARM 模型路径", False, "要填你自训的 SARM repo 或本地路径")

    if request.runner == "ssh":
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        from .ssh_config import _tcp_reachable
        from .ssh_remote_env import reward_model_status

        if not request.config_id:
            add("SSH 连接", False, "没选 SSH 连接")
        else:
            # Gate on TCP first: a stopped pod must fail in ~3 s, not after the
            # Hub lookup (which is a slow network call on its own) finishes.
            from .ssh_config import get_ssh_config

            cfg = get_ssh_config(request.config_id)
            port_open = bool(cfg) and _tcp_reachable(cfg.host, cfg.port)
            if not port_open:
                add(
                    "SSH 能连上",
                    False,
                    f"端口 {getattr(cfg, 'port', '?')} 连不上（{getattr(cfg, 'host', '?')}）"
                    "——实例可能已停止，或重开后换了 host/port",
                )
            else:
                pool = ThreadPoolExecutor(max_workers=1)
                hub_future = pool.submit(_hub_dataset_exists, request.dataset_repo_id)
                try:
                    # One SSH session proves reachability *and* answers the weights
                    # question; a second handshake just to say "connected" doubled
                    # the wait on a slow pod.
                    rms = reward_model_status(request.config_id)
                    reachable = bool(rms.get("success"))
                    add(
                        "SSH 能连上",
                        reachable,
                        "" if reachable else str(rms.get("message", "")),
                    )
                    if reachable:
                        models = rms.get("models") or []
                        present = [m for m in models if m.get("present")]
                        if request.backend == "robometer":
                            add(
                                "打分权重在 pod 上",
                                bool(models) and len(present) == len(models),
                                f"{len(present)}/{len(models)} 个模型已下载"
                                if models
                                else str(rms.get("message", "")),
                            )
                    try:
                        # 3 s is plenty for HF when the link is healthy, and the
                        # real requirement is enforced again at start time (409).
                        on_hub, _why = hub_future.result(timeout=3)
                    except FutureTimeout:
                        on_hub, _why = True, "timeout"
                    add(
                        "数据集已在 Hub（pod 从 Hub 拉）",
                        on_hub,
                        "" if on_hub else "先在 Dataset 页把它上传到 Hub",
                    )
                finally:
                    pool.shutdown(wait=False)
    else:
        free = _free_vram_mib()
        add("本地显存够（≥11GB 空闲）", free >= 11000, f"当前空闲 {free / 1024:.1f}GB")

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "command": " ".join(build_score_command(request)),
    }


def _gpu_big_enough(min_free_mib: int = 11000) -> bool:
    """Progress models need ~11GB of free VRAM; report honestly if we don't."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        free = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception as exc:
        logger.info("nvidia-smi unavailable (%s); assuming scoring can run locally", exc)
        return True
    return bool(free) and max(free) >= min_free_mib
