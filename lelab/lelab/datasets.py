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

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub.errors import HfHubHTTPError

from .utils.hf_auth import cached_whoami, shared_hf_api

logger = logging.getLogger(__name__)


def _lerobot_cache_root() -> Path:
    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _is_dataset_dir(path: Path) -> bool:
    """A directory is a LeRobot dataset iff <dir>/meta/info.json exists."""
    try:
        return (path / "meta" / "info.json").is_file()
    except OSError:
        return False


def _dir_mtime_iso(path: Path) -> str | None:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except OSError:
        return None


def list_local_datasets() -> list[dict[str, Any]]:
    """Scan the LeRobot cache for local datasets (dirs containing meta/info.json).

    Walks one level deep: a top-level dataset dir is recorded as "<name>"; if a
    top-level dir is not itself a dataset, each subdir that is a dataset is
    recorded as "<top>/<sub>". Does not descend further.
    """
    root = _lerobot_cache_root()
    if not root.is_dir():
        return []

    out: list[dict[str, Any]] = []
    try:
        top_entries = list(root.iterdir())
    except OSError as e:
        logger.warning(f"Could not read LeRobot cache root {root}: {e}")
        return []

    for top in top_entries:
        try:
            if not top.is_dir():
                continue
        except OSError:
            continue

        if _is_dataset_dir(top):
            out.append(
                {
                    "repo_id": top.name,
                    "last_modified": _dir_mtime_iso(top),
                    "private": False,
                }
            )
            continue

        # Not a dataset itself — descend one level.
        try:
            sub_entries = list(top.iterdir())
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if not sub.is_dir():
                    continue
            except OSError:
                continue
            if _is_dataset_dir(sub):
                out.append(
                    {
                        "repo_id": f"{top.name}/{sub.name}",
                        "last_modified": _dir_mtime_iso(sub),
                        "private": False,
                    }
                )

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_user_datasets() -> list[dict[str, Any]]:
    info = cached_whoami()
    if info is None:
        return []

    authors = [info["name"]] + [o["name"] for o in info.get("orgs", [])]
    api = shared_hf_api()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for author in authors:
        try:
            for ds in api.list_datasets(author=author, filter="LeRobot", limit=200):
                if ds.id in seen:
                    continue
                seen.add(ds.id)
                out.append(
                    {
                        "repo_id": ds.id,
                        "last_modified": ds.last_modified.isoformat() if ds.last_modified else None,
                        "private": bool(getattr(ds, "private", False)),
                    }
                )
        except HfHubHTTPError as e:
            logger.warning(f"list_datasets({author}) failed: {e}")

    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def list_all_datasets() -> list[dict[str, Any]]:
    """Merged listing: Hub datasets + local cache, with `source` field.

    A repo_id present in both lists is collapsed to one entry with
    source="both" and last_modified set to the more recent of the two.
    """
    hub = list_user_datasets()
    local = list_local_datasets()

    merged: dict[str, dict[str, Any]] = {}
    for item in hub:
        merged[item["repo_id"]] = {**item, "source": "hub"}
    for item in local:
        rid = item["repo_id"]
        if rid in merged:
            existing = merged[rid]
            existing["source"] = "both"
            # Keep the newer timestamp; ISO strings sort lexically.
            a = existing.get("last_modified") or ""
            b = item.get("last_modified") or ""
            existing["last_modified"] = max(a, b) or None
        else:
            merged[rid] = {**item, "source": "local"}

    out = list(merged.values())
    out.sort(key=lambda d: d["last_modified"] or "", reverse=True)
    return out


def _video_frame_count(path: Path, fps: float) -> int | None:
    """Estimate the number of frames in a video file without full decoding."""
    try:
        from torchcodec.decoders import VideoDecoder

        md = VideoDecoder(str(path)).metadata
        n = getattr(md, "num_frames", None)
        if n:
            return int(n)
        dur = getattr(md, "duration_seconds", None)
        if dur:
            return round(dur * fps)
    except Exception as e:
        logger.warning("Could not probe video %s: %s", path, e)
    return None


def analyze_dataset(repo_id: str) -> dict[str, Any]:
    """Analyze a local dataset and return statistics + anomaly detection.

    Supports both the v3 layout (``data/chunk-*/file-*.parquet`` +
    ``meta/episodes.jsonl`` + chunked videos) and the legacy v2 layout
    (per-episode parquet files).

    Returns:
        Dictionary with:
        - statistics: episode_count, episode_length stats, action_range, camera_fps
        - anomalies: list of detected anomalies with episode index and type
    """
    root = _lerobot_cache_root()
    dataset_root = root / repo_id

    if not dataset_root.is_dir():
        return {"error": f"Dataset {repo_id} not found"}

    info_json = dataset_root / "meta" / "info.json"
    if not info_json.exists():
        return {"error": "Dataset meta/info.json not found"}

    import json
    from collections import defaultdict

    info = json.loads(info_json.read_text())
    fps = float(info.get("fps", 30))

    episodes_meta: list[dict[str, Any]] = []
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                episodes_meta.append(json.loads(line))

    # v3 datasets may store episode metadata as parquet instead of jsonl. These
    # rows carry the videos/<cam>/{chunk_index,file_index,from/to_timestamp}
    # columns the video-integrity check below depends on, so load them here
    # rather than falling back to the groupby derivation (which lacks them).
    if not episodes_meta:
        ep_parquet_files = sorted(dataset_root.glob("meta/episodes/chunk-*/file-*.parquet"))
        if ep_parquet_files:
            try:
                ep_df = pd.concat(
                    [pd.read_parquet(p) for p in ep_parquet_files], ignore_index=True
                )
                episodes_meta = ep_df.to_dict(orient="records")
            except Exception as e:
                logger.warning("Failed to read episodes parquet metadata: %s", e)

    data_files = sorted(dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        data_files = sorted(dataset_root.glob("chunk-*/episode_*.parquet"))
    if not data_files:
        return {"error": "No parquet files found"}

    # Read parquet files, skipping corrupted ones (e.g. from interrupted recording)
    valid_frames: list[pd.DataFrame] = []
    corrupted_files: list[str] = []
    for p in data_files:
        try:
            valid_frames.append(pd.read_parquet(p))
        except Exception as e:
            logger.warning("Skipping corrupted parquet file %s: %s", p.name, e)
            corrupted_files.append(p.name)

    if not valid_frames:
        return {"error": "All parquet files are corrupted"}

    df = pd.concat(valid_frames, ignore_index=True)

    if not episodes_meta:
        episodes_meta = [
            {
                "episode_index": int(ep),
                "length": int(len(g)),
                "dataset_from_index": int(g["index"].min()),
                "dataset_to_index": int(g["index"].max()) + 1,
            }
            for ep, g in df.groupby("episode_index")
        ]
    # Normalize episode_index to a plain int (parquet yields numpy scalars).
    for e in episodes_meta:
        e["episode_index"] = int(e["episode_index"])
        e["length"] = int(e["length"])
    episodes_meta.sort(key=lambda e: e["episode_index"])

    anomalies_dup: list[dict[str, Any]] = []
    seen_ep: dict[int, int] = {}
    for e in episodes_meta:
        seen_ep[e["episode_index"]] = seen_ep.get(e["episode_index"], 0) + 1
    dup_eps = sorted(ep for ep, n in seen_ep.items() if n > 1)
    if dup_eps:
        anomalies_dup.append({
            "episode": dup_eps[0],
            "type": "duplicate_episode_meta",
            "severity": "high",
            "message": (
                f"{len(episodes_meta)} episode-metadata rows but only "
                f"{len(seen_ep)} unique episode_index values; duplicated: "
                f"{dup_eps[:10]}{'...' if len(dup_eps) > 10 else ''}. "
                f"The episode metadata table is corrupt (usually from merging "
                f"two recording generations)."
            ),
        })
    ep_len = {e["episode_index"]: int(e["length"]) for e in episodes_meta}

    statistics = {
        "episode_count": len(episodes_meta),
        "episode_length": {"mean": 0, "std": 0, "min": 0, "max": 0},
        "action_range": {},
        "camera_fps": {"default": fps},
        "total_frames": sum(ep_len.values()),
    }

    anomalies = list(anomalies_dup)

    # Add corrupted file anomalies
    for cf in corrupted_files:
        anomalies.append({
            "episode": -1,
            "type": "corrupted_parquet_file",
            "severity": "high",
            "message": f"Data file '{cf}' is corrupted and was skipped. This episode's data is lost.",
        })

    episode_lengths = list(ep_len.values())

    if info.get("total_frames") not in (None, statistics["total_frames"]):
        anomalies.append({
            "episode": -1,
            "type": "frame_count_mismatch",
            "severity": "high",
            "message": (
                f"info.json total_frames={info['total_frames']} but parquet "
                f"rows={statistics['total_frames']}"
            ),
        })

    # Global index span: the episode table's cumulative range must equal the
    # actual data row count. A stale or duplicated table makes the training
    # sampler draw out-of-bounds indices deep into a run (IndexError in the
    # DataLoader worker), so flag it here where it can be repaired.
    if episodes_meta and all("dataset_to_index" in e for e in episodes_meta):
        table_span = max(int(e["dataset_to_index"]) for e in episodes_meta)
        if table_span != len(df):
            anomalies.append({
                "episode": -1,
                "type": "index_range_mismatch",
                "severity": "high",
                "message": (
                    f"Episode table spans {table_span} frames but the data "
                    f"parquet holds {len(df)} rows. Stale or duplicated episode "
                    f"metadata — training would crash on out-of-bounds frame "
                    f"indices. Use 'Rebuild episode table' to repair."
                ),
            })

    # 1. Short episodes (< 50% of median length)
    if episode_lengths:
        median_length = sorted(episode_lengths)[len(episode_lengths) // 2]
        mean_length = sum(episode_lengths) / len(episode_lengths)
        statistics["episode_length"] = {
            "mean": mean_length,
            "std": (sum((x - mean_length) ** 2 for x in episode_lengths) / len(episode_lengths)) ** 0.5,
            "min": min(episode_lengths),
            "max": max(episode_lengths),
            "median": median_length,
        }
        for ep_idx, length in ep_len.items():
            if length < median_length * 0.5:
                anomalies.append({
                    "episode": ep_idx,
                    "type": "short_duration",
                    "severity": "medium",
                    "message": f"Episode length {length} is less than 50% of median {median_length}",
                })

    # 2. Action range + per-episode outliers (beyond 3 sigma)
    action_cols = [c for c in df.columns if c.startswith("action.")]
    for col in action_cols:
        joint_name = col.replace("action.", "")
        series = df[col].dropna()
        if series.empty:
            continue
        min_val, max_val = float(series.min()), float(series.max())
        mean_val = float(series.mean())
        std_val = float((((series - mean_val) ** 2).mean()) ** 0.5)
        statistics["action_range"][joint_name] = {
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val,
        }
        if std_val > 0:
            per_ep_mean = df.groupby("episode_index")[col].mean()
            for ep_idx, ep_mean in per_ep_mean.items():
                if abs(ep_mean - mean_val) > 3 * std_val:
                    anomalies.append({
                        "episode": int(ep_idx),
                        "type": "action_outlier",
                        "severity": "high",
                        "joint": joint_name,
                        "message": f"Episode mean {ep_mean:.2f} is beyond 3σ from global mean {mean_val:.2f}",
                    })

    # 3. Video integrity: chunked video files must contain every frame the
    #    parquet rows reference. Encoders occasionally drop trailing frames,
    #    leaving rows whose video frame index is out of range at decode time.
    video_keys = (
        sorted({
            k[len("videos/"):-len("/chunk_index")]
            for k in episodes_meta[0]
            if k.startswith("videos/") and k.endswith("/chunk_index")
        })
        if episodes_meta
        else []
    )
    files: dict[tuple, dict] = defaultdict(lambda: {"expected": 0, "eps": []})
    for ep in episodes_meta:
        for vk in video_keys:
            key = (vk, ep.get(f"videos/{vk}/chunk_index"), ep.get(f"videos/{vk}/file_index"))
            if key[1] is None:
                continue
            files[key]["expected"] += ep_len[ep["episode_index"]]
            files[key]["eps"].append(ep["episode_index"])

    videos_checked = 0
    for (vk, ci, fi), meta in files.items():
        path = dataset_root / "videos" / vk / f"chunk-{int(ci):03d}" / f"file-{int(fi):03d}.mp4"
        if not path.exists():
            anomalies.append({
                "episode": -1,
                "type": "video_missing_file",
                "severity": "high",
                "message": f"Video file not found: {path.relative_to(dataset_root)}",
            })
            continue
        actual = _video_frame_count(path, fps)
        if actual is None:
            continue
        videos_checked += 1
        if actual < meta["expected"]:
            cum = 0
            affected = []
            for ep_idx in meta["eps"]:
                cum += ep_len[ep_idx]
                if cum > actual:
                    affected.append(ep_idx)
            anomalies.append({
                "episode": affected[0] if affected else -1,
                "type": "video_missing_frames",
                "severity": "high",
                "message": (
                    f"{path.name} has {actual} frames but metadata references "
                    f"{meta['expected']}; undecodable tail rows in episodes {affected}"
                ),
            })
    statistics["videos_checked"] = videos_checked

    return {
        "success": True,
        "statistics": statistics,
        "anomalies": anomalies,
    }
