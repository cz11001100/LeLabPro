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

"""Rebuild the episode index of a recording that was never finalized.

``LeRobotDataset.finalize()`` is what writes ``meta/episodes/`` and the parquet
footers; frames, videos and ``meta/info.json`` are written as the session runs.
A recording interrupted before finalize therefore leaves a directory that looks
complete but carries no episode index, and ``LeRobotDataset`` reads that as
"not available locally" and tries to download it from the Hub — a 404 for a
dataset that was never pushed.

Recovery is bounded by the parquet footer: a data file the writer was still
appending to when it died cannot be read at all, so its frames and every frame
after it are gone.
"""

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHUNK_FILE_RE = re.compile(r"chunk-(\d+)/file-(\d+)\.")


class DatasetRepairError(Exception):
    """The dataset is damaged beyond what can be rebuilt from disk."""


def _chunk_file(path: Path) -> tuple[int, int]:
    match = _CHUNK_FILE_RE.search(path.as_posix())
    if match is None:
        raise DatasetRepairError(f"Unexpected dataset file layout: {path}")
    return int(match.group(1)), int(match.group(2))


def _video_files(root: Path, video_key: str) -> list[tuple[int, int, float]]:
    """Return (chunk_index, file_index, duration_s) per encoded video file."""
    import av

    files = []
    for path in sorted((root / "videos" / video_key).rglob("*.mp4"), key=_chunk_file):
        try:
            with av.open(str(path)) as container:
                duration = float(container.duration) / av.time_base if container.duration else 0.0
        except Exception as e:
            # An unclosed mp4 has no moov atom; treat it as carrying no episodes.
            logger.warning("Could not read %s (%s) — episodes in it are unrecoverable", path, e)
            duration = 0.0
        chunk_index, file_index = _chunk_file(path)
        files.append((chunk_index, file_index, duration))
    return files


def _set_aside_unreadable(path: Path) -> None:
    """Rename a file the readers can't parse out of the dataset's glob path.

    Renamed rather than deleted: a footer-less parquet is unreadable today, but
    the bytes are the only surviving copy of those frames and pyarrow may grow a
    way to salvage them. Leaving the original name in place is not an option —
    every reader globs ``data/**/*.parquet``, so it would keep breaking the
    dataset.
    """
    target = path.with_suffix(path.suffix + ".unreadable")
    suffix = 0
    while target.exists():
        suffix += 1
        target = path.with_suffix(path.suffix + f".unreadable{suffix}")
    path.rename(target)
    logger.info("Set aside unreadable file: %s", target)


def _episodes_from_data(root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Reconstruct one row per episode from the readable data files.

    Also returns the trailing files that have no parquet footer — the reader
    globs every file under ``data/``, so they have to be moved out of the way.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    # Written with the first saved episode, so its absence means the session
    # died before one completed.
    if not (root / "meta" / "tasks.parquet").exists():
        return [], []

    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_by_index = {int(index): task for task, index in tasks["task_index"].items()}

    data_files = sorted((root / "data").rglob("*.parquet"), key=_chunk_file)
    rows: list[dict[str, Any]] = []
    next_index = 0
    for position, path in enumerate(data_files):
        try:
            frames = pq.read_table(path, columns=["episode_index", "task_index"]).to_pandas()
        except Exception as e:
            # No footer: the writer died mid-file. Nothing from here on is
            # readable, since only the newest file is ever left open.
            logger.warning("Data file %s is truncated (%s) — recovering the episodes before it", path, e)
            return rows, data_files[position:]

        chunk_index, file_index = _chunk_file(path)
        for episode_index, group in frames.groupby("episode_index", sort=True):
            length = len(group)
            rows.append(
                {
                    "episode_index": int(episode_index),
                    "tasks": [task_by_index[int(i)] for i in sorted(group["task_index"].unique())],
                    "length": length,
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                    "dataset_from_index": next_index,
                    "dataset_to_index": next_index + length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )
            next_index += length
    return rows, []


def _assign_videos(rows: list[dict[str, Any]], root: Path, video_key: str, fps: int) -> int:
    """Fill in the video columns for `video_key`, in place.

    Episodes are concatenated into a video file until it hits the size limit,
    so each episode's window is its predecessors' durations within that file.
    Returns how many leading rows are actually covered by encoded video.
    """
    files = _video_files(root, video_key)
    cursor = 0
    offset = 0.0
    tolerance = 1.0 / fps

    for covered, row in enumerate(rows):
        duration = row["length"] / fps
        while cursor < len(files) and offset + duration > files[cursor][2] + tolerance:
            cursor += 1
            offset = 0.0
        if cursor >= len(files):
            return covered

        chunk_index, file_index, _ = files[cursor]
        row[f"videos/{video_key}/chunk_index"] = chunk_index
        row[f"videos/{video_key}/file_index"] = file_index
        row[f"videos/{video_key}/from_timestamp"] = offset
        row[f"videos/{video_key}/to_timestamp"] = offset + duration
        offset += duration

    return len(rows)


def _trim_data(root: Path, keep_through: int) -> int:
    """Drop data rows belonging to episodes past `keep_through`.

    The reader loads every row under data/ and push_to_hub uploads the whole
    directory, so rows the rebuilt index no longer references would otherwise
    travel with the dataset. Returns the number of frames dropped.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    dropped = 0
    for path in sorted((root / "data").rglob("*.parquet"), key=_chunk_file):
        table = pq.read_table(path)
        kept = table.filter(pc.less_equal(table["episode_index"], keep_through))
        if kept.num_rows == table.num_rows:
            continue
        dropped += table.num_rows - kept.num_rows
        if kept.num_rows == 0:
            path.unlink()
        else:
            pq.write_table(kept, path, compression="snappy", use_dictionary=True)
    return dropped


def _episode_video_frames(root: Path, video_key: str, row: dict[str, Any]) -> "np.ndarray":  # noqa: F821
    """Decode a sample of the frames one episode references, as (N, C, H, W) in [0, 1]."""
    import av
    import numpy as np

    from lerobot.datasets.compute_stats import sample_indices

    file_index = row[f"videos/{video_key}/file_index"]
    start = row[f"videos/{video_key}/from_timestamp"]
    end = row[f"videos/{video_key}/to_timestamp"]

    path = next((root / "videos" / video_key).rglob(f"*file-{file_index:03d}.mp4"))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Without this every episode re-decodes its file from frame 0, which makes
        # recomputing stats over a few hundred episodes take longer than any
        # reasonable timeout (1011_4: 190 episodes x 3 cameras, killed at 50 min).
        if start > 0 and stream.time_base:
            container.seek(int(start / stream.time_base), stream=stream, backward=True, any_frame=False)
        frames = [
            frame
            for frame in container.decode(stream)
            if start <= float(frame.time) < end
        ]

    if not frames:
        raise DatasetRepairError(f"No decodable frames for {video_key} in episode {row['episode_index']}")

    sampled = [frames[i] for i in sample_indices(len(frames))]
    return np.stack([frame.to_ndarray(format="rgb24").transpose(2, 0, 1) for frame in sampled]) / 255.0


def _recompute_stats(root: Path, info: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Rewrite meta/stats.json over the retained frames only.

    stats.json is aggregated as each episode is saved, so it still describes
    episodes that were lost with the interruption, and training normalizes with
    it. Rebuilt the way recording builds it, per episode then aggregated, so the
    numbers match what the same episodes would have produced on their own.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.io_utils import write_stats

    columns: dict[str, np.ndarray] = {}  # noqa: F821
    for path in sorted((root / "data").rglob("*.parquet"), key=_chunk_file):
        table = pq.read_table(path)
        for key in table.schema.names:
            values = np.stack(table[key].to_numpy(zero_copy_only=False).tolist())
            columns[key] = np.concatenate([columns[key], values]) if key in columns else values

    per_episode = []
    for row in rows:
        episode_stats = {}
        for key, feature in info["features"].items():
            if feature["dtype"] in {"string", "language"}:
                continue

            if feature["dtype"] in {"image", "video"}:
                values = _episode_video_frames(root, key, row)
                episode_stats[key] = {
                    name: value if name == "count" else np.squeeze(value, axis=0)
                    for name, value in get_feature_stats(values, axis=(0, 2, 3), keepdims=True).items()
                }
            elif key in columns:
                values = columns[key][row["dataset_from_index"] : row["dataset_to_index"]]
                episode_stats[key] = get_feature_stats(values, axis=0, keepdims=values.ndim == 1)
        per_episode.append(episode_stats)

    write_stats(aggregate_stats(per_episode), root)


def _write_episodes(root: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the episode index with `rows`, keeping the previous one until the
    replacement is fully on disk.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    episodes_dir = root / "meta" / "episodes"
    staging = root / "meta" / "episodes.reindexing"
    shutil.rmtree(staging, ignore_errors=True)
    path = staging / "chunk-000" / "file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy", use_dictionary=True)

    # Rebuilding the index is the last step of a repair, and losing the old one to
    # a crash before the new one lands would leave the dataset with no index at all.
    if episodes_dir.exists():
        shutil.rmtree(episodes_dir)
    os.replace(staging, episodes_dir)


def check_dataset_health(repo_id: str) -> dict[str, Any]:
    """Check if a local dataset is healthy or corrupted.

    Returns a dict with:
    - healthy: bool — whether the dataset can be loaded without errors
    - corrupted_episodes: int — number of episodes that appear corrupted
    - total_episodes: int — total episodes recorded
    - error: str | None — error message if unhealthy
    - needs_repair: bool — whether repair is recommended
    """
    from lerobot.utils.constants import HF_LEROBOT_HOME

    cache_root = Path(HF_LEROBOT_HOME).resolve()
    root = (cache_root / repo_id).resolve()

    if root == cache_root or cache_root not in root.parents:
        return {"healthy": False, "error": f"Invalid dataset id: {repo_id}", "needs_repair": False}

    if not (root / "meta" / "info.json").exists():
        return {"healthy": False, "error": "Dataset not found locally", "needs_repair": False}

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        total_episodes = info.get("total_episodes", 0)
    except Exception as e:
        return {"healthy": False, "error": f"Cannot read info.json: {e}", "needs_repair": True, "total_episodes": 0}

    # Check episodes parquet files
    episodes_healthy = True
    episodes_error = None
    try:
        from lerobot.datasets.io_utils import load_episodes
        load_episodes(root)
    except Exception as e:
        episodes_healthy = False
        episodes_error = str(e)

    # Check data parquet files
    corrupted_data_files = []
    total_data_files = 0
    data_dir = root / "data"
    if data_dir.exists():
        import pyarrow.parquet as pq
        for path in sorted(data_dir.rglob("*.parquet"), key=_chunk_file):
            total_data_files += 1
            try:
                pq.read_table(path, columns=["episode_index"]).to_pandas()
            except Exception:
                corrupted_data_files.append(str(path.relative_to(root)))

    # Check video files
    corrupted_video_files = []
    total_video_files = 0
    videos_dir = root / "videos"
    if videos_dir.exists():
        import av
        for path in sorted(videos_dir.rglob("*.mp4")):
            total_video_files += 1
            try:
                with av.open(str(path)) as container:
                    _ = container.duration
            except Exception:
                corrupted_video_files.append(str(path.relative_to(root)))

    healthy = episodes_healthy and len(corrupted_data_files) == 0
    needs_repair = not healthy

    # Estimate corrupted episodes from truncated data files
    corrupted_episodes = 0
    if corrupted_data_files:
        # Count episodes in corrupted files by checking which episode indices
        # appear in readable files vs what info.json says
        try:
            readable_episodes = set()
            for path in sorted(data_dir.rglob("*.parquet"), key=_chunk_file):
                if str(path.relative_to(root)) in corrupted_data_files:
                    continue
                try:
                    frames = pq.read_table(path, columns=["episode_index"]).to_pandas()
                    readable_episodes.update(frames["episode_index"].unique().tolist())
                except Exception:
                    pass
            all_expected = set(range(total_episodes))
            corrupted_episodes = len(all_expected - readable_episodes)
        except Exception:
            corrupted_episodes = total_episodes

    result = {
        "healthy": healthy,
        "total_episodes": total_episodes,
        "corrupted_episodes": corrupted_episodes,
        "episodes_healthy": episodes_healthy,
        "episodes_error": episodes_error,
        "corrupted_data_files": corrupted_data_files,
        "total_data_files": total_data_files,
        "corrupted_video_files": corrupted_video_files,
        "total_video_files": total_video_files,
        "needs_repair": needs_repair,
        "error": episodes_error if not healthy else None,
    }
    return result


# Repair progress tracking (similar to upload progress)
repair_progress: dict[str, Any] = {
    "active": False,
    "phase": "",
    "percent": 0,
    "current": 0,
    "total": 0,
    "dataset_repo_id": "",
    "error": None,
    "completed": False,
}


def get_repair_progress() -> dict[str, Any]:
    return dict(repair_progress)


def _update_repair_progress(**kwargs):
    repair_progress.update(kwargs)


def repair_local_dataset_with_progress(repo_id: str, force: bool = False) -> str | None:
    """Repair a dataset with progress tracking.

    Runs the repair in the current thread but updates progress as it goes.
    With force=True the episode table is rebuilt from the data parquet even
    when it parses fine — needed for stale/duplicated tables that parse but
    disagree with the actual rows (index_range_mismatch).
    """
    import threading

    from lerobot.datasets.io_utils import load_episodes
    from lerobot.utils.constants import HF_LEROBOT_HOME

    _update_repair_progress(
        active=True,
        phase="Checking dataset health...",
        percent=5,
        current=0,
        total=0,
        dataset_repo_id=repo_id,
        error=None,
        completed=False,
    )

    try:
        cache_root = Path(HF_LEROBOT_HOME).resolve()
        root = (cache_root / repo_id).resolve()

        if root == cache_root or cache_root not in root.parents:
            raise DatasetRepairError(f"Invalid dataset id: {repo_id}")

        if not (root / "meta" / "info.json").exists():
            _update_repair_progress(active=False, completed=True, percent=100, phase="Dataset not found")
            return None

        try:
            load_episodes(root)
            healthy = True
        except Exception as e:
            logger.info("%s has no usable episode index (%s) — rebuilding", repo_id, e)
            healthy = False
        if healthy and not force:
            _update_repair_progress(active=False, completed=True, percent=100, phase="Dataset is already healthy")
            return None

        _update_repair_progress(phase="Reading data files...", percent=15)
        info = json.loads((root / "meta" / "info.json").read_text())
        fps = info["fps"]
        recorded_episodes = info.get("total_episodes", 0)

        rows, truncated = _episodes_from_data(root)

        _update_repair_progress(phase="Assigning video files...", percent=30, total=len(info["features"]))
        video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
        for i, video_key in enumerate(video_keys):
            _update_repair_progress(
                phase=f"Processing video: {video_key}",
                percent=30 + int(20 * (i + 1) / max(len(video_keys), 1)),
                current=i + 1,
                total=len(video_keys),
            )
            del rows[_assign_videos(rows, root, video_key, fps):]

        if not rows:
            raise DatasetRepairError(
                f"{repo_id} was interrupted before any episode was fully written to disk."
            )

        _update_repair_progress(phase="Setting corrupted files aside...", percent=55)
        for path in truncated:
            _set_aside_unreadable(path)

        lost = recorded_episodes - len(rows)
        if lost > 0:
            _update_repair_progress(phase="Trimming orphaned data...", percent=60)
            _trim_data(root, rows[-1]["episode_index"])

            _update_repair_progress(phase="Recomputing statistics (this may take a few minutes)...", percent=65)
            _recompute_stats(root, info, rows)

        _update_repair_progress(phase="Writing episode index...", percent=90)
        _write_episodes(root, rows)

        info["total_episodes"] = len(rows)
        info["total_frames"] = sum(row["length"] for row in rows)
        info["splits"] = {"train": f"0:{len(rows)}"}
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

        message = f"Recovered {len(rows)} episode(s)"
        if lost > 0:
            message += f"; {lost} episode(s) were lost"

        _update_repair_progress(
            active=False,
            completed=True,
            percent=100,
            phase="Repair complete",
        )
        logger.info(message)
        return message

    except Exception as e:
        _update_repair_progress(
            active=False,
            completed=True,
            percent=0,
            phase="Repair failed",
            error=str(e),
        )
        raise


def repair_local_dataset(repo_id: str) -> str | None:
    """Rebuild `repo_id`'s episode index when the recording never finalized.

    Returns a description of what was recovered, or None when the dataset is
    absent or already readable. Raises DatasetRepairError when nothing is left
    to recover.
    """
    from lerobot.datasets.io_utils import load_episodes
    from lerobot.utils.constants import HF_LEROBOT_HOME

    cache_root = Path(HF_LEROBOT_HOME).resolve()
    root = (cache_root / repo_id).resolve()

    # repo_id reaches here straight from the request body, and this function
    # rewrites info.json and renames files. Keep it inside the cache.
    if root == cache_root or cache_root not in root.parents:
        raise DatasetRepairError(f"Invalid dataset id: {repo_id}")

    if not (root / "meta" / "info.json").exists():
        return None

    try:
        load_episodes(root)
        return None
    except Exception as e:
        logger.info("%s has no usable episode index (%s) — rebuilding it from disk", repo_id, e)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    recorded_episodes = info.get("total_episodes", 0)

    rows, truncated = _episodes_from_data(root)
    for video_key, feature in info["features"].items():
        if feature["dtype"] == "video":
            del rows[_assign_videos(rows, root, video_key, fps) :]

    if not rows:
        raise DatasetRepairError(
            f"{repo_id} was interrupted before any episode was fully written to disk, "
            "and the partial files it left behind cannot be read. Re-record it."
        )

    # The reader loads every parquet file under data/, so an unreadable one
    # would still break it. Move it out of the glob rather than deleting it.
    for path in truncated:
        _set_aside_unreadable(path)

    lost = recorded_episodes - len(rows)
    if lost > 0:
        # Episodes the videos don't cover still have their frames in the
        # readable shards; drop them so data, index and stats agree.
        _trim_data(root, rows[-1]["episode_index"])
        _recompute_stats(root, info, rows)

    _write_episodes(root, rows)

    info["total_episodes"] = len(rows)
    info["total_frames"] = sum(row["length"] for row in rows)
    info["splits"] = {"train": f"0:{len(rows)}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    message = f"Recovered {len(rows)} episode(s) from an interrupted recording of {repo_id}"
    if lost > 0:
        message += f"; {lost} episode(s) were lost because the recording was cut short"
    logger.info(message)
    return message


def reindex_episodes_to_contiguous(root: Path) -> str | None:
    """Compact a dataset's episode_index to a contiguous 0..N-1 range.

    Deleting episodes from the middle (or a resume that mis-numbered them) can
    leave the episode_index with gaps or duplicates, e.g. ``[0..9, 17..19]``.
    lerobot numbers the next episode as ``meta.total_episodes`` (a count), so a
    gapped index makes a later resume collide with existing indices and merge
    two generations. This renumbers every episode to ``0..N-1`` in ascending
    order of its current label, recomputes the global ``index`` column to match,
    and rebuilds the episode metadata, video windows, stats and info.json so the
    whole dataset is internally consistent again.

    Returns a description of the renumbering, or None when the index is already
    contiguous and the metadata agrees with the data (a no-op). A shard left
    unreadable by an interrupted session is moved aside and the episodes in it
    are dropped; DatasetRepairError is raised only when nothing readable is left.
    """
    import os

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    # The data files are authoritative for which episodes actually exist on disk.
    # A file the writer was still appending to when the session died has no footer
    # and can never be read, so it is set aside rather than aborting the reindex —
    # that partial file is exactly what an interrupted recording leaves behind, and
    # leaving it in data/ breaks every reader that globs the directory.
    data_files: list[Path] = []
    present: set[int] = set()
    for path in sorted((root / "data").rglob("*.parquet"), key=_chunk_file):
        try:
            col = pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()
        except Exception as e:
            logger.warning("reindex: data file %s is unreadable (%s) — setting it aside", path.name, e)
            _set_aside_unreadable(path)
            continue
        data_files.append(path)
        present.update(int(e) for e in col)

    if not data_files:
        raise DatasetRepairError(
            f"{root.name}: no readable episode data on disk, so its episode index "
            "cannot be rebuilt. Re-record it."
        )

    ordered = sorted(present)
    data_contiguous = ordered == list(range(len(ordered)))

    # The episode metadata can drift from the data — e.g. the old v2 delete left a
    # gapped index while the data stayed contiguous — so read it too and treat any
    # disagreement as work to do, not just a non-contiguous data index.
    meta_eps: list[int] | None = None
    try:
        import pandas as pd

        ep_parquets = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        if ep_parquets:
            mdf = pd.concat(
                [pd.read_parquet(p, columns=["episode_index"]) for p in ep_parquets],
                ignore_index=True,
            )
            meta_eps = sorted(int(e) for e in mdf["episode_index"].tolist())
    except Exception as e:
        logger.warning("reindex: could not read episode metadata (%s)", e)

    if data_contiguous and meta_eps == ordered:
        return None  # data and metadata already agree on a contiguous 0..N-1

    # Renumber the data shards only when the data itself is gapped/duplicated.
    renumbered = 0
    if not data_contiguous:
        mapping = {old: new for new, old in enumerate(ordered)}
        renumbered = sum(1 for old, new in mapping.items() if old != new)
        running = 0
        for path in data_files:
            table = pq.read_table(path)
            n = table.num_rows

            ep_field = table.schema.field("episode_index")
            new_ep = [mapping[int(e)] for e in table["episode_index"].to_pylist()]
            table = table.set_column(
                table.schema.get_field_index("episode_index"), ep_field, pa.array(new_ep, type=ep_field.type)
            )

            idx_field = table.schema.field("index")
            new_idx = list(range(running, running + n))
            table = table.set_column(
                table.schema.get_field_index("index"), idx_field, pa.array(new_idx, type=idx_field.type)
            )
            running += n
            # Write-then-replace: a session that dies mid-rewrite must not be able
            # to leave the shard half-written, since that costs the episodes in it.
            staging = path.with_suffix(".parquet.reindexing")
            pq.write_table(table, staging, compression="snappy", use_dictionary=True)
            os.replace(staging, path)

    # Rebuild the episode index, video windows, stats and info.json from the
    # now-contiguous data, reusing the same helpers the interrupted-recording
    # repair uses so the result matches what recording would have produced.
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info["fps"]
    rows, truncated = _episodes_from_data(root)
    for path in truncated:
        logger.warning("Truncated data file %s left in place after reindex", path)
    for video_key, feature in info["features"].items():
        if feature["dtype"] == "video":
            del rows[_assign_videos(rows, root, video_key, fps) :]
    if not rows:
        raise DatasetRepairError(f"{root.name}: no complete episodes to reindex")

    _recompute_stats(root, info, rows)
    _write_episodes(root, rows)

    info["total_episodes"] = len(rows)
    info["total_frames"] = sum(row["length"] for row in rows)
    info["splits"] = {"train": f"0:{len(rows)}"}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    message = (
        f"Reindexed {len(rows)} episode(s) to a contiguous 0..{len(rows) - 1} range "
        f"(renumbered {renumbered} in the data, rebuilt metadata from disk)"
    )
    logger.info(message)
    return message


def delete_episodes_via_rebuild(root: Path, repo_id: str, episode_indices: list[int]) -> str:
    """Remove episodes using lerobot's official editor, then swap the result into place.

    lerobot's ``delete_episodes`` copies whole video and parquet files that hold no
    deleted episode, and re-encodes only the files that mix kept and dropped
    episodes. Rebuilding by hand instead decodes and re-encodes every frame of every
    surviving episode through Python, which on a 190-episode dataset costs an order
    of magnitude more time for the same result.

    The original directory is kept as ``<name>__bak_del<indices>`` so the operation
    is reversible. Returns a summary message.
    """
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    drop = sorted(set(int(i) for i in episode_indices))
    if not drop:
        raise DatasetRepairError("No episodes selected")

    cache_root = Path(HF_LEROBOT_HOME).resolve()
    root = root.resolve()
    if root == cache_root or cache_root not in root.parents:
        raise DatasetRepairError(f"Invalid dataset id: {repo_id}")
    if not (root / "meta" / "info.json").exists():
        raise DatasetRepairError(f"{repo_id}: no info.json to rebuild from")

    _update_repair_progress(
        active=True,
        completed=False,
        error=None,
        percent=5,
        current=0,
        total=0,
        phase=f"Opening {repo_id}",
        dataset_repo_id=repo_id,
    )

    try:
        from .utils.video_backend import video_backend

        src = LeRobotDataset(repo_id, root=root, video_backend=video_backend())
    except Exception as e:
        _update_repair_progress(active=False, completed=False, error=str(e), phase="Cannot open dataset")
        raise DatasetRepairError(f"{repo_id}: cannot be opened for editing ({e})") from e

    total = src.meta.total_episodes
    missing = [i for i in drop if i not in range(total)]
    if missing:
        raise DatasetRepairError(
            f"{repo_id}: episode(s) {missing} not found; the dataset has {total} "
            f"episode(s) (0..{total - 1})"
        )
    if len(drop) >= total:
        raise DatasetRepairError(f"{repo_id}: refusing to delete all {total} episode(s)")

    tmp_root = root.with_name(root.name + "__rebuild_tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    _update_repair_progress(
        percent=15, current=len(drop), total=total, phase="Re-encoding affected files, copying the rest"
    )
    try:
        result = delete_episodes(src, drop, output_dir=tmp_root, repo_id=repo_id)
        kept = result.meta.total_episodes
    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        _update_repair_progress(
            active=False, completed=False, error=str(e), phase="Rebuild failed; original untouched"
        )
        raise DatasetRepairError(f"{repo_id}: rebuild failed ({e})") from e

    label = "-".join(str(i) for i in drop[:5]) + (f"-etc{len(drop)}" if len(drop) > 5 else "")
    backup = root.with_name(root.name + f"__bak_del{label}")
    if backup.exists():
        shutil.rmtree(backup)
    root.rename(backup)
    tmp_root.rename(root)

    _update_repair_progress(
        active=False,
        completed=True,
        error=None,
        percent=100,
        current=kept,
        total=total,
        phase=f"Rebuilt {kept} episode(s); backup kept at {backup.name}",
    )

    logger.info(
        "Deleted episodes %s from %s; %d -> %d episode(s) (backup at %s)",
        drop,
        repo_id,
        total,
        kept,
        backup.name,
    )
    return f"Deleted {len(drop)} episode(s) {drop}; {total} -> {kept} contiguous episode(s)"


def delete_episode_via_rebuild(root: Path, repo_id: str, episode_index: int) -> str:
    """Remove a single episode; see `delete_episodes_via_rebuild`."""
    return delete_episodes_via_rebuild(root, repo_id, [episode_index])
