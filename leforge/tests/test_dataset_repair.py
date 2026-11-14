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
"""Tests for leforge.dataset_repair — rebuilding the episode index of a
recording that was interrupted before `LeRobotDataset.finalize()` ran."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO_ID = "repair_test/session"
EPISODES = 2
FRAMES = 5
FPS = 10


@pytest.fixture
def lerobot_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point LeRobot's dataset cache at a tmp dir.

    HF_LEROBOT_HOME is read from the environment once at import time and then
    copied into the modules that use it, so every binding has to be patched —
    the env var alone is too late by the time a test runs.
    """
    home = tmp_path / "lerobot"
    home.mkdir()
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", home)
    monkeypatch.setattr("lerobot.datasets.dataset_metadata.HF_LEROBOT_HOME", home)
    return home


def _record(
    repo_id: str,
    *,
    video: bool,
    episodes: int = EPISODES,
    batch_encoding_size: int = 1,
    metadata_buffer_size: int = 10,
) -> Path:
    """Write a small finished dataset and return its root."""
    from lerobot.datasets import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    if video:
        features["observation.images.cam"] = {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["h", "w", "c"],
        }

    dataset = LeRobotDataset.create(
        repo_id,
        fps=FPS,
        features=features,
        use_videos=video,
        batch_encoding_size=batch_encoding_size,
        metadata_buffer_size=metadata_buffer_size,
    )
    for episode in range(episodes):
        for frame in range(FRAMES):
            values = np.full(2, episode * FRAMES + frame, dtype=np.float32)
            item = {"action": values, "observation.state": values, "task": "repair me"}
            if video:
                item["observation.images.cam"] = np.full((32, 32, 3), 100, dtype=np.uint8)
            dataset.add_frame(item)
        dataset.save_episode()
    dataset.finalize()
    return dataset.root


def test_chunk_file_parses_indices_from_path() -> None:
    from leforge.dataset_repair import _chunk_file

    assert _chunk_file(Path("data/chunk-003/file-012.parquet")) == (3, 12)


def test_chunk_file_rejects_unexpected_layout() -> None:
    from leforge.dataset_repair import DatasetRepairError, _chunk_file

    with pytest.raises(DatasetRepairError):
        _chunk_file(Path("data/whatever.parquet"))


def test_absent_dataset_is_left_to_the_normal_hub_path(lerobot_home: Path) -> None:
    from leforge.dataset_repair import repair_local_dataset

    assert repair_local_dataset("someone/never-recorded") is None


def test_readable_dataset_is_not_touched(lerobot_home: Path) -> None:
    from leforge.dataset_repair import repair_local_dataset

    _record(REPO_ID, video=False)

    assert repair_local_dataset(REPO_ID) is None


@pytest.mark.parametrize("repo_id", ["../escape", "repair_test/../../escape", "/etc"])
def test_repo_id_cannot_escape_the_cache_directory(lerobot_home: Path, repo_id: str) -> None:
    from leforge.dataset_repair import DatasetRepairError, repair_local_dataset

    with pytest.raises(DatasetRepairError, match="Invalid dataset id"):
        repair_local_dataset(repo_id)


@pytest.mark.parametrize("video", [False, True])
def test_missing_episode_index_is_rebuilt_from_the_data_files(lerobot_home: Path, video: bool) -> None:
    from leforge.dataset_repair import repair_local_dataset
    from lerobot.datasets import LeRobotDataset

    root = _record(REPO_ID, video=video)
    # What an un-finalized recording leaves behind: frames and videos on disk,
    # no episode index. Loading it in this state is what reaches for the Hub
    # and 404s, so the test never does it — it goes straight to the repair.
    shutil.rmtree(root / "meta" / "episodes")

    message = repair_local_dataset(REPO_ID)
    assert message is not None
    assert f"Recovered {EPISODES} episode(s)" in message

    repaired = LeRobotDataset(REPO_ID)
    assert repaired.num_episodes == EPISODES
    assert repaired.num_frames == EPISODES * FRAMES

    last = repaired[repaired.num_frames - 1]
    assert int(last["episode_index"]) == EPISODES - 1
    assert last["task"] == "repair me"
    np.testing.assert_allclose(last["action"].numpy(), np.full(2, EPISODES * FRAMES - 1))
    if video:
        assert tuple(last["observation.images.cam"].shape) == (3, 32, 32)


def test_missing_info_is_rebuilt_for_numeric_only_dataset(lerobot_home: Path) -> None:
    from leforge.dataset_repair import repair_local_dataset
    from lerobot.datasets import LeRobotDataset

    root = _record(REPO_ID, video=False)
    shutil.rmtree(root / "meta" / "episodes")
    (root / "meta" / "info.json").unlink()

    message = repair_local_dataset(REPO_ID)

    assert message is not None
    repaired = LeRobotDataset(REPO_ID)
    assert repaired.num_episodes == EPISODES
    assert repaired.num_frames == EPISODES * FRAMES
    assert repaired.meta.fps == FPS
    assert (root / "meta" / "stats.json").exists()


def test_missing_info_does_not_discard_existing_videos(lerobot_home: Path) -> None:
    from leforge.dataset_repair import DatasetRepairError, repair_local_dataset

    root = _record(REPO_ID, video=True)
    shutil.rmtree(root / "meta" / "episodes")
    (root / "meta" / "info.json").unlink()

    with pytest.raises(DatasetRepairError, match="video metadata"):
        repair_local_dataset(REPO_ID)

    assert not (root / "meta" / "info.json").exists()


def test_forced_repair_restores_stale_info_and_episode_indices(lerobot_home: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from leforge.dataset_repair import repair_local_dataset_with_progress
    from lerobot.datasets import LeRobotDataset

    root = _record(REPO_ID, video=False)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] += 1
    info["total_frames"] += FRAMES
    info_path.write_text(json.dumps(info))

    episode_path = next((root / "meta" / "episodes").rglob("*.parquet"))
    episodes = pq.read_table(episode_path).to_pandas().iloc[1:].copy()
    episodes.loc[episodes.index[-1], "episode_index"] = EPISODES
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False), episode_path)

    repair_local_dataset_with_progress(REPO_ID, force=True)

    repaired = LeRobotDataset(REPO_ID)
    assert repaired.num_episodes == EPISODES
    assert repaired.num_frames == EPISODES * FRAMES
    assert sorted({int(value) for value in repaired.hf_dataset["episode_index"]}) == list(range(EPISODES))


def test_forced_repair_fixes_second_shard_shifted_from_episode_44(lerobot_home: Path) -> None:
    import pandas as pd

    from leforge.dataset_repair import repair_local_dataset_with_progress
    from leforge.datasets import analyze_dataset

    root = _record(REPO_ID, video=False, episodes=100)
    episode_path = next((root / "meta" / "episodes").rglob("*.parquet"))
    episodes = pd.read_parquet(episode_path)
    first = episodes.iloc[:43].copy()
    second = episodes.iloc[43:].copy()
    second["episode_index"] += 1
    second["meta/episodes/file_index"] = 1
    first.to_parquet(episode_path, index=False)
    second_path = episode_path.with_name("file-001.parquet")
    second.to_parquet(second_path, index=False)

    analysis = analyze_dataset(REPO_ID)
    assert "episode_metadata_mismatch" in {a["type"] for a in analysis["anomalies"]}
    assert analysis["statistics"]["total_frames"] == 100 * FRAMES

    repair_local_dataset_with_progress(REPO_ID, force=True)

    repaired = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    data = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    assert repaired["episode_index"].tolist() == list(range(100))
    assert repaired["length"].tolist() == data.groupby("episode_index").size().tolist()
    assert "episode_metadata_mismatch" not in {
        a["type"] for a in analyze_dataset(REPO_ID)["anomalies"]
    }


def test_analysis_detects_length_mismatch_with_unchanged_total_and_span(lerobot_home: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from leforge.datasets import analyze_dataset

    root = _record(REPO_ID, video=False, episodes=3)
    episode_path = next((root / "meta" / "episodes").rglob("*.parquet"))
    table = pq.read_table(episode_path)
    lengths = table.schema.field("length")
    table = table.set_column(
        table.schema.get_field_index("length"),
        lengths,
        pa.array([FRAMES, FRAMES - 2, FRAMES + 2], type=lengths.type),
    )
    pq.write_table(table, episode_path)

    analysis = analyze_dataset(REPO_ID)
    anomaly_types = {a["type"] for a in analysis["anomalies"]}

    assert "episode_metadata_mismatch" in anomaly_types
    assert "frame_count_mismatch" not in anomaly_types
    assert "index_range_mismatch" not in anomaly_types


def test_batch_video_encoding_keeps_episode_44_aligned_across_shards(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import pandas as pd
    from lerobot.datasets.dataset_writer import DatasetWriter
    from lerobot.datasets.io_utils import load_episodes

    class Meta(SimpleNamespace):
        def _close_writer(self):
            pass

    root = tmp_path / "batch"
    rows = []
    for ep in range(100):
        shard = 0 if ep < 43 else 1
        rows.append({
            "episode_index": ep,
            "length": FRAMES,
            "data/chunk_index": 0,
            "data/file_index": shard,
            "dataset_from_index": ep * FRAMES,
            "dataset_to_index": (ep + 1) * FRAMES,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": shard,
        })
    for shard in (0, 1):
        path = root / f"meta/episodes/chunk-000/file-{shard:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row for row in rows if row["meta/episodes/file_index"] == shard]).to_parquet(
            path, index=False
        )

    writer = DatasetWriter.__new__(DatasetWriter)
    writer._root = root
    writer._batch_encoding_size = 100
    writer._meta = Meta(
        episodes=load_episodes(root), total_episodes=100, video_keys=["cam"], latest_episode=None
    )

    def fake_save(video_key: str, ep: int) -> dict:
        return {
            "episode_index": ep,
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
            f"videos/{video_key}/from_timestamp": ep * 0.2,
            f"videos/{video_key}/to_timestamp": (ep + 1) * 0.2,
        }

    writer._save_episode_video = fake_save
    writer._batch_save_episode_video(0, 100)

    episodes = load_episodes(root)
    assert len(episodes) == 100
    assert episodes["episode_index"] == list(range(100))
    np.testing.assert_allclose(episodes["videos/cam/from_timestamp"], np.arange(100) * 0.2)
    np.testing.assert_allclose(episodes["videos/cam/to_timestamp"], np.arange(1, 101) * 0.2)


def test_batched_recording_survives_repeated_metadata_flushes(lerobot_home: Path) -> None:
    import pandas as pd
    from lerobot.datasets import LeRobotDataset

    root = _record(
        REPO_ID,
        video=True,
        episodes=6,
        batch_encoding_size=2,
        metadata_buffer_size=3,
    )

    episodes = pd.concat(
        [pd.read_parquet(path) for path in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    assert episodes["episode_index"].tolist() == list(range(6))
    assert episodes["length"].tolist() == [FRAMES] * 6
    assert len(list((root / "meta" / "episodes").rglob("*.parquet"))) == 3
    np.testing.assert_allclose(
        np.asarray(episodes["videos/observation.images.cam/from_timestamp"], dtype=np.float64),
        np.arange(6) * FRAMES / FPS,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(episodes["videos/observation.images.cam/to_timestamp"], dtype=np.float64),
        np.arange(1, 7) * FRAMES / FPS,
        rtol=1e-5,
        atol=1e-5,
    )

    repaired = LeRobotDataset(REPO_ID)
    assert repaired.num_episodes == 6
    assert repaired.num_frames == 6 * FRAMES
    last = repaired[repaired.num_frames - 1]
    assert int(last["episode_index"]) == 5
    assert tuple(last["observation.images.cam"].shape) == (3, 32, 32)


def test_failed_repair_verification_is_not_reported_as_success(
    lerobot_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from leforge.dataset_repair import DatasetRepairError, repair_local_dataset

    root = _record(REPO_ID, video=False)
    shutil.rmtree(root / "meta" / "episodes")
    monkeypatch.setattr("leforge.dataset_repair._write_episodes", lambda root, rows: None)

    with pytest.raises(DatasetRepairError, match="Repair verification failed"):
        repair_local_dataset(REPO_ID)


def test_repair_is_idempotent(lerobot_home: Path) -> None:
    from leforge.dataset_repair import repair_local_dataset

    root = _record(REPO_ID, video=False)
    shutil.rmtree(root / "meta" / "episodes")

    assert repair_local_dataset(REPO_ID) is not None
    assert repair_local_dataset(REPO_ID) is None


def test_truncated_data_file_is_moved_out_of_the_readers_way(lerobot_home: Path) -> None:
    from leforge.dataset_repair import repair_local_dataset
    from lerobot.datasets import LeRobotDataset

    root = _record(REPO_ID, video=False)
    shutil.rmtree(root / "meta" / "episodes")

    # A file the writer was still appending to when the process died: bytes
    # present, parquet footer missing.
    orphan = root / "data" / "chunk-000" / "file-001.parquet"
    orphan.write_bytes(b"PAR1\x00\x01\x02 no footer here")

    assert repair_local_dataset(REPO_ID) is not None
    assert not orphan.exists()
    assert orphan.with_suffix(".parquet.unreadable").exists()
    assert LeRobotDataset(REPO_ID).num_episodes == EPISODES


def test_reindex_sets_aside_a_shard_the_writer_never_closed(lerobot_home: Path) -> None:
    """An interrupted recording leaves a footer-less shard behind.

    Reindexing has to heal the dataset anyway: refusing because one trailing file
    is unreadable is what turned a crash into a permanently unresumable dataset.
    """
    from lerobot.datasets import LeRobotDataset
    from leforge.dataset_repair import reindex_episodes_to_contiguous

    root = _record(REPO_ID, video=False, episodes=3)
    orphan = root / "data" / "chunk-000" / "file-001.parquet"
    orphan.write_bytes(b"PAR1\x00\x01\x02 no footer here")

    reindex_episodes_to_contiguous(root)

    assert not orphan.exists()
    assert orphan.with_suffix(".parquet.unreadable").exists()
    assert LeRobotDataset(REPO_ID).num_episodes == 3


def test_episodes_without_video_lose_their_frames_and_stats_too(lerobot_home: Path) -> None:
    """An episode the videos don't cover is dropped from the index, so its rows
    and its contribution to stats.json have to go with it."""
    from leforge.dataset_repair import repair_local_dataset
    from lerobot.datasets import LeRobotDataset

    broken = _record(REPO_ID, video=True, episodes=2)
    # Same recording, one episode shorter: stands in both for the video that
    # was never flushed and for the statistics the repair should end up with.
    reference = _record("repair_test/reference", video=True, episodes=1)

    shutil.rmtree(broken / "meta" / "episodes")
    shutil.copy(
        next((reference / "videos").rglob("*.mp4")),
        next((broken / "videos").rglob("*.mp4")),
    )

    message = repair_local_dataset(REPO_ID)
    assert message is not None
    assert "Recovered 1 episode(s)" in message

    repaired = LeRobotDataset(REPO_ID)
    assert repaired.num_episodes == 1
    assert repaired.num_frames == FRAMES
    # The dropped episode's rows are gone from the shards, not just unreferenced.
    assert len(repaired.hf_dataset) == FRAMES

    repaired_stats = json.loads((broken / "meta" / "stats.json").read_text())
    reference_stats = json.loads((reference / "meta" / "stats.json").read_text())
    assert repaired_stats["action"]["max"] == reference_stats["action"]["max"]
    assert repaired_stats["action"]["mean"] == reference_stats["action"]["mean"]


def test_unrecoverable_dataset_reports_instead_of_hitting_the_hub(lerobot_home: Path) -> None:
    from leforge.dataset_repair import DatasetRepairError, repair_local_dataset

    root = _record(REPO_ID, video=False)
    shutil.rmtree(root / "meta" / "episodes")
    for path in (root / "data").rglob("*.parquet"):
        path.write_bytes(b"PAR1 truncated")

    with pytest.raises(DatasetRepairError, match="Re-record it"):
        repair_local_dataset(REPO_ID)


def test_deleting_several_episodes_rebuilds_once(lerobot_home: Path) -> None:
    """The rebuild re-encodes the whole dataset, so N deletions must share one pass."""
    from leforge.dataset_repair import delete_episodes_via_rebuild
    from lerobot.datasets import LeRobotDataset

    root = _record(REPO_ID, video=True, episodes=4)

    msg = delete_episodes_via_rebuild(root, REPO_ID, [1, 3])

    rebuilt = LeRobotDataset(REPO_ID)
    assert rebuilt.num_episodes == 2
    assert [int(e["episode_index"]) for e in rebuilt.meta.episodes] == [0, 1]
    assert (root.parent / (root.name + "__bak_del1-3")).exists(), "backup of the original is kept"
    assert "2 episode" in msg


def test_deleting_a_missing_episode_aborts_before_touching_the_dataset(
    lerobot_home: Path,
) -> None:
    """One bad index in the selection must not cost the whole rebuild."""
    from leforge.dataset_repair import DatasetRepairError, delete_episodes_via_rebuild

    root = _record(REPO_ID, video=False, episodes=3)

    with pytest.raises(DatasetRepairError, match=r"\[9\] not found"):
        delete_episodes_via_rebuild(root, REPO_ID, [1, 9])

    assert (root / "meta" / "info.json").exists()
    assert list(root.parent.glob(root.name + "__bak_*")) == []


def test_deleting_nothing_is_an_error(lerobot_home: Path) -> None:
    from leforge.dataset_repair import DatasetRepairError, delete_episodes_via_rebuild

    root = _record(REPO_ID, video=False, episodes=2)

    with pytest.raises(DatasetRepairError, match="No episodes selected"):
        delete_episodes_via_rebuild(root, REPO_ID, [])


def test_deleting_from_a_merged_dataset_survives_merge_bookkeeping(lerobot_home: Path) -> None:
    """lerobot's merge leaves the episode index inconsistent, in two fatal ways.

    - `meta/episodes/{chunk,file}_index` is copied from the source datasets, so rows
      name shards the merged set never wrote and the editor opens a missing path;
    - a feature stripped to line the sources up keeps its `stats/<feature>/*`
      columns, and the rows that never carried it flush as a `null` column type,
      which no longer matches the file the first batch created.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from leforge.dataset_repair import delete_episodes_via_rebuild

    root = _record(REPO_ID, video=False, episodes=4)
    shard = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(shard)

    field = table.column_names.index("meta/episodes/file_index")
    table = table.set_column(
        field,
        table.schema.field(field),
        pa.array([2, 0, 2, 0], type=pa.int64()),
    )
    table = table.append_column(
        pa.field("stats/intervention/min", pa.list_(pa.float64())),
        pa.array([[0.0, 1.0], None, [0.0, 1.0], None], type=pa.list_(pa.float64())),
    )
    pq.write_table(table, shard)

    msg = delete_episodes_via_rebuild(root, REPO_ID, [2])

    assert "4 -> 3" in msg
    rebuilt = pq.read_table(shard)
    assert "stats/intervention/min" not in rebuilt.column_names
    assert set(rebuilt["meta/episodes/file_index"].to_pylist()) == {0}
    assert set(rebuilt["meta/episodes/chunk_index"].to_pylist()) == {0}
