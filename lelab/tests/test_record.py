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
"""Tests for lelab.record — request schemas and handler entry points."""

from __future__ import annotations

import pytest


def test_recording_request_rejects_missing_required_fields() -> None:
    from pydantic import ValidationError

    from lelab.record import RecordingRequest

    with pytest.raises(ValidationError):
        RecordingRequest()


def test_recording_status_handler_exposes_state_fields() -> None:
    from lelab.record import handle_recording_status

    result = handle_recording_status()
    assert isinstance(result, dict)
    # Pinning the exact keys so a rename in handle_recording_status surfaces here.
    assert "recording_active" in result
    assert "current_phase" in result
    assert "session_ended" in result
    assert "available_controls" in result


def test_handle_stop_recording_when_idle_returns_dict(tmp_lerobot_home) -> None:
    from lelab.record import handle_stop_recording

    result = handle_stop_recording()
    assert isinstance(result, dict)


def test_create_record_config_pins_dshow_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Windows, recording must use the DSHOW backend so a camera_index opens
    the same device /available-cameras enumerated (via pygrabber, DSHOW order).
    """
    import lelab.record as record
    from lerobot.cameras.configs import Cv2Backends

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(record, "setup_calibration_files", lambda leader, follower: ("leader", "follower"))

    request = record.RecordingRequest(
        leader_port="COM_LEADER",
        follower_port="COM_FOLLOWER",
        leader_config="leader",
        follower_config="follower",
        dataset_repo_id="user/dataset",
        single_task="pick up the cube",
        cameras={"wrist": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}},
    )

    config = record.create_record_config(request)
    assert config.robot.cameras["wrist"].backend == Cv2Backends.DSHOW


def test_build_camera_configs_uses_default_backend_when_unset() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "width": 640, "height": 480, "fps": 30}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.AVFOUNDATION
    assert configs["cam"].fourcc is None
    assert configs["cam"].index_or_path == 0


def test_build_camera_configs_passes_fourcc_through() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "fourcc": "MJPG"}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs["cam"].fourcc == "MJPG"


def test_build_camera_configs_explicit_backend_overrides_default() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "V4L2"}}
    configs = _build_camera_configs(cameras, Cv2Backends.AVFOUNDATION)

    assert configs["cam"].backend == Cv2Backends.V4L2


def test_build_camera_configs_invalid_backend_raises() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "opencv", "camera_index": 0, "backend": "NOPE"}}
    with pytest.raises(KeyError):
        _build_camera_configs(cameras, Cv2Backends.ANY)


def test_build_camera_configs_skips_non_opencv_type() -> None:
    from lelab.record import _build_camera_configs
    from lerobot.cameras.configs import Cv2Backends

    cameras = {"cam": {"type": "realsense", "camera_index": 0}}
    configs = _build_camera_configs(cameras, Cv2Backends.ANY)

    assert configs == {}


@pytest.fixture
def lerobot_cache(tmp_lerobot_home, monkeypatch: pytest.MonkeyPatch):
    """Also redirect the copies of HF_LEROBOT_HOME resolved at import time.

    `tmp_lerobot_home` only covers lelab's own paths, so `LeRobotDataset.create`
    and `handle_get_dataset_info` would otherwise read and write the real ~/.cache.
    """
    monkeypatch.setattr("lerobot.utils.constants.HF_LEROBOT_HOME", tmp_lerobot_home)
    monkeypatch.setattr("lerobot.datasets.dataset_metadata.HF_LEROBOT_HOME", tmp_lerobot_home)
    monkeypatch.setattr("lelab.record.HF_LEROBOT_HOME", tmp_lerobot_home)
    return tmp_lerobot_home


def _tiny_dataset(repo_id: str, episodes: int = 3, frames: int = 4):
    """Write a small finished, non-video dataset and return its root."""
    import numpy as np

    from lerobot.datasets import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
    }
    dataset = LeRobotDataset.create(repo_id, fps=10, features=features, use_videos=False)
    for episode in range(episodes):
        for frame in range(frames):
            values = np.full(2, episode * frames + frame, dtype=np.float32)
            dataset.add_frame({"action": values, "observation.state": values, "task": "push"})
        dataset.save_episode()
    dataset.finalize()
    return dataset.root


def test_resume_guard_accepts_a_consistent_dataset(lerobot_cache) -> None:
    from lelab.record import _assert_resume_index_safe

    root = _tiny_dataset("guard_test/healthy")

    _assert_resume_index_safe(root)


def test_resume_guard_reindexes_when_data_disagrees_with_metadata(lerobot_cache) -> None:
    """The next index comes from the metadata, so metadata that looks clean while
    the data carries an index beyond it still collides on resume — that is what an
    episode whose rows landed but whose metadata row did not looks like."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from lelab.record import (
        _assert_resume_index_safe,
        _read_data_episode_indices,
        _read_resume_episode_indices,
    )

    root = _tiny_dataset("guard_test/data_ahead")
    shard = sorted((root / "data").rglob("*.parquet"))[0]
    table = pq.read_table(shard)
    field = table.schema.field("episode_index")
    values = table["episode_index"].to_pylist()
    bumped = [4 if v == max(values) else v for v in values]
    pq.write_table(
        table.set_column(table.schema.get_field_index("episode_index"), field, pa.array(bumped, type=field.type)),
        shard,
    )

    assert _read_resume_episode_indices(root) == [0, 1, 2]
    assert _read_data_episode_indices(root) == [0, 1, 4]

    _assert_resume_index_safe(root)

    assert _read_data_episode_indices(root) == [0, 1, 2]
    assert _read_resume_episode_indices(root) == [0, 1, 2]


def test_release_free_memory_returns_freed_pages_to_the_os() -> None:
    """The recording loop's real bug: Python frees each episode's frames but glibc
    keeps them in its heap, so RSS climbs until the kernel OOM-kills the session.

    Frame-sized blocks only: allocations large enough for glibc to mmap come back
    on their own, which would let this test pass without malloc_trim doing anything.
    """
    import gc
    import sys

    if not sys.platform.startswith("linux"):
        pytest.skip("malloc_trim is a glibc facility")

    import psutil

    from lelab.record import _release_free_memory

    process = psutil.Process()
    blocks = [bytearray(64 * 1024) for _ in range(9000)]
    for block in blocks:
        for offset in range(0, len(block), 4096):
            block[offset] = 1
    resident = process.memory_info().rss

    del blocks
    gc.collect()
    assert process.memory_info().rss > resident - 100 * 1024 * 1024, "gc alone freed the pages; not testing trim"

    _release_free_memory()

    assert process.memory_info().rss < resident - 200 * 1024 * 1024


def test_camera_reads_retry_and_name_the_stalling_camera(caplog) -> None:
    """A UVC hiccup has to cost a pause, not the episode in flight — and the log
    has to name the camera, because lerobot only reports the device index."""
    import logging

    from lelab.record import _harden_camera_reads

    class StallingCam:
        index_or_path = 2

        def __init__(self) -> None:
            self.calls = 0

        def async_read(self, timeout_ms=200):
            self.calls += 1
            if self.calls < 3:
                raise TimeoutError("Timed out waiting for frame from camera OpenCVCamera(2) after 200 ms")
            return "frame"

        def read_latest(self, max_age_ms=500):
            raise AssertionError("read_latest must not be needed while retries succeed")

    cam = StallingCam()

    class Robot:
        cameras = {"observation.images.side": cam}

    _harden_camera_reads(Robot())

    with caplog.at_level(logging.WARNING, logger="lelab.record"):
        assert cam.async_read() == "frame"

    assert cam.calls == 3
    assert "observation.images.side" in caplog.text
    assert "device 2" in caplog.text


def test_camera_reads_reuse_a_buffered_frame_before_losing_the_session(caplog) -> None:
    """A stall past the retry budget should cost one duplicated frame, not the run."""
    import logging

    from lelab.record import _CAMERA_READ_TIMEOUTS_MS, _harden_camera_reads

    class BlankingCam:
        index_or_path = 4

        def __init__(self) -> None:
            self.async_calls = 0

        def async_read(self, timeout_ms=200):
            self.async_calls += 1
            raise TimeoutError("Timed out waiting for frame from camera OpenCVCamera(4)")

        def read_latest(self, max_age_ms=500):
            return "buffered frame"

    cam = BlankingCam()

    class Robot:
        cameras = {"observation.images.up": cam}

    _harden_camera_reads(Robot())

    with caplog.at_level(logging.WARNING, logger="lelab.record"):
        assert cam.async_read() == "buffered frame"

    assert cam.async_calls == len(_CAMERA_READ_TIMEOUTS_MS)
    assert "reused its buffered frame" in caplog.text


def test_camera_reads_give_up_when_the_camera_has_nothing_at_all() -> None:
    """A camera that cannot even produce a stale frame is gone: end the session
    rather than record minutes of frozen footage."""
    from lelab.record import _CAMERA_READ_TIMEOUTS_MS, _harden_camera_reads

    class DeadCam:
        index_or_path = 2

        def __init__(self) -> None:
            self.calls = 0

        def async_read(self, timeout_ms=200):
            self.calls += 1
            raise TimeoutError("no frame ever")

        def read_latest(self, max_age_ms=500):
            raise TimeoutError("no buffered frame either")

    cam = DeadCam()

    class Robot:
        cameras = {"observation.images.up": cam}

    _harden_camera_reads(Robot())

    with pytest.raises(TimeoutError):
        cam.async_read()

    assert cam.calls == len(_CAMERA_READ_TIMEOUTS_MS)


def test_dataset_info_prefers_the_disk_copy_over_a_failed_session(lerobot_cache, monkeypatch) -> None:
    """A failed recording session leaves an error snapshot with no num_episodes.

    Serving that snapshot for a dataset which reads fine on disk made the frontend
    hide the whole Episodes panel, so the user could no longer pick episodes.
    """
    import lelab.record as record
    from lelab.record import DatasetInfoRequest, handle_get_dataset_info

    _tiny_dataset("info_test/on_disk", episodes=4, frames=6)
    monkeypatch.setattr(
        record,
        "last_recording_info",
        {
            "success": False,
            "error": "Column 91 named stats/observation.images.up/min expected length 10 but got length 9",
            "dataset_repo_id": "info_test/on_disk",
            "saved_episodes": 3,
        },
    )

    info = handle_get_dataset_info(DatasetInfoRequest(dataset_repo_id="info_test/on_disk"))

    assert info["success"] is True
    assert info["num_episodes"] == 4
    # The UI needs each episode's real length to pick which ones to delete;
    # total_frames / num_episodes labels every row identically.
    assert info["episodes"] == [{"index": i, "length": 6} for i in range(4)]


def test_dataset_info_falls_back_to_the_session_snapshot(lerobot_cache, monkeypatch) -> None:
    """A session whose dataset is not on disk yet must still report what it has."""
    import lelab.record as record
    from lelab.record import DatasetInfoRequest, handle_get_dataset_info

    monkeypatch.setattr(
        record,
        "last_recording_info",
        {"success": True, "dataset_repo_id": "info_test/not_yet", "num_episodes": 2},
    )

    info = handle_get_dataset_info(DatasetInfoRequest(dataset_repo_id="info_test/not_yet"))

    assert info["num_episodes"] == 2
