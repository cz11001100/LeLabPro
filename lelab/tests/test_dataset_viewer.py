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
"""Tests for the local dataset viewer helpers and sim-eval plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _make_v3_dataset(root: Path, repo_id: str) -> Path:
    d = root / repo_id
    (d / "meta").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    (d / "videos" / "observation.images.UP" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": ["j0.pos", "j1.pos"]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "observation.images.UP": {"dtype": "video", "shape": [480, 640, 3]},
        },
    }
    (d / "meta" / "info.json").write_text(json.dumps(info))
    episodes = [
        {
            "episode_index": 0,
            "tasks": ["grasp the ball"],
            "length": 4,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.UP/chunk_index": 0,
            "videos/observation.images.UP/file_index": 0,
            "videos/observation.images.UP/from_timestamp": 0.0,
            "videos/observation.images.UP/to_timestamp": 4 / 30,
        },
        {
            "episode_index": 1,
            "tasks": ["grasp the ball"],
            "length": 2,
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.UP/chunk_index": 0,
            "videos/observation.images.UP/file_index": 0,
            "videos/observation.images.UP/from_timestamp": 4 / 30,
            "videos/observation.images.UP/to_timestamp": 6 / 30,
        },
    ]
    (d / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in episodes))
    rows = []
    for ep, length in ((0, 4), (1, 2)):
        for fr in range(length):
            rows.append(
                {
                    "episode_index": ep,
                    "frame_index": fr,
                    "action": [float(ep + fr), float(fr)],
                    "observation.state": [float(fr), float(ep)],
                }
            )
    pd.DataFrame(rows).to_parquet(d / "data" / "chunk-000" / "file-000.parquet")
    (d / "videos" / "observation.images.UP" / "chunk-000" / "file-000.mp4").write_bytes(b"")
    return d


def test_dataset_episodes_plain_json(tmp_lerobot_home: Path) -> None:
    from lelab.datasets import dataset_episodes

    _make_v3_dataset(tmp_lerobot_home, "user/ds")
    out = dataset_episodes("user/ds")
    assert out["fps"] == 30.0
    assert out["cameras"] == ["observation.images.UP"]
    assert [e["episode_index"] for e in out["episodes"]] == [0, 1]
    assert out["episodes"][0]["tasks"] == ["grasp the ball"]
    json.dumps(out)


def test_dataset_video_window(tmp_lerobot_home: Path) -> None:
    from lelab.datasets import dataset_video_info

    _make_v3_dataset(tmp_lerobot_home, "user/ds")
    info = dataset_video_info("user/ds", 1, "observation.images.UP")
    assert info["path"].name == "file-000.mp4"
    assert info["from_timestamp"] == pytest.approx(4 / 30)
    assert info["to_timestamp"] == pytest.approx(6 / 30)
    with pytest.raises(FileNotFoundError):
        dataset_video_info("user/ds", 99, "observation.images.UP")
    with pytest.raises(FileNotFoundError):
        dataset_video_info("user/ds", 0, "observation.images.NOPE")


def test_dataset_series_json_serializable(tmp_lerobot_home: Path) -> None:
    from lelab.datasets import dataset_episode_series

    _make_v3_dataset(tmp_lerobot_home, "user/ds")
    out = dataset_episode_series("user/ds", 0, max_points=10)
    assert out["joint_names"] == ["j0", "j1"]
    assert out["frames"] == [0, 1, 2, 3]
    assert out["action"][3] == [3.0, 3.0]
    assert out["state"][0] == [0.0, 0.0]
    json.dumps(out)


def test_dataset_root_traversal_rejected(tmp_lerobot_home: Path) -> None:
    from lelab.datasets import dataset_episodes

    _make_v3_dataset(tmp_lerobot_home, "user/ds")
    with pytest.raises(FileNotFoundError):
        dataset_episodes("../ds")


def test_sim_eval_camera_aliases(tmp_path: Path) -> None:
    from lelab.sim_eval import _make_env_kwargs

    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps(
            {
                "input_features": {
                    "observation.state": {},
                    "observation.images.up": {},
                    "observation.images.side": {},
                    "observation.images.wrist": {},
                }
            }
        )
    )
    kwargs = _make_env_kwargs(str(ckpt), "lerobot/so101-tabletop-mujoco")
    assert kwargs == {"cameras": {"up": "top", "side": "front", "wrist": "wrist_cam"}}
    # Other env repos get no tabletop-specific mapping.
    assert _make_env_kwargs(str(ckpt), "someone/other-env") == {}


def test_sim_eval_unknown_camera_rejected(tmp_path: Path) -> None:
    from lelab.sim_eval import _make_env_kwargs

    ckpt = tmp_path / "model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(
        json.dumps({"input_features": {"observation.images.ceiling": {}}})
    )
    with pytest.raises(ValueError, match="ceiling"):
        _make_env_kwargs(str(ckpt), "lerobot/so101-tabletop-mujoco")


def test_sim_eval_parse_progress(tmp_path: Path) -> None:
    from lelab.sim_eval import parse_progress

    log = tmp_path / "log.txt"
    log.write_text(
        "Running rollout with at most 600 steps:  40%|███| 240/600 [00:04<00:06, 59.9it/s]\r"
        "Running rollout with at most 600 steps:  65%|█████| 390/600 [00:06<00:03, 60.1it/s]"
    )
    assert parse_progress(log) == {"batches_done": 390, "batches_total": 600}
    assert parse_progress(tmp_path / "missing.txt") is None