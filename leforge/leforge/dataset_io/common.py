# Copyright 2026 The LeForge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared helpers for dataset format import/export.

Everything here is deliberately free of h5py/zarr imports so the module can be
imported even when an optional format dependency is missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class Episode:
    """One episode in a format-neutral shape.

    Images are RGB uint8 ``(H, W, 3)`` arrays in frame order. State/action are
    ``(T, D)`` float32 arrays.
    """

    index: int
    state: np.ndarray
    action: np.ndarray
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    task: str = ""
    fps: float | None = None

    @property
    def length(self) -> int:
        lengths = [len(self.state), len(self.action)]
        lengths += [len(v) for v in self.images.values()]
        return min(lengths) if lengths else 0

    def cropped(self) -> Episode:
        n = self.length
        return Episode(
            index=self.index,
            state=np.ascontiguousarray(np.asarray(self.state, dtype=np.float32)[:n]),
            action=np.ascontiguousarray(np.asarray(self.action, dtype=np.float32)[:n]),
            images={k: v[:n] for k, v in self.images.items()},
            task=self.task,
            fps=self.fps,
        )


def safe_repo_id(repo_id: str) -> str:
    repo_id = (repo_id or "").strip().strip("/")
    if not REPO_ID_RE.match(repo_id):
        raise ValueError(
            "数据集标识必须是 '<作者>/<名称>'，仅允许字母、数字、点、下划线和连字符，"
            f"当前为 {repo_id!r}"
        )
    return repo_id


def lerobot_home() -> Path:
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME).expanduser()


def dataset_dir(repo_id: str) -> Path:
    safe_repo_id(repo_id)
    root = lerobot_home().resolve()
    target = (root / repo_id).resolve()
    if not target.is_relative_to(root):
        raise ValueError("数据集路径越界")
    return target


def natural_key(value: str) -> list[Any]:
    """Split a string into numbers and text so episode_2 sorts before episode_10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def sanitize_name(name: str) -> str:
    """Camera/feature names may not contain '/' or leak path separators."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return out or "cam"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = sanitize_name(name)
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def decode_encoded_image(buffer: Any) -> np.ndarray:
    """Decode a JPEG/PNG byte blob into RGB uint8 (H, W, 3)."""
    import cv2

    arr = np.asarray(buffer)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    decoded = cv2.imdecode(np.frombuffer(arr.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("图像字节无法解码（不是有效的 JPEG/PNG）")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def to_rgb_uint8(value: Any) -> np.ndarray:
    """Normalize an encoded blob or an array (HWC/CHW, uint8/float) to RGB uint8."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return decode_encoded_image(value)

    arr = np.asarray(value)
    if arr.ndim == 1:
        if arr.dtype == np.uint8:
            return decode_encoded_image(arr)
        raise ValueError(f"无法把一维图像数组（dtype={arr.dtype}）识别为编码图像")

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3:
        # Channels-first inputs from torch/LeRobot are (C, H, W).
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] != 3:
            raise ValueError(f"不支持 {arr.shape[-1]} 通道图像，需要 RGB 或灰度")
    else:
        raise ValueError(f"不支持 {arr.ndim} 维图像数组: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size and float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def encode_jpeg(rgb: np.ndarray, quality: int = 95) -> bytes:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def as_float_matrix(value: Any, *, length: int | None = None) -> np.ndarray:
    """Coerce state/action data to a contiguous float32 (T, D) matrix."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if length is not None:
        arr = arr[:length]
    if not np.all(np.isfinite(arr)):
        raise ValueError("状态/动作数据包含 NaN 或 Inf")
    return np.ascontiguousarray(arr)


def read_hdf5_dataset(group: Any, key: str, *, length: int | None = None) -> np.ndarray:
    """Read a dataset by slash path and return it as a float32 (T, D) matrix."""
    if key not in group:
        raise KeyError(f"HDF5 中找不到字段 {key!r}")
    return as_float_matrix(group[key][...], length=length)


def load_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value
