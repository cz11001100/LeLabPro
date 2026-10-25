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
"""Video decoder selection that survives a broken torchcodec install.

lerobot picks "torchcodec" as soon as the *package* is importable, but its
native bindings dlopen FFmpeg 4-7 shared libraries at import time. On a box
without them the dataset still opens and then every video read raises a long
"Could not load libtorchcodec" RuntimeError, which looks like an evaluation
bug rather than a missing system package. Probing the import once lets us
decode with pyav instead.
"""

from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def probe_torchcodec() -> str | None:
    """Return None when torchcodec can actually decode, else the failure reason."""
    if importlib.util.find_spec("torchcodec") is None:
        return "torchcodec is not installed"
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception as exc:  # torchcodec raises RuntimeError with a multi-line traceback
        return str(exc).splitlines()[0].strip() or type(exc).__name__
    return None


@lru_cache(maxsize=1)
def video_backend() -> str:
    """Backend name to hand to LeRobotDataset(video_backend=...)."""
    reason = probe_torchcodec()
    if reason is None:
        return "torchcodec"
    if importlib.util.find_spec("av") is None:
        logger.warning("torchcodec unusable (%s) and pyav is not installed", reason)
        return "torchcodec"
    logger.warning(
        "torchcodec unusable (%s) — decoding videos with pyav. "
        "Install FFmpeg 4-7 (e.g. `sudo apt install ffmpeg`) to get torchcodec back.",
        reason,
    )
    return "pyav"
