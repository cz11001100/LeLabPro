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

"""Closed-loop simulation evaluation: lerobot-eval + EnvHub (MuJoCo).

Runs in a subprocess because rendering needs its own process. The tabletop
scene is vendored in this repo under ``sim/so101-tabletop-mujoco`` and is the
default environment (no Hub download); a Hub repo id or local path overrides it.
MuJoCo renders offscreen through EGL by default, so no X display is needed;
set MUJOCO_GL=glfw to render on a display instead.
Scores rank checkpoints against each other; acceptance stays on the real robot.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ENV_HUB = "lerobot/so101-tabletop-mujoco"

# Values meaning "use the scene vendored in this repo" instead of a Hub download.
BUILTIN_ALIASES = {"", "builtin", "local", "内置"}


def builtin_scene_dir() -> Path | None:
    """The vendored tabletop scene, if this checkout (or ~/LeLabPro) carries it."""
    for root in (Path(__file__).resolve().parents[2], Path.home() / "LeLabPro"):
        scene = root / "sim" / "so101-tabletop-mujoco"
        if (scene / "env.py").is_file() and (scene / "so101_tabletop.xml").is_file():
            return scene
    return None


def resolve_env_path(value: str | None) -> str:
    """Empty/builtin aliases -> the vendored scene; anything else passes through."""
    value = (value or "").strip()
    if value in BUILTIN_ALIASES:
        scene = builtin_scene_dir()
        if scene is not None:
            return str(scene)
        logger.warning("内置仿真场景缺失，回退 Hub 环境 %s", DEFAULT_ENV_HUB)
        return DEFAULT_ENV_HUB
    return value

# The tabletop MJCF scene names its cameras top/front/wrist_cam; checkpoints
# recorded on the real rig use up/side/wrist. Map checkpoint names onto the
# scene so the env renders exactly the views the policy was trained on.
_TABLETOP_SCENE_CAMERAS = {"top", "front", "wrist_cam"}
_TABLETOP_CAMERA_ALIASES = {"up": "top", "side": "front", "wrist": "wrist_cam"}


def _make_env_kwargs(checkpoint_path: str, env_hub_path: str) -> dict:
    if "so101-tabletop-mujoco" not in env_hub_path:
        return {}
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    cameras = {}
    for key in (config.get("input_features") or {}):
        if not key.startswith("observation.images."):
            continue
        name = key.removeprefix("observation.images.")
        target = _TABLETOP_CAMERA_ALIASES.get(name, name)
        if target not in _TABLETOP_SCENE_CAMERAS:
            raise ValueError(
                f"仿真场景没有相机 {name}（可用: top/front/wrist_cam），请换用匹配的环境仓库"
            )
        cameras[name] = target
    return {"cameras": cameras} if cameras else {}

# tqdm writes "Stepping through eval batches:  40%|██| 2/5 [00:30<00:45, ...]"
_PROGRESS_RE = re.compile(r"(\d+)/(\d+)\s*\[")


def _kill_proc_tree(proc: subprocess.Popen, force: bool = False) -> None:
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(sig)


def parse_progress(log_path: Path) -> dict | None:
    """Batch progress from the eval tqdm line, read off the log tail."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = _PROGRESS_RE.findall(tail)
    if not matches:
        return None
    done, total = matches[-1]
    return {"batches_done": int(done), "batches_total": int(total)}


def run_sim_eval(
    *,
    checkpoint_path: str,
    run_dir: str,
    env_hub_path: str | None = None,
    n_episodes: int = 10,
    batch_size: int = 1,
    seed: int = 1000,
    max_episode_steps: int | None = None,
    cancel_event=None,
) -> dict:
    """Run lerobot-eval once; results and videos land under ``run_dir``.

    ``run_dir`` is ``<job_dir>/sim_eval/<timestamp>`` so video paths can be
    served relative to the job directory, like the offline evaluator's charts.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    job_dir = run_dir.parent.parent
    log_path = run_dir / "log.txt"
    env_hub_path = resolve_env_path(env_hub_path)

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={checkpoint_path}",
        "--env.type=envhub",
        f"--env.hub_path={env_hub_path}",
        "--trust_remote_code=true",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--output_dir={run_dir}",
        f"--seed={seed}",
    ]
    if max_episode_steps:
        cmd.append(f"--eval.max_episode_steps={max_episode_steps}")
    env_kwargs = _make_env_kwargs(checkpoint_path, env_hub_path)
    if env_kwargs:
        cmd.append(f"--env.make_env_kwargs={json.dumps(env_kwargs)}")

    env = os.environ.copy()
    # EGL renders offscreen without an X display; glfw is the opt-in fallback.
    env.setdefault("MUJOCO_GL", "egl")
    if env["MUJOCO_GL"] == "glfw":
        env.setdefault("DISPLAY", ":0")
    env["PYTHONUNBUFFERED"] = "1"

    logger.info("Sim eval: %s", " ".join(cmd))
    with open(log_path, "wb") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(run_dir),
            start_new_session=True,
        )
    (run_dir / "proc.pid").write_text(str(proc.pid))

    cancelled = False
    cancel_sent_at: float | None = None
    while proc.poll() is None:
        if cancel_event is not None and cancel_event.is_set():
            if cancel_sent_at is None:
                cancelled = True
                cancel_sent_at = time.time()
                _kill_proc_tree(proc)
            elif time.time() - cancel_sent_at > 5:
                _kill_proc_tree(proc, force=True)
        time.sleep(0.5)
    exit_code = proc.returncode

    rel_run = run_dir.relative_to(job_dir)
    if cancelled:
        return {"cancelled": True, "exit_code": exit_code, "run_dir": str(rel_run)}

    info_path = run_dir / "eval_info.json"
    if exit_code == 0 and info_path.is_file():
        try:
            info = json.loads(info_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "success": False,
                "exit_code": exit_code,
                "error": f"eval_info.json 解析失败: {exc}",
                "run_dir": str(rel_run),
            }
        videos = []
        for p in info.get("overall", {}).get("video_paths", []):
            try:
                rel = Path(p).resolve().relative_to(job_dir.resolve())
            except ValueError:
                continue
            if (job_dir / rel).is_file():
                videos.append(str(rel))
        return {
            "success": True,
            "exit_code": 0,
            "overall": info.get("overall", {}),
            "per_group": info.get("per_group", {}),
            "videos": videos,
            "run_dir": str(rel_run),
        }

    try:
        tail = log_path.read_text(errors="replace")[-4000:]
    except OSError:
        tail = ""
    return {
        "success": False,
        "exit_code": exit_code,
        "error": f"lerobot-eval 退出码 {exit_code}",
        "log_tail": tail,
        "run_dir": str(rel_run),
    }