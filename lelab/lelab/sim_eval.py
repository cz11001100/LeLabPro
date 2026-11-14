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


def available_environments() -> list[dict]:
    """Entries for the 环境仓库 picker: builtin scene, local scene dirs, Hub default."""
    entries = [{"label": "内置场景（sim/so101-tabletop-mujoco）", "value": "", "kind": "builtin"}]
    repo_root = Path(__file__).resolve().parents[2]
    sim_root = repo_root / "sim"
    builtin = builtin_scene_dir()
    if sim_root.is_dir():
        for d in sorted(sim_root.iterdir()):
            if not (d.is_dir() and (d / "env.py").is_file()):
                continue
            if builtin is not None and d.resolve() == builtin.resolve():
                continue
            entries.append({"label": f"本地：{d.name}", "value": str(d), "kind": "local"})
    entries.append({"label": f"Hub：{DEFAULT_ENV_HUB}", "value": DEFAULT_ENV_HUB, "kind": "hub"})
    return entries


def _policy_type(checkpoint_path: str) -> str:
    try:
        cfg = json.loads((Path(checkpoint_path) / "config.json").read_text())
        return str(cfg.get("type") or "").lower()
    except (OSError, json.JSONDecodeError):
        return ""


def policy_supports_temporal_ensemble(checkpoint_path: str) -> bool:
    """Whether the checkpoint's policy implements temporal ensembling (ACT today).

    Checked against the policy config dataclass so future policies that add the
    knob work without changes here; unknown configs fall back to the type name.
    """
    ptype = _policy_type(checkpoint_path)
    if not ptype:
        return False
    try:
        import dataclasses

        from lerobot.configs.policies import PreTrainedConfig

        cls = PreTrainedConfig.get_choice_class(ptype)
        return "temporal_ensemble_coeff" in {f.name for f in dataclasses.fields(cls)}
    except Exception:
        return ptype == "act"

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


def _ball_centers_px(rgb) -> dict:
    """Red/blue ball centroids (pixels) in an up-camera frame; None if not visible."""
    import numpy as np

    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    out = {}
    for name, mask in (
        ("red", (r > 120) & (g < 90) & (b < 90)),
        ("blue", (b > 120) & (r < 90)),
    ):
        ys, xs = np.nonzero(mask)
        out[name] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)] if len(xs) > 20 else None
    return out


def diagnose_recordings(run_dir: Path, checkpoint_path: str) -> dict:
    """Did the arm react to the ball at all? Combine the eval recording with the
    checkpoint's training stats.

    ``--eval.recording=true`` saves every step's actions and camera frames as a
    LeRobot dataset under ``run_dir/recordings``. From it we compute:

    * per-episode ball positions (read off the first up frame),
    * cross-episode action-trajectory spread (normalized by the pooled action
      std): near zero means the arm replays the same motion regardless of where
      the balls are — the policy is effectively ignoring vision,
    * sim-vs-train domain gap per camera: mean/std shift of the recorded frames
      against the normalizer stats the policy was trained with.
    """
    import numpy as np

    root = run_dir / "recordings"
    # Layouts: recordings/env_i (single task) or recordings/<task>/env_i.
    dirs = sorted({p.parent for p in root.rglob("meta") if p.is_dir()}) if root.is_dir() else []
    if not dirs:
        return {"error": "没有录制数据（recordings 目录缺失）"}

    from lerobot.datasets import LeRobotDataset

    episodes: list[dict] = []
    trajs: list = []
    frames: dict[str, list] = {"up": [], "side": [], "wrist": []}
    ep_no = 0
    for d in dirs:
        ds = LeRobotDataset(repo_id="eval_recording", root=str(d))
        hf = ds.hf_dataset
        ep_ids = hf["episode_index"]
        for ep in range(int(max(ep_ids)) + 1):
            idxs = [i for i, e in enumerate(ep_ids) if e == ep]
            if not idxs:
                continue
            acts = np.asarray([hf[i]["action"] for i in idxs], dtype=np.float64)
            trajs.append(acts[:150])
            first = ds[idxs[0]]
            balls: dict = {}
            for cam in frames:
                key = f"observation.images.{cam}"
                if key not in first:
                    continue
                img = first[key]
                arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                if cam == "up":
                    balls = _ball_centers_px(arr)
                if len(frames[cam]) < 30:
                    frames[cam].append(arr)
            episodes.append({"episode": ep_no, "balls_up_px": balls, "steps": len(idxs)})
            ep_no += 1

    diag: dict = {"episodes": episodes}
    if len(trajs) >= 2:
        t = min(len(x) for x in trajs)
        a = np.stack([x[:t] for x in trajs])  # (E, T, 6)
        std = a.reshape(-1, a.shape[-1]).std(axis=0) + 1e-6
        an = (a - a.mean(axis=(0, 1))) / std
        pair = [
            float(np.linalg.norm(an[i] - an[j]) / np.sqrt(6 * t))
            for i in range(len(a))
            for j in range(i + 1, len(a))
        ]
        spread = float(np.mean(pair))
        within = float(
            np.mean([np.linalg.norm(np.diff(x, axis=0)) / np.sqrt(6 * max(t - 1, 1)) for x in an])
        )
        raw = float(
            np.mean(
                [
                    np.linalg.norm(a[i] - a[j]) / np.sqrt(6 * t)
                    for i in range(len(a))
                    for j in range(i + 1, len(a))
                ]
            )
        )
        diag.update(traj_spread_norm=round(spread, 3), traj_spread_deg=round(raw, 2), within_motion_norm=round(within, 3))
        if spread < 0.05:
            verdict = "各集轨迹几乎相同：机械臂动作对球的位置基本无响应，视觉未起作用"
        elif spread < 0.2:
            verdict = "各集轨迹差异很小：对球位置响应弱，视觉利用不足"
        else:
            verdict = "各集轨迹随画面变化，但需结合域差判断是否为有效追球"
    else:
        verdict = "只有一集录制，无法比较轨迹差异（建议 ≥3 集）"

    try:
        from safetensors.torch import load_file

        stats = load_file(
            str(Path(checkpoint_path) / "policy_preprocessor_step_3_normalizer_processor.safetensors")
        )
        gap = {}
        for cam, arrs in frames.items():
            mk, sk = f"observation.images.{cam}.mean", f"observation.images.{cam}.std"
            if not arrs or mk not in stats or sk not in stats:
                continue
            sim = np.stack(arrs).astype(np.float64) / 255.0
            sim_mean = sim.mean(axis=(0, 1, 2))
            sim_std = sim.std(axis=(0, 1, 2)) + 1e-6
            tm = stats[mk].numpy().flatten()
            ts = stats[sk].numpy().flatten() + 1e-6
            gap[cam] = {
                "mean_abs_shift": round(float(np.abs(sim_mean - tm).mean()), 3),
                "std_ratio": round(float((sim_std / ts).mean()), 2),
            }
        if gap:
            diag["domain_gap"] = gap
            worst = max(g["mean_abs_shift"] for g in gap.values())
            if worst > 0.2:
                verdict += (
                    f"；但仿真画面与训练画面差异大（最大均值差 {worst}），"
                    "策略在半盲状态下动作，轨迹变化不等于有效追球"
                )
            else:
                verdict += "；画面分布与训练接近，轨迹变化可视为对球位的响应"
    except Exception as exc:  # stats are a bonus, never fail the diagnosis over them
        logger.warning("Domain-gap stats unavailable: %s", exc)
    diag["verdict"] = verdict

    try:
        (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))
    except OSError as exc:
        logger.warning("Could not write diagnostics.json: %s", exc)
    return diag


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
    action_smooth: float = 0.0,
    temporal_ensemble: float | None = None,
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
        # Sync vector env: the async workers cannot render per camera, which
        # the multi-view eval videos need. MuJoCo stepping is cheap; policy
        # inference stays batched on GPU, so little is lost.
        "--eval.use_async_envs=false",
        f"--output_dir={run_dir}",
        f"--seed={seed}",
        # Record every step's actions + camera frames so the post-run diagnosis
        # can check whether the arm actually reacts to the ball layout.
        "--eval.recording=true",
    ]
    if max_episode_steps:
        cmd.append(f"--eval.max_episode_steps={max_episode_steps}")
    notes = ""
    if temporal_ensemble is not None:
        # Temporal ensembling is a per-policy capability (ACT today): re-predict
        # every step and weight-average chunk overlaps. Only pass the flags when
        # the checkpoint's config defines them; other policies keep their replay.
        if policy_supports_temporal_ensemble(checkpoint_path):
            cmd.append(f"--policy.temporal_ensemble_coeff={temporal_ensemble}")
            # ACT requires n_action_steps=1 when ensembling (query every step).
            if _policy_type(checkpoint_path) == "act":
                cmd.append("--policy.n_action_steps=1")
        else:
            notes = "该策略未实现时序集成（目前 ACT 支持），本次按 checkpoint 原生动作回放执行"
            logger.warning("temporal_ensemble requested but unsupported: %s", checkpoint_path)
    env_kwargs = _make_env_kwargs(checkpoint_path, env_hub_path)
    if max_episode_steps and "so101-tabletop-mujoco" in env_hub_path:
        # Let the env itself truncate at the cap: the recorder only flushes an
        # episode on env terminated/truncated, so an eval-side cap alone would
        # leave the recorded actions unwritten.
        env_kwargs["episode_length"] = int(max_episode_steps)
    if action_smooth > 0 and "so101-tabletop-mujoco" in env_hub_path:
        env_kwargs["action_smooth"] = float(action_smooth)
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
        try:
            diagnostics = diagnose_recordings(run_dir, checkpoint_path)
        except Exception as exc:
            logger.error("Sim eval diagnosis failed: %s", exc, exc_info=True)
            diagnostics = {"error": str(exc)}
        return {
            "success": True,
            "exit_code": 0,
            "overall": info.get("overall", {}),
            "per_group": info.get("per_group", {}),
            "videos": videos,
            "diagnostics": diagnostics,
            "notes": notes,
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
        "notes": notes,
        "run_dir": str(rel_run),
    }