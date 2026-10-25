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

import logging
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
_score_lock = threading.Lock()


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
    image_key: str = "observation.images.up"
    device: str = "cuda"
    stride: int = DEFAULT_STRIDE
    episodes: list[int] | None = None
    head_mode: str = "sparse"


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
    """Local datasets that hold policy rollouts, with their scoring status."""
    root = _cache_root()
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for org_dir in sorted(root.iterdir()):
        if not org_dir.is_dir():
            continue
        for ds_dir in sorted(org_dir.iterdir()):
            name = ds_dir.name
            if not ds_dir.is_dir() or not name.startswith("rollout_"):
                continue
            if not (ds_dir / "meta" / "info.json").is_file():
                continue
            lengths = _episode_lengths(ds_dir)
            out.append(
                {
                    "repo_id": f"{org_dir.name}/{name}",
                    "episodes": len(lengths),
                    "frames": sum(lengths.values()),
                    "scored": (ds_dir / PROGRESS_FILENAME).is_file(),
                    "backends": [
                        b for b, f in PROGRESS_BY_BACKEND.items() if (ds_dir / f).is_file()
                    ],
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M", time.localtime(ds_dir.stat().st_mtime)
                    ),
                }
            )
    out.sort(key=lambda d: d["modified"] or "", reverse=True)
    return out


def handle_rollout_summary(repo_id: str, success_progress: float = DEFAULT_SUCCESS_PROGRESS) -> dict[str, Any]:
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

    column = "progress_sparse" if "progress_sparse" in table.columns else "progress"
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
    return {
        "available": True,
        "repo_id": repo_id,
        "column": column,
        "episodes": scored,
        "successes": successes,
        "success_rate": round(successes / scored, 4) if scored else None,
        "threshold": success_progress,
        "details": episodes,
    }


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
    tmp = root / f".{PROGRESS_FILENAME}.tmp"
    shutil.copyfile(src, tmp)
    tmp.replace(root / PROGRESS_FILENAME)
    return {"success": True, "message": f"已把 {backend} 的进度设为训练/评估使用的版本"}


def build_score_command(request: RolloutScoreRequest) -> list[str]:
    """The lerobot progress-scoring invocation for one rollout dataset.

    Each backend writes its own parquet so two scorings coexist for comparison;
    the file RA-BC reads is only touched by promote_progress()."""
    backend = request.backend if request.backend in PROGRESS_BY_BACKEND else DEFAULT_BACKEND
    model_path = request.reward_model_path or _default_model_path(backend)
    out_file = _dataset_root(request.dataset_repo_id) / PROGRESS_BY_BACKEND[backend]
    cmd = [
        sys.executable,
        "-m",
        f"lerobot.rewards.{backend}.compute_rabc_weights",
        f"--dataset-repo-id={request.dataset_repo_id}",
        f"--reward-model-path={model_path}",
        # The Robometer checkpoint ships with ALOHA's camera naming; without
        # this the scorer looks for `observation.images.top` and finds nothing.
        f"--image-key={request.image_key}",
        f"--device={request.device}",
        f"--stride={request.stride}",
        f"--head-mode={request.head_mode}",
        f"--output-path={out_file}",
    ]
    if request.episodes:
        cmd += ["--episodes"] + [str(e) for e in request.episodes]
    return cmd


def handle_rollout_score(request: RolloutScoreRequest) -> dict[str, Any]:
    """Score a rollout dataset in the background; the parquet lands in its root.

    Refuses to start a second job while one is running, and returns the command
    verbatim so the same job can be run by hand on a bigger GPU instead."""
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
            "message": "本地显存放不下 4B 进度模型，请在 4090 上执行返回的命令",
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

    def pump() -> None:
        code = proc.wait()
        handle.close()
        with _score_lock:
            _score_state.update({"running": False, "exit_code": code, "finished_at": time.time()})
        logger.info("Rollout scoring finished (rc=%s): %s", code, request.dataset_repo_id)

    threading.Thread(target=pump, name="rollout-score-watcher", daemon=True).start()
    return {
        "success": True,
        "message": "打分已开始",
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }


def handle_rollout_score_status() -> dict[str, Any]:
    with _score_lock:
        state = dict(_score_state)
    if state.get("started_at"):
        state["elapsed_s"] = round(time.time() - state["started_at"], 1)
    state.setdefault("running", False)
    return state


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
