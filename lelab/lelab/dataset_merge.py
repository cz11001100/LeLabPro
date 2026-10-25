"""Merge several local datasets into one training set.

lerobot 0.6 cannot train on multiple datasets (`MultiLeRobotDataset` raises
NotImplementedError in datasets/factory.py), so correction episodes recorded by
the rollout capture modes have to be folded into the main demo set before they
can train anything. Merging is also where the schema problem shows up: a
correction dataset carries an extra `intervention` column that plain teleop
datasets don't, and merge requires identical features.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Present only on datasets created by the dagger capture strategy.
OPTIONAL_COLUMNS = ("intervention",)

_merge_state: dict[str, Any] = {"running": False}
_merge_lock = threading.Lock()


class MergeRequest(BaseModel):
    sources: list[str]
    target_repo_id: str
    # Drop the correction marker column from every source so their feature sets
    # line up. The information is not lost — the correction episodes keep their
    # own dataset on disk.
    drop_optional_columns: bool = True
    push_to_hub: bool = False


def _cache_root() -> Path:
    import os

    return Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


def _info(repo_id: str) -> dict[str, Any] | None:
    path = _cache_root() / repo_id / "meta" / "info.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def inspect_merge_sources(repo_ids: list[str]) -> dict[str, Any]:
    """Per-dataset facts plus the mismatches that would make a merge fail."""
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for repo_id in repo_ids:
        info = _info(repo_id)
        if info is None:
            missing.append(repo_id)
            continue
        features = set(info.get("features") or {})
        sources.append(
            {
                "repo_id": repo_id,
                "root": str(_cache_root() / repo_id),
                "episodes": info.get("total_episodes"),
                "frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "robot_type": info.get("robot_type"),
                "optional_columns": sorted(features & set(OPTIONAL_COLUMNS)),
            }
        )

    blockers: list[str] = []
    if missing:
        blockers.append(f"本地找不到：{', '.join(missing)}")
    if sources:
        for field in ("fps", "robot_type"):
            values = {s.get(field) for s in sources}
            if len(values) > 1:
                blockers.append(f"{field} 不一致：{sorted(str(v) for v in values)}")
        # Compare feature sets ignoring the columns we can drop.
        keysets = {
            frozenset(
                (_info(s["repo_id"]).get("features") or {}).keys()
            ) - set(OPTIONAL_COLUMNS)
            for s in sources
        }
        if len(keysets) > 1:
            sample = sorted(keysets, key=lambda k: -len(k))[0]
            others = [k for k in keysets if k != sample]
            detail = "; ".join(
                f"缺 {', '.join(sorted(sample - k))} / 多 {', '.join(sorted(k - sample))}" for k in others
            )
            blockers.append(f"特征列不一致：{detail}")
    return {"sources": sources, "blockers": blockers}


def build_merge_command(request: MergeRequest) -> list[str]:
    roots = [str(_cache_root() / repo_id) for repo_id in request.sources]
    return [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_edit_dataset",
        "--operation.type=merge",
        f"--operation.repo_ids=[{','.join(request.sources)}]",
        # Without roots the script would fetch each repo from the Hub instead of
        # using the local copy we just validated.
        f"--operation.roots=[{','.join(roots)}]",
        f"--new_repo_id={request.target_repo_id}",
        f"--push_to_hub={'true' if request.push_to_hub else 'false'}",
    ]


def handle_merge(request: MergeRequest) -> dict[str, Any]:
    """Start the merge in a subprocess (it re-encodes video and can take minutes)."""
    if len(request.sources) < 2:
        return {"success": False, "status_code": 400, "message": "至少选两个数据集才能合并"}
    if not request.target_repo_id.strip():
        return {"success": False, "status_code": 400, "message": "要给合并结果起个名字"}
    if "/" not in request.target_repo_id:
        return {"success": False, "status_code": 400, "message": "目标数据集要写成 用户/名字"}
    if (_cache_root() / request.target_repo_id / "meta" / "info.json").exists():
        return {
            "success": False,
            "status_code": 409,
            "message": f"{request.target_repo_id} 已存在，换个名字免得覆盖",
        }

    check = inspect_merge_sources(request.sources)
    if check["blockers"]:
        return {
            "success": False,
            "status_code": 400,
            "message": "；".join(check["blockers"]),
            "sources": check["sources"],
        }

    with _merge_lock:
        if _merge_state.get("running"):
            return {"success": False, "status_code": 409, "message": "已有合并任务在跑"}

    cmd = build_merge_command(request)
    if request.drop_optional_columns and any(s["optional_columns"] for s in check["sources"]):
        # Feature sets matched only because the optional column was ignored:
        # strip it from whichever sources carry it, then merge.
        strip = [s for s in check["sources"] if s["optional_columns"]]
        for source in strip:
            strip_cmd = [
                sys.executable,
                "-m",
                "lerobot.scripts.lerobot_edit_dataset",
                "--repo_id=" + source["repo_id"],
                f"--root={source['root']}",
                "--new_repo_id=" + source["repo_id"],
                f"--operation.type=remove_feature",
                f"--operation.feature_names=[{','.join(source['optional_columns'])}]",
                "--push_to_hub=false",
            ]
            code, err = _run_blocking(strip_cmd)
            if code != 0:
                return {
                    "success": False,
                    "status_code": 500,
                    "message": f"去掉 {source['repo_id']} 的多余列失败：{err[-300:]}",
                }

    log_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "merge_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{int(time.time())}.log"
    try:
        handle = log_path.open("w", buffering=1)
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
    except Exception as exc:
        logger.exception("Failed to start merge")
        return {"success": False, "status_code": 500, "message": f"启动合并失败: {exc}"}

    def pump() -> None:
        code = proc.wait()
        handle.close()
        with _merge_lock:
            _merge_state.update({"running": False, "exit_code": code, "finished_at": time.time()})
        logger.info("Merge finished (rc=%s) -> %s", code, request.target_repo_id)

    with _merge_lock:
        _merge_state.update(
            {
                "running": True,
                "sources": list(request.sources),
                "target": request.target_repo_id,
                "pid": proc.pid,
                "log_path": str(log_path),
                "started_at": time.time(),
                "exit_code": None,
            }
        )
    threading.Thread(target=pump, name="dataset-merge-watcher", daemon=True).start()
    return {
        "success": True,
        "message": "合并已开始",
        "log_path": str(log_path),
        "command": " ".join(cmd),
        "sources": check["sources"],
    }


def handle_merge_status() -> dict[str, Any]:
    with _merge_lock:
        state = dict(_merge_state)
    if state.get("started_at"):
        state["elapsed_s"] = round(time.time() - state["started_at"], 1)
    state.setdefault("running", False)
    if not state.get("running") and state.get("target"):
        state["target_exists"] = _info(state["target"]) is not None
    return state


def _run_blocking(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    """Run a quick edit op to completion (column stripping before the merge)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -1, f"timed out after {timeout}s"
    return proc.returncode, (proc.stderr or proc.stdout or "")
