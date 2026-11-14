"""Push a locally exported checkpoint to the Hugging Face Hub.

The point is not backup: a model on the Hub can be referenced by repo id, so a
pod pulls it through the mirror instead of receiving a 1.2 GB SFTP upload from
this machine before every training run.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"running": False}
_lock = threading.Lock()
_STATE_PATH = Path.home() / ".cache" / "huggingface" / "lerobot" / "model_push_state.json"


def _persist() -> None:
    import json

    with _lock:
        data = dict(_state)
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(_STATE_PATH)
    except OSError as exc:  # noqa: BLE001
        logger.info("Push state not persisted: %s", exc)


def _load() -> None:
    """A push left 'running' died with the previous process — say so."""
    import json

    try:
        data = json.loads(_STATE_PATH.read_text())
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        if data.get("running"):
            data.update({"running": False, "interrupted": True})
        _state.update(data)


_load()


class PushModelRequest(BaseModel):
    job_id: str
    # Empty = derive `<hf_user>/<job name>` on the server.
    repo_id: str = ""
    step: int | None = None
    # Optional per-request write token (the UI prompts when the machine has no
    # cached HF login). Never logged or persisted.
    token: str = ""


def push_status() -> dict[str, Any]:
    with _lock:
        state = dict(_state)
    if state.get("running") and state.get("started_at"):
        state["elapsed_s"] = round(time.time() - float(state["started_at"]), 1)
    return state


def _set(**fields: Any) -> None:
    with _lock:
        _state.update(fields)
    _persist()


def _local_dir_for(job_id: str, step: int | None) -> tuple[Path | None, str]:
    from .jobs import JobNotFoundError, job_registry

    try:
        checkpoints = job_registry.list_checkpoints(job_id)
    except JobNotFoundError:
        return None, f"任务 {job_id} 不存在"
    if not checkpoints:
        return None, "这个任务没有可导出的 checkpoint"
    chosen = checkpoints[-1]
    if step is not None:
        match = [c for c in checkpoints if c.step == step]
        if not match:
            return None, f"找不到 step {step}"
        chosen = match[0]
    path = Path(str(chosen.ref))
    try:
        if path.is_dir():
            return path, ""
    except OSError:
        pass  # pod paths (/root/...) are not traversable from this user
    # SSH checkpoints live on the pod; push from the exported local copy when
    # one exists, otherwise the user has to download it first.
    local = job_registry.local_checkpoint_dir(job_id, chosen.step)
    if local is not None:
        return local, ""
    return None, f"本地没有这个 checkpoint：{path}（先点下载把 checkpoint 导到本地，再上传）"


def _run(job_id: str, local_dir: Path, repo_id: str, token: str = "") -> None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token or None)
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
        with _lock:
            _state.setdefault("by_job", {})[job_id] = repo_id
            _state.update(running=False, done=True, error=None, finished_at=time.time())
        _persist()
        logger.info("Pushed %s to %s", local_dir, repo_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model push failed")
        _set(running=False, done=False, error=str(exc)[:400], finished_at=time.time())


def pushed_repo(job_id: str) -> str | None:
    """The Hub repo this job's checkpoint was pushed to, if any.

    Read from disk as well as memory: a push done in an earlier server process
    must still show up as a selectable starting point.
    """
    import json

    with _lock:
        hit = (_state.get("by_job") or {}).get(job_id)
    if hit:
        return hit
    try:
        data = json.loads(_STATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        return (data.get("by_job") or {}).get(job_id) or None
    return None


def _path_key(local_dir: Path) -> str:
    try:
        return str(local_dir.resolve())
    except OSError:
        return str(local_dir)


def pushed_repo_for_path(local_dir: Path) -> str | None:
    import json

    key = _path_key(local_dir)
    with _lock:
        hit = (_state.get("by_path") or {}).get(key)
    if hit:
        return hit
    try:
        data = json.loads(_STATE_PATH.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        return (data.get("by_path") or {}).get(key) or None
    return None


def ensure_hub_copy(local_dir: Path, name_hint: str) -> str | None:
    """Push a local checkpoint to the Hub once; later runs reuse the repo id.

    Pushing beats SFTP-to-pod: the Hub upload measured ~12 MB/s from this
    machine versus ~2 MB/s to the pod, and the pod then pulls it through the
    mirror (also once, since it caches). Returns None when the push fails so the
    caller can fall back to the direct transfer.
    """
    import re

    cached = pushed_repo_for_path(local_dir)
    if cached:
        return cached
    try:
        from huggingface_hub import HfApi

        from .jobs import _default_repo_id_for_name  # local import: avoids a cycle
    except Exception:  # noqa: BLE001
        return None
    try:
        repo_id = _default_repo_id_for_name(name_hint)
        if not repo_id:
            return None
        with _lock:
            _UPLOAD_STATE.clear()
            _UPLOAD_STATE.update(
                {
                    "active": True,
                    "phase": "uploading",
                    "sent": 0,
                    "total": sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file()),
                    "started_at": time.time(),
                    "name": repo_id,
                    "error": None,
                }
            )
        _persist()
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model")
        with _lock:
            _state.setdefault("by_path", {})[_path_key(local_dir)] = repo_id
            _state.update({"running": False, "done": True, "error": None})
        _persist()
        logger.info("Checkpoint %s published to %s", local_dir, repo_id)
        return repo_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hub copy failed (%s); falling back to SFTP upload", exc)
        _set(running=False, done=False, error=str(exc)[:300])
        return None


def _default_repo_id(job_id: str, token: str = "") -> str:
    """`<hf_user>/<job name>` — the same shape a manual push would use.

    `whoami` is a network call and was measured hanging for tens of seconds on
    this link, so it runs with a hard bound; the frontend normally supplies the
    repo id from its own auth state and this path isn't hit at all.
    """
    import re
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    from .jobs import job_registry

    def whoami() -> dict:
        if token:
            from huggingface_hub import HfApi

            return HfApi(token=token).whoami()
        from .utils.hf_auth import cached_whoami

        return cached_whoami() or {}

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        info = pool.submit(whoami).result(timeout=8) or {}
    except FutureTimeout:
        return ""
    except Exception:  # noqa: BLE001
        return ""
    finally:
        pool.shutdown(wait=False)
    user = info.get("name")
    if not user:
        return ""
    try:
        name = job_registry.get(job_id).name
    except Exception:  # noqa: BLE001
        name = "model"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower()
    return f"{user}/{slug or 'model'}"


def _remote_ckpt_dir(job_id: str, step: int | None) -> tuple[str | None, str, dict | None]:
    """(remote dir, problem, ssh connect kwargs) for a checkpoint on the pod."""
    import os

    from .jobs import JobNotFoundError, job_registry
    from .ssh_config import get_ssh_config

    try:
        record = job_registry.get(job_id)
    except JobNotFoundError:
        return None, f"任务 {job_id} 不存在", None
    if record.runner != "ssh" or not record.ssh_config_id:
        return None, "这个任务不在 SSH 远端", None
    cfg = get_ssh_config(record.ssh_config_id)
    if cfg is None:
        return None, "任务的 SSH 配置已删除", None
    try:
        checkpoints = job_registry.list_checkpoints(job_id)
    except JobNotFoundError:
        return None, f"任务 {job_id} 不存在", None
    if not checkpoints:
        return None, "这个任务没有可上传的 checkpoint", None
    chosen = checkpoints[-1]
    if step is not None:
        match = [c for c in checkpoints if c.step == step]
        if not match:
            return None, f"找不到 step {step}", None
        chosen = match[0]
    kwargs: dict = {
        "hostname": cfg.host,
        "port": cfg.port,
        "username": cfg.username,
        "timeout": 20,
    }
    if cfg.key_path:
        kwargs["key_filename"] = os.path.expanduser(cfg.key_path)
    elif cfg.password:
        kwargs["password"] = cfg.password
    # Not a paramiko kwarg: popped before connect; names the interpreter that
    # actually has huggingface_hub installed on the pod.
    kwargs["leforge_python"] = cfg.remote_python or "python3"
    return str(chosen.ref), "", kwargs


def _run_remote(job_id: str, connect_kwargs: dict, remote_dir: str, repo_id: str, token: str) -> None:
    """Upload straight from the pod to the Hub — no 1.2 GB local round-trip."""
    try:
        import paramiko

        api = paramiko.SSHClient()
        api.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        python = connect_kwargs.get("leforge_python") or "python3"
        api.connect(**{k: v for k, v in connect_kwargs.items() if k != "leforge_python"})
        try:
            # The token travels over the encrypted stdin channel, never on the
            # remote command line (visible in `ps`) nor in LeForge logs.
            script = (
                "import sys\n"
                "from huggingface_hub import HfApi\n"
                "api = HfApi(token=%r)\n"
                "api.create_repo(repo_id=%r, repo_type='model', exist_ok=True)\n"
                "api.upload_folder(folder_path=%r, repo_id=%r, repo_type='model')\n"
                "print('LEFORGE_UPLOAD_OK')\n"
            ) % (token, repo_id, remote_dir, repo_id)
            _stdin, stdout, stderr = api.exec_command(f"{python} -", timeout=3600)
            _stdin.write(script)
            _stdin.channel.shutdown_write()
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            code = stdout.channel.recv_exit_status()
        finally:
            api.close()
        if code != 0 or "LEFORGE_UPLOAD_OK" not in out:
            tail = (err or out).strip().splitlines()[-4:]
            raise RuntimeError("；".join(tail)[:400] or f"远端上传失败 (exit {code})")
        with _lock:
            _state.setdefault("by_job", {})[job_id] = repo_id
            _state.update(running=False, done=True, error=None, finished_at=time.time())
        _persist()
        logger.info("Pushed %s (remote) to %s", remote_dir, repo_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Remote model push failed")
        _set(running=False, done=False, error=str(exc)[:400], finished_at=time.time())


def handle_push_model(request: PushModelRequest) -> dict[str, Any]:
    from .jobs import JobNotFoundError, job_registry
    from .utils.hf_auth import get_token

    try:
        record = job_registry.get(request.job_id)
    except JobNotFoundError:
        return {"success": False, "status_code": 404, "message": f"任务 {request.job_id} 不存在"}

    token = request.token.strip() or (get_token() or "")
    if not token:
        return {
            "success": False,
            "status_code": 401,
            "message": "需要先登录 Hugging Face（请填写写入令牌）",
        }

    repo_id = request.repo_id.strip() or _default_repo_id(request.job_id, token)
    if "/" not in repo_id:
        return {"success": False, "status_code": 400, "message": "repo id 要写成 user/name"}

    with _lock:
        if _state.get("running"):
            return {"success": False, "status_code": 409, "message": "已经有一个上传在进行"}

    if record.runner == "ssh":
        # Direct pod → Hub: no local download needed.
        remote_dir, problem, connect_kwargs = _remote_ckpt_dir(request.job_id, request.step)
        if remote_dir is None:
            return {"success": False, "status_code": 400, "message": problem}
        with _lock:
            _state.clear()
            _state.update(
                {
                    "running": True,
                    "mode": "remote",
                    "job_id": request.job_id,
                    "repo_id": repo_id,
                    "started_at": time.time(),
                    "done": False,
                    "error": None,
                }
            )
        threading.Thread(
            target=_run_remote,
            args=(request.job_id, connect_kwargs, remote_dir, repo_id, token),
            daemon=True,
        ).start()
        _persist()
        return {
            "success": True,
            "message": f"开始从云端直传 {repo_id}",
            "repo_id": repo_id,
        }

    local_dir, problem = _local_dir_for(request.job_id, request.step)
    if local_dir is None:
        return {"success": False, "status_code": 400, "message": problem}

    total = sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())
    with _lock:
        _state.clear()
        _state.update(
            {
                "running": True,
                "mode": "local",
                "job_id": request.job_id,
                "repo_id": repo_id,
                "started_at": time.time(),
                "size_mb": round(total / 1e6),
                "done": False,
                "error": None,
            }
        )
    threading.Thread(
        target=_run, args=(request.job_id, local_dir, repo_id, token), daemon=True
    ).start()
    _persist()
    return {
        "success": True,
        "message": f"开始上传 {total / 1e6:.0f} MB 到 {repo_id}",
        "repo_id": repo_id,
    }
