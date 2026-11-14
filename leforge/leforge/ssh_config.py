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

"""Persistent storage for SSH connection configs used by the SSH job runner.

Configs are saved to ``~/.cache/huggingface/lerobot/ssh_configs.json`` so they
survive server restarts. Passwords are stored in plain text — same trust model
as the rest of the local lerobot cache directory.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SSH_CONFIGS_PATH = Path(
    os.environ.get(
        "LEFORGE_SSH_CONFIGS_PATH",
        Path.home() / ".cache" / "huggingface" / "lerobot" / "ssh_configs.json",
    )
)


class SSHConfig(BaseModel):
    id: str
    name: str
    host: str
    port: int = 22
    username: str
    password: str | None = None
    key_path: str | None = None
    remote_python: str = "python"
    remote_workdir: str | None = None
    hf_token: str | None = None
    use_cn_mirror: bool = False


class SSHConfigPublic(BaseModel):
    """Same as SSHConfig but without the password — safe to send to the frontend."""

    id: str
    name: str
    host: str
    port: int
    username: str
    has_password: bool
    key_path: str | None
    remote_python: str
    remote_workdir: str | None
    has_hf_token: bool
    use_cn_mirror: bool


def _load_all() -> list[dict]:
    if not _SSH_CONFIGS_PATH.exists():
        return []
    try:
        data = json.loads(_SSH_CONFIGS_PATH.read_text())
        if isinstance(data, list):
            # Hosts saved while LeForge tracked runs with W&B still carry that
            # key. It is a different credential from an HF token, so drop it
            # rather than let a dead secret sit in the file forever.
            dropped = False
            for raw in data:
                if raw.pop("wandb_api_key", None) is not None:
                    dropped = True
            if dropped:
                try:
                    _save_all(data)
                except Exception as exc:
                    logger.warning("Failed to clean legacy SSH configs: %s", exc)
            return data
    except Exception as exc:
        logger.warning("Failed to read SSH configs: %s", exc)
    return []


def _save_all(configs: list[dict]) -> None:
    _SSH_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SSH_CONFIGS_PATH.write_text(json.dumps(configs, indent=2))


def list_ssh_configs() -> list[SSHConfigPublic]:
    out: list[SSHConfigPublic] = []
    for raw in _load_all():
        try:
            cfg = SSHConfig.model_validate(raw)
            out.append(
                SSHConfigPublic(
                    id=cfg.id,
                    name=cfg.name,
                    host=cfg.host,
                    port=cfg.port,
                    username=cfg.username,
                    has_password=cfg.password is not None,
                    key_path=cfg.key_path,
                    remote_python=cfg.remote_python,
                    remote_workdir=cfg.remote_workdir,
                    has_hf_token=cfg.hf_token is not None,
                    use_cn_mirror=cfg.use_cn_mirror,
                )
            )
        except Exception as exc:
            logger.warning("Skipping invalid SSH config: %s", exc)
    return out


def get_ssh_config(config_id: str) -> SSHConfig | None:
    for raw in _load_all():
        if raw.get("id") == config_id:
            try:
                return SSHConfig.model_validate(raw)
            except Exception:
                return None
    return None


def set_remote_python(config_id: str, remote_python: str) -> None:
    """Patch only the interpreter field, leaving credentials untouched."""
    configs = _load_all()
    for raw in configs:
        if raw.get("id") == config_id:
            raw["remote_python"] = remote_python
            _save_all(configs)
            return
    raise ValueError(f"SSH config {config_id!r} not found")


def save_ssh_config(
    name: str,
    host: str,
    port: int,
    username: str,
    password: str | None = None,
    key_path: str | None = None,
    remote_python: str = "python",
    remote_workdir: str | None = None,
    hf_token: str | None = None,
    use_cn_mirror: bool = False,
    config_id: str | None = None,
) -> SSHConfigPublic:
    configs = _load_all()

    if config_id:
        for i, raw in enumerate(configs):
            if raw.get("id") == config_id:
                existing = SSHConfig.model_validate(raw)
                updated = SSHConfig(
                    id=existing.id,
                    name=name,
                    host=host,
                    port=port,
                    username=username,
                    password=password if password is not None else existing.password,
                    key_path=key_path,
                    remote_python=remote_python,
                    remote_workdir=remote_workdir,
                    hf_token=hf_token if hf_token is not None else existing.hf_token,
                    use_cn_mirror=use_cn_mirror,
                )
                configs[i] = updated.model_dump()
                _save_all(configs)
                return _to_public(updated)
        raise ValueError(f"SSH config {config_id!r} not found")

    new_id = uuid.uuid4().hex[:12]
    cfg = SSHConfig(
        id=new_id,
        name=name,
        host=host,
        port=port,
        username=username,
        password=password,
        key_path=key_path,
        remote_python=remote_python,
        remote_workdir=remote_workdir,
        hf_token=hf_token,
        use_cn_mirror=use_cn_mirror,
    )
    configs.append(cfg.model_dump())
    _save_all(configs)
    return _to_public(cfg)


def delete_ssh_config(config_id: str) -> bool:
    configs = _load_all()
    new_configs = [c for c in configs if c.get("id") != config_id]
    if len(new_configs) == len(configs):
        return False
    _save_all(new_configs)
    return True


def _tcp_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Cheap liveness probe before paying paramiko's handshake timeout.

    A stopped pod usually leaves the port closed (RST) or blackholed; the RST
    case fails in milliseconds, and the blackholed one costs `timeout` instead
    of the 10 s a full SSH connect would burn.
    """
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def test_ssh_connection(config_id: str) -> dict:
    """Try to connect and run a trivial command. Returns {success, message}."""
    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}
    if not _tcp_reachable(cfg.host, cfg.port):
        return {
            "success": False,
            "message": f"端口 {cfg.port} 连不上（{cfg.host}）——实例可能已停止，或重开后换了 host/port",
        }
    try:
        import paramiko
    except ImportError:
        return {"success": False, "message": "paramiko is not installed. Run: pip install paramiko"}

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs: dict = {
            "hostname": cfg.host,
            "port": cfg.port,
            "username": cfg.username,
            "timeout": 10,
        }
        if cfg.key_path:
            connect_kwargs["key_filename"] = os.path.expanduser(cfg.key_path)
        elif cfg.password:
            connect_kwargs["password"] = cfg.password
        client.connect(**connect_kwargs)
        _stdin, stdout, stderr = client.exec_command("echo ok && which python", timeout=10)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if "ok" in out:
            python_path = out.split("\n")[-1] if "\n" in out else "python"
            return {"success": True, "message": f"Connected. Python: {python_path}"}
        return {"success": False, "message": f"Unexpected output: {out} {err}"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}
    finally:
        client.close()


def _to_public(cfg: SSHConfig) -> SSHConfigPublic:
    return SSHConfigPublic(
        id=cfg.id,
        name=cfg.name,
        host=cfg.host,
        port=cfg.port,
        username=cfg.username,
        has_password=cfg.password is not None,
        key_path=cfg.key_path,
        remote_python=cfg.remote_python,
        remote_workdir=cfg.remote_workdir,
        has_hf_token=cfg.hf_token is not None,
        use_cn_mirror=cfg.use_cn_mirror,
    )
