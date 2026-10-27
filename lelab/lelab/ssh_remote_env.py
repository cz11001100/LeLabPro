# Copyright 2025 The HuggingFace Inc. library.
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

"""Remote environment checking and setup for SSH runner targets.

Provides helpers to inspect whether lerobot, CUDA, etc. are available on the
remote machine and to install lerobot when missing.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import logging
import os
import re
import shlex
import time
from typing import Generator

logger = logging.getLogger(__name__)

CN_PIP_MIRROR = "https://mirrors.aliyun.com/pypi/simple"
CN_PYTORCH_WHEELS = "https://mirror.sjtu.edu.cn/pytorch-wheels"

# Extras covering every supervised policy the training UI offers:
#   act / tdmpc / vqbet -> core deps
#   diffusion -> "diffusion"; pi0 / pi0_fast -> "pi"; smolvla -> "smolvla";
#   groot -> "groot"; reward_classifier deps -> "robometer"
# ("sac" is an RL-pipeline policy and does not go through lerobot-train.)
POLICY_UNION_EXTRAS = ("diffusion", "pi", "smolvla", "groot", "robometer")

# Extras each UI-facing policy needs beyond core + "training" deps.
# act / tdmpc / vqbet live in core lerobot and need no extra.
POLICY_INSTALL_EXTRAS: dict[str, tuple[str, ...]] = {
    "act": (),
    "tdmpc": (),
    "vqbet": (),
    "diffusion": ("diffusion",),
    "pi0": ("pi",),
    "pi0_fast": ("pi",),
    "smolvla": ("smolvla",),
    "groot": ("groot",),
    "reward_classifier": ("robometer",),
}

# Policies probed by both the env check and installer verification step.
PROBED_POLICIES = (
    "act",
    "diffusion",
    "pi0",
    "pi0_fast",
    "smolvla",
    "groot",
    "tdmpc",
    "vqbet",
    "reward_classifier",
)


def policy_union_extras(user_extra: str, policies: list[str] | None = None) -> str:
    """Merge training + policy extras with whatever the caller passed.

    Only what training actually needs: ``policies`` selects those policy types'
    extras, and an empty selection installs core + "training" alone — a job that
    later needs another extra gets it auto-installed at launch (see
    SSHJobRunner._install_policy_extras). Pass ``["*"]`` for every supervised
    policy extra up front.
    """
    extras = ["training"]
    if policies and "*" in policies:
        extras.extend(POLICY_UNION_EXTRAS)
    else:
        for p in policies or []:
            extras.extend(POLICY_INSTALL_EXTRAS.get(p, ()))
    for e in (user_extra or "").split(","):
        e = e.strip()
        if e and e not in extras:
            extras.append(e)
    return ",".join(dict.fromkeys(extras))


# Interpreter candidates on GPU pods, best first: the pod images ship miniconda
# while the distro `python3` is often too old for lerobot.
PYTHON_CANDIDATES = (
    "/usr/local/miniconda3/bin/python",
    "/opt/conda/bin/python",
    "python3.13",
    "python3.12",
    "python3",
)


def local_lerobot_version() -> str | None:
    """Version of the lerobot this LeLab process runs against."""
    with contextlib.suppress(Exception):
        return importlib.metadata.version("lerobot")
    return None


def local_python_floor() -> tuple[int, int]:
    """lerobot's own Requires-Python floor, so we don't hardcode a version."""
    with contextlib.suppress(Exception):
        spec = importlib.metadata.metadata("lerobot").get("Requires-Python") or ""
        m = re.search(r"(\d+)\.(\d+)", spec)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return (3, 12)


def local_torch_constraints() -> tuple[str, str]:
    """lerobot's own version specifiers for torch / torchvision.

    Read from the installed lerobot metadata so the remote never drifts from what
    the pinned version actually accepts (hardcoded caps had already gone stale).
    Returns (torch_spec, torchvision_spec), e.g. (">=2.7,<2.12.0", ">=0.22.0,<0.27.0").
    """
    specs = {"torch": "", "torchvision": ""}
    with contextlib.suppress(Exception):
        for r in importlib.metadata.requires("lerobot") or []:
            head, _, marker = r.partition(";")
            if "extra" in marker:
                continue  # extras-only pins are not the core training dependency
            head = head.strip()
            name = re.split(r"[\[<>!=~ ]", head, 1)[0].lower()
            if name in specs:
                spec = head[len(name):].strip()
                if spec:
                    specs[name] = spec
    return specs["torch"], specs["torchvision"]


def detect_remote_python(client, floor: tuple[int, int] | None = None) -> str | None:
    """First remote interpreter that meets lerobot's Requires-Python floor."""
    maj, minor = floor or local_python_floor()
    script = (
        "for p in " + " ".join(PYTHON_CANDIDATES) + "; do "
        'command -v "$p" >/dev/null 2>&1 || continue; '
        f'"$p" -c \'import sys; sys.exit(0 if sys.version_info >= ({maj}, {minor}) else 1)\' 2>/dev/null || continue; '
        'echo "$p"; break; done'
    )
    code, out, _err = _run_cmd(client, script, timeout=90)
    lines = [l.strip() for l in out.strip().splitlines() if l.strip().startswith(("/", "python"))]
    return lines[0] if code == 0 and lines else None


def _python_meets_floor(client, python: str, floor: tuple[int, int]) -> bool:
    """Whether a specific remote interpreter satisfies lerobot's version floor."""
    maj, minor = floor
    code, _out, _err = _run_cmd(
        client,
        f"{shlex.quote(python)} -c "
        f"'import sys; sys.exit(0 if sys.version_info >= ({maj}, {minor}) else 1)'",
        timeout=60,
    )
    return code == 0


def effective_python(cfg, client=None) -> str:
    """The interpreter to run remote lerobot commands with.

    A blank `remote_python` used to mean `python3`, which on most pods is the
    distro's 3.10: the SSH test still passes (it only checks connectivity) but
    pip then installs an ancient lerobot, or none at all. An explicitly wrong
    value is just as bad, so the configured path is *validated* too — otherwise
    filling in `python3` beats leaving it empty, which is backwards.
    """
    from .ssh_config import set_remote_python

    floor = local_python_floor()
    owned = client is None
    if owned:
        client = _connect(cfg)
    try:
        if cfg.remote_python and _python_meets_floor(client, cfg.remote_python, floor):
            return cfg.remote_python
        found = detect_remote_python(client, floor)
    finally:
        if owned and client is not None:
            with contextlib.suppress(Exception):
                client.close()
    if found and found != cfg.remote_python:
        with contextlib.suppress(Exception):
            set_remote_python(cfg.id, found)
        logger.info(
            "Remote interpreter %s → %s for config %s (Python >= %d.%d required)",
            cfg.remote_python or "(unset)", found, cfg.id, floor[0], floor[1],
        )
        return found
    return cfg.remote_python or found or "python3"


def _connect(cfg) -> "paramiko.SSHClient":
    import paramiko

    # Cloud SSH proxies (e.g. compshare) intermittently drop the banner
    # handshake; retry so a transient blip doesn't fail the whole install.
    last_exc: Exception | None = None
    for attempt in range(4):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kw: dict = {
            "hostname": cfg.host,
            "port": cfg.port,
            "username": cfg.username,
            "timeout": 15,
        }
        if cfg.key_path:
            kw["key_filename"] = os.path.expanduser(cfg.key_path)
        elif cfg.password:
            kw["password"] = cfg.password
        try:
            client.connect(**kw)
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            return client
        except Exception as exc:
            last_exc = exc
            with contextlib.suppress(Exception):
                client.close()
            logger.warning("SSH connect attempt %d failed: %s", attempt + 1, exc)
            time.sleep(3 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _run_cmd(
    client,
    cmd: str,
    timeout: int = 30,
    kill_pattern: str | None = None,
) -> tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr).

    paramiko's recv_exit_status() blocks forever, so the deadline is enforced
    here. On timeout the remote command is killed: otherwise it keeps running
    (and holds apt/dpkg locks) long after we gave up, wedging later attempts.
    """
    _stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    channel = stdout.channel
    deadline = time.time() + timeout
    while not channel.exit_status_ready():
        if time.time() > deadline:
            logger.warning("Remote command timed out after %ss: %s", timeout, cmd[:120])
            pattern = kill_pattern or cmd.split("|")[0].strip()[:60]
            # Bracket the first char so the pkill command line itself can't
            # match the pattern (it contains the pattern verbatim).
            safe_pattern = f"[{pattern[0]}]{pattern[1:]}" if pattern else ""
            with contextlib.suppress(Exception):
                client.exec_command(f"pkill -f '{safe_pattern}'", timeout=10)
            with contextlib.suppress(Exception):
                channel.close()
            return -1, "", f"timed out after {timeout}s"
        time.sleep(0.5)
    exit_code = channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    return exit_code, out, err


# Repeating these can never succeed — they are permission/lock/absent-tool
# failures, not transient network blips. Without this the installer retried them
# three times, burned 15 minutes and showed the user a bare "install failed".
_NON_RETRYABLE_APT_PATTERNS = (
    "permission denied",
    "could not open lock file",
    "unable to acquire the dpkg frontend lock",
    "are you root",
    "is not installed",
    "e: unsupported method",
)

ROOT_FREE_WORKAROUND = (
    "远端用户不是 root 且无免密 sudo，apt/yum 装不了。"
    "改用无需 root 的方式：conda install -y -c conda-forge ffmpeg，"
    "或把静态二进制放到 ~/.local/bin 并确保它在 PATH 里。"
)


def _is_non_retryable(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in _NON_RETRYABLE_APT_PATTERNS)


def _apt_privilege(client) -> tuple[str | None, str]:
    """How to invoke the remote package manager: '' , 'sudo -n ', or give up.

    GPU pods are frequently handed out as an unprivileged user, where a bare
    apt-get dies on the dpkg lock. ``sudo -n`` cannot hang: it fails instantly
    instead of prompting for a password nobody can type into an SSH exec channel.
    """
    _, uid, _ = _run_cmd(client, "id -u", timeout=10)
    if uid.strip() == "0":
        return "", ""
    exit_code, _, _ = _run_cmd(client, "sudo -n true", timeout=15)
    if exit_code == 0:
        return "sudo -n ", ""
    return None, ROOT_FREE_WORKAROUND


def _ensure_apt_cn_mirror(client, use_cn_mirror: bool, priv: str = "") -> str | None:
    """Point apt at the Aliyun mirror on CN hosts.

    The stock archive.ubuntu.com sources are frequently unreachable from CN
    clouds; apt-get then retries silently for tens of minutes and the install
    looks hung. Idempotent — safe to call before every apt invocation.

    Returns None when there is nothing to fix, otherwise the remote's own error
    line. Silently ignoring a failed rewrite left apt pointed at the unreachable
    official mirror, so the real cause surfaced later as an unexplained timeout.
    """
    if not use_cn_mirror:
        return None
    _, exists, _ = _run_cmd(
        client, "test -f /etc/apt/sources.list && echo yes || echo no", timeout=15
    )
    if exists.strip() != "yes":
        return None  # deb86 layout or non-Ubuntu: no legacy file to rewrite
    exit_code, out, err = _run_cmd(
        client,
        f"{priv}sed -i -E 's|https?://(archive|security)\\.ubuntu\\.com/ubuntu"
        "|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list",
        timeout=30,
    )
    if exit_code != 0:
        lines = (out + err).strip().splitlines()
        return lines[-1] if lines else "改写 /etc/apt/sources.list 失败"
    return None


def _env_probe_script() -> str:
    """Remote probe returning a JSON fact-sheet that diagnose() interprets.

    Deliberately collects the *reason* for each failure (driver missing vs torch
    built for another CUDA, torchcodec ABI mismatch, per-policy import error) —
    a bare "not installed" is what made people re-run the installer blindly.
    """
    return (
        "import sys, json, shutil, subprocess\n"
        "r = {}\n"
        "r['python_version'] = '%d.%d.%d' % sys.version_info[:3]\n"
        "r['nvidia_smi'] = bool(shutil.which('nvidia-smi'))\n"
        "r['driver_version'] = None\n"
        "if r['nvidia_smi']:\n"
        "    try:\n"
        "        o = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name',"
        " '--format=csv,noheader'], capture_output=True, text=True, timeout=20)\n"
        "        if o.returncode == 0 and o.stdout.strip():\n"
        "            first = o.stdout.strip().splitlines()[0]\n"
        "            r['driver_version'] = first.split(',')[0].strip()\n"
        "            r['gpu_name_smi'] = first.split(',')[-1].strip()\n"
        "        else:\n"
        "            r['nvidia_smi_error'] = ((o.stderr or '') + (o.stdout or '')).strip()[:200]\n"
        "    except Exception as e:\n"
        "        r['nvidia_smi_error'] = repr(e)[:200]\n"
        "r['ffmpeg'] = bool(shutil.which('ffmpeg'))\n"
        "try:\n"
        "    import lerobot\n"
        "    r['lerobot_installed'] = True\n"
        "    r['lerobot_version'] = getattr(lerobot, '__version__', 'unknown')\n"
        "except Exception as e:\n"
        "    r['lerobot_installed'] = False; r['lerobot_version'] = None\n"
        "    r['lerobot_error'] = repr(e)[:200]\n"
        "r['torch_version'] = None; r['cuda_available'] = False\n"
        "r['cuda_version'] = None; r['gpu_name'] = None\n"
        "try:\n"
        "    import torch\n"
        "    r['torch_version'] = torch.__version__\n"
        "    r['cuda_version'] = torch.version.cuda\n"
        "    r['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    if r['cuda_available']:\n"
        "        r['gpu_name'] = torch.cuda.get_device_name(0)\n"
        "except Exception as e:\n"
        "    r['torch_error'] = repr(e)[:200]\n"
        "r['torchcodec_error'] = None\n"
        "try:\n"
        "    from torchcodec.decoders import VideoDecoder\n"
        "except Exception as e:\n"
        "    r['torchcodec_error'] = repr(e)[:220]\n"
        "r['training_deps_missing'] = []\n"
        "for m in ('accelerate', 'datasets', 'wandb'):\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception:\n"
        "        r['training_deps_missing'].append(m)\n"
        "r['policies_ready'] = []; r['policies_missing'] = []; r['policy_errors'] = {}\n"
        "if r['lerobot_installed']:\n"
        "    for t in ('act','diffusion','pi0','pi0_fast','smolvla','groot','tdmpc','vqbet','reward_classifier'):\n"
        "        try:\n"
        "            if t == 'reward_classifier':\n"
        "                from lerobot.rewards.factory import get_reward_model_class\n"
        "                get_reward_model_class('reward_classifier')\n"
        "            else:\n"
        "                from lerobot.policies.factory import get_policy_class\n"
        "                get_policy_class(t)\n"
        "            r['policies_ready'].append(t)\n"
        "        except Exception as e:\n"
        "            r['policies_missing'].append(t)\n"
        "            r['policy_errors'][t] = repr(e)[:200]\n"
        "try:\n"
        "    r['disk_free_gb'] = round(shutil.disk_usage('/').free / 1073741824, 1)\n"
        "except Exception:\n"
        "    r['disk_free_gb'] = None\n"
        "print(json.dumps(r))\n"
    )


def _explain_connect_error(exc: Exception) -> str:
    """Translate paramiko/socket failures into what the user should do next."""
    text = str(exc)
    low = text.lower()
    if "banner" in low:
        return (
            f"SSH 握手失败（{text}）：按以往经验，这基本等于该实例已被停止或回收——"
            "端口还在，但后面没有 sshd。去 GPU 平台控制台确认实例状态，重启或新开一台，"
            "并更新 SSH 配置里的 host/port。"
        )
    if "authentication failed" in low or "auth" in low and "fail" in low:
        return f"SSH 认证失败（{text}）：密码或私钥不对，或平台改了凭据；重新填写密码/密钥路径。"
    if "timed out" in low or "timeout" in low:
        return (
            f"SSH 连接超时（{text}）：host/port 可能已变（实例重开常换端口），"
            "或被本机网络/防火墙拦截。"
        )
    if "refused" in low or "unable to connect" in low or "not found" in low:
        return (
            f"SSH 端口拒绝连接（{text}）：实例未开机，或 host/port 已变。"
            "实例每次重开都可能换端口，需要更新 SSH 配置。"
        )
    if "host key" in low:
        return f"SSH 主机密钥校验失败（{text}）：实例被重建，host key 变了。"
    return f"SSH 连接失败：{text[:240]}"


def diagnose_env(data: dict, *, python_used: str, floor=None, local_version=None) -> list[dict]:
    """Turn the probe fact-sheet into actionable issues.

    `severity`: blocking stops training, warning degrades it, info is context.
    `auto_fixable`: the 安装环境 flow repairs it.
    """
    floor = floor or local_python_floor()
    local_version = local_version or local_lerobot_version()
    issues: list[dict] = []

    def add(code, severity, message, fix, auto_fixable):
        issues.append({
            "code": code, "severity": severity, "message": message,
            "fix": fix, "auto_fixable": auto_fixable,
        })

    py = data.get("python_version") or ""
    try:
        py_tuple = tuple(int(x) for x in py.split(".")[:2])
    except ValueError:
        py_tuple = (0, 0)
    if py_tuple < floor:
        add("python_too_old", "blocking",
            f"当前解释器 {python_used} 是 Python {py}，lerobot 需要 ≥{floor[0]}.{floor[1]}",
            "remote_python 已做过校验并会自动改用达标的解释器，仍报这条说明这台机器上"
            f"根本没有 ≥{floor[0]}.{floor[1]} 的 Python：pip 会装到旧版 lerobot 或干脆装不上。"
            "先装一个（conda / uv python install 3.12 / 系统包），再重跑「检查环境」。",
            False)

    if not data.get("lerobot_installed"):
        add("lerobot_missing", "blocking",
            "远端没有可用的 lerobot" + (f"（{data.get('lerobot_error')}）" if data.get("lerobot_error") else ""),
            "运行「安装环境」；它会只装训练必需依赖，并把版本对齐到本地 lerobot "
            f"{local_version or '(未知)'}。",
            True)
    elif local_version and data.get("lerobot_version") not in (local_version, "unknown", None):
        add("lerobot_version_mismatch", "blocking",
            f"远端 lerobot {data.get('lerobot_version')} ≠ 本地 {local_version}",
            "版本不一致时，远端训出的 checkpoint 可能无法在本地 lerobot 加载（配置字段或"
            "数据格式漂移）。运行「安装环境」会把远端对齐到本地版本。",
            True)

    if not data.get("cuda_available"):
        if not data.get("nvidia_smi"):
            add("gpu_no_driver", "blocking",
                "远端找不到 nvidia-smi，实例没有挂载 GPU 或驱动未装",
                "这不是软件能修的：在 GPU 平台控制台确认实例带卡并已启动驱动，"
                "或换一台带卡实例。",
                False)
        elif data.get("nvidia_smi_error"):
            add("gpu_driver_error", "blocking",
                f"nvidia-smi 报错：{data.get('nvidia_smi_error')}",
                "驱动与容器不匹配或 GPU 被占用。重启该实例通常可恢复；反复出现就换实例。",
                False)
        else:
            add("torch_cuda_mismatch", "blocking",
                f"驱动正常（nvidia-smi {data.get('driver_version') or 'ok'}），但 torch "
                f"{data.get('torch_version') or '(未安装)'} 看不到 CUDA",
                "多为 torch 构建与驱动/CUDA 不匹配（或实例刚重启、/dev/nvidia* 未挂上）。"
                "「安装环境」会按驱动版本重装匹配的 torch；若刚重启过实例，先重启实例更省时间。",
                True)
    elif data.get("torch_version") is None:
        add("torch_missing", "blocking", "torch 未安装", "运行「安装环境」。", True)

    if data.get("torchcodec_error"):
        add("torchcodec_abi", "warning",
            f"torchcodec 无法导入：{data['torchcodec_error']}",
            "torchcodec 按 torch 版本发行但元数据不钉版本，pip 常装到不匹配的版本。"
            "「安装环境」会逐个版本回退对齐；临时可用 `--dataset.video_backend=pyav`。",
            True)

    if not data.get("ffmpeg"):
        add("ffmpeg_missing", "warning", "远端没有 ffmpeg",
            "torchcodec 解码视频必需。训练启动时会自动 apt 安装，"
            "「安装环境」也会装。", True)

    missing_deps = data.get("training_deps_missing") or []
    if data.get("lerobot_installed") and missing_deps:
        add("training_deps_missing", "blocking",
            f"训练依赖缺失：{', '.join(missing_deps)}",
            "运行「安装环境」（会装 accelerate / datasets / wandb / tensorboard）。", True)

    missing_pol = data.get("policies_missing") or []
    if missing_pol and data.get("lerobot_installed"):
        detail = "; ".join(f"{p}: {data.get('policy_errors', {}).get(p, '')}" for p in missing_pol[:3])
        add("policy_extras_missing", "warning",
            f"这些策略当前不可训：{', '.join(missing_pol)}",
            f"原因示例 — {detail}。训练任务启动时会自动补装该策略的 extras，"
            "或在「安装环境」里勾选对应策略一次装齐。", True)

    free = data.get("disk_free_gb")
    if free is not None and free < 15:
        add("disk_low", "warning", f"远端磁盘只剩 {free}GB",
            "数据集 + checkpoint 很容易写满（历史上出现过 No space left on device）。"
            "可清理：~/.cache/huggingface/datasets、旧的 ~/lelab_outputs 任务目录、pip 缓存。",
            False)

    return issues


# The progress model that scores rollout episodes (see lelab.rollout_eval) and
# the VLM backbone its config names. ~8.9 GB each; a 4B model does not fit the
# laptop GPU, so the download happens on the pod.
REWARD_MODEL_REPOS = ("lerobot/Robometer-4B", "Qwen/Qwen3-VL-4B-Instruct")
_REWARD_LOG = "/tmp/lelab_reward_download.log"
_SARM_TRAIN_LOG = "/tmp/lelab_sarm_train.log"


def _repo_cache_dir(repo: str) -> str:
    org, _, name = repo.partition("/")
    return f"~/.cache/huggingface/hub/models--{org}--{name}"


def reward_model_status(config_id: str) -> dict:
    """What of the scoring model is already on the pod, and is a pull running."""
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}
    try:
        client = _connect(cfg)
    except Exception as exc:
        return {"success": False, "message": _explain_connect_error(exc)}

    try:
        models = []
        for repo in REWARD_MODEL_REPOS:
            path = _repo_cache_dir(repo)
            code, size, _ = _run_cmd(
                client, f"du -sh {path} 2>/dev/null | cut -f1", timeout=30
            )
            models.append(
                {
                    "repo_id": repo,
                    "present": code == 0 and bool(size.strip()),
                    "size": size.strip(),
                }
            )
        _, running, _ = _run_cmd(
            client, "pgrep -f 'hf download' | head -3", timeout=20
        )
        _, log_tail, _ = _run_cmd(client, f"tail -3 {_REWARD_LOG} 2>/dev/null", timeout=20)
        return {
            "success": True,
            "models": models,
            "ready": all(m["present"] for m in models),
            "downloading": bool(running.strip()),
            "log_path": _REWARD_LOG,
            "log_tail": log_tail.strip()[-400:],
        }
    finally:
        client.close()


def start_reward_model_download(config_id: str, *, use_cn_mirror: bool = True) -> dict:
    """Pull the scoring model onto the pod in the background (nohup + log).

    Detached on purpose: 18 GB takes longer than any sane HTTP request, and a
    dropped SSH channel must not kill the download mid-file."""
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}
    endpoint = "https://hf-mirror.com" if use_cn_mirror else "https://huggingface.co"
    pulls = " && ".join(f'hf download {repo}' for repo in REWARD_MODEL_REPOS)
    script = (
        f"export HF_ENDPOINT={endpoint}; "
        f"command -v hf >/dev/null 2>&1 || python3 -m pip install -q -U huggingface_hub; "
        f"{pulls}; echo REWARD_DOWNLOAD_DONE=$?"
    )
    cmd = f"nohup bash -c {shlex.quote(script)} > {_REWARD_LOG} 2>&1 < /dev/null &"

    try:
        client = _connect(cfg)
    except Exception as exc:
        return {"success": False, "message": _explain_connect_error(exc)}
    try:
        code, _, err = _run_cmd(client, cmd, timeout=30)
        if code != 0:
            return {"success": False, "message": f"启动下载失败: {err[-300:]}"}
        return {
            "success": True,
            "message": "已在 pod 上开始下载打分模型",
            "log_path": _REWARD_LOG,
            "endpoint": endpoint,
            "repos": list(REWARD_MODEL_REPOS),
        }
    finally:
        client.close()


def start_sarm_training_remote(
    config_id: str,
    dataset_repo_id: str,
    *,
    image_key: str = "observation.images.up",
    steps: int = 5000,
    batch_size: int = 32,
    push_repo_id: str = "",
    use_cn_mirror: bool = True,
) -> dict:
    """Train a task-specific SARM scorer on the pod, detached.

    SARM is the small, trainable scorer (single_stage mode needs no subtask
    annotations: linear 0→1 progress is the label). Its output is the second
    backend in the scoring comparison — the one that has actually seen this rig.
    """
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}
    endpoint = "https://hf-mirror.com" if use_cn_mirror else "https://huggingface.co"
    # lerobot 0.6 trains reward models through the same script as policies, but
    # under its own config node: cfg.reward_model (is_reward_model_training).
    # `--policy.type=sarm` is not a thing — sarm is not in the policy registry.
    push = (
        f"--reward_model.repo_id={push_repo_id} --reward_model.push_to_hub=true"
        if push_repo_id
        else ""
    )
    script = (
        f"export HF_ENDPOINT={endpoint}; "
        f"python3 -m lerobot.scripts.lerobot_train "
        f"--dataset.repo_id={dataset_repo_id} --reward_model.type=sarm "
        f"--reward_model.annotation_mode=single_stage --reward_model.image_key={image_key} "
        f"--steps={steps} --batch_size={batch_size} --output_dir=outputs/sarm {push}; "
        "echo SARM_TRAIN_DONE=$?"
    )
    cmd = f"nohup bash -c {shlex.quote(script)} > {_SARM_TRAIN_LOG} 2>&1 < /dev/null &"
    try:
        client = _connect(cfg)
    except Exception as exc:
        return {"success": False, "message": _explain_connect_error(exc)}
    try:
        code, _, err = _run_cmd(client, cmd, timeout=30)
        if code != 0:
            return {"success": False, "message": f"启动 SARM 训练失败: {err[-300:]}"}
        return {
            "success": True,
            "message": "已在 pod 上开始训练 SARM 打分模型",
            "log_path": _SARM_TRAIN_LOG,
            "command": script,
        }
    finally:
        client.close()


def sarm_training_status(config_id: str) -> dict:
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}
    try:
        client = _connect(cfg)
    except Exception as exc:
        return {"success": False, "message": _explain_connect_error(exc)}
    try:
        _, running, _ = _run_cmd(client, "pgrep -f lerobot_train | head -3", timeout=20)
        _, tail, _ = _run_cmd(client, f"tail -4 {_SARM_TRAIN_LOG} 2>/dev/null", timeout=20)
        _, done, _ = _run_cmd(client, f"grep -c 'SARM_TRAIN_DONE=0' {_SARM_TRAIN_LOG} 2>/dev/null", timeout=20)
        return {
            "success": True,
            "training": bool(running.strip()),
            "finished_ok": done.strip() not in ("", "0"),
            "log_path": _SARM_TRAIN_LOG,
            "log_tail": tail.strip()[-400:],
        }
    finally:
        client.close()


def check_remote_environment(config_id: str) -> dict:
    """Inspect the remote machine: Python version, lerobot, CUDA, disk space.

    Returns a dict with keys: success, python_version, lerobot_installed,
    lerobot_version, cuda_available, cuda_version, gpu_name, disk_free_gb, message.
    """
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        return {"success": False, "message": "Config not found"}

    def fail(exc_text: str, kind: str) -> dict:
        return {"success": False, "ready": False, "error_kind": kind, "message": exc_text}

    try:
        python = effective_python(cfg)
        client = _connect(cfg)
    except Exception as exc:
        return fail(_explain_connect_error(exc), "connect")

    try:
        _stdin, stdout, stderr = client.exec_command(f"{python} -", timeout=120)
        _stdin.write(_env_probe_script())
        _stdin.channel.shutdown_write()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        if not out:
            return fail(
                f"探测脚本没有输出（{python} 可能不可用）：{(err or '')[-300:]}", "probe"
            )
        data = json.loads(out.splitlines()[-1])
        issues = diagnose_env(data, python_used=python)
        blocking = [i for i in issues if i["severity"] == "blocking"]
        warnings = [i for i in issues if i["severity"] in ("warning", "info")]

        data["success"] = True
        data["ready"] = not blocking
        data["issues"] = issues
        data["blocking"] = [i["code"] for i in blocking]
        data["warnings"] = [i["code"] for i in warnings]
        data["fixable"] = [i["code"] for i in issues if i["auto_fixable"]]
        data["required_python"] = ">=%d.%d" % local_python_floor()
        data["local_lerobot_version"] = local_lerobot_version()
        data["remote_python_used"] = python
        parts = [
            f"Python {data.get('python_version')}",
            f"lerobot {data.get('lerobot_version') or '未安装'}"
            f"（本地 {data['local_lerobot_version'] or '?'}）",
            f"CUDA {'可用' if data.get('cuda_available') else '不可用'}",
            f"torch {data.get('torch_version') or '未安装'}",
            f"磁盘 {data.get('disk_free_gb')}GB",
        ]
        data["message"] = "，".join(parts)
        if blocking:
            data["message"] += f" — {len(blocking)} 个阻塞问题"
        elif warnings:
            data["message"] += f" — {len(warnings)} 个提醒"
        else:
            data["message"] += " — 环境就绪"
        return data
    except Exception as exc:
        return fail(f"探测失败：{exc}", "probe")
    finally:
        client.close()


TOTAL_INSTALL_STEPS = 7


def install_lerobot_remote_streaming(
    config_id: str,
    extra: str = "",
    use_cn_mirror: bool = False,
    policies: list[str] | None = None,
) -> Generator[dict, None, None]:
    """Install a complete training environment, yielding progress updates.

    Each yielded dict has:
        step: int (1-based current step)
        total_steps: int
        label: str (what this step does)
        status: "running" | "done" | "failed"
        logs: list[str] (recent output lines)

    Yields a final dict with status="done" or status="failed".
    """
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(config_id)
    if cfg is None:
        yield {"step": 0, "total_steps": TOTAL_INSTALL_STEPS, "label": "Init",
               "status": "failed", "logs": [], "message": "Config not found"}
        return

    workdir_prefix = f"cd {cfg.remote_workdir} && " if cfg.remote_workdir else ""

    pip_mirror_flag = f" -i {CN_PIP_MIRROR} --trusted-host mirrors.aliyun.com" if use_cn_mirror else ""
    pytorch_index = f"{CN_PYTORCH_WHEELS}" if use_cn_mirror else "https://download.pytorch.org/whl"
    # Target is the lerobot this LeLab runs against: a remote on another version
    # produces checkpoints the local lerobot may not load.
    target_version = local_lerobot_version()
    python = cfg.remote_python or ""

    def step_result(
        step: int, label: str, status: str, logs: list[str], message: str = "", hint: str = ""
    ) -> dict:
        return {
            "step": step,
            "total_steps": TOTAL_INSTALL_STEPS,
            "label": label,
            "status": status,
            "logs": logs[-15:],
            "message": message,
            "hint": hint,
        }

    try:
        client = _connect(cfg)
    except Exception as exc:
        # An uncaught connect error kills the generator, and the UI is left
        # showing only "安装流中断" with no reason.
        yield step_result(0, "连接远端", "failed", [], _explain_connect_error(exc))
        return

    try:
        # A blank remote_python would otherwise mean the distro's python3 — often
        # too old for lerobot, which makes pip install nothing useful while the
        # SSH "test" still passes. Detect once and persist the choice.
        python = effective_python(cfg, client)

        # ── Step 0: Diagnose before changing anything ───────────────────────
        yield step_result(0, "诊断远端环境...", "running", [f"python: {python}"])
        facts: dict = {}
        try:
            _stdin, sout, _serr = client.exec_command(f"{python} -", timeout=120)
            _stdin.write(_env_probe_script())
            _stdin.channel.shutdown_write()
            probe_out = sout.read().decode().strip()
            if probe_out:
                facts = json.loads(probe_out.splitlines()[-1])
        except Exception as exc:
            yield step_result(0, "诊断", "done", [f"诊断脚本未返回结果：{exc}"],
                              "无法预检，按全新安装逐步执行")
        issues = diagnose_env(facts, python_used=python) if facts else []
        blocking = [i for i in issues if i["severity"] == "blocking"]
        yield step_result(
            0,
            "诊断完成",
            "done",
            [f"[{'阻塞' if i['severity'] == 'blocking' else '提醒'}] {i['message']} → {i['fix']}"
             for i in issues] or ["未发现已装环境，按全新安装执行"],
            f"共 {len(issues)} 项（阻塞 {len(blocking)}）；已满足的步骤会自动跳过，"
            "标记可自动修复的会在这轮装好",
        )

        # ── Step 1: Detect GPU ──────────────────────────────────────────────
        # A GPU is required to *train*, not to install: lerobot, ffmpeg and the
        # training deps are all GPU-independent. Only the torch build needs the
        # driver version, so a missing/broken card degrades this step to a note
        # instead of aborting the whole install.
        yield step_result(1, "Detecting GPU...", "running", [f"python: {python}"])
        exit_code, gpu_out, gpu_err = _run_cmd(
            client,
            "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 || echo 'NO_GPU'",
            timeout=30,
        )
        gpu_out = gpu_out.strip()
        first_line = gpu_out.splitlines()[0] if gpu_out else ""
        fields = [f.strip() for f in first_line.split(",")]
        driver_version = fields[-1] if len(fields) > 1 else ""
        gpu_name = fields[0] if fields else ""
        driver_ok = "NO_GPU" not in gpu_out and bool(re.match(r"^\d+\.", driver_version))
        cuda_tag = None

        if driver_ok:
            driver_major = int(driver_version.split(".")[0])

            if driver_major >= 570:
                cuda_tag = "cu128"
            elif driver_major >= 535:
                cuda_tag = "cu124"
            elif driver_major >= 520:
                cuda_tag = "cu121"
            else:
                cuda_tag = "cu118"

        if driver_ok:
            yield step_result(1, "GPU detected", "done",
                              [f"{gpu_name}", f"Driver {driver_version} → CUDA {cuda_tag}"],
                              f"{gpu_name}, Driver {driver_version}")
        else:
            detail = first_line or (gpu_err or "").strip() or "nvidia-smi 无输出"
            yield step_result(
                1, "未检测到可用 GPU（继续安装）", "done",
                [detail[:200]],
                f"软件会照常装完，但这台机器当前看不到卡：{detail[:120]}",
                hint="没卡不影响安装（lerobot / ffmpeg / 训练依赖都不需要 GPU）；"
                     "只是这次**不猜 CUDA 版本装 torch**，避免装完依旧看不到卡。"
                     "驱动类报错（如 Exec format error）通常在平台控制台重启实例后恢复，"
                     "恢复后重跑「安装环境」即可按驱动补上匹配的 torch。")

        # ── Step 2: Ensure pip is available ─────────────────────────────────
        yield step_result(2, "Checking pip...", "running", [])
        exit_code, pip_out, pip_err = _run_cmd(
            client, f"{python} -m pip --version", timeout=15
        )
        pip_logs = []
        if exit_code != 0:
            yield step_result(2, "Installing pip...", "running", ["pip not found, installing..."])
            detect_pkg_mgr = "which apt-get >/dev/null 2>&1 && echo 'apt' || (which yum >/dev/null 2>&1 && echo 'yum' || echo 'unknown')"
            _, pkg_mgr, _ = _run_cmd(client, detect_pkg_mgr, timeout=10)
            pkg_mgr = pkg_mgr.strip()

            priv: str | None = ""
            if pkg_mgr in ("apt", "yum"):
                priv, hint = _apt_privilege(client)
            if priv is None:
                # No root available: get-pip.py installs into the user's own
                # interpreter and needs no privilege at all, which beats failing
                # the whole install over a dpkg lock.
                yield step_result(
                    2, "Installing pip (无 root，改用 get-pip.py)...", "running",
                    [hint, "回退到用户级安装，不需要 root"],
                )
                install_cmd = f"curl -sS https://bootstrap.pypa.io/get-pip.py | {python} 2>&1 | tail -5"
            elif pkg_mgr == "apt":
                mirror_err = _ensure_apt_cn_mirror(client, use_cn_mirror, priv)
                if mirror_err:
                    yield step_result(
                        2, "pip install", "failed",
                        [f"改写 apt 源失败: {mirror_err}",
                         "取消勾选 Use China mirrors 后重试，或让远端管理员配置 apt 源"],
                        "apt 源不可写",
                    )
                    return
                install_cmd = (
                    f"{priv}apt-get update -qq && "
                    f"{priv}apt-get install -y -qq python3-pip python3-venv 2>&1 | tail -5"
                )
            elif pkg_mgr == "yum":
                install_cmd = f"{priv}yum install -y -q python3-pip 2>&1 | tail -5"
            else:
                install_cmd = f"curl -sS https://bootstrap.pypa.io/get-pip.py | {python} 2>&1 | tail -5"

            exit_code, out, err = _run_cmd(client, install_cmd, timeout=300, kill_pattern="apt-get")
            pip_logs = (out + err).strip().splitlines()[-10:]
            if exit_code != 0 and _is_non_retryable(out + err):
                pip_logs = pip_logs + ["该错误重试无意义（权限/锁/缺工具），已停止重试"]
            if exit_code != 0:
                yield step_result(2, "pip install", "failed", pip_logs,
                                  f"Failed to install pip (exit {exit_code})")
                return
            exit_code2, out2, _ = _run_cmd(client, f"{python} -m pip --version", timeout=15)
            if exit_code2 != 0:
                yield step_result(2, "pip install", "failed", pip_logs + ["pip still not available"],
                                  "pip installation succeeded but pip is still not importable")
                return
            pip_logs.append("pip installed successfully")
        else:
            pip_logs.append(pip_out.strip())

        yield step_result(2, "pip ready", "done", pip_logs, "pip available")

        # ── Step 3: Install FFmpeg (needed by torchcodec) ────────────────────
        yield step_result(3, "Installing FFmpeg...", "running", [])
        _, ff_which, _ = _run_cmd(client, "which ffmpeg", timeout=10)
        if ff_which.strip():
            yield step_result(3, "FFmpeg installed", "done", [ff_which.strip()], "Already present")
        else:
            detect_pkg_mgr_ff = "which apt-get >/dev/null 2>&1 && echo 'apt' || (which yum >/dev/null 2>&1 && echo 'yum' || echo 'unknown')"
            _, pkg_mgr_ff, _ = _run_cmd(client, detect_pkg_mgr_ff, timeout=10)
            pkg_mgr_ff = pkg_mgr_ff.strip()

            priv_ff: str | None = ""
            if pkg_mgr_ff in ("apt", "yum"):
                priv_ff, hint_ff = _apt_privilege(client)
                if priv_ff is None:
                    # Say so up front instead of failing three times against the
                    # same dpkg lock, which read as a flaky installer.
                    yield step_result(3, "FFmpeg install", "failed",
                                      [hint_ff, "装好 ffmpeg 后重新点击安装环境即可，"
                                               "本步检测到已有 ffmpeg 会直接跳过"],
                                      "远端无 root/免密 sudo，apt 无法安装")
                    return

            if pkg_mgr_ff == "apt":
                mirror_err = _ensure_apt_cn_mirror(client, use_cn_mirror, priv_ff)
                if mirror_err:
                    yield step_result(
                        3, "FFmpeg install", "failed",
                        [f"改写 apt 源失败: {mirror_err}",
                         "取消勾选 Use China mirrors 后重试，或让远端管理员配置 apt 源"],
                        "apt 源不可写",
                    )
                    return
                ffmpeg_cmd = (
                    f"{priv_ff}apt-get update -qq && "
                    f"{priv_ff}apt-get install -y -qq ffmpeg 2>&1 | tail -5"
                )
                ff_kill = "apt-get"
            elif pkg_mgr_ff == "yum":
                ffmpeg_cmd = f"{priv_ff}yum install -y -q ffmpeg 2>&1 | tail -5"
                ff_kill = "yum"
            else:
                ffmpeg_cmd = "echo 'Unknown package manager, please install ffmpeg manually'"
                ff_kill = None
            exit_code = -1
            ff_logs: list[str] = []
            for attempt in range(1, 4):
                exit_code, ff_out, ff_err = _run_cmd(client, ffmpeg_cmd, timeout=300, kill_pattern=ff_kill)
                ff_logs = (ff_out + ff_err).strip().splitlines()[-5:]
                if exit_code == 0:
                    break
                if _is_non_retryable(ff_out + ff_err):
                    ff_logs = ff_logs + [
                        "该错误属于权限/锁/缺少工具，重试不可能成功，已停止",
                        "无 root 时: conda install -y -c conda-forge ffmpeg",
                    ]
                    break
                if attempt < 3:
                    yield step_result(3, f"FFmpeg install failed, retry {attempt}/2...", "running", ff_logs)
            if exit_code != 0:
                yield step_result(3, "FFmpeg install", "failed", ff_logs,
                                  f"FFmpeg install failed (exit {exit_code})")
                return
            yield step_result(3, "FFmpeg installed", "done", ff_logs, "FFmpeg ready for torchcodec")

        # ── Step 4: Ensure PyTorch ──────────────────────────────────────────
        # Re-fetching the ~3GB torch wheel is the slowest part of an install, so
        # skip it when a CUDA-capable torch is already importable. Version pins
        # come from lerobot's own metadata (never hardcoded), and the CUDA-specific
        # index is only used when the driver is actually there to match against —
        # otherwise pip resolves torch exactly like the official install does.
        torch_spec, tv_spec = local_torch_constraints()
        torch_reqs = f'"torch{torch_spec}" "torchvision{tv_spec}"'
        yield step_result(4, "Checking PyTorch...", "running",
                          [f"lerobot 要求: torch{torch_spec or '(未约束)'} "
                           f"torchvision{tv_spec or '(未约束)'}",
                           f"CUDA {cuda_tag}" if cuda_tag else "无可用 GPU/驱动"])
        code_t, out_t, t_err = _run_cmd(
            client,
            f"{python} -c \"import torch, torchvision;"
            " print(torch.__version__, torch.cuda.is_available(), torchvision.__version__)\"",
            timeout=120,
        )
        torch_present = code_t == 0 and bool(out_t.strip())
        if torch_present and "True" in out_t:
            yield step_result(4, "PyTorch present", "done", [out_t.strip()],
                              "Skipped install: CUDA-capable torch already importable")
        elif torch_present and not driver_ok:
            yield step_result(
                4, "跳过 PyTorch（无可用 GPU）", "done", [out_t.strip()[:200]],
                "torch 已存在但看不到 CUDA：这是驱动/容器问题，不是包问题",
                hint="在平台控制台重启实例让 nvidia-smi 正常后重跑「安装环境」；"
                     "若版本不满足 lerobot 约束（"
                     f"torch{torch_spec}），届时会按官方约束重装。")
        else:
            index_flag = f" --index-url {pytorch_index}/{cuda_tag}" if cuda_tag else ""
            upgrade_flag = " --upgrade" if cuda_tag else ""
            note = (f"按驱动匹配 CUDA {cuda_tag}" if cuda_tag
                    else "无 GPU/驱动：不加 CUDA 索引，按 lerobot 官方依赖解析安装")
            yield step_result(4, "Installing PyTorch...", "running",
                              [note, torch_reqs.replace('"', "")])
            torch_cmd = (
                f"{workdir_prefix}{python} -m pip install{upgrade_flag}"
                f"{pip_mirror_flag} {torch_reqs}{index_flag}"
                f" 2>&1 | tail -20"
            )
            exit_code = -1
            torch_logs: list[str] = []
            for attempt in range(1, 3):
                exit_code, out, err = _run_cmd(client, torch_cmd, timeout=1800, kill_pattern=f"{python} -m pip")
                torch_logs = (out + err).strip().splitlines()[-10:]
                if exit_code == 0:
                    break
                if attempt < 2:
                    yield step_result(4, "PyTorch install failed, retry 1/1...", "running", torch_logs)
            if exit_code != 0:
                yield step_result(4, "PyTorch install", "failed", torch_logs,
                                  f"PyTorch install failed (exit {exit_code})",
                                  hint="多为网络或磁盘：~3GB 下载容易被重置（重试或勾选国内镜像），"
                                       "并用 df -h 确认空间。版本约束取自 lerobot 元数据"
                                       f"（torch{torch_spec or '(未约束)'}），不要手装越界版本。")
                return
            yield step_result(4, "PyTorch installed", "done", torch_logs,
                              "PyTorch + CUDA ready" if cuda_tag
                              else "PyTorch 已按 lerobot 约束安装（本机无 GPU，CUDA 未验证）")

        # ── Step 5: Install lerobot ─────────────────────────────────────────
        # "training" is always merged in: a policy-specific extra like "smolvla"
        # alone omits dataset deps. Only the selected policies' extras are
        # installed — whatever another job needs comes in at launch.
        extras = policy_union_extras(extra, policies)
        pkg = f"lerobot[{extras}]"
        if target_version:
            pkg += f"=={target_version}"
        yield step_result(
            5, f"Installing {pkg}...", "running",
            [f"目标版本 = 本地 lerobot {target_version or '(未知，不钉版本)'}"],
        )
        code_l, out_l, _ = _run_cmd(
            client,
            f"{python} -c \"import importlib.metadata as m; print(m.version('lerobot'))\"",
            timeout=120,
        )
        remote_v = out_l.strip().splitlines()[-1].strip() if code_l == 0 and out_l.strip() else ""
        if target_version and remote_v == target_version:
            yield step_result(5, "lerobot already aligned", "done", [f"lerobot {remote_v}"],
                              f"远端已是本地版本 {remote_v}，跳过安装")
        else:
            # No --upgrade: step 4 already installed torch/torchvision. -U would
            # re-resolve them and re-download the ~450MB torch wheel from the pypi
            # mirror — a different URL than step 4's index, so pip's HTTP cache
            # (keyed by URL) can't reuse it.
            lerobot_cmd = (
                f"{workdir_prefix}{python} -m pip install"
                f"{pip_mirror_flag} {shlex.quote(pkg)}"
                f" 2>&1 | tail -20"
            )
            exit_code = -1
            lerobot_logs: list[str] = []
            for attempt in range(1, 3):
                exit_code, out, err = _run_cmd(client, lerobot_cmd, timeout=1800, kill_pattern=f"{python} -m pip")
                lerobot_logs = (out + err).strip().splitlines()[-10:]
                if exit_code == 0:
                    break
                if attempt < 2:
                    yield step_result(5, f"{pkg} install failed, retry 1/1...", "running", lerobot_logs)
            if exit_code != 0:
                yield step_result(5, f"{pkg} install", "failed", lerobot_logs,
                                  f"{pkg} install failed (exit {exit_code})",
                                  hint="常见三种原因：镜像源上没有这个版本的 lerobot wheel（去掉版本钉"
                                       "或改用官方源）、磁盘写满（df -h 确认）、连接被重置（重试或换国内镜像）。"
                                       "日志末尾通常直接写着是哪一种。")
                return
            yield step_result(5, f"{pkg} installed", "done", lerobot_logs, f"{pkg} ready")

        # ── Step 5b: Align torchcodec with the installed torch ABI ──────────
        # torchcodec ships per-torch builds but its metadata doesn't pin
        # torch, so pip can grab one newer than our torch — importing it then
        # dies with "undefined symbol: torch_dtype_..." at train startup.
        # Probe-load; on failure walk versions down from torch's minor.
        yield step_result(5, "Checking torchcodec...", "running", [])
        align_env = (
            "export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple "
            "PIP_TRUSTED_HOST=mirrors.aliyun.com && "
            if use_cn_mirror
            else ""
        )
        align_script = (
            "import subprocess, sys, torch\n"
            "def loads():\n"
            "    return subprocess.run([sys.executable, '-c',\n"
            "        'from torchcodec.decoders import VideoDecoder'],\n"
            "        capture_output=True).returncode == 0\n"
            "if loads():\n"
            "    print('torchcodec ok')\n"
            "else:\n"
            "    minor = int(torch.__version__.split('.')[1])\n"
            "    done = False\n"
            "    for v in [f'0.{i}' for i in range(minor, 3, -1)]:\n"
            "        subprocess.run([sys.executable, '-m', 'pip', 'install',\n"
            "            '-q', '--no-deps', 'torchcodec==' + v], check=False)\n"
            "        if loads():\n"
            "            print('torchcodec aligned to', v); done = True; break\n"
            "    if not done:\n"
            "        print('torchcodec alignment failed')\n"
        )
        exit_code, tc_out, tc_err = _run_cmd(
            client,
            f"{align_env}{python} - <<'PY'\n{align_script}PY",
            timeout=600,
            kill_pattern=f"{python} -m pip",
        )
        tc_logs = (tc_out + tc_err).strip().splitlines()[-5:]
        if "alignment failed" in tc_out:
            yield step_result(
                5, "torchcodec", "done", tc_logs,
                "torchcodec could not be aligned — training may need "
                "--dataset.video_backend=pyav",
            )
        else:
            yield step_result(5, "torchcodec", "done", tc_logs, tc_logs[-1] if tc_logs else "ok")

        # ── Step 6: Install training deps ───────────────────────────────────
        deps = ["accelerate", "wandb", "tensorboard"]
        yield step_result(6, "Installing training deps...", "running", [])
        deps_cmd = (
            f"{workdir_prefix}{python} -m pip install"
            f"{pip_mirror_flag} {' '.join(deps)}"
            f" 2>&1 | tail -10"
        )
        exit_code = -1
        deps_logs: list[str] = []
        for attempt in range(1, 3):
            exit_code, out, err = _run_cmd(client, deps_cmd, timeout=900, kill_pattern=f"{python} -m pip")
            deps_logs = (out + err).strip().splitlines()[-5:]
            if exit_code == 0:
                break
            if attempt < 2:
                yield step_result(6, "Training deps install failed, retry 1/1...", "running", deps_logs)
        if exit_code != 0:
            yield step_result(6, "Training deps install", "failed", deps_logs,
                              f"Training deps install failed (exit {exit_code})",
                              hint="accelerate/wandb/tensorboard 装不上通常是网络或磁盘；"
                                   "其中只有 accelerate 是训练必需，可先跳过 wandb/tensorboard 重试。")
            return
        yield step_result(6, "Training deps installed", "done", deps_logs,
                          "accelerate, wandb, tensorboard ready")

        # ── Step 7: Verify ──────────────────────────────────────────────────
        yield step_result(7, "Verifying installation...", "running", [])
        verify_script = (
            "import torch, json, shutil\n"
            "r = {}\n"
            "r['torch'] = torch.__version__\n"
            "r['cuda'] = torch.cuda.is_available()\n"
            "r['cuda_ver'] = torch.version.cuda\n"
            "r['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'\n"
            "r['ffmpeg'] = shutil.which('ffmpeg') is not None\n"
            "try:\n"
            "    import lerobot; r['lerobot'] = getattr(lerobot, '__version__', 'ok')\n"
            "except: r['lerobot'] = 'MISSING'\n"
            "try:\n"
            "    import datasets, accelerate, torchcodec.decoders; r['training_extras'] = True\n"
            "except: r['training_extras'] = False\n"
            "ready, missing = [], []\n"
            "from lerobot.policies.factory import get_policy_class\n"
            "for t in ('act','diffusion','pi0','pi0_fast','smolvla','groot','tdmpc','vqbet'):\n"
            "    try:\n"
            "        get_policy_class(t); ready.append(t)\n"
            "    except Exception:\n"
            "        missing.append(t)\n"
            "try:\n"
            "    from lerobot.rewards.factory import get_reward_model_class\n"
            "    get_reward_model_class('reward_classifier'); ready.append('reward_classifier')\n"
            "except Exception:\n"
            "    missing.append('reward_classifier')\n"
            "r['policies_ready'] = ready\n"
            "r['policies_missing'] = missing\n"
            "try:\n"
            "    import accelerate; r['accelerate'] = accelerate.__version__\n"
            "except: r['accelerate'] = 'MISSING'\n"
            "print(json.dumps(r))\n"
        )
        _stdin, stdout, stderr = client.exec_command(f"{python} -", timeout=180)
        _stdin.write(verify_script)
        _stdin.channel.shutdown_write()
        vout = stdout.read().decode().strip()
        verr = stderr.read().decode().strip()
        verify_logs = vout.splitlines()
        if verr:
            verify_logs.extend(verr.splitlines())

        try:
            vdata = json.loads(vout.splitlines()[-1])
            missing = vdata.get("policies_missing", []) or []
            # Unselected policies are expected to be missing when the user
            # installed a subset — only the selection gates success.
            blocking_pol = [p for p in missing if (not policies or p in policies)]
            remote_lr = vdata.get("lerobot")
            version_off = bool(target_version) and remote_lr not in (target_version, "MISSING", None)

            problems: list[str] = []
            notes: list[str] = []
            if not vdata.get("cuda"):
                if not driver_ok:
                    notes.append(
                        "软件装齐了，但这台机器当前看不到 GPU/驱动：可以安装，不能训练。"
                        "重启实例让 nvidia-smi 正常后重跑「安装环境」，会按驱动补装匹配的 torch。"
                    )
                else:
                    problems.append(
                        "驱动正常但 torch 看不到 CUDA：torch 构建与驱动不匹配（或实例刚重启没挂上卡）。"
                        "重跑本安装会按驱动重装匹配的 torch。"
                    )
            if remote_lr == "MISSING":
                problems.append("lerobot 没装上：看 step 5 的 pip 日志，多为镜像站缺该版本 wheel 或磁盘写满。")
            elif version_off:
                problems.append(
                    f"远端 lerobot {remote_lr} ≠ 本地 {target_version}：checkpoint 可能无法在本地 lerobot "
                    "加载。确认镜像源有该版本 wheel 后重跑，或本地升到与之一致。"
                )
            if not vdata.get("ffmpeg"):
                problems.append("ffmpeg 缺失：torchcodec 无法解码视频，训练会报 libav 相关错误。")
            if not vdata.get("training_extras"):
                problems.append("训练依赖不全（accelerate / datasets / torchcodec）：lerobot-train 起不来。")
            if blocking_pol:
                problems.append(
                    f"这些策略仍不可训：{blocking_pol}。训练任务启动时会自动补该策略的 extras，"
                    "也可以重跑本安装并勾选它们一次装齐。"
                )

            ok = not problems
            summary = (
                f"PyTorch {vdata.get('torch')}, CUDA {vdata.get('cuda_ver')}, "
                f"GPU {vdata.get('gpu')}, lerobot {remote_lr}（本地 {target_version or '?'}）, "
                f"FFmpeg {'ok' if vdata.get('ffmpeg') else 'MISSING'}, "
                f"policies ready: {len(vdata.get('policies_ready', []))}/9"
            )
            if notes:
                summary += f" — {notes[0]}"
            yield step_result(
                7, "验证通过" if ok else "验证未通过",
                "done" if ok else "failed",
                verify_logs + [f"! {p}" for p in problems] + [f"· {n}" for n in notes],
                summary,
                hint=" ; ".join(problems or notes),
            )
        except Exception as exc:
            yield step_result(7, "Verification", "failed", verify_logs,
                              f"验证脚本输出无法解析：{exc}",
                              hint="多为远端解释器中途变了（conda 环境被切走）或 python 不可用："
                                   "重新点「检查环境」确认 remote_python。")

    except Exception as exc:
        low = str(exc).lower()
        yield step_result(
            0, "安装中断", "failed", [str(exc)[:500]],
            _explain_connect_error(exc) if any(
                k in low for k in ("banner", "auth", "connect", "socket", "timeout")
            ) else f"安装过程中断：{str(exc)[:300]}",
            hint="远端已完成的步骤不会回滚；修好原因后从「检查环境」重新开始即可，"
                 "已满足的步骤会自动跳过。",
        )
    finally:
        client.close()


def install_lerobot_remote(config_id: str, extra: str = "") -> dict:
    """Legacy non-streaming install — runs all steps and returns final result."""
    final = {"success": False, "message": "", "lines": []}
    for update in install_lerobot_remote_streaming(config_id, extra):
        if update["status"] == "failed":
            return {
                "success": False,
                "message": update.get("message", update["label"]),
                "lines": update["logs"],
            }
        if update["step"] == update["total_steps"] and update["status"] == "done":
            return {
                "success": True,
                "message": update.get("message", "Installation complete"),
                "lines": update["logs"],
            }
    return {"success": False, "message": "Installation interrupted", "lines": []}
