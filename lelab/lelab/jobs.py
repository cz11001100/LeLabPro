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

"""Job lifecycle and registry for trainings (and, in future, other long-running
work). One JobRunner instance owns one subprocess; the JobRegistry owns the
overall state, including history persisted to disk under outputs/train/."""

from __future__ import annotations

import builtins
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from .train import TrainingRequest

logger = logging.getLogger(__name__)


JobState = Literal["running", "done", "failed", "interrupted"]


class JobTarget(BaseModel):
    """Where a job should run. `local` ⇒ LocalJobRunner. `hf_cloud` requires
    a non-empty `flavor` from HfApi.list_jobs_hardware(). `ssh` requires a
    valid `ssh_config_id` referencing a saved SSH connection."""

    runner: Literal["local", "hf_cloud", "ssh"] = "local"
    flavor: str | None = None
    ssh_config_id: str | None = None


class TrainingMetrics(BaseModel):
    current_step: int = 0
    total_steps: int = 0
    current_loss: float | None = None
    current_lr: float | None = None
    grad_norm: float | None = None
    eta_seconds: float | None = None


class LogLine(BaseModel):
    timestamp: float
    message: str


class JobRecord(BaseModel):
    id: str
    name: str
    state: JobState
    config: TrainingRequest
    output_dir: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    metrics: TrainingMetrics = TrainingMetrics()
    runner: Literal["local", "hf_cloud", "ssh", "imported"] = "local"
    # SSH runner only: the saved SSH config id, needed to reconnect for SFTP
    # downloads (checkpoint export).
    ssh_config_id: str | None = None
    # SSH runner only: the remote path where checkpoints are saved.
    remote_output_dir: str | None = None
    # PID of the detached subprocess (local runner only); survives uvicorn
    # --reload so a fresh registry can re-attach by tailing the log file.
    process_pid: int | None = None
    # HF Jobs identifiers (hf_cloud runner only)
    hf_job_id: str | None = None
    hf_flavor: str | None = None
    hf_repo_id: str | None = None
    hf_job_url: str | None = None
    # Captured from training stdout the first time wandb prints the run URL.
    wandb_run_url: str | None = None
    # Number of checkpoints currently visible (local: filesystem; cloud:
    # Hub repo). Filled in by JobRegistry.list/get; persisted as zero.
    checkpoint_count: int = 0
    # Validation state and results
    validation_state: Literal["none", "pending", "running", "done", "failed"] = "none"
    validation_results: dict | None = None
    # SSH only: the last attempt ended in "could not reach the pod", which is not
    # evidence the training stopped, so the watchdog keeps re-attaching the job.
    remote_unconfirmed: bool = False
    reconcile_attempts: int = 0


class JobCheckpoint(BaseModel):
    """One checkpoint produced by a training job.

    `ref` is opaque to the frontend; the inference handler resolves it back
    to a usable `--policy.path` value (a local dir for both sources, after
    snapshot_download for hub refs)."""

    step: int
    source: Literal["local", "hub"]
    ref: str


class MetricsHistoryPoint(BaseModel):
    """One (step, metrics) sample reconstructed from a job's log.jsonl.

    Used by GET /jobs/{id}/metrics-history to seed the monitoring charts.
    A point is emitted for each log line that carried a `step: ... loss: ...`
    payload (the log-freq lines from lerobot). Tqdm progress lines are
    skipped — they carry step + ETA but no loss/lr/grdn."""

    step: int
    loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID exists. Cheap; uses signal 0."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@runtime_checkable
class JobRunner(Protocol):
    """Backend interface for running one job. LocalJobRunner is the only impl
    today; remote runners (SSH, Slurm) drop in here later. @runtime_checkable
    lets `isinstance(r, JobRunner)` work in tests / sanity checks."""

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def returncode(self) -> int | None: ...
    def stream_log_lines(self) -> list[LogLine]: ...
    def wandb_run_url(self) -> str | None: ...


# tqdm progress: "Training:   1%|▏         | 125/10000 [02:02<2:36:10,  1.05step/s]"
_TQDM_RE = re.compile(r"Training:\s*\d+%[^|]*\|[^|]*\|\s*(\d+)/(\d+)\s*\[(?:[\d:]+)<([\d:]+)")

# Wandb prints something like "wandb: 🚀 View run at https://wandb.ai/<entity>/<project>/runs/<id>"
# when it boots. We capture the first URL of that shape we see.
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s/]+/[^\s/]+/runs/[A-Za-z0-9]+")


def extract_wandb_run_url(line: str) -> str | None:
    match = _WANDB_URL_RE.search(line)
    return match.group(0) if match else None


def _parse_duration(s: str) -> float | None:
    """Parse tqdm's HH:MM:SS or MM:SS into seconds. Returns None on '?'."""
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def parse_metrics_into(line: str, metrics: TrainingMetrics) -> None:
    """Update `metrics` in-place from one stdout line.

    Two complementary sources:
      * tqdm progress for current_step + total_steps + ETA (~1s cadence).
      * 'INFO ... step:N smpl:... loss:X grdn:Y lr:Z ...' for loss/lr/grdn
        (only at log_freq cadence, default every 250 steps).
    """
    try:
        tqdm_match = _TQDM_RE.search(line)
        if tqdm_match:
            try:
                metrics.current_step = int(tqdm_match.group(1))
                total = int(tqdm_match.group(2))
                if total > 0:
                    metrics.total_steps = total
                eta = _parse_duration(tqdm_match.group(3))
                if eta is not None:
                    metrics.eta_seconds = eta
            except (ValueError, IndexError):
                pass

        if "step:" in line and "loss:" in line:
            with contextlib.suppress(ValueError):
                metrics.current_step = int(line.split("step:")[1].split()[0].replace(",", ""))
            with contextlib.suppress(ValueError):
                metrics.current_loss = float(line.split("loss:")[1].split()[0])
            if "lr:" in line:
                with contextlib.suppress(ValueError):
                    metrics.current_lr = float(line.split("lr:")[1].split()[0])
            if "grdn:" in line:
                with contextlib.suppress(ValueError):
                    metrics.grad_norm = float(line.split("grdn:")[1].split()[0])

    except Exception as exc:
        logger.debug("Error parsing log line %r: %s", line, exc)


class SubprocessJobRunner:
    """Spawn a subprocess and pump its stdout into a log file + in-memory queue.

    The shared engine behind both LocalJobRunner (which runs `lerobot-train`
    directly) and HfCloudJobRunner (which runs `lerobot-train --job.target=...`,
    a local process that submits the job and streams the remote logs to its own
    stdout). Subclasses override `_on_line` to inspect each stdout line for
    runner-specific markers (e.g. the HF job id / page URL).
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path | None = None,
    ) -> None:
        self._metrics = metrics
        self._process: subprocess.Popen | None = None
        self._log_queue: Queue[LogLine] = Queue()
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._log_file_path = log_file_path
        self._log_file = None  # type: ignore[assignment]
        self._wandb_run_url: str | None = None

    def _open_log_file(self) -> None:
        """Open the persistent log sink (one JSON line per consumed line).
        Held open for the consumer thread's lifetime so we don't reopen per
        write; _consume_lines closes it when its iterator is exhausted."""
        if self._log_file_path is not None:
            self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_file_path.open("a", buffering=1)

    def _spawn(self, cmd: list[str], thread_name: str, extra_env: dict[str, str] | None = None) -> None:
        """Open the log sink, launch `cmd`, and start the stdout pump thread."""
        if self._process is not None:
            raise RuntimeError(f"{type(self).__name__} already started")

        self._open_log_file()

        # PYTHONUNBUFFERED makes the child's stdout flush per line. Without it
        # block-buffering hides log lines from our parser for many seconds.
        child_env = os.environ.copy()
        child_env["PYTHONUNBUFFERED"] = "1"
        child_env.update(extra_env or {})

        # start_new_session=True puts the child in its own session/process
        # group. Without it, signals sent to the uvicorn worker (e.g. when
        # --reload restarts it on a .py file change) cascade to the child
        # and kill the training. With it, the child survives reloads; the
        # next worker re-attaches via the reattach path.
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=child_env,
            start_new_session=True,
        )

        self._monitor_thread = threading.Thread(target=self._pump_stdout, name=thread_name, daemon=True)
        self._monitor_thread.start()

    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def stop(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._stop_event.set()
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess did not terminate in 10s, killing")
                self._process.kill()
                self._process.wait()
        except Exception as exc:
            logger.exception("Error stopping subprocess: %s", exc)

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def returncode(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def stream_log_lines(self) -> list[LogLine]:
        """Drain whatever has accumulated since the last call."""
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    # -- internals --

    def _on_line(self, line: str) -> None:
        """Hook for subclasses to inspect each stdout line. Default: no-op."""

    def _consume_lines(self, lines: Iterable[str]) -> None:
        """Drive each text line through the metric/marker parse + log.jsonl
        append + in-memory queue. Source-agnostic: a subprocess's stdout
        (LocalJobRunner) or a remote log stream iterator (cloud reattach) feed
        the same pipeline. Closes the log file when the iterator is exhausted.
        """
        try:
            for line in lines:
                if self._stop_event.is_set():
                    break
                stripped = line.rstrip()
                if not stripped:
                    continue
                parse_metrics_into(stripped, self._metrics)
                if self._wandb_run_url is None:
                    url = extract_wandb_run_url(stripped)
                    if url is not None:
                        self._wandb_run_url = url
                self._on_line(stripped)
                log_line = LogLine(timestamp=time.time(), message=stripped)
                if self._log_file is not None:
                    try:
                        self._log_file.write(log_line.model_dump_json() + "\n")
                    except Exception as exc:  # pragma: no cover — best-effort persist
                        logger.exception("Error writing to log file: %s", exc)
                # Cap queue so a chatty source can't grow memory unbounded.
                if self._log_queue.qsize() >= 1000:
                    with contextlib.suppress(Empty):
                        self._log_queue.get_nowait()
                self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Error consuming log lines: %s", exc)
        finally:
            if self._log_file is not None:
                with contextlib.suppress(Exception):
                    self._log_file.close()
                self._log_file = None

    def _pump_stdout(self) -> None:
        assert self._process is not None
        self._consume_lines(iter(self._process.stdout.readline, ""))


class LocalJobRunner(SubprocessJobRunner):
    """Run a training as a local subprocess.

    The runner is single-shot: instantiate a fresh one per job. Lifetime of
    the underlying subprocess is bounded by this object's existence in memory.
    """

    def start(
        self,
        job_id: str,
        config: TrainingRequest,
        output_dir: str,
    ) -> None:
        # Build the command via the helper that lives in train.py.
        from .train import build_training_command, policy_weights_are_bf16  # avoid import cycle at module load

        cmd = build_training_command(config, output_dir, sys.executable)
        logger.info("Starting job %s: %s", job_id, " ".join(cmd))
        # Policies without a `dtype` config field leave the mode to the
        # environment, and an fp16 GradScaler cannot unscale bf16 gradients.
        extra_env = {"ACCELERATE_MIXED_PRECISION": "bf16"} if policy_weights_are_bf16(config.policy_type) else None
        self._spawn(cmd, thread_name=f"job-{job_id}-stdout", extra_env=extra_env)


class TailingJobRunner:
    """Re-attaches to a detached subprocess after a uvicorn reload.

    We can't recover the original Popen object across processes, so we don't
    own stdout. Instead we tail the persisted log file and watch the pid.
    Implements the JobRunner Protocol so JobRegistry can use it interchangeably
    with LocalJobRunner.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        pid: int,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._pid = pid
        self._log_queue: Queue[LogLine] = Queue()
        self._tail_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Replay everything that's already on disk so the parser catches up
        # on metrics, then tail from the current EOF.
        self._tail_offset = 0
        self._wandb_run_url: str | None = None

    def start(self, job_id: str, config: TrainingRequest, output_dir: str) -> None:
        # Required by JobRunner Protocol but irrelevant here; the subprocess
        # we're tailing was started by a previous uvicorn worker.
        raise RuntimeError("TailingJobRunner reattaches to an existing pid; use start_tailing() instead")

    def start_tailing(self) -> None:
        if self._tail_thread is not None:
            return
        self._tail_thread = threading.Thread(
            target=self._tail_loop, name=f"job-tail-{self._pid}", daemon=True
        )
        self._tail_thread.start()

    def stop(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(self._pid, signal.SIGTERM)
        self._stop_event.set()

    def is_running(self) -> bool:
        return _pid_alive(self._pid)

    def returncode(self) -> int | None:
        # We can't reap a process from another session, so we don't know the
        # actual exit code. Return 0 once the pid is gone — the watchdog
        # finalises as "done" rather than "failed", which is the better
        # default for a detached training that completed normally.
        if _pid_alive(self._pid):
            return None
        return 0

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def pid(self) -> int | None:
        return self._pid

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    # -- internals --

    def _tail_loop(self) -> None:
        """Read lines as they arrive in log_file_path. Exits when pid dies
        AND there are no more new lines to read."""
        try:
            while not self._stop_event.is_set():
                if not self._log_file_path.exists():
                    if not _pid_alive(self._pid):
                        return
                    self._stop_event.wait(0.5)
                    continue
                with self._log_file_path.open() as f:
                    f.seek(self._tail_offset)
                    while not self._stop_event.is_set():
                        raw = f.readline()
                        if not raw:
                            self._tail_offset = f.tell()
                            if not _pid_alive(self._pid):
                                return
                            self._stop_event.wait(0.5)
                            continue
                        try:
                            log_line = LogLine.model_validate_json(raw.strip())
                        except Exception:
                            continue
                        parse_metrics_into(log_line.message, self._metrics)
                        if self._wandb_run_url is None:
                            url = extract_wandb_run_url(log_line.message)
                            if url is not None:
                                self._wandb_run_url = url
                        if self._log_queue.qsize() >= 1000:
                            with contextlib.suppress(Empty):
                                self._log_queue.get_nowait()
                        self._log_queue.put(log_line)
        except Exception as exc:
            logger.exception("Tailing loop error: %s", exc)


_PERSIST_THROTTLE_SECONDS = 1.0

# SSH checkpoint discovery runs off the request path. A dead/recycled pod makes
# paramiko stall for its full connect timeout, and the jobs list fans out over
# every stored record, so probing inline once meant a multi-minute /jobs.
_SSH_CKPT_TTL_SECONDS = 60.0
_SSH_CKPT_FAIL_TTL_SECONDS = 300.0

# SSH jobs left as 'interrupted' only because the pod could not be reached are
# re-probed on this cadence, up to the attempt cap (~1h of trying).
_RECONCILE_INTERVAL_SECONDS = 60.0
_RECONCILE_MAX_ATTEMPTS = 60

# Markers our own runners write when the only fact established was "could not
# reach the remote". Used to resurrect records stranded by such a failure.
_UNCONFIRMED_MARKERS = ("无法连接远端", "读不到远端退出码", "没有确认退出", "无法确认训练")


def _looks_unconfirmed(message: str | None) -> bool:
    return bool(message) and any(marker in message for marker in _UNCONFIRMED_MARKERS)


def _list_local_checkpoints(output_dir: str) -> list[JobCheckpoint]:
    """Scan an output dir for valid checkpoint subdirectories.

    A directory under <output_dir>/checkpoints/ is a valid checkpoint iff
    its name parses to an int and it contains pretrained_model/config.json.
    """
    root = Path(output_dir) / "checkpoints"
    if not root.is_dir():
        return []
    out: list[JobCheckpoint] = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            step = int(entry.name)
        except ValueError:
            continue
        config_json = entry / "pretrained_model" / "config.json"
        if not config_json.is_file():
            continue
        out.append(
            JobCheckpoint(
                step=step,
                source="local",
                ref=str((entry / "pretrained_model").resolve()),
            )
        )
    out.sort(key=lambda c: c.step)
    return out


def _list_ssh_checkpoints(record: JobRecord) -> list[JobCheckpoint]:
    """List checkpoints on the SSH remote. Blocking; empty means "none found"."""
    return _list_ssh_checkpoints_status(record) or []


def _list_ssh_checkpoints_status(record: JobRecord) -> list[JobCheckpoint] | None:
    """SSH checkpoint discovery that tells "remote unreachable" (None) apart
    from "no checkpoints yet" ([]), so the background cache can back off from a
    recycled pod instead of repeatedly reporting zero after a 15s connect stall.

    Two cases:
    1. New jobs with remote_output_dir set — scan that path via SFTP.
    2. Old jobs without remote_output_dir — use `find` to locate checkpoint
       dirs under the lerobot default output path.
    """
    if not record.ssh_config_id:
        return None
    from .ssh_config import get_ssh_config

    cfg = get_ssh_config(record.ssh_config_id)
    if cfg is None:
        return None
    try:
        import paramiko
    except ImportError:
        logger.warning("paramiko not installed; cannot list SSH checkpoints")
        return None

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        connect_kwargs: dict = {
            "hostname": cfg.host,
            "port": cfg.port,
            "username": cfg.username,
            "timeout": 15,
        }
        if cfg.key_path:
            connect_kwargs["key_filename"] = os.path.expanduser(cfg.key_path)
        elif cfg.password:
            connect_kwargs["password"] = cfg.password
        client.connect(**connect_kwargs)

        if record.remote_output_dir:
            remote_dir = record.remote_output_dir
            if remote_dir.startswith("~"):
                _stdin, stdout, _stderr = client.exec_command("echo $HOME", timeout=10)
                home = stdout.read().decode().strip()
                if home:
                    remote_dir = remote_dir.replace("~", home, 1)
            return _ssh_list_checkpoints_sftp(client, f"{remote_dir}/checkpoints")

        # Old job: discover checkpoint path via find
        return _ssh_find_checkpoints(client, record)
    except Exception as exc:
        logger.warning("Failed to list SSH checkpoints for %s: %s", record.id, exc)
        return None
    finally:
        client.close()


def _ssh_list_checkpoints_sftp(client, remote_ckpt_dir: str) -> list[JobCheckpoint]:
    """Scan a known checkpoints/ directory via SFTP."""
    sftp = client.open_sftp()
    try:
        try:
            entries = sftp.listdir(remote_ckpt_dir)
        except FileNotFoundError:
            return []
        out: list[JobCheckpoint] = []
        for name in entries:
            try:
                step = int(name)
            except ValueError:
                continue
            config_json = f"{remote_ckpt_dir}/{name}/pretrained_model/config.json"
            try:
                sftp.stat(config_json)
            except FileNotFoundError:
                continue
            out.append(
                JobCheckpoint(
                    step=step,
                    source="local",
                    ref=f"{remote_ckpt_dir}/{name}/pretrained_model",
                )
            )
        out.sort(key=lambda c: c.step)
        return out
    finally:
        sftp.close()


def _ssh_find_checkpoints(client, record: JobRecord) -> list[JobCheckpoint]:
    """Use `find` on the remote to discover checkpoint dirs for old SSH jobs."""
    find_cmd = (
        "find /root/outputs /home -path '*/checkpoints/*/pretrained_model/config.json' "
        "-type f 2>/dev/null | head -50"
    )
    _stdin, stdout, _stderr = client.exec_command(find_cmd, timeout=30)
    lines = stdout.read().decode().strip().splitlines()
    out: list[JobCheckpoint] = []
    seen_steps: set[int] = set()
    for line in lines:
        # line: /some/path/checkpoints/001000/pretrained_model/config.json
        parts = line.split("/checkpoints/")
        if len(parts) != 2:
            continue
        rest = parts[1]  # 001000/pretrained_model/config.json
        step_str = rest.split("/")[0]
        try:
            step = int(step_str)
        except ValueError:
            continue
        if step in seen_steps:
            continue
        seen_steps.add(step)
        ckpt_base = f"{parts[0]}/checkpoints/{step_str}/pretrained_model"
        out.append(
            JobCheckpoint(step=step, source="local", ref=ckpt_base)
        )
    out.sort(key=lambda c: c.step)
    return out


_CLOUD_CKPT_TTL_SECONDS = 30.0
_CKPT_PATH_RE = re.compile(r"^checkpoints/(\d+)/pretrained_model/config\.json$")


def _hub_checkpoints_from_files(files, repo_id: str) -> list[JobCheckpoint]:
    """Parse a repo file listing into checkpoints. The ref preserves the
    original zero-padded directory name (e.g. 000050); JobCheckpoint.step is
    the int form for sorting and UI display."""
    seen: dict[int, JobCheckpoint] = {}
    for path in files:
        m = _CKPT_PATH_RE.match(path)
        if not m:
            continue
        step_dir = m.group(1)
        step = int(step_dir)
        seen[step] = JobCheckpoint(
            step=step,
            source="hub",
            ref=f"{repo_id}@checkpoints/{step_dir}",
        )
    out = list(seen.values())
    out.sort(key=lambda c: c.step)
    return out


def _list_imported_local(path: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported local directory.

    A checkpoints/<step>/pretrained_model tree → reuse _list_local_checkpoints.
    Otherwise, if the dir itself is a pretrained_model (config.json present) →
    a single step-0 checkpoint. Neither → empty (source moved/unusable)."""
    tree = _list_local_checkpoints(path)
    if tree:
        return tree
    if (Path(path) / "config.json").is_file():
        return [JobCheckpoint(step=0, source="local", ref=str(Path(path).resolve()))]
    return []


def _list_imported_hub(api, repo_id: str) -> list[JobCheckpoint]:
    """Auto-detect the layout of an imported Hub model repo.

    A checkpoints/<step>/pretrained_model tree → reuse the tree parse.
    Otherwise, a root config.json → a single step-0 checkpoint with a
    'repo@root' ref (the whole repo is the pretrained_model)."""
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except Exception:
        return []
    tree = _hub_checkpoints_from_files(files, repo_id)
    if tree:
        return tree
    if "config.json" in files:
        return [JobCheckpoint(step=0, source="hub", ref=f"{repo_id}@root")]
    return []


_LANGUAGE_CONDITIONED_POLICY_TYPES = {"smolvla", "pi0", "pi0_fast", "pi05"}


_HUB_CKPT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@checkpoints/(?P<step_dir>\d+)$")
_HUB_ROOT_REF_RE = re.compile(r"^(?P<repo>[^@]+)@root$")


def _read_train_config_dataset_repo_id(ckpt_dir: Path) -> str:
    try:
        train_config_path = ckpt_dir / "train_config.json"
        if train_config_path.is_file():
            with open(train_config_path) as f:
                return str((json.load(f).get("dataset") or {}).get("repo_id") or "")
    except Exception as exc:
        logger.warning("Failed to read dataset_repo_id from train_config.json: %s", exc)
    return ""


def _dataset_task_strings(repo_id: str) -> list[str]:
    """Verbatim task strings a dataset was recorded with, in tasks.parquet order.

    Empty when the dataset isn't cached locally; the UI then falls back to free
    text. Language-conditioned policies embed the task string, so a reworded
    prompt at inference is out-of-distribution — the UI offers these exact
    strings to keep inference inside the training distribution.
    """
    if not repo_id:
        return []
    cache_root = Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    meta = cache_root / repo_id / "meta"
    try:
        if (meta / "tasks.parquet").is_file():
            import pandas as pd

            df = pd.read_parquet(meta / "tasks.parquet")
            raw = df["task"].tolist() if "task" in df.columns else df.index.tolist()
            return [str(t) for t in raw if str(t).strip()]
        if (meta / "tasks.jsonl").is_file():
            out: list[str] = []
            for line in (meta / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    task = json.loads(line).get("task")
                    if task:
                        out.append(str(task))
            return out
    except Exception as exc:
        logger.warning("Could not read task strings for dataset %s: %s", repo_id, exc)
    return []


def _read_checkpoint_config(ckpt: JobCheckpoint) -> dict[str, object]:
    """Load the pretrained_model/config.json for one checkpoint.

    Keyed on the checkpoint's own source/ref shape so it works for training
    jobs and imports alike:
      * local  → ckpt.ref is the absolute pretrained_model dir.
      * hub    → 'repo@checkpoints/<step_dir>' (a tree) or 'repo@root' (a flat
                 model repo); both resolve via hf_hub_download.
    """
    if ckpt.source == "local":
        with open(Path(ckpt.ref) / "config.json") as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download

    m = _HUB_CKPT_REF_RE.match(ckpt.ref)
    if m:
        repo_id = m.group("repo")
        filename = f"checkpoints/{m.group('step_dir')}/pretrained_model/config.json"
    else:
        m = _HUB_ROOT_REF_RE.match(ckpt.ref)
        if not m:
            raise ValueError(f"Bad hub ref: {ckpt.ref!r}")
        repo_id = m.group("repo")
        filename = "config.json"
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    with open(local_path) as f:
        return json.load(f)


def _generate_job_id(policy_type: str, dataset_repo_id: str) -> str:
    """Build a sortable, collision-free job id from policy type and dataset slug."""
    from .train import _SLUG_RE

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_slug = _SLUG_RE.sub("_", dataset_repo_id).strip("_") or "dataset"
    return f"{policy_type}_{dataset_slug}_{timestamp}"


def _job_dir(output_root: Path, job_id: str) -> Path:
    return output_root / job_id


def _job_log_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "log.jsonl"


def _job_meta_path(output_root: Path, job_id: str) -> Path:
    return _job_dir(output_root, job_id) / "job.json"


class JobAlreadyRunningError(Exception):
    """Raised when start() is called while another local job is running."""


class JobNotFoundError(Exception):
    """Raised when a lookup hits an unknown id."""


class JobNotRunningError(Exception):
    """Raised when stop() is called on a non-running job."""


class JobRegistry:
    """Owns the registry of training jobs and their persistence.

    On instantiation, scans outputs/train/ for existing job.json files. For
    each record marked 'running': local jobs reattach if the pid is alive
    (else 'interrupted'); hf_cloud jobs always reattach and let the tail loop
    drive finalisation.
    """

    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._runners: dict[str, JobRunner] = {}
        self._last_persist_at: dict[str, float] = {}

        self._stop_watchdog = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

        # repo_id -> (expires_at_epoch, checkpoint list)
        self._cloud_ckpt_cache: dict[str, tuple[float, list[JobCheckpoint]]] = {}

        # job_id -> (expires_at_epoch, checkpoint count) for SSH jobs, probed
        # by a small background pool so no request ever waits on SSH. Guarded by
        # its own lock: finalization counts checkpoints while holding self._lock,
        # and self._lock is not reentrant.
        self._ssh_ckpt_cache: dict[str, tuple[float, int]] = {}
        self._ssh_ckpt_inflight: set[str] = set()
        self._ssh_ckpt_pool: ThreadPoolExecutor | None = None
        self._ssh_ckpt_lock = threading.Lock()

        # "job_id:step" -> progress of the (blocking) SFTP checkpoint export.
        # The export POST does not return until the download finishes, so the
        # UI polls get_export_progress() while that request is in flight.
        self._export_progress: dict[str, dict] = {}

        # Fired (best-effort) on every state change: new job, stop initiated,
        # watchdog finalisation, delete. Server wires this to a WebSocket
        # broadcast so the frontend can refetch on-event instead of polling.
        self._on_change: Callable[[], None] | None = None

        # Fired from the watchdog at ~1Hz with a compact snapshot of every
        # running job (id, state, metrics, wandb url, checkpoint count) so
        # the dashboard keeps the progress bar live without refetching /jobs.
        self._on_progress: Callable[[builtins.list[dict]], None] | None = None

        self._migrate_legacy_cwd_jobs()
        self._load_from_disk()
        self._start_watchdog()

    def _migrate_legacy_cwd_jobs(self) -> None:
        """One-shot migration from cwd-relative `outputs/train/` to the new
        absolute root.

        Older lelab versions wrote job dirs to `<cwd>/outputs/train/`, which
        meant history disappeared when you launched from a different cwd. We
        now anchor to ~/.cache/.../outputs/train. On first boot under the new
        layout, move any pre-existing cwd-relative job dirs over and rewrite
        each job.json's `output_dir` field to the new absolute path.

        Idempotent: skipped if (a) the new root is the legacy one itself
        (LELAB_OUTPUT_ROOT=outputs/train still wins for tests), or (b) the
        legacy dir is absent / already empty.
        """
        legacy_root = (Path.cwd() / "outputs" / "train").resolve()
        if legacy_root == self._output_root or not legacy_root.is_dir():
            return

        legacy_dirs = [p for p in legacy_root.iterdir() if p.is_dir()]
        if not legacy_dirs:
            return

        logger.info(
            "Migrating %d legacy job dirs from %s to %s",
            len(legacy_dirs),
            legacy_root,
            self._output_root,
        )
        for src in legacy_dirs:
            dst = self._output_root / src.name
            if dst.exists():
                logger.warning("Migration: %s already exists at destination; skipping", src.name)
                continue
            try:
                shutil.move(str(src), str(dst))
            except Exception as exc:
                logger.warning("Migration: failed to move %s: %s", src.name, exc)
                continue
            self._rewrite_output_dir_in_meta(dst)

        # If the legacy dir is now empty, remove it so subsequent boots skip
        # the scan. A leftover non-dir file keeps it around — that's fine.
        with contextlib.suppress(OSError):
            legacy_root.rmdir()

    def _rewrite_output_dir_in_meta(self, job_dir: Path) -> None:
        """Repoint `output_dir` in a migrated job.json to its new absolute
        path. Pre-migration records stored `outputs/train/<id>/run` which
        no longer resolves once cwd has moved."""
        meta = job_dir / "job.json"
        if not meta.is_file():
            return
        try:
            data = json.loads(meta.read_text())
        except Exception as exc:
            logger.warning("Migration: could not parse %s: %s", meta, exc)
            return
        data["output_dir"] = str(job_dir / "run")
        tmp = meta.with_suffix(meta.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, meta)

    def set_on_change(self, callback: Callable[[], None] | None) -> None:
        """Register a single observer fired when registry state changes."""
        self._on_change = callback

    def set_on_progress(self, callback: Callable[[builtins.list[dict]], None] | None) -> None:
        """Register an observer fired each watchdog tick with one dict per
        running job. Quiet when no job runs: a tick with no running jobs
        produces no callback."""
        self._on_progress = callback

    def _notify_change(self) -> None:
        cb = self._on_change
        if cb is None:
            return
        try:
            cb()
        except Exception as exc:
            logger.exception("JobRegistry on_change callback failed: %s", exc)

    def _notify_progress(self, snapshots: builtins.list[dict]) -> None:
        cb = self._on_progress
        if cb is None or not snapshots:
            return
        try:
            cb(snapshots)
        except Exception as exc:
            logger.exception("JobRegistry on_progress callback failed: %s", exc)

    # -- public API --

    def list(self, limit: int = 10) -> builtins.list[JobRecord]:
        with self._lock:
            records = list(self._records.values())
        records.sort(key=lambda r: r.started_at, reverse=True)
        records = records[:limit]
        for r in records:
            r.checkpoint_count = self._count_checkpoints(r)
        return records

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        record.checkpoint_count = self._count_checkpoints(record)
        return record

    def start(self, config: TrainingRequest, target: JobTarget | None = None) -> JobRecord:
        from .runners.hf_cloud import HfCloudJobRunner  # lazy import to avoid circular import
        from .runners.ssh_remote import SSHJobRunner
        from .ssh_config import get_ssh_config

        target = target or JobTarget()
        if target.runner == "hf_cloud" and not target.flavor:
            raise ValueError("flavor is required when runner is hf_cloud")
        if target.runner == "ssh" and not target.ssh_config_id:
            raise ValueError("ssh_config_id is required when runner is ssh")

        ssh_config = None
        if target.runner == "ssh":
            ssh_config = get_ssh_config(target.ssh_config_id)
            if ssh_config is None:
                raise ValueError(f"SSH config {target.ssh_config_id!r} not found")

        # Resolve resume_from_job_id: find the latest checkpoint of the
        # previous SSH job and set config_path + resume_output_dir so the
        # remote lerobot can pick up where it left off.
        resume_output_dir: str | None = None
        if config.resume_from_job_id and target.runner in ("ssh", "local"):
            prev = self._records.get(config.resume_from_job_id)
            if prev is None:
                raise JobNotFoundError(config.resume_from_job_id)
            ckpts = self.list_checkpoints(config.resume_from_job_id)
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints found for job {config.resume_from_job_id!r}"
                )
            latest = max(ckpts, key=lambda c: c.step)
            # latest.ref points to the pretrained_model dir; train_config.json
            # lives inside it.
            config.config_path = f"{latest.ref}/train_config.json"
            config.resume = True
            # Derive output_dir from the checkpoint path. For old jobs without
            # remote_output_dir, the ref is something like
            # /root/outputs/train/<date>/<time>_<policy>/checkpoints/<step>/pretrained_model
            # so we strip everything from /checkpoints/ onward.
            if prev.remote_output_dir:
                resume_output_dir = prev.remote_output_dir
            else:
                ref = latest.ref
                ckpt_idx = ref.find("/checkpoints/")
                if ckpt_idx >= 0:
                    resume_output_dir = ref[:ckpt_idx]
                else:
                    resume_output_dir = str(Path(ref).parent.parent)
            logger.info(
                "Resuming job %s from step %d, config_path=%s, output_dir=%s",
                config.resume_from_job_id,
                latest.step,
                config.config_path,
                resume_output_dir,
            )

        with self._lock:
            # Local trainings are bounded by this machine's GPU/USB resources,
            # so at most one runs at a time. SSH trainings share the pod's one
            # GPU — two concurrent jobs there OOM each other, so guard per
            # target too. Cloud trainings each get their own container.
            if target.runner in ("local", "ssh"):
                for r in self._records.values():
                    if r.state != "running":
                        continue
                    if target.runner == "local" and r.runner == "local":
                        raise JobAlreadyRunningError(r.id)
                    if (
                        target.runner == "ssh"
                        and r.runner == "ssh"
                        and r.ssh_config_id == target.ssh_config_id
                    ):
                        raise JobAlreadyRunningError(r.id)

            job_id = _generate_job_id(config.policy_type, config.dataset_repo_id)
            # Same policy+dataset started twice in one second would otherwise
            # overwrite the first record and share its dirs.
            base_id = job_id
            dup_n = 2
            while job_id in self._records:
                job_id = f"{base_id}_{dup_n}"
                dup_n += 1
            job_dir = _job_dir(self._output_root, job_id)
            # A local resume trains back into the previous job's run dir so
            # lerobot finds its checkpoints; logs/job.json stay per-job.
            if target.runner == "local" and resume_output_dir:
                lerobot_output_dir = resume_output_dir
            else:
                lerobot_output_dir = str(job_dir / "run")
            name = (config.job_name or "").strip() or f"{config.policy_type.upper()} · {config.dataset_repo_id}"
            record = JobRecord(
                id=job_id,
                name=name,
                state="running",
                config=config,
                output_dir=lerobot_output_dir,
                started_at=time.time(),
                runner=target.runner,
                hf_flavor=target.flavor,
                ssh_config_id=target.ssh_config_id,
            )

            job_dir.mkdir(parents=True, exist_ok=True)
            self._records[job_id] = record
            self._persist(record, force=True)

            log_path = _job_log_path(self._output_root, job_id)
            if target.runner == "local":
                runner = LocalJobRunner(record.metrics, log_file_path=log_path)
            elif target.runner == "ssh":
                runner = SSHJobRunner(record.metrics, log_path, ssh_config)
            else:
                runner = HfCloudJobRunner(record.metrics, log_path, target.flavor)

            try:
                if target.runner == "ssh":
                    runner.start(job_id, config, lerobot_output_dir, resume_output_dir=resume_output_dir)
                else:
                    runner.start(job_id, config, lerobot_output_dir)
            except Exception as exc:
                logger.exception("Failed to start runner for job %s", job_id)
                record.state = "failed"
                record.ended_at = time.time()
                record.error_message = f"Failed to start runner: {exc}"
                self._persist(record, force=True)
                raise

            # Capture runner-specific identifiers. For cloud jobs the HF job id
            # / page URL / model repo are printed by lerobot's submit_to_hf and
            # only appear in stdout a few seconds after start, so they're None
            # here; the watchdog (_tick) parses and persists them once they land.
            if target.runner == "local":
                record.process_pid = runner.pid()
            elif target.runner == "ssh":
                record.process_pid = runner.pid()
                record.remote_output_dir = getattr(runner, "_remote_output_dir", None)

            self._persist(record, force=True)
            self._runners[job_id] = runner
        self._notify_change()
        return record

    def register_imported(self, source: str, name: str | None = None) -> JobRecord:
        """Register an externally-trained model as a pointer-only pseudo-job.

        `source` is either an existing local directory (its path is stored in
        output_dir) or, failing that, a Hugging Face repo id (stored in
        hf_repo_id). The source must expose at least one checkpoint under the
        auto-detect rules, else ValueError. Nothing is copied; delete only
        removes the pointer."""
        src = source.strip()
        if not src:
            raise ValueError("source is required")

        local_path = Path(src).expanduser()
        if local_path.is_dir():
            resolved = str(local_path.resolve())
            ckpts = _list_imported_local(resolved)
            output_dir, hf_repo_id = resolved, None
            label = local_path.name or resolved
        else:
            from .utils.hf_auth import shared_hf_api

            ckpts = _list_imported_hub(shared_hf_api(), src)
            output_dir, hf_repo_id = "", src
            label = src

        if not ckpts:
            raise ValueError(
                f"No usable model at {src!r}. For a local path, expected a "
                "pretrained_model (config.json) or a checkpoints/<step>/"
                "pretrained_model tree. For a Hugging Face repo, the repo may "
                "not exist, be private without auth, or lack a model config."
            )

        # Best-effort policy type for the display name; inference reads the
        # real config from the checkpoint, so a wrong guess here is harmless.
        policy_type = "model"
        with contextlib.suppress(Exception):
            policy_type = str(_read_checkpoint_config(ckpts[-1]).get("type") or "model")

        job_id = _generate_job_id(policy_type, "imported")
        record = JobRecord(
            id=job_id,
            name=name or f"Imported · {label}",
            state="done",
            config=TrainingRequest(dataset_repo_id="(imported)", policy_type=policy_type),
            output_dir=output_dir,
            started_at=time.time(),
            ended_at=time.time(),
            runner="imported",
            hf_repo_id=hf_repo_id,
        )
        # Count checkpoints for imported models so validation UI can render
        record.checkpoint_count = len(self._checkpoints_for(record))
        with self._lock:
            self._records[job_id] = record
            self._persist(record, force=True)
        self._notify_change()
        return record

    def stop(self, job_id: str) -> JobRecord:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
        if record.state != "running" or runner is None:
            raise JobNotRunningError(job_id)
        runner.stop()
        # The watchdog will finalise the record (state, ended_at, exit_code).
        # Wait briefly so the caller sees the new state in the response.
        for _ in range(20):
            time.sleep(0.1)
            with self._lock:
                if record.state != "running":
                    return record
        return record

    def drain_logs(self, job_id: str) -> builtins.list[LogLine]:
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
            runner = self._runners.get(job_id)
        if runner is None:
            return []
        return runner.stream_log_lines()

    def read_persisted_logs(self, job_id: str) -> builtins.list[LogLine]:
        """Read all log lines that have been written to disk for this job.

        Used by the frontend on Monitoring-page mount to seed the log panel
        with history (e.g. after navigating away and back, or after a lelab
        restart marked the job 'interrupted').
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
        path = _job_log_path(self._output_root, job_id)
        if not path.exists():
            return []
        out: list[LogLine] = []
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(LogLine.model_validate_json(raw))
                except Exception:
                    continue  # skip a malformed line rather than 500ing
        return out

    def read_metrics_history(self, job_id: str) -> builtins.list[MetricsHistoryPoint]:
        """Reconstruct the per-step loss/lr/grad-norm series from log.jsonl.

        Used by the frontend on Monitoring-page mount to seed the curves so
        they survive page reloads, navigation, and lelab restarts. Re-parses
        on every call; cache later if a slow file ever shows up.
        """
        with self._lock:
            if job_id not in self._records:
                raise JobNotFoundError(job_id)
        path = _job_log_path(self._output_root, job_id)
        if not path.exists():
            return []
        points: list[MetricsHistoryPoint] = []
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    log_line = LogLine.model_validate_json(raw)
                except Exception:
                    continue  # skip malformed line, same as read_persisted_logs
                msg = log_line.message
                # Only the log-freq lines carry per-step metric values.
                # Tqdm lines have a step but no loss/lr — skip them so we
                # don't emit a flat-line point per tqdm tick.
                if "step:" not in msg or "loss:" not in msg:
                    continue
                fresh = TrainingMetrics()
                parse_metrics_into(msg, fresh)
                if fresh.current_step <= 0:
                    continue
                point = MetricsHistoryPoint(
                    step=fresh.current_step,
                    loss=fresh.current_loss,
                    lr=fresh.current_lr,
                    grad_norm=fresh.grad_norm,
                )
                # Dedupe by step: overwrite on consecutive same-step lines.
                if points and points[-1].step == point.step:
                    points[-1] = point
                else:
                    points.append(point)
        points.sort(key=lambda p: p.step)
        return points

    def _checkpoints_for(self, record: JobRecord) -> builtins.list[JobCheckpoint]:
        if record.runner == "imported":
            if record.hf_repo_id:
                return self._list_cloud_cached(record.hf_repo_id)
            return _list_imported_local(record.output_dir)
        if record.runner == "local":
            return _list_local_checkpoints(record.output_dir)
        if record.runner == "ssh":
            return _list_ssh_checkpoints(record)
        # Cloud: _list_imported_hub prefers the checkpoints/<step>/ tree (pushed when
        # save_checkpoint_to_hub is on) and falls back to the final model at the repo
        # root, so a finished run is always reachable even with no per-step tree.
        return self._list_cloud_cached(record.hf_repo_id)

    def list_checkpoints(self, job_id: str) -> builtins.list[JobCheckpoint]:
        """Return checkpoints saved for this job, ascending by step.

        Local jobs scan <output_dir>/checkpoints/. Cloud jobs introspect the
        Hub repo (30s TTL cache). Imported jobs auto-detect single-model vs
        checkpoints-tree from their local path or Hub repo id."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return self._checkpoints_for(record)

    def _list_cloud_cached(self, repo_id: str | None) -> builtins.list[JobCheckpoint]:
        """30s-TTL cache over the hub checkpoint listing (`_list_imported_hub`:
        the checkpoints/<step>/ tree, else the root model). All hub listings —
        cloud-trained and imported alike — share this cache + rate-limit budget."""
        if not repo_id:
            return []
        now = time.time()
        cached = self._cloud_ckpt_cache.get(repo_id)
        if cached is not None and cached[0] > now:
            return cached[1]
        from .utils.hf_auth import shared_hf_api  # lazy: keeps unit-test imports cheap

        result = _list_imported_hub(shared_hf_api(), repo_id)
        self._cloud_ckpt_cache[repo_id] = (now + _CLOUD_CKPT_TTL_SECONDS, result)
        return result

    def _count_checkpoints(self, record: JobRecord) -> int:
        if record.runner == "ssh":
            return self._ssh_checkpoint_count(record)
        return len(self._checkpoints_for(record))

    def _ssh_checkpoint_count(self, record: JobRecord) -> int:
        """Last known SSH checkpoint count, refreshed in the background.

        Serving this inline would put a 15s paramiko stall per SSH record on the
        jobs request, so a stale-but-known count is the correct answer here.
        """
        now = time.time()
        cached = self._ssh_ckpt_cache.get(record.id)
        if cached is not None and cached[0] > now:
            return cached[1]
        self._refresh_ssh_checkpoint_count(record)
        return record.checkpoint_count

    def _refresh_ssh_checkpoint_count(self, record: JobRecord) -> None:
        with self._ssh_ckpt_lock:
            if record.id in self._ssh_ckpt_inflight:
                return
            self._ssh_ckpt_inflight.add(record.id)
            if self._ssh_ckpt_pool is None:
                self._ssh_ckpt_pool = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="ssh-ckpt"
                )
            self._ssh_ckpt_pool.submit(self._probe_ssh_checkpoint_count, record)

    def _probe_ssh_checkpoint_count(self, record: JobRecord) -> None:
        changed = False
        try:
            checkpoints = _list_ssh_checkpoints_status(record)
            expires_in = (
                _SSH_CKPT_TTL_SECONDS if checkpoints is not None else _SSH_CKPT_FAIL_TTL_SECONDS
            )
            count = record.checkpoint_count if checkpoints is None else len(checkpoints)
            self._ssh_ckpt_cache[record.id] = (time.time() + expires_in, count)
            if checkpoints is not None and count != record.checkpoint_count:
                record.checkpoint_count = count
                self._persist(record, force=True)
                changed = True
        except Exception:
            logger.debug("Background SSH checkpoint probe failed for %s", record.id, exc_info=True)
        finally:
            with self._ssh_ckpt_lock:
                self._ssh_ckpt_inflight.discard(record.id)
        if changed:
            self._notify_change()

    def get_policy_config_summary(self, job_id: str, step: int) -> dict[str, object]:
        """Read the checkpoint's pretrained_model/config.json and return only
        the UX-relevant slice: policy type, expected camera names + their
        height/width, and whether the policy needs a --task string."""
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        ckpts = self.list_checkpoints(job_id)
        match = next((c for c in ckpts if c.step == step), None)
        if match is None:
            raise FileNotFoundError(f"No checkpoint at step {step} for job {record.id}")
        cfg = _read_checkpoint_config(match)
        policy_type = cfg.get("type")
        image_features: dict[str, dict[str, int]] = {}
        for full_name, feat in (cfg.get("input_features") or {}).items():
            if feat.get("type") != "VISUAL":
                continue
            shape = feat.get("shape") or []
            if len(shape) != 3:
                continue
            _channels, height, width = shape
            # The policy keys are 'observation.images.<name>'; the rollout CLI
            # takes just the suffix.
            name = full_name.split(".")[-1]
            image_features[name] = {"height": int(height), "width": int(width)}
        repo_id = record.config.dataset_repo_id
        if record.runner != "ssh" and (not repo_id or repo_id == "(imported)") and match.source == "local":
            repo_id = _read_train_config_dataset_repo_id(Path(match.ref))
        return {
            "policy_type": policy_type,
            "image_features": image_features,
            "requires_task": policy_type in _LANGUAGE_CONDITIONED_POLICY_TYPES,
            "tasks": _dataset_task_strings(repo_id),
        }

    def get_model_info(self, job_id: str, step: int) -> dict[str, object]:
        """Static model/training metadata for the detail page.

        Reads the checkpoint's config.json (architecture) and, when present on
        disk, train_config.json (hyperparameters) plus the weights file size.
        Imported checkpoints are local dirs so all three are available; hub
        refs only expose config.json.
        """
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        ckpts = self.list_checkpoints(job_id)
        match = next((c for c in ckpts if c.step == step), None)
        if match is None:
            raise FileNotFoundError(f"No checkpoint at step {step} for job {record.id}")
        cfg = _read_checkpoint_config(match)

        # Imported checkpoints are registered at step 0; the real training step
        # lives in the directory suffix (e.g. ..._s17500/model). Prefer it.
        display_step = step
        m = re.search(r"_s(\d+)(?:/|$)", match.ref)
        if m:
            display_step = int(m.group(1))

        info: dict[str, object] = {
            "step": display_step,
            "policy_type": cfg.get("type"),
            "chunk_size": cfg.get("chunk_size"),
            "n_action_steps": cfg.get("n_action_steps"),
            "max_action_dim": cfg.get("max_action_dim"),
            "num_vlm_layers": cfg.get("num_vlm_layers"),
            "num_expert_layers": cfg.get("num_expert_layers"),
            "vlm_model_name": cfg.get("vlm_model_name"),
            "normalization_mapping": cfg.get("normalization_mapping"),
            "dataset_repo_id": record.config.dataset_repo_id,
        }

        train_cfg: dict[str, object] | None = None
        size_bytes: int | None = None
        saved_at: float | None = None
        if match.source == "local":
            model_dir = Path(match.ref)
            tc_path = model_dir / "train_config.json"
            if tc_path.is_file():
                try:
                    train_cfg = json.loads(tc_path.read_text())
                except Exception as exc:  # malformed config shouldn't break the page
                    logger.warning("Bad train_config.json at %s: %s", tc_path, exc)
            weights = model_dir / "model.safetensors"
            if weights.is_file():
                size_bytes = weights.stat().st_size
                saved_at = weights.stat().st_mtime
            elif model_dir.is_dir():
                saved_at = model_dir.stat().st_mtime

        if train_cfg:
            opt = train_cfg.get("optimizer") or {}
            sched = train_cfg.get("scheduler") or {}
            info["training"] = {
                "batch_size": train_cfg.get("batch_size"),
                "steps": train_cfg.get("steps"),
                "seed": train_cfg.get("seed"),
                "num_workers": train_cfg.get("num_workers"),
                "optimizer_type": opt.get("type"),
                "lr": opt.get("lr"),
                "weight_decay": opt.get("weight_decay"),
                "grad_clip_norm": opt.get("grad_clip_norm"),
                "scheduler_type": sched.get("type"),
                "warmup_steps": sched.get("num_warmup_steps"),
                "decay_steps": sched.get("num_decay_steps"),
                "peak_lr": sched.get("peak_lr"),
                "decay_lr": sched.get("decay_lr"),
            }
            ds = train_cfg.get("dataset") or {}
            if ds.get("repo_id"):
                info["dataset_repo_id"] = ds["repo_id"]
        info["size_bytes"] = size_bytes
        info["saved_at"] = saved_at
        return info

    def export_checkpoint(self, job_id: str, step: int) -> JobRecord:
        """Download a checkpoint from the SSH remote to local via SFTP.

        Creates an 'imported' job record pointing to the downloaded checkpoint
        so it can be used for inference. Returns the new imported record.

        Blocks until the transfer completes; byte-level progress is published
        to ``self._export_progress`` for the UI to poll meanwhile.
        """
        with self._lock:
            record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        if record.runner != "ssh":
            raise ValueError(f"Job {job_id!r} is not an SSH job")
        if not record.ssh_config_id:
            raise ValueError(f"Job {job_id!r} has no SSH config")

        key = self._export_key(job_id, step)
        self._set_export_progress(
            key, state="preparing", transferred_bytes=0, total_bytes=0, current_file="", error=None
        )

        from .ssh_config import get_ssh_config

        cfg = get_ssh_config(record.ssh_config_id)
        if cfg is None:
            raise ValueError(f"SSH config {record.ssh_config_id!r} not found")

        # Find the checkpoint (works for both old and new jobs)
        ckpts = self.list_checkpoints(job_id)
        match = next((c for c in ckpts if c.step == step), None)
        if match is None:
            raise FileNotFoundError(f"No checkpoint at step {step} for job {job_id}")

        try:
            import paramiko
        except ImportError:
            self._set_export_progress(key, state="failed", error="paramiko is not installed")
            raise RuntimeError("paramiko is not installed. Run: pip install paramiko")

        remote_ckpt_dir = match.ref  # e.g. /path/to/checkpoints/000100/pretrained_model
        # Local destination: <output_root>/imported/<job_id>_step<step>/
        import_id = _generate_job_id(record.config.policy_type, record.config.dataset_repo_id)
        import_id = f"{import_id}_s{step}"
        local_dir = _job_dir(self._output_root, import_id) / "model"
        local_dir.mkdir(parents=True, exist_ok=True)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            connect_kwargs: dict = {
                "hostname": cfg.host,
                "port": cfg.port,
                "username": cfg.username,
                "timeout": 30,
            }
            if cfg.key_path:
                connect_kwargs["key_filename"] = os.path.expanduser(cfg.key_path)
            elif cfg.password:
                connect_kwargs["password"] = cfg.password
            client.connect(**connect_kwargs)
            sftp = client.open_sftp()
            try:
                self._sftp_download_dir(sftp, remote_ckpt_dir, str(local_dir), key)
            finally:
                sftp.close()
        except Exception as exc:
            self._set_export_progress(key, state="failed", error=str(exc))
            raise
        finally:
            client.close()

        self._set_export_progress(key, state="done", current_file="")
        # Register as imported job
        imported = self.register_imported(str(local_dir), name=f"{record.name} (step {step})")
        return imported

    def _sftp_download_dir(self, sftp, remote_dir: str, local_dir: str, key: str) -> None:
        """Recursively download a directory via SFTP, publishing byte progress."""
        files = self._sftp_collect_files(sftp, remote_dir)
        self._set_export_progress(
            key, state="downloading", total_bytes=sum(size for _, _, size in files)
        )
        # Throttle the updates: paramiko calls back once per ~32KB packet, which
        # would otherwise take the registry lock thousands of times per second.
        last_tick = 0.0
        done = 0
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for remote_path, rel, size in files:
            local_path = Path(local_dir) / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            def on_bytes(sent: int, file_total: int, rel: str = rel, base: int = done) -> None:
                nonlocal last_tick
                now = time.monotonic()
                if sent < file_total and now - last_tick < 0.2:
                    return
                last_tick = now
                self._set_export_progress(key, transferred_bytes=base + sent, current_file=rel)

            sftp.get(remote_path, str(local_path), callback=on_bytes)
            done += size
            self._set_export_progress(key, transferred_bytes=done, current_file=rel)

    @staticmethod
    def _sftp_collect_files(sftp, remote_dir: str, prefix: str = "") -> list[tuple[str, str, int]]:
        """Return [(remote_path, relative_local_path, size_bytes)] for remote_dir."""
        import stat as stat_mod

        out: list[tuple[str, str, int]] = []
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{entry.filename}"
            rel = f"{prefix}/{entry.filename}" if prefix else entry.filename
            if stat_mod.S_ISDIR(entry.st_mode or 0):
                out.extend(JobRegistry._sftp_collect_files(sftp, remote_path, rel))
            else:
                out.append((remote_path, rel, entry.st_size or 0))
        return out

    @staticmethod
    def _export_key(job_id: str, step: int) -> str:
        return f"{job_id}:{step}"

    def _set_export_progress(self, key: str, **fields) -> None:
        with self._lock:
            progress = self._export_progress.setdefault(
                key,
                {
                    "state": "none",
                    "transferred_bytes": 0,
                    "total_bytes": 0,
                    "current_file": "",
                    "error": None,
                },
            )
            progress.update(fields)

    def get_export_progress(self, job_id: str, step: int) -> dict:
        with self._lock:
            progress = self._export_progress.get(self._export_key(job_id, step))
            return dict(progress) if progress else {"state": "none"}

    def delete(self, job_id: str) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.state == "running":
                raise JobNotRunningError(job_id)
            self._records.pop(job_id, None)
            self._runners.pop(job_id, None)
            self._last_persist_at.pop(job_id, None)
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(_job_dir(self._output_root, job_id))
        self._notify_change()

    def start_validation(self, job_id: str, options: dict | None = None) -> dict:
        """Trigger the offline evaluator for a job's checkpoint.

        Options (all optional): ``step`` — which checkpoint, ``num_batches`` /
        ``batch_size`` / ``seed`` — sampling, ``holdout_episodes`` — exclude the
        last K episodes, ``episodes`` — explicit list, ``probe`` — run the
        vision/language sensitivity probe, ``device``.

        Runs in a background thread. Only works where the checkpoint and the
        dataset are reachable: locally, or on the SSH remote.
        """
        opts = options or {}
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            if record.runner not in ("local", "imported", "ssh"):
                raise ValueError(f"Validation is only supported for local/imported/ssh jobs, not {record.runner!r}")
            if record.validation_state == "running":
                return {"validation_state": "running", "message": "Validation already in progress"}

            ckpts = self._list_checkpoints_unlocked(job_id)
            if not ckpts:
                raise ValueError(f"No checkpoints found for job {job_id}")

            step = opts.get("step")
            if step is None:
                latest = ckpts[-1]
            else:
                latest = next((c for c in ckpts if c.step == int(step)), None)
                if latest is None:
                    raise ValueError(f"No checkpoint at step {step} for job {job_id}")

            # Charts land under the job dir so the server can serve them by
            # relative path without exposing the whole filesystem.
            chart_dir = _job_dir(self._output_root, job_id) / "evaluation" / datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            record.validation_state = "running"
            record.validation_results = None
            self._persist(record, force=True)

        self._notify_change()
        dataset_repo_id = record.config.dataset_repo_id
        runner_type = record.runner
        ssh_config_id = record.ssh_config_id
        if runner_type != "ssh" and (not dataset_repo_id or dataset_repo_id == "(imported)"):
            # For imported models, read the dataset_repo_id from train_config.json
            try:
                ckpt_dir = Path(latest.ref)
                train_config_path = ckpt_dir / "train_config.json"
                if train_config_path.exists():
                    with open(train_config_path) as f:
                        train_cfg = json.load(f)
                    dataset_repo_id = train_cfg.get("dataset", {}).get("repo_id", "")
                else:
                    logger.warning("train_config.json not found in checkpoint")
                    dataset_repo_id = ""
            except Exception as e:
                logger.warning("Failed to read dataset_repo_id from train_config.json: %s", e)
                dataset_repo_id = ""

        eval_kwargs: dict = {
            "num_batches": int(opts.get("num_batches", 20)),
            "batch_size": int(opts.get("batch_size", 8)),
            "seed": int(opts.get("seed", 0)),
            "holdout_episodes": int(opts.get("holdout_episodes", 20)),
            "frames_per_episode": int(opts.get("frames_per_episode", 0)),
            "episodes": opts.get("episodes") or None,
            "probe": bool(opts.get("probe", True)),
            "saliency_frames": int(opts.get("saliency_frames", 0)),
            "aim_frames": int(opts.get("aim_frames", 0)),
            "device": str(opts.get("device", "cuda")),
        }

        if opts.get("dataset_repo_id"):
            # Evaluate against a separately recorded validation set instead of a
            # slice of the training data.
            dataset_repo_id = str(opts["dataset_repo_id"])

        def _finalize(results: dict) -> None:
            # Charts come back as bare filenames inside chart_dir; store them
            # relative to the job directory so the serving route stays inside it.
            charts = results.pop("charts", None)
            chart_root = results.pop("chart_dir", None) or str(chart_dir)
            if charts:
                job_root = _job_dir(self._output_root, job_id).resolve()
                rel_base = Path(chart_root).resolve().relative_to(job_root)
                results["charts"] = {key: str(rel_base / name) for key, name in charts.items()}
            with self._lock:
                rec = self._records.get(job_id)
                if rec is not None:
                    rec.validation_state = "done" if results.get("success") else "failed"
                    rec.validation_results = results
                    self._persist(rec, force=True)
            self._notify_change()

        def _run():
            try:
                if runner_type == "ssh":
                    results = self._run_ssh_validation(
                        ssh_config_id=ssh_config_id,
                        remote_checkpoint=latest.ref,
                        dataset_repo_id=dataset_repo_id,
                        job_id=job_id,
                        **eval_kwargs,
                    )
                else:
                    from .validate import run_validation

                    results = run_validation(
                        checkpoint_path=latest.ref,
                        dataset_repo_id=dataset_repo_id,
                        output_dir=str(chart_dir),
                        **eval_kwargs,
                    )
                _finalize(results)
            except Exception as e:
                import traceback
                error_details = f"{str(e)}\n\n{traceback.format_exc()}"
                logger.error("Validation failed for job %s:\n%s", job_id, error_details)
                _finalize({"error": str(e), "traceback": traceback.format_exc()})

        threading.Thread(target=_run, daemon=True, name=f"validate-{job_id}").start()
        return {"validation_state": "running", "message": "Validation started"}

    def _run_ssh_validation(
        self,
        ssh_config_id: str | None,
        remote_checkpoint: str,
        dataset_repo_id: str,
        job_id: str,
        num_batches: int = 20,
        batch_size: int = 8,
        seed: int = 0,
        holdout_episodes: int = 0,
        episodes: list[int] | None = None,
        probe: bool = True,
        device: str = "cuda",
        saliency_frames: int = 0,
        aim_frames: int = 0,
    ) -> dict:
        """Run validate.py on the remote SSH machine and stream back the result.

        Uploads validate.py to /tmp, launches it under nohup with the remote
        python, then polls the remote log for the RESULT_JSON marker and the
        exit-code file. Survives SSH drops by reconnecting for the poll.
        """
        if not ssh_config_id:
            return {"success": False, "error": "Job has no SSH config"}
        if not dataset_repo_id:
            return {"success": False, "error": "No dataset repo id known for this job"}

        from .ssh_config import get_ssh_config

        cfg = get_ssh_config(ssh_config_id)
        if cfg is None:
            return {"success": False, "error": f"SSH config {ssh_config_id!r} not found"}

        try:
            import paramiko
        except ImportError:
            return {"success": False, "error": "paramiko is not installed"}

        # Upload validate.py so the remote runs the exact same code path.
        local_validate = Path(__file__).parent / "validate.py"
        remote_script = f"/tmp/lelab_validate_{job_id}.py"
        remote_log = f"/tmp/lelab_validate_{job_id}.log"
        remote_exit = f"/tmp/lelab_validate_{job_id}.exit"

        def _connect():
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            kwargs: dict = {
                "hostname": cfg.host,
                "port": cfg.port,
                "username": cfg.username,
                "timeout": 30,
            }
            if cfg.key_path:
                kwargs["key_filename"] = os.path.expanduser(cfg.key_path)
            elif cfg.password:
                kwargs["password"] = cfg.password
            client.connect(**kwargs)
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            return client

        client = _connect()
        try:
            sftp = client.open_sftp()
            try:
                sftp.put(str(local_validate), remote_script)
            finally:
                sftp.close()

            remote_python = cfg.remote_python or "python"
            env_prefix = "export HF_ENDPOINT=https://hf-mirror.com && " if cfg.use_cn_mirror else ""
            # Everything here is interpolated into a shell string, so keep the
            # numbers numeric and the device to a known set.
            safe_device = device if re.fullmatch(r"(cpu|cuda|mps|cuda:\d+)", device or "") else "cuda"
            cmd = (
                f"{env_prefix}{remote_python} {remote_script} "
                f"--checkpoint {remote_checkpoint} "
                f"--dataset {dataset_repo_id} "
                f"--num-batches {int(num_batches)} "
                f"--batch-size {int(batch_size)} "
                f"--seed {int(seed)} "
                f"--device {safe_device}"
            )
            if holdout_episodes:
                cmd += f" --holdout-episodes {int(holdout_episodes)}"
            if episodes:
                cmd += " --episodes " + ",".join(str(int(e)) for e in episodes)
            if not probe:
                cmd += " --no-probe"
            nohup_cmd = (
                f"nohup bash -c '{cmd}; echo $? > {remote_exit}' > {remote_log} 2>&1 & echo $!"
            )
            _stdin, stdout, _stderr = client.exec_command(nohup_cmd, timeout=30)
            pid_out = stdout.read().decode().strip()
            logger.info("SSH validation started (pid=%s) for job %s", pid_out, job_id)

            # Poll for completion: either the exit file appears or RESULT_JSON is logged.
            deadline = time.time() + 3600  # 1h hard cap
            result_json: dict | None = None
            while time.time() < deadline:
                time.sleep(5)
                try:
                    if client.get_transport() is None or not client.get_transport().is_active():
                        with contextlib.suppress(Exception):
                            client.close()
                        client = _connect()
                    _stdin, stdout, _stderr = client.exec_command(
                        f"grep -m1 '^RESULT_JSON ' {remote_log} 2>/dev/null; "
                        f"test -f {remote_exit} && echo EXIT=$(cat {remote_exit})",
                        timeout=15,
                    )
                    out = stdout.read().decode("utf-8", errors="replace")
                except Exception as exc:
                    logger.warning("SSH validation poll error: %s", exc)
                    continue

                exit_code: int | None = None
                for line in out.splitlines():
                    if line.startswith("RESULT_JSON "):
                        try:
                            result_json = json.loads(line[len("RESULT_JSON "):])
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith("EXIT="):
                        code = line[len("EXIT="):].strip()
                        exit_code = int(code) if code.isdigit() else None

                if result_json is not None:
                    return result_json
                if exit_code is not None:
                    # Process finished without a parseable RESULT_JSON marker.
                    tail = self._ssh_read_tail(client, remote_log, 40)
                    return {
                        "success": False,
                        "error": f"Remote validation exited with code {exit_code}",
                        "log_tail": tail,
                    }

            return {"success": False, "error": "Remote validation timed out after 1h"}
        finally:
            with contextlib.suppress(Exception):
                client.close()

    @staticmethod
    def _ssh_read_tail(client, remote_log: str, n: int = 40) -> str:
        try:
            _stdin, stdout, _stderr = client.exec_command(
                f"tail -n {n} {remote_log} 2>/dev/null", timeout=15
            )
            return stdout.read().decode("utf-8", errors="replace")
        except Exception:
            return ""

    def get_evaluation_artifact(self, job_id: str, rel_path: str) -> Path:
        """Resolve a chart PNG produced by the evaluator, inside the job dir only."""
        root = _job_dir(self._output_root, job_id).resolve()
        candidate = (root / rel_path).resolve()
        if not str(rel_path).startswith("evaluation/") or candidate.suffix != ".png":
            raise ValueError("Only evaluation chart files can be read")
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError(rel_path)
        return candidate

    def get_validation(self, job_id: str) -> dict:
        """Return validation state and results for a job."""
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                raise JobNotFoundError(job_id)
            return {
                "validation_state": record.validation_state,
                "validation_results": record.validation_results,
            }

    def _list_checkpoints_unlocked(self, job_id: str) -> list[JobCheckpoint]:
        """Internal: list checkpoints without acquiring the lock."""
        record = self._records.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)

        if record.runner == "imported":
            source = record.output_dir or record.hf_repo_id
            if not source:
                return []
            if record.output_dir and Path(record.output_dir).is_dir():
                return _list_imported_local(Path(record.output_dir))
            return []

        output_dir = Path(record.output_dir) if record.output_dir else None
        if not output_dir or not output_dir.is_dir():
            return []

        ckpts_dir = output_dir / "checkpoints"
        if not ckpts_dir.is_dir():
            return []

        results = []
        for entry in sorted(ckpts_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                step = int(entry.name)
            except ValueError:
                continue
            pretrained = entry / "pretrained_model"
            if not (pretrained / "config.json").is_file():
                continue
            results.append(JobCheckpoint(step=step, source="local", ref=str(pretrained)))
        return results

    def shutdown(self) -> None:
        """For tests / orderly process exit. Not wired to FastAPI lifespan today."""
        self._stop_watchdog.set()

    # -- internals --

    def _load_from_disk(self) -> None:
        for job_dir in self._output_root.glob("*/"):
            meta = job_dir / "job.json"
            if not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text())
                record = JobRecord.model_validate(data)
            except Exception as exc:
                logger.warning("Skipping malformed job.json at %s: %s", meta, exc)
                continue
            # Validation runs in an in-process thread that does not survive a
            # server restart. Any record persisted mid-validation is therefore a
            # zombie — without this reset it shows "validating" forever (and made
            # every imported model look like it was validating at once).
            if record.validation_state in ("running", "pending"):
                logger.info(
                    "Resetting stale validation_state=%s for job %s",
                    record.validation_state,
                    record.id,
                )
                record.validation_state = "failed"
                if record.validation_results is None:
                    record.validation_results = {
                        "error": "Validation was interrupted by a server restart"
                    }
                self._write_meta(record)
            if (
                record.runner == "ssh"
                and record.state == "interrupted"
                and _looks_unconfirmed(record.error_message)
            ):
                # Stranded by an unreachable-pod failure (possibly before this
                # flag existed): let the watchdog re-probe it.
                record.remote_unconfirmed = True
            if record.state == "running":
                if record.runner == "local":
                    pid = record.process_pid
                    if pid is not None and _pid_alive(pid):
                        logger.info(
                            "Re-attaching to detached local job %s (pid %d)",
                            record.id,
                            pid,
                        )
                        runner = TailingJobRunner(
                            record.metrics,
                            _job_log_path(self._output_root, record.id),
                            pid,
                        )
                        runner.start_tailing()
                        self._runners[record.id] = runner
                    else:
                        record.state = "interrupted"
                        if record.ended_at is None:
                            record.ended_at = time.time()
                        self._write_meta(record)
                elif record.runner == "hf_cloud" and record.hf_job_id and record.hf_flavor:
                    # Always reattach; the status poller is the source of truth
                    # for terminal state. If the HF job already finished, the
                    # next inspect_job call resolves the final stage and the
                    # watchdog finalises the record. A transient HF API hiccup
                    # at startup no longer strands the record as "interrupted".
                    logger.info(
                        "Re-attaching to HF Cloud job %s (hf_job_id=%s)",
                        record.id,
                        record.hf_job_id,
                    )
                    from .runners.hf_cloud import HfCloudJobRunner

                    runner = HfCloudJobRunner(
                        record.metrics,
                        _job_log_path(self._output_root, record.id),
                        record.hf_flavor,
                    )
                    runner.reattach(record.hf_job_id)
                    self._runners[record.id] = runner
                elif record.runner == "ssh" and record.ssh_config_id:
                    # Reattach to SSH training by checking if the remote process
                    # is still alive. If it is, resume tail -f streaming.
                    # If not, read the exit code file or mark done/interrupted.
                    if not self._attach_ssh_reattach(record):
                        record.state = "interrupted"
                        if record.ended_at is None:
                            record.ended_at = time.time()
                        self._write_meta(record)
                else:
                    # Malformed running record — mark interrupted.
                    record.state = "interrupted"
                    if record.ended_at is None:
                        record.ended_at = time.time()
                    self._write_meta(record)
            self._records[record.id] = record

    def _attach_ssh_reattach(self, record: JobRecord) -> bool:
        """Start a reattach runner for an SSH job. False if its SSH config is gone."""
        from .ssh_config import get_ssh_config
        from .runners.ssh_remote import SSHJobRunner

        ssh_cfg = get_ssh_config(record.ssh_config_id)
        if ssh_cfg is None:
            return False
        runner = SSHJobRunner(
            record.metrics,
            _job_log_path(self._output_root, record.id),
            ssh_cfg,
        )
        runner.reattach(
            job_id=record.id,
            remote_pid=record.process_pid,
            remote_log_path=f"/tmp/lelab_train_{record.id}.log",
            remote_exit_code_path=f"/tmp/lelab_exit_{record.id}.code",
            remote_output_dir=record.remote_output_dir,
        )
        self._runners[record.id] = runner
        return True

    def _reconcile_unconfirmed(self) -> None:
        """Retry SSH jobs we could only mark 'interrupted' because the remote was
        unreachable. Their training may be running fine on the pod — a transient
        SSH failure at local boot must not freeze the record for good.
        """
        with self._lock:
            due = [
                r
                for r in self._records.values()
                if r.runner == "ssh"
                and r.remote_unconfirmed
                and r.state != "running"
                and r.id not in self._runners
                and r.ssh_config_id
            ]
            for record in due:
                if record.reconcile_attempts >= _RECONCILE_MAX_ATTEMPTS:
                    record.remote_unconfirmed = False
                    record.error_message = (
                        f"连续 {record.reconcile_attempts} 次重连 pod 都失败，已停止自动对账。"
                        "训练状态请查 wandb，或重连后手动重新打开该任务。"
                    )
                    self._write_meta(record)
                    continue
                record.reconcile_attempts += 1
                record.state = "running"
                record.ended_at = None
                record.error_message = None
                record.remote_unconfirmed = False
                self._write_meta(record)
                logger.info(
                    "Reconciling SSH job %s (attempt %d) after an unreachable-remote failure",
                    record.id,
                    record.reconcile_attempts,
                )
                self._attach_ssh_reattach(record)

    def _start_watchdog(self) -> None:
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="job-registry-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        last_reconcile = 0.0
        while not self._stop_watchdog.is_set():
            try:
                self._tick()
                now = time.time()
                if now - last_reconcile >= _RECONCILE_INTERVAL_SECONDS:
                    last_reconcile = now
                    self._reconcile_unconfirmed()
            except Exception as exc:
                logger.exception("Watchdog tick failed: %s", exc)
            self._stop_watchdog.wait(1.0)

    def _tick(self) -> None:
        with self._lock:
            running_ids = [jid for jid, r in self._records.items() if r.state == "running"]

        progress_snapshots: builtins.list[dict] = []

        for jid in running_ids:
            with self._lock:
                runner = self._runners.get(jid)
                record = self._records.get(jid)
            if runner is None or record is None:
                continue
            if runner.is_running():
                # Pull the wandb run URL once it appears in stdout.
                if record.wandb_run_url is None:
                    url = runner.wandb_run_url()
                    if url is not None:
                        with self._lock:
                            record.wandb_run_url = url
                        self._persist(record, force=True)
                # Capture the remote PID once it becomes available (SSH runner
                # sets it asynchronously after the nohup command starts).
                if record.process_pid is None:
                    pid = runner.pid()
                    if pid is not None:
                        with self._lock:
                            record.process_pid = pid
                        self._persist(record, force=True)
                # Cloud jobs print their HF job id / page URL / model repo a few
                # seconds after start; capture them onto the record once parsed.
                if record.runner == "hf_cloud":
                    self._sync_cloud_ids(record, runner)
                # Persist metric snapshot at most once per second.
                self._persist(record, force=False)
                progress_snapshots.append(
                    {
                        "id": record.id,
                        "state": record.state,
                        "metrics": record.metrics.model_dump(),
                        "wandb_run_url": record.wandb_run_url,
                        "checkpoint_count": self._count_checkpoints(record),
                    }
                )
                continue

            # Subprocess exited since the last tick. Finalise.
            # Capture any cloud ids printed right before exit (e.g. a job that
            # submitted then failed fast) so checkpoint listing has the repo id.
            if record.runner == "hf_cloud":
                self._sync_cloud_ids(record, runner)
            rc = runner.returncode()
            # A remote runner can finish without ever establishing what the
            # remote did (recycled pod, changed host:port, lost connection).
            # That is not a training failure — say so instead of claiming done.
            lost_reason = getattr(runner, "remote_lost_reason", None)
            lost_reason = lost_reason() if callable(lost_reason) else None
            unconfirmed = getattr(runner, "remote_unconfirmed", None)
            unconfirmed = bool(unconfirmed()) if callable(unconfirmed) else False
            with self._lock:
                if record.wandb_run_url is None:
                    record.wandb_run_url = runner.wandb_run_url()
                if lost_reason:
                    record.state = "interrupted"
                    record.error_message = lost_reason
                    # "Couldn't reach the pod" is not evidence the training
                    # stopped; the flag lets the watchdog re-attach it later.
                    record.remote_unconfirmed = unconfirmed
                else:
                    record.state = "done" if rc == 0 else "failed"
                    record.remote_unconfirmed = False
                    record.reconcile_attempts = 0
                    if rc != 0 and record.error_message is None:
                        # 143 = the SSH runner's confirmed remote kill; -15 = the
                        # local subprocess SIGTERM. Neither is a training failure.
                        if rc == 143:
                            record.error_message = "已由用户停止（远端进程已确认终止）"
                        elif rc == -signal.SIGTERM.value:
                            record.error_message = "已由用户停止"
                        else:
                            record.error_message = f"Job exited with code {rc}"
                record.ended_at = time.time()
                record.exit_code = rc
                # Persist checkpoint count for SSH jobs so it survives restarts
                # and remote server unavailability.
                if record.runner == "ssh":
                    record.checkpoint_count = self._count_checkpoints(record)
                self._runners.pop(jid, None)
            self._persist(record, force=True)
            self._notify_change()

            # Auto-validate when training finishes successfully with checkpoints
            # and a known dataset. Runs in a background thread via start_validation.
            if rc == 0 and record.config.dataset_repo_id:
                try:
                    self.start_validation(jid)
                except Exception:
                    logger.debug("Auto-validation skipped for job %s", jid, exc_info=True)

        self._notify_progress(progress_snapshots)

    def _sync_cloud_ids(self, record: JobRecord, runner: JobRunner) -> None:
        """Copy HF job id / page URL / model repo from a cloud runner onto the
        record once lerobot's submit_to_hf has printed them. Persists on first
        appearance so the ids survive a uvicorn --reload (which drives reattach).
        """
        changed = False
        for attr in ("hf_job_id", "hf_job_url", "hf_repo_id"):
            if getattr(record, attr) is not None:
                continue
            getter = getattr(runner, attr, None)
            value = getter() if callable(getter) else None
            if value is not None:
                with self._lock:
                    setattr(record, attr, value)
                changed = True
        if changed:
            self._persist(record, force=True)

    def _persist(self, record: JobRecord, force: bool) -> None:
        now = time.time()
        last = self._last_persist_at.get(record.id, 0.0)
        if not force and (now - last) < _PERSIST_THROTTLE_SECONDS:
            return
        self._last_persist_at[record.id] = now
        self._write_meta(record)

    def _write_meta(self, record: JobRecord) -> None:
        path = _job_meta_path(self._output_root, record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-write never strands a half-written file
        # that would skip the job on next _load_from_disk.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, path)
        logger.info("Wrote job.json for %s: state=%s exit_code=%s", record.id, record.state, record.exit_code)


# Module-level singleton. Anchored to ~/.cache so history survives launches
# from different cwds. JobRegistry.__init__ migrates legacy `<cwd>/outputs/train/`
# job dirs into this root on first boot. LELAB_OUTPUT_ROOT overrides for tests.
_DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("LELAB_OUTPUT_ROOT")
    or (Path.home() / ".cache" / "huggingface" / "lerobot" / "outputs" / "train")
).expanduser()
job_registry = JobRegistry(_DEFAULT_OUTPUT_ROOT)

__all__ = [
    "JobState",
    "JobTarget",
    "TrainingMetrics",
    "LogLine",
    "JobRecord",
    "JobCheckpoint",
    "MetricsHistoryPoint",
    "JobRunner",
    "SubprocessJobRunner",
    "LocalJobRunner",
    "JobRegistry",
    "JobAlreadyRunningError",
    "JobNotFoundError",
    "JobNotRunningError",
    "job_registry",
    "parse_metrics_into",
]
