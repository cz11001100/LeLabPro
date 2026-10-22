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

"""SSH remote runner — executes a training on a remote machine via SSH.

Uses paramiko for the SSH connection. The remote machine must have lerobot
installed and accessible. Training output is streamed back over the SSH channel
and fed through the same metric-parsing / log-persisting pipeline as the local
and cloud runners, so the monitoring UI works identically.

The training process runs under setsid+nohup in its own session so it survives
SSH session timeouts *and* provider-side session cleanup (which typically kills
the whole process group of the dead SSH session — nohup alone does not protect
against that). Output is written to a remote log file and streamed back via
tail -f.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from ..jobs import JobRunner, LogLine, TrainingMetrics, extract_wandb_run_url, parse_metrics_into
from ..ssh_config import SSHConfig
from ..ssh_remote_env import (
    CN_PIP_MIRROR,
    POLICY_INSTALL_EXTRAS,
    effective_python,
    policy_union_extras,
)
from ..train import (
    TrainingRequest,
    accelerate_mixed_precision,
    build_training_command,
    resolve_hub_revision,
)

logger = logging.getLogger(__name__)

# How often the consumed-line offset is flushed to the sidecar. The offset is
# what lets a restart replay only the remote log lines written while LeLab was
# down, so a bounded staleness window is enough.
_OFFSET_FLUSH_EVERY = 25

# Exit code reported when the remote job's outcome cannot be established (pod
# recycled, host/port changed, connection lost before an exit code was written).
_EXIT_UNKNOWN = -1


class SSHJobRunner:
    """Run a training on a remote machine via SSH.

    Connects using the provided SSHConfig, builds the lerobot-train command,
    executes it remotely detached in its own session (setsid + nohup), and
    streams output back through the standard _consume_lines pipeline. Because
    the training leads its own session/process group, it survives both SSH
    timeouts and provider-side cleanup that kills the dead session's process
    group. Output is written to a remote log file and streamed via tail -f.
    """

    def __init__(
        self,
        metrics: TrainingMetrics,
        log_file_path: Path,
        ssh_config: SSHConfig,
    ) -> None:
        self._metrics = metrics
        self._log_file_path = log_file_path
        self._log_file = None  # type: ignore[assignment]
        self._ssh_config = ssh_config
        self._log_queue: Queue[LogLine] = Queue()
        self._stop_event = threading.Event()
        self._ssh_thread: threading.Thread | None = None
        self._wandb_run_url: str | None = None
        self._remote_pid: int | None = None
        self._exit_code: int | None = None
        self._ssh_client = None
        self._channel = None
        self._job_id: str | None = None
        self._policy_type: str | None = None
        self._dataset_repo_id: str | None = None
        self._dataset_root: str | None = None
        self._remote_log_path: str | None = None
        self._remote_exit_code_path: str | None = None
        self._remote_pid_path: str | None = None
        self._lines_processed: int = 0
        self._offset_path = log_file_path.with_suffix(".tail_offset")
        # Set when the remote job's outcome is unknowable, so the registry can
        # mark the record 'interrupted' instead of reading a fake exit code.
        # Set when a credential is injected into the remote command, so the
        # command echo (log file + server stdout) never carries the secret.
        self._secret: str | None = None
        # True when the only thing we know is "could not reach the remote": the
        # job might be training fine, so the registry keeps re-probing it.
        self._remote_unconfirmed: bool = False
        self._remote_lost_reason: str | None = None

    def start(
        self,
        job_id: str,
        config: TrainingRequest,
        output_dir: str,
        resume_output_dir: str | None = None,
    ) -> None:
        from ..jobs import JobTarget

        # Compute a known remote output dir so we can later list/download
        # checkpoints via SFTP. Use remote_workdir as base if set.
        # When resuming, reuse the previous job's output dir so lerobot can
        # find the existing checkpoints.
        # shlex.join quotes a leading `~/`, so the remote shell stops expanding it:
        # training would write checkpoints into a literal `~` directory while the
        # SFTP listing/export paths resolve `~` to $HOME and find nothing. Use an
        # absolute base instead.
        home = self._remote_home()
        if resume_output_dir:
            self._remote_output_dir = resume_output_dir
        else:
            base = self._ssh_config.remote_workdir or home
            self._remote_output_dir = f"{base}/lelab_outputs/{job_id}"
        if self._remote_output_dir.startswith("~/"):
            self._remote_output_dir = home + self._remote_output_dir[1:]

        # A mirror can serve the pod a cached `main` tree and resurrect files
        # deleted from the Hub (stale meta shards → index out of bounds). Pin the
        # SHA resolved straight from huggingface.co so the download is exact.
        if not config.dataset_root and not config.dataset_revision:
            sha = resolve_hub_revision(config.dataset_repo_id)
            if sha:
                logger.info("Pinned dataset %s to revision %s", config.dataset_repo_id, sha[:8])
                config = config.model_copy(update={"dataset_revision": sha})

        # Pass job_target so build_training_command knows this is a remote run
        # and excludes --output_dir (which is a local path that doesn't exist on the remote).
        # Resolve the interpreter first: a blank remote_python would otherwise emit
        # an empty argv[0], and the pod's distro python3 is usually too old for lerobot.
        self._remote_python = effective_python(self._ssh_config)
        job_target = JobTarget(runner="ssh", ssh_config_id=self._ssh_config.id)
        cmd = build_training_command(
            config,
            output_dir,
            self._remote_python,
            job_target=job_target,
        )
        # Inject --output_dir so checkpoints land at the known remote path.
        cmd.extend(["--output_dir", self._remote_output_dir])
        logger.info("SSH training command: %s", shlex.join(cmd))
        # Not " ".join: this string is executed by the remote shell, so any value
        # with a space or a glob character (--dataset.episodes=[1,2], a task
        # description) would be split or expanded instead of passed through.
        cmd_str = shlex.join(cmd)

        # If wandb is enabled and API key is configured, authenticate via the
        # environment. `wandb login` was the old path and broke two ways: the
        # console script is not on PATH in a non-interactive SSH shell (exit 127
        # killed the whole `&&` chain before training could start), and it needs
        # a writable netrc. WANDB_API_KEY is read by wandb.init directly.
        if config.wandb_enable and self._ssh_config.wandb_api_key:
            api_key = self._ssh_config.wandb_api_key
            self._secret = api_key
            cmd_str = f"export WANDB_API_KEY={self._shell_quote(api_key)} && {cmd_str}"

        # Ensure FFmpeg is available (required by torchcodec for video decoding)
        cmd_str = (
            "which ffmpeg >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq ffmpeg) 2>/dev/null; "
            f"{cmd_str}"
        )

        # Xet transfers ignore HF_ENDPOINT and talk straight to
        # cas-server.xethub.hf.co, which a mirror-routed pod cannot authenticate
        # against (401 on reconstruction). Plain HTTP through the mirror works,
        # so keep the legacy transport. Cross-border links also stall longer
        # than huggingface_hub's 10s default read timeout, which turns every
        # video fetch into a resume-retry loop.
        cmd_str = f"export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 && {cmd_str}"

        # Pod images commonly export ACCELERATE_MIXED_PRECISION=fp16, and lerobot
        # leaves the mode to the environment when the policy config has no dtype
        # field. An fp16 scaler cannot unscale a bf16 policy's gradients, so pin
        # the mode to what the policy's weights can use.
        cmd_str = (
            f"export ACCELERATE_MIXED_PRECISION={accelerate_mixed_precision(config.policy_type)} && {cmd_str}"
        )

        if self._ssh_config.use_cn_mirror:
            cmd_str = f"export HF_ENDPOINT=https://hf-mirror.com && {cmd_str}"

        if self._ssh_config.remote_workdir:
            cmd_str = f"cd {self._ssh_config.remote_workdir} && {cmd_str}"

        logger.info(
            "Starting SSH job %s on %s: %s", job_id, self._ssh_config.host, self._redact(cmd_str)
        )
        self._job_id = job_id
        self._policy_type = config.policy_type
        self._dataset_repo_id = config.dataset_repo_id
        self._dataset_root = config.dataset_root
        self._remote_log_path = f"/tmp/lelab_train_{job_id}.log"
        self._remote_exit_code_path = f"/tmp/lelab_exit_{job_id}.code"
        self._open_log_file()
        # Write the command to the log file for debugging
        if self._log_file:
            import time
            cmd_log = LogLine(timestamp=time.time(), message=f"SSH command: {self._redact(cmd_str)}")
            self._log_file.write(cmd_log.model_dump_json() + "\n")
        self._ssh_thread = threading.Thread(
            target=self._ssh_run, args=(cmd_str,), name=f"ssh-job-{job_id}", daemon=True
        )
        self._ssh_thread.start()

    def stop(self) -> None:
        """Kill the detached remote training and only then report it stopped.

        The old version fired `kill` at whatever client object it had and moved
        on. After an SSH drop or a LeLab restart that object is closed, so the
        call raised into a suppress(), the record went terminal anyway, and the
        pod kept training (and billing). Now: reconnect if needed, TERM then
        KILL, and wait for the PID to actually disappear.
        """
        self._stop_event.set()

        if self._remote_pid is None:
            self._teardown_streams()
            if self._exit_code is None:
                self._exit_code = 143
            return

        client = self._live_client_for_stop()
        if client is None:
            self._teardown_streams()
            self._exit_code = _EXIT_UNKNOWN
            self._remote_unconfirmed = True
            self._remote_lost_reason = (
                f"已请求停止，但连不上 pod（{self._ssh_config.host}:{self._ssh_config.port}），"
                f"无法确认远端进程 {self._remote_pid} 是否终止——训练可能仍在运行并继续计费。"
                f"重连后手动执行：kill -TERM -{self._remote_pid}"
            )
            self._log_error(self._remote_lost_reason)
            return

        confirmed = False
        try:
            self._signal_remote_group(client, "TERM")
            confirmed = self._wait_for_remote_exit(client, timeout=8.0)
            if not confirmed:
                logger.warning(
                    "Training %s ignored SIGTERM for 8s, escalating to SIGKILL", self._remote_pid
                )
                self._signal_remote_group(client, "KILL")
                confirmed = self._wait_for_remote_exit(client, timeout=5.0)
        except Exception as exc:
            logger.warning("Remote stop failed for %s: %s", self._job_id, exc)

        self._teardown_streams(client)
        if confirmed:
            self._exit_code = 143
        else:
            self._exit_code = _EXIT_UNKNOWN
            self._remote_lost_reason = (
                f"已向 pod 发送停止信号，但进程 {self._remote_pid} 没有确认退出——"
                "请登录 pod 检查它是否仍在运行（可能还在计费）。"
            )
            self._log_error(self._remote_lost_reason)

    def _signal_remote_group(self, client, signal_name: str) -> None:
        """Signal the training's process group (setsid makes the wrapper its leader)."""
        cmd = (
            f"kill -{signal_name} -{self._remote_pid} 2>/dev/null || "
            f"kill -{signal_name} {self._remote_pid} 2>/dev/null; true"
        )
        _stdin, stdout, _stderr = client.exec_command(cmd, timeout=10)
        stdout.channel.recv_exit_status()

    def _wait_for_remote_exit(self, client, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _stdin, stdout, _stderr = client.exec_command(
                f"kill -0 {self._remote_pid} 2>/dev/null && echo alive || echo gone", timeout=10
            )
            if stdout.read().decode("utf-8", errors="replace").strip() != "alive":
                return True
            time.sleep(0.5)
        return False

    def _live_client_for_stop(self):
        """A usable client for the kill, reconnecting once if the pooled one died."""
        client = self._ssh_client
        if client is not None:
            with contextlib.suppress(Exception):
                transport = client.get_transport()
                if transport is not None and transport.is_active():
                    return client
        try:
            client = self._connect(timeout=10)
        except Exception as exc:
            logger.warning("Reconnect for stop failed (%s): %s", self._job_id, exc)
            return None
        self._ssh_client = client
        return client

    def _teardown_streams(self, client=None) -> None:
        if self._channel is not None:
            with contextlib.suppress(Exception):
                self._channel.send("\x03")
            self._channel = None
        for candidate in (client, self._ssh_client):
            if candidate is not None:
                with contextlib.suppress(Exception):
                    candidate.close()
        self._ssh_client = None

    def is_running(self) -> bool:
        if self._exit_code is not None:
            return False
        return self._ssh_thread is not None and self._ssh_thread.is_alive()

    def returncode(self) -> int | None:
        return self._exit_code

    def stream_log_lines(self) -> list[LogLine]:
        out: list[LogLine] = []
        try:
            while True:
                out.append(self._log_queue.get_nowait())
        except Empty:
            pass
        return out

    def wandb_run_url(self) -> str | None:
        return self._wandb_run_url

    def pid(self) -> int | None:
        return self._remote_pid

    def remote_lost_reason(self) -> str | None:
        """Why the remote job's outcome could not be established, or None if the
        remote reported a real exit code. Read by the registry when finalizing."""
        return self._remote_lost_reason

    def remote_unconfirmed(self) -> bool:
        """True when the failure was only "remote unreachable" (not a confirmed
        end of training), so the registry should keep reconciling later."""
        return self._remote_unconfirmed

    def reattach(
        self,
        job_id: str,
        remote_pid: int | None,
        remote_log_path: str,
        remote_exit_code_path: str,
        remote_output_dir: str | None = None,
    ) -> None:
        """Reattach to an existing remote training after a server restart.

        Connects via SSH, probes what the remote did while LeLab was down,
        replays any log lines we have not consumed yet, then either resumes
        tail -f streaming or finalizes.
        """
        self._job_id = job_id
        self._remote_pid = remote_pid
        self._remote_log_path = remote_log_path
        self._remote_exit_code_path = remote_exit_code_path
        self._remote_output_dir = remote_output_dir
        # Seed the resume offset from the last offset we persisted, so the
        # remote tail replays only what is missing instead of jumping to the
        # end and dropping everything written while the machine was off.
        self._lines_processed = self._load_offset()
        self._open_log_file()
        self._ssh_thread = threading.Thread(
            target=self._reattach_run, name=f"ssh-reattach-{job_id}", daemon=True
        )
        self._ssh_thread.start()

    # -- internals --

    def _count_local_log_lines(self) -> int:
        try:
            with self._log_file_path.open("r", errors="replace") as fh:
                return sum(1 for _ in fh)
        except FileNotFoundError:
            return 0
        except Exception:
            return 0

    def _load_offset(self) -> int:
        """Remote-log lines already consumed, from the sidecar offset file.

        The line count of log.jsonl is only a rough proxy (blank remote lines
        are counted but not stored, LeLab's own notes are stored but not
        counted), and an inflated proxy silently skips unseen remote lines.
        Jobs predating the sidecar still fall back to it.
        """
        try:
            return max(0, int(self._offset_path.read_text().strip()))
        except (FileNotFoundError, ValueError, OSError):
            return self._count_local_log_lines()

    def _save_offset(self) -> None:
        with contextlib.suppress(Exception):
            self._offset_path.parent.mkdir(parents=True, exist_ok=True)
            self._offset_path.write_text(str(self._lines_processed))

    def _probe_remote(self, client) -> tuple[str, str]:
        """Ask the pod, in one round-trip, where this job stands.

        Returns ('exit', code) when the wrapper wrote an exit code, ('alive', '')
        while the detached process runs, ('stale', '') when the log survives but
        neither process nor exit code does, and ('missing', '') when the pod has
        no trace of the job (recycled pod, or the SSH config now points elsewhere).
        """
        pid_check = f"kill -0 {self._remote_pid} 2>/dev/null" if self._remote_pid else "false"
        exit_q = self._shell_quote(self._remote_exit_code_path)
        log_q = self._shell_quote(self._remote_log_path)
        script = (
            f"if [ -f {exit_q} ]; then echo \"exit:$(cat {exit_q} 2>/dev/null)\"; "
            f"elif {pid_check}; then echo alive; "
            f"elif [ -f {log_q} ]; then echo stale; "
            "else echo missing; fi"
        )
        _stdin, stdout, _stderr = client.exec_command(script, timeout=15)
        out = stdout.read().decode("utf-8", errors="replace").strip()
        head, _, payload = out.partition(":")
        if head in ("exit", "alive", "stale", "missing"):
            return head, payload.strip()
        logger.warning("Unexpected reattach probe output for %s: %r", self._job_id, out)
        return "missing", ""

    def _tail_resume_cmd(self) -> str:
        """tail that replays from the first unseen remote-log line, then follows."""
        start = self._lines_processed + 1
        return f"tail -n +{start} -f {self._remote_log_path}"

    def _open_log_file(self) -> None:
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_file_path.open("a", buffering=1)

    def _reattach_run(self) -> None:
        """Reconnect after LeLab went down and establish what the remote did.

        One probe classifies the job, and every terminal branch first replays the
        log lines written while we were away — otherwise a run that finished
        during the outage would finalize with a curve truncated at shutdown.
        """
        try:
            import paramiko
        except ImportError:
            self._log_error("paramiko not installed")
            self._exit_code = 1
            return

        client = None
        try:
            client = self._connect_retry()
            self._ssh_client = client

            kind, payload = self._probe_remote(client)
            logger.info("Reattach probe for %s: %s %s", self._job_id, kind, payload)

            if kind == "exit":
                self._drain_tail_from_log(client)
                self._exit_code = int(payload) if payload.lstrip("-").isdigit() else _EXIT_UNKNOWN
                if self._exit_code == _EXIT_UNKNOWN:
                    self._remote_lost_reason = "远端退出码文件为空或损坏，无法确认训练是正常结束还是被中断。"
                return

            if kind == "stale":
                self._drain_tail_from_log(client)
                self._exit_code = _EXIT_UNKNOWN
                self._remote_lost_reason = (
                    "远端训练进程已不存在，且没留下退出码（可能被 kill，或 pod 在本机关机期间重启过）。"
                    "关机期间的日志已补回，看下面最后几行判断它跑到哪了。"
                )
                return

            if kind == "missing":
                self._exit_code = _EXIT_UNKNOWN
                self._remote_lost_reason = (
                    "远端已找不到这个任务（日志和退出码都不在）——pod 可能已被回收，或 SSH 配置"
                    "换到了新 pod。真实结束状态请查 wandb；若之前开了 push_to_hub，checkpoint 在 Hub 上。"
                )
                self._log_error(self._remote_lost_reason)
                return

            logger.info("Reattaching to running training PID %s", self._remote_pid)
            channel = client.get_transport().open_session()
            channel.settimeout(0.1)
            channel.exec_command(self._tail_resume_cmd())
            self._channel = channel
            self._consume_channel_with_pid_check(channel, client)

            if self._exit_code is None:
                active_client = self._ssh_client or client
                for _ in range(10):
                    self._exit_code = self._read_remote_exit_code(active_client)
                    if self._exit_code is not None:
                        break
                    time.sleep(1)
            if self._exit_code is None:
                self._exit_code = _EXIT_UNKNOWN
                self._remote_unconfirmed = True
                self._remote_lost_reason = (
                    "监控连接断开后仍读不到远端退出码，任务状态未知——训练可能还在跑，"
                    "也可能已经停了。请查 wandb，或重新连接 pod 确认进程是否存活。"
                )
                self._log_error(self._remote_lost_reason)
            logger.info("Reattached training finished with exit code %d", self._exit_code)

        except Exception as exc:
            logger.exception("SSH reattach failed: %s", exc)
            if self._exit_code is None:
                self._exit_code = _EXIT_UNKNOWN
                self._remote_unconfirmed = True
                self._remote_lost_reason = f"无法连接远端确认任务状态（{exc}）——pod 可能已停机或地址变了。"
                self._log_error(self._remote_lost_reason)
        finally:
            self._save_offset()
            with contextlib.suppress(Exception):
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
            with contextlib.suppress(Exception):
                if client is not None:
                    client.close()

    def _ssh_run(self, cmd_str: str) -> None:
        try:
            import paramiko
        except ImportError:
            self._log_error("paramiko is not installed. Run: pip install paramiko")
            self._exit_code = 1
            return

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh_client = client

        try:
            connect_kwargs: dict = {
                "hostname": self._ssh_config.host,
                "port": self._ssh_config.port,
                "username": self._ssh_config.username,
                "timeout": 30,
            }
            if self._ssh_config.key_path:
                connect_kwargs["key_filename"] = os.path.expanduser(self._ssh_config.key_path)
            elif self._ssh_config.password:
                connect_kwargs["password"] = self._ssh_config.password

            client.connect(**connect_kwargs)
            # Enable SSH keepalive to detect dead connections
            transport = client.get_transport()
            if transport is not None:
                transport.set_keepalive(30)

            # Pre-flight: verify the requested policy actually imports with
            # the remote's installed extras. Failing here costs seconds;
            # failing after dataset download costs GPU money.
            err = self._preflight_policy(client)
            if err is not None:
                self._log_error(err)
                self._exit_code = 1
                return

            # A dataset re-push can land in multiple commits; a download that
            # races that window caches a mix (old table + new data) and
            # training dies later with a cryptic IndexError. Probe the cached
            # snapshot's internal consistency and drop it when mixed.
            self._ensure_dataset_cache_consistent(client)

            # Checkpoints fill disks: 9/22 an 18GB-full pod killed a run with
            # "No space left on device" mid-save. Fail before burning GPU hours.
            disk_err = self._preflight_disk(client)
            if disk_err is not None:
                self._log_error(disk_err)
                self._exit_code = 1
                return

            # Detach the training into its own session: setsid puts it in a new
            # session/process group so provider-side cleanup that kills the dead
            # SSH session's process group cannot reach it; nohup + redirected
            # stdio covers plain SIGHUP. The wrapper writes its own PID (which
            # equals the new pgid) to a file so we can monitor and kill the
            # whole group later — `echo $!` would race with setsid's fork.
            detach = "setsid nohup" if self._has_setsid(client) else "nohup"
            self._remote_pid_path = f"/tmp/lelab_pid_{self._job_id}.pid"
            inner = (
                f"echo $$ > {self._remote_pid_path}; "
                f"{cmd_str}; "
                f"echo $? > {self._remote_exit_code_path}"
            )
            launch_cmd = (
                f"rm -f {self._remote_pid_path}; "
                f"{detach} bash -c {self._shell_quote(inner)} "
                f"> {self._remote_log_path} 2>&1 < /dev/null & "
                f"for _ in $(seq 1 20); do [ -f {self._remote_pid_path} ] && break; sleep 0.25; done; "
                f"cat {self._remote_pid_path} 2>/dev/null || echo FAILED"
            )
            logger.info("Starting remote training detached (%s): %s", detach, launch_cmd[:200])

            _stdin, stdout, stderr = client.exec_command(launch_cmd, timeout=30)
            pid_output = stdout.read().decode().strip()
            err_output = stderr.read().decode().strip()
            if err_output:
                logger.warning("detached launch stderr: %s", err_output)

            if not pid_output or not pid_output.isdigit():
                self._log_error(f"Failed to start remote training. PID output: {pid_output}")
                self._exit_code = 1
                return

            self._remote_pid = int(pid_output)
            logger.info("Remote training started with PID %d, log: %s", self._remote_pid, self._remote_log_path)

            # Now tail the log file to stream output back
            tail_cmd = f"tail -f {self._remote_log_path}"
            channel = client.get_transport().open_session()
            channel.settimeout(0.1)
            channel.exec_command(tail_cmd)
            self._channel = channel

            # Monitor the tail channel and check if the training process is still running
            self._consume_channel_with_pid_check(channel, client)

            # Read the exit code from the file written by the nohup wrapper.
            # Use self._ssh_client since reconnects may have replaced the original.
            # Brief poll since the file may take a moment to be written after process exit.
            # A user stop has already settled the outcome; don't re-read through the
            # client it just closed, or this overwrites the stop's exit code.
            if self._exit_code is None:
                active_client = self._ssh_client or client
                for _ in range(10):
                    self._exit_code = self._read_remote_exit_code(active_client)
                    if self._exit_code is not None:
                        break
                    time.sleep(1)
            if self._exit_code is None:
                # Never assume success: the tail loop can end because the SSH
                # connection died while the detached training keeps running.
                self._exit_code = _EXIT_UNKNOWN
                self._remote_unconfirmed = True
                self._remote_lost_reason = (
                    "监控结束但读不到远端退出码，任务状态未知——训练可能仍在 pod 上运行。"
                    "请查 wandb，或重连 pod 看进程是否存活。"
                )
                self._log_error(self._remote_lost_reason)
            logger.info("Remote training finished with exit code %d", self._exit_code)

        except Exception as exc:
            logger.exception("SSH job failed: %s", exc)
            self._log_error(f"SSH error: {exc}")
            if self._exit_code is None:
                self._exit_code = 1
        finally:
            self._save_offset()
            with contextlib.suppress(Exception):
                if self._log_file is not None:
                    self._log_file.close()
                    self._log_file = None
            with contextlib.suppress(Exception):
                if self._ssh_client is not None:
                    self._ssh_client.close()
                    self._ssh_client = None

    # Resolves a policy's declared extras from the remote's lerobot metadata and
    # import-checks them. lerobot[smolvla] expands to lerobot[transformers-dep],
    # so nested `lerobot[...]` entries are followed. lerobot only raises for these
    # inside the model constructor, so a `get_policy_class` probe alone passes and
    # the run then dies after a full dataset download.
    _EXTRAS_PROBE_TEMPLATE = (
        "import importlib, importlib.metadata as md, importlib.util, re, sys\n"
        "want = {want!r}\n"
        "try:\n"
        "    reqs = md.requires('lerobot') or []\n"
        "except md.PackageNotFoundError:\n"
        "    print('LEROBOT-MISSING'); sys.exit(0)\n"
        "pending, seen, dists = list(want), set(), set()\n"
        "while pending:\n"
        "    e = pending.pop()\n"
        "    if e in seen:\n"
        "        continue\n"
        "    seen.add(e)\n"
        "    for r in reqs:\n"
        "        parts = r.split(';', 1)\n"
        "        if len(parts) < 2:\n"
        "            continue\n"
        "        if not any(n == e for n in re.findall(r'extra == \"([^\"]+)\"', parts[1])):\n"
        "            continue\n"
        "        head = parts[0].strip()\n"
        "        dist = re.split(r'[\\[<>!=~ ]', head, 1)[0]\n"
        "        if dist == 'lerobot':\n"
        "            for grp in re.findall(r'\\[([^\\]]+)\\]', head):\n"
        "                pending.extend(s.strip() for s in grp.split(','))\n"
        "        elif dist:\n"
        "            dists.add(dist)\n"
        "missing = []\n"
        "for dist in sorted(dists):\n"
        "    mod = dist.replace('-', '_')\n"
        "    try:\n"
        "        importlib.import_module(mod)\n"
        "        continue\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        if importlib.util.find_spec(mod) is None:\n"
        "            missing.append(dist)\n"
        "    except Exception:\n"
        "        pass\n"
        "print('MISSING ' + ' '.join(missing) if missing else 'POLICY-OK')\n"
    )

    def _run_python_probe(self, client, script: str, timeout: int = 300) -> str:
        python = getattr(self, "_remote_python", None) or effective_python(self._ssh_config)
        cmd = f"{python} -c {self._shell_quote(script)} 2>&1"
        _stdin, stdout, _stderr = client.exec_command(cmd, timeout=timeout)
        return stdout.read().decode("utf-8", errors="replace")

    @staticmethod
    def _extra_pkg(ptype: str) -> str:
        """pip spec installing this policy's training extras."""
        return "lerobot[" + policy_union_extras("", [ptype]) + "]"

    def _install_policy_extras(self, client, ptype: str) -> bool:
        """pip install this policy's lerobot extras on the pod, once per job."""
        pkg = self._extra_pkg(ptype)
        python = getattr(self, "_remote_python", None) or effective_python(self._ssh_config)
        mirror = (
            f" -i {CN_PIP_MIRROR} --trusted-host mirrors.aliyun.com"
            if self._ssh_config.use_cn_mirror
            else ""
        )
        self._log_error(f"远端缺少 '{ptype}' 的训练依赖，自动补装 {pkg}（每台机器一次，可能几分钟）...")
        cmd = f"{python} -m pip install{mirror} {shlex.quote(pkg)} 2>&1 | tail -40"
        try:
            _stdin, stdout, _stderr = client.exec_command(cmd, timeout=1800)
            text = stdout.read().decode("utf-8", errors="replace")
            code = stdout.channel.recv_exit_status()
        except Exception as exc:
            self._log_error(f"自动补装失败：{exc}")
            return False
        for line in [l for l in text.strip().splitlines() if l.strip()][-8:]:
            self._log_error(f"pip: {line[:200]}")
        if code != 0:
            self._log_error(f"自动补装退出码 {code}")
        return code == 0

    def _preflight_disk(self, client) -> str | None:
        """Refuse to start when the pod lacks room for a full run."""
        path = self._ssh_config.remote_workdir or self._remote_home(client)
        cmd = f"df -Pm {self._shell_quote(path)} 2>/dev/null | awk 'NR==2{{print $4}}'"
        try:
            _stdin, stdout, _stderr = client.exec_command(cmd, timeout=30)
            free_mb = int(stdout.read().decode("utf-8", errors="replace").strip() or "-1")
        except Exception as exc:
            logger.warning("Remote disk probe failed (%s) — proceeding", exc)
            return None
        if free_mb < 0:
            return None
        if free_mb < 8192:
            return (
                f"远端 {path} 只剩 {free_mb / 1024:.1f}GB，不足以存数据集和 checkpoint。"
                "请清理 pod（旧任务目录在 ~/lelab_outputs，pip 缓存 ~/.cache/huggingface）后重试。"
            )
        if free_mb < 20480:
            self._log_error(
                f"提醒：远端仅剩 {free_mb / 1024:.1f}GB。可以训练，但 save_freq 太密可能写满磁盘。"
            )
        return None

    def _preflight_policy(self, client) -> str | None:
        """Return an actionable error string if the policy can't be trained
        remotely with the installed lerobot extras; None when good to go."""
        ptype = self._policy_type or "act"
        if ptype == "sac":
            return (
                "SAC trains through lerobot's online-RL pipeline (lerobot.rl), "
                "not dataset training — LeLab cannot run it on an SSH job yet."
            )
        if ptype == "reward_classifier":
            inner = (
                "from lerobot.rewards.factory import get_reward_model_class; "
                "get_reward_model_class('reward_classifier'); print('POLICY-OK')"
            )
        else:
            inner = (
                "from lerobot.policies.factory import get_policy_class; "
                f"get_policy_class('{ptype}'); print('POLICY-OK')"
            )

        try:
            out = self._run_python_probe(client, inner, timeout=120)
        except Exception as exc:
            # Probe failure shouldn't block a training that might work; log and proceed.
            logger.warning("Policy pre-flight probe failed (%s) — proceeding anyway", exc)
            return None

        if "POLICY-OK" not in out:
            if not self._install_policy_extras(client, ptype):
                tail = " | ".join(out.strip().splitlines()[-2:])[:300]
                return (
                    f"Policy '{ptype}' cannot be imported on the remote: {tail}. "
                    "Re-run 'Install lerobot' (the installer now includes all policy extras) "
                    f"or manually: pip install '{self._extra_pkg(ptype)}'."
                )
            try:
                out = self._run_python_probe(client, inner, timeout=120)
            except Exception:
                out = ""
            if "POLICY-OK" not in out:
                tail = " | ".join(out.strip().splitlines()[-2:])[:300]
                return (
                    f"Policy '{ptype}' still cannot be imported after installing extras: {tail}. "
                    "Run the full installer (检查环境 → 安装 lerobot) on this machine."
                )

        # Constructor-time dependencies: get_policy_class succeeds without them.
        want = ["training", *POLICY_INSTALL_EXTRAS.get(ptype, ())]
        script = self._EXTRAS_PROBE_TEMPLATE.format(want=want)
        try:
            verdict = self._run_python_probe(client, script)
        except Exception as exc:
            logger.warning("Extras pre-flight failed (%s) — proceeding", exc)
            return None
        if "LEROBOT-MISSING" in verdict:
            return (
                "远端 Python 里没有安装 lerobot（metadata 查不到）。请在 UI 运行"
                "“检查环境 → 安装 lerobot”后再训练。"
            )
        if "MISSING " in verdict:
            self._install_policy_extras(client, ptype)
            try:
                verdict = self._run_python_probe(client, script)
            except Exception:
                verdict = ""
            if "MISSING " in verdict:
                self._log_error(
                    f"补装后仍报告缺少 {verdict.split('MISSING', 1)[1].strip()}；"
                    "可能是包名与导入名不同，继续尝试训练。"
                )
        return None

    def _ensure_dataset_cache_consistent(self, client) -> None:
        """Drop the remote's cached dataset snapshot if its info/table/data
        disagree (a partial re-upload race). Cheap: parquet row counts come
        from file metadata, no decoding."""
        repo = (getattr(self, "_dataset_repo_id", None) or "").strip()
        if not repo or getattr(self, "_dataset_root", None):
            return
        slug = "datasets--" + repo.replace("/", "--")
        # Two cache layouts exist: lerobot's flat `lerobot/<repo_id>` root and the
        # hub-style `lerobot/hub/datasets--<org>--<name>/snapshots/<rev>`. Probe
        # whichever holds a meta/info.json, newest last.
        probe = (
            "import glob, json, os\n"
            f"roots = [os.path.expanduser('~/.cache/huggingface/lerobot/{repo}')]\n"
            f"roots += sorted(glob.glob(os.path.expanduser('~/.cache/huggingface/lerobot/hub/{slug}/snapshots/*')))\n"
            "roots = [r for r in roots if os.path.isfile(r + '/meta/info.json')]\n"
            "if not roots:\n"
            "    print('NO-CACHE')\n"
            "else:\n"
            "    import pyarrow.parquet as pq\n"
            "    s = roots[-1]\n"
            "    info = json.load(open(s + '/meta/info.json'))\n"
            "    t = info.get('total_frames', -1)\n"
            "    import pandas as pd\n"
            "    tab = pd.concat([pd.read_parquet(p) for p in glob.glob(s + '/meta/episodes/*/*.parquet')])\n"
            "    m = int(tab.dataset_to_index.max())\n"
            "    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in glob.glob(s + '/data/*/*.parquet'))\n"
            "    print('CONSISTENT' if t == m == rows else f'INCONSISTENT {t} {m} {rows}')\n"
        )
        python = getattr(self, "_remote_python", None) or effective_python(self._ssh_config)
        try:
            _stdin, stdout, _stderr = client.exec_command(
                f"{python} -c {self._shell_quote(probe)} 2>&1", timeout=120
            )
            out = stdout.read().decode("utf-8", errors="replace").strip().splitlines()[-1:] or [""]
        except Exception as exc:
            logger.warning("Dataset cache probe failed (%s) — skipping", exc)
            return
        result = out[0].strip()
        if result.startswith("INCONSISTENT"):
            logger.warning("Remote dataset cache for %s is mixed (%s) — clearing", repo, result)
            self._log_error(
                f"Remote dataset cache for {repo} was inconsistent ({result}); "
                "clearing it and re-downloading a consistent snapshot."
            )
            # exec_command returns as soon as the channel opens, so without
            # draining it the training would start while rm is still running and
            # read the stale meta this call exists to remove. The standard HF cache
            # (`~/.cache/huggingface/hub/<slug>`) holds the resolved `main` ref —
            # leave it and snapshot_download re-materialises the deleted file tree.
            clear = (
                f"rm -rf ~/.cache/huggingface/lerobot/hub/{slug} "
                f"~/.cache/huggingface/lerobot/hub/.locks/{slug} "
                f"~/.cache/huggingface/hub/{slug} "
                f"~/.cache/huggingface/hub/.locks/{slug} "
                f"~/.cache/huggingface/lerobot/{repo} "
                "~/.cache/huggingface/datasets; "
                f"test -e ~/.cache/huggingface/lerobot/{repo}/meta/info.json "
                "&& echo STILL-THERE || echo CLEARED"
            )
            try:
                _stdin, stdout, _stderr = client.exec_command(clear, timeout=900)
                done = stdout.read().decode("utf-8", errors="replace")
                stdout.channel.recv_exit_status()
            except Exception as exc:
                logger.warning("Remote dataset cache clear failed (%s)", exc)
                done = ""
            if "CLEARED" in done:
                logger.info("Remote dataset cache for %s cleared", repo)
            else:
                self._log_error(
                    f"Could not clear the remote dataset cache for {repo} — training may "
                    "still read the inconsistent snapshot and fail with an index error."
                )
        elif result.startswith("CONSISTENT"):
            logger.info("Remote dataset cache for %s is consistent", repo)

    def _consume_channel(self, channel) -> None:
        buf = ""
        while not self._stop_event.is_set():
            if channel.recv_ready():
                chunk = channel.recv(4096).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    self._process_line(line)
            elif channel.exit_status_ready():
                remaining = channel.recv(4096).decode("utf-8", errors="replace")
                buf += remaining
                if buf.strip():
                    self._process_line(buf.strip())
                break
            else:
                time.sleep(0.1)

    def _remote_home(self, client=None) -> str:
        """Absolute remote home directory, probed once per runner and cached."""
        cached = getattr(self, "_remote_home_cache", None)
        if cached:
            return cached
        home = "~"
        owned = client is None
        try:
            if owned:
                client = self._connect()
            _stdin, stdout, _stderr = client.exec_command('printf %s "$HOME"', timeout=20)
            probed = stdout.read().decode("utf-8", errors="replace").strip()
            if probed.startswith("/"):
                home = probed
        except Exception as exc:
            logger.warning("Could not probe remote $HOME (%s); leaving path as ~", exc)
        finally:
            if owned and client is not None:
                with contextlib.suppress(Exception):
                    client.close()
        self._remote_home_cache = home
        return home

    def _connect(self, import_paramiko=None, timeout: int = 30):
        """Create a new SSH connection using the stored config."""
        if import_paramiko is None:
            import paramiko as import_paramiko
        client = import_paramiko.SSHClient()
        client.set_missing_host_key_policy(import_paramiko.AutoAddPolicy())
        connect_kwargs: dict = {
            "hostname": self._ssh_config.host,
            "port": self._ssh_config.port,
            "username": self._ssh_config.username,
            "timeout": timeout,
        }
        if self._ssh_config.key_path:
            connect_kwargs["key_filename"] = os.path.expanduser(self._ssh_config.key_path)
        elif self._ssh_config.password:
            connect_kwargs["password"] = self._ssh_config.password
        client.connect(**connect_kwargs)
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(30)
        return client

    def _connect_retry(self, attempts: int = 4, delay: float = 3.0, timeout: int = 15):
        """Connect, tolerating the provider's intermittent SSH failures.

        A single banner error or timeout right after the local machine wakes up
        must not strand a healthy detached training as 'interrupted'.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._connect(timeout=timeout)
            except Exception as exc:
                last = exc
                if attempt < attempts - 1:
                    logger.warning(
                        "SSH connect failed (%s); retry %d/%d in %.0fs",
                        exc,
                        attempt + 1,
                        attempts - 1,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 20.0)
        raise last if last else RuntimeError("SSH connect failed")

    def _has_setsid(self, client) -> bool:
        """True if the remote has util-linux setsid (almost always on Linux)."""
        try:
            _stdin, stdout, _stderr = client.exec_command("command -v setsid", timeout=10)
            return bool(stdout.read().decode().strip())
        except Exception:
            return False

    def _read_remote_exit_code(self, client) -> int | None:
        """Read the exit code file written by the nohup wrapper on the remote.

        Returns None if the file doesn't exist yet.
        """
        if self._remote_exit_code_path is None:
            return None
        try:
            _stdin, stdout, _stderr = client.exec_command(
                f"test -f {self._remote_exit_code_path} && cat {self._remote_exit_code_path} || echo NOT_READY",
                timeout=10,
            )
            result = stdout.read().decode().strip()
            if result.isdigit():
                return int(result)
        except Exception as exc:
            logger.warning("Failed to read exit code file: %s", exc)
        return None

    def _consume_channel_with_pid_check(self, channel, client) -> None:
        """Stream tail output while monitoring if the training process is still running.

        The tail command runs indefinitely, so we check if the training
        process (identified by PID) has finished. When it does, we read any
        remaining log output and exit.

        If the SSH connection drops (common during long trainings), we
        reconnect and resume tailing from the current log position.
        """
        buf = ""
        last_pid_check = 0.0
        pid_check_interval = 5.0
        reconnect_delays = [5, 10, 30, 60, 120]

        while not self._stop_event.is_set():
            try:
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    if not chunk:
                        if channel.exit_status_ready():
                            break
                        time.sleep(0.1)
                        continue
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        self._process_line(line)
                else:
                    now = time.time()
                    if now - last_pid_check >= pid_check_interval:
                        last_pid_check = now
                        if self._remote_pid is not None:
                            try:
                                _stdin, stdout, _stderr = client.exec_command(
                                    f"ps -p {self._remote_pid} > /dev/null && echo running || echo finished",
                                    timeout=5,
                                )
                                status = stdout.read().decode().strip()
                                if status == "finished":
                                    logger.info("Remote training process %d finished", self._remote_pid)
                                    time.sleep(1)
                                    self._drain_tail(channel, client, buf)
                                    return
                            except Exception as exc:
                                logger.warning("PID check failed: %s — attempting reconnect", exc)
                                channel, client, buf = self._reconnect_and_resume_tail(
                                    client, channel, buf, reconnect_delays
                                )
                                if channel is None:
                                    return
                                last_pid_check = time.time()
                    time.sleep(0.1)
            except Exception as exc:
                logger.warning("Channel read error: %s — attempting reconnect", exc)
                channel, client, buf = self._reconnect_and_resume_tail(
                    client, channel, buf, reconnect_delays
                )
                if channel is None:
                    return
                last_pid_check = time.time()

    def _drain_tail(self, channel, client, buf: str) -> None:
        """Read remaining tail output after the training process has finished."""
        try:
            for _ in range(10):
                if channel.recv_ready():
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                    if not chunk:
                        break
                    buf += chunk
                else:
                    time.sleep(0.3)
        except Exception:
            pass
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            self._process_line(line)
        if buf.strip():
            self._process_line(buf.strip())

    def _reconnect_and_resume_tail(self, client, channel, buf: str, delays: list[int]):
        """Try to reconnect SSH and resume tail -f. Returns (channel, client, buf).

        Returns (None, client, buf) if all reconnect attempts fail — the caller
        should exit the monitoring loop.
        """
        with contextlib.suppress(Exception):
            channel.close()
        with contextlib.suppress(Exception):
            client.close()

        for delay in delays:
            if self._stop_event.is_set():
                return None, client, ""
            logger.info("Reconnecting in %ds...", delay)
            time.sleep(delay)
            try:
                import paramiko
                new_client = self._connect(paramiko)
                self._ssh_client = new_client

                if self._remote_pid is not None:
                    _stdin, stdout, _stderr = new_client.exec_command(
                        f"ps -p {self._remote_pid} > /dev/null && echo running || echo finished",
                        timeout=10,
                    )
                    status = stdout.read().decode().strip()
                    if status == "finished":
                        logger.info("Training %d finished (detected during reconnect)", self._remote_pid)
                        self._drain_tail_from_log(new_client)
                        return None, new_client, ""

                new_channel = new_client.get_transport().open_session()
                new_channel.settimeout(0.1)
                new_channel.exec_command(self._tail_resume_cmd())
                logger.info(
                    "Reconnected and resumed tail from line %d", self._lines_processed + 1
                )
                self._channel = new_channel
                return new_channel, new_client, buf
            except Exception as exc:
                logger.warning("Reconnect attempt failed: %s", exc)

        logger.error("All reconnect attempts exhausted")
        self._log_error("SSH connection lost and could not reconnect")
        return None, client, ""

    def _drain_tail_from_log(self, client) -> None:
        """Read unseen remote-log lines to catch up after the process finished.

        Resumes from the first line we have not processed yet, so metrics are
        not double-counted into the curve. Splitting must match the live path's
        "\n" framing: splitlines() would also break on the "\r" characters that
        fill tqdm's progress lines and inflate the offset past the remote file's
        real line count, silently skipping unseen lines on the next resume.
        """
        try:
            start = self._lines_processed + 1
            _stdin, stdout, _stderr = client.exec_command(
                f"tail -n +{start} {self._shell_quote(self._remote_log_path)}", timeout=30
            )
            full_log = stdout.read().decode("utf-8", errors="replace")
            if full_log.endswith("\n"):
                full_log = full_log[:-1]
            if not full_log:
                return
            for line in full_log.split("\n"):
                self._process_line(line)
            self._save_offset()
        except Exception as exc:
            logger.warning("Failed to read remote log: %s", exc)

    def _process_line(self, line: str) -> None:
        # Count every line emitted by the remote log (including blanks) so that
        # reconnect/reattach can resume with `tail -n +N` and replay exactly the
        # lines we have not yet seen. This keeps the loss curve / info intact
        # across SSH drops instead of losing everything written during the gap.
        self._lines_processed += 1
        if self._lines_processed % _OFFSET_FLUSH_EVERY == 0:
            self._save_offset()
        stripped = line.rstrip()
        if not stripped:
            return
        if "is out of bounds for size" in stripped and not getattr(self, "_meta_hint_shown", False):
            self._meta_hint_shown = True
            self._log_error(
                "数据集索引越界：远端 meta/episodes 分片与 data 行数不一致。多为镜像把删除前的"
                "旧 `main` 文件树发回给 pod（陈旧残片会让索引超过 total_frames）。"
                "本 runner 已改为启动前用 huggingface.co 解析 commit SHA 并钉住 revision；"
                "若仍出现，请手动设置 Dataset revision 为当前提交号，或稍后重试等镜像刷新。"
            )
        parse_metrics_into(stripped, self._metrics)
        if self._wandb_run_url is None:
            url = extract_wandb_run_url(stripped)
            if url is not None:
                self._wandb_run_url = url
        log_line = LogLine(timestamp=time.time(), message=stripped)
        if self._log_file is not None:
            try:
                self._log_file.write(log_line.model_dump_json() + "\n")
            except Exception as exc:
                logger.exception("Error writing to log file: %s", exc)
        if self._log_queue.qsize() >= 1000:
            with contextlib.suppress(Empty):
                self._log_queue.get_nowait()
        self._log_queue.put(log_line)

    def _log_error(self, message: str) -> None:
        log_line = LogLine(timestamp=time.time(), message=message)
        if self._log_file is not None:
            try:
                self._log_file.write(log_line.model_dump_json() + "\n")
            except Exception:
                pass
        self._log_queue.put(log_line)

    @staticmethod
    def _shell_quote(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    def _redact(self, text: str) -> str:
        """Keep an injected credential out of the command echo (log file + server stdout)."""
        if self._secret:
            return text.replace(self._secret, "<redacted>")
        return text
