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

"""Training-specific helpers: the request schema and the LeRobot CLI builder.

The actual job lifecycle (subprocess management, registry, log streaming)
lives in app/jobs.py.
"""

import importlib
import json
import logging
import platform
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lelab.jobs import JobTarget

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")

# Policies whose weights sit in bfloat16 and that expose no `dtype` config field
# (pi0/pi05/pi0_fast/xvla do, so lerobot picks the mode for them):
#   smolvla — SmolVLMWithExpert is built with torch_dtype="bfloat16"
#   wall_x  — `self.to(dtype=torch.bfloat16)`
#   groot   — use_bf16=True by default
# Accelerate's fp16 path builds a GradScaler, and its CUDA kernel
# `_amp_foreach_non_finite_check_and_unscale_` has no bf16 support, so the very
# first `clip_grad_norm_` aborts the run. Mixed precision has to follow the
# policy's native weight dtype, and we must not leave it to the environment:
# lerobot passes `mixed_precision=None` whenever the config has no dtype field,
# which lets a pod image's ACCELERATE_MIXED_PRECISION default pick the mode.
_BF16_NATIVE_POLICIES = {"smolvla", "groot", "wall_x"}


def policy_weights_are_bf16(policy_type: str) -> bool:
    """Whether lerobot builds this policy's weights in bfloat16."""
    return policy_type in _BF16_NATIVE_POLICIES


def accelerate_mixed_precision(policy_type: str) -> str:
    """Accelerate mixed-precision mode matching the policy's weight dtype."""
    return "bf16" if policy_weights_are_bf16(policy_type) else "fp16"


def resolve_hub_revision(repo_id: str, repo_type: str = "dataset") -> str | None:
    """Current commit SHA of a Hub repo, read from huggingface.co directly.

    A pod that resolves `main` through a mirror can be handed a cached, stale
    file tree — after remote files were deleted it still downloads them, which
    reintroduces the index-out-of-bounds crash. Pinning the SHA skips that
    resolution entirely. Returns None when the Hub can't be reached.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi(endpoint="https://huggingface.co").repo_info(repo_id, repo_type=repo_type).sha
    except Exception as exc:
        logger.warning("Could not resolve Hub revision for %s (%s)", repo_id, str(exc)[:160])
        return None


def _resolve_video_backend(requested: str | None) -> str:
    """Return the video backend to pass to LeRobot.

    If the caller explicitly chose a backend, honour it.  Otherwise, probe
    whether ``torchcodec`` can actually load at runtime (it may be installed
    but broken on Windows when FFmpeg shared DLLs are missing) and fall back
    to ``pyav`` when it can't.
    """
    if requested:
        return requested
    if importlib.util.find_spec("torchcodec"):
        try:
            importlib.import_module("torchcodec")
            return "torchcodec"
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "torchcodec is installed but cannot be loaded (%s). Falling back to pyav.", exc
            )
            return "pyav"
    return "pyav"


class TrainingRequest(BaseModel):
    # Dataset configuration
    dataset_repo_id: str
    dataset_revision: str | None = None
    dataset_root: str | None = None
    dataset_episodes: list[int] | None = None
    dataset_image_transforms_enable: bool = False
    dataset_video_backend: str | None = None  # "torchcodec" or "pyav"; auto-detected when None

    # Policy configuration
    policy_type: str = "act"

    # Core training parameters
    steps: int = 10000
    batch_size: int = 8
    seed: int | None = 1000
    num_workers: int = 4

    # Logging and checkpointing
    log_freq: int = 250
    save_freq: int = 1000
    env_eval_freq: int = 0
    save_checkpoint: bool = True

    # Output configuration
    output_dir: str = "outputs/train"
    resume: bool = False
    resume_from_job_id: str | None = None
    job_name: str | None = None

    # Weights & Biases
    wandb_enable: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_notes: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = "online"
    wandb_disable_artifact: bool = False

    # Environment / evaluation
    env_type: str | None = None
    env_task: str | None = None
    eval_n_episodes: int = 10
    eval_batch_size: int = 10  # Must be <= eval_n_episodes to pass lerobot validation
    eval_use_async_envs: bool = False

    # Policy-specific
    policy_device: str | None = "cuda"
    policy_use_amp: bool = False
    # Finetune from an existing checkpoint's weights (fresh optimizer/scheduler/
    # step counter, new dataset allowed). Mutually exclusive with resume:
    # resume restores the old lr schedule, this starts a new run from weights.
    policy_pretrained_path: str | None = None
    # Hub upload (set by HfCloudJobRunner; not exposed in the form)
    policy_push_to_hub: bool = False
    policy_repo_id: str | None = None

    # GR00T-specific policy options (only emitted when policy_type == "groot";
    # these flags don't exist on other policy configs, so draccus would reject
    # them). All optional — unset fields fall back to the groot config defaults.
    policy_base_model_path: str | None = None
    policy_embodiment_tag: str | None = None
    policy_chunk_size: int | None = None
    policy_n_action_steps: int | None = None
    policy_use_relative_actions: bool | None = None
    policy_relative_exclude_joints: list[str] | None = None
    policy_use_bf16: bool | None = None

    # Data augmentation (image transforms)
    augmentation_enable: bool = False
    augmentation_max_num_transforms: int = 3
    augmentation_random_order: bool = False
    augmentation_brightness: bool = True
    augmentation_contrast: bool = True
    augmentation_saturation: bool = True
    augmentation_hue: bool = True
    augmentation_sharpness: bool = True
    augmentation_affine: bool = True

    # Optimizer — None = defer to the policy's training preset (e.g. SmolVLA
    # uses AdamW lr=1e-4/betas=(0.9,0.95)). lerobot applies the preset only on
    # fresh runs; on a *resume* an explicit flag here would win, so always
    # sending "adam" silently switched resumed runs off the preset optimizer.
    optimizer_type: str | None = None
    optimizer_lr: float | None = None
    optimizer_weight_decay: float | None = None
    optimizer_grad_clip_norm: float | None = None

    # Scheduler (unset = policy preset defaults)
    scheduler_warmup_steps: int | None = None
    scheduler_decay_steps: int | None = None
    scheduler_decay_lr: float | None = None

    # Advanced
    use_policy_training_preset: bool = True
    config_path: str | None = None


def build_training_command(
    request: TrainingRequest,
    output_dir: str,
    python_executable: str = "python",
    job_target: "JobTarget | None" = None,
) -> list[str]:
    """Build the argv list to invoke `<python_executable> -m lerobot.scripts.lerobot_train`.

    `output_dir` is supplied separately from the request so the caller (the
    JobRegistry) can pin it to the per-job directory rather than relying on
    request.output_dir, which the frontend doesn't even send in the new world.

    `python_executable` defaults to "python" for the cloud runner (whose
    container has lerobot on PATH); the local runner must pass sys.executable
    so the subprocess uses the same interpreter as lelab itself — otherwise
    PATH lookup picks up a different env (uv tool venv, miniforge3 base, etc.)
    that lacks lerobot.
    """
    cmd: list[str] = [python_executable, "-m", "lerobot.scripts.lerobot_train"]

    # Dataset
    cmd.extend(["--dataset.repo_id", request.dataset_repo_id])
    if request.dataset_revision:
        cmd.extend(["--dataset.revision", request.dataset_revision])
    if request.dataset_root:
        cmd.extend(["--dataset.root", request.dataset_root])
    if request.dataset_episodes:
        # draccus parses this as a single YAML token. Spreading the indices over
        # separate argv entries made it read only the first one and reject the
        # rest as unrecognized arguments, so the selection silently did nothing.
        cmd.append("--dataset.episodes=[" + ",".join(str(ep) for ep in request.dataset_episodes) + "]")
    # Data augmentation (image transforms)
    aug_enabled = request.augmentation_enable or request.dataset_image_transforms_enable
    if aug_enabled:
        cmd.extend(["--dataset.image_transforms.enable", "true"])
        cmd.extend(["--dataset.image_transforms.max_num_transforms", str(request.augmentation_max_num_transforms)])
        cmd.extend(["--dataset.image_transforms.random_order", "true" if request.augmentation_random_order else "false"])

        tfs = {}
        if request.augmentation_brightness:
            tfs["brightness"] = {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}
        if request.augmentation_contrast:
            tfs["contrast"] = {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}
        if request.augmentation_saturation:
            tfs["saturation"] = {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}
        if request.augmentation_hue:
            tfs["hue"] = {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}
        if request.augmentation_sharpness:
            tfs["sharpness"] = {"weight": 1.0, "type": "SharpnessJitter", "kwargs": {"sharpness": [0.5, 1.5]}}
        if request.augmentation_affine:
            tfs["affine"] = {"weight": 1.0, "type": "RandomAffine", "kwargs": {"degrees": [-5.0, 5.0], "translate": [0.05, 0.05]}}

        if tfs:
            cmd.extend(["--dataset.image_transforms.tfs", json.dumps(tfs)])

    # Video backend – auto-detect when not explicitly set (fixes torchcodec/FFmpeg
    # DLL loading failures on Windows where the package is installed but broken).
    video_backend = _resolve_video_backend(request.dataset_video_backend)
    cmd.extend(["--dataset.video_backend", video_backend])

    # Policy / reward-model routing. lerobot 0.6 trains the reward classifier
    # through --reward_model.type (rewards factory), not --policy.type; and
    # emitting any --policy.* sub-key without --policy.type makes draccus
    # abort, so the whole policy block is skipped for reward training.
    is_reward_model = request.policy_type == "reward_classifier"
    if is_reward_model:
        cmd.extend(["--reward_model.type", "reward_classifier"])
    else:
        if request.policy_pretrained_path and not request.resume:
            cmd.append(f"--policy.path={request.policy_pretrained_path}")
        cmd.extend(["--policy.type", request.policy_type])

    # Core training params
    cmd.extend(["--steps", str(request.steps)])
    cmd.extend(["--batch_size", str(request.batch_size)])
    cmd.extend(["--num_workers", str(request.num_workers)])
    if request.seed is not None:
        cmd.extend(["--seed", str(request.seed)])

    # Policy device / AMP / hub — reward-model training has no --policy
    # section; any policy sub-key would make draccus abort.
    is_cloud = job_target is not None and job_target.runner == "hf_cloud"
    if not is_reward_model:
        if request.policy_device:
            cmd.extend(["--policy.device", request.policy_device])
        # use_amp drives fp16 autocast, which a bf16-native policy can't use.
        use_amp = request.policy_use_amp and not policy_weights_are_bf16(request.policy_type)
        if request.policy_use_amp and not use_amp:
            logger.warning(
                "Ignoring policy_use_amp for %s: its weights are bf16, which fp16 autocast cannot train.",
                request.policy_type,
            )
        cmd.extend(["--policy.use_amp", "true" if use_amp else "false"])
        # On HF Cloud, lerobot's submit_to_hf owns the model repo and sets push_to_hub on
        # the pod itself; _pod_forwarded_args drops any --policy.push_to_hub/--policy.repo_id
        # we'd pass, so we must not emit them. Local and SSH runs keep the existing behavior:
        # LeRobot defaults push_to_hub=True and demands --policy.repo_id when so.
        if not is_cloud:
            cmd.extend(["--policy.push_to_hub", "true" if request.policy_push_to_hub else "false"])
            if request.policy_push_to_hub and request.policy_repo_id:
                cmd.extend(["--policy.repo_id", request.policy_repo_id])

    # GR00T-specific policy flags. These options only exist on the groot config,
    # so emit them only for --policy.type=groot; sending them to another policy
    # would make draccus abort the run. Each is skipped when unset so the groot
    # config's own defaults apply.
    if request.policy_type == "groot":
        if request.policy_base_model_path:
            cmd.extend(["--policy.base_model_path", request.policy_base_model_path])
        if request.policy_embodiment_tag:
            cmd.extend(["--policy.embodiment_tag", request.policy_embodiment_tag])
        if request.policy_chunk_size is not None:
            cmd.extend(["--policy.chunk_size", str(request.policy_chunk_size)])
        if request.policy_n_action_steps is not None:
            cmd.extend(["--policy.n_action_steps", str(request.policy_n_action_steps)])
        if request.policy_use_relative_actions is not None:
            cmd.extend(
                ["--policy.use_relative_actions", "true" if request.policy_use_relative_actions else "false"]
            )
        if request.policy_relative_exclude_joints is not None:
            # draccus parses list values from a single JSON token, e.g. '["gripper"]'.
            cmd.extend(
                ["--policy.relative_exclude_joints", json.dumps(request.policy_relative_exclude_joints)]
            )
        if request.policy_use_bf16 is not None:
            cmd.extend(["--policy.use_bf16", "true" if request.policy_use_bf16 else "false"])

    # Logging / checkpointing
    cmd.extend(["--log_freq", str(request.log_freq)])
    cmd.extend(["--save_freq", str(request.save_freq)])
    # Only include env_eval_freq if non-zero; older lerobot versions don't have this flag.
    if request.env_eval_freq:
        cmd.extend(["--env_eval_freq", str(request.env_eval_freq)])
    cmd.extend(["--save_checkpoint", "true" if request.save_checkpoint else "false"])

    # Output. On HF Cloud the pod, not this host, runs the trainer: an absolute host
    # output_dir (e.g. ~/.cache/.../outputs/train) is baked into the staged config and
    # the pod crashes trying to mkdir it under /Users. Checkpoints land on the Hub repo
    # anyway, so we omit it for cloud and let lerobot pick its in-pod default.
    # For SSH the same logic applies: the local output_dir path doesn't exist on the
    # remote machine, so we omit it and let the remote lerobot use its default.
    is_remote = job_target is not None and job_target.runner in ("hf_cloud", "ssh")
    if not is_remote:
        cmd.extend(["--output_dir", output_dir])
    cmd.extend(["--resume", "true" if request.resume else "false"])
    if request.job_name:
        cmd.extend(["--job_name", request.job_name])

    # W&B
    cmd.extend(["--wandb.enable", "true" if request.wandb_enable else "false"])
    if request.wandb_enable:
        if request.wandb_project:
            cmd.extend(["--wandb.project", request.wandb_project])
        if request.wandb_entity:
            cmd.extend(["--wandb.entity", request.wandb_entity])
        if request.wandb_notes:
            cmd.extend(["--wandb.notes", request.wandb_notes])
        if request.wandb_run_id:
            cmd.extend(["--wandb.run_id", request.wandb_run_id])
        if request.wandb_mode:
            cmd.extend(["--wandb.mode", request.wandb_mode])
        cmd.extend(["--wandb.disable_artifact", "true" if request.wandb_disable_artifact else "false"])

    # Env
    if request.env_type:
        cmd.extend(["--env.type", request.env_type])
    if request.env_task:
        cmd.extend(["--env.task", request.env_task])

    # Eval
    cmd.extend(["--eval.n_episodes", str(request.eval_n_episodes)])
    cmd.extend(["--eval.batch_size", str(request.eval_batch_size)])
    cmd.extend(["--eval.use_async_envs", "true" if request.eval_use_async_envs else "false"])

    # Optimizer
    if request.optimizer_type:
        cmd.extend(["--optimizer.type", request.optimizer_type])
    if request.optimizer_lr is not None:
        cmd.extend(["--optimizer.lr", str(request.optimizer_lr)])
    if request.optimizer_weight_decay is not None:
        cmd.extend(["--optimizer.weight_decay", str(request.optimizer_weight_decay)])
    if request.optimizer_grad_clip_norm is not None:
        cmd.extend(["--optimizer.grad_clip_norm", str(request.optimizer_grad_clip_norm)])

    # Scheduler (overrides; unset = policy preset defaults)
    if request.scheduler_warmup_steps is not None:
        cmd.extend(["--scheduler.num_warmup_steps", str(request.scheduler_warmup_steps)])
    if request.scheduler_decay_steps is not None:
        cmd.extend(["--scheduler.num_decay_steps", str(request.scheduler_decay_steps)])
    if request.scheduler_decay_lr is not None:
        cmd.extend(["--scheduler.decay_lr", str(request.scheduler_decay_lr)])

    # Advanced
    cmd.extend(["--use_policy_training_preset", "true" if request.use_policy_training_preset else "false"])
    if request.config_path:
        cmd.append(f"--config_path={request.config_path}")

    # HF Jobs: --job.target=<flavor> dispatches the run remotely (lerobot commit #3856).
    # Image/timeout use lerobot's JobConfig defaults. lelab tags its jobs; lerobot always
    # adds a "lerobot" tag too. A pod's local checkpoints die with it, so push each one to
    # the model repo's checkpoints/<step>/ tree (the native replacement for lelab's old
    # in-pod uploader) — that's what makes the trained checkpoints reachable afterwards.
    if is_cloud and job_target.flavor:
        cmd.extend(["--job.target", job_target.flavor])
        cmd.extend(["--job.tags", '["lelab"]'])
        # save_checkpoint_to_hub needs policy.repo_id, which submit_to_hf only sets on the
        # fresh-run path; on a resume it isn't set before validate(), so the flag would
        # abort the submit. A resume already pushes back to its source repo, so skip it.
        if request.save_checkpoint and not request.resume:
            cmd.extend(["--save_checkpoint_to_hub", "true"])

    return cmd
