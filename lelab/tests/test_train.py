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
"""Tests for lelab.train — request schema and CLI builder."""

from __future__ import annotations

import pytest


def _arg_value(cmd: list[str], flag: str) -> str:
    """Return the value passed to `--flag`. Fails the test if absent."""
    assert flag in cmd, f"{flag} missing from {cmd}"
    return cmd[cmd.index(flag) + 1]


def test_minimal_request_yields_well_formed_argv() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, output_dir="/tmp/out")

    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert _arg_value(cmd, "--dataset.repo_id") == "lerobot/pusht"
    assert _arg_value(cmd, "--policy.type") == "act"
    assert _arg_value(cmd, "--steps") == "10000"
    assert _arg_value(cmd, "--output_dir") == "/tmp/out"


def test_optional_dataset_fields_only_present_when_set() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="lerobot/pusht")
    cmd = build_training_command(req, "/tmp/out")
    assert "--dataset.revision" not in cmd
    assert "--dataset.root" not in cmd
    assert "--dataset.episodes" not in cmd

    req2 = TrainingRequest(
        dataset_repo_id="lerobot/pusht",
        dataset_revision="v2",
        dataset_root="/data",
        dataset_episodes=[0, 1, 2],
    )
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--dataset.revision") == "v2"
    assert _arg_value(cmd2, "--dataset.root") == "/data"
    # draccus parses this as a single YAML token. Spreading the indices over
    # separate argv entries makes it read only the first and reject the rest,
    # which silently turned the whole episode selection into a no-op.
    assert "--dataset.episodes=[0,1,2]" in cmd2

    # The SSH runner joins the command into one string for the remote shell, so
    # the brackets have to survive a shell round-trip without being globbed.
    import shlex

    assert shlex.split(shlex.join(cmd2)) == cmd2
    assert "--dataset.episodes=[0,1,2]" in shlex.split(shlex.join(cmd2))


def test_wandb_block_only_serialized_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(TrainingRequest(dataset_repo_id="x", wandb_enable=False), "/tmp/out")
    assert _arg_value(off, "--wandb.enable") == "false"
    assert "--wandb.project" not in off

    on = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            wandb_enable=True,
            wandb_project="proj",
            wandb_entity="me",
            wandb_run_id="abc",
        ),
        "/tmp/out",
    )
    assert _arg_value(on, "--wandb.enable") == "true"
    assert _arg_value(on, "--wandb.project") == "proj"
    assert _arg_value(on, "--wandb.entity") == "me"
    assert _arg_value(on, "--wandb.run_id") == "abc"


def test_push_to_hub_emits_repo_id_only_when_enabled() -> None:
    from lelab.train import TrainingRequest, build_training_command

    off = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=False, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(off, "--policy.push_to_hub") == "false"
    assert "--policy.repo_id" not in off

    on = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x"),
        "/tmp/out",
    )
    assert _arg_value(on, "--policy.push_to_hub") == "true"
    assert _arg_value(on, "--policy.repo_id") == "me/x"


def test_seed_omitted_when_none() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", seed=None)
    cmd = build_training_command(req, "/tmp/out")
    assert "--seed" not in cmd

    req2 = TrainingRequest(dataset_repo_id="x", seed=42)
    cmd2 = build_training_command(req2, "/tmp/out")
    assert _arg_value(cmd2, "--seed") == "42"


def test_training_request_validates_required_field() -> None:
    from pydantic import ValidationError

    from lelab.train import TrainingRequest

    with pytest.raises(ValidationError):
        TrainingRequest()  # dataset_repo_id is required


def test_env_eval_freq_flag() -> None:
    from lelab.train import TrainingRequest, build_training_command

    cmd = build_training_command(TrainingRequest(dataset_repo_id="x", env_eval_freq=5000), "/tmp/out")
    # LeRobot main renamed eval_freq -> env_eval_freq (top-level flag, underscore form).
    assert _arg_value(cmd, "--env_eval_freq") == "5000"
    assert "--eval_freq" not in cmd


def test_cloud_target_emits_job_flags_and_skips_push_to_hub() -> None:
    from lelab.jobs import JobTarget
    from lelab.train import TrainingRequest, build_training_command

    # push_to_hub is requested, but for a cloud target lerobot's submit_to_hf owns the
    # model repo and _pod_forwarded_args drops --policy.* — so we must NOT emit them.
    req = TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x")
    cmd = build_training_command(
        req, "/tmp/out", job_target=JobTarget(runner="hf_cloud", flavor="a10g-small")
    )
    assert _arg_value(cmd, "--job.target") == "a10g-small"
    assert _arg_value(cmd, "--job.tags") == '["lelab"]'
    assert "--policy.push_to_hub" not in cmd
    assert "--policy.repo_id" not in cmd
    # An absolute host output_dir would be baked into the staged config and crash the
    # pod (mkdir /Users ...); checkpoints go to the Hub repo, so it must be omitted.
    assert "--output_dir" not in cmd
    # Pod checkpoints are ephemeral, so they must be pushed to the Hub to be reachable.
    assert _arg_value(cmd, "--save_checkpoint_to_hub") == "true"


def test_cloud_resume_omits_save_checkpoint_to_hub() -> None:
    from lelab.jobs import JobTarget
    from lelab.train import TrainingRequest, build_training_command

    # On a cloud resume, submit_to_hf never sets policy.repo_id before validate(), so
    # --save_checkpoint_to_hub would abort the submit. It must be suppressed.
    req = TrainingRequest(dataset_repo_id="x", save_checkpoint=True, resume=True)
    cmd = build_training_command(
        req, "/tmp/out", job_target=JobTarget(runner="hf_cloud", flavor="a10g-small")
    )
    assert "--save_checkpoint_to_hub" not in cmd
    assert _arg_value(cmd, "--job.target") == "a10g-small"


def test_local_target_keeps_push_to_hub() -> None:
    from lelab.jobs import JobTarget
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(dataset_repo_id="x", policy_push_to_hub=True, policy_repo_id="me/x")
    cmd = build_training_command(req, "/tmp/out", job_target=JobTarget(runner="local"))
    assert _arg_value(cmd, "--policy.push_to_hub") == "true"
    assert _arg_value(cmd, "--policy.repo_id") == "me/x"
    assert "--job.target" not in cmd


def test_groot_policy_flags_are_emitted_only_for_groot() -> None:
    from lelab.train import TrainingRequest, build_training_command

    req = TrainingRequest(
        dataset_repo_id="x",
        policy_type="groot",
        policy_base_model_path="nvidia/GR00T-N1.7-3B",
        policy_embodiment_tag="new_embodiment",
        policy_chunk_size=16,
        policy_n_action_steps=16,
        policy_use_relative_actions=True,
        policy_relative_exclude_joints=["gripper"],
        policy_use_bf16=True,
    )
    cmd = build_training_command(req, "/tmp/out")

    assert _arg_value(cmd, "--policy.base_model_path") == "nvidia/GR00T-N1.7-3B"
    assert _arg_value(cmd, "--policy.embodiment_tag") == "new_embodiment"
    assert _arg_value(cmd, "--policy.chunk_size") == "16"
    assert _arg_value(cmd, "--policy.n_action_steps") == "16"
    assert _arg_value(cmd, "--policy.use_relative_actions") == "true"
    assert _arg_value(cmd, "--policy.relative_exclude_joints") == '["gripper"]'
    assert _arg_value(cmd, "--policy.use_bf16") == "true"

    non_groot = build_training_command(
        TrainingRequest(
            dataset_repo_id="x",
            policy_type="act",
            policy_base_model_path="nvidia/GR00T-N1.7-3B",
            policy_embodiment_tag="new_embodiment",
            policy_chunk_size=16,
            policy_n_action_steps=16,
            policy_use_relative_actions=True,
            policy_relative_exclude_joints=["gripper"],
            policy_use_bf16=True,
        ),
        "/tmp/out",
    )
    assert "--policy.base_model_path" not in non_groot
    assert "--policy.embodiment_tag" not in non_groot
    assert "--policy.chunk_size" not in non_groot
    assert "--policy.n_action_steps" not in non_groot
    assert "--policy.use_relative_actions" not in non_groot
    assert "--policy.relative_exclude_joints" not in non_groot
    assert "--policy.use_bf16" not in non_groot


def test_bf16_native_policies_never_receive_fp16_flags() -> None:
    """bf16 weights + an fp16 GradScaler abort training at the first clip_grad_norm_.

    smolvla/groot/wall_x build bf16 weights yet expose no `dtype` config field, so
    lerobot hands the mixed-precision choice to the environment. pi0/pi05/pi0_fast/
    xvla do have that field, so lerobot passes the mode to Accelerator itself and
    our environment value never applies to them.
    """
    from lelab.train import (
        TrainingRequest,
        accelerate_mixed_precision,
        build_training_command,
        policy_weights_are_bf16,
    )

    assert [accelerate_mixed_precision(p) for p in ("smolvla", "groot", "wall_x")] == ["bf16"] * 3
    assert accelerate_mixed_precision("act") == "fp16"
    assert not policy_weights_are_bf16("diffusion")

    smolvla = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_type="smolvla", policy_use_amp=True),
        "/tmp/out",
    )
    assert _arg_value(smolvla, "--policy.use_amp") == "false"

    act = build_training_command(
        TrainingRequest(dataset_repo_id="x", policy_type="act", policy_use_amp=True),
        "/tmp/out",
    )
    assert _arg_value(act, "--policy.use_amp") == "true"
