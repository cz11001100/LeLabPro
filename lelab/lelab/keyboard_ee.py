"""Keyboard end-effector teleoperation.

Turns the ``keyboard_ee`` teleoperator's Cartesian deltas into joint targets
through the same IK steps the upstream ``so100_to_so100_EE`` example uses, so
teleoperation and dataset recording can both drive the follower without a
leader arm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.teleoperators.keyboard import KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig

TELEOP_MODE_LEADER = "leader"
TELEOP_MODE_KEYBOARD_EE = "keyboard_ee"


def normalize_teleop_mode(value: str | None) -> str:
    """Map an incoming mode string onto a known mode; anything else is leader."""
    return TELEOP_MODE_KEYBOARD_EE if value == TELEOP_MODE_KEYBOARD_EE else TELEOP_MODE_LEADER


def so101_urdf_path() -> Path:
    """Official SO101 URDF (motor joint names + ``gripper_frame_link``)."""
    return Path(__file__).resolve().parent / "assets" / "so101_new_calib.urdf"


class LeLabKeyboardEE(KeyboardEndEffectorTeleop):
    """Keyboard EE teleoperator with ESC left to the web UI.

    The upstream ``_on_release`` disconnects on ESC. In LeLab ESC is the
    stop-recording shortcut, so disconnecting the listener there would kill the
    active session; the page already owns shutdown.
    """

    def _on_release(self, key):
        char = getattr(key, "char", None)
        self.event_queue.put((char if isinstance(char, str) else key, False))


def make_keyboard_ee_teleop() -> LeLabKeyboardEE:
    return LeLabKeyboardEE(KeyboardEndEffectorTeleopConfig(id="lelab_keyboard"))


@ProcessorStepRegistry.register("lelab_keyboard_ee_to_target")
@dataclass
class KeyboardEEToTarget(RobotActionProcessorStep):
    """Adapt ``{delta_x, delta_y, delta_z, gripper}`` to the EE-target schema.

    ``EEReferenceAndDelta`` expects an enabled flag, absolute-frame deltas and
    a discrete gripper velocity (0 close, 1 stay, 2 open).
    """

    def action(self, action: RobotAction) -> RobotAction:
        dx = float(action.pop("delta_x", 0.0))
        dy = float(action.pop("delta_y", 0.0))
        dz = float(action.pop("delta_z", 0.0))
        gripper = float(action.pop("gripper", 1.0))

        out = dict(action)
        out["enabled"] = bool(dx or dy or dz)
        out["target_x"] = dx
        out["target_y"] = dy
        out["target_z"] = dz
        # Keep orientation fixed: the parallel jaw only needs translation.
        out["target_wx"] = 0.0
        out["target_wy"] = 0.0
        out["target_wz"] = 0.0
        out["gripper_vel"] = gripper
        return out

    def transform_features(self, features):
        return features


def build_keyboard_ee_pipeline(
    robot,
    *,
    step_m: float = 0.005,
    max_step_m: float = 0.05,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    """Deltas -> EE target -> bounds -> gripper -> IK -> ``<motor>.pos`` joints."""
    motor_names = list(robot.bus.motors.keys())
    kinematics = RobotKinematics(
        urdf_path=str(so101_urdf_path()),
        target_frame_name="gripper_frame_link",
        joint_names=motor_names,
    )

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            KeyboardEEToTarget(),
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes={"x": step_m, "y": step_m, "z": step_m},
                motor_names=motor_names,
                use_latched_reference=False,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=max_step_m,
            ),
            GripperVelocityToJoint(discrete_gripper=True),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=False,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
