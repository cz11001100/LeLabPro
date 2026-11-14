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

import logging
import math
import threading
import time
from typing import Any

from pydantic import BaseModel

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from .keyboard_ee import (
    TELEOP_MODE_KEYBOARD_EE,
    TELEOP_MODE_LEADER,
    build_keyboard_ee_pipeline,
    make_keyboard_ee_teleop,
    normalize_teleop_mode,
)
from .utils.config import setup_calibration_files, setup_follower_calibration_file
from .utils.devices import safe_disconnect_device

logger = logging.getLogger(__name__)

# Global variables for teleoperation state
teleoperation_active = False
teleoperation_thread: threading.Thread | None = None
current_robot = None
current_teleop = None
current_teleop_mode: str = TELEOP_MODE_LEADER
# Last accepted request, so a mode switch can restart the same session.
_last_request: "TeleoperateRequest | None" = None
# Guards the start path; the worker owns disconnect so stop() does not race.
_state_lock = threading.Lock()


class TeleoperateRequest(BaseModel):
    leader_port: str
    follower_port: str
    leader_config: str
    follower_config: str
    ema_alpha: float | None = None  # EMA low-pass filter for jitter reduction
    teleop_mode: str = TELEOP_MODE_LEADER  # "leader" or "keyboard_ee"


def get_joint_positions_from_robot(robot, observation: dict | None = None) -> dict[str, float]:
    """
    Extract current joint positions from the robot and convert to URDF joint format.

    lerobot drives the SO-101 with ``use_degrees=True`` by default, so each
    ``observation["<motor>.pos"]`` is already the joint angle in degrees relative
    to the calibration center — which is also the URDF's zero pose. The URDF
    joint value is therefore just that angle converted to radians, for every
    joint. (The gripper reports 0–100 rather than degrees, but that range lands
    inside the Jaw limit, matching the open/closed sweep.)

    Args:
        robot: The robot instance (SO101Follower)

    Returns:
        Dictionary mapping URDF joint names to radian values
    """
    motor_to_urdf_mapping = {
        "shoulder_pan": "Rotation",
        "shoulder_lift": "Pitch",
        "elbow_flex": "Elbow",
        "wrist_flex": "Wrist_Pitch",
        "wrist_roll": "Wrist_Roll",
        "gripper": "Jaw",
    }

    try:
        if observation is None:
            observation = robot.get_observation()

        joint_positions: dict[str, float] = {}
        debug_rows = []
        for motor_name, urdf_joint_name in motor_to_urdf_mapping.items():
            motor_key = f"{motor_name}.pos"
            if motor_key not in observation:
                logger.warning(f"Motor {motor_key} not found in observation")
                joint_positions[urdf_joint_name] = 0.0
                continue

            angle_degrees = observation[motor_key]
            joint_positions[urdf_joint_name] = angle_degrees * math.pi / 180.0
            debug_rows.append(f"{motor_name:14s} {angle_degrees:+8.2f}° → {urdf_joint_name:11s}")

        # Throttled debug print (~once per second at 20 Hz broadcast).
        now = time.time()
        if now - getattr(get_joint_positions_from_robot, "_last_log", 0) > 1.0:
            get_joint_positions_from_robot._last_log = now
            logger.info("[joint-debug]\n  " + "\n  ".join(debug_rows))

        return joint_positions

    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return dict.fromkeys(motor_to_urdf_mapping.values(), 0.0)


def _safe_disconnect(device) -> None:
    """Disconnect a robot/teleop device, swallowing (but logging) any error.

    Used on the connection-failure cleanup path so one device's failure can't
    leave the other holding its serial port open.
    """
    safe_disconnect_device(device, logger)


def handle_start_teleoperation(request: TeleoperateRequest, websocket_manager=None) -> dict[str, Any]:
    """Handle start teleoperation request.

    Connects to both arms *synchronously* so that a connection failure (arm
    unplugged, port busy, power off) is reported back to the caller, rather than
    dying silently in the worker thread while the API has already claimed
    success. Only the teleoperation loop runs in the background thread.
    """
    global teleoperation_active, teleoperation_thread, current_robot, current_teleop
    global current_teleop_mode, _last_request

    from . import record as _record, rollout as _rollout

    mode = normalize_teleop_mode(request.teleop_mode)

    with _state_lock:
        if teleoperation_active:
            return {"success": False, "message": "Teleoperation is already active"}
        if _record.recording_active:
            return {"success": False, "message": "Recording is currently active. Stop it first."}
        if _rollout.inference_active:
            return {"success": False, "message": "Inference is currently active. Stop it first."}
        teleoperation_active = True

    robot = None
    teleop_device = None
    try:
        logger.info(
            f"Starting teleoperation (mode={mode}) with leader port: {request.leader_port}, "
            f"follower port: {request.follower_port}"
        )
        logger.info(f"EMA alpha received from frontend: {request.ema_alpha}")

        # Setup calibration files. Keyboard mode never touches the leader, so a
        # missing/unplugged leader calibration must not block it.
        if mode == TELEOP_MODE_KEYBOARD_EE:
            leader_config_name = ""
            follower_config_name = setup_follower_calibration_file(request.follower_config)
        else:
            leader_config_name, follower_config_name = setup_calibration_files(
                request.leader_config, request.follower_config
            )

        # Create robot and teleop configs
        # EMA lives in the robot's send_action, one implementation shared with
        # recording and rollout; /update-ema reaches it through current_robot.
        robot_config = SO101FollowerConfig(
            port=request.follower_port,
            id=follower_config_name,
            ema_alpha=request.ema_alpha,
        )
        logger.info(f"Robot config created (EMA alpha={request.ema_alpha})")

        # Connect synchronously. If either device fails to connect, clean up the
        # other (so its serial port is released) and report the error — do NOT
        # leave the caller thinking teleoperation started.
        logger.info("Initializing robot and teleop device...")
        robot = SO101Follower(robot_config)
        if mode == TELEOP_MODE_KEYBOARD_EE:
            teleop_device = make_keyboard_ee_teleop()
        else:
            teleop_device = SO101Leader(
                SO101LeaderConfig(port=request.leader_port, id=leader_config_name)
            )

        # Connect each arm separately so the error names which one failed and
        # tells the user what to do, instead of a generic "failed to start".
        logger.info("Connecting to follower arm...")
        try:
            robot.bus.connect()
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the follower arm on {request.follower_port}. "
                "Make sure it's plugged in and powered on, then try again."
            ) from e

        if mode == TELEOP_MODE_KEYBOARD_EE:
            logger.info("Starting keyboard listener...")
            teleop_device.connect()
            if not teleop_device.is_connected:
                raise RuntimeError(
                    "Keyboard teleoperation is unavailable: pynput needs an X11/Windows/macOS "
                    "desktop session (it cannot capture keys on Wayland or headless machines)."
                )
        else:
            logger.info("Connecting to leader arm...")
            try:
                teleop_device.bus.connect()
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the leader arm on {request.leader_port}. "
                    "Make sure it's plugged in and powered on, then try again."
                ) from e

        # Write calibration to motors' memory
        logger.info("Writing calibration to motors...")
        robot.bus.write_calibration(robot.calibration)
        if mode != TELEOP_MODE_KEYBOARD_EE:
            teleop_device.bus.write_calibration(teleop_device.calibration)

        # Connect cameras and configure motors
        logger.info("Connecting cameras and configuring motors...")
        for cam in robot.cameras.values():
            cam.connect()
        robot.configure()
        teleop_device.configure()

        # Keyboard deltas need the FK/IK chain; the leader already speaks joints.
        keyboard_pipeline = None
        if mode == TELEOP_MODE_KEYBOARD_EE:
            logger.info("Building keyboard -> IK pipeline...")
            keyboard_pipeline = build_keyboard_ee_pipeline(robot)

        logger.info("Successfully connected to both devices")

        current_robot = robot
        current_teleop = teleop_device
        current_teleop_mode = mode

        # Stream the arms in the background; the worker owns disconnect so stop()
        # does not race the serial bus from the request thread.
        def teleoperation_worker():
            global teleoperation_active, current_robot, current_teleop

            logger.info("Starting teleoperation loop...")
            try:
                last_broadcast_time = 0
                broadcast_interval = 0.05  # 20 FPS
                # Keyboard mode reads the follower every iteration for IK, so cap
                # it at the control rate instead of spinning the serial bus.
                loop_period = 1.0 / 30 if keyboard_pipeline is not None else 0.0

                while teleoperation_active:
                    loop_start = time.perf_counter()

                    if keyboard_pipeline is not None:
                        obs = robot.get_observation()
                        action = keyboard_pipeline((teleop_device.get_action(), obs))
                    else:
                        obs = None
                        action = teleop_device.get_action()

                    robot.send_action(action)

                    current_time = time.time()
                    if current_time - last_broadcast_time >= broadcast_interval:
                        try:
                            joint_positions = get_joint_positions_from_robot(robot, observation=obs)
                            joint_data = {
                                "type": "joint_update",
                                "joints": joint_positions,
                                "timestamp": current_time,
                            }
                            if websocket_manager and websocket_manager.active_connections:
                                websocket_manager.broadcast_joint_data_sync(joint_data)
                            last_broadcast_time = current_time
                        except Exception as e:
                            logger.error(f"Error broadcasting joint data: {e}")

                    if loop_period:
                        remaining = loop_period - (time.perf_counter() - loop_start)
                        if remaining > 0:
                            time.sleep(remaining)
                    else:
                        time.sleep(0.001)
            except Exception as e:
                logger.error(f"Error during teleoperation loop: {e}")
            finally:
                _safe_disconnect(robot)
                _safe_disconnect(teleop_device)
                logger.info("Teleoperation stopped")
                teleoperation_active = False
                current_robot = None
                current_teleop = None

        teleoperation_thread = threading.Thread(
            target=teleoperation_worker, name="teleoperation-worker", daemon=True
        )
        teleoperation_thread.start()

        _last_request = request.model_copy(update={"teleop_mode": mode})

        return {
            "success": True,
            "message": "Teleoperation started successfully",
            "leader_port": request.leader_port,
            "follower_port": request.follower_port,
            "teleop_mode": mode,
        }

    except Exception as e:
        # Connection (or setup) failed before the loop started: release any
        # device that did open, reset state, and surface the error.
        _safe_disconnect(robot)
        _safe_disconnect(teleop_device)
        teleoperation_active = False
        current_robot = None
        current_teleop = None
        logger.error(f"Failed to start teleoperation: {e}")
        # str(e) is already a user-facing message for the connection failures
        # raised above; the toast title supplies the "error starting" context.
        return {"success": False, "message": str(e)}


def handle_stop_teleoperation() -> dict[str, Any]:
    """Handle stop teleoperation request.

    Signals the worker via `teleoperation_active = False` and waits for it to
    exit. The worker owns the disconnect call, so this avoids racing the
    serial bus from the request thread.
    """
    global teleoperation_active, teleoperation_thread

    if not teleoperation_active:
        return {"success": False, "message": "No teleoperation session is active"}

    logger.info("Stop teleoperation triggered from web interface")
    teleoperation_active = False

    worker = teleoperation_thread
    if worker is not None and worker.is_alive():
        worker.join(timeout=5.0)
        if worker.is_alive():
            logger.warning("Teleoperation worker did not exit within 5s")
    teleoperation_thread = None

    return {"success": True, "message": "Teleoperation stopped successfully"}


def handle_teleoperation_status() -> dict[str, Any]:
    """Handle teleoperation status request"""
    return {
        "teleoperation_active": teleoperation_active,
        "teleop_mode": current_teleop_mode,
        "available_controls": {
            "stop_teleoperation": teleoperation_active,
        },
        "message": "Teleoperation status retrieved successfully",
    }


class SetTeleopModeRequest(BaseModel):
    mode: str


def handle_set_teleop_mode(
    request: SetTeleopModeRequest, websocket_manager=None
) -> dict[str, Any]:
    """Switch the global teleop mode.

    Both teleoperation and dataset recording read this setting. An active
    session is restarted in the new mode so the switch is not silently ignored
    until the next start; a stopped session only records the choice.
    """
    global current_teleop_mode

    mode = normalize_teleop_mode(request.mode)
    previous = current_teleop_mode

    if not teleoperation_active:
        current_teleop_mode = mode
        # The page that owns the switch is reachable only after a session
        # started, so a stopped session means a crashed/ended one: bring it
        # back in the new mode rather than leaving the user with a dead page.
        if _last_request is None:
            return {
                "success": True,
                "mode": mode,
                "restarted": False,
                "message": "Teleop mode saved. It applies the next time teleoperation or recording starts.",
            }
        result = handle_start_teleoperation(
            _last_request.model_copy(update={"teleop_mode": mode}), websocket_manager
        )
        result["mode"] = mode
        result["restarted"] = bool(result.get("success"))
        return result

    if mode == previous:
        return {"success": True, "mode": mode, "restarted": False, "message": "Teleop mode unchanged."}

    if _last_request is None:
        return {"success": False, "message": "No previous teleoperation request to restart from."}

    logger.info(f"Switching teleop mode {previous} -> {mode}; restarting session")
    current_teleop_mode = mode
    stop_result = handle_stop_teleoperation()
    if not stop_result.get("success"):
        return {"success": False, "message": stop_result.get("message", "Could not stop the current session.")}

    result = handle_start_teleoperation(
        _last_request.model_copy(update={"teleop_mode": mode}), websocket_manager
    )
    result["mode"] = mode
    result["restarted"] = bool(result.get("success"))
    return result


class UpdateEmaRequest(BaseModel):
    ema_alpha: float | None = None


def handle_update_ema(request: UpdateEmaRequest) -> dict[str, Any]:
    """Update the EMA alpha for the active teleop session in real time."""
    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    current_robot.set_ema_alpha(request.ema_alpha)
    logger.info(f"EMA alpha updated to {request.ema_alpha}")
    return {"success": True, "ema_alpha": request.ema_alpha}


def handle_get_joint_positions() -> dict[str, Any]:
    """Handle get current robot joint positions request"""
    global current_robot

    if not teleoperation_active or current_robot is None:
        return {"success": False, "message": "No active teleoperation session"}

    try:
        joint_positions = get_joint_positions_from_robot(current_robot)
        return {"success": True, "joint_positions": joint_positions, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting joint positions: {e}")
        return {"success": False, "message": f"Failed to get joint positions: {str(e)}"}
