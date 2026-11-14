# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""EnvHub environment: an SO-101 arm grasping a red or blue ball on a MuJoCo tabletop.

The arm is MuJoCo Menagerie's ``robotstudio_so101`` (bundled in this repository, Apache-2.0),
attached at the scene's ``so101_mount`` site. Objects, lights, cameras and physics options live in
``so101_tabletop.xml``. Each reset restores the XML state and applies seeded offsets (``randomize``).

State and actions use the units of LeRobot's ``so101_follower`` with ``use_degrees=True``: degrees
for the arm joints and 0-100 for the gripper. Actions are absolute joint position targets at 30 fps.
The task is ``grasp the red ball`` or ``grasp the blue ball``: reward is 1.0 when the target ball's
center rises 5 cm above its reset height, the episode terminates on success and is truncated after
``episode_length`` steps. A static 10 cm plate sits on the table as a distractor.
"""

import copy
import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

REPO_ID = "lerobot/so101-tabletop-mujoco"
# Not resolved: in the Hugging Face cache, repository files are symlinks into blobs/ named by hash.
HERE = Path(__file__).absolute().parent
# Bundled from MuJoCo Menagerie at commit 0059d4335f8156206f63a35662313385f7ad6d74.
MENAGERIE_MODEL = "robotstudio_so101"
SCENE_FILE = "so101_tabletop.xml"
MOUNT_SITE = "so101_mount"
INSTRUCTION_TEXT = "instruction"
# The real 1021_8 arm is white; the motors stay dark.
ARM_COLOR = (0.90, 0.90, 0.88)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Median first-frame joint pose of CZ11001100/1021_8 (degrees, gripper 0-100).
HOME = (-1.98, -103.43, 91.21, 71.16, -0.97, 37.55)
JOINT_NAMES = [f"{joint}.pos" for joint in JOINTS]

# How LeRobot describes this environment, locally and when served.
LEROBOT_SIM = {
    "fps": 30,
    "control_mode": "position",
    "semantics": "so101-joint-degrees-v1",
    "state_names": JOINT_NAMES,
    "action_names": JOINT_NAMES,
    "robot_type": "so_follower",
}
BALL_BODIES = ("red_ball", "blue_ball")
TARGET_BALLS = {
    "grasp the red ball": "red_ball",
    "grasp the blue ball": "blue_ball",
}
# A ball counts as grasped once its center rises this far above its reset height.
LIFT_SUCCESS = 0.05
# User frame: x=0 at the base right edge, y=0 at the base front edge, +x right,
# +y forward. MuJoCo: X = y + BASE_FRONT_X, Y = -x + BASE_RIGHT_Y.
BASE_FRONT_X = 0.083
BASE_RIGHT_Y = -0.0555
BALL_RADIUS = 0.02


def user_cm_to_mj(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Convert the user's tabletop frame (cm) to MuJoCo world X/Y (m)."""
    return y_cm / 100.0 + BASE_FRONT_X, -x_cm / 100.0 + BASE_RIGHT_Y


def mj_to_user_cm(x_mj: float, y_mj: float) -> tuple[float, float]:
    """Convert MuJoCo world X/Y (m) back to the user's tabletop frame (cm)."""
    return -(y_mj - BASE_RIGHT_Y) * 100.0, (x_mj - BASE_FRONT_X) * 100.0


def repo_root() -> Path:
    """Directory holding the scene and the arm model.

    LeRobot's in-process loader downloads ``env.py`` alone, into ``snapshots/<commit>/`` of the Hub
    cache; fetch the rest of that same commit. A local checkout or a full snapshot needs nothing.
    """
    if (HERE / SCENE_FILE).is_file():
        return HERE
    from huggingface_hub import snapshot_download

    revision = HERE.name if HERE.parent.name == "snapshots" else None
    return Path(snapshot_download(REPO_ID, revision=revision))


def build_model(scene: Path, root: Path) -> mujoco.MjModel:
    """Compile the scene with the Menagerie arm attached at its mount site."""
    spec = mujoco.MjSpec.from_file(str(scene))
    arm = mujoco.MjSpec.from_file(str(root / MENAGERIE_MODEL / "so101.xml"))
    for material in arm.materials:
        if not material.name.startswith("sts3215"):
            material.rgba = [*ARM_COLOR, 1.0]
    # Attaching drops the arm's tuned physics options; keep them unless the scene sets its own.
    if ET.parse(scene).getroot().find("option") is None:  # nosec B314
        for name in dir(arm.option):
            if not name.startswith("_"):
                setattr(spec.option, name, getattr(arm.option, name))
    if spec.site(MOUNT_SITE) is not None:
        spec.attach(arm, site=MOUNT_SITE, prefix="")
    else:
        spec.attach(arm, frame=spec.worldbody.add_frame(), prefix="")
    return spec.compile()


@dataclass
class _BodyNoise:
    """Seeded offsets around one body's XML pose: free bodies move through qpos, static ones directly."""

    body: int
    qadr: int | None
    pos_range: np.ndarray
    yaw_range: float
    footprint: float


def _vector(cfg: Mapping, key: str) -> np.ndarray:
    value = np.asarray(cfg.get(key, (0.0, 0.0, 0.0)), dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"randomize {key!r} must be three half-ranges (x, y, z), got {value.tolist()}")
    return value


class _World:
    """Compiled model, physics state, renderer and reset randomization of one SO-101 world."""

    def __init__(
        self,
        scene: Path,
        root: Path,
        fps: int,
        height: int,
        width: int,
        randomize: Mapping,
        camera_fovy: Mapping[str, float],
    ):
        self.model = build_model(scene, root)
        # The control period must be a whole number of physics steps.
        self.substeps = max(1, round(1 / (fps * self.model.opt.timestep)))
        self.model.opt.timestep = 1 / (fps * self.substeps)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.qadr = np.array([self.model.joint(name).qposadr[0] for name in JOINTS])
        self.ball_bodies = {
            name: self._id(mujoco.mjtObj.mjOBJ_BODY, name, "the task") for name in BALL_BODIES
        }
        self.actuators = np.array([self.model.actuator(name).id for name in JOINTS])
        self.lo, self.hi = self.model.actuator_ctrlrange[self.actuators].T.copy()
        text = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXT, INSTRUCTION_TEXT)
        self.instruction = ""
        if text >= 0:
            start, size = self.model.text_adr[text], self.model.text_size[text]
            self.instruction = self.model.text_data[start : start + size - 1].decode()

        unknown = set(randomize) - {
            "joints",
            "bodies",
            "cameras",
            "ball_workspace",
            "ball_keepouts",
            "ball_spacing",
        }
        if unknown:
            raise ValueError(f"Unknown randomize keys {sorted(unknown)}; expected joints, bodies, cameras, ball_*")
        self.joint_noise = float(randomize.get("joints", 0.0))
        self.ball_workspace = randomize.get("ball_workspace")
        self.ball_keepouts = list(randomize.get("ball_keepouts") or [])
        self.ball_spacing = float(randomize.get("ball_spacing", 0.06))
        self.bodies = [self._body_noise(name, cfg) for name, cfg in (randomize.get("bodies") or {}).items()]
        self.cameras = {
            self._id(mujoco.mjtObj.mjOBJ_CAMERA, name, "randomize.cameras"): _vector(cfg, "pos")
            for name, cfg in (randomize.get("cameras") or {}).items()
        }
        for name, fovy in camera_fovy.items():
            self.model.cam_fovy[self._id(mujoco.mjtObj.mjOBJ_CAMERA, name, "camera_fovy")] = fovy
        self.body_pos0 = self.model.body_pos.copy()
        self.body_quat0 = self.model.body_quat.copy()
        self.cam_pos0 = self.model.cam_pos.copy()
        randomized = {noise.body for noise in self.bodies}
        # Free bodies left at their XML pose are still obstacles for the randomized ones.
        self.fixed_free = [
            body
            for body in range(1, self.model.nbody)
            if self._free_qadr(body) is not None and body not in randomized
        ]

    def _id(self, kind: mujoco.mjtObj, name: str, section: str) -> int:
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"{section} names {name!r}, which this scene does not define")
        return index

    def _free_qadr(self, body: int) -> int | None:
        model = self.model
        if (
            model.body_jntnum[body] == 1
            and model.jnt_type[model.body_jntadr[body]] == mujoco.mjtJoint.mjJNT_FREE
        ):
            return int(model.jnt_qposadr[model.body_jntadr[body]])
        return None

    def _body_noise(self, name: str, cfg: Mapping) -> _BodyNoise:
        unknown = set(cfg) - {"pos", "yaw"}
        if unknown:
            raise ValueError(f"Unknown randomize keys {sorted(unknown)} for body {name!r}; expected pos, yaw")
        body = self._id(mujoco.mjtObj.mjOBJ_BODY, name, "randomize.bodies")
        qadr = self._free_qadr(body)
        if qadr is None and (self.model.body_jntnum[body] or self.model.body_parentid[body] != 0):
            raise ValueError(
                f"Randomized body {name!r} must have a freejoint or be a static child of the world"
            )
        return _BodyNoise(body, qadr, _vector(cfg, "pos"), float(cfg.get("yaw", 0.0)), self._footprint(body))

    def _footprint(self, body: int) -> float:
        """Radius of the body's footprint on the table, from its geoms' bounding boxes."""
        radius = 0.0
        for geom in np.flatnonzero(self.model.geom_bodyid == body):
            center, half = self.model.geom_aabb[geom, :3], self.model.geom_aabb[geom, 3:]
            corner = np.abs(self.model.geom_pos[geom, :2] + center[:2]) + half[:2]
            radius = max(radius, float(np.linalg.norm(corner)))
        return radius

    def _xy(self, body: int) -> np.ndarray:
        qadr = self._free_qadr(body)
        return self.data.qpos[qadr : qadr + 2] if qadr is not None else self.model.body_pos[body, :2]

    def _randomize_balls(self, rng: np.random.Generator) -> None:
        """Place red/blue balls in the user-frame workspace with rejection sampling.

        Rejects poses inside configured keep-out rectangles (plate, stand bases)
        and poses closer than ``ball_spacing`` to an already placed ball.
        """
        if not self.ball_workspace:
            return

        x0, x1 = sorted(self.ball_workspace["x"])
        y0, y1 = sorted(self.ball_workspace["y"])
        placed: list[tuple[float, float]] = []
        for name in BALL_BODIES:
            body = self.ball_bodies[name]
            qadr = self._free_qadr(body)
            if qadr is None:
                raise ValueError(f"Ball {name!r} must have a freejoint")
            for _ in range(300):
                x_cm = rng.uniform(x0, x1)
                y_cm = rng.uniform(y0, y1)

                blocked = False
                for keep in self.ball_keepouts:
                    kx0, kx1 = sorted(keep["x"])
                    ky0, ky1 = sorted(keep["y"])
                    margin = float(keep.get("margin", 0.0))
                    if kx0 - margin <= x_cm <= kx1 + margin and ky0 - margin <= y_cm <= ky1 + margin:
                        blocked = True
                        break
                if blocked:
                    continue

                x_mj, y_mj = user_cm_to_mj(x_cm, y_cm)
                if any(
                    (x_mj - px) ** 2 + (y_mj - py) ** 2 < self.ball_spacing**2
                    for px, py in placed
                ):
                    continue

                self.data.qpos[qadr : qadr + 3] = [x_mj, y_mj, BALL_RADIUS]
                self.data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                placed.append((x_mj, y_mj))
                break
            else:
                raise ValueError(
                    f"Could not place ball {name!r} in workspace {self.ball_workspace} "
                    f"without hitting a keep-out or another ball"
                )

    def reset(self, rng: np.random.Generator, initial_joints: np.ndarray) -> None:
        """Restore the XML state, then apply seeded joint, body, camera and ball offsets."""
        model, data = self.model, self.data
        model.body_pos[:], model.body_quat[:], model.cam_pos[:] = (
            self.body_pos0,
            self.body_quat0,
            self.cam_pos0,
        )
        mujoco.mj_resetData(model, data)
        joints = initial_joints + rng.uniform(-self.joint_noise, self.joint_noise, len(JOINTS))
        data.qpos[self.qadr] = self.to_ctrl(joints)
        data.ctrl[self.actuators] = data.qpos[self.qadr]

        placed = [(self._xy(body).copy(), self._footprint(body)) for body in self.fixed_free]
        for noise in self.bodies:
            for _ in range(100):
                self._place(noise, rng)
                xy = self._xy(noise.body).copy()
                if all(np.linalg.norm(xy - other) > noise.footprint + radius for other, radius in placed):
                    break
            else:
                raise ValueError(f"Could not place body {model.body(noise.body).name!r} without overlap")
            placed.append((xy, noise.footprint))
        for camera, pos_range in self.cameras.items():
            model.cam_pos[camera] = self.cam_pos0[camera] + rng.uniform(-pos_range, pos_range)
        self._randomize_balls(rng)
        mujoco.mj_forward(model, data)

    def _place(self, noise: _BodyNoise, rng: np.random.Generator) -> None:
        offset = rng.uniform(-noise.pos_range, noise.pos_range)
        yaw = rng.uniform(-noise.yaw_range, noise.yaw_range)
        turn = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        if noise.qadr is not None:
            a = noise.qadr
            self.data.qpos[a : a + 3] = self.model.qpos0[a : a + 3] + offset
            mujoco.mju_mulQuat(self.data.qpos[a + 3 : a + 7], turn, self.model.qpos0[a + 3 : a + 7])
        else:
            self.model.body_pos[noise.body] = self.body_pos0[noise.body] + offset
            mujoco.mju_mulQuat(self.model.body_quat[noise.body], turn, self.body_quat0[noise.body])

    def to_ctrl(self, values: np.ndarray) -> np.ndarray:
        """Convert degrees and the 0-100 gripper opening to actuator radians."""
        ctrl = np.deg2rad(values.astype(np.float64))
        ctrl[5] = self.lo[5] + values[5] / 100 * (self.hi[5] - self.lo[5])
        return np.clip(ctrl, self.lo, self.hi)

    def joint_state(self) -> np.ndarray:
        """Joint positions in the units of ``to_ctrl``."""
        q = self.data.qpos[self.qadr]
        state = np.rad2deg(q)
        state[5] = (q[5] - self.lo[5]) / (self.hi[5] - self.lo[5]) * 100
        return state.astype(np.float32)

    def ball_height(self, name: str) -> float:
        """World-space height of a ball's center, in meters."""
        return float(self.data.xpos[self.ball_bodies[name]][2])

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()


class SO101Env(gym.Env):
    """SO-101 joint position control in an MJCF scene, observed through the scene's named cameras."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene: str | Path | None = None,
        fps: int = 30,
        episode_length: int = 400,
        observation_height: int = 480,
        observation_width: int = 640,
        cameras: Mapping[str, str] | None = None,
        instruction: str | None = None,
        initial_joints: Sequence[float] = HOME,
        randomize: Mapping[str, Any] | None = None,
        camera_fovy: Mapping[str, float] | None = None,
        action_smooth: float = 0.0,
    ):
        super().__init__()
        # None: the repository's tabletop scene, located on first use.
        self.scene = Path(scene) if scene is not None else None
        self.task = "so101_tabletop"
        self.fps = fps
        self._max_episode_steps = episode_length
        self._steps = 0
        self.height = observation_height
        self.width = observation_width
        # Observation key -> MJCF camera name.
        self.cameras = dict(cameras or {"up": "top", "side": "front", "wrist": "wrist_cam"})
        self.instruction = instruction
        self.initial_joints = np.asarray(initial_joints, dtype=np.float32)
        if self.initial_joints.shape != (len(JOINTS),):
            raise ValueError(f"initial_joints needs one value per joint {JOINTS}")
        if randomize is None:
            randomize = copy.deepcopy(DEFAULTS["randomize"])
        self.randomize = dict(randomize)
        # MJCF camera name -> vertical field of view in degrees, e.g. to widen the arm's wrist_cam.
        self.camera_fovy = dict(camera_fovy or {})
        # Exponential low-pass on joint targets (0 = off). Trades a little lag
        # for visibly smoother arm motion; ACT temporal ensembling is the
        # preferred smoother and lives in the policy config.
        self.action_smooth = float(np.clip(action_smooth, 0.0, 0.95))
        self._smoothed_action: np.ndarray | None = None
        self._world: _World | None = None
        self._target_ball: str | None = None
        self._ball_z0 = 0.0

        image = spaces.Box(0, 255, (self.height, self.width, 3), np.uint8)
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32),
                "pixels": spaces.Dict(dict.fromkeys(self.cameras, image)),
            }
        )
        self.action_space = spaces.Box(-360.0, 360.0, (len(JOINTS),), np.float32)

    @property
    def task_description(self) -> str:
        """The configured instruction, else the scene's ``<custom><text name="instruction">``."""
        if self.instruction is not None:
            return self.instruction
        return self._ensure_world().instruction

    def _ensure_world(self) -> _World:
        """Build the world on first use, so async workers each own their GL context."""
        if self._world is None:
            root = repo_root()
            self._world = _World(
                self.scene or root / SCENE_FILE,
                root,
                self.fps,
                self.height,
                self.width,
                self.randomize,
                self.camera_fovy,
            )
        return self._world

    def render(self, camera: str | None = None) -> np.ndarray:
        """Render one observation camera, the first one by default."""
        key = camera or next(iter(self.cameras))
        return self._ensure_world().render(self.cameras.get(key, key))

    def _obs(self, world: _World) -> dict[str, Any]:
        return {
            "agent_pos": world.joint_state(),
            "pixels": {key: world.render(camera) for key, camera in self.cameras.items()},
        }

    def reset(self, seed: int | None = None, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore the scene's XML state and apply the seeded randomization."""
        world = self._ensure_world()
        self._smoothed_action = None
        super().reset(seed=seed)
        world.reset(self.np_random, self.initial_joints)

        task = self.instruction if self.instruction is not None else world.instruction
        if task not in TARGET_BALLS:
            raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TARGET_BALLS)}")
        self._target_ball = TARGET_BALLS[task]
        self._ball_z0 = world.ball_height(self._target_ball)
        self._steps = 0
        ball_positions_cm = {}
        for name, body in world.ball_bodies.items():
            x_mj, y_mj = world.data.xpos[body][:2]
            x_cm, y_cm = mj_to_user_cm(x_mj, y_mj)
            ball_positions_cm[name] = [round(float(x_cm), 2), round(float(y_cm), 2)]
        return self._obs(world), {
            "is_success": False,
            "lift": 0.0,
            "target_ball": self._target_ball,
            "ball_positions_cm": ball_positions_cm,
        }

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Hold the joint targets for one control period."""
        if self._target_ball is None:
            raise RuntimeError("Call reset() before step()")
        world = self._ensure_world()
        action = np.asarray(action)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action of shape {self.action_space.shape}, got {action.shape}")
        if self.action_smooth > 0:
            prev = self._smoothed_action if self._smoothed_action is not None else action
            action = self.action_smooth * prev + (1 - self.action_smooth) * action
            self._smoothed_action = action
        world.data.ctrl[world.actuators] = world.to_ctrl(action)
        mujoco.mj_step(world.model, world.data, nstep=world.substeps)
        self._steps += 1
        lift = world.ball_height(self._target_ball) - self._ball_z0
        success = lift >= LIFT_SUCCESS
        truncated = self._steps >= self._max_episode_steps and not success
        info = {"is_success": success, "lift": lift, "target_ball": self._target_ball}
        return self._obs(world), 1.0 if success else 0.0, success, truncated, info

    def close(self) -> None:
        if self._world is not None:
            self._world.renderer.close()
            self._world = None


# Seeded reset randomization of the tabletop scene.
DEFAULTS: dict[str, Any] = {
    "cameras": {"up": "top", "side": "front", "wrist": "wrist_cam"},
    "randomize": {
        "joints": 2.0,
        "bodies": {},
        "cameras": {},
        # Ball centers are sampled in the user frame (cm): x right, y forward.
        # Keep the whole ball inside the up camera: it covers y <= 31 cm (v=480 is
        # the bottom edge), and the 2 cm radius needs ~2 cm of margin.
        "ball_workspace": {"x": [-10.0, 24.0], "y": [10.0, 28.0]},
        # Reject poses inside these rectangles (cm) plus margin (cm).
        "ball_keepouts": [
            {"x": [15.0, 25.0], "y": [20.0, 30.0], "margin": 2.0},  # plate
            {"x": [-6.0, 6.0], "y": [39.0, 51.0], "margin": 1.0},   # up stand base
        ],
        "ball_spacing": 0.06,
    },
}


def make_env(n_envs: int = 1, use_async_envs: bool = False, **kwargs: Any) -> dict[str, dict[int, Any]]:
    """EnvHub entry point: ``{"so101_tabletop": {0: vector_env}}``; ``kwargs`` override ``SO101Env``'s."""
    factory = partial(SO101Env, **{**DEFAULTS, **kwargs})
    vector_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return {"so101_tabletop": {0: vector_cls([factory] * n_envs)}}
