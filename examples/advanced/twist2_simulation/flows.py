from dataclasses import dataclass
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import time

from retriever import Flow
from retriever.flow import io, gui_flow
from retriever.lib.rerun import rerun_loggable


# --- Data Schemas ---
@io
@dataclass
class MotionOutput:
    action_mimic: Optional[Any]  # np.array(35)


@io
@dataclass
class PolicyOutput:
    policy_action: Optional[Any]  # np.array(29)


@rerun_loggable({"dof_pos": "Scalars", "target": "Scalars"})
@io
@dataclass
class VisState:
    dof_pos: Any
    target: Any
    qpos: Any  # Not logged to Rerun (for native viewer only)
    qvel: Any


@rerun_loggable({"vis": "TimeSeries"})
@io
@dataclass
class EnvOutput:
    proprio: Optional[Any]  # np.array(92)
    vis: Optional[VisState]


@io
@dataclass
class Twist2EnvInput:
    action_mimic: Optional[Any] = None
    policy_action: Optional[Any] = None


@io
@dataclass
class Twist2PolicyInput:
    proprio: Optional[Any] = None
    action_mimic: Optional[Any] = None


@rerun_loggable({"vis": "TimeSeries"})
@io
@dataclass
class VisInput:
    vis: Optional[VisState] = None


# Import rotation utils if available, else copy implementation
try:
    from data_utils.rot_utils import quatToEuler
except ImportError:
    # Minimal implementation of quatToEuler if module not found
    def quatToEuler(quat):
        # quat: [w, x, y, z] -> rpy
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return np.array([roll, pitch, yaw], dtype=np.float32)


class Twist2EnvFlow(Flow[Twist2EnvInput, EnvOutput]):
    """
    High-frequency Physics Environment (1000 Hz).
    Wraps MuJoCo simulation of Unitree G1.
    """

    def __init__(self, xml_path: str, render: bool = False):
        super().__init__()
        self.xml_path = xml_path
        self.render = render
        self.model = None
        self.data = None
        self.viewer = None

        # G1 Constants (from TWIST2)
        self.num_actions = 29
        self.default_dof_pos = np.array(
            [
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,  # left leg (6)
                -0.2,
                0.0,
                0.0,
                0.4,
                -0.2,
                0.0,  # right leg (6)
                0.0,
                0.0,
                0.0,  # torso (3)
                0.0,
                0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,  # left arm (7)
                0.0,
                -0.4,
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,  # right arm (7)
            ],
            dtype=np.float32,
        )

        self.torque_limits = np.array(
            [
                100,
                100,
                100,
                150,
                40,
                40,
                100,
                100,
                100,
                150,
                40,
                40,
                150,
                150,
                150,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
            ],
            dtype=np.float32,
        )

        self.action_scale = np.array(
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=np.float32,
        )

        self.stiffness = np.array(
            [
                100,
                100,
                100,
                150,
                40,
                40,
                100,
                100,
                100,
                150,
                40,
                40,
                150,
                150,
                150,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
                40,
                40,
                40,
                40,
                4.0,
                4.0,
                4.0,
            ],
            dtype=np.float32,
        )
        self.damping = np.array(
            [
                2,
                2,
                2,
                4,
                2,
                2,
                2,
                2,
                2,
                4,
                2,
                2,
                4,
                4,
                4,
                5,
                5,
                5,
                5,
                0.2,
                0.2,
                0.2,
                5,
                5,
                5,
                5,
                0.2,
                0.2,
                0.2,
            ],
            dtype=np.float32,
        )

        self.ankle_idx = [4, 5, 10, 11]

        # State cache for PD control
        self.pd_target = (
            self.default_dof_pos.copy()
        )  # Gets updated by incoming messages
        self.last_action_raw = np.zeros(self.num_actions, dtype=np.float32)

    def init_config(self) -> Dict[str, Any]:
        return {"xml_path": self.xml_path, "render": self.render}

    def init(self):
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = 0.001  # 1000 Hz physics
        self.data = mujoco.MjData(self.model)

        # Initial reset
        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose (approximate standing)
        mujoco_default_dof_pos = np.concatenate(
            [
                np.array([0, 0, 0.793]),  # root pos
                np.array([1, 0, 0, 0]),  # root quat
                self.default_dof_pos,
            ]
        )

        # If we can't set full qpos easily due to size mismatch (e.g. if xml has extra joints),
        # we try to be careful. The XML has 29 DOF + 7 (free joint) = 36 qpos?
        # Let's hope the indices match TWIST2 script assumptions.
        try:
            self.data.qpos[: len(mujoco_default_dof_pos)] = mujoco_default_dof_pos
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
        except Exception as e:
            print(f"Warning: Could not set exact initial pose: {e}")

        if self.render:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print("Twist2EnvFlow: Native Viewer Launched")
            except Exception as e:
                print(f"Twist2EnvFlow: Failed to launch viewer: {e}")
                self.viewer = None

    def run(self, inputs: Twist2EnvInput) -> EnvOutput:
        """
        Main tick (running at 1000 Hz).
        """
        action_mimic = inputs.action_mimic
        policy_action = inputs.policy_action
        # 1. Update PD Target if we got a new Policy Action
        if policy_action is not None:
            # policy_action is the raw action [-1, 1], we need to scale it
            raw = np.array(policy_action, dtype=np.float32)
            self.last_action_raw = raw  # store for next observation

            raw = np.clip(raw, -10.0, 10.0)
            scaled = raw * self.action_scale
            self.pd_target = scaled + self.default_dof_pos

        # 2. Extract Data
        n_dof = self.num_actions
        dof_pos = self.data.qpos[7 : 7 + n_dof].astype(np.float32)
        dof_vel = self.data.qvel[6 : 6 + n_dof].astype(np.float32)
        quat = self.data.qpos[3:7].astype(np.float32)
        ang_vel = self.data.qvel[3:6].astype(np.float32)

        # 3. Step Physics (PD Control)
        torque = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)

        self.data.ctrl[:] = torque
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        # 4. Construct Proprioception Output (for Policy)
        rpy = quatToEuler(quat)
        obs_body_dof_vel = dof_vel.copy()
        obs_body_dof_vel[self.ankle_idx] = 0.0

        # From TWIST2: obs_proprio = [ang_vel*0.25, rpy[:2], (dof_pos - default), obs_body_dof_vel*0.05, last_action]
        obs_proprio = np.concatenate(
            [
                ang_vel * 0.25,
                rpy[:2],
                (dof_pos - self.default_dof_pos),
                obs_body_dof_vel * 0.05,
                self.last_action_raw,
            ]
        ).astype(np.float32)

        return EnvOutput(
            proprio=obs_proprio,
            vis=VisState(
                dof_pos=dof_pos,
                target=self.pd_target,
                qpos=self.data.qpos.astype(np.float32),
                qvel=self.data.qvel.astype(np.float32),
            ),
        )


class Twist2PolicyFlow(Flow[Twist2PolicyInput, PolicyOutput]):
    """
    Policy Inference (50 Hz).
    Wraps ONNXRuntime session.
    """

    def __init__(self, policy_path: str, device: str = "cpu"):
        super().__init__()
        self.policy_path = policy_path
        self.device = device
        self.session = None
        self.input_name = None

        # Buffer Config
        self.n_mimic_obs = 35
        self.n_proprio = 3 + 2 + 3 * 29
        self.n_obs_single = 35 + 3 + 2 + 3 * 29  # 127
        self.history_len = 10
        self.total_obs_size = (
            self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs
        )  # 1402

        # History Buffer
        self.obs_history = []  # deque logic

    def init_config(self) -> Dict[str, Any]:
        return {"policy_path": self.policy_path, "device": self.device}

    def init(self):
        try:
            import onnxruntime as ort

            providers = []
            if (
                self.device == "cuda"
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

            self.session = ort.InferenceSession(self.policy_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            print(f"Twist2PolicyFlow: Loaded {self.policy_path} on {providers[0]}")

            # Init buffer
            self.obs_history = [
                np.zeros(self.n_obs_single, dtype=np.float32)
                for _ in range(self.history_len)
            ]

        except Exception as e:
            print(f"Twist2PolicyFlow Error: {e}")
            self.session = None

    def run(self, inputs: Twist2PolicyInput) -> PolicyOutput:
        proprio = inputs.proprio
        action_mimic = inputs.action_mimic

        if self.session is None or proprio is None or action_mimic is None:
            return PolicyOutput(policy_action=np.zeros(29, dtype=np.float32))

        # 1. Build Single Observation (127 dims)
        # obs_full = [mimic (35), proprio (92)]
        obs_full = np.concatenate([action_mimic, proprio])

        # 2. Update History
        self.obs_history.append(obs_full)
        if len(self.obs_history) > self.history_len:
            self.obs_history.pop(0)

        # 3. Construct Network Input (1432 dims)
        # TWIST2 structure: [obs, priv, history]
        # 127 + 35 + 1270 = 1432

        hist_arr = np.array(self.obs_history)
        hist_flat = hist_arr.flatten()

        obs_buf = np.concatenate([obs_full, action_mimic, hist_flat])

        if obs_buf.shape[0] != self.total_obs_size:
            # Buffer mismatch, can happen during initialization if config differs
            pass

        # 4. Inference
        try:
            obs_np = obs_buf.astype(np.float32).reshape(1, -1)
            outputs = self.session.run(None, {self.input_name: obs_np})
            raw_action = outputs[0][0]  # batch 0
            return PolicyOutput(policy_action=raw_action)
        except Exception as e:
            print(f"Inference Error: {e}")
            return PolicyOutput(policy_action=np.zeros(29, dtype=np.float32))


class MotionPlayerFlow(Flow[None, MotionOutput]):
    """
    Streams motion frames (35 dims) from a pickle file.
    action_mimic = [root_vel_xy(2), root_pos_z(1), roll_pitch(2), yaw_ang_vel(1), dof_pos(29)]
    """

    def __init__(self, motion_file: str):
        super().__init__()
        self.motion_file = motion_file
        self.motion_data = None
        self.idx = 0
        self.num_frames = 0
        self.fps = 50.0
        self.dt = 1.0 / 50.0

    def init_config(self) -> Dict[str, Any]:
        return {"motion_file": self.motion_file}

    def init(self):
        import pickle

        try:
            with open(self.motion_file, "rb") as f:
                self.motion_data = pickle.load(f)

            # Extract data
            self.fps = float(self.motion_data.get("fps", 50.0))
            self.dt = 1.0 / self.fps
            self.root_pos = self.motion_data["root_pos"]  # (N, 3)
            self.root_rot = self.motion_data["root_rot"]  # (N, 4) quaternion wxyz
            self.dof_pos = self.motion_data["dof_pos"]  # (N, 29)
            self.num_frames = len(self.dof_pos)

            # Precompute velocities (finite difference)
            self.root_vel = np.zeros_like(self.root_pos)
            self.root_vel[1:] = (self.root_pos[1:] - self.root_pos[:-1]) / self.dt

            print(
                f"MotionPlayerFlow: Loaded {self.num_frames} frames at {self.fps} FPS"
            )

        except Exception as e:
            print(f"Motion Load Error: {e}")
            self.motion_data = None

    def run(self, *args, **kwargs) -> MotionOutput:
        if self.motion_data is None or self.num_frames == 0:
            return MotionOutput(action_mimic=np.zeros(35, dtype=np.float32))

        # Get current frame (loop)
        frame_idx = self.idx % self.num_frames
        self.idx += 1

        # Extract data for this frame
        root_vel_xy = self.root_vel[frame_idx, :2]  # (2,)
        root_pos_z = self.root_pos[frame_idx, 2:3]  # (1,)

        # Convert quaternion to roll-pitch (simplified)
        quat = self.root_rot[frame_idx]  # wxyz
        w, x, y, z = quat
        # Roll-Pitch from quaternion
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        roll_pitch = np.array([roll, pitch], dtype=np.float32)

        # Yaw angular velocity (approximate from quat changes)
        # For simplicity, use zero or finite diff on yaw
        yaw_ang_vel = np.array([0.0], dtype=np.float32)

        # DOF positions
        dof = self.dof_pos[frame_idx]  # (29,)

        # Construct action_mimic (35 dims)
        action_mimic = np.concatenate(
            [
                root_vel_xy.astype(np.float32),
                root_pos_z.astype(np.float32),
                roll_pitch,
                yaw_ang_vel,
                dof.astype(np.float32),
            ]
        )

        return MotionOutput(action_mimic=action_mimic)


@gui_flow
class Twist2VisFlow(Flow[VisInput, None]):
    """
    Visualization flow with native MuJoCo viewer.

    Uses @gui_flow to run in main thread, enabling native GL rendering.
    Rerun visualization is handled automatically by the pipeline inspection of inputs.
    """

    def __init__(
        self,
        xml_path: str = "assets/g1/g1_sim2sim_29dof.xml",
        max_fps: int = 60,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.model = None
        self.data = None
        self.viewer = None
        self.max_fps = max_fps
        self.last_render_time = 0.0

    def init_config(self) -> Dict[str, Any]:
        return {"xml_path": self.xml_path, "max_fps": self.max_fps}

    def init(self):
        import mujoco
        import mujoco.viewer

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Launch native viewer (requires main thread on macOS; handled specially by @gui_flow)
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )
        print("Twist2VisFlow: Native MuJoCo viewer launched")

    def run(self, inputs: VisInput):
        """Update viewer."""
        vis_state = inputs.vis
        if vis_state is None:
            return

        # Update MuJoCo state for rendering
        if vis_state.qpos is not None:
            self.data.qpos[:] = vis_state.qpos
        if vis_state.qvel is not None:
            self.data.qvel[:] = vis_state.qvel

        import mujoco

        mujoco.mj_forward(self.model, self.data)

        # Sync viewer
        now = time.time()
        if now - self.last_render_time >= 1.0 / self.max_fps:
            self.viewer.sync()
            self.last_render_time = now

    def finalize(self):
        if self.viewer:
            self.viewer.close()
        print("Twist2VisFlow: Viewer closed.")
