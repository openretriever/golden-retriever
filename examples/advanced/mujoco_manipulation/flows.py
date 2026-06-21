from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
import rerun as rr
from retriever.flow import Flow, io
from env import MujocoEnv


@io
@dataclass
class Control:
    ctrl: np.ndarray

@io
@dataclass
class State:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    image: Optional[np.ndarray] = None
    # New fields for Reaching Task
    tip_pos: Optional[np.ndarray] = None
    target_pos: Optional[np.ndarray] = None
    jacobian: Optional[np.ndarray] = None

class MujocoEnvFlow(Flow[Control, State]):
    def init(self):
        self.env = MujocoEnv()
        self.env.reset()
        self.render_every = 10
        self.step_count = 0 

    def run(self, inp: Control) -> State:
        # Move the target in a circle
        t = self.env.data.time
        import math
        target_pos = np.array([
            0.5 + 0.2 * math.cos(t),
            0.2 * math.sin(t),
            0.05
        ])
        
        # Default control if none received
        ctrl = inp.ctrl if inp.ctrl is not None else np.zeros(2)
        
        # Step physics with moving target
        state_dict = self.env.step(ctrl, target_pos=target_pos)
        
        # Render occasionally
        self.step_count += 1
        image = None
        if self.step_count % self.render_every == 0:
            image = self.env.render()
            
        return State(
            time=state_dict["time"],
            qpos=state_dict["qpos"],
            qvel=state_dict["qvel"],
            image=image,
            tip_pos=state_dict["tip_pos"],
            target_pos=state_dict["target_pos"],
            jacobian=state_dict["jacobian"],
        )

class ControllerFlow(Flow[State, Control]):
    def init(self):
        # Jacobian Transpose Control Gains
        self.kp_cart = 200.0  # Cartesian stiffness (N/m)
        self.kd_joint = 5.0   # Joint damping (Nms/rad)
        self.tick = 0

    def run(self, inp: State) -> Control:
        if inp.qpos is None or inp.jacobian is None:
            return Control(ctrl=np.zeros(2))

        # 1. Calculate Cartesian Error
        # We only care about X/Y for this planar arm
        # tip_pos is 3D, target_pos is 3D
        err_cart = inp.target_pos - inp.tip_pos
        # Ignore Z error since it's planar
        err_cart[2] = 0.0

        # 2. Virtual Spring Force (F = Kp * dx)
        f_cart = self.kp_cart * err_cart
        
        # 3. Jacobian Transpose (Tau = J^T * F)
        # Jacobian is 3x2 (3D pos, 2 joints)
        # f_cart is 3x1
        # tau is 2x1
        J = inp.jacobian
        tau = J.T @ f_cart 

        # 4. Joint Damping (Stability)
        tau -= self.kd_joint * inp.qvel

        # Clip control
        tau = np.clip(tau, -200, 200)
        
        self.tick += 1
        if self.tick % 50 == 0:
            dist = np.linalg.norm(err_cart)
            print(f"[Controller] t={inp.time:.2f}s, dist={dist:.3f}m, F={np.linalg.norm(f_cart):.1f}N")
             
        return Control(ctrl=tau)

class RerunLoggerFlow(Flow[State, None]):
    """Logs MuJoCo state to Rerun for visualization."""
    def init(self):
        rr.init("mujoco_manipulation", spawn=True)
        print("[Rerun] Visualization started. Check the Rerun window!")
        self.start_time = time.time()
        
        # Setup static world info
        rr.log("world/base", rr.Points3D([0,0,0], colors=[0,0,0], radii=0.02))

    def run(self, inp: State):
        if inp is None or inp.time is None:
            return

        # Use actual wall-clock time offset by sim time
        # This matches sim duration (0, dt, 2dt...) to real time (now, now+dt...)
        real_time = self.start_time + inp.time
        rr.set_time_seconds("sim_time", real_time)
        
        # Add a sequence timeline (sim_step) to avoid "1970" date confusion if desired
        # We estimate step from time since we don't pass step count explicitly in State
        # Or just let it be. 'sim_time' is correct seconds.
        # But let's add it for clarity.
        sim_step = int(inp.time / 0.005)
        rr.set_time_sequence("sim_step", sim_step)
        
        if inp.qpos is not None:
             rr.log("state/joint1", rr.Scalars([float(inp.qpos[0])]))
             rr.log("state/joint2", rr.Scalars([float(inp.qpos[1])]))
             
        if inp.image is not None:
            rr.log("camera/render", rr.Image(inp.image))
            
        if inp.tip_pos is not None:
            rr.log("world/tip", rr.Points3D([inp.tip_pos], colors=[0,0,255], radii=0.03))
            
        if inp.target_pos is not None:
            rr.log("world/target", rr.Points3D([inp.target_pos], colors=[255,0,0], radii=0.03))
