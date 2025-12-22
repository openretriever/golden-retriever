
import mujoco
import numpy as np

# Simple 2-link arm model
MODEL_XML = """
<mujoco>
  <option timestep="0.005" integrator="RK4" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".9 .8 .7 1"/>
    
    <!-- Target Object (Red Sphere) -->
    <body name="target" pos="0.5 0.5 0.05" mocap="true">
        <geom type="sphere" size="0.05" rgba="1 0 0 1"/>
    </body>

    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0 0.7 0.7 1"/>
      <body name="link2" pos="0.5 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.05" rgba="0.7 0 0.7 1"/>
        <!-- End Effector Site -->
        <site name="tip" pos="0.5 0 0" size="0.01" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="100"/>
    <motor name="motor2" joint="joint2" gear="100"/>
  </actuator>
</mujoco>
"""

class MujocoEnv:
    def __init__(self, xml_string=MODEL_XML):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_state()

    def step(self, ctrl: np.ndarray, target_pos: np.ndarray = None):
        # Update target visualization if provided
        if target_pos is not None:
             self.data.mocap_pos[0] = target_pos
             
        # Apply control
        self.data.ctrl[:] = ctrl
        # Step physics
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def get_state(self):
        # Compute Jacobian for the tip
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        
        # Get site position
        tip_pos = self.data.site_xpos[site_id].copy()
        
        # Get target position
        target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        # For mocap bodies, their position is in mocap_pos
        # But here 'target' is the body name. We bind mocap_pos[0] to it? 
        # Actually in XML `body name="target" mocap="true"`, so it corresponds to `data.mocap_pos[0]`.
        target_pos = self.data.mocap_pos[0].copy()

        return {
            "time": self.data.time,
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "tip_pos": tip_pos,
            "target_pos": target_pos,
            "jacobian": jacp.copy(), # 3x2 Position Jacobian
        }

    def render(self):
        self.renderer.update_scene(self.data)
        return self.renderer.render()
