import robosuite
import mujoco

# 1. Create a standard Robosuite environment
# "Lift" is a classic task; we'll use the Franka Panda robot
env = robosuite.make(
    env_name="PickPlace",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# 2. Reset the environment
# This compiles the MJCF and populates env.sim
env.reset()

# 3. Access the underlying MuJoCo objects
# env.sim is the Robosuite simulation wrapper
mj_model = env.sim.model._model
mj_data = env.sim.data._data

# 4. Use it as a "regular" MuJoCo model
print(f"Model successfully grabbed!")
print(f"Robot joints: {mj_model.njnt}")
print(f"Simulation timestep: {mj_model.opt.timestep}")

# Example: Manually stepping the physics via MuJoCo directly
mujoco.mj_step(mj_model, mj_data)

# 6. (Optional) Export the raw XML for external MuJoCo use
xml_string = env.sim.model.get_xml()

# Launch the official MuJoCo viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.opt.geomgroup[0] = 0  # Disable collisions

    # Close the viewer by closing the window or breaking the loop
    while viewer.is_running():
        # Sync the viewer with the latest physics state
        viewer.sync()
