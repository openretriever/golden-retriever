"""Inspect a RoboSuite scene with MuJoCo's passive native viewer."""

import time

import mujoco
import mujoco.viewer
import robosuite


def main() -> None:
    env = robosuite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    try:
        env.reset()
        model = env.sim.model._model
        data = env.sim.data._data
        print(f"Robot joints: {model.njnt}")
        print(f"Simulation timestep: {model.opt.timestep}")

        mujoco.mj_step(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.geomgroup[0] = 0
            while viewer.is_running():
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
