"""Verify the locked RoboCasa environment without downloading kitchen assets."""

from __future__ import annotations

from importlib import import_module, metadata

from .mjviser_bridge import MjviserBridge


PACKAGES = (
    ("mujoco", "mujoco"),
    ("robosuite", "robosuite"),
    ("robocasa", "robocasa"),
    ("mjviser", "mjviser"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    for distribution, module in PACKAGES:
        import_module(module)
        print(f"[ok] {distribution} {_version(distribution)}")

    import_module("mujoco.viewer")
    robosuite = import_module("robosuite")
    env = robosuite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    bridge = MjviserBridge(port=0)
    try:
        env.reset()
        numpy = import_module("numpy")
        env.step(numpy.zeros(env.action_dim))
        print(f"[ok] headless RoboSuite Lift step (action_dim={env.action_dim})")
        bridge.update(env.sim)
        print("[ok] mjviser bridge start/update/stop")
    finally:
        bridge.close()
        env.close()

    import_module("robocasa.scripts.download_kitchen_assets")
    print("[ok] RoboCasa asset downloader")


if __name__ == "__main__":
    main()
