"""Golden Hub pack smoke test.

This source-checkout demo exercises the same manifest shape that public
``hub.use("openretriever/golden-retriever:WorldState")`` will use after the
repository and Hub index are public. It avoids network access by loading this
repo's local ``[tool.retriever.module]`` manifest directly through the runtime
Hub loader.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from retriever.error import HubError
from retriever.hub._loader import load_exports
from retriever.registry.types import get_type_info, list_types
from retriever.types.spatial import Quaternion, SE3Pose, Vector3


def _load_local_exports() -> dict[str, object]:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    manifest = config["tool"]["retriever"]["module"]
    kwargs = {
        "namespace": "golden_hub_pack_smoke",
        "hub_meta": {
            "org": "openretriever",
            "name": "golden-retriever",
            "commit": "local",
        },
    }
    try:
        return load_exports(Path("."), manifest["module"], manifest["exports"], **kwargs)
    except HubError as exc:
        # Compatibility path for older local runtimes that do not yet search
        # repo-root src/ layouts. The public runtime supports the repo root.
        if "Package directory" not in str(exc):
            raise
        return load_exports(Path("src"), manifest["module"], manifest["exports"], **kwargs)


def main() -> None:
    exports = _load_local_exports()
    selected = [
        "WorldState",
        "BeliefGraph",
        "Skill",
        "Plan",
        "Trajectory",
        "convert_to_arrow",
        "convert_from_arrow",
    ]
    print("Golden Hub exports:", ", ".join(selected))

    WorldState = exports["WorldState"]
    Skill = exports["Skill"]
    Plan = exports["Plan"]
    Action = exports["Action"]
    convert_to_arrow = exports["convert_to_arrow"]
    convert_from_arrow = exports["convert_from_arrow"]

    pose = SE3Pose(
        position=Vector3(x=0.1, y=0.2, z=0.3),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    world = WorldState(object_poses={"cup": pose}, robot_pose=pose, timestamp=0.0)
    plan = Plan(skills=[Skill(name="pick", params={"object": "cup"}, confidence=0.95)])
    action = Action(type="pick", parameters={"object": "cup"}, timestamp=0.0, priority=1)

    restored_action = convert_from_arrow(convert_to_arrow(action), Action)
    assert restored_action == action

    world_info = get_type_info("WorldState")
    type_names = list_types()
    assert "WorldState" in type_names
    assert "Plan" in type_names

    print("Registry WorldState:", world_info.type_class.__module__ + "." + world_info.type_class.__name__)
    print("Constructed WorldState:", sorted(world.object_poses.keys()))
    print("Constructed Plan skills:", [skill.name for skill in plan.skills])
    print("Arrow round-trip: Action OK")
    print('Public path after launch: hub.use("openretriever/golden-retriever:WorldState")')
    print("Graph proof: run `pixi run demo-pipeline-html-viz` to validate and render an IR HTML artifact.")


if __name__ == "__main__":
    main()
