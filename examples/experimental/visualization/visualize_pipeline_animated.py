"""Mock demo: export an animated, draggable pipeline graph.

Companion to ``visualize_pipeline.py``. That one renders the static IR graph;
this one illustrates configured clocks, not execution replay or measured rates.
Only constructs and validates passthrough mock flows; no model or actuator runs.

Run from the repository root::

    python examples/experimental/visualization/visualize_pipeline_animated.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io
from retriever.ir import IR

from animated_graph import save_animated_html


# Each hop carries a differently-named payload so the exported edges get
# meaningful labels instead of six edges all reading "data".
@io
@dataclass
class TorqueIO:
    torque: Dict[str, Any]


@io
@dataclass
class ObsIO:
    obs: Dict[str, Any]


@io
@dataclass
class ImageIO:
    image: Dict[str, Any]


@io
@dataclass
class BeliefIO:
    belief: Dict[str, Any]


@io
@dataclass
class PlanIO:
    plan: Dict[str, Any]


@io
@dataclass
class ActionIO:
    action: Dict[str, Any]


class EnvFlow(Flow[TorqueIO, ObsIO]):
    def step(self, inp: TorqueIO) -> ObsIO:
        return ObsIO(obs=inp.torque)


class CameraFlow(Flow[ObsIO, ImageIO]):
    def step(self, inp: ObsIO) -> ImageIO:
        return ImageIO(image=inp.obs)


class BeliefFlow(Flow[ImageIO, BeliefIO]):
    def step(self, inp: ImageIO) -> BeliefIO:
        return BeliefIO(belief=inp.image)


class PlannerFlow(Flow[BeliefIO, PlanIO]):
    def step(self, inp: BeliefIO) -> PlanIO:
        return PlanIO(plan=inp.belief)


class SkillFlow(Flow[PlanIO, ActionIO]):
    def step(self, inp: PlanIO) -> ActionIO:
        return ActionIO(action=inp.plan)


class ControllerFlow(Flow[ActionIO, TorqueIO]):
    def step(self, inp: ActionIO) -> TorqueIO:
        return TorqueIO(torque=inp.action)


def build_demo() -> Tuple[IR, Dict[str, str]]:
    """Build the demo loop and return its IR plus a name -> node id map.

    Node ids are assigned by the builder (``EnvFlow__0`` and friends), so the
    presentation maps below are keyed off what the builder actually chose
    rather than off names we hoped for.
    """
    pipe = Pipeline("animated_visualization_demo")

    env = EnvFlow() @ Rate(hz=200.0)
    camera = CameraFlow() @ Rate(hz=30.0)
    belief = BeliefFlow() @ Trigger("image")
    planner = PlannerFlow() @ Rate(hz=0.5)
    skill = SkillFlow() @ Rate(hz=2.0)
    controller = ControllerFlow() @ Rate(hz=200.0)

    pipe.connect(env, camera, map={"obs": "obs"}, sync=Latest())
    pipe.connect(camera, belief, map={"image": "image"}, sync=Latest())
    pipe.connect(belief, planner, map={"belief": "belief"}, sync=Latest())
    pipe.connect(planner, skill, map={"plan": "plan"}, sync=Latest())
    pipe.connect(skill, controller, map={"action": "action"}, sync=Latest())
    pipe.connect(controller, env, map={"torque": "torque"}, sync=Latest())

    handles = {
        "env": env,
        "camera": camera,
        "belief": belief,
        "planner": planner,
        "skill": skill,
        "controller": controller,
    }
    ids = {name: pipe.get_node_id(h) for name, h in handles.items()}
    return pipe.validate(), ids


def main() -> None:
    ir, ids = build_demo()

    output = save_animated_html(
        ir,
        Path("out/golden_retriever_animated_graph.html"),
        # Presentation the IR cannot infer: which box is the world rather than
        # a Flow, and which two layers the figure is actually about.
        kinds={ids["env"]: "env", ids["planner"]: "major", ids["skill"]: "major"},
    )
    print(f"Animated visualization written to {output}")
    print("Configured-clock illustration only; not execution replay.")


if __name__ == "__main__":
    main()
