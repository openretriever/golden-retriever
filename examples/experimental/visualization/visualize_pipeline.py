from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io
from retriever.ir import IR
from retriever.ir.viz import generate_ascii_graph, save_interactive_html


@io
@dataclass
class DummyIO:
    data: Dict[str, Any]


class DummyNode(Flow[DummyIO, DummyIO]):
    def step(self, inp: DummyIO) -> DummyIO:
        return inp


def build_demo_ir() -> IR:
    # Small cyclic pipeline: env -> perception -> planner -> executor -> env.
    pipe = Pipeline("visualization_demo")
    env = DummyNode() @ Rate(hz=10.0)
    perception = DummyNode() @ Trigger("obs")
    planner = DummyNode() @ Trigger("state")
    executor = DummyNode() @ Trigger("plan", "state")

    pipe.connect(env, perception, map={"data": "data"}, sync=Latest())
    pipe.connect(perception, planner, map={"data": "data"}, sync=Latest())
    pipe.connect(planner, executor, map={"data": "data"}, sync=Latest())
    pipe.connect(perception, executor, map={"data": "data"}, sync=Latest())
    pipe.connect(executor, env, map={"data": "data"}, sync=Latest())

    return pipe._build_ir()


def main() -> None:
    ir = build_demo_ir()

    print("\n--- Text Visualization (from IR) ---\n")
    print(generate_ascii_graph(ir))

    output = Path("out/golden_retriever_closed_loop_viz.html")
    output.parent.mkdir(exist_ok=True)
    save_interactive_html(ir, output)
    print(f"\nHTML visualization written to {output}")


if __name__ == "__main__":
    main()
