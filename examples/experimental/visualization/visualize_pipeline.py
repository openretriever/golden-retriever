import sys
from typing import Dict, Any
from dataclasses import dataclass

from retriever.flow import Pipeline, Flow, Rate, Trigger, io
from retriever.ir import IR
from retriever.ir.viz import generate_ascii_graph, save_interactive_html


# --- Dummy Objects for Demo ---
@io
@dataclass
class DummyIO:
    data: Dict[str, Any]

class DummyNode(Flow[DummyIO, DummyIO]):
    def run(self, inp: DummyIO) -> DummyIO: return inp

def main():
    # Build a small cyclic dummy pipeline for IR visualization. The old
    # closed-loop planning prototype was removed from the public example path;
    # this script now stays self-contained.
    pipe = Pipeline("visualization_demo")
    env = DummyNode() @ Rate(hz=10.0)
    perception = DummyNode() @ Trigger("obs")
    planner = DummyNode() @ Trigger("state")
    executor = DummyNode() @ Trigger("plan", "state")

    pipe.connect(env, perception, map={"data": "data"})
    pipe.connect(perception, planner, map={"data": "data"})
    pipe.connect(planner, executor, map={"data": "data"})
    pipe.connect(perception, executor, map={"data": "data"})
    pipe.connect(executor, env, map={"data": "data"})

    # Build IR once
    ir: IR = pipe.build_ir()

    # 2. Render ASCII
    print("\n--- Text Visualization (from IR) ---\n")
    print(generate_ascii_graph(ir))

    # 3. Export HTML
    save_interactive_html(ir, "closed_loop_viz.html")

if __name__ == "__main__":
    main()
