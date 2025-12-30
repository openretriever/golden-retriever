import sys
from typing import Dict, Any
from dataclasses import dataclass

from retriever.flow import Pipeline, Flow, Rate, Trigger, flow_io
from retriever.ir.struct import IRStruct
from retriever.ir.viz import generate_ascii_graph, save_interactive_html


# --- Dummy Objects for Demo ---
@flow_io
@dataclass
class DummyIO:
    data: Dict[str, Any]

class DummyNode(Flow[DummyIO, DummyIO]):
    def run(self, inp: DummyIO) -> DummyIO: return inp

def main():
    # 1. Import the REAL closed-loop pipeline
    try:
        # Since PYTHONPATH is ".", "examples" is top-level.
        # But this script is in examples/experimental/visualization/
        # and we want examples/experimental/closed_loop_planning/main.py
        
        # We need to make sure we can import from 'examples' as a package 
        # OR as a path. Since we set PYTHONPATH=".", 'examples' should be
        # a package if it has __init__.py, or just a namespace package.
        
        from examples.experimental.closed_loop_planning.pipelines.complete import build_complete_pipeline
        pipe = build_complete_pipeline()
        print(f"Loaded real pipeline: {pipe._name}")
    except ImportError as e:
        print(f"Could not import real pipeline, falling back to dummy: {e}")
        # Fallback to dummy
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
    ir: IRStruct = pipe.build_ir()
    
    # 2. Render ASCII
    print("\n--- Text Visualization (from IR) ---\n")
    print(generate_ascii_graph(ir))
    
    # 3. Export HTML
    save_interactive_html(ir, "closed_loop_viz.html")

if __name__ == "__main__":
    main()
