import sys
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass

from retriever.flow import Pipeline, Flow, Rate, Trigger, flow_io
from retriever.flow.graph import FlowGraph, FlowNode, FlowEdge

@flow_io
@dataclass
class DummyIO:
    data: Dict[str, Any]

class DummyNode(Flow[DummyIO, DummyIO]):
    def run(self, inp: DummyIO) -> DummyIO: return inp

def generate_ascii_graph(pipeline: Pipeline) -> str:
    """
    Generates a simple ASCII representation of the Pipeline's FlowGraph.
    It performs a topological sort (or uses one from the graph) and prints nodes
    along with their outgoing edges.
    """
    graph = pipeline.context.graph
    
    # Check for cycles first
    if graph.has_cycles():
        print("WARNING: Cycle detected in graph. Topological order might not include all nodes.")
        
    # Get nodes and edges
    nodes = graph.nodes
    
    # Build adjacency list for display
    adj: Dict[str, List[FlowEdge]] = {nid: [] for nid in nodes}
    for edge in graph.edges:
        adj[edge.source].append(edge)
        
    # Simple list-based view
    output = []
    output.append(f"Pipeline: {pipeline.name}")
    output.append("=" * (len(pipeline.name) + 10))
    
    for node_id in nodes:
        node = nodes[node_id]
        
        # Determine Trigger/Clock info
        clock_info = "Unknown"
        if node.clock_type:
            clock_info = f"{node.clock_type.__name__}"
            if hasattr(node, "rate_hz") and node.rate_hz:
                 clock_info += f"({node.rate_hz}Hz)"
            elif hasattr(node, "trigger_ports"):
                 clock_info += f"(on {node.trigger_ports})"
        
        output.append(f"[{node_id}] <{clock_info}>")
        
        outgoing = adj[node_id]
        if not outgoing:
            output.append("   (no outputs)")
        else:
            for edge in outgoing:
                adapter_str = f" via {edge.adapter.__class__.__name__}" if edge.adapter else ""
                output.append(f"   --({edge.output_port} -> {edge.input_port}{adapter_str})--> [{edge.target}]")
        output.append("")
        
    return "\n".join(output)

def main():
    # 1. Create a dummy pipeline to visualize
    pipe = Pipeline("visualization_demo")
    
    # Create some nodes
    env = DummyNode() @ Rate(hz=10.0)
    perception = DummyNode() @ Trigger("obs")
    planner = DummyNode() @ Trigger("state")
    executor = DummyNode() @ Trigger("plan", "state")
    
    # Add connections
    pipe.connect(env, perception, map={"data": "data"})
    pipe.connect(perception, planner, map={"data": "data"})
    pipe.connect(planner, executor, map={"data": "data"})
    pipe.connect(perception, executor, map={"data": "data"})
    pipe.connect(executor, env, map={"data": "data"})
    
    # 2. Render ASCII
    print("\n--- Text Visualization ---\n")
    print(generate_ascii_graph(pipe))
    
    # 3. Mention HTML export
    print("\n--- Next Steps ---")
    print("This structure can be exported to JSON/Vis.js/Cytoscape for the HTML UI.")

if __name__ == "__main__":
    main()
