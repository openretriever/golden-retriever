"""
Flow to Dora Conversion Utilities

This module provides utilities for converting Retriever Flow computation graphs
into dora-rs dataflow graphs and YAML configurations.

Key Components:
- FlowToDoraConverter: Converts Flow graphs to dora node/edge structure
- PipelineToYamlConverter: Generates dora YAML configuration files
- DoraGraphOptimizer: Optimizes dora graphs for performance
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple, Union
import yaml

from ...core.flow import Flow, FlowNode, ModuleNode, FanoutNode, ThenNode


@dataclass
class DoraNode:
    """Represents a single node in a dora dataflow graph."""
    id: str
    name: str
    module_type: str  # "python", "rust", "c++", etc.
    source_code: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DoraEdge:
    """Represents a data connection between dora nodes."""
    id: str
    source_node: str
    source_output: str
    target_node: str
    target_input: str
    data_type: str = "arrow"  # Apache Arrow format by default


@dataclass
class DoraGraph:
    """Complete dora dataflow graph representation."""
    nodes: List[DoraNode] = field(default_factory=list)
    edges: List[DoraEdge] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FlowToDoraConverter:
    """
    Converts Retriever Flow computation graphs to dora-rs dataflow graphs.
    
    This converter recursively walks the Flow graph structure and creates
    corresponding dora nodes and edges for distributed execution.
    
    Example:
        ```python
        converter = FlowToDoraConverter()
        dora_graph = converter.convert(perception_pipeline)
        yaml_config = converter.to_yaml(dora_graph)
        ```
    """
    
    def __init__(self, optimize: bool = True):
        """
        Initialize the converter.
        
        Args:
            optimize: Whether to apply graph optimizations for performance.
        """
        self.optimize = optimize
        self._node_counter = 0
        self._edge_counter = 0
    
    def convert(self, flow: Flow) -> DoraGraph:
        """
        Convert a Flow computation graph to a dora dataflow graph.
        
        Args:
            flow: The Flow to convert
            
        Returns:
            A DoraGraph representing the same computation
        """
        self._node_counter = 0
        self._edge_counter = 0
        
        # Create the graph structure
        dora_graph = DoraGraph()
        
        # Convert the flow recursively
        input_id = self._generate_node_id("input")
        output_id = self._generate_node_id("output")
        
        # Add input/output placeholders
        dora_graph.inputs = [input_id]
        dora_graph.outputs = [output_id]
        
        # Convert the main flow structure
        self._convert_node(
            flow._node, 
            dora_graph, 
            input_connection=input_id,
            output_connection=output_id
        )
        
        # Apply optimizations if requested
        if self.optimize:
            self._optimize_graph(dora_graph)
        
        return dora_graph
    
    def _convert_node(
        self, 
        node: FlowNode, 
        graph: DoraGraph,
        input_connection: str,
        output_connection: str
    ) -> None:
        """
        Recursively convert a Flow node to dora nodes and edges.
        
        Args:
            node: The FlowNode to convert
            graph: The DoraGraph being built
            input_connection: ID of the input connection
            output_connection: ID of the output connection
        """
        if isinstance(node, ModuleNode):
            # Base case: create a single dora node for the module
            dora_node = self._create_module_node(node)
            graph.nodes.append(dora_node)
            
            # Connect input and output
            if input_connection:
                edge_in = DoraEdge(
                    id=self._generate_edge_id(),
                    source_node=input_connection,
                    source_output="output",
                    target_node=dora_node.id,
                    target_input="input"
                )
                graph.edges.append(edge_in)
            
            if output_connection:
                edge_out = DoraEdge(
                    id=self._generate_edge_id(),
                    source_node=dora_node.id,
                    source_output="output",
                    target_node=output_connection,
                    target_input="input"
                )
                graph.edges.append(edge_out)
        
        elif isinstance(node, ThenNode):
            # Sequential composition: create intermediate connection
            intermediate_id = self._generate_node_id("intermediate")
            
            # Convert first node
            self._convert_node(
                node.first._node,
                graph,
                input_connection=input_connection,
                output_connection=intermediate_id
            )
            
            # Convert second node
            self._convert_node(
                node.second._node,
                graph,
                input_connection=intermediate_id,
                output_connection=output_connection
            )
        
        elif isinstance(node, FanoutNode):
            # Parallel composition: create split and merge nodes
            split_id = self._generate_node_id("split")
            merge_id = self._generate_node_id("merge")
            
            # Create split node (duplicates input to both branches)
            split_node = DoraNode(
                id=split_id,
                name=f"fanout_split_{self._node_counter}",
                module_type="python",
                source_code=self._generate_split_code(),
                inputs=["input"],
                outputs=["left", "right"]
            )
            graph.nodes.append(split_node)
            
            # Create merge node (combines outputs from both branches)
            merge_node = DoraNode(
                id=merge_id,
                name=f"fanout_merge_{self._node_counter}",
                module_type="python",
                source_code=self._generate_merge_code(),
                inputs=["left", "right"],
                outputs=["output"]
            )
            graph.nodes.append(merge_node)
            
            # Connect input to split
            if input_connection:
                edge_to_split = DoraEdge(
                    id=self._generate_edge_id(),
                    source_node=input_connection,
                    source_output="output",
                    target_node=split_id,
                    target_input="input"
                )
                graph.edges.append(edge_to_split)
            
            # Convert left branch
            left_intermediate = self._generate_node_id("left_branch")
            self._convert_node(
                node.first._node,
                graph,
                input_connection=split_id + ":left",
                output_connection=left_intermediate
            )
            
            # Convert right branch
            right_intermediate = self._generate_node_id("right_branch")
            self._convert_node(
                node.second._node,
                graph,
                input_connection=split_id + ":right",
                output_connection=right_intermediate
            )
            
            # Connect branches to merge
            edge_left_merge = DoraEdge(
                id=self._generate_edge_id(),
                source_node=left_intermediate,
                source_output="output",
                target_node=merge_id,
                target_input="left"
            )
            graph.edges.append(edge_left_merge)
            
            edge_right_merge = DoraEdge(
                id=self._generate_edge_id(),
                source_node=right_intermediate,
                source_output="output",
                target_node=merge_id,
                target_input="right"
            )
            graph.edges.append(edge_right_merge)
            
            # Connect merge to output
            if output_connection:
                edge_from_merge = DoraEdge(
                    id=self._generate_edge_id(),
                    source_node=merge_id,
                    source_output="output",
                    target_node=output_connection,
                    target_input="input"
                )
                graph.edges.append(edge_from_merge)
        
        else:
            raise TypeError(f"Unknown FlowNode type: {type(node)}")
    
    def _create_module_node(self, module_node: ModuleNode) -> DoraNode:
        """Create a dora node from a Flow ModuleNode."""
        node_id = self._generate_node_id("module")
        
        # Generate Python code for the module
        source_code = self._generate_module_code(module_node)
        
        return DoraNode(
            id=node_id,
            name=f"module_{self._node_counter}",
            module_type="python",
            source_code=source_code,
            inputs=["input"],
            outputs=["output"],
            metadata={
                "original_function": str(module_node.func),
                "flow_type": "ModuleNode"
            }
        )
    
    def _generate_module_code(self, module_node: ModuleNode) -> str:
        """Generate Python code for executing a module in dora."""
        return f'''
#!/usr/bin/env python3
from dora import Node
import pickle

# Initialize dora node
node = Node()

# Main execution loop
for event in node:
    if event["type"] == "INPUT":
        # Deserialize input data
        input_data = pickle.loads(event["data"])
        
        # Execute the original function
        # TODO: Replace with actual function execution
        result = module_function(input_data)
        
        # Serialize and send output
        output_data = pickle.dumps(result)
        node.send_output("output", output_data, event["metadata"])
    
    elif event["type"] == "STOP":
        break
'''
    
    def _generate_split_code(self) -> str:
        """Generate code for fanout split node."""
        return '''
#!/usr/bin/env python3
from dora import Node

node = Node()

for event in node:
    if event["type"] == "INPUT":
        # Duplicate input to both outputs
        node.send_output("left", event["data"], event["metadata"])
        node.send_output("right", event["data"], event["metadata"])
    elif event["type"] == "STOP":
        break
'''
    
    def _generate_merge_code(self) -> str:
        """Generate code for fanout merge node."""
        return '''
#!/usr/bin/env python3
from dora import Node
import pickle

node = Node()
left_result = None
right_result = None

for event in node:
    if event["type"] == "INPUT":
        if event["id"] == "left":
            left_result = pickle.loads(event["data"])
        elif event["id"] == "right":
            right_result = pickle.loads(event["data"])
        
        # Send combined result when both are available
        if left_result is not None and right_result is not None:
            combined = (left_result, right_result)
            output_data = pickle.dumps(combined)
            node.send_output("output", output_data, event["metadata"])
            
            # Reset for next iteration
            left_result = None
            right_result = None
    
    elif event["type"] == "STOP":
        break
'''
    
    def _generate_node_id(self, prefix: str) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}_{uuid.uuid4().hex[:8]}"
    
    def _generate_edge_id(self) -> str:
        """Generate unique edge ID."""
        self._edge_counter += 1
        return f"edge_{self._edge_counter}_{uuid.uuid4().hex[:8]}"
    
    def _optimize_graph(self, graph: DoraGraph) -> None:
        """Apply optimization passes to the dora graph."""
        # TODO: Implement graph optimizations
        # - Remove redundant nodes
        # - Merge sequential operations where possible
        # - Optimize data flow paths
        pass


class PipelineToYamlConverter:
    """
    Converts dora graphs to YAML configuration files.
    
    This converter takes DoraGraph objects and generates the YAML
    configuration files that dora-rs needs for execution.
    """
    
    def convert(self, dora_graph: DoraGraph) -> str:
        """
        Convert a DoraGraph to dora YAML configuration.
        
        Args:
            dora_graph: The graph to convert
            
        Returns:
            YAML configuration string
        """
        config = {
            "nodes": [],
            "edges": []
        }
        
        # Convert nodes
        for node in dora_graph.nodes:
            node_config = {
                "id": node.id,
                "name": node.name,
                "source": node.source_code,
                "inputs": node.inputs,
                "outputs": node.outputs
            }
            if node.metadata:
                node_config["metadata"] = node.metadata
            config["nodes"].append(node_config)
        
        # Convert edges  
        for edge in dora_graph.edges:
            edge_config = {
                "source": edge.source_node,
                "source_output": edge.source_output,
                "target": edge.target_node,
                "target_input": edge.target_input
            }
            config["edges"].append(edge_config)
        
        return yaml.dump(config, default_flow_style=False)