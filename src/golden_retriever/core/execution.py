"""
Execution Backends for Retriever Flow v2.7 - Dora-Focused Architecture

This module provides execution strategies for Flow pipelines with primary focus
on conversion to Dora dataflow graphs. The architecture supports development
with simple backends while targeting production deployment on Dora-rs.

Design Principles:
- Primary goal: Convert Flow[I,O] compositions to Dora YAML dataflows
- Development support: Simple sequential/threading backends for testing
- Dora-first: All patterns designed for efficient Dora conversion
- Time-aware execution: Integrated with timing primitives for real-time coordination
"""

import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from rich import print
except ImportError:
    # Fallback to regular print if rich not available
    pass

from .temporal import ExecutionTimer, FrequencyCoordinator, TimeConstraint

I = TypeVar('I')
O = TypeVar('O')
X = TypeVar('X')
Y = TypeVar('Y')
S = TypeVar('S')

# Forward reference - will be resolved when types.py imports this
if False:  # TYPE_CHECKING
    from .types import Pipeline


class FRPCoordinator:
    """Internal FRP coordination engine for Flow execution.
    
    Handles rate coordination, event triggers, and feedback loops
    without exposing FRP complexity to users.
    """
    
    def __init__(self):
        self.flow_configs = {}  # flow_id -> FRPConfig
        self.event_triggers = {}  # trigger_name -> List[flow_id]
        self.feedback_connections = {}  # source_id -> List[target_id]
        self.rate_groups = {}  # rate -> List[flow_id]
    
    def register_flow(self, flow: 'Flow', flow_id: str = None):
        """Register a flow with its FRP configuration."""
        if not hasattr(flow, 'frp_config'):
            return  # Not an FRP-enabled flow
        
        effective_id = flow_id or flow.flow_id or f"flow_{id(flow)}"
        self.flow_configs[effective_id] = flow.frp_config
        
        # Group by rate
        if flow.frp_config.rate:
            self.rate_groups.setdefault(flow.frp_config.rate, []).append(effective_id)
        
        # Group by trigger
        if flow.frp_config.trigger:
            self.event_triggers.setdefault(flow.frp_config.trigger, []).append(effective_id)
        
        # Register feedback connections
        if flow.frp_config.feedback_to:
            self.feedback_connections.setdefault(effective_id, []).append(flow.frp_config.feedback_to)
    
    def get_dora_timer_config(self, rate: str) -> str:
        """Convert rate specification to Dora timer format."""
        if rate.endswith('hz'):
            # Convert Hz to milliseconds
            hz = float(rate[:-2])
            ms = int(1000 / hz)
            return f"dora/timer/millis/{ms}"
        elif rate.endswith('ms'):
            # Direct milliseconds
            ms = int(rate[:-2])
            return f"dora/timer/millis/{ms}"
        elif rate.endswith('s'):
            # Convert seconds to milliseconds
            s = float(rate[:-1])
            ms = int(s * 1000)
            return f"dora/timer/millis/{ms}"
        else:
            # Default to 50ms (20Hz)
            return "dora/timer/millis/50"
    
    def generate_feedback_connections(self) -> List[Tuple[str, str]]:
        """Generate list of (source, target) feedback connections for YAML."""
        connections = []
        for source_id, target_ids in self.feedback_connections.items():
            for target_id in target_ids:
                connections.append((source_id, target_id))
        return connections


class ExecutionBackend(Enum):
    """Available execution backends."""
    SEQUENTIAL = "sequential"  # Single-threaded, immediate execution - for development
    THREADING = "threading"    # Multi-threaded with timing coordination - for testing
    DORA = "dora"             # Dora-rs dataflow backend with FRP - for production


@dataclass
class ExecutionConfig:
    """Configuration for Flow execution."""
    backend: ExecutionBackend = ExecutionBackend.SEQUENTIAL
    timing_enabled: bool = False
    performance_monitoring: bool = True
    max_concurrent_flows: int = 10
    default_timeout: float = 10.0
    dora_config_path: Optional[str] = None  # Path for generated Dora YAML


class FlowExecutor(ABC, Generic[I, O]):
    """Abstract base for Flow execution backends."""
    
    @abstractmethod
    def execute_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Execute a single flow with input data."""
        pass
    
    @abstractmethod  
    def execute_pipeline(self, pipeline: 'Pipeline', input_data: Any) -> Any:
        """Execute a pipeline of composed flows."""
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this executor."""
        pass


class SequentialExecutor(FlowExecutor[I, O]):
    """Sequential execution backend for development and debugging.
    
    Simple, synchronous execution perfect for:
    - Development and debugging
    - Unit testing  
    - Flow validation before Dora conversion
    - Prototyping new Flow patterns
    """
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.execution_count = 0
        self.total_time = 0.0
    
    def execute_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Execute flow sequentially."""
        flow_name = flow.__class__.__name__
        print(f"🚀 [bold blue]Starting:[/bold blue] {flow_name}")
        
        start_time = time.time()
        
        try:
            # Use timed execution if flow supports it and timing is enabled
            if hasattr(flow, 'run_timed') and self.config.timing_enabled:
                timer = ExecutionTimer(name=f"sequential_{flow.__class__.__name__}")
                result = flow.run_timed(input_data, timer)
            else:
                result = flow(input_data)
            
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_time += execution_time
            
            time_str = f"{execution_time:.3f}s" if execution_time >= 0.001 else f"{execution_time*1000:.1f}ms"
            print(f"✅ [bold green]Completed:[/bold green] {flow_name} in {time_str}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.execution_count += 1
            self.total_time += execution_time
            
            time_str = f"{execution_time:.3f}s" if execution_time >= 0.001 else f"{execution_time*1000:.1f}ms"
            print(f"❌ [bold red]Failed:[/bold red] {flow_name} after {time_str}")
            print(f"   [dim]Error: {str(e)}[/dim]")
            raise
    
    def execute_pipeline(self, pipeline: 'Pipeline', input_data: Any) -> Any:
        """Execute pipeline of flows sequentially."""
        # Extract flows from pipeline composition tree and execute sequentially
        flows = pipeline.get_flows()  # This method needs to exist in Pipeline class
        result = input_data
        
        for flow in flows:
            result = self.execute_flow(flow, result)
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_time = self.total_time / max(self.execution_count, 1)
        
        return {
            "backend": "sequential",
            "total_executions": self.execution_count,
            "total_time": self.total_time,
            "average_time_per_execution": avg_time,
            "overhead_percentage": 0.0  # Sequential has minimal overhead
        }


class ThreadingExecutor(FlowExecutor[I, O]):
    """Threading execution backend for time-aware execution testing.
    
    Uses Python threading to simulate multi-rate coordination that will
    be implemented efficiently in Dora. Helps validate timing patterns
    before Dora deployment.
    """
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.coordinator: Optional[FrequencyCoordinator] = None
        self.execution_metrics = {}
        self._lock = threading.Lock()
    
    def execute_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Execute flow with threading support."""
        flow_name = flow.__class__.__name__
        
        # Initialize metrics for this flow
        if flow_name not in self.execution_metrics:
            with self._lock:
                self.execution_metrics[flow_name] = ExecutionTimer(name=flow_name)
        
        timer = self.execution_metrics[flow_name]
        
        with timer.execution_context():
            # Use timed execution if flow supports it
            if hasattr(flow, 'run_timed'):
                return flow.run_timed(input_data, timer)
            else:
                return flow(input_data)
    
    def execute_pipeline(self, pipeline: 'Pipeline', input_data: Any) -> Any:
        """Execute pipeline with timing coordination."""
        flows = pipeline.get_flows()
        result = input_data
        
        for flow in flows:
            result = self.execute_flow(flow, result)
        
        return result
    
    def execute_with_frequency_coordination(self, frequency_configs: Dict[str, float], 
                                          flow_assignments: Dict[str, List['Flow']], 
                                          input_data: Any) -> Dict[str, Any]:
        """Execute flows with frequency coordination - models Dora timer behavior."""
        self.coordinator = FrequencyCoordinator(frequency_configs)
        results = {}
        
        # Execute flows at each frequency
        for freq_name, flows in flow_assignments.items():
            with self.coordinator.frequency_context(freq_name) as timer:
                freq_result = input_data
                for flow in flows:
                    freq_result = self.execute_flow(flow, freq_result)
                results[freq_name] = freq_result
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "backend": "threading",
            "flow_metrics": {},
            "coordination_metrics": {}
        }
        
        # Collect individual flow metrics
        with self._lock:
            for flow_name, timer in self.execution_metrics.items():
                metrics["flow_metrics"][flow_name] = {
                    "executions": timer.metrics.total_executions,
                    "success_rate": timer.metrics.successful_executions / max(timer.metrics.total_executions, 1),
                    "average_duration": timer.metrics.average_duration,
                    "violations": len(timer.metrics.violations)
                }
        
        # Collect coordination metrics
        if self.coordinator:
            for freq_name, timer in self.coordinator.timers.items():
                metrics["coordination_metrics"][freq_name] = {
                    "target_hz": self.coordinator.frequency_configs[freq_name],
                    "executions": timer.metrics.total_executions,
                    "average_duration": timer.metrics.average_duration,
                    "violations": len(timer.metrics.violations)
                }
        
        return metrics



class DoraExecutor(FlowExecutor[I, O]):
    """
    Dora execution backend - THE true FRP executor with full temporal coordination.
    
    This IS the "true FRP executor" - Dora provides all FRP functionality natively:
    - Rate coordination via dora/timer/millis/X
    - Event streams via Dora's event-driven dataflow  
    - Behavior composition via node connections
    - Temporal coordination via Dora's runtime
    - Feedback loops via dataflow connections
    
    Conversion Strategy (based on temp/dora/examples/python-operator-dataflow):
    - @flow(rate="30hz") -> dora/timer/millis/33 timer input
    - Flow.run(input) -> Operator.on_event() method 
    - camera >> detector -> Sequential nodes with input/output connections
    - Flow[I,O] compositions -> Complete Dora dataflow YAML
    - FRP annotations -> Native Dora timing and coordination
    """
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.generated_yamls = []
        self.compiled_operators = {}
        self.frp_coordinator = FRPCoordinator()  # Internal FRP engine
    
    def execute_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Execute flow via Dora (converts Flow to operators first)."""
        # For now, delegate to sequential execution while we build conversion
        sequential = SequentialExecutor(self.config)
        return sequential.execute_flow(flow, input_data)
    
    def execute_pipeline(self, pipeline: 'Pipeline', input_data: Any) -> Any:
        """Execute pipeline via Dora dataflow."""
        # Generate YAML and run through Dora
        yaml_path = self.compile_pipeline_to_dora_yaml(pipeline)
        
        # Try Python-native Dora execution first
        try:
            from dora import start_runtime
            return self._execute_with_python_dora(yaml_path, input_data)
        except ImportError:
            print("⚠️  dora-rs not installed, falling back to sequential execution")
            sequential = SequentialExecutor(self.config)
            return sequential.execute_pipeline(pipeline, input_data)
    
    def compile_pipeline_to_dora_yaml(self, pipeline: 'Pipeline', output_path: Optional[str] = None) -> str:
        """Convert Pipeline to Dora YAML configuration with working operators."""
        # Use new translation layer for working operators
        try:
            from ..integrations.dora.translation import PipelineTranslator
            
            # For examples directory, use local output for easy validation
            # For production use, use temp directory to avoid cluttering workspace
            if output_path:
                output_dir = output_path
            else:
                cwd = os.getcwd()
                if "examples" in cwd:
                    output_dir = "generated_dora"  # Local for examples
                else:
                    output_dir = os.path.join(tempfile.gettempdir(), "generated_dora")  # Temp for production
            translator = PipelineTranslator()
            
            config = translator.translate_pipeline(pipeline, output_dir)
            
            print(f"✅ Generated working Dora operators in: {output_dir}")
            return config.yaml_path
            
        except ImportError as e:
            print(f"⚠️  Translation layer not available: {e}, falling back to template generation")
            # Fallback to old method
            flows = pipeline.get_flows()
            return self.compile_flows_to_dora_yaml(flows, output_path)
    
    def compile_flows_to_dora_yaml(self, flows: List['Flow'], output_path: Optional[str] = None) -> str:
        """Convert Flow pipeline to Dora YAML configuration.
        
        Based on the pattern from temp/dora/examples/python-operator-dataflow/dataflow.yml:
        - Each Flow becomes a Dora node with python operator
        - Sequential flows are chained via input/output connections
        - Parallel flows (fanout) share inputs
        - TimedFlows get timer inputs
        """
        if output_path is None:
            # Default to temp directory for generated files
            import os
            os.makedirs("temp/generated_dora", exist_ok=True)
            output_path = self.config.dora_config_path or "temp/generated_dora/dataflow.yml"
        
        yaml_content = self._generate_dora_yaml_from_flows(flows)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
        
        self.generated_yamls.append(output_path)
        return output_path
    
    def _generate_dora_yaml_from_flows(self, flows: List['Flow']) -> str:
        """
        Generate complete FRP-aware Dora YAML from Flow list.
        
        This implements the core FRP → Dora conversion:
        - @flow(rate="30hz") → dora/timer/millis/33 timer inputs
        - Flow.run() methods → Operator.on_event() implementations  
        - Pipeline composition → Node connections with proper data flow
        - FRP rate coordination → Native Dora timing via multiple timers
        - Event streams → Dora's event-driven execution model
        - Feedback loops → Dataflow connections between nodes
        """
        # Register all flows with FRP coordinator for rate analysis
        for i, flow in enumerate(flows):
            flow_id = getattr(flow, 'flow_id', None) or f"flow_{i}_{flow.__class__.__name__}"
            self.frp_coordinator.register_flow(flow, flow_id)
        
        yaml_content = "# Generated Dora dataflow with FRP temporal coordination\n"
        yaml_content += "# Flows → Operators with native Dora timing\n\n"
        yaml_content += "nodes:\n"
        
        for i, flow in enumerate(flows):
            flow_name = flow.__class__.__name__.lower()
            operator_file = f"{flow_name}_op.py"
            flow_id = getattr(flow, 'flow_id', None) or f"flow_{i}"
            
            # Extract FRP annotations
            rate_annotation = self._extract_flow_rate(flow)
            trigger_annotation = self._extract_flow_trigger(flow)
            feedback_config = self._extract_feedback_config(flow)
            
            # Generate node with full FRP support
            yaml_content += f"  - id: {flow_name}\n"
            yaml_content += f"    operator:\n"
            yaml_content += f"      python: {operator_file}\n"
            yaml_content += f"      inputs:\n"
            
            # Primary input - timer or data connection
            if i == 0:  # First flow - needs timer input for rate coordination
                timer_config = self._convert_rate_to_dora_timer(rate_annotation or "50ms")
                yaml_content += f"        tick: {timer_config}  # FRP rate: {rate_annotation or 'default'}\n"
            else:
                # Data input from previous flow
                prev_flow_name = flows[i-1].__class__.__name__.lower()
                yaml_content += f"        input: {prev_flow_name}/output  # Sequential flow composition\n"
            
            # Event trigger input (FRP event streams)
            if trigger_annotation:
                yaml_content += f"        trigger: event/{trigger_annotation}  # FRP event trigger\n"
            
            # Feedback input (FRP feedback loops)
            if feedback_config.get('feedback_from'):
                yaml_content += f"        feedback: {feedback_config['feedback_from']}/feedback  # FRP feedback loop\n"
            
            # Outputs
            yaml_content += f"      outputs:\n"
            yaml_content += f"        - output  # Primary data output\n"
            
            # Feedback output
            if feedback_config.get('feedback_to'):
                yaml_content += f"        - feedback  # FRP feedback to {feedback_config['feedback_to']}\n"
            
            # Event output for reactive patterns
            if self._flow_generates_events(flow):
                yaml_content += f"        - events  # FRP event stream output\n"
            
            yaml_content += "\n"
            
            # Generate operator file for this flow
            self._generate_frp_aware_operator(flow, operator_file, rate_annotation)
        
        # Add FRP coordination comment
        yaml_content += "\n# FRP Features implemented via Dora:\n"
        yaml_content += "# - Rate coordination: Multiple dora/timer/millis/X inputs\n"
        yaml_content += "# - Event streams: Dora's native event-driven execution\n"
        yaml_content += "# - Behaviors: Node connections with temporal data flow\n"
        yaml_content += "# - Feedback loops: Dataflow connections between nodes\n"
        
        return yaml_content
    
    def _get_flow_input_config(self, flow: 'Flow', index: int, flows: List['Flow']) -> Dict[str, str]:
        """Get input configuration for a flow with FRP support."""
        if index == 0:
            # First flow: check for custom rate or use default timer
            if hasattr(flow, 'frp_config') and flow.frp_config.rate:
                timer_config = self.frp_coordinator.get_dora_timer_config(flow.frp_config.rate)
                return {"input_source": timer_config, "input_name": "tick"}
            else:
                return {"input_source": "dora/timer/millis/50", "input_name": "tick"}
        else:
            # Subsequent flows: get output from previous flow
            prev_flow = flows[index-1].__class__.__name__.lower()
            return {"input_source": f"{prev_flow}/output", "input_name": "input"}
    
    def _extract_flow_rate(self, flow: 'Flow') -> Optional[str]:
        """Extract rate annotation from @flow decorator."""
        if hasattr(flow, '_rate'):
            return flow._rate
        return None
    
    def _extract_flow_trigger(self, flow: 'Flow') -> Optional[str]:
        """Extract trigger annotation from @flow decorator."""
        if hasattr(flow, '_trigger'):
            return flow._trigger
        return None
    
    def _extract_feedback_config(self, flow: 'Flow') -> Dict[str, Optional[str]]:
        """Extract feedback configuration from @flow decorator."""
        return {
            'feedback_from': getattr(flow, '_feedback_from', None),
            'feedback_to': getattr(flow, '_feedback_to', None)
        }
    
    def _flow_generates_events(self, flow: 'Flow') -> bool:
        """Check if flow generates events (returns Optional[T])."""
        # Check if flow's return type annotation suggests events
        if hasattr(flow, 'run'):
            import inspect
            sig = inspect.signature(flow.run)
            return_annotation = sig.return_annotation
            # Check if return type is Optional[T] or similar
            return str(return_annotation).startswith('typing.Union') or 'Optional' in str(return_annotation)
        return False
    
    def _convert_rate_to_dora_timer(self, rate: str) -> str:
        """
        Convert rate annotation to Dora timer format.
        
        This is the core FRP rate → Dora conversion:
        - "30hz" → "dora/timer/millis/33"
        - "50ms" → "dora/timer/millis/50"  
        - "1s" → "dora/timer/millis/1000"
        """
        if rate.endswith('hz'):
            hz = float(rate[:-2])
            ms = int(1000 / hz)
            return f"dora/timer/millis/{ms}"
        elif rate.endswith('ms'):
            ms = int(rate[:-2])
            return f"dora/timer/millis/{ms}"
        elif rate.endswith('s'):
            s = float(rate[:-1])
            ms = int(s * 1000)
            return f"dora/timer/millis/{ms}"
        else:
            # Default to 50ms (20Hz)
            return "dora/timer/millis/50"
    
    def _generate_frp_aware_operator(self, flow: 'Flow', filename: str, rate_annotation: Optional[str]):
        """
        Generate FRP-aware Dora operator from Flow.
        
        This is the core Flow.run() → Operator.on_event() conversion:
        - Flow.run(input) → Operator.on_event(dora_event, send_output)
        - @flow annotations → Timing-aware operator behavior
        - Type conversions → PyArrow ↔ Flow data types
        """
        if filename in self.compiled_operators:
            return  # Already generated
        
        # Ensure filename includes temp directory path
        if not filename.startswith("temp/"):
            import os
            os.makedirs("temp/generated_dora", exist_ok=True)
            filename = f"temp/generated_dora/{filename}"
        
        flow_class = flow.__class__.__name__
        rate_comment = f" # FRP rate: {rate_annotation}" if rate_annotation else ""
        
        operator_code = f'''"""
FRP-Aware Dora Operator for {flow_class}
Generated from Retriever Flow with FRP temporal coordination.

Rate annotation: {rate_annotation or 'default'}
Dora timer: {self._convert_rate_to_dora_timer(rate_annotation or "50ms")}
"""

import numpy as np
import pyarrow as pa
from dora import DoraStatus
import time

# TODO: Import the original flow class
# from retriever.flows import {flow_class}

class Operator:
    """
    FRP-aware Dora operator wrapping {flow_class}.
    
    This implements the FRP → Dora conversion:
    - Flow.run(input) becomes on_event(dora_event, send_output)
    - FRP rate coordination handled by Dora timer inputs{rate_comment}
    - Event streams handled by Dora's event-driven execution
    """
    
    def __init__(self):
        """Initialize the wrapped flow."""
        # self.flow = {flow_class}()
        self.execution_count = 0
        self.last_execution_time = time.time()
        
    def on_event(self, dora_event, send_output) -> DoraStatus:
        """
        Handle Dora events by executing the wrapped Flow.
        
        This is the core FRP execution: Dora events → Flow.run() → outputs
        """
        try:
            if dora_event["type"] == "INPUT":
                # FRP rate coordination - Dora handles timing
                current_time = time.time()
                
                # Convert Dora input to Flow input format
                input_id = dora_event["id"]
                if input_id == "tick":
                    # Timer input - sensor flow with no data input
                    flow_input = None
                elif input_id == "input":
                    # Data input from previous flow
                    flow_input = self._convert_dora_input(dora_event["value"])
                elif input_id == "trigger":
                    # Event trigger input (FRP event streams)
                    flow_input = self._convert_event_input(dora_event["value"])
                elif input_id == "feedback":
                    # Feedback input (FRP feedback loops)
                    flow_input = self._convert_feedback_input(dora_event["value"])
                else:
                    flow_input = self._convert_dora_input(dora_event["value"])
                
                # Execute the Flow - this is where user's simple run() method is called
                # result = self.flow.run(flow_input)
                result = flow_input  # Placeholder for demo
                
                # Convert result to Dora format and send
                if result is not None:
                    output_array = self._convert_to_arrow(result)
                    send_output("output", output_array, dora_event["metadata"])
                    
                    # Handle event outputs (FRP event streams)
                    if self._is_event_result(result):
                        event_array = self._convert_event_to_arrow(result)
                        send_output("events", event_array, dora_event["metadata"])
                
                # Update execution metrics
                self.execution_count += 1
                self.last_execution_time = current_time
                
                # FRP timing validation
                if "{rate_annotation}":
                    expected_period = self._get_expected_period("{rate_annotation}")
                    actual_period = current_time - self.last_execution_time
                    if actual_period > expected_period * 1.5:  # 50% tolerance
                        print(f"⚠️  FRP timing violation: {{actual_period:.3f}}s > {{expected_period:.3f}}s")
                
            return DoraStatus.CONTINUE
            
        except Exception as e:
            print(f"❌ Flow execution error in {flow_class}: {{e}}")
            return DoraStatus.CONTINUE  # Continue execution despite errors
    
    def _convert_dora_input(self, arrow_value):
        """Convert Dora PyArrow input to Flow input format."""
        try:
            # Handle different Arrow types
            if hasattr(arrow_value, 'to_numpy'):
                return arrow_value.to_numpy()
            elif hasattr(arrow_value, 'to_pylist'):
                return arrow_value.to_pylist()
            else:
                return arrow_value
        except Exception:
            return arrow_value
    
    def _convert_to_arrow(self, result):
        """Convert Flow result to PyArrow format for Dora."""
        try:
            if isinstance(result, np.ndarray):
                return pa.array(result.ravel())
            elif isinstance(result, (list, tuple)):
                return pa.array(result)
            else:
                # Handle custom types by converting to string representation
                return pa.array([str(result)])
        except Exception:
            return pa.array([str(result)])
    
    def _convert_event_input(self, event_value):
        """Convert event trigger input to Flow format."""
        return self._convert_dora_input(event_value)
    
    def _convert_feedback_input(self, feedback_value):
        """Convert feedback input to Flow format."""
        return self._convert_dora_input(feedback_value)
    
    def _is_event_result(self, result):
        """Check if result should be sent as event."""
        # Simple heuristic - customize based on Flow types
        return result is not None and hasattr(result, '__iter__')
    
    def _convert_event_to_arrow(self, event_result):
        """Convert event result to Arrow format."""
        return self._convert_to_arrow(event_result)
    
    def _get_expected_period(self, rate_annotation):
        """Get expected period from rate annotation."""
        if rate_annotation.endswith('hz'):
            hz = float(rate_annotation[:-2])
            return 1.0 / hz
        elif rate_annotation.endswith('ms'):
            ms = float(rate_annotation[:-2])
            return ms / 1000.0
        elif rate_annotation.endswith('s'):
            return float(rate_annotation[:-1])
        else:
            return 0.05  # Default 50ms
'''
        
        # Write operator file
        with open(filename, 'w') as f:
            f.write(operator_code)
        
        self.compiled_operators[filename] = operator_code
        
        print(f"✅ Generated FRP-aware operator: {filename}")
        if rate_annotation:
            timer_config = self._convert_rate_to_dora_timer(rate_annotation)
            print(f"   Rate: {rate_annotation} → {timer_config}")
    
    def _generate_operator_file(self, flow: 'Flow', filename: str):
        """Legacy operator generation - use _generate_frp_aware_operator instead."""
        # Delegate to FRP-aware version
        rate = self._extract_flow_rate(flow)
        self._generate_frp_aware_operator(flow, filename, rate)
        
        # Write operator file
        with open(filename, 'w') as f:
            f.write(operator_code)
        
        self.compiled_operators[filename] = operator_code
    
    def start_dora_subprocess(self, yaml_path: str):
        """Start Dora pipeline as subprocess and return process handle."""
        original_dir = os.getcwd()
        dora_dir = os.path.dirname(yaml_path)
        yaml_name = os.path.basename(yaml_path)
        
        os.chdir(dora_dir)
        
        try:
            process = subprocess.Popen(
                ["dora", "start", yaml_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(2)  # Let Dora initialize
            return process, original_dir
            
        except Exception as e:
            os.chdir(original_dir)
            raise e
    
    def stop_dora_subprocess(self, process, original_dir: str):
        """Stop Dora subprocess and restore directory."""
        if process:
            process.terminate()
            process.wait()
        os.chdir(original_dir)
    
    def _execute_with_python_dora(self, yaml_path: str, input_data: Any) -> Any:
        """Execute pipeline using Dora's Python runtime."""
        process, original_dir = self.start_dora_subprocess(yaml_path)
        
        try:
            # Run for 10 seconds showing live status
            for i in range(10):
                time.sleep(1.0)
                if process.poll() is not None:
                    # Process died
                    stdout, stderr = process.communicate()
                    print(f"❌ Dora process died with code: {process.returncode}")
                    if stderr:
                        print(f"ERROR: {stderr}")
                    raise RuntimeError(f"Dora execution failed: {stderr}")
                else:
                    print(f"⚡ Second {i+1}: FRP pipeline running")
            
            return "FRP pipeline executed for 10 seconds"
            
        except FileNotFoundError as e:
            if "dora" in str(e):
                raise RuntimeError("Dora CLI not found. Please install dora-rs: 'cargo install dora-cli'")
            raise
            
        except Exception as e:
            print(f"⚠️  Dora execution failed: {e}")
            raise
        
        finally:
            self.stop_dora_subprocess(process, original_dir)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get Dora execution metrics."""
        return {
            "backend": "dora",
            "generated_yamls": len(self.generated_yamls),
            "compiled_operators": len(self.compiled_operators),
            "status": "python_execution_ready"
        }


class ExecutionEngine:
    """Main execution engine that manages backend selection and Flow execution."""
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self._executor = None
        self._running_pipeline = None
        self._dora_process = None
        self._dora_original_dir = None
        self._initialize_executor()
    
    def _initialize_executor(self):
        """Initialize the appropriate executor based on configuration."""
        if self.config.backend == ExecutionBackend.SEQUENTIAL:
            self._executor = SequentialExecutor(self.config)
        elif self.config.backend == ExecutionBackend.THREADING:
            self._executor = ThreadingExecutor(self.config)
        elif self.config.backend == ExecutionBackend.DORA:
            self._executor = DoraExecutor(self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def run_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Run a single flow using the configured backend."""
        return self._executor.execute_flow(flow, input_data)
    
    def execute_flow(self, flow: 'Flow[I, O]', input_data: I) -> O:
        """Execute a single flow using the configured backend (alias for run_flow)."""
        return self.run_flow(flow, input_data)
    
    def run_pipeline(self, pipeline: 'Pipeline[I, O]', input_data: I) -> O:
        """Run a Pipeline using the configured backend."""
        return self._executor.execute_pipeline(pipeline, input_data)
    
    def execute_pipeline(self, pipeline: 'Pipeline', input_data: Any) -> Any:
        """Execute a Pipeline using the configured backend (alias for run_pipeline)."""
        return self.run_pipeline(pipeline, input_data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics from the current backend."""
        return self._executor.get_performance_metrics()
    
    def switch_backend(self, new_backend: ExecutionBackend):
        """Switch to a different execution backend."""
        self.config.backend = new_backend
        self._initialize_executor()
    
    def compile_to_dora(self, pipeline: 'Pipeline', output_path: str = None) -> str:
        """Compile Pipeline to Dora YAML with working operators (switches to Dora backend if needed)."""
        if not isinstance(self._executor, DoraExecutor):
            self.switch_backend(ExecutionBackend.DORA)
        
        # Use enhanced compilation with working operators
        yaml_path = self._executor.compile_pipeline_to_dora_yaml(pipeline, output_path)
        
        # Store the generated directory for process execution
        self._dora_output_dir = os.path.dirname(yaml_path)
        
        return yaml_path
    
    # High-level pipeline execution methods that wrap all complexity
    
    def start_pipeline(self, pipeline: 'Pipeline', background: bool = True) -> bool:
        """
        Start a pipeline in the background (for Dora) or run once (for other backends).
        
        This wraps all the complexity of Dora subprocess management including daemon startup.
        Returns True if started successfully, False otherwise.
        """
        try:
            if isinstance(self._executor, DoraExecutor):
                # Dora: ensure daemon is running, compile and start subprocess
                if not self._ensure_dora_daemon():
                    print("❌ Failed to start Dora daemon")
                    return False
                    
                yaml_path = self.compile_to_dora(pipeline)
                self._dora_process, self._dora_original_dir = self._executor.start_dora_subprocess(yaml_path)
                self._running_pipeline = pipeline
                return True
            else:
                # Other backends: immediate execution
                # Note: For non-Dora backends, this would need input data
                print("⚠️  Non-Dora backends require input data - use run_pipeline() instead")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start pipeline: {e}")
            return False
    
    def stop_pipeline(self) -> bool:
        """
        Stop the running pipeline. Wraps all cleanup complexity.
        Returns True if stopped successfully, False otherwise.
        """
        try:
            if self._dora_process and isinstance(self._executor, DoraExecutor):
                self._executor.stop_dora_subprocess(self._dora_process, self._dora_original_dir)
                self._dora_process = None
                self._dora_original_dir = None
                self._running_pipeline = None
                return True
            else:
                print("⚠️  No Dora pipeline running to stop")
                return False
                
        except Exception as e:
            print(f"❌ Failed to stop pipeline: {e}")
            return False
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of running pipeline. Wraps process monitoring complexity.
        Returns status dict with 'running', 'process_id', 'backend', etc.
        """
        status = {
            "backend": self.config.backend.value,
            "has_running_pipeline": self._running_pipeline is not None,
            "pipeline_type": type(self._running_pipeline).__name__ if self._running_pipeline else None
        }
        
        if isinstance(self._executor, DoraExecutor) and self._dora_process:
            poll_result = self._dora_process.poll()
            status.update({
                "dora_running": poll_result is None,
                "dora_exit_code": poll_result,
                "dora_pid": self._dora_process.pid if poll_result is None else None
            })
        
        return status
    
    def check_pipeline_health(self) -> bool:
        """
        Check if pipeline is running healthily. Wraps error detection.
        Returns True if healthy, False if failed/stopped.
        """
        if not self._running_pipeline:
            return False
            
        if isinstance(self._executor, DoraExecutor) and self._dora_process:
            return self._dora_process.poll() is None
        
        return True
    
    def get_pipeline_logs(self) -> Dict[str, str]:
        """
        Get logs from running pipeline. Wraps log collection complexity.
        Returns dict with 'stdout' and 'stderr' keys.
        """
        logs = {"stdout": "", "stderr": ""}
        
        if isinstance(self._executor, DoraExecutor) and self._dora_process:
            try:
                # Non-blocking read if process is still running
                if self._dora_process.poll() is None:
                    # Process still running - can't get full output yet
                    logs["stdout"] = "Process still running..."
                    logs["stderr"] = "Process still running..."
                else:
                    # Process finished - get full output
                    stdout, stderr = self._dora_process.communicate()
                    logs["stdout"] = stdout or ""
                    logs["stderr"] = stderr or ""
            except Exception as e:
                logs["stderr"] = f"Error reading logs: {e}"
        
        return logs
    
    def _ensure_dora_daemon(self) -> bool:
        """
        Ensure Dora daemon is running. Start it if needed.
        Returns True if daemon is running, False otherwise.
        """
        try:
            # First, try to clean up any stuck daemons
            self._cleanup_stuck_dora_processes()
            
            # Check if daemon is already running properly
            result = subprocess.run(
                ["dora", "daemon", "--status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Dora daemon is already running")
                return True
            else:
                print("🚀 Starting Dora daemon...")
                # Start daemon in background
                daemon_process = subprocess.Popen(
                    ["dora", "daemon"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Give daemon time to start
                time.sleep(3)
                
                # Check if daemon started successfully
                if daemon_process.poll() is None:
                    print("✅ Dora daemon started successfully")
                    return True
                else:
                    stdout, stderr = daemon_process.communicate()
                    if "already a connected daemon" in stderr:
                        print("✅ Dora daemon already running (detected via error)")
                        return True
                    else:
                        print(f"❌ Dora daemon failed to start: {stderr}")
                        return False
                    
        except FileNotFoundError:
            print("❌ Dora CLI not found. Install with: cargo install dora-cli")
            return False
        except subprocess.TimeoutExpired:
            print("❌ Dora daemon status check timed out")
            return False
        except Exception as e:
            print(f"❌ Error checking/starting Dora daemon: {e}")
            return False
    
    def _cleanup_stuck_dora_processes(self):
        """Clean up any stuck Dora processes."""
        try:
            # Try to stop any existing daemon gracefully
            subprocess.run(
                ["dora", "daemon", "--stop"],
                capture_output=True,
                text=True,
                timeout=5
            )
            time.sleep(1)
        except Exception:
            pass  # Ignore errors, this is just cleanup


# Global execution engine instance
_global_engine: Optional[ExecutionEngine] = None


def init(backend: str = "sequential", timing_enabled: bool = False, **kwargs):
    """Initialize the global execution engine.
    
    Args:
        backend: Backend name ("sequential", "threading", "dora")
        timing_enabled: Whether to enable timing for flows that support it
        **kwargs: Additional configuration options
    """
    global _global_engine
    
    backend_enum = ExecutionBackend(backend)
    config = ExecutionConfig(
        backend=backend_enum,
        timing_enabled=timing_enabled,
        **kwargs
    )
    
    _global_engine = ExecutionEngine(config)


def get_engine() -> ExecutionEngine:
    """Get the global execution engine, initializing with defaults if needed."""
    global _global_engine
    
    if _global_engine is None:
        init()  # Initialize with defaults
    
    return _global_engine


# Convenience functions for backend selection
def use_sequential_backend():
    """Switch to sequential execution backend."""
    init(backend="sequential")


def use_threading_backend(timing_enabled: bool = True):
    """Switch to threading execution backend."""
    init(backend="threading", timing_enabled=timing_enabled)


def use_dora_backend(dora_config_path: str = None):
    """Switch to Dora execution backend."""
    init(backend="dora", dora_config_path=dora_config_path)


# Auto-detection of appropriate backend
def auto_select_backend() -> ExecutionBackend:
    """Automatically select the most appropriate backend based on environment."""
    
    # Check for Dora availability
    if os.environ.get("DORA_AVAILABLE") == "true":
        return ExecutionBackend.DORA
    
    # Check if we're in a development environment
    if os.environ.get("RETRIEVER_ENV") == "development":
        return ExecutionBackend.SEQUENTIAL
    
    # Default to threading for time-aware execution testing
    return ExecutionBackend.THREADING


def init_auto():
    """Initialize with automatically selected backend."""
    backend = auto_select_backend()
    init(backend=backend.value, timing_enabled=(backend != ExecutionBackend.SEQUENTIAL))


# Dora conversion utilities
def flows_to_dora_yaml(flows: List['Flow'], output_path: str) -> str:
    """Convert a list of Flow objects to Dora YAML configuration."""
    engine = get_engine()
    return engine.compile_to_dora(flows, output_path)


def flow_to_dora_operator(flow: 'Flow', operator_filename: str):
    """Convert a single Flow to a Dora operator Python file."""
    dora_executor = DoraExecutor()
    return dora_executor._generate_operator_file(flow, operator_filename)
