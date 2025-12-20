"""
FRP Engine for Retriever Flow v5.0

This module provides the compilation layer that converts simple Flow[I,O] definitions
into sophisticated FRP behaviors with temporal coordination. This is the "hidden
complexity" that provides sophisticated timing while users write simple code.

Design Principles:
- Compile Flow.run() methods to time-aware Behavior[O]
- Handle rate coordination automatically  
- Support event-driven behaviors and feedback loops
- Provide execution backends for different deployment scenarios
"""

import time
import inspect
from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, Dict, List, Optional, Any, Callable, Union, Type, Tuple
)
from dataclasses import dataclass
from enum import Enum

from .frp import Behavior, EventStream, RateCoordinator, EventManager, TimestampedBuffer
from .temporal import ExecutionTimer
from .flow import Flow
from .types import Pipeline
from .frp_config import FRPConfig

class CompositionType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

I = TypeVar('I')
O = TypeVar('O')
T = TypeVar('T')


"""
Note: FRPConfig is centralized in retriever.core.frp_config.
This module imports and uses that unified definition.
"""


class FlowCompilationContext:
    """Context information for compiling flows to behaviors."""
    
    def __init__(self):
        self.input_providers: Dict[str, Callable[[float], Any]] = {}
        self.output_consumers: Dict[str, Callable[[Any, float], None]] = {}
        self.flow_dependencies: Dict[str, List[str]] = {}
    
    def add_input_provider(self, flow_id: str, provider: Callable[[float], Any]):
        """Add input provider for a flow."""
        self.input_providers[flow_id] = provider
    
    def add_output_consumer(self, flow_id: str, consumer: Callable[[Any, float], None]):
        """Add output consumer for a flow."""
        self.output_consumers[flow_id] = consumer
    
    def get_input_at_time(self, flow_id: str, t: float) -> Any:
        """Get input for flow at given time."""
        if flow_id in self.input_providers:
            return self.input_providers[flow_id](t)
        return None


class FRPEngine:
    """Compiles Flow[I,O] definitions to time-aware FRP behaviors.
    
    This is the core "compilation" engine that transforms simple Python Flow
    definitions into sophisticated time-aware behaviors with rate coordination,
    event handling, and feedback loops.
    """
    
    def __init__(self):
        self.rate_coordinator = RateCoordinator()
        self.event_manager = EventManager()
        self.compiled_behaviors: Dict[str, Behavior] = {}
        self.flow_configs: Dict[str, FRPConfig] = {}
    
    def register_flow(self, flow: Flow[I, O], flow_id: Optional[str] = None) -> str:
        """Register a flow with the engine and extract its configuration."""
        if flow_id is None:
            flow_id = f"{flow.__class__.__name__}_{id(flow)}"
        
        # Extract FRP configuration from flow
        config = self._extract_frp_config(flow)
        self.flow_configs[flow_id] = config
        
        # Register with rate coordinator if rate specified
        if config.rate is not None:
            hz = config.rate_hz()
            if hz is not None:
                self.rate_coordinator.register_flow(flow_id, hz)
        
        return flow_id
    
    def create_behavior_from_flow(self, 
                                 flow: Flow[I, O], 
                                 context: FlowCompilationContext,
                                 flow_id: Optional[str] = None) -> Behavior[O]:
        """Convert a Flow to FRP Behavior with temporal coordination."""
        
        if flow_id is None:
            flow_id = self.register_flow(flow)
        
        config = self.flow_configs.get(flow_id, FRPConfig())
        
        def sample_function(t: float) -> O:
            # Get input data at time t
            input_data = context.get_input_at_time(flow_id, t)
            
            # Execute flow - this is where the simple user code runs
            if hasattr(flow, 'run_timed'):
                timer = ExecutionTimer(target_hz=config.rate_hz(), name=flow_id)
                timer.start()
                result = flow.run_timed(input_data, timer)
                
                # Log execution metrics
                duration = timer.elapsed()
                timer.log_execution(flow_id, duration, success=True)
            else:
                # Fallback to simple run method
                result = flow.run(input_data)
            
            # Update rate coordinator
            if config.rate is not None:
                self.rate_coordinator.update_value(flow_id, result, t)
            
            # Handle output consumers
            if flow_id in context.output_consumers:
                context.output_consumers[flow_id](result, t)
            
            return result
        
        behavior = Behavior(sample_function)
        
        # Apply rate limiting if specified
        if config.rate is not None:
            hz = config.rate_hz()
            if hz is not None:
                behavior = behavior.sample_at_rate(hz)
        
        # Store compiled behavior
        self.compiled_behaviors[flow_id] = behavior
        
        return behavior
    
    def create_event_stream_from_flow(self, 
                                     flow: Flow[I, Optional[T]], 
                                     context: FlowCompilationContext,
                                     flow_id: Optional[str] = None) -> EventStream[T]:
        """Convert Flow that outputs Optional[T] to EventStream[T].
        
        This is used for flows that generate events (like obstacle detection)
        rather than continuous data streams.
        """
        
        if flow_id is None:
            flow_id = self.register_flow(flow)
        
        event_buffer = TimestampedBuffer(max_size=1000)
        
        def event_generator() -> List[Tuple[float, T]]:
            current_time = time.time()
            input_data = context.get_input_at_time(flow_id, current_time)
            
            # Execute flow
            result = flow.run(input_data)
            
            # Generate event if result is not None
            if result is not None:
                event_buffer.add(current_time, result)
            
            # Return all events in buffer
            return event_buffer.get_all_in_window(current_time, window=60.0)
        
        event_stream = EventStream(event_generator)
        
        # Register with event manager
        self.event_manager.register_event_stream(flow_id, event_stream)
        
        return event_stream
    
    def compile_pipeline(self, pipeline: Pipeline[I, O]) -> Behavior[O]:
        """Compile entire pipeline to coordinated behavior.
        
        This handles the composition of multiple flows with different rates
        and coordination requirements.
        """
        
        # Get flows from pipeline
        flows = pipeline.get_flows()
        # If a single composed Flow was wrapped into Pipeline, try to decompose its node
        if len(flows) == 1 and hasattr(flows[0], '_node'):
            try:
                flows = self._extract_flows_from_node(flows[0]._node)
            except Exception:
                pass
        context = FlowCompilationContext()
        behaviors = {}
        
        # Compile each flow to behavior
        for i, flow in enumerate(flows):
            flow_id = f"pipeline_flow_{i}_{flow.__class__.__name__}"
            
            # Set up input provider for this flow
            if i == 0:
                # First flow gets external input
                context.add_input_provider(flow_id, lambda t: None)  # Will be provided by caller
            else:
                # Subsequent flows get input from previous flow
                prev_flow_id = f"pipeline_flow_{i-1}_{flows[i-1].__class__.__name__}"
                def make_input_provider(prev_id):
                    def provider(t):
                        if prev_id in self.compiled_behaviors:
                            return self.compiled_behaviors[prev_id].at_time(t)
                        return None
                    return provider
                context.add_input_provider(flow_id, make_input_provider(prev_flow_id))
            
            # Compile flow to behavior
            behavior = self.create_behavior_from_flow(flow, context, flow_id)
            behaviors[flow_id] = behavior
        
        # Create final composed behavior based on pipeline type
        if pipeline.composition_type == CompositionType.SEQUENTIAL:
            return self._create_sequential_behavior(behaviors, flows, context)
        elif pipeline.composition_type == CompositionType.PARALLEL:
            return self._create_parallel_behavior(behaviors, flows, context)
        else:
            raise ValueError(f"Unsupported composition type: {pipeline.composition_type}")

    def _extract_flows_from_node(self, node) -> List[Flow]:
        """Extract Flow instances from a composed Flow node.

        Supports ThenNode (sequential composition) and falls back gracefully.
        """
        flows: List[Flow] = []
        try:
            from .flow import ModuleNode, ThenNode
        except Exception:
            return flows

        if isinstance(node, ThenNode):
            # Recursively flatten left then right
            flows.extend(self._extract_flows_from_node(node.first._node))
            flows.extend(self._extract_flows_from_node(node.second._node))
        elif isinstance(node, ModuleNode):
            # Cannot reconstruct the original Flow from a bare function node here
            # The parent ThenNode paths will append the actual Flow instances.
            pass
        else:
            # Try to access direct Flow references if present
            for attr in ("first", "second"):
                f = getattr(node, attr, None)
                if f is not None:
                    flows.append(f)

        return flows or []
    
    def _create_sequential_behavior(self, 
                                   behaviors: Dict[str, Behavior], 
                                   flows: List[Flow],
                                   context: FlowCompilationContext) -> Behavior:
        """Create behavior for sequential pipeline composition."""
        
        def sequential_sample(t: float):
            # Execute flows in sequence, threading data through
            result = None  # This will be set by the pipeline executor
            
            # For sequential composition, we need to execute flows in order
            # and pass the output of each flow as input to the next
            for i, flow in enumerate(flows):
                flow_id = f"pipeline_flow_{i}_{flow.__class__.__name__}"
                
                if i == 0:
                    # First flow uses external input (provided by executor)
                    input_data = context.get_input_at_time(flow_id, t)
                else:
                    # Subsequent flows use output of previous flow
                    input_data = result
                
                # Update context with current input
                context.add_input_provider(flow_id, lambda t, data=input_data: data)
                
                # Sample the behavior
                if flow_id in behaviors:
                    result = behaviors[flow_id].at_time(t)
            
            return result
        
        return Behavior(sequential_sample)
    
    def _create_parallel_behavior(self, 
                                 behaviors: Dict[str, Behavior], 
                                 flows: List[Flow],
                                 context: FlowCompilationContext) -> Behavior:
        """Create behavior for parallel pipeline composition."""
        
        def parallel_sample(t: float):
            results = []
            
            # Execute all flows in parallel with same input
            input_data = context.get_input_at_time("parallel_input", t)
            
            for i, flow in enumerate(flows):
                flow_id = f"pipeline_flow_{i}_{flow.__class__.__name__}"
                
                # All flows get same input in parallel composition
                context.add_input_provider(flow_id, lambda t, data=input_data: data)
                
                if flow_id in behaviors:
                    result = behaviors[flow_id].at_time(t)
                    results.append(result)
            
            return tuple(results)
        
        return Behavior(parallel_sample)
    
    def create_feedback_loop(self, 
                            high_level_flow: Flow,
                            low_level_flow: Flow,
                            feedback_delay: float = 0.1) -> Behavior:
        """Create feedback loop between high-level and low-level flows.
        
        This implements the two-level architecture with automatic feedback
        coordination between strategic (1Hz) and tactical (30Hz) levels.
        """
        
        context = FlowCompilationContext()
        
        # Register flows
        high_level_id = self.register_flow(high_level_flow, "high_level")
        low_level_id = self.register_flow(low_level_flow, "low_level")
        
        # Feedback buffer for communication
        feedback_buffer = TimestampedBuffer(max_size=100)
        
        # High-level behavior (strategic planning)
        def high_level_provider(t: float):
            # Get recent feedback from low-level execution
            recent_feedback = feedback_buffer.get_latest_before(t)
            return recent_feedback
        
        context.add_input_provider(high_level_id, high_level_provider)
        high_level_behavior = self.create_behavior_from_flow(high_level_flow, context, high_level_id)
        
        # Low-level behavior (tactical execution)  
        def low_level_provider(t: float):
            # Get current high-level plan
            return high_level_behavior.at_time(t)
        
        def low_level_consumer(result, t: float):
            # Send feedback to high-level with delay
            feedback_time = t + feedback_delay
            feedback_buffer.add(feedback_time, result)
        
        context.add_input_provider(low_level_id, low_level_provider)
        context.add_output_consumer(low_level_id, low_level_consumer)
        low_level_behavior = self.create_behavior_from_flow(low_level_flow, context, low_level_id)
        
        # Combined behavior returns low-level actions
        return low_level_behavior
    
    def _extract_frp_config(self, flow: Flow) -> FRPConfig:
        """Extract FRP configuration from flow annotations or attributes."""
        config = FRPConfig()
        
        # Check for @flow decorator annotations
        if hasattr(flow, '_rate'):
            config.rate = self._parse_rate(flow._rate)
        if hasattr(flow, '_feedback_to'):
            config.feedback_to = flow._feedback_to
        if hasattr(flow, '_feedback_from'):
            config.feedback_from = flow._feedback_from
        if hasattr(flow, '_trigger'):
            config.trigger = flow._trigger
        
        # Check for instance attributes
        if hasattr(flow, 'rate') and config.rate is None:
            config.rate = float(flow.rate)
        
        return config
    
    def _parse_rate(self, rate: Union[str, float, int]) -> float:
        """Parse rate specification to float Hz."""
        if isinstance(rate, str):
            if rate.endswith('hz'):
                return float(rate[:-2])
            else:
                return float(rate)
        elif isinstance(rate, (int, float)):
            return float(rate)
        else:
            raise ValueError(f"Invalid rate specification: {rate}")
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics from all compiled flows."""
        metrics = {}
        
        # Get rate coordination metrics
        metrics['rate_coordination'] = {
            'registered_flows': len(self.flow_configs),
            'active_rates': list(set(
                config.rate for config in self.flow_configs.values() 
                if config.rate is not None
            ))
        }
        
        # Get event processing metrics
        metrics['event_processing'] = {
            'registered_streams': len(self.event_manager.event_streams),
            'active_handlers': sum(
                len(handlers) for handlers in self.event_manager.event_handlers.values()
            )
        }
        
        return metrics
    
    def cleanup(self, max_age: float = 300.0):
        """Clean up old data from buffers and caches."""
        current_time = time.time()
        
        # Clean up rate coordinator
        self.rate_coordinator.cleanup_old_data(current_time, max_age)
        
        # Process recent events
        self.event_manager.process_events(current_time)


# Flow decorator with integrated FRP and Resource Management
def flow(rate: Optional[Union[str, float]] = None,
         feedback_to: Optional[str] = None,
         feedback_from: Optional[str] = None,
         trigger: Optional[str] = None,
         # Resource management parameters
         cpu: float = 1.0,
         gpu: float = 0.0,
         memory: float = 1.0,
         gpu_memory: float = 0.0,
         disk: float = 0.0,
         custom: Optional[Dict[str, float]] = None,
         node_type: str = "cpu",
         host_affinity: Optional[List[str]] = None,
         priority: int = 0,
         max_runtime: Optional[float] = None,
         preemptible: bool = True):
    """Unified decorator for Flow classes with FRP timing and resource management.
    
    FRP Parameters:
        rate: Execution frequency (e.g., "30hz", 30.0, "1hz")
        feedback_to: Flow ID to send feedback to
        feedback_from: Flow ID to receive feedback from  
        trigger: Event name that triggers execution
    
    Resource Parameters:
        cpu: Number of CPU cores required
        gpu: Number of GPUs required
        memory: Memory in GB required
        gpu_memory: GPU memory in GB required
        disk: Disk space in GB required
        custom: Custom resources (e.g., {"camera": 1, "robot_arm": 1})
        node_type: Type of compute node ("cpu", "gpu", "edge")
        host_affinity: List of preferred host nodes
        priority: Execution priority (higher = scheduled first)
        max_runtime: Maximum execution time in seconds
        preemptible: Whether the task can be preempted
    
    Examples:
        @flow(rate="30hz", cpu=2, memory=4)
        class PerceptionFlow(Flow[None, RGBImage]): ...
        
        @flow(rate="1hz", feedback_from="execution", cpu=4, memory=16)
        class PlanningFlow(Flow[ExecutionFeedback, Plan]): ...
        
        @flow(cpu=2, gpu=1, memory=8, custom={"camera": 1})
        class VisionFlow(Flow[RGBImage, List[Detection]]): ...
    """
    
    def decorator(cls):
        from .resources import ResourceSpec
        
        # Store FRP annotations as class attributes
        if rate is not None:
            # Primary attribute used by execution paths
            cls._rate = rate
            # Secondary attribute used by Dora translation layer (compat)
            cls._flow_rate = rate
        if feedback_to is not None:
            cls._feedback_to = feedback_to
            # Compatibility alias
            cls._flow_feedback_to = feedback_to
        if feedback_from is not None:
            cls._feedback_from = feedback_from
            # Compatibility alias
            cls._flow_feedback_from = feedback_from
        if trigger is not None:
            cls._trigger = trigger
            # Compatibility alias
            cls._flow_trigger = trigger
        
        # Create and store resource specification
        cls._resource_spec = ResourceSpec(
            cpu=cpu,
            gpu=gpu,
            memory=memory,
            gpu_memory=gpu_memory,
            disk=disk,
            custom=custom or {},
            node_type=node_type,
            host_affinity=host_affinity or [],
            priority=priority,
            max_runtime=max_runtime,
            preemptible=preemptible
        )
        
        return cls
    
    return decorator
