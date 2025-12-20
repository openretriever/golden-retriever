"""
Retriever Flow: Composable Computation Graphs for Robotics Pipelines

This module implements Flow combinators - a functional programming pattern for building
composable, type-safe computation pipelines. Think of it like "PyTorch for robotics"
where instead of neural network layers, you compose perception, planning, and control modules.

## Why Use Flows in Robotics?

Traditional robotics code often has tangled dependencies:
- Hard to test individual components
- Difficult to reuse across different robots
- No type safety between pipeline stages
- Complex parallel processing logic

Flows solve these problems by providing:
1. **Type Safety**: Catch mismatched pipeline connections at development time
2. **Composability**: Build complex systems from simple, reusable parts  
3. **Parallelization**: Easy parallel execution with `fanout()`
4. **Testability**: Each component can be mocked and tested independently

## Key Operations:

- `Flow.from_module(f)`: Wrap any function into a Flow
- `a.then(b)`: Sequential composition (a's output → b's input)
- `a.fanout(b)`: Parallel composition (same input → both a and b)

## Robotics Examples:

```python
# Basic perception pipeline
camera_input = Flow.from_module(capture_image)
object_detection = Flow.from_module(yolo_model.predict)  
pose_estimation = Flow.from_module(estimate_6dof_poses)

perception_pipeline = camera_input.then(object_detection).then(pose_estimation)

# Multi-camera fusion
left_camera = Flow.from_module(process_left_stereo)
right_camera = Flow.from_module(process_right_stereo)
stereo_vision = left_camera.fanout(right_camera)

# Planning with multiple algorithms
rrt_planner = Flow.from_module(rrt_plan)
prm_planner = Flow.from_module(prm_plan)
multi_planning = rrt_planner.fanout(prm_planner)

# Full system composition
robot_system = (
    stereo_vision
    .then(perception_pipeline)
    .then(multi_planning)
    .then(Flow.from_module(select_best_plan))
    .then(Flow.from_module(execute_trajectory))
)
```

The Flow abstraction lets you build complex robotics systems as compositions
of simple, well-typed, testable components.

## Backward Compatibility

For backward compatibility, we maintain the Arrow class as an alias to Flow.
This will be deprecated in future versions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar

# Type variables for generic programming
# These represent the input and output types flowing through the computation graph
X = TypeVar("X")  # Input type
Y = TypeVar("Y")  # Output type  
Z = TypeVar("Z")  # Intermediate type for composition

from .frp_config import FRPConfig


# ========================= Computation Graph Nodes =========================
# These represent the internal structure of the computation graph.
# Users typically don't interact with these directly.


@dataclass(frozen=True)
class FlowNode:
    """
    Base class for nodes in the Flow computation graph.
    
    This is an internal representation - think of it as the "computation graph"
    that gets built when you compose Flows with .then() and .fanout().
    Similar to how PyTorch builds computation graphs for backpropagation.
    """


@dataclass(frozen=True)
class ModuleNode(FlowNode, Generic[X, Y]):
    """
    A leaf node that wraps a simple function.
    
    This represents a single "module" in your pipeline - could be a
    YOLO detector, a motion planner, a robot controller, etc.
    
    Args:
        func: The actual function that does the computation
              (e.g., a perception model, planner algorithm, etc.)
    """

    func: Callable[[X], Y]


@dataclass(frozen=True)
class ThenNode(FlowNode, Generic[X, Y, Z]):
    """
    A node representing sequential composition: first → second.
    
    This is like a pipeline stage where the output of 'first' becomes
    the input to 'second'. Common in robotics:
    - Raw images → Object detection → Pose estimation
    - Goal specification → Path planning → Trajectory optimization
    
    Args:
        first: The first computation in the sequence
        second: The second computation that processes first's output
    """

    first: Flow[X, Y]
    second: Flow[Y, Z]


@dataclass(frozen=True)
class FanoutNode(FlowNode, Generic[X, Y, Z]):
    """
    A node representing parallel composition: input goes to both branches.
    
    This is like running multiple processing streams in parallel:
    - Multi-camera processing: same timestamp → [left_cam, right_cam]
    - Multi-hypothesis planning: same goal → [plan_A, plan_B, plan_C]
    - Redundant perception: same image → [yolo_detection, faster_rcnn]
    
    Args:
        first: The first parallel branch
        second: The second parallel branch (gets same input as first)
    """

    first: Flow[X, Y]
    second: Flow[X, Z]


@dataclass(frozen=True)
class TriggeredNode(FlowNode, Generic[X, Y]):
    """
    A node representing event-driven composition: flow triggered by condition.
    
    This represents reactive behavior where one flow monitors conditions
    and triggers another flow when the condition is met:
    - Safety monitor triggers emergency stop
    - Execution monitor triggers replanning
    - Object detection triggers grasping behavior
    
    Args:
        target_flow: The flow to execute when triggered
        trigger_flow: The flow that monitors for trigger conditions  
        condition: Function that determines when to trigger (takes trigger output)
        action: Optional action parameter passed to target flow
    """
    target_flow: Flow[X, Y]
    trigger_flow: Flow[X, Any]
    condition: Callable[[Any], bool]
    action: Optional[Any] = None


@dataclass(frozen=True) 
class MultiInputNode(FlowNode, Generic[X, Y]):
    """
    A node representing multi-input composition: coordinator with multiple inputs.
    
    This represents coordination patterns where one flow needs inputs from
    multiple other flows for comprehensive system integration:
    - System coordinator receiving: safety, perception, planning, execution
    - Sensor fusion receiving: camera, lidar, IMU, GPS
    - Decision maker receiving: multiple planning hypotheses
    
    Args:
        coordinator_flow: The flow that coordinates multiple inputs
        input_flows: List of flows that provide inputs to coordinator
    """
    coordinator_flow: Flow[Tuple, Y]  # Takes tuple of all inputs
    input_flows: List[Flow[X, Any]]


# ========================= The Main Flow Class =========================


class Flow(Generic[X, Y]):
    """
    PyTorch-style base class for composable robotics computations.
    
    This is the main abstraction for building robotics pipelines with v3.5 design:
    - PyTorch-style interface: familiar __init__ + run() pattern
    - Hidden complexity management: framework handles Eff monads automatically  
    - Type safety: Flow[Input, Output] catches pipeline mismatches at compile time
    - Easy composition: camera >> detection >> planning with >> operator
    
    Key benefits for robotics systems:
    - Write simple classes like PyTorch modules, get sophisticated robot systems
    - Type safety catches pipeline mismatches at development time
    - Reusable components work across different robots and tasks
    - Easy parallelization for multi-sensor or multi-hypothesis processing
    - Individual components can be mocked and tested independently
    
    Common robotics patterns:
    - Sequential: camera >> detection >> pose_estimation >> planning
    - Parallel: left_cam & right_cam for stereo vision
    - Stateful: robot belief management with hidden Eff monads
    - Multi-rate: @flow(rate="30hz") for automatic process coordination
    """

    def __init__(self, node: Optional[FlowNode] = None):
        """
        Initialize a Flow with PyTorch-style pattern.
        
        If node is None, automatically creates a ModuleNode from the run() method.
        This enables clean PyTorch-style inheritance with zero boilerplate:
        
        ```python
        # v3.5 PyTorch-style pattern (zero boilerplate!)
        @flow(rate="30hz")
        class CameraFlow(Flow[None, RGBImage]):
            def __init__(self, camera_id=0):
                super().__init__()  # Clean PyTorch-style
                self.camera_id = camera_id
            
            def run(self, _: None) -> RGBImage:
                return capture_camera(self.camera_id)
        
        # Use immediately like PyTorch
        camera = CameraFlow(camera_id=1)
        image = camera(None)
        ```
        """
        if node is None:
            # Auto-create ModuleNode from run method (eliminates boilerplate)
            if hasattr(self, 'run'):
                self._node = ModuleNode(self.run)
            else:
                raise ValueError("Flow must have either a node or a run() method")
        else:
            self._node = node
        
        # Initialize FRP configuration for @flow decorator support
        self.frp_config = FRPConfig()
        self.flow_id: Optional[str] = None

    # Optional advanced timing interface (used by some executors)
    # Default implementation delegates to simple run() for convenience.
    def run_timed(self, input_data: X, timer=None) -> Y:  # type: ignore[override]
        return self.run(input_data)  # type: ignore[misc]

    @staticmethod
    def from_module(f: Callable[[X], Y]) -> Flow[X, Y]:
        """
        Lift a simple function into a Flow.
        
        This is how you create the basic building blocks of your pipeline.
        Any function can become a Flow - perception models, planners,
        controllers, data processors, etc.
        
        Args:
            f: Any function that transforms X to Y
        
        Returns:
            A Flow that represents this computation
            
        Example:
            ```python
            # Wrap existing robotics functions
            detect = Flow.from_module(yolo_model.predict)
            plan = Flow.from_module(rrt_planner.plan)
            control = Flow.from_module(mpc_controller.compute)
            
            # Wrap lambdas for simple transformations
            to_meters = Flow.from_module(lambda cm: cm / 100.0)
            filter_detections = Flow.from_module(lambda dets: [d for d in dets if d.confidence > 0.8])
            ```
        """
        return Flow(ModuleNode(f))

    @staticmethod 
    def arr(f: Callable[[X], Y]) -> Flow[X, Y]:
        """
        DEPRECATED: Use Flow.from_module() instead.
        
        This method is maintained for backward compatibility with Arrow-based code.
        """
        warnings.warn(
            "Flow.arr() is deprecated. Use Flow.from_module() instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        return Flow.from_module(f)

    # ------------------------- Convenience APIs -------------------------
    def to_pipeline(self):
        """
        Convert this composed Flow into a Pipeline.

        This provides a small convenience to align with documentation that
        shows composition producing a Pipeline. Internally, we compose Flows
        and allow conversion to Pipeline explicitly when needed.
        """
        try:
            # Local import to avoid circular dependencies at module import time
            from .types import Pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Pipeline class not available: {e}")
        return Pipeline.from_flow(self)  # type: ignore[arg-type]

    def then(self, g: Flow[Y, Z]) -> Flow[X, Z]:
        """
        Compose this Flow with another one sequentially: self → g.
        
        This creates a pipeline where the output of this Flow becomes
        the input to the next Flow. Essential for robotics workflows.
        
        Args:
            g: The Flow to execute after this one
            
        Returns:
            A new Flow representing the sequential composition
            
        Example:
            ```python
            # Build a perception → planning pipeline
            perception = Flow.from_module(detect_objects)
            planning = Flow.from_module(plan_path)
            pipeline = perception.then(planning)
            
            # Chain multiple stages for complete autonomy
            full_system = (
                camera_processing
                .then(object_detection)      # Raw image → detections
                .then(pose_estimation)       # Detections → 6D poses
                .then(motion_planning)       # Poses → path plan
                .then(trajectory_optimization) # Plan → smooth trajectory
                .then(robot_control)         # Trajectory → joint commands
            )
            ```
        """
        return Flow(ThenNode(self, g))

    def fanout(self, g: Flow[X, Z]) -> Flow[X, Tuple[Y, Z]]:
        """
        Compose this Flow with another one in parallel: both process same input.
        
        This creates parallel processing where both Flows receive the same
        input and their outputs are combined as a tuple (left_result, right_result).
        Perfect for multi-sensor fusion, redundant processing, or exploring multiple hypotheses.
        
        Args:
            g: The Flow to execute in parallel with this one
            
        Returns:
            A new Flow that outputs (self_result, g_result)
            
        Example:
            ```python
            # Multi-camera processing for stereo vision
            left_cam = Flow.from_module(process_left_camera)
            right_cam = Flow.from_module(process_right_camera)
            stereo = left_cam.fanout(right_cam)
            # Input: timestamp → Output: (left_features, right_features)
            
            # Multiple detection algorithms for robustness
            yolo = Flow.from_module(yolo_detect)
            faster_rcnn = Flow.from_module(rcnn_detect)
            multi_detect = yolo.fanout(faster_rcnn)
            # Input: image → Output: (yolo_detections, rcnn_detections)
            
            # Parallel planning strategies
            rrt = Flow.from_module(rrt_planner)
            prm = Flow.from_module(prm_planner)
            a_star = Flow.from_module(astar_planner)
            multi_plan = rrt.fanout(prm).fanout(a_star)
            # Input: (start, goal) → Output: ((rrt_path, prm_path), astar_path)
            ```
        """
        return Flow(FanoutNode(self, g))

    def __rshift__(self, g: Flow[Y, Z]) -> Flow[X, Z]:
        """Provides `>>` as a more readable syntax for `then`."""
        return self.then(g)

    def __and__(self, g: Flow[X, Z]) -> Flow[X, Tuple[Y, Z]]:
        """Provides `&` as a more readable syntax for `fanout`."""
        return self.fanout(g)

    def triggered_by(self, trigger: Flow[X, Any], condition: Callable[[Any], bool], action: Any = None) -> Flow[X, Y]:
        """
        Event-driven composition: execute this flow when trigger condition is met.
        
        This creates reactive behavior where one flow monitors conditions and triggers
        another flow when specific conditions are detected. Essential for event-driven
        robotics systems with safety monitoring, failure detection, and replanning.
        
        Args:
            trigger: The flow that monitors for trigger conditions
            condition: Function that takes trigger output and returns bool (True = trigger)
            action: Optional action parameter passed to execution context
            
        Returns:
            A new Flow representing the event-driven composition
            
        Example:
            ```python
            # Safety-triggered emergency stop
            @flow(rate="30hz")
            class SafetyMonitor(Flow[RobotState, bool]):
                def run(self, state): return state.force < 80.0  # Safe when True
            
            emergency_stop = EmergencyStopFlow().triggered_by(
                trigger=SafetyMonitor(),
                condition=lambda safe: not safe,  # Trigger when not safe
                action="EMERGENCY_STOP"
            )
            
            # Execution monitoring triggering replanning
            replanner = ReplannerFlow().triggered_by(
                trigger=ExecutionMonitor(),
                condition=lambda status: status.failed,  # Trigger on failure
                action="REPLAN"
            )
            
            # Object detection triggering grasping
            grasp_action = GraspFlow().triggered_by(
                trigger=ObjectDetector(),
                condition=lambda detections: len(detections) > 0,  # Trigger when objects found
                action="GRASP"
            )
            ```
        """
        return Flow(TriggeredNode(
            target_flow=self,
            trigger_flow=trigger,
            condition=condition,
            action=action
        ))

    def with_inputs(self, input_flows: List[Flow[X, Any]]) -> Flow[X, Y]:
        """
        Multi-input composition: coordinate inputs from multiple flows.
        
        This creates coordination patterns where this flow receives inputs from
        multiple other flows for comprehensive system integration. Perfect for
        system coordinators, sensor fusion, and multi-hypothesis decision making.
        
        Args:
            input_flows: List of flows that provide inputs to this coordinator flow
            
        Returns:
            A new Flow representing the multi-input coordination
            
        Example:
            ```python
            # System coordinator receiving multiple inputs
            system_coordinator = SystemCoordinatorFlow().with_inputs([
                SafetyMonitorFlow(),      # Safety status
                PerceptionFlow(),         # Robot state  
                PlanningFlow(),          # Planning state
                ExecutionFlow(),         # Execution status
                ObjectTrackerFlow()      # Object tracking
            ])
            
            # Sensor fusion coordinator
            sensor_fusion = SensorFusionFlow().with_inputs([
                CameraFlow(),            # Visual data
                LidarFlow(),            # Point cloud data
                IMUFlow(),              # Inertial data
                GPSFlow()               # Position data
            ])
            
            # Multi-hypothesis decision maker
            decision_maker = DecisionFlow().with_inputs([
                RRTPlanner(),           # RRT planning hypothesis
                PRMPlanner(),           # PRM planning hypothesis
                AStarPlanner()          # A* planning hypothesis
            ])
            ```
            
        Note:
            The coordinator flow should expect a tuple of inputs in its run() method:
            ```python
            class SystemCoordinatorFlow(Flow[Tuple, Actions]):
                def run(self, inputs: Tuple) -> Actions:
                    safety, perception, planning, execution, tracking = inputs
                    # Coordinate all inputs to produce final actions
                    return self.coordinate(safety, perception, planning, execution, tracking)
            ```
        """
        return Flow(MultiInputNode(
            coordinator_flow=self, 
            input_flows=input_flows
        ))

    def __call__(self, arg: X) -> Y:
        """
        Execute the Flow computation graph directly (synchronous).
        
        This provides a simple synchronous execution method for development
        and testing. For production robotics systems with real-time constraints,
        use the LocalExecutor instead for async execution and parallelization.
        
        Args:
            arg: The input to the computation
            
        Returns:
            The result of the computation
            
        Example:
            ```python
            # Quick testing during development
            perception_flow = Flow.from_module(yolo_detect).then(Flow.from_module(estimate_poses))
            result = perception_flow(test_image)
            
            # For production, use async executor:
            # result = await executor.execute(perception_flow, sensor_data)
            ```
            
        Note:
            This executes synchronously and doesn't parallelize fanout operations.
            For real robotics systems, use LocalExecutor.execute() for better performance.
        """
        return self._execute(self._node, arg)

    def _execute(self, node: FlowNode, arg: Any) -> Any:
        """
        Private execution method to traverse the computation graph.
        
        This implements a simple interpreter for the Flow computation graph.
        It's a recursive tree walk that executes each node type appropriately.
        Includes support for event-driven patterns (v3.6).
        """
        if isinstance(node, ModuleNode):
            # Base case: execute the wrapped function
            return node.func(arg)
        if isinstance(node, ThenNode):
            # Sequential: execute first, then second with first's output
            res1: Any = self._execute(node.first._node, arg)
            return self._execute(node.second._node, res1)
        if isinstance(node, FanoutNode):
            # Parallel: execute both with same input, combine outputs
            res1 = self._execute(node.first._node, arg)
            res2 = self._execute(node.second._node, arg)
            return (res1, res2)
        if isinstance(node, TriggeredNode):
            # Event-driven: check trigger condition, execute target if triggered
            trigger_result = self._execute(node.trigger_flow._node, arg)
            should_trigger = node.condition(trigger_result)
            
            if should_trigger:
                # Execute target flow when condition is met
                return self._execute(node.target_flow._node, arg)
            else:
                # Return None or default value when not triggered
                # In production, this would be handled by the async backend
                return None
        if isinstance(node, MultiInputNode):
            # Multi-input: execute all input flows, collect results, pass tuple to coordinator
            input_results = []
            for input_flow in node.input_flows:
                result = self._execute(input_flow._node, arg)
                input_results.append(result)
            
            # Pass tuple of all input results to coordinator flow
            return self._execute(node.coordinator_flow._node, tuple(input_results))
        
        raise TypeError(f"Unknown node type: {type(node)}") 


# ========================= Backward Compatibility =========================

class Arrow(Flow[X, Y]):
    """
    DEPRECATED: Arrow is now an alias for Flow.
    
    This class is maintained for backward compatibility. All new code should use Flow.
    Arrow will be removed in a future version.
    """
    
    def __init__(self, node: FlowNode):
        warnings.warn(
            "Arrow class is deprecated. Use Flow instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        super().__init__(node)

    @staticmethod
    def arr(f: Callable[[X], Y]) -> Arrow[X, Y]:
        """DEPRECATED: Use Flow.from_module() instead."""
        warnings.warn(
            "Arrow.arr() is deprecated. Use Flow.from_module() instead.", 
            DeprecationWarning, 
            stacklevel=2
        )
        return Arrow(ModuleNode(f))


# Backward compatibility aliases
ArrowNode = FlowNode  # For existing code that references ArrowNode
ArrNode = ModuleNode  # For existing code that references ArrNode 
