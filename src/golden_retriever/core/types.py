"""
Retriever Core Types: Flow Framework Types Only

This module defines ONLY the core Flow framework types. All data types 
(RGBImage, Detection, etc.) have been moved to retriever/types/

Core Framework Types:
- Module[I, O]: Base protocol for all pipeline components  
- Eff[S, A]: Effect monad for stateful operations
- TimedFlow: Time-aware flow with simple and advanced interfaces
- FRPConfig: Configuration for reactive flows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
    Optional,
)
from abc import ABC, abstractmethod

# ############################# Core Protocol ##############################
I = TypeVar("I")
O = TypeVar("O")


@runtime_checkable
class Module(Protocol, Generic[I, O]):
    """
    A typed, side-effect-free operator in the Retriever graph.

    This is the core protocol for any component that can be part of a Retriever
    computation graph. It's a simple, typed, callable object.

    This is philosophically similar to a "Layer" in a neural network framework
    or a "Module" in dspy-ai, representing a unit of computation.
    """

    def __call__(self, inp: I) -> O:
        ...


# ############################# Effect Monad ###############################
S = TypeVar("S")
A = TypeVar("A")
B = TypeVar("B")


@dataclass(frozen=True)
class Eff(Generic[S, A]):
    """
    A class for handling state and side effects in a functional way.

    The name "Eff" is short for "Effect". This is a "State Monad," a well-known
    functional programming pattern. It encapsulates a computation that threads
    a state `S` through a series of operations, ultimately producing a result `A`.

    The core idea is to make state management explicit. Instead of functions
    modifying a global or shared state, they take the current state as input
    and return the new state along with their result. The Eff class wraps this
    `S -> (A, S)` function, and the `>>` (bind) operator chains these
    computations together, hiding the state-passing boilerplate.

    This allows us to build complex, stateful workflows (like a robot
    interacting with the world) from simple, testable, pure-looking components.
    """

    run: Callable[[S], Tuple[A, S]]

    def __rshift__(self, k: Callable[[A], Eff[S, B]]) -> Eff[S, B]:
        """
        The "bind" operator (often written as `>>=` in Haskell or `flatMap`
        in Scala/Spark). It chains an effectful computation `k` after this one.

        Args:
            k: A function that takes the result of this computation (`A`) and
               returns the next computation (`Eff[S, B]`). This is also known
               as a "continuation."
        """

        def new_run(s0: S) -> Tuple[B, S]:
            a, s1 = self.run(s0)
            return k(a).run(s1)

        return Eff(new_run)


def pure(x: A) -> Eff[S, A]:
    """
    Lifts a pure, stateless value `x` into the Eff monad.

    It creates a computation that, when run, returns the value `x` without
    modifying the state at all.
    """
    return Eff(lambda s: (x, s))


def bind(e: Eff[S, A], k: Callable[[A], Eff[S, B]]) -> Eff[S, B]:
    """
    The implementation of the bind operator for the Eff monad.

    This function defines the core logic of the State Monad. To chain two
    computations:
    1. It defines a `new_run` function that takes the initial state `s0`.
    2. It runs the first computation `e` with `s0` to get a result `a` and a
       new state `s1`.
    3. It uses the result `a` to create the *next* computation by calling `k(a)`.
    4. It runs that new computation with the updated state `s1`.

    Args:
        e: The first effectful computation.
        k: The continuation that defines the next computation based on the
           result of the first.
    """

    def new_run(s0: S) -> tuple[B, S]:
        a, s1 = e.run(s0)
        return k(a).run(s1)

    return Eff(new_run)


from .frp_config import FRPConfig


# ########################## Time-Aware Flow ##########################
# Import timing types (assuming these will stay in temporal module)
try:
    from .temporal import ExecutionTimer, TimeConstraint
except ImportError:
    # Fallback definitions if temporal module not available
    ExecutionTimer = None
    TimeConstraint = None


from .flow import Flow  # Use Flow as the single flow abstraction


# ########################## Pipeline Type ##########################
# Pipeline is part of the core framework, represents composed flows
from typing import Union, List

Y = TypeVar("Y")


@dataclass
class PipelineNode:
    """Internal representation of pipeline composition tree."""
    flow: Optional['Flow'] = None
    left: Optional['PipelineNode'] = None  
    right: Optional['PipelineNode'] = None
    operation: Optional[str] = None  # "sequential" or "parallel"


class Pipeline(Generic[I, O]):
    """Pipeline represents a composition of flows with execution context.
    
    A Pipeline = Flow composition + execution environment information.
    Supports >> and & operators following functional programming conventions:
    - & (parallel) binds tighter than >> (sequential), like * vs +
    - Expressions like: a >> b & c >> d  parse as: a >> (b & c) >> d
    
    Pipeline can be executed directly with pipeline.execute() and contains
    execution context (environment, backend preferences, etc.).
    """
    
    def __init__(self, composition_tree: PipelineNode, execution_env: str = "development"):
        self.composition_tree = composition_tree
        self.execution_env = execution_env  # "development", "simulation", "real_robot"
        self.frp_config = FRPConfig()
        self.pipeline_id: Optional[str] = None
        self.backend_preference: Optional[str] = None
        # Default composition type for compatibility with FRP engine
        # Always treat as sequential unless a more advanced composition is introduced.
        self._composition_type_default = "sequential"
    
    @classmethod
    def from_flow(cls, flow: 'Flow') -> 'Pipeline[I, O]':
        """Create pipeline from single flow."""
        node = PipelineNode(flow=flow)
        return cls(node)
    
    def __call__(self, input_data: I) -> O:
        """Execute pipeline with given input."""
        # Simplified execution - just call the flow if single flow
        if self.composition_tree.flow:
            return self.composition_tree.flow(input_data)
        else:
            # For complex pipelines, would need execution engine
            raise NotImplementedError("Complex pipeline execution requires ExecutionEngine")
    
    def get_flows(self) -> List['Flow']:
        """Get all flows in this pipeline (required by ExecutionEngine)."""
        flows = []
        self._collect_flows(self.composition_tree, flows)
        return flows
    
    def _collect_flows(self, node: PipelineNode, flows: List['Flow']) -> None:
        """Recursively collect all flows from the composition tree."""
        if node.flow is not None:
            flows.append(node.flow)
        if node.left is not None:
            self._collect_flows(node.left, flows)
        if node.right is not None:
            self._collect_flows(node.right, flows)

    @property
    def composition_type(self):
        """Compatibility property used by some FRP compilation paths.

        Returns the FRP engine's CompositionType.SEQUENTIAL enum when available,
        otherwise a plain string "sequential" as a safe fallback.
        """
        try:
            # Local import to avoid import cycles at module load
            from .frp_engine import CompositionType  # type: ignore
            return CompositionType.SEQUENTIAL
        except Exception:
            return self._composition_type_default


# ########################## Forward References ##########################
# These will be defined in other modules but referenced here
if False:  # TYPE_CHECKING
    from ..types import RGBImage, Detection, Action
    from .flow import Flow
    from .execution import ExecutionEngine
