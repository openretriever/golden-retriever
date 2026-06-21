# Belief Space Types for Closed-Loop Planning
#
# This module provides belief state tracking for partially observable environments,
# integrating with visual predicates and epistemic (Known/Unknown) tracking.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from retriever.flow import io
from retriever.types.options import Action, Option
from retriever.types.symbolic import GroundAtom, State

from .vlm import EpistemicState, EpistemicValue, VisualGroundAtom


@io
@dataclass
class BeliefState(State):
    """State augmented with epistemic (belief) information.
    
    Attributes:
        data: Object → feature vector mapping (from State)
        visual_atoms: Visual ground atoms with epistemic values
        epistemic: Tracks Known/Unknown with regression prevention
        action_history: Past actions for temporal reasoning
        raw_observation: Optional raw observation data (images, point clouds, etc.)
    """

    # Visual predicate evaluations
    visual_atoms: Dict[VisualGroundAtom, EpistemicValue] = field(default_factory=dict)

    # Epistemic state tracker
    epistemic: EpistemicState = field(default_factory=EpistemicState)

    # History for temporal reasoning
    action_history: List[Action] = field(default_factory=list)

    # Perception data (generic - can be images, point clouds, etc.)
    raw_observation: Optional[Any] = None
    timestamp: Optional[float] = None

    def get_epistemic_value(self, atom: GroundAtom) -> EpistemicValue:
        """Get the epistemic value of a ground atom."""
        return self.epistemic.get_value(atom)

    def is_known_true(self, atom: GroundAtom) -> bool:
        """Check if atom is known to be TRUE."""
        return self.epistemic.get_value(atom) == EpistemicValue.TRUE

    def is_known_false(self, atom: GroundAtom) -> bool:
        """Check if atom is known to be FALSE."""
        return self.epistemic.get_value(atom) == EpistemicValue.FALSE

    def is_unknown(self, atom: GroundAtom) -> bool:
        """Check if atom has UNKNOWN truth value."""
        return self.epistemic.get_value(atom) == EpistemicValue.UNKNOWN

    def __getitem__(self, key: Any) -> Any:
        """Allow dict-like access to state data (e.g. state[obj])."""
        return self.data[key]

    def get(self, key: Any, default: Any = None) -> Any:
        """Allow safe dict-like access (e.g. state.get(obj))."""
        return self.data.get(key, default)

    def __contains__(self, key: Any) -> bool:
        """Allow 'obj in state' checks."""
        return key in self.data


@io
@dataclass
class BeliefUpdateInput:
    """Input to BeliefUpdaterFlow."""

    observation: Optional[State] = None
    visible_atoms: Set[GroundAtom] = field(default_factory=set)
    prev_belief: Optional[BeliefState] = None
    action: Optional[Action] = None
    plan: List[Option] = field(default_factory=list)  # Changed from Optional
    raw_observation: Optional[Any] = None  # Raw observation data (images, etc.)


@io
@dataclass
class BeliefUpdateOutput:
    """Output from BeliefUpdaterFlow."""
    belief: BeliefState





