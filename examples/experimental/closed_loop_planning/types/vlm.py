# Visual Predicate Types for Belief-Space Planning
#
# This module provides types for predicates evaluated by vision-language models
# (VLMs) with proper epistemic (Known/Unknown) state tracking.
#
# Key concepts:
# - VisualPredicate: A predicate evaluated by VLM with prompt template
# - EpistemicValue: Three-valued logic (TRUE, FALSE, UNKNOWN)
# - Known/Unknown pairs: For belief-space planning operators
#
# Based on Predicators' VLMPredicate and belief-space patterns.

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from retriever.types.symbolic import GroundAtom, Object, ObjectType, Predicate


class EpistemicValue(Enum):
    """Three-valued epistemic truth value."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self) -> bool:
        """For boolean contexts, UNKNOWN is treated as False."""
        return self == EpistemicValue.TRUE


@dataclass(frozen=True, order=False, repr=False)
class VisualPredicate(Predicate):
    """A predicate evaluated by a vision-language model.

    Visual predicates are partially observable - their truth value requires
    VLM inference on images. They support epistemic (belief-space) tracking
    with Known/Unknown state.

    Attributes:
        name: Predicate name (e.g., "ContainsWater")
        types: Expected object types for grounding
        prompt_template: VLM prompt with {obj0}, {obj1} placeholders
        is_belief_predicate: If True, creates Known_X/Unknown_X pairs

    Example:
        ContainsWater = VisualPredicate(
            name="ContainsWater",
            types=[container_type],
            prompt_template="Does {obj0} contain water?",
            is_belief_predicate=True,  # Creates Known_ContainsWater
        )
    """

    prompt_template: str = ""
    is_belief_predicate: bool = False

    def format_prompt(self, objects: Sequence[Object]) -> str:
        """Format the prompt template with object names."""
        replacements = {f"obj{i}": obj.name for i, obj in enumerate(objects)}
        prompt = self.prompt_template
        for key, val in replacements.items():
            prompt = prompt.replace(f"{{{key}}}", val)
        return prompt

    def __call__(self, entities: Sequence[Object]) -> "VisualGroundAtom":
        """Create a VisualGroundAtom from this predicate and objects."""
        return VisualGroundAtom(self, entities)

    def get_known_predicate(self) -> "VisualPredicate":
        """Get the Known_X version of this predicate."""
        return VisualPredicate(
            name=f"Known_{self.name}",
            types=self.types,
            _classifier=lambda s, o: False,  # Placeholder
            prompt_template=self.prompt_template,
            is_belief_predicate=False,
        )

    def get_unknown_predicate(self) -> "VisualPredicate":
        """Get the Unknown_X version of this predicate."""
        return VisualPredicate(
            name=f"Unknown_{self.name}",
            types=self.types,
            _classifier=lambda s, o: True,  # Default to unknown
            prompt_template="",
            is_belief_predicate=False,
        )


@dataclass(frozen=True, repr=False, eq=False)
class VisualGroundAtom(GroundAtom):
    """A grounded visual predicate with epistemic tracking.

    Extends GroundAtom with:
    - Three-valued epistemic state (TRUE/FALSE/UNKNOWN)
    - Confidence score from VLM
    - VLM prompt for evaluation
    """

    # Mutable metadata stored in dict (keeps frozen dataclass hashable)
    _metadata: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def epistemic_value(self) -> EpistemicValue:
        """The epistemic truth value."""
        return self._metadata.get("epistemic", EpistemicValue.UNKNOWN)

    @epistemic_value.setter
    def epistemic_value(self, v: EpistemicValue) -> None:
        self._metadata["epistemic"] = v

    @property
    def confidence(self) -> Optional[float]:
        """VLM confidence score (0.0 to 1.0)."""
        return self._metadata.get("confidence")

    @confidence.setter
    def confidence(self, c: float) -> None:
        self._metadata["confidence"] = c

    @property
    def is_known(self) -> bool:
        """True if the atom's truth value is known (not UNKNOWN)."""
        return self.epistemic_value != EpistemicValue.UNKNOWN

    @property
    def is_true(self) -> bool:
        """True if the atom is known to be true."""
        return self.epistemic_value == EpistemicValue.TRUE

    @property
    def prompt(self) -> str:
        """Get the formatted VLM prompt for this atom."""
        if isinstance(self.predicate, VisualPredicate):
            return self.predicate.format_prompt(self.objects)
        return f"{self.predicate.name}({', '.join(o.name for o in self.objects)})"


# --- Epistemic State Tracking ---


@dataclass
class EpistemicState:
    """Tracks Known/Unknown atoms for belief-space planning.

    Maintains the invariant: Known atoms cannot regress to Unknown.
    """

    known_true: Set[GroundAtom] = field(default_factory=set)
    known_false: Set[GroundAtom] = field(default_factory=set)
    unknown: Set[GroundAtom] = field(default_factory=set)

    def get_value(self, atom: GroundAtom) -> EpistemicValue:
        """Get the epistemic value of an atom."""
        if atom in self.known_true:
            return EpistemicValue.TRUE
        if atom in self.known_false:
            return EpistemicValue.FALSE
        return EpistemicValue.UNKNOWN

    def update(self, atom: GroundAtom, value: EpistemicValue) -> bool:
        """Update atom's epistemic value. Returns True if changed.

        Enforces: Known atoms cannot become Unknown.
        """
        current = self.get_value(atom)

        # Key invariant: Known cannot regress to Unknown
        if current != EpistemicValue.UNKNOWN and value == EpistemicValue.UNKNOWN:
            return False  # Reject regression

        # Remove from current set
        self.known_true.discard(atom)
        self.known_false.discard(atom)
        self.unknown.discard(atom)

        # Add to new set
        if value == EpistemicValue.TRUE:
            self.known_true.add(atom)
        elif value == EpistemicValue.FALSE:
            self.known_false.add(atom)
        else:
            self.unknown.add(atom)

        return current != value


# --- VLM Evaluation ---

VisualEvaluator = Callable[
    [List[VisualGroundAtom], Any], Dict[VisualGroundAtom, EpistemicValue]
]
"""Type for visual evaluation: (atoms, images) -> {atom: epistemic_value}"""


def mock_visual_evaluator(
    atoms: List[VisualGroundAtom], images: Any = None
) -> Dict[VisualGroundAtom, EpistemicValue]:
    """Mock evaluator for testing - returns all UNKNOWN."""
    return dict.fromkeys(atoms, EpistemicValue.UNKNOWN)


def get_visual_atom_combinations(
    objects: Set[Object], predicates: Set[VisualPredicate]
) -> List[VisualGroundAtom]:
    """Generate all valid visual ground atoms from objects and predicates."""
    atoms = []
    obj_list = list(objects)

    for pred in predicates:
        for grounding in _get_groundings(pred.types, obj_list):
            atoms.append(VisualGroundAtom(pred, grounding))

    return atoms


def _get_groundings(
    types: Sequence[ObjectType], objects: List[Object]
) -> List[Sequence[Object]]:
    """Get all type-respecting object combinations."""
    if not types:
        return [()]

    first_type, rest_types = types[0], types[1:]
    result = []

    for obj in objects:
        if obj.is_instance(first_type):
            for rest in _get_groundings(rest_types, objects):
                result.append((obj,) + tuple(rest))

    return result
