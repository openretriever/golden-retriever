# Belief State Architecture

## Overview

The BeliefState module provides epistemic state tracking for partially observable environments.
It follows patterns from [Predicators](https://github.com/Learning-and-Intelligent-Systems/predicators),
particularly the `MockSpotPerceiver` and `AugmentedState` implementations.

## Key Design Decisions

### 1. BeliefState extends State

```python
@flow_io
@dataclass
class BeliefState(State):
    data: dict[Object, NDArray]        # From State
    vlm_atoms: Dict[GroundAtom, Optional[bool]]  # VLM evaluations
    known_atoms: Set[GroundAtom]       # Known to be TRUE
    unknown_atoms: Set[GroundAtom]     # Unknown truth value
    action_history: List[Action]       # Temporal reasoning
    state_history: List[BeliefState]   # Belief trajectory
```

**Rationale**: By extending `State`, BeliefState is compatible with all existing
flows that expect `State` inputs. The additional epistemic fields are additive.

### 2. BeliefUpdaterFlow as State Estimation

The flow pattern is:
```
Perception → BeliefUpdater → Planner/Executor/Monitor
```

**Key invariant** (from Predicators):
- **Known atoms cannot regress to Unknown**
- Once a predicate is known, it stays known unless explicitly reset

### 3. Epistemic Predicate Naming Convention

Following Predicators' pattern for belief predicates:
- `Known_Inside(x, y)` - Agent knows x is inside y
- `Unknown_ContainerEmpty(c)` - Agent doesn't know if container is empty

This allows planning with belief-space operators that explicitly
gather information (observation actions).

## Integration with Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ GridEnvironmentFlow @ 10Hz                                  │
│   └→ PerceptionFlow @ Trigger("data")                       │
│        └→ BeliefUpdaterFlow @ Trigger("observation")        │
│             ├→ PlannerFlow @ Trigger("replan_config")       │
│             ├→ SkillExecutorFlow @ Trigger("state")         │
│             └→ ExecutionMonitorFlow @ Hybrid(2Hz, triggers) │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

1. **VLM Predicate Integration**: Add VLM-based predicate evaluation
2. **Observation Actions**: Implement info-gathering operators
3. **Belief Regression Prevention**: Stricter enforcement of knowledge persistence
