# Visual Predicate Integration

## Overview

This document describes the VLM (Vision-Language Model) predicate integration
for belief-space planning.

## Key Types

### VisualPredicate
A predicate evaluated by a VLM with a prompt template:
```python
ContainsWater = VisualPredicate(
    name="ContainsWater",
    types=[container_type],
    prompt_template="Does {obj0} contain water?",
)
```

### EpistemicValue (Three-Valued Logic)
- `TRUE` - Known to be true
- `FALSE` - Known to be false  
- `UNKNOWN` - Truth value not determined

### EpistemicState
Tracks Known/Unknown atoms with **regression prevention**:
```python
# Key invariant: Known atoms cannot become Unknown
if current != UNKNOWN and value == UNKNOWN:
    return False  # Reject regression
```

## Integration Points

1. **PerceptionFlow** → Outputs visual predicates with epistemic values
2. **BeliefState** → Tracks `EpistemicState` for planning
3. **TaskPlannerFlow** → Uses Known atoms for precondition checking

## File Locations

| File | Purpose |
|------|---------|
| `vlm.py` | VisualPredicate, EpistemicValue, EpistemicState |
| `belief.py` | BeliefState integrating visual atoms |
| `planning/task_planner.py` | TaskPlannerFlow using belief state |
