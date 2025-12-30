# Predicators Integration Notes

## Components Integrated from Predicators

### Core Types (in `retriever.types.symbolic`)

| Type | Description | Status |
|------|-------------|--------|
| `ObjectType` | Type definitions | ✅ Integrated |
| `Object` | Typed entities | ✅ Integrated |
| `Variable` | Lifted planning | ✅ Integrated |
| `State` | Object→Features mapping | ✅ Integrated |
| `Predicate` | State classifiers | ✅ Integrated |
| `GroundAtom` | Ground predicates | ✅ Integrated |
| `LiftedAtom` | Lifted predicates | ✅ Integrated |

### Options Framework (in `retriever.types.options`)

| Type | Description | Status |
|------|-------------|--------|
| `Action` | Action wrapper | ✅ Integrated |
| `Option` | Temporally extended action | ✅ Integrated |
| `ParameterizedOption` | Option factory | ✅ Integrated |
| `Task` | Init + Goal | ✅ Integrated |

### Perception (in example `belief.py`)

| Type | Description | Status |
|------|-------------|--------|
| `BeliefState` | AugmentedState-like | ✅ Experimental |
| `BeliefUpdaterFlow` | Perceiver-like | ✅ Experimental |

## Components To Integrate

### High Priority
- `VLMPredicate` - Vision-language predicates
- `VLMGroundAtom` - VLM ground atoms  
- `NSRT` - Neuro-symbolic operators

### Medium Priority
- `Operator` - STRIPS-like operators
- `Monitor` - Execution monitors (predicators/execution_monitoring/)

### Lower Priority
- `LiftedDecisionList` - Policy representation
- `Approach` - Planning approaches

## Key Patterns from Predicators

### 1. Perceiver Pattern (perception/base_perceiver.py)
```python
class BasePerceiver:
    def reset(self, env_task: EnvironmentTask) -> Task
    def step(self, observation: Observation) -> State
    def update_perceiver_with_action(self, action: Action) -> None
```

Our `BeliefUpdaterFlow` maps to this pattern:
- `step()` → `step(inp: BeliefUpdateInput)` 
- History tracking built into BeliefState

### 2. Belief Update (perception/mock_spot_perceiver.py)
Key insight: **Known predicates cannot become Unknown**

```python
# From Predicators:
# Rule: Known predicates cannot become unknown
if self._vlm_atom_dict.get(known_atom, False):
    updated_vlm_atom_values[known_atom] = True
    updated_vlm_atom_values[unknown_atom] = False
```

### 3. State Augmentation (structs.py AugmentedState)
Predicators' `AugmentedState` includes:
- `camera_images`
- `camera_images_history` 
- `action_history`
- `vlm_atom_dict`
- `vlm_predicates`

Our `BeliefState` follows this pattern with slightly different naming.
