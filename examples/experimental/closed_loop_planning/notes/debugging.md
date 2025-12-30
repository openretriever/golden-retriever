# Debugging Notes

Issues encountered during implementation and their solutions.

## 2025-12-29: Signal Sampling Fix

### Problem
`PlannerFlow` wasn't receiving `state` input even though `Latest()` adapter was configured.

### Root Cause  
In `signal.py`, the `sample()` method only sampled fields in `fields_filter` (trigger fields),
ignoring non-triggering inputs with `Latest()` adapters.

### Solution
Changed `sample()` to ALWAYS sample ALL subscriber fields:
```python
# Before: only sample trigger fields
fields_to_sample = self.fields_filter

# After: sample ALL fields
fields_to_sample = list(self.subscribers.keys())
```

## 2025-12-29: Option Pickling Fix

### Problem
`Options` couldn't be serialized across processes in Dora backend.

### Root Cause
`ParameterizedOption.ground()` created closures that captured `self`, `memory`, etc.

### Solution
Made `Option` store raw data and call parent methods directly:
```python
class Option:
    def policy(self, state: State) -> Action:
        return self.parent.policy(state, self.memory, self.objects, self.params)
```

## 2025-12-29: Trigger Port Name Mismatch

### Problem
`SkillExecutorFlow` with `@Trigger("belief")` wasn't receiving data.

### Root Cause
Connection mapped `belief → state` but trigger listened for `"belief"`.

### Solution
Trigger must match the **input field name**, not the source output name:
```python
# Wrong:
executor = SkillExecutorFlow() @ Trigger("belief")

# Correct:
executor = SkillExecutorFlow() @ Trigger("state")
```

## 2025-12-29: State.get() vs State[]

### Problem
`AttributeError: 'State' object has no attribute 'get'`

### Root Cause
Used `state.get(robot)` instead of `state[robot]` (State uses `__getitem__`).

### Solution
```python
# Wrong: state.get(robot)
# Correct: 
current_pos = state[robot]
```
