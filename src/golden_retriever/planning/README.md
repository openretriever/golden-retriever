# Retriever Planning System

**Clean bilevel closed-loop planning architecture built on validated patterns**

## Overview

This planning system implements the complete bilevel architecture from Proposal v2.7:

- **Strategic Layer (1Hz)**: High-level task decomposition and skill selection
- **Tactical Layer (10Hz-30Hz)**: Real-time action planning and skill execution  
- **FRP Coordination**: Automatic temporal coordination between layers
- **Closed-Loop Control**: Continuous monitoring and adaptive replanning

## Architecture

### Single-Level Foundation
Simple closed-loop pattern that validates core concepts:
- Planning → Execution → Monitoring → Replanning cycle
- FRP timing coordination (10Hz planning, 30Hz execution, 5Hz monitoring)
- State management through Eff monad
- Monitor-driven replanning on failures

### Bilevel Extension
Full strategic/tactical separation built on single-level patterns:
- **Strategic Planner**: LLM-based task decomposition (1Hz)
- **Tactical Planner**: Primitive action planning (10Hz)  
- **Skill Policy**: RT-1/π0/Custom execution (30Hz)
- **Monitoring**: Strategic and tactical failure detection

## Key Components

### Core Types
```python
# Task specification
TaskRequest(description="pick up the red cup", constraints=["gentle"])

# Strategic planning output  
StrategicPlan(skills=[navigate_skill, pick_skill, place_skill])

# Tactical planning output
TacticalPlan(action_sequence=[approach, grasp, lift])

# Execution state management
RobotState(pose={...}, holding="cup", execution_history=[...])
```

### Strategic Planning
```python
# LLM-based strategic planning
strategic_planner = create_llm_strategic_planner("gpt-4")

# Template-based planning for common tasks
strategic_planner = create_template_strategic_planner()

# VLA-based planning with vision integration
strategic_planner = create_vla_strategic_planner("rt1")
```

### Tactical Execution
```python
# Primitive action planning
tactical_planner = create_primitive_tactical_planner()

# RT-1 skill policy for VLA control
skill_policy = create_rt1_skill_policy()

# π0 foundation model for general robotics
skill_policy = create_pi0_skill_policy()

# Custom skill policies for domain-specific needs
skill_policy = create_custom_skill_policy(custom_implementations)
```

### Bilevel Coordination
```python
# Complete bilevel system
bilevel_system = create_bilevel_system(
    strategic_planner=strategic_planner,
    tactical_planner=tactical_planner,
    skill_policy=skill_policy,
    monitor=monitor
)

# Main closed-loop flow
closed_loop_flow = bilevel_system.create_closed_loop_flow()

# Execute task with automatic FRP coordination
task = TaskRequest(description="organize the kitchen")
result_eff = closed_loop_flow.run(task)
result, final_state = result_eff.run(initial_state)
```

## Examples

### Single-Level Validation
`examples/03_frp_coordination/02_single_level_closed_loop.py`
- Basic closed-loop pattern validation
- Planning → execution → monitoring cycle
- Monitor-driven replanning
- Multi-rate FRP coordination

### Bilevel Capability Test  
`examples/capability_tests/bilevel_closed_loop_demo.py`
- Complete bilevel architecture demonstration
- Strategic/tactical planning variants
- RT-1/π0 skill policy integration
- Dora FRP backend deployment

## Key Benefits

1. **Simple Interfaces**: Complex FRP coordination hidden behind familiar Flow[I,O] patterns
2. **Progressive Complexity**: Single-level → bilevel → production deployment
3. **Validated Patterns**: Built on tested single-level closed-loop foundations
4. **Production Ready**: Automatic Dora FRP backend integration
5. **Flexible**: Support for LLM, VLA, and custom planners/policies

## Migration from Legacy

Legacy planning modules have been moved to `archive_legacy/`:
- `modules.py`, `pipelines.py`, `combined_pipelines.py` → Use bilevel architecture instead
- `llm_planner.py`, `dspy_llm_planner.py` → Use `create_llm_strategic_planner()`
- `examples/`, `grounding/`, `utils/` → Use new capability tests and examples

The new bilevel architecture provides all functionality with cleaner interfaces and better testing.