# 06 Feedback Loops (tutorial)

Runtime-aligned feedback-loop examples built on top of `Pipeline` cycles.

## Run

```bash
# Minimal closed-loop intro (proportional controller)
pixi run python -m examples.tutorial.06_feedback_loops.00_feedback_intro --backend multiprocessing --duration 3

# Dora backend + strict lag handling (panic is an alias for error)
pixi run python -m examples.tutorial.06_feedback_loops.00_feedback_intro --backend dora --on-lag panic --duration 3

# Event-driven replanning (only emits a plan when an "event" occurs)
pixi run python -m examples.tutorial.06_feedback_loops.01_event_driven_replan --backend multiprocessing --duration 2

# Dora backend
pixi run python -m examples.tutorial.06_feedback_loops.01_event_driven_replan --backend dora --duration 2

# Execution monitoring (only emits alerts when "stuck")
pixi run python -m examples.tutorial.06_feedback_loops.02_execution_monitoring --backend multiprocessing --duration 3

# One-shot time trigger (deterministic, stepper)
pixi run python -m examples.tutorial.06_feedback_loops.03_time_triggers --steps 12 --dt 0.1 --delay 0.6

# Safety monitoring (event-driven actions)
pixi run python -m examples.tutorial.06_feedback_loops.04_safety_monitoring --steps 12 --dt 0.1

# Stateful replanning (planner keeps internal memory)
pixi run python -m examples.tutorial.06_feedback_loops.05_stateful_replanning --steps 10 --dt 0.2
```
