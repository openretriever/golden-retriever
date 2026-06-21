# Closed-Loop Planning System Notes

## Architecture Overview

This example demonstrates a **Closed-Loop Planning** system using the Retriever framework. It integrates symbolic planning with continuous execution/control, allowing a robot to reason about high-level goals (like "Unlock Door") while reacting to low-level state changes.

### Core Components

1. **Environment (`GridEnvironment`)**:
    - Simulates a 2D grid world.
    - Manages state: Robot position `(x,y)`, `has_key`, `door_open`.
    - Inputs: `Action` (move, pick, unlock).
    - Outputs: `Observation` (raw sensor data).
    - **Loop frequency**: 10Hz.

2. **Perception (`Perception`)**:
    - Translates raw `Observation` into a `State`.
    - Generates predicates like `At(1,1)`, `HasKey()`, `DoorOpen()`.
    - This abstraction allows the Planner to work in a simplified state space.

3. **Planner (`Planner`)**:
    - Inputs: `State`.
    - Outputs: `Plan` (sequence of `Option`s).
    - Logic:
        - Checks if the Goal (`DoorOpen()`) is satisfied in the current state.
        - If not satisfied and no active plan (or failure), generates a new plan.
        - Uses `ParameterizedOption`s (`Move`, `Pick`, `Unlock`) grounded to `Objects`.

### 4. VLM Perception (`VisualPredicate`):
    - Uses **Google Gemini API** (via `google-genai` SDK) to evaluate predicates on images.
    - Example: `VisualPredicate("IsOpen", ...)` sends the image and prompt "Is the door open?" to the VLM.
    - **Note**: Requires `GEMINI_API_KEY` environment variable.

5. **ExecutionMonitor (`ExecutionMonitorFlow`)**:
    - Inputs: `State`, `ExecutorStatus`.
    - Outputs: `ReplanConfig` (flag + reason).
    - Logic:
        - **Global/Slow Loop**: Periodically checks state against goal (e.g., 2Hz).
        - **Event-Driven**: Reacts to `ExecutorStatus` (e.g., "failure").
        - **Signal**: Decides **when** to replan, emitting `ReplanConfig(should_replan=True)`.

#### Relation to Predicators (`CogMan`)

This architecture implements the **`CogMan`** (Cognitive Manager) pattern from Predicators using distributed Flow nodes:
- **`CogMan.perceiver`** $\rightarrow$ `PerceptionFlow` (Geometric + VLM)
- **`CogMan.approach`** $\rightarrow$ `PlannerFlow`
- **`CogMan.execution_monitor`** $\rightarrow$ `ExecutionMonitorFlow`
- **`CogMan.step()`** (Policy Execution) $\rightarrow$ `SkillExecutorFlow`

6. **SkillExecutor (`SkillExecutorFlow`)**:
    - The central coordinator for **Skill Execution**.
    - Inputs: `Plan`, `State`.
    - Outputs: `Action`, `Status`.
    - Logic:
        - **Local/Fast Loop**: Checks `initiable()` and `terminal()` at every step (10Hz+).
        - **Skill Switching**: Iterates through the `Plan`'s options (similar to `skill_switching` example).
        - **Policy Execution**: Runs `policy(state)` for the current option.
        - **Feedback**: Reports status to the Monitor.

### Running the Demo

To run the closed loop planning demo:

```bash
# RISE Sim (VLM + Rerun + Sim Spot)
export GEMINI_API_KEY=your_key_here
pixi run demo-rise-pipeline

# Real Spot Robot
export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=pass
export SPOT_IP=192.168.80.3
pixi run demo-spot-pipeline
```

This uses the `dora` backend by default. You can switch to `backend="python"` in the pipeline script for easier debugging (single process).

### Shared Rerun Architecture

The system uses a **Distributed Rerun Architecture** to visualize data from multiple processes in a single viewer:
1.  **Dora Compiler**: Injects `env_overrides` (like API keys) and `rerun_config` into the generated Dora YAML.
2.  **Runtime**: The main pipeline process acts as the "Rerun Server" (if `spawn=True`).
3.  **Nodes**: Each distributed node connects to the central Rerun instance using `rr.connect()`.
4.  **Result**: A unified timeline showing Plan, Belief, Images, and VLM Reasoning in one dashboard.

### Distributed Execution Monitoring

The system uses a **bilevel monitoring strategy**:

1. **Fast Loop (Executor)**: The Executor runs at the rate of observations (e.g. 10Hz). It performs *local* monitoring:
    - Is the current option still `initiable`?
    - Has the option become `terminal`?
    - This allows immediate reaction to local failures or completion.

2. **Slow Loop (Monitor)**: The Monitor runs at a slower rate (e.g. 2Hz) and on specific events. It performs *global* monitoring:
    - Has the robot deviated significantly from the plan?
    - Did the Executor report a failure?
    - It triggers the expensive Planner only when necessary via `ReplanConfig`.

### Troubleshooting & Debugging

#### 1. Dora Timeout Errors
If you see errors like `Timeout event stream error: Receiver timed out`, this usually means a node (often `Planner` or `Executor`) is not receiving inputs fast enough relative to its poll timeout.
- **Fix**: This is now handled in `src/retriever/rt/backend/dora/executor.py` by ignoring these specific errors. If they persist, verify that `node.next(timeout=...)` is set sufficiently high (e.g., 1.0s).

#### 2. System Hangs / Deadlocks
If the system starts but no components step:
- **Cause**: Likely a cycle dependency where A waits for B, and B waits for A.
- **Fix**: Ensure `GridEnvironment` uses the `"latestornone"` adapter in `main.py`. This allows it to tick even if `Executor` hasn't sent an action yet.

#### 3. No Plans Generated
If `Planner` is stepping but producing empty plans:
*   **Cause**: The state might not technically satisfy the goal conditions, or the `ParameterizedOption` grounding logic failed.
*   **Debug**: Check `planner.py` debug prints or verify `Perception` output atoms.

### 6. Implementation & Troubleshooting Notes

#### VLM Integration
- **SDK**: Migrated to `google-genai` (v1beta) to support Gemini 2.0 Flash.
- **Multiprocessing**: The `genai.Client` is **not picklable**. We use **lazy initialization** in `vlm.py` (initializing the client inside `classifier()` rather than `__init__`) to ensure compatibility with `dora`'s multiprocessing backend.
- **Environment Variables**: The `GEMINI_API_KEY` must be propagated to worker processes. This is handled via `env_overrides` in the Dora compiler.

#### Backend & Visualization
- **Distributed Rerun**: `runtime.py` now supports a `rerun_config`.
    - `spawn=True`: The main pipeline process starts the Rerun Viewer/Server.
    - `connect_addr=...`: Worker nodes (like `PerceptionFlow`) connect to this address.
- **Environment Injection**: The `dora` compiler was updated to accept `env_overrides`. This is crucial for passing API keys to isolated worker processes that don't inherit the full parent shell environment in some configurations.

### Data Flow & Types

The system relies on strong typing defined in `flow_types.py` and `retriever.types.symbolic`:
-   **Data Structures**: `State`, `Action`, `Option`, `Object`, `GroundAtom`.
-   **Flow IO**: Wrappers like `EnvInput`/`EnvOutput` that are decorated with `@io`.
