# Skill Switching with Retriever

This folder contains **two implementations** of the same skill-switching logic:

1.  **Classic (Default)**: Uses explicit manual wiring (`approach_packet`, `manipulate_packet`).
    ```bash
    pixi run demo-skill-switching
    ```
2.  **Fan-in (`--fan-in`)**: Uses the new **Fan-in** feature (single `packet` port) to simplify the Arbiter and wiring.
    ```bash
    pixi run demo-skill-switching-fanin
    ```

## 🎯 Objective

To implement a system where a high-level `Commander` can switch the robot's mode, and only the relevant skill executes, without "stale data" or race conditions.

## 🏗 Architecture

The pipeline consists of a single closed-loop graph:

```mermaid
graph TD
    Env[RobotEnv] -->|State| Router[SkillRouter]
    Cmd[Commander] -->|Mode/Target| Router
    
    subgraph Skills
        Router -->|SkillInput (Active=T/F)| App[ApproachSkill]
        Router -->|SkillInput (Active=T/F)| Manip[ManipulateSkill]
    end
    
    App -->|Action (or Idle)| Arbiter[ActionArbiter]
    Manip -->|Action (or Idle)| Arbiter
    
    Arbiter -->|Selected Action| Env
```

### Key Patterns

1.  **Explicit State Signaling**:
    Instead of relying on `None` to indicate inactivity (which can be "sticky" in some runtimes), the `Router` sends a valid `SkillInput` structure to *all* skills every tick.
    -   **Active Skill**: Receives `SkillInput(active=True, ...)`
    -   **Inactive Skills**: Receive `SkillInput(active=False, ...)`

2.  **Idle Action Filtering**:
    Skills that receive `active=False` output a special `RobotAction` with `skill_name="idle"`. The `ActionArbiter` explicitly filters these out, ensuring the robot only receives valid commands from the active skill.

3.  **Consolidated Components**:
    To reduce boilerplate, all granular types (`RobotState`, `SkillInput`) and Flow logic (`Router`, `Skills`) are defined in `components.py`. `main.py` focuses purely on **wiring**.

## 📂 File Structure

- **`main.py`**: The entry point. Defines the `Pipeline`, instantiates flows, and connects them using `pipe.connect()`.
- **`components.py`**: Contains:
    -   **Dataclasses**: `RobotState`, `UserCommand`, `SkillInput` (with `active` flag).
    -   **Flows**:
        -   `RobotEnv`: Simulates 2D holonomic robot + gripper.
        -   `Commander`: Scripted behavior switching (Time-based).
        -   `SkillRouter`: Routes inputs to skills based on mode.
        -   `ApproachSkill`: P-controller for navigation.
        -   `ManipulateSkill`: Logic for opening/closing gripper.
        -   `ActionArbiter`: Priority selector for actions.

## 🚀 Running the Example

From the root of the repository:

```bash
pixi run demo-skill-switching
```

Or manually:

```bash
PYTHONPATH=src python -m examples.advanced.skill_switching.main
```

### Expected Output

You will see the robot transition through modes over 15 seconds:

1.  **t=0-5s**: `APPROACH` (Move to 5,5)
2.  **t=5-8s**: `MANIPULATE` (Toggle gripper)
3.  **t=8-12s**: `APPROACH` (Return to 0,0)
4.  **t>12s**: `IDLE` (No action)

```text
[Router] Input: ... mode='approach' ...
[Arbiter] Selected APPROACH: ...
[Env] t=1.20 State=... Action from approach: vx=1.1 vy=1.1 ...

... (Switching) ...

[Router] Input: ... mode='manipulate' ...
[Arbiter] Selected MANIPULATE: ...
[Env] t=5.10 State=... Action from manipulate: ... grip=False
```
