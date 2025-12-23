# Code as Policies (CaP) Example

This example demonstrates how to implement "Code as Policies" (CaP) using `retriever`. It allows an LLM to control a robot by generating executable Python code, which is executed safely in a separate thread to maintain the responsiveness of the main application loop.

## Overview

The example consists of:
- **`env.py`**: A simple 2D Tabletop simulation environment.
- **`agent.py`**: An LLM agent (using Google Gemini) that translates natural language into Python code.
- **`executor.py`**: A threaded executor that runs the generated code and bridges blocking API calls to the main event loop.
- **`flows.py`**: Retriever Flows that wrap the environment and the policy/executor.

## Usage

### Prerequisites
1.  **Gemini API Key**: You need a valid Google Gemini API key.
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```
    *Note: If no key is provided, the agent runs in **Mock Mode**, executing a hardcoded script for demonstration.*

2.  **Dependencies**: Uses standard `retriever` environment.

### Running the Demo

Run the application with a task description:

```bash
# Default (uses gemini-robotics-er-1.5-preview)
pixi run -e ravens python -m examples.advanced.code_as_policies.app --task "Put the block in the bowl"

# Use free tier model
pixi run -e ravens python -m examples.advanced.code_as_policies.app --model gemini-2.0-flash-exp --task "Put the block in the bowl"
```
*Note: The default model `gemini-robotics-er-1.5-preview` is optimized for spatial reasoning.*

### How it Works

1.  **Prompting**: The `CodeGenAgent` constructs a prompt with the available API (`pick`, `place`, `move_to`) and the current scene objects.
2.  **Generation**: The LLM generates a Python script (e.g., `pick("red_block"); place(blue_pos)`).
3.  **Execution**: The `PolicyExecutor` runs this script in a background thread.
4.  **Bridging**: When the script calls `pick()`, the executor pauses and sends a request to the `CodePolicyFlow` running in the main thread.
5.  **Action**: The Flow processes the request, updates the environment, and signals the executor to continue.

## Structure

- `app.py`: Main entry point.
- `flows.py`: Flow definitions.
- `executor.py`: Threaded execution logic.
- `agent.py`: LLM client.
- `prompts.py`: System prompt and API docs.
- `env.py`: Simulation logic.

## TODO

- [ ] Clean up `ravens_wrapper.py` lint warnings (trailing whitespace, unused imports)
- [ ] Update `flows.py` to use `step()` instead of `run()` (new Flow API)
- [ ] Extract RAVENS environment into self-contained `ravens/` subdirectory (see below)

### RAVENS Reorganization Plan

**Goal:** Move minimal RAVENS code from `golden_retriever` into this example.

**Proposed structure:**
```
examples/advanced/code_as_policies/
├── ravens/                    ← [NEW] Self-contained module
│   ├── environment.py         ← Core sim (UR5, PyBullet, IK)
│   ├── task.py                ← Base task class
│   ├── put_block_in_bowl.py   ← Task definition
│   ├── primitives.py          ← PickPlace primitive
│   ├── grippers.py            ← Suction gripper
│   ├── cameras.py             ← Camera configs  
│   ├── utils.py               ← Transform utilities
│   └── assets/                ← URDF/mesh files (symlink or subset)
├── ravens_wrapper.py          ← Updated imports
└── ...
```

**Files to move:**

| Source | Destination |
|--------|-------------|
| `envs/environment.py` | `ravens/environment.py` |
| `tasks/task.py` | `ravens/task.py` |
| `tasks/put_block_in_bowl.py` | `ravens/put_block_in_bowl.py` |
| `tasks/primitives.py` | `ravens/primitives.py` |
| `tasks/grippers.py` | `ravens/grippers.py` |
| `tasks/cameras.py` | `ravens/cameras.py` |
| `utils/cliport/utils.py` | `ravens/utils.py` |
| `envs/assets/` | `ravens/assets/` (symlink) |
