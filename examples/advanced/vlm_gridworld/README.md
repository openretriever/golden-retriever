# VLM GridWorld Navigation

A visual reinforcement learning example demonstrating **Vision-Language Model** integration with Retriever's reactive flow system.

> [!WARNING]
> **Status: Experimental**
> This example is currently under active development. VLM reasoning extraction may be inconsistent, and Rerun visualization timelines may require manual adjustment (switch to `log_time`). Full stability is not yet guaranteed.

## Overview

An agent navigates an 8×8 gridworld using visual observations processed by a VLM (GPT-4o or Gemini). The VLM perceives the rendered grid image and outputs navigation actions with reasoning.

```
┌─────────────────────────────────┐
│  GridEnvFlow                    │
│  (Renders image observations)   │
└─────────────┬───────────────────┘
              │ GridObservation (image, position, goal)
              ▼
┌─────────────────────────────────┐
│  VLMAgentFlow                   │
│  (GPT-4o / Gemini / Mock)       │
│  Processes image → outputs      │
│  action + reasoning             │
└─────────────┬───────────────────┘
              │ AgentAction (action, reasoning)
              ▼
        (back to env)
```

## Quick Start

### Mock Mode (No API Keys)

```bash
pixi run -e llm demo-vlm-gridworld --mock
```

### With Gemini (Free Tier)

1. Get API Key: <https://aistudio.google.com/app/apikey>

2. Run with Pixi (Dora backend):

```bash
export GEMINI_API_KEY="your-key-here"
pixi run -e llm demo-vlm-gridworld --client gemini
```

### With OpenAI GPT-4o

```bash
export OPENAI_API_KEY="your-key-here"
pixi run -e llm demo-vlm-gridworld --client openai
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mock` | False | Use heuristic policy (no VLM) |
| `--client` | gemini | VLM provider: `openai` or `gemini` |
| `--model` | auto | Specific model (e.g., `gpt-4o-mini`) |
| `--grid-size` | 8 | Grid dimensions (N×N) |
| `--max-steps` | 50 | Max steps per episode |
| `--duration` | 60 | Run duration in seconds |
| `--hz` | 0.5 | Step rate (0.5 = 1 step per 2s) |
| `--no-rerun` | False | Disable Rerun visualization |

## Architecture

The example demonstrates several Retriever patterns:

### 1. VLM-in-the-Loop Control

The VLMAgentFlow wraps VLM inference as a reactive Flow, enabling closed-loop visual control:

```python
class VLMAgentFlow(Flow[GridObservation, AgentAction]):
    def run(self, obs: GridObservation) -> AgentAction:
        # Encode image → call VLM → parse JSON response
        response = self._vlm_policy(obs)
        return AgentAction(action=response["action"], reasoning=response["reasoning"])
```

### 2. Multi-Rate Execution

- **Environment**: Runs at 0.5 Hz (slow, matches VLM inference time)
- **Visualization**: Triggered by new observations
- **Agent**: Triggered by new images, waits for VLM response

### 3. Graceful Degradation

The agent automatically falls back to mock policy if:
- API key is missing
- VLM call fails
- Response parsing fails

## VLM Prompt

The agent uses this system prompt:

```
You are a navigation agent in a grid world.
Your goal is to reach the green square (goal) from your current position (blue circle).

The grid shows:
- Blue circle: Your current position
- Green square: The goal you need to reach
- Gray trail: Your previous path

Analyze the image and choose the best action to reach the goal efficiently.
Respond ONLY with valid JSON: {"action": "<up|down|left|right>", "reasoning": "..."}
```

## Visualization (Rerun)

When running with Rerun enabled, you can observe:

- **grid/image**: Live grid visualization with agent and goal
- **metrics/reward**: Per-step reward signal
- **metrics/total_reward**: Cumulative episode reward
- **vlm/action**: Current action taken
- **vlm/reasoning**: VLM chain-of-thought explanation

## File Structure

```
vlm_gridworld/
├── __init__.py
├── app.py          # Main entry point
├── env.py          # GridWorld environment
├── flows.py        # Retriever Flow definitions
└── README.md       # This file
```

## Key Concepts Demonstrated

1. **Vision-Language Model as Agent**: VLMs can serve as decision-making agents in visual RL
2. **Image Observation Processing**: Encoding/decoding images for VLM APIs
3. **JSON Action Parsing**: Structured output from VLMs
4. **Closed-Loop Control**: Perception → Decision → Action → Observation cycle
5. **Reactive Dataflow**: Event-driven execution with Triggers

## Troubleshooting

- **Rerun Timeline Empty?**: Switch Rerun viewer timeline to `log_time` or `recording_time`. The `ReasoningLoggerFlow` logs events based on wall-clock time, which may not align with the `step` timeline if steps are sparse.
- **Empty Reasoning?**: The VLM may occasionally fail to output JSON. Check console logs for `[VLMAgent] Raw Gemini Response` to diagnose.
- **Backend**: Defaults to `dora`. If issues arise, try `--backend multiprocessing`.

## Design Notes
For more context on LLM/VLM integration prototypes, see:
- [LLM & VLM Prototypes](../../../../docs/temp_notes/2025-12-22_llm_vlm_prototypes.md)
