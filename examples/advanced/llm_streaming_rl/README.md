# LLM Streaming RL - 20 Questions

A text-based reinforcement learning example demonstrating **LLM streaming** integration with Retriever's reactive flow system.

> [!WARNING]
> **Status: Experimental**
> This example is currently under active development. Stability and performance are being verified. Rerun visualization may require manual timeline adjustment.

## Overview

An LLM agent plays "20 Questions" - asking strategic yes/no questions to guess a secret word. The example showcases:

1. **Streaming Token Generation**: Watch LLM tokens appear progressively
2. **Multi-Turn Dialogue**: Stateful conversation with history tracking
3. **RL Environment Pattern**: Observation → Action → Reward loop
4. **EventStream Integration**: Bridging streaming APIs to Retriever's FRP model

```text
┌─────────────────────────────────┐
│  TextEnvFlow                    │
│  (20 Questions Game)            │
│  Answers questions, tracks      │
│  score, resets on game end      │
└─────────────┬───────────────────┘
              │ TextObservation (text, history, reward)
              ▼
┌─────────────────────────────────┐
│  LLMAgentFlow                   │
│  (GPT-4o-mini / Mock)           │
│  Streaming token generation     │
│  Strategic Q&A reasoning        │
└─────────────┬───────────────────┘
              │ LLMAction (question/guess, reasoning)
              ▼
        (back to env)
```

## Quick Start

### Mock Mode (No API Keys)

```bash
pixi run -e llm demo-llm-streaming --mock
```

### With OpenAI (Streaming)

```bash
export OPENAI_API_KEY="your-key-here"
pixi run -e llm demo-llm-streaming
```

### With Batch Inference (No Streaming)

```bash
export OPENAI_API_KEY="your-key-here"
pixi run -e llm demo-llm-streaming --no-stream
```

## Troubleshooting
- **Backend Issues**: This example defaults to the `dora` backend. If you encounter issues, try running with `--backend multiprocessing`.
- **Rerun**: If visualization is empty, check that `Rerun` is installed and running. Use `--no-rerun` to disable it.

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mock` | False | Use simulated LLM (no API) |
| `--model` | gpt-4o-mini | OpenAI model name |
| `--no-stream` | False | Disable streaming output |
| `--max-questions` | 15 | Questions per game |
| `--episodes` | 2 | Number of games to play |
| `--hz` | 0.1 | Turn rate (0.1 = 1 turn per 10s) |
| `--no-rerun` | False | Disable Rerun visualization |
| `--seed` | None | Random seed for reproducibility |

## Streaming Architecture

The example demonstrates two streaming integration patterns:

### 1. Token-by-Token Streaming

```python
for token in client.stream("Ask a question"):
    print(token, end="", flush=True)
```

### 2. Chunked Streaming (Recommended)

```python
for chunk in client.stream_chunks("Ask a question"):
    print(f"[Chunk {chunk.chunk_index}] {chunk.tokens}")
```

The chunked approach integrates better with Retriever's Rate/Trigger system.

### StreamingConfig Options

```python
StreamingConfig(
    chunk_size=5,              # Tokens per chunk
    emit_on_newline=True,      # Emit on newlines
    emit_on_punctuation=True,  # Emit on .!?
    max_buffer_time_ms=500,    # Max buffer time
)
```

## Game Mechanics

### Word Categories

- Animals: elephant, penguin, giraffe, dolphin, butterfly, octopus
- Fruits: apple, banana, strawberry, mango, pineapple, watermelon
- Objects: laptop, umbrella, telescope, bicycle, piano, lighthouse
- Places: mountain, ocean, desert, forest, island, volcano
- Vehicles: helicopter, submarine, motorcycle, sailboat, rocket, train

### Reward Structure
| Event | Reward |
|-------|--------|
| Correct guess | +10.0 + bonus for fewer questions |
| Good question | +0.1 |
| Repeated question | -0.1 |
| Wrong guess | -1.0 |
| Out of questions | -5.0 |

### LLM Strategy Prompt

The agent uses strategic reasoning:

```text
You are playing 20 Questions...

Strategy tips:
- Binary search: divide the possibility space in half
- Track what you've learned from previous answers
- Don't waste questions on unlikely guesses
```

## Key Files

```text
llm_streaming_rl/
├── __init__.py
├── app.py          # Main entry point
├── env.py          # 20 Questions environment
├── flows.py        # Retriever Flow definitions
├── streaming.py    # Streaming LLM utilities
└── README.md       # This file
```

## Key Concepts Demonstrated

1. **LLM Streaming**: Real-time token generation with progressive display
2. **Chunked Emission**: Buffering tokens for smoother Flow integration
3. **Multi-Turn RL**: Stateful environment with conversation history
4. **Strategic Reasoning**: LLM performs binary search over possibility space
5. **Graceful Degradation**: Mock client for testing without API keys

## Example Game Session

```text
🎮 Welcome to 20 Questions!
I'm thinking of a word from the category: **animals**

[LLM Streaming] ask: Is it a mammal?
(Reasoning: Starting with broad category to divide animals)

Q1: Is it a mammal?
A: No
[14 questions remaining]

[LLM Streaming] ask: Does it live in water?
(Reasoning: Non-mammal, checking aquatic vs terrestrial)

Q2: Does it live in water?
A: Yes
[13 questions remaining]

[LLM Streaming] guess: octopus
(Reasoning: Aquatic non-mammal with 8 legs)

🎉 Correct! The word was **octopus**!
```

## Visualization (Rerun)

When Rerun is enabled, you can observe:

- **game/observation**: Current game state text
- **metrics/reward**: Per-turn reward
- **metrics/total_reward**: Cumulative score
- **llm/response**: Full LLM response
- **llm/chunks**: Number of streaming chunks

## Design Notes
For more context on LLM/VLM integration prototypes, see:
- [Design Notes](DESIGN.md)
