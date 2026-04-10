# LLM Streaming RL - 20 Questions

A text RL example showing streaming LLM output, turn-based state updates, and Retriever-based closed-loop execution.

## Quick Start

```bash
# Mock mode, no API keys required
pixi run -e llm demo-llm-streaming --mock

# OpenAI streaming
OPENAI_API_KEY=... pixi run -e llm demo-llm-streaming

# Batch mode (no token streaming)
OPENAI_API_KEY=... pixi run -e llm demo-llm-streaming --no-stream
```

## Direct command

```bash
pixi run -e llm python -m examples.advanced.llm_streaming_rl.app --backend multiprocessing --mock
```

## Notes

- The app defaults to `multiprocessing` and that is the recommended local path.
- Use `--backend dora` only when you specifically want to exercise the distributed backend.
- Rerun visualization is optional; pass `--no-rerun` to keep the demo purely terminal-based.
