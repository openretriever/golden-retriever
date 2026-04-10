# VLM GridWorld Navigation

A visual closed-loop control demo where a VLM (or mock policy) receives rendered grid observations and chooses actions.

## Quick Start

```bash
# Mock mode
pixi run -e llm demo-vlm-gridworld --mock

# Gemini
GEMINI_API_KEY=... pixi run -e llm demo-vlm-gridworld --client gemini

# OpenAI
OPENAI_API_KEY=... pixi run -e llm demo-vlm-gridworld --client openai
```

## Direct command

```bash
pixi run -e llm python -m examples.advanced.vlm_gridworld.app --backend multiprocessing --mock
```

## What it demonstrates

- image observations flowing through a Retriever pipeline
- a VLM-powered agent policy with a mock fallback
- optional Rerun visualization of the environment and reasoning stream

The recommended local backend is `multiprocessing`. Switch to `--backend dora` only if you are explicitly testing the distributed runtime.
