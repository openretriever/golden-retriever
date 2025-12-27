# Design Notes: LLM Streaming RL

## Objective
To demonstrate handling continuous data streams (tokens) within the discrete event-based model of Retriever.

## Streaming Architecture

### The Mismatch
*   **LLMs** produce a stream of tokens: `["H", "e", "l", "l", "o"]`.
*   **Retriever Flows** typically process discrete messages: `msg_in -> msg_out`.

### Solution: Chunked Event Generation
We wrap the LLM client's generator in an adapter that:
1.  Buffers tokens into **StreamChunks** (e.g., size 5, or split by punctuation).
2.  Yields these chunks as individual Flow outputs.

```python
@dataclass
class StreamChunk:
    chunk_index: int
    tokens: str
    is_final: bool
```

### State Management
The `LLMAgentFlow` is **stateful**. It maintains the conversation history (system prompt + user/assistant messages) across turns.
*   **Turn-taking**: The Environment sends a `Trigger`. The Agent generates a full response (multiple chunks) and then waits.
*   **Feedback Loop**: The simplified "20 Questions" environment provides rewards based on the semantic content of the *completed* question (reconstructed from chunks).
