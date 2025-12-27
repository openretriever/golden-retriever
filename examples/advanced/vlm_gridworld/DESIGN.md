# Design Notes: VLM GridWorld

## Objective
To explore patterns for integrating Vision-Language Models (VLM) into a reactive loop for decision making.

## Key Patterns

### 1. VLM-in-the-Loop
We treat the VLM API execution (which is blocking and slow, taking 1-2s) as a standard Flow run step.
*   **Async Isolation**: In `multiprocessing` or `dora` backends, this blocking call happens in a separate process, so it does not freeze the Environment or Visualization.

### 2. Reasoning Extraction
The VLM is prompted to output JSON containing both `action` and `reasoning`.
*   **Robustness**: We implement fallback parsing to handle cases where the VLM wraps JSON in markdown blocks (e.g., ```json ... ```).

### 3. Rerun Integration
We use a dedicated `ReasoningLoggerFlow` to send text logs to Rerun.
*   **Timeline Issue**: VLM steps are sparse (every few seconds). Rerun's default view often follows high-frequency streams. Viewing the `log_time` timeline is often necessary to correlate VLM thoughts with agent movements.
