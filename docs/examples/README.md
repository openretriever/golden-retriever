# Example Guides

These guides stay close to the runnable surfaces under `examples/advanced/`.

They follow one design rule throughout: start from a small shared payload vocabulary from Retriever core, then use structural composition (including composite `Flow[...]` IO) before inventing new named envelopes.

## Start Here

- `perception_and_memory_v1.md`: concise perception, belief, replay, and composition ladder.
- `language_and_grounding_v1.md`: concise caption, grounding, and primitive plan-text ladder.
- `pipeline_composition_v1.md`: newer registry-backed composition surfaces.

## Example Families

- `examples/advanced/perception_examples/`: concise detection, segmentation, and pointing flows.
- `examples/advanced/memory_examples/`: concise belief, dropout-memory, and remembered-pointing flows.
- `examples/advanced/perception_debug/`: deterministic record/replay and windowed stats.
- `examples/advanced/state_management/`: older state and belief-update examples.
- `examples/advanced/core_composition/`: registry-backed pipeline composition.

## Recommended Order

1. `perception_and_memory_v1.md`
2. `language_and_grounding_v1.md`
3. `pipeline_composition_v1.md`
