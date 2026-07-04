# Example Guides

These guides stay close to the runnable surfaces under `examples/advanced/`. They are applied examples, not the canonical core runtime manual. If you are new to Retriever, run the core visual quickstart first, then return here for robot-facing examples.

They follow one design rule throughout: start from the generic standard payloads in Retriever core, then put applied robotics/planning payloads in Golden type packs or local example modules before inventing new named envelopes.

## Start Here

- `perception_and_memory_v1.md`: concise perception, belief, replay, and composition ladder.
- `language_and_grounding_v1.md`: concise caption, grounding, and primitive plan-text ladder.
- `pipeline_composition_v1.md`: newer registry-backed composition surfaces.
- `simulation_and_visualization_v1.md`: optional visual/simulator examples after the concise ladder.

## Maintained Example Families

- `examples/advanced/perception_examples/`: concise detection, segmentation, and pointing flows.
- `examples/advanced/memory_examples/`: concise belief, dropout-memory, and remembered-pointing flows.
- `examples/advanced/language_examples/`: caption, grounding, and primitive plan-text examples.
- `examples/advanced/perception_debug/`: deterministic record/replay and windowed stats.
- `examples/advanced/state_management/`: state, reset, and belief-update examples.
- `examples/advanced/core_composition/`: registry-backed pipeline composition surfaces.
- `examples/advanced/webcam_rerun/`: webcam/model/Rerun visualization path.
- `examples/advanced/twist2_simulation/`: MuJoCo/TWIST2 simulator and Rerun path.
- `examples/advanced/mujoco_manipulation/`: MuJoCo manipulation with Rerun logging.
- `examples/advanced/robosuite_lift/`: mock-safe robosuite Lift smoke demo.
- `examples/advanced/hierarchical_physics_demo/`: physics demos with Rerun and HTML pipeline visualization.

## Experimental Utilities And Design Notes

- `examples/advanced/web_command_interface/`: local browser command surface; keep it secondary until it has a README and a packaged task.
- `examples/experimental/visualization/`: small self-contained IR visualization utility; run `pixi run demo-pipeline-html-viz`.
- `examples/advanced/closed_loop_planning/`: extracted design patterns, not a runnable first-run example.

## Recommended Order

Each page should name its runnable command and expected artifact: terminal output, Rerun viewer, HTML pipeline visualization, or mock-safe simulator trace.

1. `perception_and_memory_v1.md`
2. `language_and_grounding_v1.md`
3. `pipeline_composition_v1.md`
4. `simulation_and_visualization_v1.md`
