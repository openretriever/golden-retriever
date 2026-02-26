# 02 Vision Processing (tutorial)

Runtime-aligned vision examples.

For a real camera + detection demo, see `examples/tutorial/009_dora_perception.py`.

## Run

```bash
# Windowed detection stats (deterministic synthetic frames)
pixi run python -m examples.tutorial.02_vision_processing.01_detection_window_stats --backend multiprocessing --duration 3

# Dora backend
pixi run python -m examples.tutorial.02_vision_processing.01_detection_window_stats --backend dora --duration 10
```
