# Webcam + Rerun Perception Demo

This example demonstrates how to build a real-time perception pipeline using `retriever` and visualize the results in [Rerun](https://rerun.io/).

It integrates:
- **Source**: Webcam input (or mock fallback).
- **Perception**: 
    - [OwlViT v2](https://huggingface.co/google/owlv2-base-patch16-ensemble) for Open-Vocabulary Object Detection.
    - [SAM (Segment Anything)](https://huggingface.co/facebook/sam-vit-base) for segmentation based on detection boxes.
- **Visualization**: `rerun-sdk` to log images, bounding boxes, and masks.

## Dependencies

This example requires the `rerun-sdk`, `scipy`, and `transformers` packages, which are handled by `pixi`.

## Usage

Run the example using the defined `pixi` task:

```bash
pixi run demo-webcam-rerun
```

This will:
1. Download the models (if not present).
2. Start the Rerun Viewer.
3. Start the pipeline.

### Arguments

You can pass arguments to the script explicitly if running via python (or editing the task):

- `--queries`: Comma-separated list of text queries for detection. Default: `"person,face,cell phone"`.
- `--cleanup`: **Important**: Deletes the downloaded model checkpoints (Hugging Face cache) for OwlViT and SAM after the script exits. Use this to save disk space if you don't plan to run it frequently.

Example with arguments:
```bash
pixi run python examples/advanced/webcam_rerun/app.py --queries "cup,keyboard" --cleanup
```

## Troubleshooting

- **Empty Image Warnings**: You might see "Received empty image" warnings during the first few seconds of startup while the webcam initializes. This is normal.
- **Model Download**: The first run will take time to download the models (~600MB).

## Record + Replay with MCAP

This folder also includes `record_replay.py` which demonstrates the MCAP recording workflow:

```bash
# Record 50 steps to MCAP
pixi run python examples/advanced/webcam_rerun/record_replay.py record --steps 50

# View in Rerun
pixi run python examples/advanced/webcam_rerun/record_replay.py view
# Or: retriever.view("webcam_session.mcap")

# Record with LIVE Rerun visualization
pixi run python examples/advanced/webcam_rerun/record_replay.py record --stream

# Replay from MCAP (code-level replay)
pixi run python examples/advanced/webcam_rerun/record_replay.py replay
```

### API

```python
# In code
pipe.record("session.mcap", steps=50)           # Save only
pipe.record("session.mcap", steps=50, visualize=True)  # Save + Rerun
pipe.view("session.mcap")                               # View in Rerun
retriever.view("session.mcap")                          # Global version
```

### Commands

| Command | Purpose |
|---------|---------|
| `record` | Capture camera frames to MCAP |
| `view` | Open MCAP in Rerun viewer |
| `replay` | Re-run pipeline from recorded data |
| `record --stream` | Record + live Rerun visualization |

