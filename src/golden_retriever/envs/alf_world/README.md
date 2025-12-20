# ALFWorld

Benchmark using VLM as a planner in the ALFWorld environment.

---
### Setup
```
conda create -n retriever-alfworld python=3.10
conda activate retriever-alfworld
pip install uv
cd ../../..
# This environment lives in the "golden/system" dependency set.
# Use `pixi-golden.toml` (env: `vlm`) or the future `retriever-golden` repo once split.
# Example (Pixi):
#   pixi install --manifest-path pixi-golden.toml -e vlm
#   pixi run --manifest-path pixi-golden.toml -e vlm python -m pip install -e .
export ALFWORLD_DATA=<storage_path>
alfworld-download
```
---
### Features

- **Agent Support**:
  - **PureVLMAgent**: A vision-and-language agent using only VLM models.
  - **UnawareVLMAgent**: Integrates VLM and LLM models but lacks environmental awareness.
  - **AwareVLMAgent**: Fully integrates VLM and LLM with environmental context awareness.

- **Environment Tasks**:
  - `pick_and_place`
  - `pick_two_obj_and_place`
  - `look_at_obj_in_light`
  - `pick_heat_then_place`
  - `pick_cool_then_place`
  - `pick_clean_then_place`

- **Metrics**:
  - Success rates and goal-condition completion rates per task.

- **Logging**:
  - Supports logging with `wandb` for tracking and visualization.

---

### Directory Structure

```
alf_world/
├── agents/                # Agent classes and logic
├── alf-config.yaml        # Configuration file for ALFWorld
├── eval.py                # Evaluation script
├── utils.py               # Utility functions for environment setup and VLM/LLM integration
├── prompt/                # Task prompts and templates
└── README.md              # Project documentation
```

---

### Script Descriptions

- **`eval.py`**:
  Used to evaluate different agents in predefined tasks using VLM and LLM models. Outputs success metrics and allows for optional logging to `wandb`.

- **`utils.py`**:
  Provides helper functions for environment configuration, string and image processing, and interaction between VLM/LLM models and the ALFWorld framework.

---

### Arguments for `eval.py`

| Argument                  | Type       | Description                                                                                      |
|---------------------------|------------|--------------------------------------------------------------------------------------------------|
| `--alf-config`            | `string`   | Path to the configuration YAML file specifying environment and agent details.                   |
| `--agent-type`            | `string`   | Type of agent to use: `pure_vlm`, `unaware_vlm_llm`, or `aware_vlm_llm`.                        |
| `--vlm-model`            | `string`   | Name of VLM model.
| `--llm-model`            | `string`   | Name of LLM model.
| `--eval-num-per-episode`  | `int`      | Number of evaluation episodes to run per task (default: 134).                                   |
| `--use-wandb`             | `flag`     | Enable logging to Weights & Biases for tracking results.                                        |
| `--wandb-project`         | `string`   | Name of the Weights & Biases project to log results.                                            |
| `--wandb-run`             | `string`   | Run name for Weights & Biases logging.                                                         |
| `--seed`                  | `int`      | Seed value for reproducibility.                                                                |

---

### Usage

#### Evaluation

Run the evaluation script to test an agent on specific tasks:

```bash
python eval.py --alf-config ./alf-config.yaml \
               --agent-type pure_vlm \
               --vlm-model gpt-4o \
               --llm-model gpt-4o
```
