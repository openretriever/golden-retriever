# Retriever-Examples

OpenPI, Libero, and Controller Integration Package

This package provides a unified interface for:
- **OpenPI**: Core physical intelligence models and policies
- **Libero**: Simulation environment and benchmark suite  
- **OpenPI Controller**: High-level controller abstractions

## Installation

### Prerequisites
```bash
# Activate the retriever conda environment
conda activate retriever
```

### Basic Installation
```bash
# Navigate to the package directory
cd /mnt/arc/yygx/pkgs_baselines/Retriever/retriever_examples/pi0

# Install the package
pip install -e .
```

### With Specific Components
```bash
# OpenPI only
pip install -e .[openpi]

# Libero only  
pip install -e .[libero]

# Everything
pip install -e .[all]
```

## Quick Start

### Command Line Usage
```bash
# Navigate to the package directory
cd /mnt/arc/yygx/pkgs_baselines/Retriever/retriever_examples/pi0

# Activate environment
conda activate retriever

# Run Libero evaluation with OpenPI controller
python examples/libero_demo.py --controller openpi --num-tasks 1 --num-trials 3 --max-steps 100

# Run with mock controller (no dependencies)
python examples/libero_demo.py --controller mock --num-tasks 1 --num-trials 3 --max-steps 50

# Run simple integration demo
python examples/integration_demo.py
```

### Python API Usage
```python
from openpi_controller.flows import OpenPIControllerFlow, MockControllerFlow
from openpi_controller.types import RobotObservation, RobotAction
from libero.envs import OffScreenRenderEnv

# Initialize controller
controller = OpenPIControllerFlow()

# Use with Libero environment
env = OffScreenRenderEnv(task_name="KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet")
```

## Structure

- `openpi/`: Core OpenPI library
- `libero/`: Libero simulation environment
- `openpi_controller/`: Controller abstractions and utilities
  - `flows/`: Controller flow implementations
  - `types/`: Data type definitions
  - `converters/`: Format conversion utilities
  - `inference/`: Inference wrappers
- `examples/`: Usage examples and demos

## Examples

### Command Line Examples
```bash
# Full evaluation with OpenPI (recommended for high success rate)
python examples/libero_demo.py --controller openpi --num-tasks 1 --num-trials 3 --max-steps 100

# Quick test with mock controller
python examples/libero_demo.py --controller mock --num-tasks 1 --num-trials 1 --max-steps 10 --no-videos

# Random controller test
python examples/libero_demo.py --controller random --num-tasks 1 --num-trials 3 --max-steps 50
```

### Available Controllers
- `openpi`: Uses actual OpenPI Pi0 model (requires checkpoint)
- `mock`: Sinusoidal movement patterns for testing
- `random`: Random bounded actions for testing

### Command Line Arguments
- `--controller`: Controller type (openpi, mock, random)
- `--num-tasks`: Number of tasks to evaluate
- `--num-trials`: Number of trials per task
- `--max-steps`: Maximum steps per episode (100 recommended for OpenPI)
- `--no-videos`: Disable video recording
- `--task-suite`: Task suite (libero_10, libero_90, etc.)

## Requirements

- Python >= 3.8
- Conda environment: `retriever`
- See `requirements-*.txt` for component-specific dependencies

## Notes

- **High Success Rate**: Use `--max-steps 100` for OpenPI controller to achieve high success rates
- **Videos**: Videos are saved to `examples/openpi/videos/mock_test/` by default
- **Checkpoints**: OpenPI checkpoints should be available at `/mnt/arc/yygx/pkgs_baselines/openpi/checkpoints/`
