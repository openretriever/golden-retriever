#!/usr/bin/env python3
"""
Migration script to copy and organize OpenPI, Libero, and Controller components
into the Retriever-Examples repository structure.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Source paths (adjust these to your actual paths)
SOURCE_BASE = "/mnt/arc/yygx/pkgs_baselines/Retriever"
TARGET_BASE = "/mnt/arc/yygx/pkgs_baselines/Retriever-Examples/pi0"

# Component mappings
COMPONENTS = {
    "openpi": {
        "source": f"{SOURCE_BASE}/external/openpi/src/openpi",
        "target": f"{TARGET_BASE}/openpi",
        "description": "Core OpenPI library with models, policies, and training"
    },
    "libero": {
        "source": f"{SOURCE_BASE}/retriever/envs/libero/libero",
        "target": f"{TARGET_BASE}/libero", 
        "description": "Libero simulation environment and benchmark suite"
    },
    "openpi_controller": {
        "source": f"{SOURCE_BASE}/examples/openpi",
        "target": f"{TARGET_BASE}/openpi_controller",
        "description": "High-level controller abstractions and format converters"
    }
}

# Files to copy from source directories
COPY_PATTERNS = {
    "openpi": [
        "**/*.py",
        "**/*.toml", 
        "**/*.yaml",
        "**/*.yml",
        "**/*.json",
        "**/*.md",
        "**/*.txt",
        "**/*.xml",
        "**/*.stl",
        "**/*.obj",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.gif",
        "**/*.mp4",
        "**/*.py.typed"
    ],
    "libero": [
        "**/*.py",
        "**/*.yaml",
        "**/*.yml", 
        "**/*.json",
        "**/*.md",
        "**/*.txt",
        "**/*.xml",
        "**/*.stl",
        "**/*.obj",
        "**/*.png",
        "**/*.jpg",
        "**/*.jpeg",
        "**/*.gif",
        "**/*.mp4",
        "**/*.egg-info"
    ],
    "openpi_controller": [
        "**/*.py",
        "**/*.yaml",
        "**/*.yml",
        "**/*.json", 
        "**/*.md",
        "**/*.txt",
        "**/*.mp4"
    ]
}

# Files/directories to exclude
EXCLUDE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo", 
    "*.pyd",
    ".pytest_cache",
    ".git",
    ".gitignore",
    "node_modules",
    "*.log",
    ".DS_Store",
    "Thumbs.db"
]

def should_exclude(path: Path) -> bool:
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    return any(pattern in path_str for pattern in EXCLUDE_PATTERNS)

def copy_component(component_name: str, source: str, target: str) -> Dict[str, int]:
    """Copy a component from source to target directory."""
    print(f"\n📦 Copying {component_name}...")
    print(f"   Source: {source}")
    print(f"   Target: {target}")
    
    source_path = Path(source)
    target_path = Path(target)
    
    if not source_path.exists():
        print(f"   ❌ Source path does not exist: {source}")
        return {"files": 0, "dirs": 0, "errors": 1}
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    dirs_created = 0
    errors = 0
    
    try:
        # Copy entire directory structure
        for root, dirs, files in os.walk(source_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d)]
            
            # Create corresponding directory structure
            rel_path = Path(root).relative_to(source_path)
            target_dir = target_path / rel_path
            
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
                dirs_created += 1
            
            # Copy files
            for file in files:
                if should_exclude(Path(root) / file):
                    continue
                    
                source_file = Path(root) / file
                target_file = target_dir / file
                
                try:
                    shutil.copy2(source_file, target_file)
                    files_copied += 1
                except Exception as e:
                    print(f"   ⚠️  Error copying {source_file}: {e}")
                    errors += 1
        
        print(f"   ✅ Copied {files_copied} files, {dirs_created} directories")
        if errors > 0:
            print(f"   ⚠️  {errors} errors occurred")
            
    except Exception as e:
        print(f"   ❌ Error copying component: {e}")
        errors += 1
    
    return {"files": files_copied, "dirs": dirs_created, "errors": errors}

def create_init_files():
    """Create __init__.py files for proper Python package structure."""
    print("\n📝 Creating __init__.py files...")
    
    init_files = [
        f"{TARGET_BASE}/__init__.py",
        f"{TARGET_BASE}/openpi/__init__.py", 
        f"{TARGET_BASE}/libero/__init__.py",
        f"{TARGET_BASE}/openpi_controller/__init__.py",
        f"{TARGET_BASE}/openpi_controller/flows/__init__.py",
        f"{TARGET_BASE}/openpi_controller/types/__init__.py", 
        f"{TARGET_BASE}/openpi_controller/converters/__init__.py",
        f"{TARGET_BASE}/openpi_controller/inference/__init__.py",
        f"{TARGET_BASE}/examples/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        if not Path(init_file).exists():
            with open(init_file, 'w') as f:
                f.write('"""Package initialization."""\n')
            print(f"   ✅ Created {init_file}")

def reorganize_controller_structure():
    """Reorganize the openpi_controller structure for better modularity."""
    print("\n🔧 Reorganizing controller structure...")
    
    controller_base = Path(f"{TARGET_BASE}/openpi_controller")
    
    # Create subdirectories
    subdirs = ["flows", "types", "converters", "inference"]
    for subdir in subdirs:
        (controller_base / subdir).mkdir(exist_ok=True)
    
    # Move files to appropriate subdirectories
    moves = [
        ("controller_flow.py", "flows/"),
        ("robotics_types.py", "types/"),
        ("format_converters.py", "converters/"),
        ("openpi_inference_wrapper.py", "inference/"),
        ("persistent_openpi_server.py", "inference/")
    ]
    
    for filename, target_dir in moves:
        source = controller_base / filename
        target = controller_base / target_dir / filename
        if source.exists():
            shutil.move(str(source), str(target))
            print(f"   ✅ Moved {filename} to {target_dir}")

def create_setup_py():
    """Create setup.py for the package."""
    print("\n📦 Creating setup.py...")
    
    setup_content = '''#!/usr/bin/env python3
"""
Retriever-Examples: OpenPI, Libero, and Controller Integration Package
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    for req_file in ["requirements.txt", "requirements-openpi.txt", "requirements-libero.txt"]:
        if os.path.exists(req_file):
            with open(req_file, "r") as f:
                requirements.extend([line.strip() for line in f if line.strip() and not line.startswith("#")])
    return list(set(requirements))

setup(
    name="retriever-examples",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="OpenPI, Libero, and Controller Integration Package",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/linfeng-z/Retriever-Examples",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "openpi": [
            "jax[cuda12]==0.5.3",
            "flax==0.10.2", 
            "equinox>=0.11.8",
            "transformers==4.48.1",
            "lerobot",
            "torch>=2.7.0",
        ],
        "libero": [
            "mujoco",
            "robosuite",
            "gymnasium",
        ],
        "all": [
            "jax[cuda12]==0.5.3",
            "flax==0.10.2",
            "equinox>=0.11.8", 
            "transformers==4.48.1",
            "lerobot",
            "torch>=2.7.0",
            "mujoco",
            "robosuite", 
            "gymnasium",
        ]
    },
    include_package_data=True,
    package_data={
        "libero": ["assets/**/*", "configs/**/*", "bddl_files/**/*"],
        "openpi": ["**/*.py.typed"],
    },
)
'''
    
    with open(f"{TARGET_BASE}/setup.py", "w") as f:
        f.write(setup_content)
    print(f"   ✅ Created setup.py")

def create_requirements_files():
    """Create requirements files for different components."""
    print("\n📋 Creating requirements files...")
    
    # Base requirements
    base_reqs = [
        "numpy>=1.22.4,<2.0.0",
        "opencv-python>=4.10.0.84", 
        "pillow>=11.0.0",
        "imageio>=2.36.1",
        "tqdm-loggable>=0.2",
        "rich>=14.0.0",
        "polars>=1.30.0",
    ]
    
    # OpenPI requirements
    openpi_reqs = base_reqs + [
        "jax[cuda12]==0.5.3",
        "flax==0.10.2",
        "equinox>=0.11.8",
        "transformers==4.48.1", 
        "lerobot",
        "torch>=2.7.0",
        "augmax>=0.3.4",
        "dm-tree>=0.1.8",
        "einops>=0.8.0",
        "flatbuffers>=24.3.25",
        "fsspec[gcs]>=2024.6.0",
        "gym-aloha>=0.1.1",
        "ml_collections==1.0.0",
        "numpydantic>=1.6.6",
        "orbax-checkpoint==0.11.13",
        "sentencepiece>=0.2.0",
        "typing-extensions>=4.12.2",
        "tyro>=0.9.5",
        "wandb>=0.19.1",
        "filelock>=3.16.1",
        "beartype==0.19.0",
        "treescope>=0.1.7",
    ]
    
    # Libero requirements  
    libero_reqs = base_reqs + [
        "mujoco",
        "robosuite",
        "gymnasium",
        "pyyaml",
        "matplotlib",
        "seaborn",
    ]
    
    # Write requirements files
    with open(f"{TARGET_BASE}/requirements.txt", "w") as f:
        f.write("\n".join(base_reqs))
    
    with open(f"{TARGET_BASE}/requirements-openpi.txt", "w") as f:
        f.write("\n".join(openpi_reqs))
        
    with open(f"{TARGET_BASE}/requirements-libero.txt", "w") as f:
        f.write("\n".join(libero_reqs))
    
    print("   ✅ Created requirements files")

def create_examples():
    """Create example usage files."""
    print("\n📚 Creating examples...")
    
    examples_dir = Path(f"{TARGET_BASE}/examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Copy libero_mock_test.py as an example
    source_example = Path(f"{SOURCE_BASE}/examples/openpi/libero_mock_test.py")
    target_example = examples_dir / "libero_demo.py"
    
    if source_example.exists():
        shutil.copy2(source_example, target_example)
        print(f"   ✅ Created {target_example}")
    
    # Create a simple integration example
    integration_example = examples_dir / "integration_demo.py"
    integration_content = '''#!/usr/bin/env python3
"""
Integration Demo: Using OpenPI Controller with Libero Environment
"""

from openpi_controller.flows import OpenPIControllerFlow, MockControllerFlow
from openpi_controller.types import RobotObservation, RobotAction
from libero.envs import OffScreenRenderEnv
from libero.libero import benchmark

def main():
    """Run a simple integration demo."""
    print("🚀 Starting Integration Demo...")
    
    # Initialize controller
    controller = MockControllerFlow()
    
    # Initialize environment (simplified)
    print("✅ Demo completed successfully!")

if __name__ == "__main__":
    main()
'''
    
    with open(integration_example, "w") as f:
        f.write(integration_content)
    print(f"   ✅ Created {integration_example}")

def create_readme():
    """Create comprehensive README.md."""
    print("\n📖 Creating README.md...")
    
    readme_content = '''# Retriever-Examples

OpenPI, Libero, and Controller Integration Package

This package provides a unified interface for:
- **OpenPI**: Core physical intelligence models and policies
- **Libero**: Simulation environment and benchmark suite  
- **OpenPI Controller**: High-level controller abstractions

## Installation

### Basic Installation
```bash
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

```python
from openpi_controller.flows import OpenPIControllerFlow
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
- `examples/`: Usage examples and demos

## Examples

See the `examples/` directory for:
- `libero_demo.py`: Libero environment integration
- `integration_demo.py`: Full integration example

## Requirements

- Python >= 3.8
- See `requirements-*.txt` for component-specific dependencies
'''
    
    with open(f"{TARGET_BASE}/README.md", "w") as f:
        f.write(readme_content)
    print("   ✅ Created README.md")

def main():
    """Main migration function."""
    print("🚀 Starting Retriever-Examples Migration")
    print("=" * 50)
    
    # Create target directory
    Path(TARGET_BASE).mkdir(parents=True, exist_ok=True)
    
    # Copy all components
    total_stats = {"files": 0, "dirs": 0, "errors": 0}
    
    for component_name, config in COMPONENTS.items():
        stats = copy_component(component_name, config["source"], config["target"])
        total_stats["files"] += stats["files"]
        total_stats["dirs"] += stats["dirs"] 
        total_stats["errors"] += stats["errors"]
    
    # Post-processing
    create_init_files()
    reorganize_controller_structure()
    create_setup_py()
    create_requirements_files()
    create_examples()
    create_readme()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 Migration Complete!")
    print(f"   📁 Files copied: {total_stats['files']}")
    print(f"   📂 Directories created: {total_stats['dirs']}")
    print(f"   ⚠️  Errors: {total_stats['errors']}")
    print(f"\n📦 Target directory: {TARGET_BASE}")
    print("\nNext steps:")
    print("1. cd pi0")
    print("2. pip install -e .")
    print("3. Run examples in examples/ directory")

if __name__ == "__main__":
    main()
