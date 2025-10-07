#!/usr/bin/env python3
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
