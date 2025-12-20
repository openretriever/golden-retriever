"""Dora-rs integration for high-performance robotics dataflow.

This package exposes multiple modules. To avoid importing heavy optional
dependencies (like PyYAML or PyArrow) at package import time, we keep imports
lazy. Accessing attributes will import underlying modules as needed.
"""

__all__ = [
    "DoraExecutor",
    "FlowToDoraConverter",
    "PipelineToYamlConverter",
    "ArrowMessageSerializer",
]

def __getattr__(name):
    if name == "DoraExecutor":
        from .executor import DoraExecutor
        return DoraExecutor
    if name in ("FlowToDoraConverter", "PipelineToYamlConverter"):
        from .converters import FlowToDoraConverter, PipelineToYamlConverter
        return {"FlowToDoraConverter": FlowToDoraConverter,
                "PipelineToYamlConverter": PipelineToYamlConverter}[name]
    if name == "ArrowMessageSerializer":
        from .serialization import ArrowMessageSerializer
        return ArrowMessageSerializer
    raise AttributeError(name)
