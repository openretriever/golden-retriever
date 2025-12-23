"""
Canonical Type Fusion Example.

Demonstrates how to fuse multiple inputs of the same type into a single `List[T]` input port.
This allows clean "Fan-In" without custom adapter nodes.

Usage:
    python examples/basic/fusion.py
"""

from dataclasses import dataclass
from typing import List
import logging

import retriever
from retriever import Flow, Latest, Rate
from retriever.flow import flow_io

# Configure logging
logging.basicConfig(level=logging.INFO)

@flow_io
@dataclass
class IntData:
    """Standard integer data packet."""
    val: int

@flow_io
@dataclass
class FusionInput:
    """
    Fusion Flow Input.
    
    Accepts a list of integers from multiple sources.
    The runtime automatically aggregates multiple connections into this list.
    """
    values: List[int]

class Source(Flow[None, IntData]):
    """Generates an integer value."""
    def __init__(self, val: int):
        self.val = val
        
    def init_config(self):
        return {"val": self.val}

    def run(self, inp) -> IntData:
        return IntData(val=self.val)

class Fusion(Flow[FusionInput, None]):
    """Fuses multiple integer inputs."""
    def run(self, inp: FusionInput):
        # inputs are automatically aggregated into a list
        print(f"Fusion received: {inp.values} (Sum: {sum(inp.values)})")

def main():
    # 1. Create Flows
    # We use @ Rate to set execution frequency
    s1 = Source(10) @ Rate(1.0)
    s2 = Source(20) @ Rate(1.0)
    s3 = Source(30) @ Rate(1.0)
    
    fusion = Fusion() @ Rate(1.0)
    
    # 2. Connect Flows
    # Connect all sources to the SAME input port 'values'.
    # map={"val": "values"} means:
    #   Source output 'val' (int) -> Fusion input 'values' (List[int])
    # The runtime detects the list type and allows multiple connections.
    
    # Option 1: Functional API (Recommended)
    # Implicitly connects s1, s2, s3 to the 'values' list port
    print("Wiring with Functional API...")
    fusion(s1, s2, s3)
    
    # Option 2: Explicit Connection (Legacy/Advanced)
    # retriever.connect(s1, fusion, map={"val": "values"}, sync=Latest())
    # ...
    
    # 3. Run Pipeline
    print("Running Fusion Flow (1.0s)...")
    retriever.run(duration=1.1)

if __name__ == "__main__":
    main()
