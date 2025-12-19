"""
Verification: Migrated Types

This script verifies that the migrated types (Module, FRPConfig) are
accessible and functional in the new location.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from typing import Any
from retriever.types import Module, FRPConfig, Eff, pure

class TestModule(Module[str, int]):
    """Testing Module protocol."""
    def __call__(self, inp: str) -> int:
        return len(inp)

def main():
    print("🚀 Verification: Migrated Types")
    
    # 1. Verify Module protocol
    mod = TestModule()
    val = mod("hello")
    print(f"✅ Module check: 'hello' -> {val}")
    assert val == 5
    assert isinstance(mod, Module)
    
    # 2. Verify FRPConfig (Compat) -> Functional Form
    config = FRPConfig(rate="30hz")
    hz = config.rate_hz()
    clock = config.as_clock()
    print(f"✅ FRPConfig check: 30hz -> {hz}")
    print(f"✅ FRPConfig -> Clock: {clock}")
    
    # Check if clock is correct type (string repr check is sufficient for now)
    assert hz == 30.0
    assert "Rate" in str(clock) and "30.0" in str(clock)
    
    # 3. Verify Eff
    e = pure(10)
    res, _ = e.run(None)
    print(f"✅ Eff check: pure(10) -> {res}")
    assert res == 10
    
    print("\n✅ All types verified!")

if __name__ == "__main__":
    main()
