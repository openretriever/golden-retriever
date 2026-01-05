"""
Verification: FRP Combinators

Tests the high-level FRP combinators.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import time
from retriever.flow.types import constant_behavior, switch_behavior, Behavior
from retriever.flow.types import EventStream

def main():
    print("🚀 Verification: FRP Combinators")
    
    # 1. Constant
    b_const = constant_behavior(42)
    assert b_const.at(0) == 42
    print("✅ constant_behavior working")
    
    # 2. Switch
    b_true = constant_behavior("A")
    b_false = constant_behavior("B")
    
    # Control behavior toggles based on time check
    def control_fn(t): 
        return t > 10.0
    
    b_control = Behavior(control_fn)
    b_switch = switch_behavior(b_control, b_true, b_false)
    
    val_before = b_switch.at(5.0)
    val_after = b_switch.at(15.0)
    
    print(f"✅ switch_behavior: at 5.0='{val_before}', at 15.0='{val_after}'")
    assert val_before == "B"
    assert val_after == "A"
    
    print("✅ All FRP combinators verified!")

if __name__ == "__main__":
    main()
