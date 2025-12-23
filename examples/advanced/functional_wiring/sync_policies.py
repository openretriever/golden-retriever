from dataclasses import dataclass
from retriever.flow import Flow, Rate
from retriever import run
from retriever.flow.adapter import Window
from retriever.flow.io import flow_io

@flow_io
@dataclass
class FastOutput:
    val: float

class FastSource(Flow[None, FastOutput]):
    def init_config(self):
        return {}

    def execution(self, _):
        # We need internal state. Flow is stateless by default unless we assign to self.
        if not hasattr(self, 'count'):
            self.count = 0.0
        
        out = FastOutput(val=self.count)
        self.count += 1.0
        return out

    def run(self, input):
        # The runtime calls run(), which calls execution() in some mental model,
        # but here we just implement run().
        return self.execution(input)

@flow_io
@dataclass
class WindowInput:
    val: float # or List[float] if windowed?

# When using Window adapter, the input type in the handler receives the aggregated value.
# Window(agg='mean') -> float
@flow_io
@dataclass
class WindowFusionOut:
    mean: float

class WindowFusion(Flow[WindowInput, WindowFusionOut]):
    def init_config(self):
        return {}
        
    def run(self, input: WindowInput) -> WindowFusionOut:
        # Note: input fields are populated by the runtime/adapter.
        # If 'val' is windowed, 'input.val' should be the aggregated result.
        if hasattr(input, 'val'):
            print(f"Fusion received window mean: {input.val}")
            return WindowFusionOut(mean=input.val)
        return WindowFusionOut(mean=0.0)

def main():
    # 1. Setup Flows
    # Source runs at 10Hz
    fast = FastSource() @ Rate(10.0)
    
    # Fusion runs at 1Hz
    fusion = WindowFusion() @ Rate(1.0)
    
    # 2. Wire with Functional API
    # We want to connect fast.val -> fusion.val
    # Using a Window(0.5s, mean) policy.
    # fast.val is float. fusion.val is float (mean).
    # The adapter transforms Stream[float] -> Window -> float.
    
    print("Wiring with Window(duration=0.5, agg='mean')...")
    # Note: we must map explicit fields if names differ, or use Same names.
    # FastOutput has 'val', WindowInput has 'val'.
    
    fusion(val=fast, sync=Window(buffer_size=20, duration=0.5, agg="mean"))
    
    # 3. Run
    print("Running pipeline for 2.2s...")
    run(duration=2.2)

if __name__ == "__main__":
    main()
