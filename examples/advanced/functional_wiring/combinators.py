"""
FRP Combinators Example - Single-Expression Graph Building

Demonstrates Arrow-style combinators for building dataflow graphs in one expression:
- `>>` (compose): Sequential connection
- `&` (fanout): Split signal to multiple destinations

Examples:
    source >> detector                    # Linear chain
    source >> (detector & logger)         # Fan-out to 2
    source >> (a & b & c)                 # Fan-out to 3
    source.fanout(detector, logger, recorder)  # Method-based fanout
"""
from dataclasses import dataclass
from retriever.flow import Flow, Rate
from retriever import run
from retriever.flow.io import flow_io
from retriever.flow.pipeline import default_pipeline

# -----------------------------------------------------------------------------
# I/O Types
# -----------------------------------------------------------------------------
@flow_io
@dataclass
class SensorData:
    value: float
    seq: int

@flow_io
@dataclass
class SensorIn:
    value: float
    seq: int

@flow_io
@dataclass
class AlertResult:
    alert: bool

# -----------------------------------------------------------------------------
# Flows
# -----------------------------------------------------------------------------
class Sensor(Flow[None, SensorData]):
    """Simulated sensor producing readings."""
    def init_config(self):
        return {}
    
    def run(self, _) -> SensorData:
        import time
        import random
        if not hasattr(self, 'seq'):
            self.seq = 0
        self.seq += 1
        value = random.uniform(0, 100)
        print(f"Sensor: {value:.1f} (seq={self.seq})")
        return SensorData(value=value, seq=self.seq)

class Detector(Flow[SensorIn, AlertResult]):
    """Detects high values."""
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
    
    def init_config(self):
        return {"threshold": self.threshold}
    
    def run(self, input: SensorIn) -> AlertResult:
        if input.value is None:
            return AlertResult(alert=False)
        detected = input.value > self.threshold
        if detected:
            print(f"  Detector: ALERT! {input.value:.1f} > {self.threshold}")
        return AlertResult(alert=detected)

class Logger(Flow[SensorIn, None]):
    """Logs all readings."""
    def init_config(self):
        return {}
    
    def run(self, input: SensorIn) -> None:
        if input.value is not None:
            print(f"  Logger: recorded seq={input.seq}")
        return None

class Recorder(Flow[SensorIn, None]):
    """Simulates recording to file."""
    def init_config(self):
        return {}
    
    def run(self, input: SensorIn) -> None:
        if input.value is not None:
            print(f"  Recorder: saved {input.value:.1f}")
        return None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== FRP Combinators Example ===")
    print("Demonstrating single-expression graph building\n")
    
    # Create flows
    sensor = Sensor() @ Rate(2.0)
    detector = Detector(60.0) @ Rate(2.0)
    logger = Logger() @ Rate(2.0)
    recorder = Recorder() @ Rate(2.0)
    
    # Use default pipeline context
    with default_pipeline():
        # ---------------------------------------------------------------------
        # Style 1: Classic FRP Fanout using `&` operator
        # ---------------------------------------------------------------------
        print("Graph: sensor >> (detector & logger & recorder)")
        print("       sensor splits to all 3 destinations\n")
        
        # Single expression builds entire graph!
        sensor >> (detector & logger & recorder)
        
        # Equivalent to:
        #   sensor.then(detector)
        #   sensor.then(logger)
        #   sensor.then(recorder)
        
        # Or using the method:
        #   sensor.fanout(detector, logger, recorder)
    
    print("Running pipeline for 2s...")
    run(duration=2.0)

if __name__ == "__main__":
    main()
