"""
Fan-Out Example - Signal Splitting

Demonstrates one source feeding multiple destinations.
This is the dual of Fan-In (fusion).

Pattern:
    source ─┬─> detector
            └─> logger
"""
from dataclasses import dataclass
from retriever.flow import Flow, Rate
from retriever import run
from retriever.flow.io import flow_io

# -----------------------------------------------------------------------------
# I/O Types
# -----------------------------------------------------------------------------
@flow_io
@dataclass
class SensorReading:
    value: float
    timestamp: float

@flow_io
@dataclass
class DetReading:
    value: float
    timestamp: float

@flow_io
@dataclass  
class LogReading:
    value: float
    timestamp: float

@flow_io
@dataclass
class DetectionResult:
    detected: bool
    confidence: float

# -----------------------------------------------------------------------------
# Flows
# -----------------------------------------------------------------------------
class Sensor(Flow[None, SensorReading]):
    """Simulated sensor producing readings."""
    def init_config(self):
        return {}
    
    def run(self, _) -> SensorReading:
        import time
        import random
        value = random.uniform(0, 100)
        ts = time.time()
        print(f"Sensor: {value:.1f}")
        return SensorReading(value=value, timestamp=ts)

class Detector(Flow[DetReading, DetectionResult]):
    """Detects if value exceeds threshold."""
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
    
    def init_config(self):
        return {"threshold": self.threshold}
    
    def run(self, input: DetReading) -> DetectionResult:
        # Handle None on first tick (no data yet)
        if input.value is None:
            return DetectionResult(detected=False, confidence=0.0)
        detected = input.value > self.threshold
        confidence = min(1.0, input.value / 100.0)
        if detected:
            print(f"Detector: ALERT! {input.value:.1f} > {self.threshold}")
        return DetectionResult(detected=detected, confidence=confidence)

class DataLogger(Flow[LogReading, None]):
    """Logs all readings to storage."""
    def init_config(self):
        return {}
    
    def run(self, input: LogReading) -> None:
        print(f"Logger: recorded {input.value:.1f} at {input.timestamp:.3f}")
        return None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== Fan-Out Example ===")
    print("Pattern: Sensor --> Detector (for alerts)")
    print("                --> Logger (for recording)\n")
    
    # Create flow handles
    sensor = Sensor() @ Rate(2.0)        # 2 Hz source
    detector = Detector(60.0) @ Rate(2.0) # Alert if > 60
    logger = DataLogger() @ Rate(2.0)     # Log all data
    
    # Use default pipeline context
    from retriever.flow.pipeline import default_pipeline
    with default_pipeline():
        # Fan-Out: sensor feeds BOTH detector AND logger
        # Each .then() creates a separate edge in the graph
        sensor.then(detector)
        sensor.then(logger)
    
    # The graph now looks like:
    #   sensor ─┬─> detector
    #           └─> logger
    
    print("Running pipeline for 3s...")
    run(duration=3.0)

if __name__ == "__main__":
    main()
