"""
Chaining Example - Sequential Flow Composition

Demonstrates FRP-style chaining: source.then(processor).then(logger)
This builds a pipeline in a single expression.
"""
from dataclasses import dataclass
from retriever.flow import Flow, Rate
from retriever import run
from retriever.flow.io import io

# -----------------------------------------------------------------------------
# I/O Types
# -----------------------------------------------------------------------------
@io
@dataclass
class NumberOut:
    value: int

@io
@dataclass  
class NumberIn:
    value: int

@io
@dataclass
class ProcessedOut:
    result: int
    tag: str

@io
@dataclass
class ProcessedIn:
    result: int
    tag: str

# -----------------------------------------------------------------------------
# Flows
# -----------------------------------------------------------------------------
class Counter(Flow[None, NumberOut]):
    """Source flow that counts up."""
    def init_config(self):
        return {}
    
    def run(self, _) -> NumberOut:
        if not hasattr(self, 'count'):
            self.count = 0
        self.count += 1
        print(f"Counter: {self.count}")
        return NumberOut(value=self.count)

class Doubler(Flow[NumberIn, ProcessedOut]):
    """Processor that doubles input value."""
    def init_config(self):
        return {}
    
    def run(self, input: NumberIn) -> ProcessedOut:
        if input is None or input.value is None:
            return None
        result = input.value * 2
        print(f"Doubler: {input.value} -> {result}")
        return ProcessedOut(result=result, tag="doubled")

class Logger(Flow[ProcessedIn, None]):
    """Sink flow that logs received values."""
    def init_config(self):
        return {}
    
    def run(self, input: ProcessedIn) -> None:
        if input is None:
            print("Logger: received None")
            return None
        print(f"Logger: received {input.result} ({input.tag})")
        return None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=== Chaining Example ===")
    print("Building: Counter >> Doubler >> Logger\n")
    
    # Create flow handles with clocks
    counter = Counter() @ Rate(2.0)   # 2 Hz
    doubler = Doubler() @ Rate(2.0)   # 2 Hz
    logger = Logger() @ Rate(2.0)     # 2 Hz
    
    # Use default pipeline context for top-level scripts
    from retriever.flow.pipeline import default_pipeline
    with default_pipeline():
        # Chain them together in one expression!
        # This is equivalent to:
        #   counter.then(doubler)
        #   doubler.then(logger)
        counter.then(doubler).then(logger)
        
        # Alternative syntax using >> operator:
        # counter >> doubler >> logger
    
    print("Running pipeline for 2s...")
    run(duration=2.0)

if __name__ == "__main__":
    main()
