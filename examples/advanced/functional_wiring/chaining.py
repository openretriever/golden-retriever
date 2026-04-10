"""
Chaining Example - Sequential Flow Composition

Demonstrates FRP-style chaining: source.then(processor).then(logger)
This builds a pipeline in a single expression.
"""
from dataclasses import dataclass

from retriever.flow import Flow, Pipeline, Rate
from retriever.flow.io import io


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


class Counter(Flow[None, NumberOut]):
    """Source flow that counts up."""

    def step(self, _) -> NumberOut:
        if not hasattr(self, 'count'):
            self.count = 0
        self.count += 1
        print(f"Counter: {self.count}")
        return NumberOut(value=self.count)


class Doubler(Flow[NumberIn, ProcessedOut]):
    """Processor that doubles input value."""

    def step(self, input: NumberIn) -> ProcessedOut:
        if input is None or input.value is None:
            return None
        result = input.value * 2
        print(f"Doubler: {input.value} -> {result}")
        return ProcessedOut(result=result, tag="doubled")


class Logger(Flow[ProcessedIn, None]):
    """Sink flow that logs received values."""

    def step(self, input: ProcessedIn) -> None:
        if input is None:
            print("Logger: received None")
            return None
        print(f"Logger: received {input.result} ({input.tag})")
        return None


def main():
    print("=== Chaining Example ===")
    print("Building: Counter >> Doubler >> Logger\n")

    pipe = Pipeline('chaining_demo')
    with pipe:
        counter = Counter() @ Rate(2.0)
        doubler = Doubler() @ Rate(2.0)
        logger = Logger() @ Rate(2.0)
        counter.then(doubler).then(logger)

    print("Running pipeline for 2s...")
    pipe.run(backend='multiprocessing', duration=2.0)


if __name__ == "__main__":
    main()
