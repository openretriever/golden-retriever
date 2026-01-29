"""
Benchmarking script for Retriever.

Backends are configurable with the --backend command-line argument.
"""

import argparse
from dataclasses import dataclass
import sys
import time
import os

import csv
import numpy as np

from retriever.flow import Flow, flow_io, Rate, Trigger, Pipeline
from retriever.flow.adapter import Latest

# NOTE: SIZES defines the number of uint64 elements in the payload.
# Since each uint64 is 8 bytes, the actual payload size in bytes is: Element Count * 8.
# Examples:
#   i=6  -> 2^6  = 64 elements   -> 512 Bytes
#   i=20 -> 2^20 = 1M elements   -> 8 MB
#   i=24 -> 2^24 = 16M elements  -> 128 MB
SIZES = [2**i for i in range(6, 25)] 
NUM_POINTS_PER_SIZE = 100
DATA_RATE_S = 0.05  # seconds

NAME = "Retriever"
PLATFORM = "COMPUTER_PERF"
LATENCY = True

p = argparse.ArgumentParser(
    description="Perception demo (camera -> detection -> display)"
)
p.add_argument("--backend", default="dora", choices=["dora", "multiprocessing", "in-process"])
p.add_argument("--duration", type=float, default=120.0)
args = p.parse_args()


@flow_io
@dataclass
class RandomSequence:
    data: np.ndarray


class SourceFlow(Flow[None, RandomSequence]):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 0

    def run(self, _):
        random_data = np.array(
            np.random.randint(255, size=SIZES[self.i], dtype=np.uint64)
        )
        random_data[0] = time.perf_counter_ns()
        if self.j == NUM_POINTS_PER_SIZE:
            self.i += 1
            self.j = 0
            if self.i >= len(SIZES):
                print("Benchmarking data collection complete!")
                sys.exit(0)
        else:
            self.j += 1

        return RandomSequence(data=random_data)


class SinkFlow(Flow[RandomSequence, None]):
    def __init__(self):
        super().__init__()
        self.latencies = []
        self.current_size = 0
        self.n = 0

    def run(self, input: RandomSequence):
        t_received = time.perf_counter_ns()
        length = len(input.data) * 8  # As it is Uint64
        if length != self.current_size:
            if self.n > 0:
                self.record_results([], self.current_size, self.latencies, LATENCY)
            self.current_size = length
            self.n = 0
            self.latencies = []
        t_send = int(input.data[0])
        self.latencies.append((t_received - t_send) / 1000.0)
        self.n += 1

    def record_results(self, start, current_size, latencies, latency):
        csv_file = f"experiments/benchmarks/results/retriever_{args.backend}_benchmark_results.csv"
        append = os.path.isfile(csv_file)
        log_header = ["name", "platform", "size", "latency_ns"]
        log_row = [f"{NAME} {args.backend}", PLATFORM, current_size, latencies]
        if append:
            with open(csv_file, "a", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(log_row)
        else:
            with open(csv_file, "w+", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(log_header)
                w.writerow(log_row)


def main():
    pipe = Pipeline("retriever_benchmarking_dora")
    with pipe:
        source = SourceFlow() @ Rate(hz=1.0 / DATA_RATE_S)
        sink = SinkFlow() @ Trigger("data")
        source >> sink  # TODO: Why does this go back to using dora backend?
        # pipe.connect(source, sink, sync=Latest(), qsize=1)

    pipe.run(backend=args.backend, duration=args.duration, blocking=True)


if __name__ == "__main__":
    main()
