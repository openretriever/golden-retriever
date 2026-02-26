import csv
import os
import platform
from datetime import datetime

import numpy as np
from dora import __version__

LATENCY = True

DATE = str(datetime.now())
LANGUAGE = f"Python {platform.python_version()}"
PLATFORM = platform.platform()
DORA_VERSION = __version__
LOG_HEADER = [
    "Date",
    "Language",
    "Dora Version",
    "Platform",
    "Name",
    "Size (bit)",
    "Latency (μs)",
]


def record_results(name, current_size, latencies):
    # Determine output path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path: experiments/benchmarks/dora_external/dora-rs/py-latency/helper.py
    # Results: experiments/benchmarks/results/
    results_dir = os.path.join(script_dir, "../../../results")
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "dora_benchmark_results.csv")

    append = os.path.isfile(csv_file)
    log_header = ["name", "platform", "size", "latency_ns"]
    # match schema of benchmark_retriever.py
    log_row = [name, PLATFORM, current_size, latencies]

    if append:
        with open(csv_file, "a", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(log_row)
    else:
        with open(csv_file, "w+", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(log_header)
            w.writerow(log_row)
