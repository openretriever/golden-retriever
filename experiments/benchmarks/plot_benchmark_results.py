"""
Script to plot benchmarking results.

This by default looks at .csv files found in the experiments/benchmark/results folder.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Find all CSV files under the results folder.
current_file = Path(__file__).resolve()
current_dir = current_file.parent
results_dir = current_dir / "results"
csv_files = list(results_dir.glob("*.csv"))

# Plot the results for each file.
plt.figure()

legends = []
for file in csv_files:
    df = pd.read_csv(file)
    plt.plot(df["size"], df["latency_ns"] / 1000.0, marker="o")
    legends.append(df["name"][0])

ax = plt.gca()
plt.xscale("log")
exponents = [6, 8, 12, 16, 20, 24]
ticks = 2 ** np.array(exponents)
ax.set_xticks(ticks)
ax.set_xticklabels([fr"$2^{{{e}}}$" for e in exponents])

plt.xlabel("Message Size (bytes)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Message Size")
plt.legend(legends)
plt.grid(True, which="both")
plt.show()
