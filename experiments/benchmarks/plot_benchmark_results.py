"""
Script to plot benchmarking results.

This by default looks at .csv files found in the experiments/benchmark/results folder.
"""

import ast
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
lower_percentile = 10
upper_percentile = 90

plt.figure()

legends = []
for file in csv_files:
    df = pd.read_csv(file)
    latency_ns_arrays = df["latency_ns"].apply(ast.literal_eval)
    max_len = max(len(a) for a in latency_ns_arrays)
    latency_ms = np.array(
        [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in latency_ns_arrays]
    ) / 1000.0

    median = np.nanpercentile(latency_ms, 50, axis=1)
    lower = np.nanpercentile(latency_ms, lower_percentile, axis=1)
    upper = np.nanpercentile(latency_ms, upper_percentile, axis=1)
    plt.errorbar(df["size"], median, yerr=[median - lower, upper - median], fmt="o-", capsize=5)
    legends.append(df["name"][0])

ax = plt.gca()
plt.xscale("log")
# exponents = [6, 8, 12, 16, 20, 24]
exponents = [6, 8, 10, 12, 14, 16, 18, 20, 22]
ticks = 2 ** np.array(exponents)
ax.set_xticks(ticks)
ax.set_xticklabels([fr"$2^{{{e}}}$" for e in exponents])
# plt.yscale("log")

plt.xlabel("Message Size (bytes)")
plt.ylabel("Latency (ms)")
plt.title(f"Latency vs Message Size ({lower_percentile}-{upper_percentile} percentile)")
plt.legend(legends)
plt.grid(True, which="both")
plt.show()

plt.savefig("latency_vs_message_size.pdf")