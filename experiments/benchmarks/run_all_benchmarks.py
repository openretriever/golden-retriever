
import subprocess
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
import sys

# Define paths
dora_benchmark_dir = "experiments/benchmarks/dora_benchmark/dora-rs/py-latency"
results_dir = "experiments/benchmarks/results"
os.makedirs(results_dir, exist_ok=True)
PYTHON = sys.executable  # Ensure we use the same python environment

def run_command(cmd, cwd=None, env=None):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")

def parse_latencies(latency_str):
    try:
        if isinstance(latency_str, str):
            return ast.literal_eval(latency_str)
        return latency_str # Handle if already list (though read_csv makes strings)
    except:
        return []

def plot_results():
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    files = {
        "Retriever (dora)": f"{results_dir}/retriever_dora_benchmark_results.csv",
        "Retriever (mp)": f"{results_dir}/retriever_mp_benchmark_results.csv",
        "dora-rs (native)": f"{results_dir}/dora_benchmark_results.csv"
    }
    
    for label, filepath in files.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"Processing {filepath}...")
        try:
            df = pd.read_csv(filepath)
            
            # Dora-rs legacy format check
            # Our updated node_1.py sends bytes, helper.py logs it.
            
            sizes = []
            medians = []
            p10s = []
            p90s = []
            
            # Using groupby to handle multiple runs/entries
            grouped = df.groupby("size").last().reset_index()
            grouped = grouped.sort_values("size")
            
            for _, row in grouped.iterrows():
                # Handle size units
                # benchmark_retriever.py logs BYTES (length = len(data) * 8 / 8 ? No, it logs len(data)*8 which is BITS?)
                # Let's check benchmark_retriever.py line 84: `length = len(input.data) * 8` (Uint64).
                # `len(input.data)` is number of elements. Each is 8 bytes.
                # So `len(input.data) * 8` is BYTES.
                
                # dora-rs node_2.py line 34: `length = len(data) * 8`.
                # `data` in dora python is `bytes` (pyarrow buffer -> bytes).
                # `len(data)` is bytes.
                # `len(data) * 8` is BITS?
                # Wait, `node_2.py`: `length = len(data) * 8`.
                # If `data` is bytes, then len(data) is bytes.
                # Why Multiply by 8? Maybe legacy code assumed bits?
                # helper.py header says "Size (bit)".
                
                # We want BYTES on x-axis.
                # If benchmark_retriever.py logs bytes (as proper size), we are good.
                # If dora-rs logs bits, we need to divide by 8.
                
                # benchmark_retriever.py:
                # SIZES = [2**6, ...] (64)
                # SinkFlow: `length = len(input.data) * 8`.
                # `data` is numpy array of uint64.
                # `len(data)` is number of elements.
                # `len * 8` is BYTES.
                # So benchmark_retriever logs BYTES. Headers say "size".
                
                # dora-rs node_2.py:
                # `data` is `event["value"]` which is bytes/arrow buffer.
                # `len(data)` is bytes.
                # `length = len(data) * 8`. This is BITS.
                # helper.py logs this as "size".
                # So dora-rs logs BITS.
                
                size_val = row["size"]
                if "dora-rs" in label:
                    size_bytes = int(size_val) / 8
                else:
                    size_bytes = int(size_val)
                
                latencies = parse_latencies(row["latency_ns"])
                if not latencies:
                    continue
                
                # Convert to ms
                latencies_ms = np.array(latencies) 
                # benchmark_retriever writes microseconds?
                # `(t_received - t_send) / 1000.0`. Validated earlier as us?
                # 1ms = 1,000,000 ns.
                # (ns) / 1000.0 = us.
                # So latencies are in MICROSECONDS.
                
                # Helper.py also does `/ 1000`. So also MICROSECONDS.
                
                # Plot usually wants MILLISECONDS.
                latencies_ms = latencies_ms / 1000.0 # us -> ms
                
                # Use Median for robustness
                median = np.median(latencies_ms)
                p10 = np.percentile(latencies_ms, 10)
                p90 = np.percentile(latencies_ms, 90)
                
                sizes.append(size_bytes)
                medians.append(median)
                p10s.append(p10)
                p90s.append(p90)
            
            if sizes:
                # Ensure sizes are sorted and aligned
                # We might have duplicates if not grouped properly? Groupby fixed that.
                
                yerr_lower = np.array(medians) - np.array(p10s)
                yerr_upper = np.array(p90s) - np.array(medians)
                
                # Clamp to 0 just in case
                yerr_lower = np.maximum(yerr_lower, 0)
                yerr_upper = np.maximum(yerr_upper, 0)
                
                plt.errorbar(
                    sizes, 
                    medians, 
                    yerr=[yerr_lower, yerr_upper], 
                    label=label, 
                    marker='o', 
                    capsize=5,
                    alpha=0.7
                )
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()

    plt.xscale("log", base=2)
    plt.xlabel("Message Size (Bytes)")
    plt.ylabel("Latency (ms)")
    plt.title("Median Latency vs Message Size (10-90 percentile)")
    plt.grid(True, which="both", linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    output_path = f"{results_dir}/combined_benchmark_results.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    # 1. Run Retriever (dora)
    print("\n--- Running Retriever (dora) Benchmark ---")
    # Use same python executable
    run_command(f"{PYTHON} experiments/benchmarks/benchmark_retriever.py --backend dora")
    
    # 2. Run Retriever (mp)
    print("\n--- Running Retriever (mp) Benchmark ---")
    run_command(f"{PYTHON} experiments/benchmarks/benchmark_retriever.py --backend mp")
    
    # 3. Run dora-rs (native)
    print("\n--- Running dora-rs (native) Benchmark ---")
    
    # Ensure any previous dora is dead
    subprocess.run("dora destroy", shell=True)
    subprocess.run("dora up", shell=True)
    
    # Run the suite
    # dora start needs to use the python inside the env too?
    # dataflow_node.yml specifies `python: ./node_1.py`. 
    # Dora operator runs this command. It might use system python if not careful.
    # But usually it uses the python from the shebang or the path.
    # node_1.py has `#!/usr/bin/env python`.
    # So if `dora` is run from the env, it *should* work.
    
    run_command("dora start dataflow_node.yml --attach", cwd=dora_benchmark_dir)
    
    # Clean up
    subprocess.run("dora destroy", shell=True)
    
    # 4. Plot
    plot_results()

if __name__ == "__main__":
    main()
