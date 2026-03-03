"""
Script to plot benchmarking results.

This by default looks at .csv files found in the experiments/benchmark/results folder.
"""

import ast
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Find all CSV files under the results folder.
current_file = Path(__file__).resolve()
current_dir = current_file.parent
results_dir = current_dir / "results"

def parse_latencies(latency_str):
    try:
        if isinstance(latency_str, str):
            val = ast.literal_eval(latency_str)
            if isinstance(val, (list, tuple)):
                return val
        return latency_str 
    except:
        return []

def plot_results():
    print("Plotting results...")
    # Paper-friendly settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'lines.markersize': 5
    })
    
    # Fit for single column (approx 3.5 inches width, but 5 inches gives good resolution for scaling down)
    plt.figure(figsize=(6, 4.5))
    
    # Mapping for known files to cleaner legends
    name_mapping = {
        "retriever_dora_benchmark_results.csv": "Retriever [dora backend]",
        "retriever_multiprocessing_benchmark_results.csv": "Retriever [multiprocessing]",
        "retriever_mp_benchmark_results.csv": "Retriever [multiprocessing]",
        "retriever_in-process_benchmark_results.csv": "Retriever [single-process]",
        "dora_benchmark_results.csv": "dora-rs (native)",
        "ros_cpp_benchmark_results.csv": "ROS2 (C++)",
        "ros_python_benchmark_results.csv": "ROS2 (Python)"
    }
    
    # Define order for plotting (determines legend order)
    # 1. Retriever [single-process]
    # 2. Retriever [dora backend]
    # 3. Retriever [multiprocessing]
    # 4. dora-rs (native)
    priority_order = [
        "retriever_in-process_benchmark_results.csv",
        "retriever_dora_benchmark_results.csv",
        "retriever_multiprocessing_benchmark_results.csv",
        "retriever_mp_benchmark_results.csv",
        "dora_benchmark_results.csv",
        "ros_cpp_benchmark_results.csv",
        "ros_python_benchmark_results.csv"
    ]
    
    def get_sort_key(filepath):
        fname = filepath.name
        if fname in priority_order:
            return priority_order.index(fname)
        return len(priority_order) # Put unknown last

    csv_files = sorted(list(results_dir.glob("*.csv")), key=get_sort_key)
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return

    # csv_files is now sorted, proceed

    has_data = False
    
    # Colors for paper clarity
    colors = {
        "Retriever [dora backend]": "#1f77b4",          # Blue
        "Retriever [multiprocessing]": "#2ca02c", # Green
        "Retriever [single-process]": "#d62728",    # Red
        "dora-rs (native)": "#ff7f0e",          # Orange
        "ROS2 (C++)": "#9467bd",                 # Purple
        "ROS2 (Python)": "#8c564b"               # Brown
    }

    file_data = {}

    for filepath in csv_files:
        filename = filepath.name
        label = name_mapping.get(filename, filename)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            continue
            
        print(f"Processing {filepath}...")
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"File empty: {filepath}")
                continue
            
            # Check if 'size' column exists
            if 'size' not in df.columns:
                print(f"Column 'size' not found in {filepath}. Columns: {df.columns}")
                continue

            # Group by size and take latest run
            grouped = df.groupby("size").last().reset_index()
            grouped = grouped.sort_values("size")
            
            sizes = []
            medians = []
            yerr_lower = []
            yerr_upper = []
            
            
            for _, row in grouped.iterrows():
                size_val = row["size"]
                
                # NOTE: CRITICAL UNIT CLARIFICATION
                # The 'size' column in the CSV files represents BYTES.
                # Source Logic:
                # 1. Benchmark scripts generate 'N' uint64 elements.
                # 2. dora-rs helper records size as `len(data) * 8`. 
                #    Since data is a uint64 array, len is element count.
                #    Element Count * 8 = Total Bytes.
                # Therefore, size_val matches the payload size in Bytes.
                size_bytes = int(size_val)

                if "latency_ns" in row:
                    lat_col = "latency_ns"
                elif "Latency (μs)" in row: # legacy helper format
                    lat_col = "Latency (μs)"
                else:
                    lat_col = df.columns[-1]

                latencies = parse_latencies(row[lat_col])
                if not latencies:
                    continue
                
                # Convert latencies to milliseconds
                # NOTE: Both benchmark_retriever.py and dora-rs/node_2.py divide the 
                # nanosecond difference by 1000 BEFORE logging.
                # So the values in 'latency_ns' column are actually in MICROSECONDS.
                if lat_col == "latency_ns":
                    latencies_ms = [l / 1_000 for l in latencies] # us -> ms
                elif lat_col == "Latency (μs)":
                    latencies_ms = [l / 1_000 for l in latencies]
                else: # Fallback, assume us if it matches others, or ns?
                    # Given the project consistency, assume us -> ms like others
                    latencies_ms = [l / 1_000 for l in latencies]

                # FIXME: ROS2 C++ benchmark results are off by 10^3 (reported in ns instead of us)
                if filename in ["ros_cpp_benchmark_results.csv", "ros_python_benchmark_results.csv"]:
                    latencies_ms = [l / 1_000 for l in latencies_ms]

                median = np.median(latencies_ms)
                p10 = np.percentile(latencies_ms, 10)
                p90 = np.percentile(latencies_ms, 90)

                sizes.append(size_bytes)
                medians.append(median)
                yerr_lower.append(median - p10)
                yerr_upper.append(p90 - median)
            
            if sizes:
                has_data = True
                
                # Filter out outliers for multiprocessing > 64MB as requested
                if "multiprocessing" in label.lower():
                    filtered_indices = [i for i, s in enumerate(sizes) if (s <= 64 * 1024 * 1024)]
                    sizes = [sizes[i] for i in filtered_indices]
                    medians = [medians[i] for i in filtered_indices]
                    yerr_lower = [yerr_lower[i] for i in filtered_indices]
                    yerr_upper = [yerr_upper[i] for i in filtered_indices]
                
                # Sort
                if sizes:
                     sorted_data = sorted(zip(sizes, medians, yerr_lower, yerr_upper))
                     sizes, medians, yerr_lower, yerr_upper = zip(*sorted_data)
                     
                     # Store for later
                     file_data[filename] = (sizes, medians, yerr_lower, yerr_upper)

                # Check for "single-process" to color correctly if not in map
                color = colors.get(label)
                if not color:
                    if "single-process" in label:
                        color = colors["Retriever [single-process]"]
                    elif "in-process" in label: # Fallback
                         color = colors["Retriever [single-process]"]
                
                # Plot immediately (but ticks are calculated later)
                if sizes:
                    # Filter plot data to max 64MB for display
                    display_indices = [i for i, s in enumerate(sizes) if (s <= 64 * 1024 * 1024 and s >= 512)]
                    d_sizes = [sizes[i] for i in display_indices]
                    d_medians = [medians[i] for i in display_indices]
                    d_yerr_lower = [yerr_lower[i] for i in display_indices]
                    d_yerr_upper = [yerr_upper[i] for i in display_indices]
                    
                    if d_sizes:
                         plt.errorbar(
                            d_sizes, 
                            d_medians, 
                            yerr=[d_yerr_lower, d_yerr_upper], 
                            label=label, 
                            marker='o', 
                            capsize=3,
                            alpha=0.8,
                            color=color
                        )
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()

    if has_data:
        # X-axis formatting
        plt.xscale("log", base=2)
        
        # Determine global min/max for ticks
        global_min_size = float('inf')
        global_max_size = 0
        
        # Scan through all data again to find the range
        for filename, label in name_mapping.items(): # Use filename from name_mapping to match file_data keys
             if filename in file_data:
                sizes, _, _, _ = file_data[filename]
                # Filter to max 64MB as requested for plotting range
                valid_sizes = [s for s in sizes if s <= 64 * 1024 * 1024]
                if valid_sizes:
                    global_min_size = min(global_min_size, min(valid_sizes))
                    global_max_size = max(global_max_size, max(valid_sizes))

        if global_max_size > 0:
            min_pow = int(np.floor(np.log2(global_min_size)))
            max_pow = int(np.ceil(np.log2(global_max_size)))
        else:
             min_pow = 6
             max_pow = 26 # Default to 64MB if no data

        ticks = [2**i for i in range(min_pow, max_pow + 1)]
        plt.xticks(ticks)
        
        plt.yscale("log") # Log Y-axis as requested
        
        # Custom label formatter
        def human_readable(x, pos):
            if x >= 1024**2:
                return f'{int(x/1024**2)}MB'
            elif x >= 1024:
                return f'{int(x/1024)}KB'
            else:
                return f'{int(x)}B'
        
        plt.gca().get_xaxis().set_major_formatter(FuncFormatter(human_readable))
        plt.xticks(rotation=45)
        
        plt.xlabel("Message Size")
        plt.ylabel("Latency (ms)")
        # Clean title for paper (often redundant with caption, but good for standalone)
        plt.title("Latency vs Message Size")
        
        # Minor grid for log scale usually looks good
        plt.grid(True, which="major", linestyle='-', linewidth=0.5, alpha=0.8)
        plt.grid(True, which="minor", linestyle=':', linewidth=0.3, alpha=0.5)
        
        plt.legend(frameon=True, fancybox=False, edgecolor='k')
        plt.tight_layout()
        
        output_path = results_dir / "combined_benchmark_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

        pdf_output_path = results_dir / "combined_benchmark_results.pdf"
        plt.savefig(pdf_output_path, bbox_inches='tight')
        print(f"Plot saved to {pdf_output_path}")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    plot_results()