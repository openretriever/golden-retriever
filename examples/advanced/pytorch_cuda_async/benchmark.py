
import time
import torch
import argparse
import sys
import os
import csv
import logging

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Ensure dependencies
sys.path.insert(0, os.getcwd())
try:
    from retriever.rt.backend.dora.serde import serialize_arrow, deserialize_arrow
except ImportError:
    # Attempt to add project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    src_root = os.path.join(project_root, "src")
    sys.path.insert(0, src_root)
    from retriever.rt.backend.dora.serde import serialize_arrow, deserialize_arrow

logging.basicConfig(level=logging.ERROR)

def benchmark_transfer(size_mb: int, iterations: int = 100, device_str: str = "cpu"):
    """
    Benchmark the serialization speed using native serde.py.
    """
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return 0.0

    # Create random tensor
    num_elements = int((size_mb * 1024 * 1024) / 4)
    data = torch.randn(num_elements, device=device)

    print(f"Benchmarking {size_mb} MB on {device} ({iterations} iters)...")

    # Warmup
    for _ in range(5):
        arrow, meta = serialize_arrow(data)
        _ = deserialize_arrow(arrow, meta)

    start_time = time.time()
    for _ in range(iterations):
        # 1. Send (Serialize to Arrow/Handle)
        arrow, meta = serialize_arrow(data)

        # 2. Recv (Deserialize)
        # Note: In real Dora, 'arrow' is passed via shared memory. 
        # Here we just pass the object reference, but it simulates the wrapping cost.
        _ = deserialize_arrow(arrow, meta)

    end_time = time.time()
    duration = end_time - start_time

    if duration < 1e-6:
        duration = 1e-6

    total_mb = size_mb * iterations
    throughput = total_mb / duration

    print(f"  Duration: {duration:.4f}s")
    print(f"  Throughput: {throughput:.2f} MB/s")
    return throughput


def generate_plot(input_csv: str, output_image: str):
    if not HAS_MPL:
        print("\nMatplotlib not installed, skipping plot generation.")
        return

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return
        
    data = {}
    devices = set()
    sizes = set()
    
    with open(input_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dev = row["device"]
            size = int(row["size_mb"])
            tp = float(row["throughput_mbs"])
            
            data[(dev, size)] = tp
            devices.add(dev)
            sizes.add(size)
            
    sorted_sizes = sorted(list(sizes))
    sorted_devices = sorted(list(devices))
    
    print("\n--- Benchmark Summary ---")
    print(f"{'Device':<10} | {'Size (MB)':<10} | {'Throughput (MB/s)':<20}")
    print("-" * 50)
    for dev in sorted_devices:
        for size in sorted_sizes:
            tp = data.get((dev, size), 0)
            print(f"{dev:<10} | {size:<10} | {tp:<20.2f}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    indices = range(len(sorted_sizes))
    
    for i, dev in enumerate(sorted_devices):
        vals = [data.get((dev, s), 0) for s in sorted_sizes]
        pos = [x + i * bar_width for x in indices]
        ax.bar(pos, vals, width=bar_width, label=dev)
    
    ax.set_xlabel('Payload Size (MB)')
    ax.set_ylabel('Throughput (MB/s)')
    ax.set_title('Zero-Copy Tensor Transfer Throughput (Native)')
    ax.set_xticks([x + bar_width * (len(devices) - 1) / 2 for x in indices])
    ax.set_xticklabels([f"{s}MB" for s in sorted_sizes])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(output_image)
    print(f"\nPlot saved to {output_image}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--plot", default="benchmark_plot.png", help="Output Plot file (optional)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--skip-bench", action="store_true", help="Skip running benchmark (plot only)")
    args = parser.parse_args()

    sizes = [1, 10, 100]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    elif torch.backends.mps.is_available():
        devices.append("mps")

    if not args.skip_bench:
        with open(args.output, "w") as f:
            f.write("device,size_mb,throughput_mbs\n")

            for dev in devices:
                for size in sizes:
                    tp = benchmark_transfer(size, args.iterations, dev)
                    f.write(f"{dev},{size},{tp}\n")

        print(f"\nResults saved to {args.output}")

    if args.plot:
        generate_plot(args.output, args.plot)


if __name__ == "__main__":
    main()
