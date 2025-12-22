#!/usr/bin/env python3
"""
Benchmark script for Native Controller backends (Python, Rust, C++).
"""

import subprocess
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
DURATION = 5

def check_binaries():
    print("Checking native binaries...")
    rust_bin = SCRIPT_DIR / "target/release/rust-controller"
    cpp_bin = SCRIPT_DIR / "build/cpp-controller"
    
    if rust_bin.exists():
        print(f"  [OK] Rust binary found: {rust_bin}")
    else:
        print(f"  [FAIL] Rust binary NOT found: {rust_bin}\n         Run 'pixi run -e rust binding-build-rust' first.")
        
    if cpp_bin.exists():
        print(f"  [OK] C++ binary found: {cpp_bin}")
    else:
        print(f"  [FAIL] C++ binary NOT found: {cpp_bin}\n         Run 'pixi run -e cpp binding-build-cpp' first.")

def run_benchmark(backend: str, duration: int, rate: float) -> Optional[float]:
    """
    Runs the app with the specified backend and rate, returns the measured Hz.
    """
    print("\n" + "="*60)
    print(f"Benchmarking {backend.upper()} backend at {rate} Hz for {duration} seconds...")
    print("="*60)
    
    cmd = [
        "python", 
        str(SCRIPT_DIR / "app.py"),
        "--backend", backend,
        "--rate", str(rate),
        "--duration", str(duration)
    ]
    
    try:
        # Run process and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 5  # buffer time
        )
        
        # Parse output for stats
        # Expected: [RobotDriver] STATS: Total=100, Duration=2.00s, Rate=50.00 Hz
        # Retriever logs might capture stdout and print to stderr, so check both.
        combined_output = result.stdout + "\n" + result.stderr
        
        for line in combined_output.splitlines():
            if "[RobotDriver] STATS:" in line:
                # Extract Hz
                try:
                    parts = line.split("Rate=")[1].split(" Hz")[0]
                    hz = float(parts)
                    print(f"--> Result: {hz:.2f} Hz")
                    return hz
                except (IndexError, ValueError) as e:
                    print(f"--> Failed to parse Hz from line: {line} ({e})")
                    # Keep trying other lines (e.g. if we have multiple logs)
                    
        print("--> No stats found in output.")
        # Debug: print last few lines of stderr
        print(f"Last stderr lines:\n{result.stderr[-500:]}")
        return 0.0
        
    except subprocess.TimeoutExpired:
        print("--> Process timed out!")
        return 0.0
    except Exception as e:
        print(f"--> Error: {e}")
        return 0.0

def main():
    check_binaries()
    
    backends = ["python", "rust", "cpp"]
    rates = [50, 200, 1000, 5000]
    results = {rate: {} for rate in rates}
    
    for rate in rates:
        for backend in backends:
            hz = run_benchmark(backend, DURATION, rate)
            results[rate][backend] = hz

    print("\n\n" + "="*60)
    print("BENCHMARK RESULTS (Message Throughput)")
    print("="*60)
    
    # Header
    header = f"{'Rate (Hz)':<10} |"
    for backend in backends:
        header += f" {backend.upper():<12} |"
    print(header)
    print("-" * len(header))
    
    # Rows
    for rate in rates:
        row = f"{rate:<10.0f} |"
        base_hz = results[rate].get("python", 0.0)
        
        for backend in backends:
            hz = results[rate].get(backend, 0.0)
            if backend == "python":
                row += f" {hz:<12.2f} |"
            else:
                speedup = hz / base_hz if base_hz > 0 else 0.0
                row += f" {hz:<12.2f} ({speedup:.1f}x) |" 
        print(row)
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

