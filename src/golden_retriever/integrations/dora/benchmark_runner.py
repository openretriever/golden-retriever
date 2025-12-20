#!/usr/bin/env python3
"""
Dora Executor Benchmark Runner

This script runs comprehensive benchmarks comparing LocalExecutor vs DoraExecutor
performance across various robotics-relevant workloads.

Usage:
    python benchmark_runner.py [--output results.json] [--quick]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from retriever.core.flow import Flow
from retriever.integrations.dora.benchmarks import PerformanceBenchmarker


def create_test_flows():
    """Create various Flow patterns for benchmarking."""
    
    # Simple computation flows
    simple_multiply = Flow.from_module(lambda x: x * 2)
    
    array_processing = Flow.from_module(lambda arr: np.sum(arr, axis=0))
    
    # Sequential pipeline
    sequential = (
        Flow.from_module(lambda x: x * 2)
        .then(Flow.from_module(lambda x: x + 10))
        .then(Flow.from_module(lambda x: x / 3))
    )
    
    # Parallel fanout
    fanout_simple = (
        Flow.from_module(lambda x: x * 2)
        .fanout(Flow.from_module(lambda x: x + 5))
    )
    
    # Complex nested pipeline
    left_branch = (
        Flow.from_module(lambda x: np.sin(x))
        .then(Flow.from_module(lambda x: x * 100))
    )
    right_branch = (
        Flow.from_module(lambda x: np.cos(x))
        .then(Flow.from_module(lambda x: x * 200))
    )
    complex_pipeline = left_branch.fanout(right_branch)
    
    # Robotics-inspired flows
    image_processing = (
        Flow.from_module(lambda img: img.astype(np.float32) / 255.0)  # Normalize
        .then(Flow.from_module(lambda img: np.mean(img, axis=2)))     # Grayscale
        .then(Flow.from_module(lambda img: img > 0.5))                # Threshold
    )
    
    return {
        "simple_multiply": (simple_multiply, 42),
        "array_processing": (array_processing, np.random.random((1000, 100)).astype(np.float32)),
        "sequential": (sequential, 10.0),
        "fanout_simple": (fanout_simple, 7),
        "complex_pipeline": (complex_pipeline, np.random.random(100).astype(np.float64)),
        "image_processing": (image_processing, np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)),
    }


async def run_basic_benchmarks(benchmarker: PerformanceBenchmarker, quick: bool = False):
    """Run basic performance benchmarks."""
    print("Running basic Flow performance benchmarks...")
    
    test_flows = create_test_flows()
    
    # If quick mode, only test a subset
    if quick:
        test_flows = {k: v for k, v in list(test_flows.items())[:3]}
    
    for name, (flow, input_data) in test_flows.items():
        await benchmarker.benchmark_flow(
            flow, 
            name, 
            input_data,
            compare_executors=True,
            measure_memory=True
        )
        print()  # Add spacing between benchmarks


async def run_scalability_benchmarks(benchmarker: PerformanceBenchmarker):
    """Run scalability benchmarks with increasing data sizes."""
    print("Running scalability benchmarks...")
    
    # Test image processing with different sizes
    image_flow = Flow.from_module(lambda img: np.mean(img, axis=2))
    
    sizes = [
        (240, 320),    # Small
        (480, 640),    # Medium  
        (720, 1280),   # HD
        (1080, 1920),  # Full HD
    ]
    
    for height, width in sizes:
        test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        await benchmarker.benchmark_flow(
            image_flow,
            f"image_{height}x{width}",
            test_image,
            compare_executors=True
        )


async def run_data_type_benchmarks(benchmarker: PerformanceBenchmarker):
    """Run benchmarks focusing on data serialization performance."""
    print("Running data type serialization benchmarks...")
    
    # This will test various robotics data types
    benchmarker.benchmark_data_types()


async def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run Dora Executor benchmarks")
    parser.add_argument(
        "--output", 
        default="benchmark_results.json",
        help="Output file for benchmark results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick subset of benchmarks"
    )
    parser.add_argument(
        "--scalability",
        action="store_true", 
        help="Include scalability benchmarks"
    )
    parser.add_argument(
        "--data-types",
        action="store_true",
        help="Include data type serialization benchmarks"
    )
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    if args.quick:
        benchmarker = PerformanceBenchmarker(warmup_iterations=1, test_iterations=3)
    else:
        benchmarker = PerformanceBenchmarker(warmup_iterations=3, test_iterations=10)
    
    print("Dora Executor Performance Benchmark Suite")
    print("=" * 50)
    
    try:
        # Run basic benchmarks
        await run_basic_benchmarks(benchmarker, args.quick)
        
        # Run additional benchmarks if requested
        if args.scalability and not args.quick:
            await run_scalability_benchmarks(benchmarker)
        
        if args.data_types and not args.quick:
            await run_data_type_benchmarks(benchmarker)
        
        # Print results summary
        benchmarker.print_summary()
        
        # Save results
        benchmarker.save_results(args.output)
        
        print(f"\nBenchmark complete! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())