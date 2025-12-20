"""
Test the benchmarking infrastructure to ensure it works correctly.

This script validates that the benchmark system can properly measure
performance and compare different executors.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from retriever.core.flow import Flow
from retriever.integrations.dora.benchmarks import PerformanceBenchmarker


async def test_basic_benchmarking():
    """Test basic benchmarking functionality."""
    print("Testing basic benchmarking infrastructure...")
    
    # Create a simple test flow
    simple_flow = Flow.from_module(lambda x: x * 2)
    
    # Initialize benchmarker with minimal iterations for testing
    benchmarker = PerformanceBenchmarker(warmup_iterations=1, test_iterations=3)
    
    # Run benchmark (only LocalExecutor for now since DoraExecutor is not fully implemented)
    await benchmarker.benchmark_flow(
        simple_flow,
        "test_simple",
        input_data=42,
        compare_executors=False,  # Only test LocalExecutor for now
        measure_memory=True
    )
    
    # Check results
    results = benchmarker.get_results()
    assert len(results.results) == 1, "Should have one benchmark result"
    
    result = results.results[0]
    assert result.name == "test_simple_local", f"Unexpected name: {result.name}"
    assert result.executor_type == "LocalExecutor", f"Unexpected executor: {result.executor_type}"
    assert result.success_rate == 1.0, f"Should have 100% success rate, got {result.success_rate}"
    assert result.avg_time > 0, f"Should have positive execution time, got {result.avg_time}"
    
    print(f"✓ Basic benchmark test passed")
    print(f"  - Average time: {result.avg_time*1000:.3f}ms")
    print(f"  - Throughput: {result.throughput:.2f} ops/sec")
    print(f"  - Memory peak: {result.memory_peak_mb:.1f}MB")
    
    return benchmarker


async def test_array_processing(benchmarker):
    """Test benchmarking with numpy array processing."""
    print("\nTesting array processing benchmark...")
    
    # Create array processing flow
    array_flow = Flow.from_module(lambda arr: np.sum(arr, axis=0))
    test_array = np.random.random((1000, 100)).astype(np.float32)
    
    initial_count = len(benchmarker.get_results().results)
    
    await benchmarker.benchmark_flow(
        array_flow,
        "test_array",
        test_array,
        compare_executors=False,
        measure_memory=True
    )
    
    results = benchmarker.get_results()
    # Should have one more result now
    assert len(results.results) == initial_count + 1, f"Should have {initial_count + 1} results, got {len(results.results)}"
    
    array_result = results.results[-1]  # Last result
    assert array_result.success_rate == 1.0, "Array benchmark should succeed"
    
    print(f"✓ Array processing benchmark test passed")
    print(f"  - Average time: {array_result.avg_time*1000:.3f}ms")


async def test_sequential_pipeline(benchmarker):
    """Test benchmarking with sequential pipeline."""
    print("\nTesting sequential pipeline benchmark...")
    
    # Create sequential pipeline
    sequential_flow = (
        Flow.from_module(lambda x: x * 2)
        .then(Flow.from_module(lambda x: x + 10))
        .then(Flow.from_module(lambda x: x / 3))
    )
    
    await benchmarker.benchmark_flow(
        sequential_flow,
        "test_sequential",
        input_data=5.0,
        compare_executors=False,
        measure_memory=True
    )
    
    results = benchmarker.get_results()
    sequential_result = results.results[-1]  # Last result
    assert sequential_result.success_rate == 1.0, "Sequential benchmark should succeed"
    
    print(f"✓ Sequential pipeline benchmark test passed")
    print(f"  - Average time: {sequential_result.avg_time*1000:.3f}ms")


async def test_fanout_pipeline(benchmarker):
    """Test benchmarking with fanout (parallel) pipeline."""
    print("\nTesting fanout pipeline benchmark...")
    
    # Create fanout pipeline
    fanout_flow = (
        Flow.from_module(lambda x: x * 2)
        .fanout(Flow.from_module(lambda x: x + 5))
    )
    
    await benchmarker.benchmark_flow(
        fanout_flow,
        "test_fanout",
        input_data=7,
        compare_executors=False,
        measure_memory=True
    )
    
    results = benchmarker.get_results()
    fanout_result = results.results[-1]  # Last result
    assert fanout_result.success_rate == 1.0, "Fanout benchmark should succeed"
    
    print(f"✓ Fanout pipeline benchmark test passed")
    print(f"  - Average time: {fanout_result.avg_time*1000:.3f}ms")


def test_benchmark_suite_functions():
    """Test BenchmarkSuite utility functions."""
    print("\nTesting BenchmarkSuite functions...")
    
    from retriever.integrations.dora.benchmarks import BenchmarkSuite, BenchmarkResult
    
    suite = BenchmarkSuite()
    
    # Add mock results
    result1 = BenchmarkResult(
        name="test_local",
        executor_type="LocalExecutor",
        iterations=10,
        total_time=1.0,
        avg_time=0.1,
        min_time=0.05,
        max_time=0.15,
        std_time=0.02,
        throughput=10.0,
        memory_peak_mb=100.0,
        memory_avg_mb=95.0,
        success_rate=1.0
    )
    
    result2 = BenchmarkResult(
        name="test_dora",
        executor_type="DoraExecutor", 
        iterations=10,
        total_time=0.5,
        avg_time=0.05,
        min_time=0.025,
        max_time=0.075,
        std_time=0.01,
        throughput=20.0,
        memory_peak_mb=80.0,
        memory_avg_mb=75.0,
        success_rate=1.0
    )
    
    suite.add_result(result1)
    suite.add_result(result2)
    
    # Test speedup calculation
    speedup = suite.get_speedup("test_local", "test_dora")
    assert speedup == 2.0, f"Expected 2x speedup, got {speedup}"
    
    # Test serialization
    suite_dict = suite.to_dict()
    assert len(suite_dict["results"]) == 2, "Should serialize 2 results"
    assert suite_dict["results"][0]["name"] == "test_local", "Should preserve result names"
    
    print("✓ BenchmarkSuite functions test passed")
    print(f"  - Calculated speedup: {speedup}x")


async def main():
    """Run all benchmark infrastructure tests."""
    print("Dora Benchmarking Infrastructure Test Suite")
    print("=" * 50)
    
    try:
        # Test basic functionality and get shared benchmarker
        benchmarker = await test_basic_benchmarking()
        await test_array_processing(benchmarker)
        await test_sequential_pipeline(benchmarker)
        await test_fanout_pipeline(benchmarker)
        
        # Test utility functions
        test_benchmark_suite_functions()
        
        print("\n" + "=" * 50)
        print("✓ All benchmarking infrastructure tests passed!")
        print("The benchmark system is ready for use.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())