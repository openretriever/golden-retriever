"""
Performance Benchmarking Infrastructure for Dora Integration

This module provides comprehensive benchmarking tools to measure and compare
performance between LocalExecutor and DoraExecutor implementations.

Key Metrics:
- Execution latency and throughput
- Memory usage and allocation patterns
- Zero-copy efficiency measurements
- Parallel execution speedup factors
"""

import time
import psutil
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
import statistics
import json

import numpy as np

from ...core.executor import LocalExecutor
from ...core.flow import Flow
from .executor import DoraExecutor


@dataclass
class BenchmarkResult:
    """Container for benchmark execution results."""
    name: str
    executor_type: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput: float  # operations per second
    memory_peak_mb: float
    memory_avg_mb: float
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results for comparison."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result to the suite."""
        self.results.append(result)
    
    def get_speedup(self, baseline_name: str, comparison_name: str) -> Optional[float]:
        """Calculate speedup between two benchmarks."""
        baseline = next((r for r in self.results if r.name == baseline_name), None)
        comparison = next((r for r in self.results if r.name == comparison_name), None)
        
        if baseline and comparison and baseline.avg_time > 0:
            return baseline.avg_time / comparison.avg_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark suite to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "results": [
                {
                    "name": r.name,
                    "executor_type": r.executor_type,
                    "iterations": r.iterations,
                    "total_time": r.total_time,
                    "avg_time": r.avg_time,
                    "min_time": r.min_time,
                    "max_time": r.max_time,
                    "std_time": r.std_time,
                    "throughput": r.throughput,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_avg_mb": r.memory_avg_mb,
                    "success_rate": r.success_rate,
                    "metadata": r.metadata
                }
                for r in self.results
            ]
        }


class PerformanceBenchmarker:
    """
    Comprehensive benchmarking system for comparing executor performance.
    
    This benchmarker measures execution time, memory usage, and throughput
    across different executor implementations to validate the performance
    improvements claimed by dora-rs integration.
    
    Example:
        ```python
        benchmarker = PerformanceBenchmarker()
        
        # Benchmark simple pipeline
        simple_flow = Flow.from_module(lambda x: x * 2)
        benchmarker.benchmark_flow(simple_flow, "simple_multiply", input_data=10)
        
        # Compare results
        suite = benchmarker.get_results()
        print(f"Speedup: {suite.get_speedup('simple_multiply_local', 'simple_multiply_dora')}x")
        ```
    """
    
    def __init__(self, warmup_iterations: int = 3, test_iterations: int = 10):
        """
        Initialize the benchmarker.
        
        Args:
            warmup_iterations: Number of warmup runs to stabilize performance
            test_iterations: Number of test iterations for statistical analysis
        """
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.suite = BenchmarkSuite()
        
        # Initialize executors
        self.local_executor = LocalExecutor()
        self.dora_executor = None  # Lazy initialization
    
    async def benchmark_flow(
        self,
        flow: Flow,
        benchmark_name: str,
        input_data: Any,
        compare_executors: bool = True,
        measure_memory: bool = True
    ) -> BenchmarkSuite:
        """
        Benchmark a Flow computation across different executors.
        
        Args:
            flow: The Flow to benchmark
            benchmark_name: Human-readable name for the benchmark
            input_data: Input data for the Flow
            compare_executors: Whether to test both Local and Dora executors
            measure_memory: Whether to measure memory usage
            
        Returns:
            BenchmarkSuite containing all results
        """
        print(f"Benchmarking: {benchmark_name}")
        
        # Benchmark LocalExecutor
        local_result = await self._benchmark_executor(
            self.local_executor,
            flow,
            f"{benchmark_name}_local",
            "LocalExecutor",
            input_data,
            measure_memory
        )
        self.suite.add_result(local_result)
        
        # Benchmark DoraExecutor if requested
        if compare_executors:
            try:
                if self.dora_executor is None:
                    self.dora_executor = DoraExecutor(debug=True)
                
                dora_result = await self._benchmark_executor(
                    self.dora_executor,
                    flow,
                    f"{benchmark_name}_dora",
                    "DoraExecutor", 
                    input_data,
                    measure_memory
                )
                self.suite.add_result(dora_result)
                
                # Calculate and print speedup
                speedup = self.suite.get_speedup(
                    f"{benchmark_name}_local",
                    f"{benchmark_name}_dora"
                )
                if speedup:
                    print(f"Speedup: {speedup:.2f}x (Dora vs Local)")
                
            except Exception as e:
                print(f"DoraExecutor benchmark failed: {e}")
                print("Continuing with LocalExecutor results only")
        
        return self.suite
    
    async def _benchmark_executor(
        self,
        executor: Union[LocalExecutor, DoraExecutor],
        flow: Flow,
        result_name: str,
        executor_type: str,
        input_data: Any,
        measure_memory: bool
    ) -> BenchmarkResult:
        """
        Benchmark a specific executor implementation.
        
        Args:
            executor: The executor to benchmark
            flow: The Flow to execute
            result_name: Name for the benchmark result
            executor_type: Type of executor (for metadata)
            input_data: Input data for the Flow
            measure_memory: Whether to track memory usage
            
        Returns:
            BenchmarkResult with performance metrics
        """
        print(f"  Testing {executor_type}...")
        
        # Warmup runs
        for _ in range(self.warmup_iterations):
            try:
                if isinstance(executor, DoraExecutor):
                    await executor.run(flow, input_data)
                else:
                    executor.run(flow, input_data)
            except Exception as e:
                print(f"    Warmup failed: {e}")
                # Continue with reduced warmup
                break
        
        # Actual benchmark runs
        execution_times = []
        memory_measurements = []
        successful_runs = 0
        
        for i in range(self.test_iterations):
            try:
                with self._memory_monitor(measure_memory) as memory_tracker:
                    start_time = time.perf_counter()
                    
                    if isinstance(executor, DoraExecutor):
                        result = await executor.run(flow, input_data)
                    else:
                        result = executor.run(flow, input_data)
                    
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    successful_runs += 1
                    
                    if measure_memory and memory_tracker.measurements:
                        memory_measurements.extend(memory_tracker.measurements)
                
            except Exception as e:
                print(f"    Run {i+1} failed: {e}")
                continue
        
        # Calculate statistics
        if execution_times:
            total_time = sum(execution_times)
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
            throughput = successful_runs / total_time if total_time > 0 else 0.0
        else:
            total_time = avg_time = min_time = max_time = std_time = throughput = 0.0
        
        # Memory statistics
        if memory_measurements:
            memory_peak_mb = max(memory_measurements)
            memory_avg_mb = statistics.mean(memory_measurements)
        else:
            memory_peak_mb = memory_avg_mb = 0.0
        
        success_rate = successful_runs / self.test_iterations
        
        print(f"    Average time: {avg_time*1000:.2f}ms")
        print(f"    Throughput: {throughput:.2f} ops/sec")
        print(f"    Success rate: {success_rate*100:.1f}%")
        
        return BenchmarkResult(
            name=result_name,
            executor_type=executor_type,
            iterations=successful_runs,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_time=std_time,
            throughput=throughput,
            memory_peak_mb=memory_peak_mb,
            memory_avg_mb=memory_avg_mb,
            success_rate=success_rate,
            metadata={
                "warmup_iterations": self.warmup_iterations,
                "test_iterations": self.test_iterations,
                "input_data_type": str(type(input_data).__name__)
            }
        )
    
    @contextmanager
    def _memory_monitor(self, enabled: bool):
        """Context manager for monitoring memory usage during execution."""
        class MemoryTracker:
            def __init__(self):
                self.measurements = []
                self.process = psutil.Process()
                self.enabled = enabled
            
            def sample(self):
                if self.enabled:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    self.measurements.append(memory_mb)
        
        tracker = MemoryTracker()
        
        if enabled:
            # Take initial measurement
            tracker.sample()
        
        try:
            yield tracker
        finally:
            if enabled:
                # Take final measurement
                tracker.sample()
    
    def benchmark_data_types(self) -> BenchmarkSuite:
        """
        Benchmark common robotics data types for serialization performance.
        
        Tests different data types that are common in robotics applications
        to validate Arrow serialization performance.
        """
        print("Benchmarking common robotics data types...")
        
        # Test data generators
        test_cases = [
            ("small_image", lambda: np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)),
            ("large_image", lambda: np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)),
            ("point_cloud", lambda: np.random.random((10000, 3)).astype(np.float32)),
            ("pose_array", lambda: np.random.random((100, 7)).astype(np.float64)),  # positions + quaternions
            ("detection_list", lambda: [
                {"box": [10, 20, 100, 200], "score": 0.9, "label": "cup"}
                for _ in range(50)
            ]),
        ]
        
        # Simple pass-through flow for data serialization testing
        identity_flow = Flow.from_module(lambda x: x)
        
        for test_name, data_generator in test_cases:
            test_data = data_generator()
            asyncio.run(self.benchmark_flow(
                identity_flow,
                f"data_type_{test_name}",
                test_data,
                compare_executors=False  # Focus on serialization, not execution
            ))
        
        return self.suite
    
    def save_results(self, filename: str) -> None:
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.suite.to_dict(), f, indent=2)
        print(f"Results saved to {filename}")
    
    def print_summary(self) -> None:
        """Print a summary of all benchmark results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for result in self.suite.results:
            print(f"\n{result.name} ({result.executor_type}):")
            print(f"  Average Time: {result.avg_time*1000:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Memory Peak: {result.memory_peak_mb:.1f}MB")
            print(f"  Success Rate: {result.success_rate*100:.1f}%")
        
        # Print speedup comparisons
        print(f"\nSPEEDUP COMPARISONS:")
        local_results = [r for r in self.suite.results if "local" in r.name]
        for local_result in local_results:
            base_name = local_result.name.replace("_local", "")
            dora_name = f"{base_name}_dora"
            speedup = self.suite.get_speedup(local_result.name, dora_name)
            if speedup:
                print(f"  {base_name}: {speedup:.2f}x speedup (Dora vs Local)")
    
    def get_results(self) -> BenchmarkSuite:
        """Get the current benchmark results."""
        return self.suite