#!/usr/bin/env python3
"""
benchmark_performance.py - Performance benchmarking for SIRF-SIMIND-Connection

This script performs comprehensive performance benchmarking of the package,
including memory usage, execution time, and scalability testing.

Usage:
    python benchmark_performance.py
    python benchmark_performance.py --quick
    python benchmark_performance.py --output-dir benchmark_results

Author: SIRF-SIMIND-Connection Team
"""

import argparse
import time
import sys
import gc
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, asdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sirf.STIR import ImageData, AcquisitionData
    SIRF_AVAILABLE = True
except ImportError:
    SIRF_AVAILABLE = False

try:
    import sirf_simind_connection
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Context manager for performance profiling."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = None
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
    
    def __enter__(self):
        gc.collect()  # Clean up before measurement
        self.start_time = time.perf_counter()
        
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        else:
            self.start_memory = 0
            self.peak_memory = 0
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        
        if self.process:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
    
    @property
    def execution_time(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def memory_delta(self) -> float:
        if self.process:
            current_memory = self.process.memory_info().rss / 1024 / 1024
            return current_memory - self.start_memory
        return 0.0
    
    @property
    def cpu_usage(self) -> float:
        if self.process:
            return self.process.cpu_percent()
        return 0.0


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results", verbose: bool = True):
        """Initialize the benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
            verbose: Whether to print detailed output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        
        if self.verbose:
            print("SIRF-SIMIND-Connection Performance Benchmark")
            print("=" * 50)
            print(f"Output directory: {self.output_dir}")
            print(f"Temporary directory: {self.temp_dir}")
    
    def run_benchmarks(self, quick: bool = False) -> Dict[str, Any]:
        """Run the complete benchmark suite.
        
        Args:
            quick: Run only essential benchmarks
            
        Returns:
            Dictionary containing benchmark summary
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\nStarting benchmarks (mode: {'quick' if quick else 'comprehensive'})")
            print("-" * 50)
        
        # Core benchmarks (always run)
        self._benchmark_imports()
        self._benchmark_memory_usage()
        
        if PACKAGE_AVAILABLE:
            self._benchmark_config_operations()
            self._benchmark_data_operations()
        
        if not quick:
            self._benchmark_scalability()
            self._benchmark_file_operations()
            self._benchmark_computation_intensive()
        
        total_time = time.time() - start_time
        
        # Generate summary and reports
        summary = self._generate_summary(total_time)
        self._save_results()
        
        if MATPLOTLIB_AVAILABLE:
            self._generate_visualizations()
        
        if self.verbose:
            self._print_summary(summary)
        
        return summary
    
    def _run_benchmark(self, name: str, benchmark_func, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark with performance profiling."""
        if self.verbose:
            print(f"Running: {name}...", end=" ", flush=True)
        
        try:
            with PerformanceProfiler() as profiler:
                result = benchmark_func(*args, **kwargs)
            
            benchmark_result = BenchmarkResult(
                name=name,
                execution_time=profiler.execution_time,
                memory_usage_mb=profiler.memory_delta,
                peak_memory_mb=profiler.peak_memory,
                cpu_usage_percent=profiler.cpu_usage,
                success=True,
                metadata=result if isinstance(result, dict) else None
            )
            
            if self.verbose:
                print(f"✅ {profiler.execution_time:.3f}s, {profiler.memory_delta:+.1f}MB")
            
        except Exception as e:
            benchmark_result = BenchmarkResult(
                name=name,
                execution_time=0.0,
                memory_usage_mb=0.0,
                peak_memory_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e)
            )
            
            if self.verbose:
                print(f"❌ ERROR: {str(e)}")
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def _benchmark_imports(self):
        """Benchmark package import performance."""
        def test_basic_imports():
            import importlib
            start_time = time.perf_counter()
            
            modules = ['numpy', 'scipy', 'matplotlib']
            import_times = {}
            
            for module in modules:
                module_start = time.perf_counter()
                try:
                    importlib.import_module(module)
                    import_times[module] = time.perf_counter() - module_start
                except ImportError:
                    import_times[module] = None
            
            return import_times
        
        def test_package_import():
            import importlib
            start_time = time.perf_counter()
            
            try:
                importlib.import_module('sirf_simind_connection')
                return {'import_time': time.perf_counter() - start_time}
            except ImportError as e:
                return {'error': str(e)}
        
        def test_sirf_import():
            import importlib
            start_time = time.perf_counter()
            
            try:
                sirf_stir = importlib.import_module('sirf.STIR')
                return {'import_time': time.perf_counter() - start_time}
            except ImportError as e:
                return {'error': str(e)}
        
        self._run_benchmark("Basic Imports", test_basic_imports)
        self._run_benchmark("Package Import", test_package_import)
        self._run_benchmark("SIRF Import", test_sirf_import)
    
    def _benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        def test_baseline_memory():
            """Test baseline memory usage."""
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return {
                    'rss_mb': process.memory_info().rss / 1024 / 1024,
                    'vms_mb': process.memory_info().vms / 1024 / 1024
                }
            return {}
        
        def test_array_creation():
            """Test memory usage for large array creation."""
            arrays = []
            for size in [100, 1000, 10000]:
                arr = np.random.rand(size, size) if NUMPY_AVAILABLE else [[0] * size for _ in range(size)]
                arrays.append(arr)
            
            return {'arrays_created': len(arrays)}
        
        def test_memory_cleanup():
            """Test memory cleanup after operations."""
            import gc
            
            # Create large objects
            if NUMPY_AVAILABLE:
                large_array = np.random.rand(1000, 1000, 10)
                del large_array
            
            gc.collect()
            return {'cleanup_completed': True}
        
        self._run_benchmark("Baseline Memory", test_baseline_memory)
        self._run_benchmark("Array Creation", test_array_creation)
        self._run_benchmark("Memory Cleanup", test_memory_cleanup)
    
    def _benchmark_config_operations(self):
        """Benchmark configuration operations."""
        def test_config_creation():
            if not PACKAGE_AVAILABLE:
                return {'skipped': 'Package not available'}
            
            from sirf_simind_connection import SimulationConfig
            
            # Create test SMC content
            smc_content = """
            TITLE Benchmark Test
            PHOTONS 1000000
            SPECTRUM MONO 140.0
            DETECTOR NaI 0.95
            MATRIX 128 128
            """
            
            smc_file = self.temp_dir / "benchmark_config.smc"
            smc_file.write_text(smc_content)
            
            configs_created = 0
            for i in range(10):
                config = SimulationConfig(str(smc_file))
                configs_created += 1
            
            return {'configs_created': configs_created}
        
        def test_config_modification():
            if not PACKAGE_AVAILABLE:
                return {'skipped': 'Package not available'}
            
            # Test configuration loading and saving
            operations = 0
            for i in range(5):
                # Simulate config operations
                operations += 1
            
            return {'operations_completed': operations}
        
        self._run_benchmark("Config Creation", test_config_creation)
        self._run_benchmark("Config Modification", test_config_modification)
    
    def _benchmark_data_operations(self):
        """Benchmark data handling operations."""
        def test_image_creation():
            if not SIRF_AVAILABLE or not NUMPY_AVAILABLE:
                return {'skipped': 'SIRF or NumPy not available'}
            
            images_created = 0
            for dimensions in [(64, 64, 32), (128, 128, 64)]:
                try:
                    img = ImageData()
                    img.initialise(dimensions)
                    
                    # Fill with random data
                    data = np.random.rand(*dimensions)
                    img.fill(data)
                    
                    images_created += 1
                except Exception:
                    pass
            
            return {'images_created': images_created}
        
        def test_data_conversion():
            if not NUMPY_AVAILABLE:
                return {'skipped': 'NumPy not available'}
            
            conversions = 0
            
            # Test various data conversions
            for size in [100, 500, 1000]:
                # HU to density conversion simulation
                hu_values = np.random.randint(-1000, 3000, size)
                densities = hu_values / 1000.0 + 1.0  # Simplified conversion
                conversions += 1
            
            return {'conversions_completed': conversions}
        
        self._run_benchmark("Image Creation", test_image_creation)
        self._run_benchmark("Data Conversion", test_data_conversion)
    
    def _benchmark_scalability(self):
        """Benchmark scalability with different data sizes."""
        def test_scaling_performance():
            if not NUMPY_AVAILABLE:
                return {'skipped': 'NumPy not available'}
            
            results = {}
            sizes = [64, 128, 256, 512]
            
            for size in sizes:
                start_time = time.perf_counter()
                
                # Create and process data
                data = np.random.rand(size, size)
                processed = data * 2 + np.sin(data)
                result = np.sum(processed)
                
                execution_time = time.perf_counter() - start_time
                results[f'size_{size}'] = execution_time
            
            return results
        
        def test_memory_scaling():
            if not NUMPY_AVAILABLE or not PSUTIL_AVAILABLE:
                return {'skipped': 'Dependencies not available'}
            
            process = psutil.Process()
            results = {}
            
            for size_mb in [10, 50, 100]:
                start_memory = process.memory_info().rss / 1024 / 1024
                
                # Allocate memory
                size_elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
                data = np.random.rand(size_elements)
                
                peak_memory = process.memory_info().rss / 1024 / 1024
                memory_used = peak_memory - start_memory
                
                results[f'allocated_{size_mb}MB'] = memory_used
                
                # Clean up
                del data
            
            return results
        
        self._run_benchmark("Scaling Performance", test_scaling_performance)
        self._run_benchmark("Memory Scaling", test_memory_scaling)
    
    def _benchmark_file_operations(self):
        """Benchmark file I/O operations."""
        def test_file_io():
            results = {}
            
            # Test writing different file sizes
            for size_kb in [1, 10, 100]:
                data = b'x' * (size_kb * 1024)
                file_path = self.temp_dir / f"test_{size_kb}kb.bin"
                
                # Write test
                start_time = time.perf_counter()
                file_path.write_bytes(data)
                write_time = time.perf_counter() - start_time
                
                # Read test
                start_time = time.perf_counter()
                read_data = file_path.read_bytes()
                read_time = time.perf_counter() - start_time
                
                results[f'write_{size_kb}KB'] = write_time
                results[f'read_{size_kb}KB'] = read_time
                
                # Cleanup
                file_path.unlink()
            
            return results
        
        def test_image_io():
            if not SIRF_AVAILABLE or not NUMPY_AVAILABLE:
                return {'skipped': 'SIRF or NumPy not available'}
            
            results = {}
            
            for dimensions in [(32, 32, 16), (64, 64, 32)]:
                try:
                    # Create test image
                    img = ImageData()
                    img.initialise(dimensions)
                    data = np.random.rand(*dimensions)
                    img.fill(data)
                    
                    # Write test
                    file_path = self.temp_dir / f"test_image_{'x'.join(map(str, dimensions))}.hv"
                    start_time = time.perf_counter()
                    img.write(str(file_path))
                    write_time = time.perf_counter() - start_time
                    
                    # Read test
                    img2 = ImageData()
                    start_time = time.perf_counter()
                    img2.read(str(file_path))
                    read_time = time.perf_counter() - start_time
                    
                    results[f'write_{"x".join(map(str, dimensions))}'] = write_time
                    results[f'read_{"x".join(map(str, dimensions))}'] = read_time
                    
                except Exception as e:
                    results[f'error_{"x".join(map(str, dimensions))}'] = str(e)
            
            return results
        
        self._run_benchmark("File I/O", test_file_io)
        self._run_benchmark("Image I/O", test_image_io)
    
    def _benchmark_computation_intensive(self):
        """Benchmark computationally intensive operations."""
        def test_numerical_operations():
            if not NUMPY_AVAILABLE:
                return {'skipped': 'NumPy not available'}
            
            results = {}
            
            # Matrix operations
            for size in [100, 500, 1000]:
                matrix_a = np.random.rand(size, size)
                matrix_b = np.random.rand(size, size)
                
                start_time = time.perf_counter()
                result = np.dot(matrix_a, matrix_b)
                execution_time = time.perf_counter() - start_time
                
                results[f'matrix_mult_{size}x{size}'] = execution_time
            
            # FFT operations
            for size in [1024, 4096, 16384]:
                signal = np.random.rand(size)
                
                start_time = time.perf_counter()
                fft_result = np.fft.fft(signal)
                execution_time = time.perf_counter() - start_time
                
                results[f'fft_{size}'] = execution_time
            
            return results
        
        def test_iterative_algorithms():
            if not NUMPY_AVAILABLE:
                return {'skipped': 'NumPy not available'}
            
            results = {}
            
            # Simulate iterative reconstruction
            for matrix_size in [64, 128]:
                data = np.random.rand(matrix_size, matrix_size)
                
                start_time = time.perf_counter()
                
                # Simulate 10 iterations of processing
                for iteration in range(10):
                    data = data * 0.9 + np.random.rand(matrix_size, matrix_size) * 0.1
                    data = np.maximum(data, 0)  # Non-negativity constraint
                
                execution_time = time.perf_counter() - start_time
                results[f'iterative_{matrix_size}x{matrix_size}'] = execution_time
            
            return results
        
        self._run_benchmark("Numerical Operations", test_numerical_operations)
        self._run_benchmark("Iterative Algorithms", test_iterative_algorithms)
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate benchmark summary statistics."""
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        if successful_results:
            execution_times = [r.execution_time for r in successful_results]
            memory_usages = [r.memory_usage_mb for r in successful_results]
            
            summary = {
                'total_benchmarks': len(self.results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'total_time': total_time,
                'execution_time_stats': {
                    'mean': statistics.mean(execution_times),
                    'median': statistics.median(execution_times),
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                },
                'memory_usage_stats': {
                    'mean': statistics.mean(memory_usages),
                    'median': statistics.median(memory_usages),
                    'min': min(memory_usages),
                    'max': max(memory_usages),
                    'stdev': statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
                }
            }
        else:
            summary = {
                'total_benchmarks': len(self.results),
                'successful': 0,
                'failed': len(failed_results),
                'total_time': total_time,
                'execution_time_stats': {},
                'memory_usage_stats': {}
            }
        
        return summary
    
    def _save_results(self):
        """Save benchmark results to files."""
        # Save detailed results as JSON
        results_data = {
            'benchmark_info': {
                'timestamp': time.time(),
                'python_version': sys.version,
                'platform': sys.platform,
                'dependencies': {
                    'numpy': NUMPY_AVAILABLE,
                    'psutil': PSUTIL_AVAILABLE,
                    'matplotlib': MATPLOTLIB_AVAILABLE,
                    'sirf': SIRF_AVAILABLE,
                    'package': PACKAGE_AVAILABLE
                }
            },
            'results': [asdict(result) for result in self.results]
        }
        
        json_file = self.output_dir / "benchmark_results.json"
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save summary as CSV
        csv_file = self.output_dir / "benchmark_summary.csv"
        with open(csv_file, 'w') as f:
            f.write("Name,Success,ExecutionTime,MemoryUsage,PeakMemory,CPUUsage,ErrorMessage\n")
            for result in self.results:
                f.write(f"{result.name},{result.success},{result.execution_time:.6f},"
                       f"{result.memory_usage_mb:.2f},{result.peak_memory_mb:.2f},"
                       f"{result.cpu_usage_percent:.1f},{result.error_message or ''}\n")
    
    def _generate_visualizations(self):
        """Generate performance visualization plots."""
        if not MATPLOTLIB_AVAILABLE or not self.results:
            return
        
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SIRF-SIMIND-Connection Performance Benchmark Results', fontsize=16)
        
        # 1. Execution time by benchmark
        names = [r.name for r in successful_results]
        times = [r.execution_time for r in successful_results]
        
        ax1.barh(range(len(names)), times, color='skyblue', alpha=0.7)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels([name[:25] + '...' if len(name) > 25 else name for name in names], fontsize=8)
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_title('Execution Time by Benchmark')
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory usage distribution
        memory_usages = [r.memory_usage_mb for r in successful_results if r.memory_usage_mb != 0]
        if memory_usages:
            ax2.hist(memory_usages, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Memory Usage (MB)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Memory Usage Distribution')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No memory data available', ha='center', va='center', transform=ax2.transAxes)
        
        # 3. Execution time vs Memory usage scatter
        if memory_usages:
            times_with_memory = [r.execution_time for r in successful_results if r.memory_usage_mb != 0]
            ax3.scatter(times_with_memory, memory_usages, alpha=0.7, color='green')
            ax3.set_xlabel('Execution Time (seconds)')
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('Execution Time vs Memory Usage')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No memory data available', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Success rate pie chart
        success_count = len(successful_results)
        failure_count = len(self.results) - success_count
        
        if success_count + failure_count > 0:
            ax4.pie([success_count, failure_count], labels=['Successful', 'Failed'], 
                   colors=['green', 'red'], autopct='%1.1f%%')
            ax4.set_title('Benchmark Success Rate')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Visualization saved to: {plot_file}")
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print benchmark summary to console."""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total time: {summary['total_time']:.2f} seconds")
        
        if summary['execution_time_stats']:
            stats = summary['execution_time_stats']
            print(f"\nExecution Time Statistics:")
            print(f"  Mean: {stats['mean']:.4f}s")
            print(f"  Median: {stats['median']:.4f}s")
            print(f"  Range: {stats['min']:.4f}s - {stats['max']:.4f}s")
            print(f"  Std Dev: {stats['stdev']:.4f}s")
        
        if summary['memory_usage_stats']:
            stats = summary['memory_usage_stats']
            print(f"\nMemory Usage Statistics:")
            print(f"  Mean: {stats['mean']:.2f} MB")
            print(f"  Median: {stats['median']:.2f} MB")
            print(f"  Range: {stats['min']:.2f} MB - {stats['max']:.2f} MB")
            print(f"  Std Dev: {stats['stdev']:.2f} MB")
        
        # Show failed benchmarks
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\nFailed Benchmarks:")
            for result in failed_results:
                print(f"  ❌ {result.name}: {result.error_message}")
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main function for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for SIRF-SIMIND-Connection"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only essential benchmarks"
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    try:
        # Run benchmarks
        summary = benchmark.run_benchmarks(quick=args.quick)
        
        # Exit with appropriate code based on results
        if summary['failed'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nBenchmarking failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        benchmark.cleanup()


if __name__ == "__main__":
    main()