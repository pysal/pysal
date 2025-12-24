import time
import tracemalloc
from unittest.mock import MagicMock, patch

from pysal.base import _installed_versions, memberships


def benchmark_version_checking():
    print("=" * 70)
    print("PySAL Parallel Version Checking Benchmark")
    print("=" * 70)

    packages = list(memberships.keys())
    print(f"\nTotal packages to check: {len(packages)}")
    print(f"Packages: {', '.join(packages[:5])}...")

    print("\n" + "-" * 70)
    print("Benchmark 1: Real Package Version Checking")
    print("-" * 70)

    tracemalloc.start()
    start_time = time.perf_counter()
    start_mem = tracemalloc.get_traced_memory()[0]

    versions = _installed_versions()

    end_time = time.perf_counter()
    end_mem = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    elapsed = end_time - start_time
    mem_used = (end_mem - start_mem) / 1024 / 1024

    print(f"Time: {elapsed:.3f}s")
    print(f"Memory: {mem_used:.2f} MB")
    found = sum(1 for v in versions.values() if v != 'NA')
    print(f"Packages found: {found}/{len(versions)}")

    print("\n" + "-" * 70)
    print("Benchmark 2: Simulated Slow Imports (0.1s each)")
    print("-" * 70)

    def slow_import(_name):
        time.sleep(0.1)
        mock_mod = MagicMock()
        mock_mod.__version__ = "1.0.0"
        return mock_mod

    mock_path = 'pysal.base.importlib.import_module'
    with patch(mock_path, side_effect=slow_import):
        tracemalloc.start()
        start_time = time.perf_counter()
        start_mem = tracemalloc.get_traced_memory()[0]

        _installed_versions()

        end_time = time.perf_counter()
        end_mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        parallel_time = end_time - start_time
        parallel_mem = (end_mem - start_mem) / 1024 / 1024

    sequential_time_estimate = len(packages) * 0.1
    speedup = sequential_time_estimate / parallel_time

    print(f"Parallel execution time: {parallel_time:.3f}s")
    print(f"Sequential estimate: {sequential_time_estimate:.3f}s")
    print(f"Speedup: {speedup:.1f}x")
    time_saved = sequential_time_estimate - parallel_time
    pct_improvement = (1 - parallel_time/sequential_time_estimate) * 100
    print(f"Time saved: {time_saved:.3f}s ({pct_improvement:.1f}%)")
    print(f"Memory overhead: {parallel_mem:.2f} MB")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Parallel version checking is {speedup:.1f}x faster")
    overhead_reduction = (1 - parallel_time/sequential_time_estimate) * 100
    print(f"✓ Reduces import overhead by {overhead_reduction:.1f}%")
    print(f"✓ Memory overhead is minimal ({parallel_mem:.2f} MB)")
    print("=" * 70)

    return {
        "real_time_seconds": elapsed,
        "real_memory_mb": mem_used,
        "simulated_parallel_time_seconds": parallel_time,
        "simulated_sequential_estimate_seconds": sequential_time_estimate,
        "speedup_factor": speedup,
        "time_saved_seconds": sequential_time_estimate - parallel_time,
        "percent_improvement": (1 - parallel_time/sequential_time_estimate) * 100,
        "parallel_memory_mb": parallel_mem,
        "total_packages": len(packages),
        "packages_found": sum(1 for v in versions.values() if v != 'NA')
    }


if __name__ == "__main__":
    import json
    results = benchmark_version_checking()

    output_file = "agent_logs/bench_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
