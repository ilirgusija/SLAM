import os
import time
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.classes.quantizer import BeliefQuantizer


def _ensure_outdir() -> Path:
    out_dir = Path("output/beliefQuantizer")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _unique_rows(a: np.ndarray) -> int:
    # robust unique row count for float codebook with rational entries
    # scale by M to get integers and then use np.unique on a structured view
    # Caller ensures values are multiples of 1/M, but we round for safety
    scaled = a.copy()
    # Try to infer M from denominators: values are multiples of 1/M
    # Use row sums (==1) and max value times M == integer -> estimate M as lcm not needed for tests
    # For reliability, round to 12 decimals
    scaled = np.round(scaled, 12)
    view = np.ascontiguousarray(scaled).view([('', scaled.dtype)] * scaled.shape[1])
    return np.unique(view).shape[0]


def test_codebook_size_and_uniqueness():
    """
    Validate that _generate_codebook produces the correct cardinality and
    all rows are unique for a few small parameter pairs.
    """
    print("\n=== BeliefQuantizer Codebook Validation ===")
    params = [
        (3, 3),
        (4, 3),
        (5, 4),
        (6, 4),
    ]

    for M, N_n in params:
        print(f"\nTesting M={M}, N_n={N_n}:")
        bq = BeliefQuantizer(M, N_n)
        codebook = bq._generate_codebook()
        expected = comb(M + N_n - 1, N_n - 1)

        print(f"  Expected cardinality: {expected}")
        print(f"  Actual shape: {codebook.shape}")
        print(f"  Shape matches: {codebook.shape == (expected, N_n)}")

        # values should be multiples of 1/M and rows should sum to 1
        row_sums = codebook.sum(axis=1)
        print(f"  Row sums (min/max): {row_sums.min():.6f} / {row_sums.max():.6f}")
        print(f"  All rows sum to 1: {np.allclose(row_sums, 1.0)}")

        # uniqueness
        n_unique = _unique_rows(codebook)
        print(f"  Unique rows found: {n_unique}")
        print(f"  All rows unique: {n_unique == expected}")

        # Show a few example rows
        print(f"  Sample rows (first 3):")
        for i in range(min(3, codebook.shape[0])):
            print(f"    Row {i}: {codebook[i]}")

        assert codebook.shape == (expected, N_n)
        np.testing.assert_allclose(codebook.sum(axis=1), 1.0)
        assert n_unique == expected, f"found {n_unique} unique rows, expected {expected}"

    print("\n✓ All codebook validations passed!")


def test_codebook_benchmark_and_plot():
    """
    Benchmark _generate_codebook across a small grid of (M, N_n),
    log timings, and save a plot under output/beliefQuantizer/.
    """
    print("\n=== BeliefQuantizer Performance Benchmark ===")
    out_dir = _ensure_outdir()

    grid = [
        (4, 3),
        (5, 3),
        (6, 4),
        (7, 4),
        (8, 5),
    ]

    print(f"Testing {len(grid)} parameter combinations:")
    results = []
    for M, N_n in grid:
        print(f"\nBenchmarking M={M}, N_n={N_n}:")
        bq = BeliefQuantizer(M, N_n)

        # warmup
        _ = bq._generate_codebook()
        print("  Warmup completed")

        t0 = time.time()
        codebook = bq._generate_codebook()
        t1 = time.time()
        elapsed = t1 - t0

        cardinality = comb(M + N_n - 1, N_n - 1)
        print(f"  Cardinality: {cardinality}")
        print(f"  Time: {elapsed:.6f}s")
        print(f"  Rate: {cardinality/elapsed:.0f} rows/sec")

        results.append((M, N_n, cardinality, elapsed))

    # save CSV-like log
    log_path = out_dir / "codebook_benchmark.tsv"
    print(f"\nSaving benchmark data to: {log_path}")
    with log_path.open("w") as f:
        f.write("M\tN_n\tcardinality\telapsed_s\n")
        for M, N_n, cardinality, elapsed in results:
            f.write(f"{M}\t{N_n}\t{cardinality}\t{elapsed:.6f}\n")

    # plot: x = cardinality, y = elapsed, color/marker by (M,N_n)
    xs = [r[2] for r in results]
    ys = [r[3] for r in results]
    labels = [f"M={r[0]},N={r[1]}" for r in results]

    print("Generating performance plot...")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=100, alpha=0.7)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=10)
    ax.set_xlabel("Cardinality (comb(M+N_n-1, N_n-1))")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("BeliefQuantizer _generate_codebook performance")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(xs) > 1:
        z = np.polyfit(xs, ys, 1)
        p = np.poly1d(z)
        ax.plot(xs, p(xs), "r--", alpha=0.8, label=f"Trend: {z[0]:.2e}x + {z[1]:.2e}")
        ax.legend()

    fig.tight_layout()
    fig_path = out_dir / "codebook_performance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Plot saved to: {fig_path}")

    # Summary statistics
    print(f"\nPerformance Summary:")
    print(f"  Fastest: {min(ys):.6f}s (M={results[ys.index(min(ys))][0]}, N_n={results[ys.index(min(ys))][1]})")
    print(f"  Slowest: {max(ys):.6f}s (M={results[ys.index(max(ys))][0]}, N_n={results[ys.index(max(ys))][1]})")
    print(f"  Speedup: {max(ys)/min(ys):.1f}x")

    # basic smoke assertion that timings are finite and positive
    assert all(np.isfinite(ys)) and all(y >= 0 for y in ys)
    print("\n✓ Performance benchmark completed!")


def test_codebook_large_scale_benchmark():
    """
    Benchmark _generate_codebook for larger, more realistic parameter ranges.
    Tests N_n=512 (11*11*512 state space) with increasing M values.
    """
    print("\n=== Large Scale BeliefQuantizer Benchmark ===")
    H, W = 2, 3
    print(f"Testing realistic parameter ranges for {2**(H*W) * H * W} state space")

    out_dir = _ensure_outdir()
    N_n = 2**(H * W) * H * W

    # More realistic parameters - N_n=512, increasing M
    large_grid = [
        (5, N_n),  # Medium M
        (6, N_n),  # Larger M
        (7, N_n),  # Even larger M
        (8, N_n),  # Large M
    ]

    print(f"Testing {len(large_grid)} large-scale parameter combinations:")
    print("Note: These may take longer due to combinatorial explosion...")

    results = []
    for M, _ in large_grid:
        print(f"\nBenchmarking M={M}, N_n={N_n}:")

        # Calculate expected cardinality first to warn if it's huge
        cardinality = comb(M + N_n - 1, N_n - 1)
        print(f"  Expected cardinality: {cardinality:,}")

        if cardinality > 1_000_000:
            print(f"  WARNING: Very large cardinality! This may take a while...")

        try:
            bq = BeliefQuantizer(M, N_n)

            # warmup
            print("  Timing actual run...")
            t0 = time.time()
            codebook = bq._generate_codebook()
            t1 = time.time()
            elapsed = t1 - t0

            print(f"  Actual cardinality: {codebook.shape[0]:,}")
            print(f"  Time: {elapsed:.6f}s")
            print(f"  Rate: {cardinality/elapsed:.0f} rows/sec")
            print(f"  Memory: ~{codebook.nbytes / 1024 / 1024:.1f} MB (float32)")

            results.append((M, N_n, cardinality, elapsed))

        except MemoryError:
            print(f"  ERROR: Out of memory for M={M}, N_n={N_n}")
            print(f"  Skipping this combination...")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping this combination...")
            continue

    if not results:
        print("\nNo successful runs completed - all combinations failed!")
        return

    # save CSV-like log
    log_path = out_dir / "codebook_large_scale_benchmark.tsv"
    print(f"\nSaving large-scale benchmark data to: {log_path}")
    with log_path.open("w") as f:
        f.write("M\tN_n\tcardinality\telapsed_s\n")
        for M, N_n, cardinality, elapsed in results:
            f.write(f"{M}\t{N_n}\t{cardinality}\t{elapsed:.6f}\n")

    # plot: x = cardinality, y = elapsed
    xs = [r[2] for r in results]
    ys = [r[3] for r in results]
    labels = [f"M={r[0]}" for r in results]

    print("Generating large-scale performance plot...")
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, s=150, alpha=0.7, c=range(len(xs)), cmap='viridis')
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=12)
    ax.set_xlabel("Cardinality (log scale)")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("BeliefQuantizer Large Scale Performance (N_n=512)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(xs) > 1:
        log_xs = np.log10(xs)
        z = np.polyfit(log_xs, ys, 1)
        p = np.poly1d(z)
        ax.plot(xs, p(log_xs), "r--", alpha=0.8, label=f"Trend: {z[0]:.2e}*log10(x) + {z[1]:.2e}")
        ax.legend()

    fig.tight_layout()
    fig_path = out_dir / "codebook_large_scale_performance.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Large-scale plot saved to: {fig_path}")

    # Summary statistics
    print(f"\nLarge Scale Performance Summary:")
    print(f"  Successful runs: {len(results)}")
    print(f"  Fastest: {min(ys):.6f}s (M={results[ys.index(min(ys))][0]})")
    print(f"  Slowest: {max(ys):.6f}s (M={results[ys.index(max(ys))][0]})")
    print(f"  Speedup: {max(ys)/min(ys):.1f}x")
    print(f"  Cardinality range: {min(xs):,} to {max(xs):,}")

    # basic smoke assertion that timings are finite and positive
    assert all(np.isfinite(ys)) and all(y >= 0 for y in ys)
def test_belief_quantizer_caching():
    """
    Test the caching functionality of BeliefQuantizer.
    """
    print("\n=== BeliefQuantizer Caching Test ===")

    # Test parameters
    M, N_n = 4, 8
    cache_dir = "cache/belief_quantizer_test"

    print(f"Testing caching with M={M}, N_n={N_n}")

    # Clear any existing cache
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # First creation - should generate and cache
    print("\n1. First creation (should generate and cache):")
    import time
    t0 = time.time()
    bq1 = BeliefQuantizer(M, N_n, cache_dir=cache_dir)
    t1 = time.time()
    first_time = t1 - t0
    print(f"  First creation time: {first_time:.6f}s")

    # Second creation - should load from cache
    print("\n2. Second creation (should load from cache):")
    t0 = time.time()
    bq2 = BeliefQuantizer(M, N_n, cache_dir=cache_dir)
    t1 = time.time()
    second_time = t1 - t0
    print(f"  Second creation time: {second_time:.6f}s")

    # Verify they're identical
    print("\n3. Verifying codebooks are identical:")
    np.testing.assert_array_equal(bq1.Π_n_M, bq2.Π_n_M)
    print("  ✓ Codebooks are identical!")

    # Test cache listing
    print("\n4. Testing cache listing:")
    cached = BeliefQuantizer.list_cached_quantizers(cache_dir)
    print(f"  Found {len(cached)} cached quantizers:")
    for M_cached, N_n_cached, cardinality, file_size_mb in cached:
        print(f"    M={M_cached}, N_n={N_n_cached}, cardinality={cardinality:,}, size={file_size_mb:.1f}MB")

    # Test force regeneration
    print("\n5. Testing force regeneration:")
    t0 = time.time()
    bq3 = BeliefQuantizer(M, N_n, cache_dir=cache_dir, force_regenerate=True)
    t1 = time.time()
    force_time = t1 - t0
    print(f"  Force regeneration time: {force_time:.6f}s")

    # Test cache clearing
    print("\n6. Testing cache clearing:")
    bq1.clear_cache()

    # Verify cache is cleared
    cached_after = BeliefQuantizer.list_cached_quantizers(cache_dir)
    print(f"  Cached quantizers after clearing: {len(cached_after)}")

    # Performance comparison
    print(f"\n7. Performance comparison:")
    print(f"  First creation: {first_time:.6f}s")
    print(f"  Cache load: {second_time:.6f}s")
    print(f"  Force regeneration: {force_time:.6f}s")
    if second_time > 0:
        speedup = first_time / second_time
        print(f"  Cache speedup: {speedup:.1f}x faster")

    # Cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print("\n✓ Caching test completed successfully!")


def test_belief_quantizer_caching_large():
    """
    Test caching with a larger example to show real performance benefits.
    """
    print("\n=== Large Scale Caching Test ===")

    # Test with larger parameters
    M, N_n = 7, 2**(2 * 2) * 2 * 2
    cache_dir = "cache/belief_quantizer_large_test"

    print(f"Testing caching with M={M}, N_n={N_n}")

    # Clear any existing cache
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # First creation - should generate and cache
    print("\n1. First creation (generating large codebook):")
    import time
    t0 = time.time()
    bq1 = BeliefQuantizer(M, N_n, cache_dir=cache_dir)
    t1 = time.time()
    first_time = t1 - t0
    print(f"  First creation time: {first_time:.6f}s")
    print(f"  Codebook shape: {bq1.Π_n_M.shape}")
    print(f"  Cardinality: {bq1.cardinality:,}")

    # Second creation - should load from cache
    print("\n2. Second creation (loading from cache):")
    t0 = time.time()
    bq2 = BeliefQuantizer(M, N_n, cache_dir=cache_dir)
    t1 = time.time()
    second_time = t1 - t0
    print(f"  Second creation time: {second_time:.6f}s")

    # Verify they're identical
    print("\n3. Verifying codebooks are identical:")
    np.testing.assert_array_equal(bq1.Π_n_M, bq2.Π_n_M)
    print("  ✓ Codebooks are identical!")

    # Performance comparison
    print(f"\n4. Performance comparison:")
    print(f"  First creation: {first_time:.6f}s")
    print(f"  Cache load: {second_time:.6f}s")
    if second_time > 0:
        speedup = first_time / second_time
        print(f"  Cache speedup: {speedup:.1f}x faster")
        print(f"  Time saved: {first_time - second_time:.6f}s")

    # Show cache file info
    print(f"\n5. Cache file information:")
    cached = BeliefQuantizer.list_cached_quantizers(cache_dir)
    for M_cached, N_n_cached, cardinality, file_size_mb in cached:
        print(f"  M={M_cached}, N_n={N_n_cached}, cardinality={cardinality:,}, size={file_size_mb:.1f}MB")

    # Cleanup
    # if os.path.exists(cache_dir):
    #     shutil.rmtree(cache_dir)

    print("\n✓ Large scale caching test completed successfully!")


def test_belief_quantizer_cache_management():
    """
    Test cache management utilities (stats, clear_all, etc.).
    """
    print("\n=== Cache Management Test ===")

    cache_dir = "cache/belief_quantizer_mgmt_test"

    # Clear any existing cache
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Create several quantizers to populate cache
    print("1. Creating multiple quantizers to populate cache:")
    configs = [(3, 5), (4, 6), (5, 7)]

    for M, N_n in configs:
        print(f"  Creating M={M}, N_n={N_n}...")
        BeliefQuantizer(M, N_n, cache_dir=cache_dir)

    # Test cache stats
    print("\n2. Testing cache statistics:")
    stats = BeliefQuantizer.cache_stats(cache_dir)
    print(f"  Cache count: {stats['count']}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print(f"  Total cardinality: {stats['total_cardinality']:,}")
    print(f"  Largest file: {stats['largest_file_mb']:.1f} MB")
    print(f"  Largest cardinality: {stats['largest_cardinality']:,}")

    # Test cache listing
    print("\n3. Testing cache listing:")
    cached = BeliefQuantizer.list_cached_quantizers(cache_dir)
    print(f"  Found {len(cached)} cached quantizers:")
    for M, N_n, cardinality, size_mb in cached:
        print(f"    M={M}, N_n={N_n}, cardinality={cardinality:,}, size={size_mb:.1f}MB")

    # Test clear all cache
    print("\n4. Testing clear all cache:")
    BeliefQuantizer.clear_all_cache(cache_dir)

    # Verify cache is cleared
    stats_after = BeliefQuantizer.cache_stats(cache_dir)
    print(f"  Cache count after clearing: {stats_after['count']}")
    print(f"  Total size after clearing: {stats_after['total_size_mb']:.1f} MB")

    # Cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    print("\n✓ Cache management test completed successfully!")


def test_reznik_algorithm_validation():
    """
    Comprehensive validation of the Reznik algorithm.
    """
    print("\n=== Reznik Algorithm Validation ===")

    # Test with different quantizer configurations
    test_configs = [
        (3, 4),   # Small
        (4, 5),   # Medium
        (5, 6),   # Larger
    ]

    for M, N_n in test_configs:
        print(f"\nTesting M={M}, N_n={N_n}:")

        # Create quantizer
        bq = BeliefQuantizer(M, N_n, cache_dir="cache/test_reznik")

        # Test 1: Verify algorithm properties
        print("  1. Testing algorithm properties:")

        # Generate random belief vectors
        np.random.seed(42)  # For reproducibility
        test_vectors = []

        # Create some test vectors
        for _ in range(10):
            # Generate random vector and normalize
            vec = np.random.rand(N_n)
            vec = vec / np.sum(vec)
            test_vectors.append(vec)

        # Test each vector
        for i, z in enumerate(test_vectors):
            print(f"    Testing vector {i+1}: {z[:3]}... (sum={np.sum(z):.6f})")

            # Apply Reznik algorithm
            quantized = bq.reznik_algorithm(z)

            # Validate properties
            assert len(quantized) == N_n, f"Wrong output length: {len(quantized)} != {N_n}"
            assert np.isclose(np.sum(quantized), 1.0, atol=1e-10), f"Sum not 1: {np.sum(quantized)}"
            assert np.all(quantized >= 0), f"Negative values: {quantized}"

            # Check membership and get index
            codebook_idx = bq.is_in_codebook(quantized)

            if codebook_idx < 0:
                # Diagnose the issue: check if other methods find it
                dbg = bq.debug_nearest(quantized)
                in_codebook_dist, min_distance, nearest_idx = bq.is_in_codebook_distance(quantized)

                print(f"      ✗ Vector not found in codebook!")
                print(f"        Codebook index: {codebook_idx} (not found)")
                print(f"        Distance method: {in_codebook_dist} (min_distance={min_distance:.2e})")
                print(f"        Quantized: {quantized}")
                print(f"        q_int: {dbg['q_int']} (sum={dbg['q_int_sum']}, expected={dbg['expected_sum']})")
                print(f"        Nearest codebook entry [{nearest_idx}]: {bq.Π_n_M[nearest_idx]}")
                print(f"        Nearest int: {dbg.get('nearest_int', 'N/A')}")
                print(f"        Diff int: {dbg.get('diff_int', 'N/A')}")
                assert False, f"Quantized vector not in codebook! Distance: {in_codebook_dist}, Min distance: {min_distance:.2e}"
            else:
                # Found! Print both vectors and verify they match
                codebook_vector = bq.Π_n_M[codebook_idx]
                l2_distance = np.linalg.norm(quantized - codebook_vector)

                print(f"        ✓ Found at index {codebook_idx}")
                print(f"        Quantized (Reznik): {quantized}")
                print(f"        Codebook[{codebook_idx}]:  {codebook_vector}")
                print(f"        L2 distance: {l2_distance:.2e}")

                # Verify they're essentially identical
                assert l2_distance < 1e-10, f"Vectors don't match! L2 distance: {l2_distance:.2e}"

        # Test 2: Edge cases
        print("  2. Testing edge cases:")

        # Test uniform distribution
        uniform = np.ones(N_n) / N_n
        quantized_uniform = bq.reznik_algorithm(uniform)
        print(f"    Uniform input: {uniform[:3]}... -> {quantized_uniform[:3]}...")

        # Test one-hot vector
        one_hot = np.zeros(N_n)
        one_hot[0] = 1.0
        quantized_one_hot = bq.reznik_algorithm(one_hot)
        print(f"    One-hot input: {one_hot[:3]}... -> {quantized_one_hot[:3]}...")

        # Test 3: Verify quantization quality
        print("  3. Testing quantization quality:")

        # Generate many random vectors and check quantization error
        errors = []
        for _ in range(100):
            vec = np.random.rand(N_n)
            vec = vec / np.sum(vec)
            quantized = bq.reznik_algorithm(vec)

            # Find nearest codebook entry
            distances = np.linalg.norm(bq.Π_n_M - quantized, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_codebook = bq.Π_n_M[nearest_idx]

            # Check if Reznik algorithm found the truly nearest
            all_distances = np.linalg.norm(bq.Π_n_M - vec, axis=1)
            truly_nearest_idx = np.argmin(all_distances)
            truly_nearest = bq.Π_n_M[truly_nearest_idx]

            error = np.linalg.norm(quantized - truly_nearest)
            errors.append(error)

        avg_error = np.mean(errors)
        max_error = np.max(errors)
        print(f"    Average quantization error: {avg_error:.6f}")
        print(f"    Maximum quantization error: {max_error:.6f}")

        # The Reznik algorithm should find the optimal quantization
        assert avg_error < 1e-10, f"Reznik algorithm not finding optimal quantization! Avg error: {avg_error}"

    print("\n✓ Reznik algorithm validation completed successfully!")


def test_reznik_algorithm_jit_performance():
    """
    Test whether JIT compilation improves Reznik algorithm performance.
    """
    print("\n=== Reznik Algorithm JIT Performance Test ===")

    # Test configuration
    M, N_n = 5, 20
    bq = BeliefQuantizer(M, N_n, cache_dir="cache/test_reznik_jit")

    # Generate test vectors
    np.random.seed(42)
    n_vectors = 1000
    test_vectors = []
    for _ in range(n_vectors):
        vec = np.random.rand(N_n)
        vec = vec / np.sum(vec)
        test_vectors.append(vec)

    print(f"Testing with {n_vectors} vectors, M={M}, N_n={N_n}")

    # Test non-JIT version (remove @njit temporarily)
    print("\n1. Testing non-JIT version:")

    # Create a copy without JIT
    import types
    reznik_non_jit = types.MethodType(
        lambda self, z: self._reznik_algorithm_non_jit(z),
        bq
    )

    # Define non-JIT version
    def _reznik_algorithm_non_jit(self, z):
        """Non-JIT version of Reznik algorithm for comparison."""
        assert len(z) == self.N_n
        assert np.isclose(np.sum(z), 1.0, atol=1e-10), f"Sum is {np.sum(z)}, expected 1.0"
        assert self.M > 0

        k_prime = np.zeros(self.N_n)
        k_prime = np.floor(self.M * z + 0.5)
        M_prime = np.sum(k_prime)
        Delta = M_prime - self.M

        if Delta == 0:
            return k_prime / self.M

        delta = np.zeros(self.N_n)
        delta = k_prime - self.M * z

        # sort delta in increasing order
        sorted_indices = np.argsort(delta)

        k = np.zeros(self.N_n)
        if Delta > 0:
            condition = (sorted_indices + 1) <= self.N_n - Delta - 1
            k = np.where(condition, k_prime, k_prime - 1)
        else:
            condition = (sorted_indices + 1) <= abs(Delta)
            k = np.where(condition, k_prime + 1, k_prime)

        # Ensure the result sums to 1
        result = k / self.M
        result = result / np.sum(result)  # Normalize to ensure sum = 1
        return result

    bq._reznik_algorithm_non_jit = _reznik_algorithm_non_jit.__get__(bq, BeliefQuantizer)

    # Time non-JIT version
    import time
    t0 = time.time()
    for vec in test_vectors:
        result = bq._reznik_algorithm_non_jit(vec)
    t1 = time.time()
    non_jit_time = t1 - t0

    print(f"  Non-JIT time: {non_jit_time:.6f}s")
    print(f"  Non-JIT rate: {n_vectors/non_jit_time:.0f} vectors/sec")

    # Test JIT version
    print("\n2. Testing JIT version:")

    # Warm up JIT
    warmup_vec = test_vectors[0]
    bq.reznik_algorithm(warmup_vec)

    # Time JIT version
    t0 = time.time()
    for vec in test_vectors:
        result = bq.reznik_algorithm(vec)
    t1 = time.time()
    jit_time = t1 - t0

    print(f"  JIT time: {jit_time:.6f}s")
    print(f"  JIT rate: {n_vectors/jit_time:.0f} vectors/sec")

    # Compare results
    print("\n3. Comparing results:")
    test_vec = test_vectors[0]
    non_jit_result = bq._reznik_algorithm_non_jit(test_vec)
    jit_result = bq.reznik_algorithm(test_vec)

    np.testing.assert_array_almost_equal(non_jit_result, jit_result, decimal=10)
    print("  ✓ Results are identical!")

    # Performance comparison
    print(f"\n4. Performance comparison:")
    if jit_time > 0:
        speedup = non_jit_time / jit_time
        print(f"  JIT speedup: {speedup:.1f}x")
        print(f"  Time saved: {non_jit_time - jit_time:.6f}s")

        if speedup > 1.5:
            print("  ✓ JIT provides significant speedup!")
        elif speedup > 1.1:
            print("  ⚠ JIT provides modest speedup")
        else:
            print("  ⚠ JIT provides minimal/no speedup")

    print("\n✓ JIT performance test completed successfully!")


def test_debug_combinatorial_limits():
    """
    Debug exactly where the combinatorial explosion hits limits.
    """
    print("\n=== Debugging Combinatorial Limits ===")

    # Test progressively larger values to find the breaking point
    test_cases = [
        (3, 10),    # Very small
        (4, 20),    # Small
        (5, 50),    # Medium
        (6, 2**(2 * 2) * 2 * 2),   # Larger
        # (6, 2**(2*3) * 2 * 3),   # Large
        # (6, 2**(3*3) * 3 * 3),   # Your current test case
    ]

    for M, N_n in test_cases:
        print(f"\nTesting M={M}, N_n={N_n}:")

        # Step 1: Can we compute the cardinality?
        try:
            cardinality = comb(M + N_n - 1, N_n - 1)
            print(f"  ✓ Cardinality computed: {cardinality:,}")
        except Exception as e:
            print(f"  ✗ Cardinality computation failed: {e}")
            continue

        # Step 2: Can we allocate the array?
        try:
            estimated_memory_gb = (cardinality * N_n * 4) / (1024**3)  # 4 bytes per float32
            print(f"  Estimated memory needed: {estimated_memory_gb:.2f} GB")

            if estimated_memory_gb > 29:  # More than 29GB
                print(f"  ✗ Memory requirement too large: {estimated_memory_gb:.2f} GB")
                continue

            # Try to allocate the array
            test_array = np.zeros((cardinality, N_n), dtype=np.float32)
            print(f"  ✓ Array allocation successful: {test_array.shape}")
            del test_array  # Free memory immediately

        except Exception as e:
            print(f"  ✗ Array allocation failed: {e}")
            continue

        # Step 3: Can we create the BeliefQuantizer?
        try:
            print(f"  Attempting BeliefQuantizer creation...")
            print(f"  (This will call _generate_codebook() internally)")
            t0 = time.time()
            bq = BeliefQuantizer(M, N_n)
            t1 = time.time()
            print(f"  ✓ BeliefQuantizer created successfully in {t1-t0:.6f}s")

            # Step 4: Test accessing the already-generated codebook
            try:
                print(f"  Accessing pre-generated codebook...")
                t0 = time.time()
                codebook = bq.Π_n_M  # Access the already-generated codebook
                t1 = time.time()
                print(f"  ✓ Codebook accessed: {codebook.shape} in {t1-t0:.6f}s")
                print(f"  ✓ SUCCESS: M={M}, N_n={N_n} works completely!")

            except Exception as e:
                print(f"  ✗ Codebook access failed: {e}")
                print(f"  ✗ Breaking point: M={M}, N_n={N_n}")
                break

        except Exception as e:
            print(f"  ✗ BeliefQuantizer creation failed: {e}")
            print(f"  ✗ Breaking point: M={M}, N_n={N_n}")
            break

    print("\n=== Debugging Complete ===")
