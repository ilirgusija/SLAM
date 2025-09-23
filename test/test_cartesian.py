from utils.misc import cartesian_pairs
import itertools
import time
import numpy as np


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def cartesian_product_transpose(*arrays):
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    rows, cols = np.prod(broadcasted[0].shape), len(broadcasted)
    dtype = np.result_type(*arrays)

    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def cartesian_product_transpose_pp(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty((la, *map(len, arrays)), dtype=dtype)
    idx = slice(None), *itertools.repeat(None, la)
    for i, a in enumerate(arrays):
        arr[i, ...] = a[idx[:la-i]]
    return arr.reshape(la, -1).T


def cartesian_dot_product(arr1, arr2):
    """
    Compute the cartesian dot product between two arrays.

    Args:
        arr1: Array of shape (m, n) - first set of vectors
        arr2: Array of shape (k, n) - second set of vectors

    Returns:
        Array of shape (m, k) containing dot products between all pairs
        of vectors from arr1 and arr2
    """
    # Use broadcasting to compute all pairwise dot products efficiently
    # arr1[:, None, :] has shape (m, 1, n)
    # arr2[None, :, :] has shape (1, k, n)
    # The multiplication gives (m, k, n), then sum along last axis gives (m, k)
    return np.sum(arr1[:, None, :] * arr2[None, :, :], axis=-1)


def test_cartesian_product_functions():
    # Test all three cartesian product functions
    print("\n" + "="*50)
    print("COMPARING CARTESIAN PRODUCT FUNCTIONS")
    print("="*50)

    # Test data for cartesian product
    a = np.array([1, 2])
    b = np.array([4, 5])
    c = np.array([6, 7])

    print(f"Input arrays: a={a}, b={b}, c={c}")
    print(f"Expected shape: ({len(a)*len(b)*len(c)}, 3)")

    # Test all three functions
    print("\n1. Original cartesian_product:")
    start_time = time.time()
    result1 = cartesian_product(a, b, c)
    time1 = time.time() - start_time
    print(f"   Result shape: {result1.shape}")
    print(f"   Time: {time1:.6f} seconds")
    print(f"   First few rows: {result1[:5]}")

    print("\n2. cartesian_product_transpose:")
    start_time = time.time()
    result2 = cartesian_product_transpose(a, b, c)
    time2 = time.time() - start_time
    print(f"   Result shape: {result2.shape}")
    print(f"   Time: {time2:.6f} seconds")
    print(f"   First few rows: {result2[:5]}")

    print("\n3. cartesian_product_transpose_pp:")
    start_time = time.time()
    result3 = cartesian_product_transpose_pp([a, b, c])
    time3 = time.time() - start_time
    print(f"   Result shape: {result3.shape}")
    print(f"   Time: {time3:.6f} seconds")
    print(f"   First few rows: {result3[:5]}")

    # Verify all results are identical
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    print(f"Results 1 and 2 identical: {np.array_equal(result1, result2)}")
    print(f"Results 1 and 3 identical: {np.array_equal(result1, result3)}")
    print(f"Results 2 and 3 identical: {np.array_equal(result2, result3)}")

    # Performance comparison
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    times = [time1, time2, time3]
    functions = ["cartesian_product",
                 "cartesian_product_transpose", "cartesian_product_transpose_pp"]
    fastest_idx = np.argmin(times)

    print(f"Fastest: {functions[fastest_idx]} ({times[fastest_idx]:.6f}s)")
    print(f"Slowest: {functions[np.argmax(times)]} ({np.max(times):.6f}s)")
    print(f"Speedup: {np.max(times)/times[fastest_idx]:.2f}x")

    # Test with larger arrays for better timing
    print("\n" + "="*50)
    print("LARGER ARRAY TEST")
    print("="*50)

    large_a = np.arange(50)
    large_b = np.arange(30)
    large_c = np.arange(20)

    print(
        f"Large arrays: a.shape={large_a.shape}, b.shape={large_b.shape}, c.shape={large_c.shape}")
    print(
        f"Expected result size: {len(large_a)*len(large_b)*len(large_c)} rows")

    # Time with larger arrays
    print("\n1. Original cartesian_product:")
    start_time = time.time()
    large_result1 = cartesian_product(large_a, large_b, large_c)
    large_time1 = time.time() - start_time
    print(f"   Result shape: {large_result1.shape}")
    print(f"   Time: {large_time1:.6f} seconds")

    print("\n2. cartesian_product_transpose:")
    start_time = time.time()
    large_result2 = cartesian_product_transpose(large_a, large_b, large_c)
    large_time2 = time.time() - start_time
    print(f"   Result shape: {large_result2.shape}")
    print(f"   Time: {large_time2:.6f} seconds")

    print("\n3. cartesian_product_transpose_pp:")
    start_time = time.time()
    large_result3 = cartesian_product_transpose_pp([large_a, large_b, large_c])
    large_time3 = time.time() - start_time
    print(f"   Result shape: {large_result3.shape}")
    print(f"   Time: {large_time3:.6f} seconds")

    # Large array performance comparison
    print("\n" + "="*50)
    print("LARGE ARRAY PERFORMANCE")
    print("="*50)
    large_times = [large_time1, large_time2, large_time3]
    fastest_large_idx = np.argmin(large_times)

    print(
        f"Fastest: {functions[fastest_large_idx]} ({large_times[fastest_large_idx]:.6f}s)")
    print(
        f"Slowest: {functions[np.argmax(large_times)]} ({np.max(large_times):.6f}s)")
    print(
        f"Speedup: {np.max(large_times)/large_times[fastest_large_idx]:.2f}x")

    # Verify large results are identical
    print(
        f"\nLarge results identical: {np.array_equal(large_result1, large_result2) and np.array_equal(large_result1, large_result3)}")


def test_cartesian_dot_product():
    print("\n" + "="*50)
    print("TESTING CARTESIAN DOT PRODUCT FUNCTIONS")
    print("="*50)

    # Test data
    arr1 = np.array([[1, 1], [2, 2], [3, 3]])
    arr2 = np.array([[1, 2], [3, 4]])

    print(f"Input arrays:")
    print(f"arr1 shape: {arr1.shape}, arr1: {arr1}")
    print(f"arr2 shape: {arr2.shape}, arr2: {arr2}")
    print(
        f"Expected result shape: ({arr1.shape[0]}, {arr2.shape[0]}) = (3, 2)")

    # Test 1: Direct broadcasting method
    print("\n1. Direct broadcasting method (cartesian_dot_product):")
    start_time = time.time()
    result1 = cartesian_dot_product(arr1, arr2)
    time1 = time.time() - start_time
    print(f"   Result shape: {result1.shape}")
    print(f"   Time: {time1:.6f} seconds")
    print(f"   Result: {result1}")

    # Test 2: Cartesian product transpose pp method
    print("\n2. Cartesian product transpose pp method:")
    start_time = time.time()
    intermediate = cartesian_product_transpose_pp([arr1, arr2])
    result2 = np.dot(intermediate[:, :2], intermediate[:, 2:])
    time2 = time.time() - start_time
    print(f"   Result shape: {result2.shape}")

    # Test 2: Cartesian product transpose method for comparison
    print("\n3. Cartesian product transpose method:")
    start_time = time.time()
    intermediate = cartesian_product_transpose(arr1, arr2)
    result3 = np.dot(intermediate[:, :2], intermediate[:, 2:])
    time3 = time.time() - start_time
    print(f"   Result shape: {result3.shape}")
    print(f"   Time: {time3:.6f} seconds")
    print(f"   Result: {result3}")

    # Manual verification
    print("\n" + "="*50)
    print("MANUAL VERIFICATION")
    print("="*50)
    print("Expected results (computed manually):")
    expected = np.array([[3, 7], [6, 14], [9, 21]])
    print(f"Expected: {expected}")

    # Verify all results are identical (only for methods 1 and 2)
    print("\n" + "="*50)
    print("CORRECTNESS CHECKING")
    print("="*50)
    print(f"Result 1 matches expected: {np.array_equal(result1, expected)}")
    print(f"Result 2 matches expected: {np.array_equal(result2, expected)}")
    print(f"Result 3 matches expected: {np.array_equal(result3, expected)}")
    print(f"Results 1 and 2 identical: {np.array_equal(result1, result2)}")
    print(f"Results 1 and 3 identical: {np.array_equal(result1, result3)}")
    print(f"Results 2 and 3 identical: {np.array_equal(result2, result3)}")

    # Performance comparison (only for methods 1 and 2)
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    times = [time1, time2, time3]
    functions = ["Direct broadcasting",
                 "Cartesian product transpose pp", "Cartesian product transpose"]
    fastest_idx = np.argmin(times)

    print(f"Fastest: {functions[fastest_idx]} ({times[fastest_idx]:.6f}s)")
    print(f"Slowest: {functions[np.argmax(times)]} ({np.max(times):.6f}s)")
    print(f"Speedup: {np.max(times)/times[fastest_idx]:.2f}x")

    # Test with larger arrays for better timing
    print("\n" + "="*50)
    print("LARGER ARRAY TEST")
    print("="*50)

    # Create larger test arrays
    large_arr1 = np.random.rand(100, 50)  # 100 vectors of dimension 50
    large_arr2 = np.random.rand(80, 50)   # 80 vectors of dimension 50

    print(f"Large arrays:")
    print(f"arr1 shape: {large_arr1.shape}")
    print(f"arr2 shape: {large_arr2.shape}")
    print(
        f"Expected result shape: ({large_arr1.shape[0]}, {large_arr2.shape[0]}) = (100, 80)")

    # Time with larger arrays
    print("\n1. Direct broadcasting method:")
    start_time = time.time()
    large_result1 = cartesian_dot_product(large_arr1, large_arr2)
    large_time1 = time.time() - start_time
    print(f"   Result shape: {large_result1.shape}")
    print(f"   Time: {large_time1:.6f} seconds")

    print("\n2. Cartesian product transpose pp method:")
    start_time = time.time()
    intermediate = cartesian_product_transpose_pp([large_arr1, large_arr2])
    large_result2 = np.dot(intermediate[:, :2], intermediate[:, 2:])
    large_time2 = time.time() - start_time
    print(f"   Result shape: {large_result2.shape}")
    print(f"   Time: {large_time2:.6f} seconds")

    print("\n3. Cartesian product transpose method:")
    start_time = time.time()
    intermediate = cartesian_product_transpose(large_arr1, large_arr2)
    large_result3 = np.dot(intermediate[:, :2], intermediate[:, 2:])
    large_time3 = time.time() - start_time
    print(f"   Result shape: {large_result3.shape}")
    print(f"   Time: {large_time3:.6f} seconds")

    # Large array performance comparison
    print("\n" + "="*50)
    print("LARGE ARRAY PERFORMANCE")
    print("="*50)
    large_times = [large_time1, large_time2, large_time3]
    fastest_large_idx = np.argmin(large_times)

    print(
        f"Fastest: {functions[fastest_large_idx]} ({large_times[fastest_large_idx]:.6f}s)")
    print(
        f"Slowest: {functions[np.argmax(large_times)]} ({np.max(large_times):.6f}s)")
    print(
        f"Speedup: {np.max(large_times)/large_times[fastest_large_idx]:.2f}x")

    # Verify large results are identical
    print(
        f"\nLarge results identical: {np.allclose(large_result1, large_result2) and np.allclose(large_result1, large_result3)}")

    # Memory usage comparison (rough estimate)
    print("\n" + "="*50)
    print("MEMORY USAGE ESTIMATES")
    print("="*50)
    print(
        f"Direct method: ~{large_result1.nbytes / 1024:.1f} KB (result only)")
    print(
        f"Cartesian product transpose: ~{large_result2.nbytes / 1024:.1f} KB (result only)")
    print(
        f"Cartesian product transpose pp: ~{large_result3.nbytes / 1024:.1f} KB (result only)")
    print(f"Memory usage is identical for both methods")


if __name__ == "__main__":
    # test_cartesian_product_functions()
    test_cartesian_dot_product()

    # Mixed-shape cartesian pairs test
    print("\n" + "="*50)
    print("TESTING MIXED-SHAPE CARTESIAN PAIRS")
    print("="*50)

    # Define state space X_n: (m, 2)
    m = 3
    X_n = np.random.rand(m, 2)

    # Define maps M: (k, H, W) -> flatten to (k, H*W)
    k, H, W = 5, 2, 2
    M_3d = (np.random.rand(k, H, W) > 0.5).astype(int)
    M = M_3d.reshape(k, H * W)

    X_rep, M_tile = cartesian_pairs(X_n, M)

    print(f"X_n shape: {X_n.shape}")
    print(f"M shape (flattened): {M.shape} (from {k} maps of {H}x{W})")
    print(f"X_rep shape: {X_rep.shape}")
    print(f"M_tile shape: {M_tile.shape}")

    # Shape assertions
    assert X_rep.shape == (m * k, 2)
    assert M_tile.shape == (m * k, H * W)

    # Content assertions: blocks of k rows in X_rep equal each original state
    for i in range(m):
        block = X_rep[i * k:(i + 1) * k]
        assert np.allclose(block, np.repeat(X_n[i][None, :], k, axis=0))

    # Content assertions: M_tile cycles the maps identically for each state
    for i in range(m):
        block = M_tile[i * k:(i + 1) * k]
        assert np.array_equal(block, M)

    print("Mixed-shape cartesian_pairs passed shape and content checks.")
