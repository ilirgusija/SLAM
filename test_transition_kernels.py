#!/usr/bin/env python3
"""
Comprehensive test function to compare all transition kernel implementations.
This script tests performance, accuracy, and correctness of different T() methods.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from src.classes.pomdp import POMDP
from src.classes.model import VelocityIntegratorModel, LIDAR
from src.classes.mapping import LidarGridMapVec
from src.classes.obstacle import Obstacle


def test_transition_kernels():
    """
    Comprehensive test of all transition kernel implementations.
    """
    print("=" * 80)
    print("TRANSITION KERNEL PERFORMANCE COMPARISON")
    print("=" * 80)

    # Setup POMDP
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=5, B=4)
    obstacles = []
    map_obj = LidarGridMapVec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)

    pomdp = POMDP(motion_model, sensor, obstacles, map_obj, sigma_w=0.1, sigma_v=0.1)

    # Test parameters
    B = np.array([[1.0, 2.0], [1.0, 2.0]])  # [x_min, x_max], [y_min, y_max]
    u = np.array([1.0, 0.0])

    # Test different state space sizes
    state_counts = [5, 10, 25, 50, 100, 200]
    iterations = 20

    # Available methods
    methods = {
        'T (default)': pomdp.T,
        'T_ot (OpenTURNS)': pomdp.T_ot,
        'T_fortran (mvnun)': pomdp.T_fortran,
    }

    print(f"Testing {len(methods)} methods with {iterations} iterations each")
    print(f"Bounds: {B}")
    print()

    # Results storage
    results = {name: {'times': [], 'accuracies': []} for name in methods.keys()}

    for n_states in state_counts:
        print(f"States: {n_states:3d}")

        # Generate random states
        np.random.seed(42)  # For reproducibility
        X = np.random.uniform(0, 5, (n_states, 2))

        # Get reference result (using default method)
        reference_result = methods['T (default)'](B, X, u)

        for method_name, method_func in methods.items():
            # Time the method
            start_time = time.time()
            for _ in range(iterations):
                result = method_func(B, X, u)
            end_time = time.time()

            avg_time = (end_time - start_time) / iterations
            results[method_name]['times'].append(avg_time)

            # Check accuracy
            if method_name != 'T (default)':
                accuracy = np.max(np.abs(result - reference_result))
                results[method_name]['accuracies'].append(accuracy)
            else:
                results[method_name]['accuracies'].append(0.0)

            print(f"  {method_name:20s}: {avg_time:.6f}s, accuracy: {results[method_name]['accuracies'][-1]:.2e}")

        print()

    # Performance analysis
    print("=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Find fastest method for each state count
    for i, n_states in enumerate(state_counts):
        times = [results[method]['times'][i] for method in methods.keys()]
        fastest_idx = np.argmin(times)
        fastest_method = list(methods.keys())[fastest_idx]
        fastest_time = times[fastest_idx]

        print(f"States {n_states:3d}: Fastest = {fastest_method} ({fastest_time:.6f}s)")

        # Calculate speedups
        speedups = [fastest_time / t for t in times]
        print(f"           Speedups: {[f'{s:.2f}x' for s in speedups]}")
        print()

    # Overall winner
    avg_times = {method: np.mean(results[method]['times']) for method in methods.keys()}
    overall_fastest = min(avg_times, key=avg_times.get)
    print(f"Overall fastest method: {overall_fastest} (avg: {avg_times[overall_fastest]:.6f}s)")

    # Accuracy analysis
    print("\n" + "=" * 80)
    print("ACCURACY ANALYSIS")
    print("=" * 80)

    for method_name in methods.keys():
        if method_name != 'T (default)':
            max_error = np.max(results[method_name]['accuracies'])
            avg_error = np.mean(results[method_name]['accuracies'])
            print(f"{method_name:20s}: Max error: {max_error:.2e}, Avg error: {avg_error:.2e}")

    # Create performance plot
    try:
        plt.figure(figsize=(12, 8))

        # Plot 1: Performance vs State Count
        plt.subplot(2, 2, 1)
        for method_name in methods.keys():
            plt.loglog(state_counts, results[method_name]['times'], 'o-', label=method_name)
        plt.xlabel('Number of States')
        plt.ylabel('Time (seconds)')
        plt.title('Performance vs State Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Speedup vs State Count
        plt.subplot(2, 2, 2)
        reference_times = results['T (default)']['times']
        for method_name in methods.keys():
            if method_name != 'T (default)':
                speedups = [ref / t for ref, t in zip(reference_times, results[method_name]['times'])]
                plt.semilogx(state_counts, speedups, 'o-', label=method_name)
        plt.xlabel('Number of States')
        plt.ylabel('Speedup vs Default')
        plt.title('Speedup vs State Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Accuracy vs State Count
        plt.subplot(2, 2, 3)
        for method_name in methods.keys():
            if method_name != 'T (default)':
                plt.semilogx(state_counts, results[method_name]['accuracies'], 'o-', label=method_name)
        plt.xlabel('Number of States')
        plt.ylabel('Max Absolute Error')
        plt.title('Accuracy vs State Count')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Method comparison bar chart
        plt.subplot(2, 2, 4)
        method_names = list(methods.keys())
        avg_times_list = [avg_times[method] for method in method_names]
        bars = plt.bar(range(len(method_names)), avg_times_list)
        plt.xticks(range(len(method_names)), method_names, rotation=45)
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Performance Comparison')

        # Add value labels on bars
        for bar, time_val in zip(bars, avg_times_list):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                     f'{time_val:.4f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('transition_kernel_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nPerformance plot saved as 'transition_kernel_comparison.png'")

    except ImportError:
        print("\nMatplotlib not available - skipping plots")

    return results


def test_correctness():
    """
    Test correctness of all methods with known analytical results.
    """
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)

    # Setup
    motion_model = VelocityIntegratorModel(i_x=0.0, i_y=0.0, dt=0.1, max_v=1.0)
    sensor = LIDAR(fov=360, r_max=5, B=4)
    obstacles = []
    map_obj = LidarGridMapVec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)

    pomdp = POMDP(motion_model, sensor, obstacles, map_obj, sigma_w=0.1, sigma_v=0.1)

    # Test cases with known results
    test_cases = [
        {
            'name': 'Centered distribution',
            'B': np.array([[0.0, 1.0], [0.0, 1.0]]),
            'X': np.array([[0.0, 0.0]]),
            'u': np.array([0.0, 0.0]),
            'expected_range': (0.1, 0.2)  # Approximate range for 2D normal
        },
        {
            'name': 'Off-center distribution',
            'B': np.array([[1.0, 2.0], [1.0, 2.0]]),
            'X': np.array([[1.0, 1.0]]),
            'u': np.array([0.0, 0.0]),
            'expected_range': (0.1, 0.3)
        },
        {
            'name': 'Large bounds',
            'B': np.array([[-2.0, 2.0], [-2.0, 2.0]]),
            'X': np.array([[0.0, 0.0]]),
            'u': np.array([0.0, 0.0]),
            'expected_range': (0.8, 1.0)
        }
    ]

    methods = {
        'T (default)': pomdp.T,
        'T_ot (OpenTURNS)': pomdp.T_ot,
        'T_fortran (mvnun)': pomdp.T_fortran,
    }

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Bounds: {test_case['B']}")
        print(f"State: {test_case['X']}")
        print(f"Action: {test_case['u']}")

        results = {}
        for method_name, method_func in methods.items():
            result = method_func(test_case['B'], test_case['X'], test_case['u'])
            results[method_name] = result[0]  # Single state result
            print(f"  {method_name:20s}: {result[0]:.6f}")

        # Check if results are consistent
        values = list(results.values())
        max_diff = max(values) - min(values)
        print(f"  Max difference: {max_diff:.2e}")

        # Check if within expected range
        for method_name, value in results.items():
            in_range = test_case['expected_range'][0] <= value <= test_case['expected_range'][1]
            print(f"  {method_name:20s}: {'✓' if in_range else '✗'} (in expected range)")


if __name__ == "__main__":
    # Run comprehensive tests
    results = test_transition_kernels()
    test_correctness()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
