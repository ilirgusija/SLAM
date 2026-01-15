#!/usr/bin/env python3
"""
Quick test script to verify multi-GPU setup is working.

Run this before your main code to ensure both GPUs are configured correctly.
"""

import sys
import os

# Set CUDA_VISIBLE_DEVICES if not already set
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    print("⚠ CUDA_VISIBLE_DEVICES not set. Setting to 0,1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print("=" * 60)
print("Multi-GPU Configuration Test")
print("=" * 60)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Import and configure
try:
    from src.utils.array_backend import configure_multi_gpu, get_num_gpus, set_device
    from src.utils.multi_gpu import get_available_devices

    # Configure multi-GPU
    devices = configure_multi_gpu()
    num_gpus = get_num_gpus()

    if num_gpus < 2:
        print(f"\n❌ Only {num_gpus} GPU(s) detected!")
        print("\nTroubleshooting:")
        print("1. Check nvidia-smi shows multiple GPUs")
        print("2. Verify CUDA_VISIBLE_DEVICES=0,1 is set")
        print("3. Ensure CuPy can access both devices")
        sys.exit(1)

    print(f"\n✓ {num_gpus} GPUs detected: {devices}")

    # Test basic operations on each GPU
    print("\n" + "=" * 60)
    print("Testing GPU Operations")
    print("=" * 60)

    import cupy as cp
    import numpy as np

    for device_id in devices:
        set_device(device_id)
        print(f"\nGPU {device_id}:")

        # Create array
        x = cp.array([1, 2, 3, 4, 5])
        print(f"  Created array: {x.get()}")

        # Simple computation
        result = cp.sum(x ** 2)
        print(f"  Sum of squares: {result.get()}")

        # Memory info
        mempool = cp.get_default_memory_pool()
        print(f"  Free memory: {mempool.free_bytes() / 1024**3:.2f} GB")

    # Test parallel processing
    print("\n" + "=" * 60)
    print("Testing Parallel Processing")
    print("=" * 60)

    from src.utils.array_backend import batch_process_across_gpus, get_current_device

    def test_batch(batch_data):
        """Test function for batch processing."""
        # Device is already set by batch_process_across_gpus
        x = cp.array(batch_data)
        return float(cp.sum(x ** 2).get())

    # Create test batches
    batches = [np.random.randn(100, 100) for _ in range(4)]
    print(f"Processing {len(batches)} batches across {num_gpus} GPUs...")

    results = batch_process_across_gpus(test_batch, batches, num_gpus=num_gpus)
    print(f"Results: {[f'{r:.2e}' for r in results]}")

    print("\n" + "=" * 60)
    print("✓ Multi-GPU setup verified!")
    print("=" * 60)
    print("\nYou can now use multi-GPU in your code:")
    print("  from src.utils.array_backend import configure_multi_gpu, batch_process_across_gpus")
    print("  configure_multi_gpu()")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Make sure you're in the SLAM directory and dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
