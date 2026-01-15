"""
Multi-GPU utilities for parallel computation across multiple GPUs.

This module provides utilities for distributing work across multiple GPUs
using CuPy's device management capabilities.

Usage:
    # Enable global multi-GPU mode (automatic parallelization)
    from src.utils.multi_gpu import enable_multi_gpu, get_available_devices
    
    enable_multi_gpu()  # Automatically uses all available GPUs
    devices = get_available_devices()
    print(f"Available GPUs: {devices}")
    
    # Now operations automatically use multiple GPUs when beneficial
    # No need to manually manage devices!
"""

import os
from typing import List, Callable, Any, Tuple, Optional
import numpy as _numpy
from contextlib import contextmanager

try:
    import cupy as _cupy
    _cupy_available = True
except ImportError:
    _cupy_available = False
    _cupy = None

# Global multi-GPU state
_multi_gpu_enabled = False
_multi_gpu_devices = []
_multi_gpu_num_gpus = 0


def get_available_devices() -> List[int]:
    """
    Get list of available GPU device IDs.

    Returns:
        List of device IDs (e.g., [0, 1] for 2 GPUs)
    """
    if not _cupy_available:
        return []

    try:
        num_devices = _cupy.cuda.runtime.getDeviceCount()
        return list(range(num_devices))
    except Exception:
        return []


def get_num_gpus() -> int:
    """Get the number of available GPUs."""
    return len(get_available_devices())


def set_device(device_id: int) -> None:
    """
    Set the current CUDA device.

    Args:
        device_id: GPU device ID (0, 1, etc.)
    """
    if not _cupy_available:
        raise RuntimeError("CuPy not available, cannot set device")
    _cupy.cuda.Device(device_id).use()


def get_current_device() -> int:
    """Get the current CUDA device ID."""
    if not _cupy_available:
        return -1
    return _cupy.cuda.Device().id


def split_data_across_devices(data: Any, num_devices: int = None) -> List[Tuple[int, Any]]:
    """
    Split data across multiple devices for parallel processing.

    Args:
        data: Data to split (array, list, or other iterable)
        num_devices: Number of devices to use (default: all available)

    Returns:
        List of tuples (device_id, data_chunk) for each device
    """
    if not _cupy_available:
        return [(0, data)]

    devices = get_available_devices()
    if not devices:
        return [(0, data)]

    if num_devices is None:
        num_devices = len(devices)
    else:
        num_devices = min(num_devices, len(devices))

    # Convert to list if needed
    if hasattr(data, '__len__') and not isinstance(data, (str, bytes)):
        data_list = list(data) if not isinstance(data, _numpy.ndarray) else data
    else:
        # Single item - duplicate across devices
        return [(devices[i % len(devices)], data) for i in range(num_devices)]

    # Split data into chunks
    chunk_size = len(data_list) // num_devices
    chunks = []

    for i in range(num_devices):
        start_idx = i * chunk_size
        if i == num_devices - 1:
            # Last chunk gets remainder
            end_idx = len(data_list)
        else:
            end_idx = (i + 1) * chunk_size

        device_id = devices[i % len(devices)]
        chunk = data_list[start_idx:end_idx]
        chunks.append((device_id, chunk))

    return chunks


def process_on_device(device_id: int, func: Callable, *args, **kwargs) -> Any:
    """
    Execute a function on a specific GPU device.

    Args:
        device_id: GPU device ID
        func: Function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from func
    """
    if not _cupy_available:
        return func(*args, **kwargs)

    # Save current device
    current_device = get_current_device()

    try:
        # Switch to target device
        set_device(device_id)

        # Execute function
        result = func(*args, **kwargs)

        return result
    finally:
        # Restore original device
        if current_device >= 0:
            set_device(current_device)


def parallel_map(func: Callable, data_list: List[Any], num_workers: int = None) -> List[Any]:
    """
    Process a list of data items in parallel across multiple GPUs.

    Args:
        func: Function to apply to each data item
        data_list: List of data items to process
        num_workers: Number of GPUs to use (default: all available)

    Returns:
        List of results, one per input item
    """
    if not _cupy_available or not get_available_devices():
        # Fallback to sequential processing
        return [func(item) for item in data_list]

    devices = get_available_devices()
    if num_workers is None:
        num_workers = len(devices)
    else:
        num_workers = min(num_workers, len(devices))

    # Split data across devices
    chunks = split_data_across_devices(data_list, num_workers)

    # Process each chunk on its device
    results = []
    for device_id, chunk in chunks:
        chunk_results = process_on_device(device_id, lambda: [func(item) for item in chunk])
        results.extend(chunk_results)

    return results


def batch_process_across_gpus(func: Callable, batches: List[Any],
                              num_gpus: int = None) -> List[Any]:
    """
    Process batches in parallel across multiple GPUs.

    This is useful when you have multiple independent batches that can be
    processed simultaneously on different GPUs.

    Args:
        func: Function that processes a single batch
        batches: List of batches to process
        num_gpus: Number of GPUs to use (default: all available)

    Returns:
        List of results, one per batch
    """
    if not _cupy_available or not get_available_devices():
        # Fallback to sequential processing
        return [func(batch) for batch in batches]

    devices = get_available_devices()
    if num_gpus is None:
        num_gpus = len(devices)
    else:
        num_gpus = min(num_gpus, len(devices), len(batches))

    # Distribute batches across GPUs
    results = []
    for i, batch in enumerate(batches):
        device_id = devices[i % num_gpus]
        result = process_on_device(device_id, func, batch)
        results.append(result)

    return results


def configure_multi_gpu():
    """
    Configure environment for multi-GPU usage.

    Sets CUDA_VISIBLE_DEVICES if not already set and prints GPU info.
    """
    if not _cupy_available:
        print("⚠ CuPy not available, cannot configure multi-GPU")
        return

    devices = get_available_devices()
    num_gpus = len(devices)

    if num_gpus == 0:
        print("⚠ No GPUs detected")
        return

    print(f"✓ Found {num_gpus} GPU(s): {devices}")

    # Print device info
    for device_id in devices:
        set_device(device_id)
        device = _cupy.cuda.Device()
        mempool = _cupy.get_default_memory_pool()
        print(f"  GPU {device_id}: {device.compute_capability}, "
              f"Free memory: {mempool.free_bytes() / 1024**3:.2f} GB")

    # Set default device back to 0
    if devices:
        set_device(0)

    return devices


def enable_multi_gpu(num_gpus: Optional[int] = None, auto_configure: bool = True) -> bool:
    """
    Enable global multi-GPU mode for automatic parallelization.

    When enabled, operations that can benefit from multi-GPU will automatically
    distribute work across available GPUs without requiring manual device management.

    Args:
        num_gpus: Number of GPUs to use (default: all available)
        auto_configure: If True, automatically configure and print GPU info

    Returns:
        True if multi-GPU was successfully enabled, False otherwise

    Example:
        # Enable multi-GPU at the start of your script/notebook
        from src.utils.multi_gpu import enable_multi_gpu

        enable_multi_gpu()  # Uses all available GPUs automatically

        # Now operations automatically use multiple GPUs when beneficial
        # No need to manually manage devices!
    """
    global _multi_gpu_enabled, _multi_gpu_devices, _multi_gpu_num_gpus

    if not _cupy_available:
        print("⚠ CuPy not available, cannot enable multi-GPU")
        _multi_gpu_enabled = False
        return False

    devices = get_available_devices()
    if not devices:
        print("⚠ No GPUs detected, cannot enable multi-GPU")
        _multi_gpu_enabled = False
        return False

    if num_gpus is None:
        num_gpus = len(devices)
    else:
        num_gpus = min(num_gpus, len(devices))

    _multi_gpu_devices = devices[:num_gpus]
    _multi_gpu_num_gpus = num_gpus
    _multi_gpu_enabled = True

    if auto_configure:
        print(f"✓ Multi-GPU mode enabled: Using {num_gpus} GPU(s) {_multi_gpu_devices}")
        for device_id in _multi_gpu_devices:
            set_device(device_id)
            device = _cupy.cuda.Device()
            mempool = _cupy.get_default_memory_pool()
            print(f"  GPU {device_id}: {device.compute_capability}, "
                  f"Free memory: {mempool.free_bytes() / 1024**3:.2f} GB")
        # Reset to device 0
        if _multi_gpu_devices:
            set_device(0)

    return True


def disable_multi_gpu():
    """Disable global multi-GPU mode."""
    global _multi_gpu_enabled, _multi_gpu_devices, _multi_gpu_num_gpus
    _multi_gpu_enabled = False
    _multi_gpu_devices = []
    _multi_gpu_num_gpus = 0
    print("Multi-GPU mode disabled")


def is_multi_gpu_enabled() -> bool:
    """Check if global multi-GPU mode is enabled."""
    return _multi_gpu_enabled


def get_multi_gpu_devices() -> List[int]:
    """Get the list of devices used for multi-GPU operations."""
    if _multi_gpu_enabled:
        return _multi_gpu_devices.copy()
    return get_available_devices()


def auto_batch_process(func: Callable, batches: List[Any]) -> List[Any]:
    """
    Automatically process batches across GPUs if multi-GPU is enabled.

    If multi-GPU mode is enabled, distributes batches across GPUs.
    Otherwise, processes sequentially on the default device.

    Args:
        func: Function to process each batch
        batches: List of batches to process

    Returns:
        List of results, one per batch
    """
    if _multi_gpu_enabled and _multi_gpu_num_gpus > 1:
        return batch_process_across_gpus(func, batches, num_gpus=_multi_gpu_num_gpus)
    else:
        # Sequential processing
        return [func(batch) for batch in batches]


@contextmanager
def multi_gpu_context():
    """
    Context manager that temporarily enables multi-GPU mode.

    Example:
        with multi_gpu_context():
            # Operations here automatically use multiple GPUs
            results = process_batches(batches)
    """
    was_enabled = _multi_gpu_enabled
    if not was_enabled:
        enable_multi_gpu(auto_configure=False)
    try:
        yield
    finally:
        if not was_enabled:
            disable_multi_gpu()
