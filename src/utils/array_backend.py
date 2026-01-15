"""
Array backend utility for switching between NumPy and CuPy.

This module provides a unified interface for array operations that can
use either NumPy (CPU) or CuPy (GPU) based on availability and configuration.

Usage:
    from src.utils.array_backend import np, use_cupy, is_cupy
    
    # Use np as normal
    x = np.array([1, 2, 3])
    
    # Check if using CuPy
    if is_cupy:
        print("Using GPU acceleration")
    
    # Force CPU mode (for file I/O, etc.)
    import numpy as np_cpu
    np_cpu.savez(...)
"""

import numpy as _numpy
import os

# Check if CuPy should be used
USE_CUPY = os.getenv("USE_CUPY", "true").lower() in ("true", "1", "yes")

# Ensure CUDA_PATH is set if nvcc is available (for CuPy kernel compilation)
# This helps CuPy find CUDA headers when compiling kernels at runtime
if USE_CUPY and "CUDA_PATH" not in os.environ:
    # Try to find nvcc and infer CUDA_PATH
    import shutil
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # nvcc is typically in bin/nvcc, so CUDA_PATH is parent of bin
        potential_cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
        if os.path.exists(os.path.join(potential_cuda_path, "include", "cuda.h")):
            os.environ["CUDA_PATH"] = potential_cuda_path
            # Also set CUDA_HOME for compatibility
            if "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = potential_cuda_path
    else:
        # If nvcc is not in PATH, try to initialize module system and retry
        # This handles cases where modules need to be loaded first
        try:
            # Try to source /etc/profile and get environment variables
            import subprocess
            result = subprocess.run(
                'source /etc/profile && env',
                shell=True,
                executable='/bin/bash',
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Update environment with variables from profile
                for line in result.stdout.split('\n'):
                    if '=' in line and 'CUDA' in line:
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value
                # Retry finding nvcc
                nvcc_path = shutil.which("nvcc")
                if nvcc_path:
                    potential_cuda_path = os.path.dirname(os.path.dirname(nvcc_path))
                    if os.path.exists(os.path.join(potential_cuda_path, "include", "cuda.h")):
                        os.environ["CUDA_PATH"] = potential_cuda_path
                        if "CUDA_HOME" not in os.environ:
                            os.environ["CUDA_HOME"] = potential_cuda_path
        except Exception:
            # If module initialization fails, continue without it
            # The user should load modules before running Python
            pass

# Try to import CuPy
_cupy_available = False
_cupy = None

if USE_CUPY:
    try:
        import cupy as _cupy
        # Test if CuPy actually works by trying to create a small array
        # This catches cases where CuPy is installed but CUDA is not available
        try:
            _cupy.array([1, 2, 3])
            _cupy_available = True
            # Only print once when first imported
            import sys
            if not hasattr(sys.modules[__name__], '_cupy_imported'):
                print("✓ Using CuPy for GPU acceleration")
                sys.modules[__name__]._cupy_imported = True
        except Exception as e:
            # CuPy is installed but CUDA is not working (driver issues, etc.)
            _cupy_available = False
            import sys
            if not hasattr(sys.modules[__name__], '_cupy_runtime_error_printed'):
                print(f"⚠ CuPy installed but CUDA not available ({type(e).__name__}), falling back to NumPy")
                sys.modules[__name__]._cupy_runtime_error_printed = True
    except ImportError:
        _cupy_available = False
        # Only print once when first imported
        import sys
        if not hasattr(sys.modules[__name__], '_numpy_fallback_printed'):
            print("⚠ CuPy not available, falling back to NumPy")
            sys.modules[__name__]._numpy_fallback_printed = True

# Always import NumPy as fallback

# Export the appropriate backend
if _cupy_available:
    np = _cupy
    is_cupy = True
else:
    np = _numpy
    is_cupy = False

# Export utilities
def use_cupy():
    """Check if CuPy is being used."""
    return _cupy_available

# Random number generator - use appropriate backend
random = np.random

# Export multi-GPU utilities if CuPy is available
if _cupy_available:
    from .multi_gpu import (
        get_available_devices,
        get_num_gpus,
        set_device,
        get_current_device,
        configure_multi_gpu,
        batch_process_across_gpus,
        enable_multi_gpu,
        disable_multi_gpu,
        is_multi_gpu_enabled,
        get_multi_gpu_devices,
        auto_batch_process,
        multi_gpu_context
    )
