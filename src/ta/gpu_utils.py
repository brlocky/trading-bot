"""
GPU Acceleration Utilities
===========================

Provides GPU-accelerated operations for TA calculations with automatic CPU fallback.

Supported backends:
1. CuPy (CUDA) - Best performance, requires NVIDIA GPU
2. NumPy (CPU) - Always available fallback

Operations accelerated:
- Vectorized price comparisons
- Rolling calculations
- Distance computations
- Binning/histograms for volume profiles
- Array filtering and masking
"""

import numpy as np
from typing import Tuple
import warnings

# Try to import GPU libraries
GPU_AVAILABLE = False
GPU_BACKEND = "numpy"

try:
    import cupy as cp
    # Suppress CUDA path warning - it's not critical
    import warnings
    warnings.filterwarnings('ignore', message='CUDA path could not be detected')

    # Test basic GPU operations (avoid compilation-dependent ops)
    test_arr = cp.array([1.0, 2.0, 3.0])
    _ = cp.sum(test_arr)  # Basic operation that doesn't need NVRTC

    GPU_AVAILABLE = True
    GPU_BACKEND = "cupy"
    print("✅ GPU acceleration available (CuPy)")
except (ImportError, Exception) as e:
    cp = None
    GPU_AVAILABLE = False
    GPU_BACKEND = "numpy"
    if isinstance(e, ImportError):
        print("ℹ️  GPU acceleration not available - using NumPy (CPU only)")
    else:
        print(f"⚠️  GPU found but CuPy failed ({type(e).__name__}) - using NumPy (CPU only)")


class GPUArrayOps:
    """
    Unified interface for array operations with automatic GPU/CPU selection.

    Usage:
        ops = GPUArrayOps()
        result = ops.sum(array)  # Automatically uses GPU if available
    """

    def __init__(self, force_cpu: bool = False):
        """
        Args:
            force_cpu: Force CPU even if GPU is available (for testing/debugging)
        """
        self.use_gpu = GPU_AVAILABLE and not force_cpu
        self.backend = cp if self.use_gpu else np

    def asarray(self, arr):
        """Convert array to GPU/CPU array"""
        return self.backend.asarray(arr)

    def asnumpy(self, arr):
        """Convert back to numpy array"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def sum(self, arr, axis=None):
        """Sum array elements"""
        return self.backend.sum(arr, axis=axis)

    def mean(self, arr, axis=None):
        """Mean of array elements"""
        return self.backend.mean(arr, axis=axis)

    def max(self, arr, axis=None):
        """Maximum value"""
        return self.backend.max(arr, axis=axis)

    def min(self, arr, axis=None):
        """Minimum value"""
        return self.backend.min(arr, axis=axis)

    def argmax(self, arr, axis=None):
        """Index of maximum value"""
        return self.backend.argmax(arr, axis=axis)

    def argmin(self, arr, axis=None):
        """Index of minimum value"""
        return self.backend.argmin(arr, axis=axis)

    def where(self, condition, x=None, y=None):
        """Return indices where condition is true, or conditional selection"""
        if x is None and y is None:
            return self.backend.where(condition)
        return self.backend.where(condition, x, y)

    def abs(self, arr):
        """Absolute value"""
        return self.backend.abs(arr)

    def log(self, arr):
        """Natural logarithm"""
        return self.backend.log(arr)

    def exp(self, arr):
        """Exponential"""
        return self.backend.exp(arr)

    def digitize(self, arr, bins):
        """Return indices of bins to which each value belongs"""
        return self.backend.digitize(arr, bins)

    def bincount(self, arr, weights=None, minlength=0):
        """Count occurrences of each value"""
        return self.backend.bincount(arr, weights=weights, minlength=minlength)

    def clip(self, arr, min_val, max_val):
        """Clip array values"""
        return self.backend.clip(arr, min_val, max_val)

    def arange(self, *args, **kwargs):
        """Create range array"""
        return self.backend.arange(*args, **kwargs)

    def zeros(self, shape, dtype=None):
        """Create array of zeros"""
        return self.backend.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        """Create array of ones"""
        return self.backend.ones(shape, dtype=dtype)


def calculate_distances_gpu(
    prices: np.ndarray,
    level_prices: np.ndarray,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Calculate distances from each price to each level (vectorized).

    Args:
        prices: Array of shape (N,) - current prices
        level_prices: Array of shape (M,) - level prices
        use_gpu: Whether to use GPU if available

    Returns:
        Array of shape (N, M) - distances[i, j] = distance from price i to level j
    """
    ops = GPUArrayOps(force_cpu=not use_gpu)

    # Move to GPU
    prices_gpu = ops.asarray(prices.reshape(-1, 1))
    levels_gpu = ops.asarray(level_prices.reshape(1, -1))

    # Vectorized distance calculation
    distances = ops.abs(prices_gpu - levels_gpu)

    # Move back to CPU
    return ops.asnumpy(distances)


def calculate_volume_profile_gpu(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 100,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate volume profile using GPU acceleration.

    Args:
        prices: Price array
        volumes: Volume array
        bins: Number of price bins
        use_gpu: Whether to use GPU

    Returns:
        (bin_edges, bin_volumes) tuple
    """
    ops = GPUArrayOps(force_cpu=not use_gpu)

    # Move to GPU
    prices_gpu = ops.asarray(prices)
    volumes_gpu = ops.asarray(volumes)

    # Calculate bin edges
    min_price = float(ops.min(prices_gpu))
    max_price = float(ops.max(prices_gpu))
    price_range = max_price - min_price

    if price_range == 0.0:
        # All prices the same
        bin_edges = np.array([min_price, min_price])
        bin_volumes = np.array([float(ops.sum(volumes_gpu))])
        return bin_edges, bin_volumes

    bin_size = price_range / bins
    bin_edges_gpu = ops.arange(min_price, max_price + bin_size, bin_size)

    # Vectorized binning
    bin_indices = ops.digitize(prices_gpu, bin_edges_gpu) - 1
    bin_indices = ops.clip(bin_indices, 0, len(bin_edges_gpu) - 2)

    # Fast histogram
    bin_volumes_gpu = ops.bincount(
        bin_indices,
        weights=volumes_gpu,
        minlength=len(bin_edges_gpu) - 1
    )

    # Move back to CPU
    return ops.asnumpy(bin_edges_gpu), ops.asnumpy(bin_volumes_gpu)


def rolling_calculation_gpu(
    arr: np.ndarray,
    window: int,
    operation: str = 'mean',
    use_gpu: bool = True
) -> np.ndarray:
    """
    Perform rolling calculations with GPU acceleration.

    Args:
        arr: Input array
        window: Rolling window size
        operation: 'mean', 'sum', 'max', 'min'
        use_gpu: Whether to use GPU

    Returns:
        Result array (same shape as input)
    """
    ops = GPUArrayOps(force_cpu=not use_gpu)

    if len(arr) < window:
        warnings.warn(f"Array length {len(arr)} < window {window}")
        return arr

    # Move to GPU
    arr_gpu = ops.asarray(arr)

    # For now, use simple implementation
    # TODO: Implement optimized GPU rolling window using convolution
    result = np.zeros_like(arr)

    for i in range(len(arr)):
        start = max(0, i - window + 1)
        end = i + 1
        window_data = arr_gpu[start:end]

        if operation == 'mean':
            result[i] = float(ops.mean(window_data))
        elif operation == 'sum':
            result[i] = float(ops.sum(window_data))
        elif operation == 'max':
            result[i] = float(ops.max(window_data))
        elif operation == 'min':
            result[i] = float(ops.min(window_data))

    return result


def batch_process_timeframes(
    data_dict: dict,
    process_func,
    use_gpu: bool = True
) -> dict:
    """
    Process multiple timeframes in parallel using GPU batching.

    This can significantly speed up processing when you have multiple
    timeframes to calculate at once.

    Args:
        data_dict: {timeframe: data} mapping
        process_func: Function to process each dataset
        use_gpu: Whether to use GPU

    Returns:
        {timeframe: result} mapping
    """
    # For now, process sequentially
    # TODO: Implement true parallel GPU processing
    results = {}
    for timeframe, data in data_dict.items():
        results[timeframe] = process_func(data, use_gpu=use_gpu)
    return results


def get_gpu_info() -> dict:
    """Get information about GPU availability and capabilities"""
    info = {
        'available': GPU_AVAILABLE,
        'backend': GPU_BACKEND,
    }

    if GPU_AVAILABLE and cp:
        try:
            device = cp.cuda.Device()
            info.update({
                'name': device.attributes.get('Name', 'Unknown'),
                'compute_capability': device.compute_capability,
                'memory_total': device.mem_info[1] / 1024**3,  # GB
                'memory_free': device.mem_info[0] / 1024**3,   # GB
            })
        except Exception as e:
            info['error'] = str(e)

    return info


# Global singleton instance
_default_ops = GPUArrayOps()


def get_array_ops(force_cpu: bool = False) -> GPUArrayOps:
    """Get default array operations instance"""
    if force_cpu:
        return GPUArrayOps(force_cpu=True)
    return _default_ops
