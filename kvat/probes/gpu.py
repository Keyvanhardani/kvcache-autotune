"""
GPU resource probes for VRAM monitoring.

Supports:
- PyTorch CUDA memory tracking
- NVML for detailed GPU info (optional)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    total_memory_mb: float
    compute_capability: tuple[int, int]
    driver_version: str | None = None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_device_count() -> int:
    """Get number of CUDA devices."""
    if not is_cuda_available():
        return 0
    import torch
    return torch.cuda.device_count()


def get_cuda_memory_mb(device: int = 0) -> float:
    """
    Get current CUDA memory allocation in MB.

    Args:
        device: CUDA device index

    Returns:
        Current allocated memory in megabytes
    """
    if not is_cuda_available():
        return 0.0

    import torch
    return torch.cuda.memory_allocated(device) / (1024 * 1024)


def get_cuda_max_memory_mb(device: int = 0) -> float:
    """
    Get peak CUDA memory allocation in MB.

    Args:
        device: CUDA device index

    Returns:
        Peak allocated memory in megabytes
    """
    if not is_cuda_available():
        return 0.0

    import torch
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def get_cuda_reserved_memory_mb(device: int = 0) -> float:
    """
    Get reserved (cached) CUDA memory in MB.

    Args:
        device: CUDA device index

    Returns:
        Reserved memory in megabytes
    """
    if not is_cuda_available():
        return 0.0

    import torch
    return torch.cuda.memory_reserved(device) / (1024 * 1024)


def reset_cuda_peak_memory(device: int = 0) -> None:
    """
    Reset CUDA peak memory statistics.

    Args:
        device: CUDA device index
    """
    if not is_cuda_available():
        return

    import torch
    torch.cuda.reset_peak_memory_stats(device)


def empty_cuda_cache() -> None:
    """Empty CUDA cache to free unused memory."""
    if not is_cuda_available():
        return

    import torch
    torch.cuda.empty_cache()


def synchronize_cuda(device: int = 0) -> None:
    """Synchronize CUDA device."""
    if not is_cuda_available():
        return

    import torch
    torch.cuda.synchronize(device)


def get_gpu_info(device: int = 0) -> GPUInfo | None:
    """
    Get detailed GPU information.

    Args:
        device: CUDA device index

    Returns:
        GPUInfo if available, None otherwise
    """
    if not is_cuda_available():
        return None

    import torch

    if device >= torch.cuda.device_count():
        return None

    props = torch.cuda.get_device_properties(device)

    return GPUInfo(
        index=device,
        name=props.name,
        total_memory_mb=props.total_memory / (1024 * 1024),
        compute_capability=(props.major, props.minor),
        driver_version=None,  # Would need NVML for this
    )


def get_all_gpu_info() -> list[GPUInfo]:
    """Get info for all available GPUs."""
    count = get_cuda_device_count()
    return [
        info for i in range(count)
        if (info := get_gpu_info(i)) is not None
    ]


def get_cuda_memory_summary(device: int = 0) -> dict[str, float]:
    """
    Get comprehensive CUDA memory summary.

    Args:
        device: CUDA device index

    Returns:
        Dictionary with memory statistics
    """
    if not is_cuda_available():
        return {}

    import torch

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 * 1024),
    }


# =============================================================================
# NVML Support (Optional)
# =============================================================================

_nvml_initialized = False


def _init_nvml() -> bool:
    """Initialize NVML library."""
    global _nvml_initialized
    if _nvml_initialized:
        return True

    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialized = True
        return True
    except (ImportError, Exception):
        return False


def get_nvml_memory_info(device: int = 0) -> dict[str, float] | None:
    """
    Get GPU memory info via NVML.

    More accurate than PyTorch for total/free memory.

    Args:
        device: GPU device index

    Returns:
        Memory info dict or None if NVML unavailable
    """
    if not _init_nvml():
        return None

    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return {
            "total_mb": info.total / (1024 * 1024),
            "used_mb": info.used / (1024 * 1024),
            "free_mb": info.free / (1024 * 1024),
        }
    except Exception:
        return None


def get_nvml_utilization(device: int = 0) -> dict[str, int] | None:
    """
    Get GPU utilization via NVML.

    Args:
        device: GPU device index

    Returns:
        Utilization dict (gpu%, memory%) or None
    """
    if not _init_nvml():
        return None

    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        return {
            "gpu_percent": util.gpu,
            "memory_percent": util.memory,
        }
    except Exception:
        return None


def create_vram_probe(device: int = 0):
    """
    Create a VRAM probe function for MetricsCollector.

    Args:
        device: CUDA device index

    Returns:
        Callable that returns current VRAM in MB
    """
    def probe() -> float:
        return get_cuda_max_memory_mb(device)

    return probe
