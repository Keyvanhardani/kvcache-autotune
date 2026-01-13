"""
CPU and system memory probes.

Monitors:
- Process RSS (Resident Set Size)
- System memory availability
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SystemRAMInfo:
    """System RAM information."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float


def get_process_ram_mb() -> float:
    """
    Get current process RAM usage in MB.

    Uses psutil if available, falls back to /proc on Linux.

    Returns:
        Process RSS in megabytes
    """
    # Try psutil first
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    # Fallback: /proc/self/status on Linux
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return float(line.split()[1]) / 1024
    except (OSError, FileNotFoundError):
        pass

    # Windows fallback via ctypes
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        process_handle = kernel32.GetCurrentProcess()

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        psapi = ctypes.windll.psapi
        if psapi.GetProcessMemoryInfo(
            process_handle,
            ctypes.byref(counters),
            ctypes.sizeof(counters)
        ):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass

    return 0.0


def get_system_ram_info() -> SystemRAMInfo | None:
    """
    Get system-wide RAM information.

    Returns:
        SystemRAMInfo or None if unavailable
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return SystemRAMInfo(
            total_mb=mem.total / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            used_mb=mem.used / (1024 * 1024),
            percent_used=mem.percent,
        )
    except ImportError:
        pass

    # Linux fallback: /proc/meminfo
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = float(parts[1])  # kB
                    meminfo[key] = value

        total = meminfo.get("MemTotal", 0) / 1024  # MB
        available = meminfo.get("MemAvailable", 0) / 1024
        used = total - available

        return SystemRAMInfo(
            total_mb=total,
            available_mb=available,
            used_mb=used,
            percent_used=(used / total * 100) if total > 0 else 0,
        )
    except (OSError, FileNotFoundError, KeyError):
        pass

    return None


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def get_cpu_percent() -> float:
    """Get current CPU utilization percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return 0.0


def create_ram_probe():
    """
    Create a RAM probe function for MetricsCollector.

    Returns:
        Callable that returns current process RAM in MB
    """
    return get_process_ram_mb
