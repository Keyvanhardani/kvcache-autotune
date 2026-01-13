"""
KVCache Auto-Tuner - Automatic KV-Cache Optimization for Transformers

A production-ready plugin that automatically finds the optimal KV-cache
configuration for any model + hardware + workload combination.
"""

__version__ = "0.1.0"
__author__ = "KVCache Auto-Tuner Contributors"

from kvat.core.profiles import get_profile, list_profiles
from kvat.core.schema import (
    AttentionBackend,
    CacheStrategy,
    TuneConfig,
    TuneResult,
    WorkloadProfile,
)
from kvat.engines.base import EngineAdapter

__all__ = [
    "__version__",
    "TuneConfig",
    "TuneResult",
    "CacheStrategy",
    "AttentionBackend",
    "WorkloadProfile",
    "EngineAdapter",
    "get_profile",
    "list_profiles",
]
