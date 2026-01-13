"""
Metrics collection and scoring for KVCache Auto-Tuner.

Handles:
- Time measurement (TTFT, decode throughput)
- Resource monitoring (VRAM, RAM)
- Score calculation with weighted profiles
"""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from kvat.core.schema import BenchmarkResult, Metrics, WorkloadProfile


@dataclass
class TimingContext:
    """Context for timing a generation run."""

    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float = 0.0
    tokens_generated: int = 0

    def start(self) -> None:
        """Mark the start of generation."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = 0.0
        self.tokens_generated = 0

    def mark_first_token(self) -> None:
        """Mark when the first token is generated."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def mark_token(self) -> None:
        """Mark each generated token."""
        self.tokens_generated += 1
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()

    def stop(self) -> None:
        """Mark the end of generation."""
        self.end_time = time.perf_counter()

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_time_ms(self) -> float:
        """Total generation time in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def decode_time_ms(self) -> float:
        """Decode phase time in milliseconds."""
        if self.first_token_time is None:
            return 0.0
        return (self.end_time - self.first_token_time) * 1000

    @property
    def decode_tokens_per_sec(self) -> float:
        """Decode throughput in tokens per second."""
        decode_time_sec = self.decode_time_ms / 1000
        if decode_time_sec <= 0 or self.tokens_generated <= 1:
            return 0.0
        # Subtract 1 because first token is part of prefill
        return (self.tokens_generated - 1) / decode_time_sec


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""

    vram_mb: float | None = None
    ram_mb: float | None = None
    gpu_utilization: float | None = None
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collects and aggregates metrics during benchmark runs.

    Thread-safe collection of timing and resource metrics
    with support for streaming token callbacks.
    """

    def __init__(self) -> None:
        self._timing = TimingContext()
        self._peak_vram_mb: float | None = None
        self._peak_ram_mb: float | None = None
        self._error: str | None = None
        self._timeout: bool = False

        # Callbacks for resource probes
        self._vram_probe: Callable[[], float] | None = None
        self._ram_probe: Callable[[], float] | None = None

    def set_vram_probe(self, probe: Callable[[], float]) -> None:
        """Set the VRAM measurement probe."""
        self._vram_probe = probe

    def set_ram_probe(self, probe: Callable[[], float]) -> None:
        """Set the RAM measurement probe."""
        self._ram_probe = probe

    def reset(self) -> None:
        """Reset all metrics for a new run."""
        self._timing = TimingContext()
        self._peak_vram_mb = None
        self._peak_ram_mb = None
        self._error = None
        self._timeout = False
        gc.collect()

    def start(self) -> None:
        """Start timing a generation run."""
        self._sample_resources()
        self._timing.start()

    def mark_first_token(self) -> None:
        """Mark first token generation."""
        self._timing.mark_first_token()
        self._sample_resources()

    def mark_token(self) -> None:
        """Mark token generation (for streaming)."""
        self._timing.mark_token()

    def stop(self) -> None:
        """Stop timing and collect final resources."""
        self._timing.stop()
        self._sample_resources()

    def set_tokens_generated(self, count: int) -> None:
        """Set the total tokens generated (if not using streaming)."""
        self._timing.tokens_generated = count

    def mark_error(self, error: str) -> None:
        """Mark that an error occurred."""
        self._error = error
        self._timing.stop()

    def mark_timeout(self) -> None:
        """Mark that the run timed out."""
        self._timeout = True
        self._timing.stop()

    def _sample_resources(self) -> None:
        """Sample current resource usage."""
        if self._vram_probe is not None:
            try:
                vram = self._vram_probe()
                if self._peak_vram_mb is None or vram > self._peak_vram_mb:
                    self._peak_vram_mb = vram
            except Exception:
                pass

        if self._ram_probe is not None:
            try:
                ram = self._ram_probe()
                if self._peak_ram_mb is None or ram > self._peak_ram_mb:
                    self._peak_ram_mb = ram
            except Exception:
                pass

    def get_metrics(self) -> Metrics:
        """Get the collected metrics."""
        return Metrics(
            ttft_ms=self._timing.ttft_ms,
            decode_tokens_per_sec=self._timing.decode_tokens_per_sec,
            total_time_ms=self._timing.total_time_ms,
            tokens_generated=self._timing.tokens_generated,
            peak_vram_mb=self._peak_vram_mb,
            peak_ram_mb=self._peak_ram_mb,
            error=self._error,
            timeout=self._timeout,
        )

    @contextmanager
    def measure(self):
        """Context manager for measuring a generation run."""
        self.reset()
        self.start()
        try:
            yield self
        except Exception as e:
            self.mark_error(str(e))
            raise
        finally:
            self.stop()


# =============================================================================
# Scoring Functions
# =============================================================================

def calculate_score(
    result: BenchmarkResult,
    profile: WorkloadProfile,
    baseline_ttft: float | None = None,
    baseline_throughput: float | None = None,
) -> float:
    """
    Calculate weighted score for a benchmark result.

    Uses profile-specific weights to balance:
    - Time to first token (TTFT)
    - Decode throughput
    - Memory efficiency / stability

    Args:
        result: Benchmark result to score
        profile: Workload profile with weights
        baseline_ttft: Optional baseline TTFT for normalization
        baseline_throughput: Optional baseline throughput for normalization

    Returns:
        Normalized score (higher is better, 0-100 scale)
    """
    if result.success_rate == 0:
        return 0.0

    # Normalize metrics to 0-1 scale
    # For TTFT, lower is better (invert)
    ttft_score = _normalize_ttft(result.ttft_mean_ms, baseline_ttft)

    # For throughput, higher is better
    throughput_score = _normalize_throughput(
        result.throughput_mean,
        baseline_throughput
    )

    # Memory score (combination of efficiency and stability)
    memory_score = _calculate_memory_score(result)

    # Apply profile weights
    weighted_score = (
        profile.weight_ttft * ttft_score +
        profile.weight_throughput * throughput_score +
        profile.weight_memory * memory_score
    )

    # Apply stability penalty
    stability_factor = result.success_rate

    return weighted_score * stability_factor * 100


def _normalize_ttft(ttft_ms: float, baseline: float | None = None) -> float:
    """Normalize TTFT score (lower is better)."""
    if ttft_ms <= 0:
        return 0.0

    if baseline is not None and baseline > 0:
        # Score relative to baseline: 1.0 at baseline, higher if faster
        return min(2.0, baseline / ttft_ms)
    else:
        # Heuristic: <100ms excellent, >2000ms poor
        if ttft_ms < 100:
            return 1.0
        elif ttft_ms > 2000:
            return 0.1
        else:
            # Linear interpolation
            return 1.0 - (ttft_ms - 100) / 1900 * 0.9


def _normalize_throughput(
    throughput: float,
    baseline: float | None = None
) -> float:
    """Normalize throughput score (higher is better)."""
    if throughput <= 0:
        return 0.0

    if baseline is not None and baseline > 0:
        # Score relative to baseline
        return min(2.0, throughput / baseline)
    else:
        # Heuristic: >100 tok/s excellent, <5 tok/s poor
        if throughput >= 100:
            return 1.0
        elif throughput < 5:
            return 0.1
        else:
            return 0.1 + (throughput - 5) / 95 * 0.9


def _calculate_memory_score(result: BenchmarkResult) -> float:
    """
    Calculate memory efficiency score.

    Factors:
    - Success rate (stability)
    - VRAM efficiency
    - Variance in performance (jitter)
    """
    score = 1.0

    # Penalize high variance (jitter)
    if result.ttft_std_ms > 0 and result.ttft_mean_ms > 0:
        cv = result.ttft_std_ms / result.ttft_mean_ms  # Coefficient of variation
        if cv > 0.3:  # High jitter
            score *= 0.8
        elif cv > 0.5:
            score *= 0.6

    # Success rate is already factored in main score
    # but we can give bonus for 100% success
    if result.success_rate == 1.0:
        score *= 1.05

    return min(1.0, score)


def compare_results(
    results: list[BenchmarkResult],
    profile: WorkloadProfile,
) -> list[tuple[BenchmarkResult, float]]:
    """
    Compare and rank benchmark results.

    Args:
        results: List of benchmark results
        profile: Workload profile for scoring

    Returns:
        List of (result, score) tuples, sorted by score descending
    """
    if not results:
        return []

    # Find baselines from first result or median
    baseline_ttft = None
    baseline_throughput = None

    valid_results = [r for r in results if r.success_rate > 0]
    if valid_results:
        ttfts = sorted([r.ttft_mean_ms for r in valid_results])
        throughputs = sorted([r.throughput_mean for r in valid_results])

        baseline_ttft = ttfts[len(ttfts) // 2]  # Median
        baseline_throughput = throughputs[len(throughputs) // 2]

    # Score all results
    scored = []
    for result in results:
        score = calculate_score(result, profile, baseline_ttft, baseline_throughput)
        result.score = score
        scored.append((result, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


def is_dominated(
    candidate: BenchmarkResult,
    other: BenchmarkResult,
    tolerance: float = 0.05,
) -> bool:
    """
    Check if candidate is dominated by other result.

    A result is dominated if it's worse in all metrics.

    Args:
        candidate: Candidate to check
        other: Result to compare against
        tolerance: Relative tolerance for "worse"

    Returns:
        True if candidate is dominated
    """
    if candidate.success_rate == 0:
        return other.success_rate > 0

    if other.success_rate == 0:
        return False

    # Check TTFT (lower is better)
    ttft_worse = candidate.ttft_mean_ms > other.ttft_mean_ms * (1 + tolerance)

    # Check throughput (higher is better)
    throughput_worse = candidate.throughput_mean < other.throughput_mean * (1 - tolerance)

    # Check VRAM (lower is better, if available)
    vram_worse = True
    if candidate.peak_vram_mb is not None and other.peak_vram_mb is not None:
        vram_worse = candidate.peak_vram_mb > other.peak_vram_mb * (1 + tolerance)

    return ttft_worse and throughput_worse and vram_worse
