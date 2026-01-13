"""Tests for metrics module."""

import pytest
import time

from kvat.core.metrics import (
    TimingContext,
    MetricsCollector,
    calculate_score,
    is_dominated,
)
from kvat.core.schema import (
    WorkloadProfile,
    BenchmarkResult,
    CandidateConfig,
    Metrics,
    CacheStrategy,
    AttentionBackend,
    DType,
    ProfileType,
)


class TestTimingContext:
    """Tests for TimingContext."""

    def test_basic_timing(self):
        """Test basic timing operations."""
        ctx = TimingContext()

        ctx.start()
        time.sleep(0.01)  # 10ms
        ctx.mark_first_token()
        time.sleep(0.02)  # 20ms
        ctx.stop()

        assert ctx.ttft_ms > 5  # Should be ~10ms
        assert ctx.total_time_ms > 25  # Should be ~30ms

    def test_token_counting(self):
        """Test token counting."""
        ctx = TimingContext()

        ctx.start()
        for _ in range(10):
            ctx.mark_token()
        ctx.stop()

        assert ctx.tokens_generated == 10

    def test_throughput_calculation(self):
        """Test decode throughput calculation."""
        ctx = TimingContext()
        ctx.start_time = 0.0
        ctx.first_token_time = 0.1  # 100ms prefill
        ctx.end_time = 1.1  # 1000ms decode
        ctx.tokens_generated = 101  # 100 decode tokens + 1 first token

        # Should be ~100 tok/s
        assert 95 < ctx.decode_tokens_per_sec < 105


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_basic_collection(self):
        """Test basic metrics collection."""
        collector = MetricsCollector()

        collector.start()
        time.sleep(0.01)
        collector.mark_first_token()
        collector.set_tokens_generated(50)
        collector.stop()

        metrics = collector.get_metrics()

        assert metrics.ttft_ms > 0
        assert metrics.tokens_generated == 50
        assert metrics.error is None

    def test_error_handling(self):
        """Test error marking."""
        collector = MetricsCollector()

        collector.start()
        collector.mark_error("Out of memory")

        metrics = collector.get_metrics()

        assert metrics.error == "Out of memory"

    def test_timeout_handling(self):
        """Test timeout marking."""
        collector = MetricsCollector()

        collector.start()
        collector.mark_timeout()

        metrics = collector.get_metrics()

        assert metrics.timeout is True

    def test_context_manager(self):
        """Test context manager usage."""
        collector = MetricsCollector()

        with collector.measure():
            time.sleep(0.01)
            collector.mark_first_token()
            collector.set_tokens_generated(10)

        metrics = collector.get_metrics()

        assert metrics.total_time_ms > 0
        assert metrics.tokens_generated == 10


class TestScoring:
    """Tests for scoring functions."""

    @pytest.fixture
    def sample_profile(self):
        """Create a sample profile for testing."""
        return WorkloadProfile(
            name="test",
            profile_type=ProfileType.CUSTOM,
            weight_ttft=0.5,
            weight_throughput=0.35,
            weight_memory=0.15,
        )

    @pytest.fixture
    def sample_config(self):
        """Create a sample config for testing."""
        return CandidateConfig(
            cache_strategy=CacheStrategy.DYNAMIC,
            attention_backend=AttentionBackend.SDPA_FLASH,
            dtype=DType.FP16,
        )

    def test_score_calculation(self, sample_profile, sample_config):
        """Test basic score calculation."""
        result = BenchmarkResult(
            candidate=sample_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=100.0,
            throughput_mean=50.0,
            success_rate=1.0,
        )

        score = calculate_score(result, sample_profile)

        assert 0 <= score <= 100

    def test_score_zero_on_failure(self, sample_profile, sample_config):
        """Test that failed results get zero score."""
        result = BenchmarkResult(
            candidate=sample_config,
            context_length=2048,
            output_length=128,
            success_rate=0.0,
        )

        score = calculate_score(result, sample_profile)

        assert score == 0.0

    def test_score_penalizes_high_ttft(self, sample_profile, sample_config):
        """Test that high TTFT reduces score."""
        fast_result = BenchmarkResult(
            candidate=sample_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=50.0,
            throughput_mean=50.0,
            success_rate=1.0,
        )

        slow_result = BenchmarkResult(
            candidate=sample_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=500.0,
            throughput_mean=50.0,
            success_rate=1.0,
        )

        fast_score = calculate_score(fast_result, sample_profile)
        slow_score = calculate_score(slow_result, sample_profile)

        assert fast_score > slow_score


class TestDominance:
    """Tests for dominance checking."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        return CandidateConfig(
            cache_strategy=CacheStrategy.DYNAMIC,
            attention_backend=AttentionBackend.SDPA_FLASH,
            dtype=DType.FP16,
        )

    def test_dominated_result(self, base_config):
        """Test detecting a dominated result."""
        better = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=50.0,
            throughput_mean=100.0,
            peak_vram_mb=1000.0,
            success_rate=1.0,
        )

        worse = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=200.0,  # Worse TTFT
            throughput_mean=50.0,  # Worse throughput
            peak_vram_mb=2000.0,  # Worse VRAM
            success_rate=1.0,
        )

        assert is_dominated(worse, better)
        assert not is_dominated(better, worse)

    def test_non_dominated_tradeoff(self, base_config):
        """Test that tradeoffs are not dominated."""
        fast = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=30.0,  # Better TTFT
            throughput_mean=50.0,  # Worse throughput
            success_rate=1.0,
        )

        high_throughput = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=100.0,  # Worse TTFT
            throughput_mean=100.0,  # Better throughput
            success_rate=1.0,
        )

        # Neither should dominate the other
        assert not is_dominated(fast, high_throughput)
        assert not is_dominated(high_throughput, fast)

    def test_failed_result_dominated(self, base_config):
        """Test that failed results are always dominated."""
        successful = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            ttft_mean_ms=100.0,
            throughput_mean=50.0,
            success_rate=1.0,
        )

        failed = BenchmarkResult(
            candidate=base_config,
            context_length=2048,
            output_length=128,
            success_rate=0.0,
        )

        assert is_dominated(failed, successful)
