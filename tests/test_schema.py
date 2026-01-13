"""Tests for schema module."""

import pytest
from pydantic import ValidationError

from kvat.core.schema import (
    WorkloadProfile,
    CandidateConfig,
    BenchmarkResult,
    Metrics,
    CacheStrategy,
    AttentionBackend,
    DType,
    ProfileType,
)


class TestWorkloadProfile:
    """Tests for WorkloadProfile schema."""

    def test_valid_profile_creation(self):
        """Test creating a valid profile."""
        profile = WorkloadProfile(
            name="test-profile",
            profile_type=ProfileType.CUSTOM,
            context_lengths=[2048, 4096],
            output_lengths=[128, 256],
            weight_ttft=0.5,
            weight_throughput=0.35,
            weight_memory=0.15,
        )

        assert profile.name == "test-profile"
        assert profile.context_lengths == [2048, 4096]
        assert profile.weight_ttft == 0.5

    def test_weights_must_sum_to_one(self):
        """Test that weights validation works."""
        with pytest.raises(ValidationError) as exc_info:
            WorkloadProfile(
                name="bad-profile",
                profile_type=ProfileType.CUSTOM,
                weight_ttft=0.5,
                weight_throughput=0.5,
                weight_memory=0.5,  # Sum = 1.5, should fail
            )

        assert "Weights must sum to 1.0" in str(exc_info.value)

    def test_default_values(self):
        """Test default values are applied."""
        profile = WorkloadProfile(
            name="minimal",
            profile_type=ProfileType.CHAT_AGENT,
            weight_ttft=0.5,
            weight_throughput=0.35,
            weight_memory=0.15,
        )

        assert profile.warmup_runs == 2
        assert profile.measurement_runs == 5


class TestCandidateConfig:
    """Tests for CandidateConfig schema."""

    def test_candidate_creation(self):
        """Test creating a candidate config."""
        config = CandidateConfig(
            cache_strategy=CacheStrategy.DYNAMIC,
            attention_backend=AttentionBackend.SDPA_FLASH,
            dtype=DType.FP16,
        )

        assert config.cache_strategy == CacheStrategy.DYNAMIC
        assert config.use_torch_compile is False

    def test_to_identifier(self):
        """Test identifier generation."""
        config = CandidateConfig(
            cache_strategy=CacheStrategy.STATIC,
            attention_backend=AttentionBackend.FLASH_ATTENTION,
            dtype=DType.BF16,
            use_torch_compile=True,
        )

        identifier = config.to_identifier()

        assert "cache=static" in identifier
        assert "attn=flash_attention" in identifier
        assert "dtype=bfloat16" in identifier
        assert "compile=true" in identifier

    def test_identifier_uniqueness(self):
        """Test that different configs have different identifiers."""
        config1 = CandidateConfig(
            cache_strategy=CacheStrategy.DYNAMIC,
            attention_backend=AttentionBackend.SDPA_FLASH,
            dtype=DType.FP16,
        )

        config2 = CandidateConfig(
            cache_strategy=CacheStrategy.STATIC,
            attention_backend=AttentionBackend.SDPA_FLASH,
            dtype=DType.FP16,
        )

        assert config1.to_identifier() != config2.to_identifier()


class TestBenchmarkResult:
    """Tests for BenchmarkResult schema."""

    def test_compute_statistics_empty(self):
        """Test statistics computation with no runs."""
        result = BenchmarkResult(
            candidate=CandidateConfig(
                cache_strategy=CacheStrategy.DYNAMIC,
                attention_backend=AttentionBackend.SDPA_FLASH,
                dtype=DType.FP16,
            ),
            context_length=2048,
            output_length=128,
        )

        result.compute_statistics()

        assert result.ttft_mean_ms == 0.0
        assert result.success_rate == 1.0  # Default

    def test_compute_statistics_with_runs(self):
        """Test statistics computation with successful runs."""
        result = BenchmarkResult(
            candidate=CandidateConfig(
                cache_strategy=CacheStrategy.DYNAMIC,
                attention_backend=AttentionBackend.SDPA_FLASH,
                dtype=DType.FP16,
            ),
            context_length=2048,
            output_length=128,
            runs=[
                Metrics(
                    ttft_ms=100.0,
                    decode_tokens_per_sec=50.0,
                    total_time_ms=1000.0,
                    tokens_generated=128,
                ),
                Metrics(
                    ttft_ms=110.0,
                    decode_tokens_per_sec=48.0,
                    total_time_ms=1100.0,
                    tokens_generated=128,
                ),
            ],
        )

        result.compute_statistics()

        assert result.ttft_mean_ms == 105.0
        assert result.throughput_mean == 49.0
        assert result.success_rate == 1.0

    def test_compute_statistics_with_errors(self):
        """Test statistics computation with some failed runs."""
        result = BenchmarkResult(
            candidate=CandidateConfig(
                cache_strategy=CacheStrategy.DYNAMIC,
                attention_backend=AttentionBackend.SDPA_FLASH,
                dtype=DType.FP16,
            ),
            context_length=2048,
            output_length=128,
            runs=[
                Metrics(
                    ttft_ms=100.0,
                    decode_tokens_per_sec=50.0,
                    total_time_ms=1000.0,
                    tokens_generated=128,
                ),
                Metrics(
                    ttft_ms=0.0,
                    decode_tokens_per_sec=0.0,
                    total_time_ms=0.0,
                    tokens_generated=0,
                    error="OOM",
                ),
            ],
        )

        result.compute_statistics()

        # Only successful run should be counted
        assert result.ttft_mean_ms == 100.0
        assert result.success_rate == 0.5


class TestEnums:
    """Tests for enum values."""

    def test_cache_strategy_values(self):
        """Test CacheStrategy enum values."""
        assert CacheStrategy.DYNAMIC.value == "dynamic"
        assert CacheStrategy.STATIC.value == "static"
        assert CacheStrategy.SLIDING_WINDOW.value == "sliding_window"

    def test_attention_backend_values(self):
        """Test AttentionBackend enum values."""
        assert AttentionBackend.SDPA_FLASH.value == "sdpa_flash"
        assert AttentionBackend.FLASH_ATTENTION.value == "flash_attention"

    def test_dtype_values(self):
        """Test DType enum values."""
        assert DType.FP16.value == "float16"
        assert DType.BF16.value == "bfloat16"
        assert DType.FP32.value == "float32"
