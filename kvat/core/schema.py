"""
Pydantic schemas for KVCache Auto-Tuner.

Defines all data models used throughout the tuning process:
- Configuration schemas
- Benchmark results
- Tuning plans
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CacheStrategy(str, Enum):
    """Available KV-Cache strategies."""

    DYNAMIC = "dynamic"
    STATIC = "static"
    SLIDING_WINDOW = "sliding_window"
    OFFLOAD_CPU = "offload_cpu"
    QUANTIZED = "quantized"  # P2 feature


class AttentionBackend(str, Enum):
    """Available attention backends."""

    SDPA_MATH = "sdpa_math"
    SDPA_FLASH = "sdpa_flash"
    SDPA_MEM_EFFICIENT = "sdpa_mem_efficient"
    FLASH_ATTENTION = "flash_attention"
    XFORMERS = "xformers"
    EAGER = "eager"


class DType(str, Enum):
    """Supported data types."""

    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"  # P2 feature


class DeviceType(str, Enum):
    """Supported device types."""

    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class ProfileType(str, Enum):
    """Built-in workload profile types."""

    CHAT_AGENT = "chat-agent"
    RAG = "rag"
    LONGFORM = "longform"
    CUSTOM = "custom"


class WorkloadProfile(BaseModel):
    """Defines a workload profile for tuning."""

    name: str = Field(..., description="Profile name")
    profile_type: ProfileType = Field(..., description="Profile type")
    context_lengths: list[int] = Field(
        default=[2048, 4096],
        description="Context lengths to test"
    )
    output_lengths: list[int] = Field(
        default=[128, 256],
        description="Output token lengths to test"
    )
    system_prompt_tokens: int = Field(
        default=500,
        description="Approximate system prompt length"
    )
    warmup_runs: int = Field(default=2, ge=1, le=10)
    measurement_runs: int = Field(default=5, ge=1, le=20)

    # Scoring weights (must sum to 1.0)
    weight_ttft: float = Field(default=0.5, ge=0.0, le=1.0)
    weight_throughput: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_memory: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("weight_memory")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Ensure weights sum to approximately 1.0."""
        data = info.data
        if "weight_ttft" in data and "weight_throughput" in data:
            total = data["weight_ttft"] + data["weight_throughput"] + v
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0, got {total:.3f}"
                )
        return v


class CandidateConfig(BaseModel):
    """A single tuning candidate configuration."""

    cache_strategy: CacheStrategy
    attention_backend: AttentionBackend
    dtype: DType
    use_torch_compile: bool = False
    max_batch_size: int = 1

    # Cache-specific settings
    cache_max_length: int | None = None
    sliding_window_size: int | None = None
    offload_to_cpu: bool = False

    # Additional settings
    extra_config: dict[str, Any] = Field(default_factory=dict)

    def to_identifier(self) -> str:
        """Generate unique identifier for this config."""
        parts = [
            f"cache={self.cache_strategy.value}",
            f"attn={self.attention_backend.value}",
            f"dtype={self.dtype.value}",
        ]
        if self.use_torch_compile:
            parts.append("compile=true")
        if self.offload_to_cpu:
            parts.append("offload=cpu")
        return "_".join(parts)


class Metrics(BaseModel):
    """Benchmark metrics for a single run."""

    ttft_ms: float = Field(..., description="Time to first token in milliseconds")
    decode_tokens_per_sec: float = Field(..., description="Decode throughput")
    total_time_ms: float = Field(..., description="Total generation time")
    tokens_generated: int = Field(..., description="Number of tokens generated")

    # Memory metrics
    peak_vram_mb: float | None = Field(None, description="Peak VRAM usage in MB")
    peak_ram_mb: float | None = Field(None, description="Peak RAM usage in MB")

    # Stability metrics
    error: str | None = Field(None, description="Error message if failed")
    timeout: bool = Field(default=False, description="Whether run timed out")


class BenchmarkResult(BaseModel):
    """Complete benchmark result for a candidate configuration."""

    candidate: CandidateConfig
    context_length: int
    output_length: int

    # Aggregated metrics
    runs: list[Metrics] = Field(default_factory=list)

    # Computed statistics
    ttft_mean_ms: float = 0.0
    ttft_std_ms: float = 0.0
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    peak_vram_mb: float | None = None
    peak_ram_mb: float | None = None

    # Scoring
    score: float = 0.0
    success_rate: float = 1.0

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def compute_statistics(self) -> None:
        """Compute aggregate statistics from runs."""
        if not self.runs:
            return

        successful_runs = [r for r in self.runs if r.error is None and not r.timeout]

        if not successful_runs:
            self.success_rate = 0.0
            return

        self.success_rate = len(successful_runs) / len(self.runs)

        ttfts = [r.ttft_ms for r in successful_runs]
        throughputs = [r.decode_tokens_per_sec for r in successful_runs]

        self.ttft_mean_ms = sum(ttfts) / len(ttfts)
        self.throughput_mean = sum(throughputs) / len(throughputs)

        if len(ttfts) > 1:
            self.ttft_std_ms = (
                sum((x - self.ttft_mean_ms) ** 2 for x in ttfts)
                / (len(ttfts) - 1)
            ) ** 0.5
            self.throughput_std = (
                sum((x - self.throughput_mean) ** 2 for x in throughputs)
                / (len(throughputs) - 1)
            ) ** 0.5

        vrams = [r.peak_vram_mb for r in successful_runs if r.peak_vram_mb is not None]
        if vrams:
            self.peak_vram_mb = max(vrams)

        rams = [r.peak_ram_mb for r in successful_runs if r.peak_ram_mb is not None]
        if rams:
            self.peak_ram_mb = max(rams)


class FallbackRule(BaseModel):
    """Defines a fallback rule for adaptive configuration."""

    condition: str = Field(..., description="Condition description")
    threshold: float = Field(..., description="Threshold value")
    action: str = Field(..., description="Action to take")
    target_config: CandidateConfig


class TuneResult(BaseModel):
    """Complete tuning result with best plan and recommendations."""

    # Winner configuration
    best_config: CandidateConfig
    best_score: float

    # All benchmark results
    all_results: list[BenchmarkResult] = Field(default_factory=list)

    # Fallback recommendations
    fallback_rules: list[FallbackRule] = Field(default_factory=list)

    # Metadata
    model_id: str
    device: DeviceType
    profile: WorkloadProfile
    tuning_duration_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # System info
    system_info: dict[str, Any] = Field(default_factory=dict)

    # Confidence metrics
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation (0-1)"
    )


class TuneConfig(BaseModel):
    """Main configuration for a tuning run."""

    model_id: str = Field(..., description="HuggingFace model ID or path")
    device: DeviceType = Field(default=DeviceType.CUDA)
    profile: WorkloadProfile

    # Resource limits
    max_vram_mb: float | None = Field(
        None,
        description="Maximum VRAM to use (soft limit)"
    )
    max_ram_mb: float | None = Field(
        None,
        description="Maximum RAM to use (soft limit)"
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="Timeout per candidate"
    )

    # Search configuration
    enable_torch_compile: bool = Field(
        default=False,
        description="Include torch.compile candidates"
    )
    enable_quantized_cache: bool = Field(
        default=False,
        description="Include quantized cache candidates (experimental)"
    )
    early_stopping: bool = Field(
        default=True,
        description="Enable early stopping for dominated candidates"
    )

    # Output configuration
    output_dir: str = Field(default="./kvat_results")
    generate_html_report: bool = Field(default=True)

    # Safety & Privacy
    log_prompts: bool = Field(
        default=False,
        description="Whether to log actual prompt content"
    )
    air_gapped_mode: bool = Field(
        default=True,
        description="No external network calls"
    )
