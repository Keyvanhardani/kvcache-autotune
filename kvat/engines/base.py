"""
Base Engine Adapter interface for KVCache Auto-Tuner.

Defines the contract that all inference engine adapters must implement.
This abstraction enables support for multiple backends:
- Transformers (primary)
- vLLM (P2)
- llama.cpp (P2)
- Ollama (P2)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from kvat.core.schema import (
    AttentionBackend,
    CacheStrategy,
    CandidateConfig,
    DeviceType,
    DType,
)


@dataclass
class GenerationOutput:
    """Output from a generation run."""

    text: str
    tokens_generated: int
    prompt_tokens: int
    finish_reason: str  # "length", "stop", "error"


@dataclass
class ResourceUsage:
    """Resource usage snapshot."""

    vram_mb: float | None = None
    ram_mb: float | None = None
    gpu_utilization: float | None = None


class EngineAdapter(ABC):
    """
    Abstract base class for inference engine adapters.

    Each adapter handles:
    - Model loading with specific configurations
    - KV-cache setup and management
    - Generation (prefill + decode)
    - Resource measurement
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name (e.g., 'transformers', 'vllm')."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        pass

    @abstractmethod
    def get_supported_cache_strategies(self) -> list[CacheStrategy]:
        """Get list of supported cache strategies for this engine."""
        pass

    @abstractmethod
    def get_supported_attention_backends(self) -> list[AttentionBackend]:
        """Get list of supported attention backends."""
        pass

    @abstractmethod
    def get_supported_dtypes(self) -> list[DType]:
        """Get list of supported data types."""
        pass

    @abstractmethod
    def load_model(
        self,
        model_id: str,
        device: DeviceType,
        dtype: DType,
        attention_backend: AttentionBackend,
        **kwargs: Any,
    ) -> None:
        """
        Load a model with specific configuration.

        Args:
            model_id: HuggingFace model ID or local path
            device: Target device (cuda, cpu, mps)
            dtype: Data type for model weights
            attention_backend: Attention implementation to use
            **kwargs: Engine-specific options
        """
        pass

    @abstractmethod
    def prepare_cache(self, config: CandidateConfig) -> None:
        """
        Prepare KV-cache according to configuration.

        Args:
            config: Cache configuration
        """
        pass

    @abstractmethod
    def run_prefill(
        self,
        prompt: str,
        max_new_tokens: int = 0,
    ) -> tuple[GenerationOutput, float]:
        """
        Run prefill phase (encode prompt, optionally generate tokens).

        Args:
            prompt: Input prompt text
            max_new_tokens: Number of tokens to generate (0 = prefill only)

        Returns:
            Tuple of (output, prefill_time_ms)
        """
        pass

    @abstractmethod
    def run_decode(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        stream: bool = False,
    ) -> Iterator[tuple[str, int]] | GenerationOutput:
        """
        Run full generation (prefill + decode).

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            stream: If True, yield tokens as they're generated

        Returns:
            If stream=False: GenerationOutput
            If stream=True: Iterator of (token_text, total_tokens_so_far)
        """
        pass

    @abstractmethod
    def measure_resources(self) -> ResourceUsage:
        """
        Measure current resource usage.

        Returns:
            ResourceUsage snapshot
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources (unload model, free memory).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        pass

    def get_max_sequence_length(self) -> int | None:
        """
        Get the maximum sequence length supported by the model.

        Returns:
            Max sequence length or None if unknown
        """
        info = self.get_model_info()
        return info.get("max_position_embeddings")

    def validate_config(self, config: CandidateConfig) -> tuple[bool, str]:
        """
        Validate if a configuration is supported.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if config.cache_strategy not in self.get_supported_cache_strategies():
            return False, f"Cache strategy {config.cache_strategy} not supported"

        if config.attention_backend not in self.get_supported_attention_backends():
            return False, f"Attention backend {config.attention_backend} not supported"

        if config.dtype not in self.get_supported_dtypes():
            return False, f"Data type {config.dtype} not supported"

        return True, ""


class EngineError(Exception):
    """Base exception for engine errors."""
    pass


class ModelLoadError(EngineError):
    """Error loading a model."""
    pass


class GenerationError(EngineError):
    """Error during generation."""
    pass


class CacheConfigError(EngineError):
    """Error configuring KV-cache."""
    pass
