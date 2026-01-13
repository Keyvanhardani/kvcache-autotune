"""
Transformers Engine Adapter for KVCache Auto-Tuner.

Implements the EngineAdapter interface for HuggingFace Transformers 4.57+,
supporting various cache strategies and attention backends.

Updated for latest Transformers API with cache_implementation parameter.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator
from typing import Any

from kvat.core.schema import (
    AttentionBackend,
    CacheStrategy,
    CandidateConfig,
    DeviceType,
    DType,
)
from kvat.engines.base import (
    CacheConfigError,
    EngineAdapter,
    GenerationError,
    GenerationOutput,
    ModelLoadError,
    ResourceUsage,
)
from kvat.probes.cpu import get_process_ram_mb
from kvat.probes.gpu import (
    get_cuda_max_memory_mb,
    is_cuda_available,
)

logger = logging.getLogger(__name__)


# Lazy imports for transformers
_transformers_available = None
_flash_attn_available = None
_xformers_available = None


def _check_transformers() -> bool:
    """Check if transformers is available."""
    global _transformers_available
    if _transformers_available is None:
        try:
            import transformers
            _transformers_available = True
        except ImportError:
            _transformers_available = False
    return _transformers_available


def _check_flash_attn() -> bool:
    """Check if flash attention is available."""
    global _flash_attn_available
    if _flash_attn_available is None:
        try:
            import flash_attn
            _flash_attn_available = True
        except ImportError:
            _flash_attn_available = False
    return _flash_attn_available


def _check_xformers() -> bool:
    """Check if xformers is available."""
    global _xformers_available
    if _xformers_available is None:
        try:
            import xformers
            _xformers_available = True
        except ImportError:
            _xformers_available = False
    return _xformers_available


class TransformersAdapter(EngineAdapter):
    """
    HuggingFace Transformers adapter for KV-cache tuning.

    Uses latest Transformers 4.57+ API with:
    - cache_implementation parameter for cache selection
    - Proper inference_mode handling
    - Modern generation config
    """

    def __init__(self) -> None:
        if not _check_transformers():
            raise ImportError(
                "transformers is required for TransformersAdapter. "
                "Install with: pip install transformers>=4.45.0"
            )

        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._model_id = None
        self._cache_implementation = "dynamic"  # Default
        self._attention_backend = None
        self._generation_config = None

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_supported_cache_strategies(self) -> list[CacheStrategy]:
        """Get supported cache strategies."""
        # Modern Transformers supports these via cache_implementation
        strategies = [
            CacheStrategy.DYNAMIC,
            CacheStrategy.STATIC,
        ]

        # Check for sliding window / sink cache support
        try:
            from transformers import SinkCache
            strategies.append(CacheStrategy.SLIDING_WINDOW)
        except ImportError:
            pass

        return strategies

    def get_supported_attention_backends(self) -> list[AttentionBackend]:
        """Get supported attention backends."""
        backends = [
            AttentionBackend.EAGER,
            AttentionBackend.SDPA_MATH,
        ]

        if is_cuda_available():
            backends.extend([
                AttentionBackend.SDPA_FLASH,
                AttentionBackend.SDPA_MEM_EFFICIENT,
            ])

            if _check_flash_attn():
                backends.append(AttentionBackend.FLASH_ATTENTION)

            if _check_xformers():
                backends.append(AttentionBackend.XFORMERS)

        return backends

    def get_supported_dtypes(self) -> list[DType]:
        """Get supported data types."""
        dtypes = [DType.FP32, DType.FP16]

        if is_cuda_available():
            import torch
            if torch.cuda.is_bf16_supported():
                dtypes.append(DType.BF16)

        return dtypes

    def load_model(
        self,
        model_id: str,
        device: DeviceType,
        dtype: DType,
        attention_backend: AttentionBackend,
        **kwargs: Any,
    ) -> None:
        """Load model with specified configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        # Cleanup previous model
        self.cleanup()

        logger.info(f"Loading model: {model_id}")

        # Map dtype - use 'dtype' parameter (new API)
        torch_dtype = self._get_torch_dtype(dtype)

        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
            "low_cpu_mem_usage": kwargs.get("low_cpu_mem_usage", True),
        }

        # Device mapping
        if device == DeviceType.CUDA:
            model_kwargs["device_map"] = "auto"
        elif device == DeviceType.MPS:
            model_kwargs["device_map"] = "mps"
        # CPU: no device_map, will move manually

        # Set attention implementation
        attn_impl = self._get_attention_implementation(attention_backend)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

            # Ensure pad token exists
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            )

            # Move to device if CPU (no device_map)
            if device == DeviceType.CPU:
                self._model = self._model.to("cpu")

            # Set eval mode
            self._model.eval()

            # Create generation config
            self._generation_config = GenerationConfig(
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            self._device = device
            self._dtype = dtype
            self._model_id = model_id
            self._attention_backend = attention_backend

            logger.info(f"Model loaded successfully on {device.value}")

        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def prepare_cache(self, config: CandidateConfig) -> None:
        """Prepare KV-cache strategy for generation."""
        if not self.is_loaded:
            raise CacheConfigError("No model loaded")

        # Map cache strategy to cache_implementation string
        cache_map = {
            CacheStrategy.DYNAMIC: "dynamic",
            CacheStrategy.STATIC: "static",
            CacheStrategy.SLIDING_WINDOW: "sliding_window",
            CacheStrategy.OFFLOAD_CPU: "offloaded",
            CacheStrategy.QUANTIZED: "quantized",
        }

        self._cache_implementation = cache_map.get(
            config.cache_strategy,
            "dynamic"
        )

        # Check if cache_implementation is supported (requires triton on some platforms)
        # Fall back to simple use_cache=True if not available
        import platform
        self._use_cache_implementation = platform.system() == "Linux"

        # Update generation config with cache settings
        if self._generation_config is not None and self._use_cache_implementation:
            try:
                self._generation_config.cache_implementation = self._cache_implementation

                # Static cache needs max length
                if config.cache_strategy == CacheStrategy.STATIC:
                    if config.cache_max_length:
                        self._generation_config.cache_config = {
                            "max_cache_len": config.cache_max_length
                        }
            except Exception as e:
                logger.warning(f"cache_implementation not supported: {e}")
                self._use_cache_implementation = False

        logger.debug(f"Cache prepared: {self._cache_implementation} (native={self._use_cache_implementation})")

    def run_prefill(
        self,
        prompt: str,
        max_new_tokens: int = 0,
    ) -> tuple[GenerationOutput, float]:
        """Run prefill phase."""
        if not self.is_loaded:
            raise GenerationError("No model loaded")

        import torch

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()

        with torch.inference_mode():
            if max_new_tokens == 0:
                # Prefill only - run forward pass
                _ = self._model(**inputs, use_cache=False)
                generated_tokens = 0
                text = ""
            else:
                # Generate tokens - use simple API for Windows compatibility
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "pad_token_id": self._tokenizer.pad_token_id,
                }

                if hasattr(self, '_use_cache_implementation') and self._use_cache_implementation:
                    gen_kwargs["generation_config"] = self._generation_config

                outputs = self._model.generate(**gen_kwargs)

                generated_tokens = outputs.shape[1] - prompt_tokens
                text = self._tokenizer.decode(
                    outputs[0, prompt_tokens:],
                    skip_special_tokens=True,
                )

        prefill_time = (time.perf_counter() - start_time) * 1000

        return GenerationOutput(
            text=text,
            tokens_generated=generated_tokens,
            prompt_tokens=prompt_tokens,
            finish_reason="length" if generated_tokens > 0 else "prefill",
        ), prefill_time

    def run_decode(
        self,
        prompt: str,
        max_new_tokens: int,
        *,
        stream: bool = False,
    ) -> Iterator[tuple[str, int]] | GenerationOutput:
        """Run full generation."""
        if not self.is_loaded:
            raise GenerationError("No model loaded")


        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        if stream:
            return self._stream_generate(inputs, prompt_tokens, max_new_tokens)
        else:
            return self._batch_generate(inputs, prompt_tokens, max_new_tokens)

    def _batch_generate(
        self,
        inputs: dict,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> GenerationOutput:
        """Non-streaming generation using latest API."""
        import torch

        # Build generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }

        # Only use generation_config if cache_implementation is supported
        if hasattr(self, '_use_cache_implementation') and self._use_cache_implementation:
            gen_kwargs["generation_config"] = self._generation_config

        with torch.inference_mode():
            outputs = self._model.generate(**gen_kwargs)

        generated_tokens = outputs.shape[1] - prompt_tokens
        text = self._tokenizer.decode(
            outputs[0, prompt_tokens:],
            skip_special_tokens=True,
        )

        # Determine finish reason
        finish_reason = "length" if generated_tokens >= max_new_tokens else "stop"

        return GenerationOutput(
            text=text,
            tokens_generated=generated_tokens,
            prompt_tokens=prompt_tokens,
            finish_reason=finish_reason,
        )

    def _stream_generate(
        self,
        inputs: dict,
        prompt_tokens: int,
        max_new_tokens: int,
    ) -> Iterator[tuple[str, int]]:
        """Streaming generation with token-by-token output."""
        from threading import Thread

        import torch
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Clone generation config and add streamer
        gen_config = self._generation_config.to_dict() if self._generation_config else {}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            **gen_config,
        }

        # Run generation in background thread
        def generate_thread():
            with torch.inference_mode():
                self._model.generate(**generation_kwargs)

        thread = Thread(target=generate_thread)
        thread.start()

        # Yield tokens as they come
        total_tokens = 0
        for text in streamer:
            total_tokens += 1
            yield text, total_tokens

        thread.join()

    def measure_resources(self) -> ResourceUsage:
        """Measure current resource usage."""
        vram_mb = None
        if is_cuda_available() and self._device == DeviceType.CUDA:
            vram_mb = get_cuda_max_memory_mb()

        ram_mb = get_process_ram_mb()

        return ResourceUsage(
            vram_mb=vram_mb,
            ram_mb=ram_mb,
        )

    def cleanup(self) -> None:
        """Clean up model and free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._model_id = None
        self._device = None
        self._dtype = None
        self._attention_backend = None
        self._generation_config = None
        self._cache_implementation = "dynamic"

        gc.collect()

        if is_cuda_available():
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def get_model_config(self, model_id: str) -> dict[str, Any]:
        """
        Get model config info WITHOUT loading the full model.
        Uses AutoConfig to fetch only the configuration.
        """
        from transformers import AutoConfig

        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return {
                "max_position_embeddings": getattr(
                    config, "max_position_embeddings", None
                ),
                "vocab_size": getattr(config, "vocab_size", None),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            }
        except Exception as e:
            logger.warning(f"Could not fetch model config: {e}")
            return {}

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded model."""
        if not self.is_loaded:
            return {}

        config = self._model.config

        info = {
            "model_id": self._model_id,
            "device": self._device.value if self._device else None,
            "dtype": self._dtype.value if self._dtype else None,
            "attention_backend": (
                self._attention_backend.value if self._attention_backend else None
            ),
            "cache_implementation": self._cache_implementation,
            "vocab_size": getattr(config, "vocab_size", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "max_position_embeddings": getattr(
                config, "max_position_embeddings", None
            ),
        }

        # Model memory footprint
        if hasattr(self._model, "get_memory_footprint"):
            info["memory_footprint_mb"] = (
                self._model.get_memory_footprint() / (1024 * 1024)
            )

        return info

    def _get_torch_dtype(self, dtype: DType):
        """Convert DType enum to torch dtype."""
        import torch

        dtype_map = {
            DType.FP32: torch.float32,
            DType.FP16: torch.float16,
            DType.BF16: torch.bfloat16,
        }
        return dtype_map.get(dtype, torch.float16)

    def _get_attention_implementation(
        self,
        backend: AttentionBackend,
    ) -> str | None:
        """Get attention implementation string for model config."""
        impl_map = {
            AttentionBackend.EAGER: "eager",
            AttentionBackend.SDPA_MATH: "sdpa",
            AttentionBackend.SDPA_FLASH: "sdpa",
            AttentionBackend.SDPA_MEM_EFFICIENT: "sdpa",
            AttentionBackend.FLASH_ATTENTION: "flash_attention_2",
            AttentionBackend.XFORMERS: "sdpa",
        }
        return impl_map.get(backend)

    def reset_memory_stats(self) -> None:
        """Reset GPU memory statistics for accurate measurement."""
        if is_cuda_available() and self._device == DeviceType.CUDA:
            import torch
            torch.cuda.reset_peak_memory_stats()
