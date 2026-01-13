"""
Transformers Engine Adapter for KVCache Auto-Tuner.

Implements the EngineAdapter interface for HuggingFace Transformers,
supporting various cache strategies and attention backends.
"""

from __future__ import annotations

import gc
import time
from typing import Optional, Any, Iterator
import logging

from kvat.engines.base import (
    EngineAdapter,
    GenerationOutput,
    ResourceUsage,
    ModelLoadError,
    GenerationError,
    CacheConfigError,
)
from kvat.core.schema import (
    CandidateConfig,
    CacheStrategy,
    AttentionBackend,
    DType,
    DeviceType,
)
from kvat.probes.gpu import (
    get_cuda_max_memory_mb,
    reset_cuda_peak_memory,
    empty_cuda_cache,
    is_cuda_available,
)
from kvat.probes.cpu import get_process_ram_mb

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

    Supports:
    - Multiple cache strategies (Dynamic, Static, Sliding Window)
    - Multiple attention backends (SDPA, Flash, xFormers)
    - Various data types (fp16, bf16, fp32)
    """

    def __init__(self) -> None:
        if not _check_transformers():
            raise ImportError(
                "transformers is required for TransformersAdapter. "
                "Install with: pip install transformers"
            )

        self._model = None
        self._tokenizer = None
        self._device = None
        self._dtype = None
        self._model_id = None
        self._cache = None
        self._attention_backend = None

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_supported_cache_strategies(self) -> list[CacheStrategy]:
        """Get supported cache strategies."""
        strategies = [
            CacheStrategy.DYNAMIC,
            CacheStrategy.STATIC,
        ]

        # Check for sliding window support
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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Cleanup previous model
        self.cleanup()

        logger.info(f"Loading model: {model_id}")

        # Map dtype
        torch_dtype = self._get_torch_dtype(dtype)

        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device.value if device == DeviceType.CUDA else None,
            "trust_remote_code": kwargs.get("trust_remote_code", True),
        }

        # Set attention implementation
        attn_impl = self._get_attention_implementation(attention_backend)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        # Low memory loading option
        if kwargs.get("low_cpu_mem_usage", True):
            model_kwargs["low_cpu_mem_usage"] = True

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

            # Move to device if not using device_map
            if device == DeviceType.CPU:
                self._model = self._model.to("cpu")
            elif device == DeviceType.MPS:
                self._model = self._model.to("mps")

            # Set eval mode
            self._model.eval()

            self._device = device
            self._dtype = dtype
            self._model_id = model_id
            self._attention_backend = attention_backend

            logger.info(f"Model loaded successfully on {device.value}")

        except Exception as e:
            self.cleanup()
            raise ModelLoadError(f"Failed to load model: {e}") from e

    def prepare_cache(self, config: CandidateConfig) -> None:
        """Prepare KV-cache according to configuration."""
        if not self.is_loaded:
            raise CacheConfigError("No model loaded")

        from transformers import DynamicCache, StaticCache

        self._cache = None

        try:
            if config.cache_strategy == CacheStrategy.DYNAMIC:
                self._cache = DynamicCache()

            elif config.cache_strategy == CacheStrategy.STATIC:
                # Static cache requires knowing max length upfront
                max_length = config.cache_max_length or 4096

                # Get model config for cache dimensions
                model_config = self._model.config

                self._cache = StaticCache(
                    config=model_config,
                    batch_size=config.max_batch_size,
                    max_cache_len=max_length,
                    device=self._model.device,
                    dtype=self._get_torch_dtype(config.dtype),
                )

            elif config.cache_strategy == CacheStrategy.SLIDING_WINDOW:
                # Use SinkCache for sliding window behavior
                try:
                    from transformers import SinkCache
                    window_size = config.sliding_window_size or 1024
                    self._cache = SinkCache(
                        window_length=window_size,
                        num_sink_tokens=4,
                    )
                except ImportError:
                    raise CacheConfigError(
                        "SinkCache not available in this transformers version"
                    )

            logger.debug(f"Cache prepared: {config.cache_strategy.value}")

        except Exception as e:
            raise CacheConfigError(f"Failed to prepare cache: {e}") from e

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

        # Reset cache if using static
        if self._cache is not None and hasattr(self._cache, "reset"):
            self._cache.reset()

        start_time = time.perf_counter()

        with torch.inference_mode():
            if max_new_tokens == 0:
                # Prefill only - just run forward pass
                outputs = self._model(
                    **inputs,
                    past_key_values=self._cache,
                    use_cache=True,
                )
                generated_tokens = 0
                text = ""
            else:
                # Generate tokens
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    past_key_values=self._cache,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

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

        # Reset cache
        if self._cache is not None and hasattr(self._cache, "reset"):
            self._cache.reset()

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
        """Non-streaming generation."""
        import torch

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                past_key_values=self._cache,
                use_cache=True,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated_tokens = outputs.shape[1] - prompt_tokens
        text = self._tokenizer.decode(
            outputs[0, prompt_tokens:],
            skip_special_tokens=True,
        )

        # Determine finish reason
        if generated_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

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
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "past_key_values": self._cache,
            "use_cache": True,
            "do_sample": False,
            "pad_token_id": self._tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # Run generation in background thread
        thread = Thread(
            target=self._model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        # Yield tokens as they come
        total_tokens = 0
        for text in streamer:
            total_tokens += 1  # Approximate
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

        if self._cache is not None:
            del self._cache
            self._cache = None

        self._model_id = None
        self._device = None
        self._dtype = None
        self._attention_backend = None

        gc.collect()

        if is_cuda_available():
            empty_cuda_cache()

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
    ) -> Optional[str]:
        """Get attention implementation string for model config."""
        # Map to transformers attention implementations
        impl_map = {
            AttentionBackend.EAGER: "eager",
            AttentionBackend.SDPA_MATH: "sdpa",
            AttentionBackend.SDPA_FLASH: "sdpa",
            AttentionBackend.SDPA_MEM_EFFICIENT: "sdpa",
            AttentionBackend.FLASH_ATTENTION: "flash_attention_2",
            AttentionBackend.XFORMERS: "sdpa",  # xformers via SDPA
        }
        return impl_map.get(backend)

    def reset_memory_stats(self) -> None:
        """Reset GPU memory statistics for accurate measurement."""
        if is_cuda_available() and self._device == DeviceType.CUDA:
            reset_cuda_peak_memory()
