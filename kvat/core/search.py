"""
Search and tuning orchestration for KVCache Auto-Tuner.

Implements:
- Grid search with early stopping
- Dominance pruning
- Candidate generation
- Benchmark execution
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable

from kvat.core.metrics import (
    MetricsCollector,
    calculate_score,
    is_dominated,
)
from kvat.core.schema import (
    AttentionBackend,
    BenchmarkResult,
    CacheStrategy,
    CandidateConfig,
    DeviceType,
    DType,
    Metrics,
    TuneConfig,
    TuneResult,
)
from kvat.engines.base import EngineAdapter
from kvat.probes.cpu import create_ram_probe
from kvat.probes.gpu import (
    create_vram_probe,
    empty_cuda_cache,
    is_cuda_available,
    reset_cuda_peak_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchProgress:
    """Progress information for search callbacks."""

    total_candidates: int
    completed_candidates: int
    current_candidate: CandidateConfig | None
    current_context_length: int
    current_output_length: int
    best_score: float
    elapsed_seconds: float


ProgressCallback = Callable[[SearchProgress], None]


@dataclass
class TuningSearch:
    """
    Orchestrates the KV-cache tuning search process.

    Handles:
    - Candidate generation
    - Benchmark execution
    - Early stopping and pruning
    - Result collection
    """

    config: TuneConfig
    adapter: EngineAdapter
    progress_callback: ProgressCallback | None = None

    # Internal state
    _candidates: list[CandidateConfig] = field(default_factory=list)
    _results: list[BenchmarkResult] = field(default_factory=list)
    _pruned: set[str] = field(default_factory=set)
    _start_time: float = 0.0

    def generate_candidates(self) -> list[CandidateConfig]:
        """
        Generate candidate configurations to test.

        Returns:
            List of candidate configurations
        """
        candidates = []

        # Get supported options from adapter
        cache_strategies = self.adapter.get_supported_cache_strategies()
        attention_backends = self.adapter.get_supported_attention_backends()
        dtypes = self.adapter.get_supported_dtypes()

        # Filter to relevant strategies
        cache_strategies = self._filter_cache_strategies(cache_strategies)
        attention_backends = self._filter_attention_backends(attention_backends)
        dtypes = self._filter_dtypes(dtypes)

        logger.info(
            f"Generating candidates: "
            f"{len(cache_strategies)} cache strategies, "
            f"{len(attention_backends)} attention backends, "
            f"{len(dtypes)} dtypes"
        )

        # Generate combinations
        for cache, attn, dtype in product(cache_strategies, attention_backends, dtypes):
            config = CandidateConfig(
                cache_strategy=cache,
                attention_backend=attn,
                dtype=dtype,
                use_torch_compile=False,
            )

            # Validate with adapter
            valid, error = self.adapter.validate_config(config)
            if valid:
                candidates.append(config)
            else:
                logger.debug(f"Skipping invalid config: {error}")

        # Optionally add torch.compile variants
        if self.config.enable_torch_compile:
            compile_candidates = []
            for c in candidates[:3]:  # Limit compile candidates
                cc = CandidateConfig(
                    cache_strategy=c.cache_strategy,
                    attention_backend=c.attention_backend,
                    dtype=c.dtype,
                    use_torch_compile=True,
                )
                compile_candidates.append(cc)
            candidates.extend(compile_candidates)

        self._candidates = candidates
        logger.info(f"Generated {len(candidates)} candidate configurations")

        return candidates

    def _filter_cache_strategies(
        self,
        strategies: list[CacheStrategy],
    ) -> list[CacheStrategy]:
        """Filter cache strategies based on config."""
        # Always include Dynamic and Static
        filtered = [
            s for s in strategies
            if s in [CacheStrategy.DYNAMIC, CacheStrategy.STATIC]
        ]

        # Add sliding window for RAG profile with large contexts
        if self.config.profile.profile_type.value == "rag":
            if CacheStrategy.SLIDING_WINDOW in strategies:
                filtered.append(CacheStrategy.SLIDING_WINDOW)

        return filtered or strategies[:2]

    def _filter_attention_backends(
        self,
        backends: list[AttentionBackend],
    ) -> list[AttentionBackend]:
        """Filter attention backends based on availability."""
        # Prioritize faster backends
        priority = [
            AttentionBackend.SDPA_FLASH,
            AttentionBackend.FLASH_ATTENTION,
            AttentionBackend.SDPA_MEM_EFFICIENT,
            AttentionBackend.SDPA_MATH,
            AttentionBackend.EAGER,
        ]

        filtered = [b for b in priority if b in backends]
        return filtered[:4]  # Limit to top 4

    def _filter_dtypes(self, dtypes: list[DType]) -> list[DType]:
        """Filter dtypes based on device and config."""
        # Prefer fp16/bf16 on GPU
        if self.config.device == DeviceType.CUDA:
            priority = [DType.BF16, DType.FP16]
            return [d for d in priority if d in dtypes] or dtypes[:1]
        else:
            return [DType.FP32]

    def run(self) -> TuneResult:
        """
        Execute the tuning search.

        Returns:
            TuneResult with best configuration and all results
        """
        self._start_time = time.time()
        self._results = []
        self._pruned = set()

        # Generate candidates if not already done
        if not self._candidates:
            self.generate_candidates()

        if not self._candidates:
            raise ValueError("No valid candidates to test")

        profile = self.config.profile

        # Get model's max sequence length from config (without loading full model)
        model_config = {}
        if hasattr(self.adapter, 'get_model_config'):
            model_config = self.adapter.get_model_config(self.config.model_id)

        max_seq_len = model_config.get("max_position_embeddings")
        context_lengths = profile.context_lengths
        output_lengths = profile.output_lengths

        logger.info(f"Model max_position_embeddings: {max_seq_len}")

        if max_seq_len:
            # Filter context lengths to fit within model's max
            valid_ctx = [c for c in context_lengths if c <= max_seq_len]
            if len(valid_ctx) < len(context_lengths):
                filtered_out = [c for c in context_lengths if c > max_seq_len]
                logger.warning(
                    f"Model max_position_embeddings={max_seq_len}. "
                    f"Filtering out context lengths: {filtered_out}"
                )
                context_lengths = valid_ctx if valid_ctx else [max_seq_len - 128]

            # Also ensure context + output doesn't exceed max
            valid_out = [o for o in output_lengths if min(context_lengths) + o <= max_seq_len]
            if len(valid_out) < len(output_lengths):
                filtered_out = [o for o in output_lengths if min(context_lengths) + o > max_seq_len]
                logger.warning(
                    f"Filtering out output lengths (would exceed max): {filtered_out}"
                )
                output_lengths = valid_out if valid_out else [32]

        total_tests = (
            len(self._candidates)
            * len(context_lengths)
            * len(output_lengths)
        )

        logger.info(f"Starting tuning search with {total_tests} total tests")

        # Setup metrics collector
        collector = MetricsCollector()
        if is_cuda_available() and self.config.device == DeviceType.CUDA:
            collector.set_vram_probe(create_vram_probe())
        collector.set_ram_probe(create_ram_probe())

        completed = 0
        best_score = 0.0

        # Test each candidate
        for candidate in self._candidates:
            candidate_id = candidate.to_identifier()

            # Skip if pruned
            if candidate_id in self._pruned:
                logger.debug(f"Skipping pruned candidate: {candidate_id}")
                continue

            # Test across context/output lengths
            for ctx_len in context_lengths:
                for out_len in output_lengths:
                    # Skip if pruned mid-flight
                    if candidate_id in self._pruned:
                        break

                    completed += 1
                    self._report_progress(
                        completed,
                        total_tests,
                        candidate,
                        ctx_len,
                        out_len,
                        best_score,
                    )

                    result = self._benchmark_candidate(
                        candidate,
                        ctx_len,
                        out_len,
                        collector,
                    )

                    if result is not None:
                        self._results.append(result)

                        # Calculate score
                        result.compute_statistics()
                        result.score = calculate_score(result, profile)

                        if result.score > best_score:
                            best_score = result.score

                        # Check for early pruning
                        if self.config.early_stopping:
                            self._check_pruning(result)

        # Find best result
        if not self._results:
            raise ValueError("No successful benchmark results")

        best_result = max(self._results, key=lambda r: r.score)

        elapsed = time.time() - self._start_time

        return TuneResult(
            best_config=best_result.candidate,
            best_score=best_result.score,
            all_results=self._results,
            model_id=self.config.model_id,
            device=self.config.device,
            profile=profile,
            tuning_duration_seconds=elapsed,
            confidence=self._calculate_confidence(best_result),
            system_info=self._get_system_info(),
        )

    def _benchmark_candidate(
        self,
        candidate: CandidateConfig,
        context_length: int,
        output_length: int,
        collector: MetricsCollector,
    ) -> BenchmarkResult | None:
        """
        Benchmark a single candidate configuration.

        Uses separate prefill and decode measurements for accurate TTFT.

        Returns:
            BenchmarkResult or None if failed
        """
        profile = self.config.profile

        result = BenchmarkResult(
            candidate=candidate,
            context_length=context_length,
            output_length=output_length,
        )

        # Generate test prompt
        prompt = self._generate_prompt(context_length)

        try:
            # Load model with candidate config
            self.adapter.load_model(
                self.config.model_id,
                self.config.device,
                candidate.dtype,
                candidate.attention_backend,
            )

            # Prepare cache
            candidate.cache_max_length = context_length + output_length
            self.adapter.prepare_cache(candidate)

            # Reset memory stats
            if is_cuda_available():
                reset_cuda_peak_memory()

            # Warmup runs
            for _ in range(profile.warmup_runs):
                try:
                    self.adapter.run_decode(prompt, max_new_tokens=min(32, output_length))
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")

            # Measurement runs
            for run_idx in range(profile.measurement_runs):
                # Reset for this run
                if is_cuda_available():
                    reset_cuda_peak_memory()
                    self.adapter.reset_memory_stats()

                try:
                    # Measure TTFT using prefill
                    _, ttft_ms = self.adapter.run_prefill(prompt, max_new_tokens=1)

                    # Measure decode throughput with full generation
                    start_time = time.perf_counter()
                    output = self.adapter.run_decode(
                        prompt,
                        max_new_tokens=output_length,
                        stream=False,
                    )
                    total_time_ms = (time.perf_counter() - start_time) * 1000

                    # Calculate throughput (tokens/second)
                    tokens_generated = output.tokens_generated
                    if tokens_generated > 0 and total_time_ms > 0:
                        # Decode throughput excludes first token time
                        decode_time_ms = max(1.0, total_time_ms - ttft_ms)
                        throughput = (tokens_generated / decode_time_ms) * 1000
                    else:
                        throughput = 0.0

                    # Get resource usage
                    resources = self.adapter.measure_resources()

                    # Create metrics directly with measured values
                    metrics = Metrics(
                        ttft_ms=ttft_ms,
                        decode_tokens_per_sec=throughput,
                        total_time_ms=total_time_ms,
                        tokens_generated=tokens_generated,
                        peak_vram_mb=resources.vram_mb,
                        peak_ram_mb=resources.ram_mb,
                    )

                    result.runs.append(metrics)

                except Exception as e:
                    logger.warning(f"Run {run_idx} failed: {e}")
                    result.runs.append(Metrics(
                        ttft_ms=0,
                        decode_tokens_per_sec=0,
                        total_time_ms=0,
                        tokens_generated=0,
                        error=str(e),
                    ))

        except Exception as e:
            logger.error(f"Benchmark failed for {candidate.to_identifier()}: {e}")
            return None

        finally:
            # Cleanup between candidates
            self.adapter.cleanup()
            gc.collect()
            if is_cuda_available():
                empty_cuda_cache()

        return result

    def _generate_prompt(self, target_length: int) -> str:
        """Generate a prompt of approximately target length in tokens."""
        # Base system prompt
        base_prompt = (
            "You are a helpful AI assistant. Please provide detailed, "
            "accurate, and well-structured responses to the user's questions. "
            "Consider multiple perspectives and explain your reasoning clearly."
        )

        # Estimate tokens (rough approximation: 4 chars per token)
        chars_needed = target_length * 4

        # Pad with filler content
        filler = (
            " The following context provides additional background information "
            "that may be relevant to the user's query. Please consider this "
            "context when formulating your response."
        )

        prompt = base_prompt
        while len(prompt) < chars_needed:
            prompt += filler

        # Truncate to approximate length
        prompt = prompt[:chars_needed]

        # Add actual question
        prompt += "\n\nQuestion: Please summarize the key points discussed above."

        return prompt

    def _check_pruning(self, result: BenchmarkResult) -> None:
        """Check if other candidates should be pruned based on this result."""
        if result.success_rate == 0:
            return

        for other in self._results:
            if other.candidate.to_identifier() == result.candidate.to_identifier():
                continue

            # Check if other is dominated by this result
            if is_dominated(other, result):
                self._pruned.add(other.candidate.to_identifier())
                logger.debug(
                    f"Pruned {other.candidate.to_identifier()} "
                    f"(dominated by {result.candidate.to_identifier()})"
                )

    def _report_progress(
        self,
        completed: int,
        total: int,
        candidate: CandidateConfig,
        ctx_len: int,
        out_len: int,
        best_score: float,
    ) -> None:
        """Report progress via callback."""
        if self.progress_callback is None:
            return

        progress = SearchProgress(
            total_candidates=total,
            completed_candidates=completed,
            current_candidate=candidate,
            current_context_length=ctx_len,
            current_output_length=out_len,
            best_score=best_score,
            elapsed_seconds=time.time() - self._start_time,
        )

        try:
            self.progress_callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")

    def _calculate_confidence(self, best_result: BenchmarkResult) -> float:
        """Calculate confidence in the recommendation."""
        if not self._results:
            return 0.0

        # Factors affecting confidence:
        # 1. Success rate of best result
        # 2. Gap to second best
        # 3. Consistency (low variance)

        confidence = best_result.success_rate

        # Check gap to second best
        sorted_results = sorted(self._results, key=lambda r: r.score, reverse=True)
        if len(sorted_results) > 1:
            gap = best_result.score - sorted_results[1].score
            if gap > 10:  # Clear winner
                confidence *= 1.1
            elif gap < 2:  # Too close
                confidence *= 0.8

        # Check variance
        if best_result.ttft_mean_ms > 0 and best_result.ttft_std_ms > 0:
            cv = best_result.ttft_std_ms / best_result.ttft_mean_ms
            if cv > 0.3:
                confidence *= 0.9

        return min(1.0, max(0.0, confidence))

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        import platform

        info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
        }

        # Add GPU info
        if is_cuda_available():
            from kvat.probes.gpu import get_gpu_info
            gpu_info = get_gpu_info()
            if gpu_info:
                info["gpu"] = {
                    "name": gpu_info.name,
                    "memory_mb": gpu_info.total_memory_mb,
                    "compute_capability": gpu_info.compute_capability,
                }

        # Add RAM info
        from kvat.probes.cpu import get_system_ram_info
        ram_info = get_system_ram_info()
        if ram_info:
            info["ram_total_mb"] = ram_info.total_mb

        return info
