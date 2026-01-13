"""
Plan builder for KVCache Auto-Tuner.

Creates deployment-ready configuration plans from tuning results,
including fallback rules and code snippets.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kvat.core.schema import (
    AttentionBackend,
    BenchmarkResult,
    CacheStrategy,
    CandidateConfig,
    DType,
    FallbackRule,
    TuneResult,
)


class PlanBuilder:
    """
    Builds deployment plans from tuning results.

    Generates:
    - Best configuration as JSON
    - Transformers code snippet
    - Fallback rules for different scenarios
    """

    def __init__(self, result: TuneResult) -> None:
        self.result = result

    def build_plan(self) -> dict[str, Any]:
        """
        Build complete deployment plan.

        Returns:
            Dictionary containing full plan
        """
        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(),
            "model_id": self.result.model_id,
            "device": self.result.device.value,
            "profile": self.result.profile.name,

            "best_config": self._config_to_dict(self.result.best_config),
            "best_score": self.result.best_score,
            "confidence": self.result.confidence,

            "fallback_rules": [
                self._fallback_to_dict(rule)
                for rule in self._generate_fallback_rules()
            ],

            "code_snippet": self.generate_code_snippet(),

            "benchmarks": {
                "summary": self._get_benchmark_summary(),
                "tuning_duration_seconds": self.result.tuning_duration_seconds,
            },

            "system_info": self.result.system_info,
        }

    def _config_to_dict(self, config: CandidateConfig) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "cache_strategy": config.cache_strategy.value,
            "attention_backend": config.attention_backend.value,
            "dtype": config.dtype.value,
            "use_torch_compile": config.use_torch_compile,
            "max_batch_size": config.max_batch_size,
            "cache_max_length": config.cache_max_length,
            "sliding_window_size": config.sliding_window_size,
            "offload_to_cpu": config.offload_to_cpu,
        }

    def _fallback_to_dict(self, rule: FallbackRule) -> dict[str, Any]:
        """Convert fallback rule to dictionary."""
        return {
            "condition": rule.condition,
            "threshold": rule.threshold,
            "action": rule.action,
            "target_config": self._config_to_dict(rule.target_config),
        }

    def _generate_fallback_rules(self) -> list[FallbackRule]:
        """Generate fallback rules from benchmark results."""
        rules = []

        # Find alternative configurations
        sorted_results = sorted(
            self.result.all_results,
            key=lambda r: r.score,
            reverse=True,
        )

        if len(sorted_results) < 2:
            return rules

        best = sorted_results[0]
        second = sorted_results[1]

        # VRAM fallback rule
        if best.peak_vram_mb and second.peak_vram_mb:
            if best.peak_vram_mb > second.peak_vram_mb:
                rules.append(FallbackRule(
                    condition="VRAM usage exceeds limit",
                    threshold=best.peak_vram_mb * 0.9,
                    action="Switch to lower memory configuration",
                    target_config=second.candidate,
                ))

        # Find memory-efficient alternative
        mem_efficient = self._find_memory_efficient_config(sorted_results)
        if mem_efficient and mem_efficient.candidate != best.candidate:
            rules.append(FallbackRule(
                condition="Memory constrained environment",
                threshold=0.0,
                action="Use memory-efficient configuration",
                target_config=mem_efficient.candidate,
            ))

        # Latency-sensitive fallback
        low_ttft = self._find_low_ttft_config(sorted_results)
        if low_ttft and low_ttft.candidate != best.candidate:
            rules.append(FallbackRule(
                condition="Latency-sensitive workload",
                threshold=100.0,  # TTFT threshold in ms
                action="Use low-latency configuration",
                target_config=low_ttft.candidate,
            ))

        return rules

    def _find_memory_efficient_config(
        self,
        results: list[BenchmarkResult],
    ) -> BenchmarkResult | None:
        """Find the most memory-efficient configuration."""
        valid = [r for r in results if r.peak_vram_mb and r.success_rate > 0.5]
        if not valid:
            return None
        return min(valid, key=lambda r: r.peak_vram_mb)

    def _find_low_ttft_config(
        self,
        results: list[BenchmarkResult],
    ) -> BenchmarkResult | None:
        """Find configuration with lowest TTFT."""
        valid = [r for r in results if r.ttft_mean_ms > 0 and r.success_rate > 0.5]
        if not valid:
            return None
        return min(valid, key=lambda r: r.ttft_mean_ms)

    def _get_benchmark_summary(self) -> dict[str, Any]:
        """Get summary of benchmark results."""
        best = None
        for r in self.result.all_results:
            if r.candidate == self.result.best_config:
                best = r
                break

        if not best:
            return {}

        return {
            "ttft_mean_ms": best.ttft_mean_ms,
            "ttft_std_ms": best.ttft_std_ms,
            "throughput_mean_tok_s": best.throughput_mean,
            "throughput_std_tok_s": best.throughput_std,
            "peak_vram_mb": best.peak_vram_mb,
            "peak_ram_mb": best.peak_ram_mb,
            "success_rate": best.success_rate,
        }

    def generate_code_snippet(self) -> str:
        """
        Generate Python code snippet for applying the best configuration.

        Returns:
            Python code as string
        """
        config = self.result.best_config

        # Build imports
        imports = [
            "import torch",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
        ]

        # Add cache import
        if config.cache_strategy == CacheStrategy.DYNAMIC:
            imports.append("from transformers import DynamicCache")
        elif config.cache_strategy == CacheStrategy.STATIC:
            imports.append("from transformers import StaticCache")
        elif config.cache_strategy == CacheStrategy.SLIDING_WINDOW:
            imports.append("from transformers import SinkCache")

        # Build dtype mapping
        dtype_map = {
            DType.FP32: "torch.float32",
            DType.FP16: "torch.float16",
            DType.BF16: "torch.bfloat16",
        }
        torch_dtype = dtype_map.get(config.dtype, "torch.float16")

        # Build attention implementation
        attn_impl_map = {
            AttentionBackend.EAGER: '"eager"',
            AttentionBackend.SDPA_MATH: '"sdpa"',
            AttentionBackend.SDPA_FLASH: '"sdpa"',
            AttentionBackend.SDPA_MEM_EFFICIENT: '"sdpa"',
            AttentionBackend.FLASH_ATTENTION: '"flash_attention_2"',
            AttentionBackend.XFORMERS: '"sdpa"',
        }
        attn_impl = attn_impl_map.get(config.attention_backend, '"sdpa"')

        # Generate code
        code_lines = [
            "# KVCache Auto-Tuner - Optimized Configuration",
            f"# Model: {self.result.model_id}",
            f"# Profile: {self.result.profile.name}",
            f"# Score: {self.result.best_score:.2f}",
            "",
            *imports,
            "",
            "# Load model with optimized settings",
            f'model_id = "{self.result.model_id}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "if tokenizer.pad_token is None:",
            "    tokenizer.pad_token = tokenizer.eos_token",
            "",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            f"    torch_dtype={torch_dtype},",
            f"    attn_implementation={attn_impl},",
            '    device_map="auto",',
            "    low_cpu_mem_usage=True,",
            ")",
            "model.eval()",
            "",
        ]

        # Add cache setup
        if config.cache_strategy == CacheStrategy.DYNAMIC:
            code_lines.extend([
                "# Setup KV-Cache (Dynamic)",
                "cache = DynamicCache()",
            ])
        elif config.cache_strategy == CacheStrategy.STATIC:
            max_len = config.cache_max_length or 4096
            code_lines.extend([
                "# Setup KV-Cache (Static)",
                "cache = StaticCache(",
                "    config=model.config,",
                "    batch_size=1,",
                f"    max_cache_len={max_len},",
                "    device=model.device,",
                f"    dtype={torch_dtype},",
                ")",
            ])
        elif config.cache_strategy == CacheStrategy.SLIDING_WINDOW:
            window = config.sliding_window_size or 1024
            code_lines.extend([
                "# Setup KV-Cache (Sliding Window)",
                "cache = SinkCache(",
                f"    window_length={window},",
                "    num_sink_tokens=4,",
                ")",
            ])

        # Add generation example
        code_lines.extend([
            "",
            "# Example generation",
            "def generate(prompt: str, max_new_tokens: int = 256) -> str:",
            "    inputs = tokenizer(prompt, return_tensors='pt')",
            "    inputs = {k: v.to(model.device) for k, v in inputs.items()}",
            "",
            "    with torch.inference_mode():",
            "        outputs = model.generate(",
            "            **inputs,",
            "            max_new_tokens=max_new_tokens,",
            "            past_key_values=cache,",
            "            use_cache=True,",
            "            do_sample=False,",
            "            pad_token_id=tokenizer.pad_token_id,",
            "        )",
            "",
            "    return tokenizer.decode(",
            "        outputs[0, inputs['input_ids'].shape[1]:],",
            "        skip_special_tokens=True,",
            "    )",
        ])

        # Add torch.compile if enabled
        if config.use_torch_compile:
            code_lines.insert(-15, "# Apply torch.compile optimization")
            code_lines.insert(-15, "model = torch.compile(model, mode='reduce-overhead')")
            code_lines.insert(-15, "")

        return "\n".join(code_lines)

    def save_plan(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Save plan to files.

        Args:
            output_dir: Directory to save files

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save JSON plan
        plan = self.build_plan()
        plan_path = output_dir / "best_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, default=str)
        saved["plan"] = plan_path

        # Save code snippet
        snippet_path = output_dir / "optimized_config.py"
        with open(snippet_path, "w", encoding="utf-8") as f:
            f.write(self.generate_code_snippet())
        saved["snippet"] = snippet_path

        return saved


def load_plan(path: str | Path) -> dict[str, Any]:
    """
    Load a saved plan from JSON.

    Args:
        path: Path to plan JSON file

    Returns:
        Plan dictionary
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_from_plan(plan: dict[str, Any]) -> CandidateConfig:
    """
    Create CandidateConfig from a plan dictionary.

    Args:
        plan: Plan dictionary

    Returns:
        CandidateConfig instance
    """
    config_dict = plan["best_config"]

    return CandidateConfig(
        cache_strategy=CacheStrategy(config_dict["cache_strategy"]),
        attention_backend=AttentionBackend(config_dict["attention_backend"]),
        dtype=DType(config_dict["dtype"]),
        use_torch_compile=config_dict.get("use_torch_compile", False),
        max_batch_size=config_dict.get("max_batch_size", 1),
        cache_max_length=config_dict.get("cache_max_length"),
        sliding_window_size=config_dict.get("sliding_window_size"),
        offload_to_cpu=config_dict.get("offload_to_cpu", False),
    )
