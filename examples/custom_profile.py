"""
Custom profile example for KVCache Auto-Tuner.

This example shows how to create and use custom workload profiles
for specific use cases not covered by built-in profiles.
"""

from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import create_custom_profile, save_profile_to_json
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch


def main():
    """Run custom profile example."""
    # Create a custom profile for document summarization
    # - Medium context (5-10k tokens from documents)
    # - Medium output (300-500 token summaries)
    # - Balance between TTFT and throughput

    summarization_profile = create_custom_profile(
        name="document-summarization",
        context_lengths=[4096, 6144, 8192],
        output_lengths=[256, 384, 512],
        system_prompt_tokens=200,
        warmup_runs=2,
        measurement_runs=5,
        # Custom weights: balanced approach
        weight_ttft=0.40,
        weight_throughput=0.40,
        weight_memory=0.20,
    )

    # Optionally save for reuse
    save_profile_to_json(summarization_profile, "./profiles/summarization.json")
    print("Profile saved to ./profiles/summarization.json")

    # Create tuning config with custom profile
    config = TuneConfig(
        model_id="meta-llama/Llama-3.2-1B",
        device=DeviceType.CUDA,
        profile=summarization_profile,
        output_dir="./results_summarization",
    )

    # Run tuning
    adapter = TransformersAdapter()
    search = TuningSearch(config=config, adapter=adapter)

    print("\nTuning for document summarization workload...")
    result = search.run()

    print(f"\nBest config for summarization:")
    print(f"  Cache: {result.best_config.cache_strategy.value}")
    print(f"  Attention: {result.best_config.attention_backend.value}")
    print(f"  Score: {result.best_score:.2f}")


if __name__ == "__main__":
    main()
