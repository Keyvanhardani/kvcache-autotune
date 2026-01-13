"""
Basic usage example for KVCache Auto-Tuner.

This example demonstrates how to use the library programmatically
to find the optimal KV-cache configuration for your model.
"""

from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import get_profile
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch
from kvat.core.planner import PlanBuilder
from kvat.core.report import ReportGenerator


def main():
    """Run basic tuning example."""
    # 1. Configure the tuning run
    profile = get_profile("chat-agent")  # or "rag", "longform"

    config = TuneConfig(
        model_id="meta-llama/Llama-3.2-1B",  # Replace with your model
        device=DeviceType.CUDA,
        profile=profile,
        output_dir="./results",
        enable_torch_compile=False,  # Set True to test compiled variants
    )

    # 2. Create the engine adapter
    adapter = TransformersAdapter()

    # 3. Run the tuning search
    print("Starting KV-cache optimization...")

    def on_progress(p):
        pct = (p.completed_candidates / p.total_candidates) * 100
        print(f"Progress: {pct:.0f}% - Testing {p.current_candidate.cache_strategy.value}")

    search = TuningSearch(
        config=config,
        adapter=adapter,
        progress_callback=on_progress,
    )

    result = search.run()

    # 4. Display results
    print(f"\n{'=' * 50}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"\nBest Configuration:")
    print(f"  Cache Strategy: {result.best_config.cache_strategy.value}")
    print(f"  Attention Backend: {result.best_config.attention_backend.value}")
    print(f"  Data Type: {result.best_config.dtype.value}")
    print(f"  Score: {result.best_score:.2f}")
    print(f"  Confidence: {result.confidence * 100:.0f}%")

    # 5. Generate and save outputs
    planner = PlanBuilder(result)
    plan_files = planner.save_plan(config.output_dir)
    print(f"\nPlan saved to: {plan_files['plan']}")
    print(f"Code snippet saved to: {plan_files['snippet']}")

    reporter = ReportGenerator(result)
    report_files = reporter.save(config.output_dir)
    print(f"Report saved to: {report_files['markdown']}")

    # 6. Print the code snippet
    print(f"\n{'=' * 50}")
    print("OPTIMIZED CONFIGURATION SNIPPET")
    print(f"{'=' * 50}")
    print(planner.generate_code_snippet())


if __name__ == "__main__":
    main()
