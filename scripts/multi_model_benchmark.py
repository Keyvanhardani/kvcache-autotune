#!/usr/bin/env python3
"""
Multi-Model Benchmark Suite for KVCache Auto-Tuner.

Benchmarks popular LLM models to showcase optimization across different architectures.
Designed for 8GB VRAM (RTX 4060/4070/3070).
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# =============================================================================
# Model Configurations for 8GB VRAM
# =============================================================================

BENCHMARK_MODELS = [
    # Tiny models (< 1GB VRAM)
    {
        "id": "gpt2",
        "name": "GPT-2",
        "family": "OpenAI",
        "params": "124M",
        "vram_estimate": "0.5GB",
        "profile": "ci-micro",
    },

    # Small models (1-2GB VRAM)
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5-0.5B",
        "family": "Qwen",
        "params": "0.5B",
        "vram_estimate": "1.5GB",
        "profile": "ci-micro",
    },
    {
        "id": "microsoft/phi-1_5",
        "name": "Phi-1.5",
        "family": "Microsoft",
        "params": "1.3B",
        "vram_estimate": "2.5GB",
        "profile": "ci-micro",
    },

    # Medium models (2-4GB VRAM)
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen2.5-1.5B",
        "family": "Qwen",
        "params": "1.5B",
        "vram_estimate": "3GB",
        "profile": "ci-micro",
    },
    {
        "id": "google/gemma-2b",
        "name": "Gemma-2B",
        "family": "Google",
        "params": "2B",
        "vram_estimate": "4GB",
        "profile": "ci-micro",
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "name": "Llama-3.2-1B",
        "family": "Meta",
        "params": "1B",
        "vram_estimate": "2.5GB",
        "profile": "ci-micro",
    },

    # Larger models (4-6GB VRAM) - with bf16
    {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "name": "Llama-3.2-3B",
        "family": "Meta",
        "params": "3B",
        "vram_estimate": "6GB",
        "profile": "ci-micro",
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "family": "Qwen",
        "params": "3B",
        "vram_estimate": "6GB",
        "profile": "ci-micro",
    },
    {
        "id": "microsoft/phi-2",
        "name": "Phi-2",
        "family": "Microsoft",
        "params": "2.7B",
        "vram_estimate": "5.5GB",
        "profile": "ci-micro",
    },

    # Vision-Language Models (if VRAM allows)
    {
        "id": "Qwen/Qwen2-VL-2B-Instruct",
        "name": "Qwen2-VL-2B",
        "family": "Qwen",
        "params": "2B",
        "vram_estimate": "4.5GB",
        "profile": "ci-micro",
        "skip_reason": "VLM - requires special handling",
    },
]

# Quick benchmark models (for faster testing)
QUICK_MODELS = [
    "gpt2",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/phi-1_5",
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Full benchmark models (comprehensive)
FULL_MODELS = [m["id"] for m in BENCHMARK_MODELS if "skip_reason" not in m]


@dataclass
class BenchmarkResult:
    model_id: str
    model_name: str
    family: str
    params: str
    ttft_ms: float
    throughput: float
    vram_mb: float
    score: float
    best_config: str
    success: bool
    error: str | None = None
    duration_s: float = 0.0


def run_single_benchmark(model_config: dict, output_dir: Path) -> BenchmarkResult:
    """Run benchmark for a single model."""
    model_id = model_config["id"]
    model_name = model_config["name"]
    profile = model_config.get("profile", "ci-micro")

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name} ({model_id})")
    print(f"{'='*60}")

    # Create output directory for this model
    model_output = output_dir / model_name.replace("/", "_").replace(" ", "_")
    model_output.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        # Run kvat tune
        cmd = [
            sys.executable, "-m", "kvat", "tune",
            model_id,
            "--profile", profile,
            "-o", str(model_output),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per model
            env={**dict(__import__('os').environ), "PYTHONIOENCODING": "utf-8"},
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            print(f"  [FAILED] {result.stderr[:500]}")
            return BenchmarkResult(
                model_id=model_id,
                model_name=model_name,
                family=model_config["family"],
                params=model_config["params"],
                ttft_ms=0,
                throughput=0,
                vram_mb=0,
                score=0,
                best_config="N/A",
                success=False,
                error=result.stderr[:200],
                duration_s=duration,
            )

        # Load results
        plan_file = model_output / "best_plan.json"
        if plan_file.exists():
            with open(plan_file) as f:
                plan = json.load(f)

            summary = plan.get("benchmarks", {}).get("summary", {})
            best_config = plan.get("best_config", {})

            config_str = f"{best_config.get('cache_strategy', 'dynamic')}/{best_config.get('attention_backend', 'sdpa')}"

            result = BenchmarkResult(
                model_id=model_id,
                model_name=model_name,
                family=model_config["family"],
                params=model_config["params"],
                ttft_ms=summary.get("ttft_mean_ms", 0),
                throughput=summary.get("throughput_mean_tok_s", 0),
                vram_mb=summary.get("peak_vram_mb", 0),
                score=plan.get("best_score", 0),
                best_config=config_str,
                success=True,
                duration_s=duration,
            )

            print(f"  [SUCCESS] TTFT: {result.ttft_ms:.1f}ms | Throughput: {result.throughput:.1f} tok/s | VRAM: {result.vram_mb:.0f}MB")
            return result
        else:
            return BenchmarkResult(
                model_id=model_id,
                model_name=model_name,
                family=model_config["family"],
                params=model_config["params"],
                ttft_ms=0,
                throughput=0,
                vram_mb=0,
                score=0,
                best_config="N/A",
                success=False,
                error="No results file generated",
                duration_s=duration,
            )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"  [TIMEOUT] Model took too long (>10 min)")
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            family=model_config["family"],
            params=model_config["params"],
            ttft_ms=0,
            throughput=0,
            vram_mb=0,
            score=0,
            best_config="N/A",
            success=False,
            error="Timeout (>10 min)",
            duration_s=duration,
        )
    except Exception as e:
        duration = time.time() - start_time
        print(f"  [ERROR] {e}")
        return BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            family=model_config["family"],
            params=model_config["params"],
            ttft_ms=0,
            throughput=0,
            vram_mb=0,
            score=0,
            best_config="N/A",
            success=False,
            error=str(e),
            duration_s=duration,
        )


def generate_comparison_report(results: list[BenchmarkResult], output_dir: Path):
    """Generate comparison report and charts."""
    # Filter successful results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    # Generate markdown report
    report_path = output_dir / "BENCHMARK_COMPARISON.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# KVCache Auto-Tuner - Multi-Model Benchmark\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Models Tested:** {len(results)}\n")
        f.write(f"- **Successful:** {len(successful)}\n")
        f.write(f"- **Failed:** {len(failed)}\n\n")

        if successful:
            f.write("## Results\n\n")
            f.write("| Model | Family | Params | TTFT (ms) | Throughput | VRAM (MB) | Best Config |\n")
            f.write("|-------|--------|--------|-----------|------------|-----------|-------------|\n")

            for r in sorted(successful, key=lambda x: x.throughput, reverse=True):
                f.write(f"| **{r.model_name}** | {r.family} | {r.params} | ")
                f.write(f"{r.ttft_ms:.1f} | {r.throughput:.1f} tok/s | ")
                f.write(f"{r.vram_mb:.0f} | {r.best_config} |\n")

            f.write("\n")

            # Top performers
            f.write("## Top Performers\n\n")

            # Fastest TTFT
            fastest = min(successful, key=lambda x: x.ttft_ms if x.ttft_ms > 0 else float('inf'))
            f.write(f"### Fastest TTFT: **{fastest.model_name}** ({fastest.ttft_ms:.1f}ms)\n\n")

            # Highest throughput
            highest_tp = max(successful, key=lambda x: x.throughput)
            f.write(f"### Highest Throughput: **{highest_tp.model_name}** ({highest_tp.throughput:.1f} tok/s)\n\n")

            # Best efficiency (throughput / VRAM)
            for r in successful:
                r.efficiency = r.throughput / max(r.vram_mb, 1) * 1000
            most_efficient = max(successful, key=lambda x: x.efficiency)
            f.write(f"### Most Efficient: **{most_efficient.model_name}** ({most_efficient.efficiency:.2f} tok/s per GB)\n\n")

        if failed:
            f.write("## Failed Models\n\n")
            for r in failed:
                f.write(f"- **{r.model_name}**: {r.error}\n")
            f.write("\n")

        f.write("---\n")
        f.write("*Generated by KVCache Auto-Tuner*\n")

    print(f"\nReport saved: {report_path}")

    # Save JSON results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([{
            "model_id": r.model_id,
            "model_name": r.model_name,
            "family": r.family,
            "params": r.params,
            "ttft_ms": r.ttft_ms,
            "throughput": r.throughput,
            "vram_mb": r.vram_mb,
            "score": r.score,
            "best_config": r.best_config,
            "success": r.success,
            "error": r.error,
            "duration_s": r.duration_s,
        } for r in results], f, indent=2)

    print(f"JSON saved: {json_path}")

    return successful


def generate_comparison_charts(results: list[BenchmarkResult], output_dir: Path):
    """Generate comparison charts."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping charts")
        return

    successful = [r for r in results if r.success and r.throughput > 0]
    if not successful:
        return

    # Color scheme
    COLORS = {
        'Qwen': '#FF6B6B',
        'Meta': '#4ECDC4',
        'Microsoft': '#45B7D1',
        'Google': '#96CEB4',
        'OpenAI': '#FFEAA7',
        'default': '#95A5A6',
    }

    plt.style.use('dark_background')

    # Sort by throughput
    sorted_results = sorted(successful, key=lambda x: x.throughput, reverse=True)

    # 1. Throughput comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    models = [r.model_name for r in sorted_results]
    throughputs = [r.throughput for r in sorted_results]
    colors = [COLORS.get(r.family, COLORS['default']) for r in sorted_results]

    bars = ax.barh(range(len(models)), throughputs, color=colors, edgecolor='white', linewidth=0.5)

    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{tp:.0f}', va='center', fontsize=10, color='white')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Throughput (tokens/second)')
    ax.set_title('Model Throughput Comparison', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_throughput.png', dpi=150, bbox_inches='tight',
                facecolor='#1F2937', edgecolor='none')
    plt.close()

    # 2. TTFT comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_by_ttft = sorted(successful, key=lambda x: x.ttft_ms if x.ttft_ms > 0 else float('inf'))
    models = [r.model_name for r in sorted_by_ttft]
    ttfts = [r.ttft_ms for r in sorted_by_ttft]
    colors = [COLORS.get(r.family, COLORS['default']) for r in sorted_by_ttft]

    bars = ax.barh(range(len(models)), ttfts, color=colors, edgecolor='white', linewidth=0.5)

    for bar, ttft in zip(bars, ttfts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{ttft:.1f}ms', va='center', fontsize=10, color='white')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Time to First Token (ms) ← Lower is better')
    ax.set_title('Model TTFT Comparison', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_ttft.png', dpi=150, bbox_inches='tight',
                facecolor='#1F2937', edgecolor='none')
    plt.close()

    # 3. VRAM usage
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_by_vram = sorted(successful, key=lambda x: x.vram_mb)
    models = [r.model_name for r in sorted_by_vram]
    vrams = [r.vram_mb for r in sorted_by_vram]
    colors = [COLORS.get(r.family, COLORS['default']) for r in sorted_by_vram]

    bars = ax.barh(range(len(models)), vrams, color=colors, edgecolor='white', linewidth=0.5)

    for bar, vram in zip(bars, vrams):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f'{vram:.0f} MB', va='center', fontsize=10, color='white')

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Peak VRAM Usage (MB)')
    ax.set_title('Model VRAM Comparison', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=8000, color='red', linestyle='--', alpha=0.5, label='8GB Limit')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_vram.png', dpi=150, bbox_inches='tight',
                facecolor='#1F2937', edgecolor='none')
    plt.close()

    # 4. Scatter: TTFT vs Throughput
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in successful:
        color = COLORS.get(r.family, COLORS['default'])
        size = 100 + (r.vram_mb / 1000) * 50
        ax.scatter(r.ttft_ms, r.throughput, s=size, c=color, alpha=0.8,
                   edgecolors='white', linewidths=1, label=r.family)
        ax.annotate(r.model_name, (r.ttft_ms, r.throughput),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, color='white')

    ax.set_xlabel('Time to First Token (ms) ← Lower is better')
    ax.set_ylabel('Throughput (tok/s) ↑ Higher is better')
    ax.set_title('Performance Trade-off: TTFT vs Throughput', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_scatter.png', dpi=150, bbox_inches='tight',
                facecolor='#1F2937', edgecolor='none')
    plt.close()

    # 5. Hero chart with all info
    fig = plt.figure(figsize=(16, 9))

    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('KVCache Auto-Tuner - Multi-Model Benchmark', fontsize=18, fontweight='bold', y=0.98)

    # Throughput (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    top_tp = sorted_results[:6]
    ax1.barh(range(len(top_tp)), [r.throughput for r in top_tp],
             color=[COLORS.get(r.family, COLORS['default']) for r in top_tp],
             edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(top_tp)))
    ax1.set_yticklabels([r.model_name for r in top_tp])
    ax1.set_xlabel('Throughput (tok/s)')
    ax1.set_title('Highest Throughput', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # TTFT (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    top_ttft = sorted(successful, key=lambda x: x.ttft_ms if x.ttft_ms > 0 else float('inf'))[:6]
    ax2.barh(range(len(top_ttft)), [r.ttft_ms for r in top_ttft],
             color=[COLORS.get(r.family, COLORS['default']) for r in top_ttft],
             edgecolor='white', linewidth=0.5)
    ax2.set_yticks(range(len(top_ttft)))
    ax2.set_yticklabels([r.model_name for r in top_ttft])
    ax2.set_xlabel('TTFT (ms)')
    ax2.set_title('Fastest TTFT', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # VRAM (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    top_vram = sorted(successful, key=lambda x: x.vram_mb)[:6]
    ax3.barh(range(len(top_vram)), [r.vram_mb for r in top_vram],
             color=[COLORS.get(r.family, COLORS['default']) for r in top_vram],
             edgecolor='white', linewidth=0.5)
    ax3.set_yticks(range(len(top_vram)))
    ax3.set_yticklabels([r.model_name for r in top_vram])
    ax3.set_xlabel('VRAM (MB)')
    ax3.set_title('Lowest VRAM Usage', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # Summary table (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Create summary text
    summary_text = f"""
    Models Tested: {len(successful)}

    Best Throughput:
    {sorted_results[0].model_name}
    {sorted_results[0].throughput:.0f} tok/s

    Fastest TTFT:
    {top_ttft[0].model_name}
    {top_ttft[0].ttft_ms:.1f} ms

    Most Efficient:
    {top_vram[0].model_name}
    {top_vram[0].vram_mb:.0f} MB VRAM
    """

    ax4.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
             transform=ax4.transAxes, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#374151', alpha=0.8))
    ax4.set_title('Summary', fontweight='bold')

    plt.savefig(output_dir / 'comparison_hero.png', dpi=150, bbox_inches='tight',
                facecolor='#1F2937', edgecolor='none')
    plt.close()

    print(f"\nCharts saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (4 models)")
    parser.add_argument("--full", action="store_true", help="Run full benchmark (all models)")
    parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    parser.add_argument("--output", "-o", default="benchmark_multi", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select models
    if args.models:
        models = [m for m in BENCHMARK_MODELS if m["id"] in args.models or m["name"] in args.models]
    elif args.quick:
        models = [m for m in BENCHMARK_MODELS if m["id"] in QUICK_MODELS]
    elif args.full:
        models = [m for m in BENCHMARK_MODELS if "skip_reason" not in m]
    else:
        # Default: quick benchmark
        models = [m for m in BENCHMARK_MODELS if m["id"] in QUICK_MODELS]

    print(f"\n{'#'*60}")
    print(f"# KVCache Auto-Tuner - Multi-Model Benchmark")
    print(f"# Models: {len(models)}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}\n")

    # Run benchmarks
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Starting benchmark...")
        result = run_single_benchmark(model, output_dir)
        results.append(result)

    # Generate reports
    print("\n" + "="*60)
    print("Generating comparison report...")
    successful = generate_comparison_report(results, output_dir)

    if successful:
        print("\nGenerating comparison charts...")
        generate_comparison_charts(results, output_dir)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print(f"Results: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
