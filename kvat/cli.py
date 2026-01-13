"""
Command-line interface for KVCache Auto-Tuner.

Provides commands:
- tune: Run optimization search
- apply: Apply a saved plan
- compare: Compare two plans
- profiles: List available profiles
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from kvat import __version__
from kvat.core.profiles import (
    get_profile,
    get_profile_description,
    list_profiles,
    load_profile_from_json,
)
from kvat.core.schema import (
    DeviceType,
    TuneConfig,
    WorkloadProfile,
)

app = typer.Typer(
    name="kvat",
    help="KVCache Auto-Tuner - Automatic KV-cache optimization for Transformers",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"KVCache Auto-Tuner v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """KVCache Auto-Tuner - Find optimal KV-cache configuration."""
    pass


@app.command()
def tune(
    model: str = typer.Argument(
        ...,
        help="HuggingFace model ID or local path",
    ),
    profile: str = typer.Option(
        "chat-agent",
        "--profile",
        "-p",
        help="Workload profile (chat-agent, rag, longform, or path to JSON)",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help="Device to use (cuda, cpu, mps)",
    ),
    output: str = typer.Option(
        "./kvat_results",
        "--out",
        "-o",
        help="Output directory for results",
    ),
    context: str | None = typer.Option(
        None,
        "--context",
        "-c",
        help="Context lengths to test (comma-separated, e.g., 2048,4096,8192)",
    ),
    max_vram: float | None = typer.Option(
        None,
        "--max-vram",
        help="Maximum VRAM in MB (soft limit)",
    ),
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        "-t",
        help="Timeout per candidate in seconds",
    ),
    compile: bool = typer.Option(
        False,
        "--compile",
        help="Include torch.compile candidates (experimental)",
    ),
    no_html: bool = typer.Option(
        False,
        "--no-html",
        help="Skip HTML report generation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Run KV-cache optimization for a model.

    Examples:

        kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

        kvat tune ./my_model --profile rag --context 8192,16384

        kvat tune mistralai/Mistral-7B-v0.1 --device cuda --max-vram 8000
    """
    setup_logging(verbose)

    console.print(
        Panel.fit(
            "[bold blue]KVCache Auto-Tuner[/bold blue]\n"
            f"Model: [green]{model}[/green]",
            border_style="blue",
        )
    )

    # Load or create profile
    workload_profile = _load_profile(profile, context)

    # Map device
    device_type = _parse_device(device)

    # Create config
    config = TuneConfig(
        model_id=model,
        device=device_type,
        profile=workload_profile,
        max_vram_mb=max_vram,
        timeout_seconds=timeout,
        enable_torch_compile=compile,
        output_dir=output,
        generate_html_report=not no_html,
    )

    # Run tuning
    try:
        from kvat.core.planner import PlanBuilder
        from kvat.core.report import ReportGenerator
        from kvat.core.search import TuningSearch
        from kvat.engines.transformers import TransformersAdapter

        adapter = TransformersAdapter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Tuning...", total=100)

            def update_progress(p):
                pct = (p.completed_candidates / p.total_candidates) * 100
                desc = f"Testing {p.current_candidate.cache_strategy.value if p.current_candidate else '...'}"
                progress.update(task, completed=pct, description=desc)

            search = TuningSearch(
                config=config,
                adapter=adapter,
                progress_callback=update_progress,
            )

            result = search.run()

        # Display results
        _display_results(result)

        # Generate outputs
        console.print("\n[bold]Saving results...[/bold]")

        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save plan
        planner = PlanBuilder(result)
        plan_files = planner.save_plan(output_path)
        console.print(f"  Plan: [green]{plan_files['plan']}[/green]")
        console.print(f"  Snippet: [green]{plan_files['snippet']}[/green]")

        # Save reports
        reporter = ReportGenerator(result)
        report_files = reporter.save(output_path, html=not no_html)
        console.print(f"  Report: [green]{report_files['markdown']}[/green]")
        if "html" in report_files:
            console.print(f"  HTML: [green]{report_files['html']}[/green]")

        console.print("\n[bold green]Tuning complete![/bold green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Install with: pip install transformers torch")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def apply(
    plan: str = typer.Argument(
        ...,
        help="Path to plan JSON file",
    ),
    print_snippet: bool = typer.Option(
        False,
        "--print-snippet",
        "-p",
        help="Print code snippet to stdout",
    ),
    output: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file for snippet (default: stdout)",
    ),
) -> None:
    """
    Apply a saved optimization plan.

    Examples:

        kvat apply ./kvat_results/best_plan.json --print-snippet

        kvat apply plan.json --out config.py
    """
    from kvat.core.planner import load_plan

    try:
        plan_data = load_plan(plan)

        console.print(f"[bold]Plan:[/bold] {plan}")
        console.print(f"  Model: {plan_data['model_id']}")
        console.print(f"  Profile: {plan_data['profile']}")
        console.print(f"  Best Config: {plan_data['best_config']['cache_strategy']}")
        console.print(f"  Score: {plan_data['best_score']:.2f}")

        if print_snippet or output:
            snippet = plan_data.get("code_snippet", "# No snippet available")

            if output:
                with open(output, "w") as f:
                    f.write(snippet)
                console.print(f"\n[green]Snippet saved to: {output}[/green]")
            else:
                console.print("\n[bold]Code Snippet:[/bold]")
                console.print(snippet)

    except FileNotFoundError:
        console.print(f"[red]Plan file not found: {plan}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error loading plan: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    baseline: str = typer.Argument(
        ...,
        help="Baseline plan JSON",
    ),
    candidate: str = typer.Argument(
        ...,
        help="Candidate plan JSON",
    ),
) -> None:
    """
    Compare two optimization plans.

    Examples:

        kvat compare old_plan.json new_plan.json
    """
    from kvat.core.planner import load_plan

    try:
        baseline_data = load_plan(baseline)
        candidate_data = load_plan(candidate)

        table = Table(title="Plan Comparison")
        table.add_column("Metric", style="cyan")
        table.add_column("Baseline", style="yellow")
        table.add_column("Candidate", style="green")
        table.add_column("Delta", style="magenta")

        # Score comparison
        b_score = baseline_data["best_score"]
        c_score = candidate_data["best_score"]
        delta_score = c_score - b_score
        delta_str = f"+{delta_score:.2f}" if delta_score > 0 else f"{delta_score:.2f}"
        table.add_row("Score", f"{b_score:.2f}", f"{c_score:.2f}", delta_str)

        # Config comparison
        b_cfg = baseline_data["best_config"]
        c_cfg = candidate_data["best_config"]
        table.add_row(
            "Cache Strategy",
            b_cfg["cache_strategy"],
            c_cfg["cache_strategy"],
            "changed" if b_cfg["cache_strategy"] != c_cfg["cache_strategy"] else "-",
        )
        table.add_row(
            "Attention Backend",
            b_cfg["attention_backend"],
            c_cfg["attention_backend"],
            "changed" if b_cfg["attention_backend"] != c_cfg["attention_backend"] else "-",
        )

        # Benchmark comparison
        if "benchmarks" in baseline_data and "benchmarks" in candidate_data:
            b_bench = baseline_data["benchmarks"].get("summary", {})
            c_bench = candidate_data["benchmarks"].get("summary", {})

            if "ttft_mean_ms" in b_bench and "ttft_mean_ms" in c_bench:
                b_ttft = b_bench["ttft_mean_ms"]
                c_ttft = c_bench["ttft_mean_ms"]
                delta_ttft = c_ttft - b_ttft
                pct = (delta_ttft / b_ttft * 100) if b_ttft > 0 else 0
                table.add_row(
                    "TTFT (ms)",
                    f"{b_ttft:.2f}",
                    f"{c_ttft:.2f}",
                    f"{pct:+.1f}%",
                )

            if "throughput_mean_tok_s" in b_bench and "throughput_mean_tok_s" in c_bench:
                b_tput = b_bench["throughput_mean_tok_s"]
                c_tput = c_bench["throughput_mean_tok_s"]
                delta_tput = c_tput - b_tput
                pct = (delta_tput / b_tput * 100) if b_tput > 0 else 0
                table.add_row(
                    "Throughput (tok/s)",
                    f"{b_tput:.2f}",
                    f"{c_tput:.2f}",
                    f"{pct:+.1f}%",
                )

        console.print(table)

        # Verdict
        if c_score > b_score:
            console.print(
                f"\n[bold green]Candidate is better by {c_score - b_score:.2f} points[/bold green]"
            )
        elif c_score < b_score:
            console.print(
                f"\n[bold yellow]Baseline is better by {b_score - c_score:.2f} points[/bold yellow]"
            )
        else:
            console.print("\n[bold]Plans have equal scores[/bold]")

    except FileNotFoundError as e:
        console.print(f"[red]File not found: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("profiles")
def list_profiles_cmd() -> None:
    """List available workload profiles."""
    table = Table(title="Available Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description", style="white")

    for name in list_profiles():
        profile = get_profile(name)
        if profile:
            desc = get_profile_description(name)
            # Truncate description
            if len(desc) > 60:
                desc = desc[:57] + "..."
            table.add_row(name, profile.profile_type.value, desc)

    console.print(table)

    console.print("\n[dim]Use --profile <name> or --profile <path.json> for custom profiles[/dim]")


@app.command()
def info() -> None:
    """Show system information and capabilities."""
    from kvat.probes.cpu import get_cpu_count, get_system_ram_info
    from kvat.probes.gpu import get_all_gpu_info, is_cuda_available

    console.print("[bold]System Information[/bold]\n")

    # Python version
    console.print(f"Python: {sys.version.split()[0]}")

    # CPU
    console.print(f"CPU Cores: {get_cpu_count()}")

    # RAM
    ram_info = get_system_ram_info()
    if ram_info:
        console.print(f"RAM: {ram_info.total_mb / 1024:.1f} GB total, {ram_info.available_mb / 1024:.1f} GB available")

    # GPU
    console.print(f"\nCUDA Available: {is_cuda_available()}")
    if is_cuda_available():
        gpus = get_all_gpu_info()
        for gpu in gpus:
            console.print(f"  GPU {gpu.index}: {gpu.name} ({gpu.total_memory_mb / 1024:.1f} GB)")

    # Dependencies
    console.print("\n[bold]Dependencies[/bold]")

    deps = [
        ("transformers", True),
        ("torch", True),
        ("flash_attn", False),
        ("xformers", False),
        ("pynvml", False),
        ("psutil", False),
    ]

    for dep, required in deps:
        try:
            __import__(dep)
            status = "[green]installed[/green]"
        except ImportError:
            status = "[yellow]not installed[/yellow]" if not required else "[red]missing (required)[/red]"
        console.print(f"  {dep}: {status}")


# =============================================================================
# Helper Functions
# =============================================================================

def _load_profile(profile_str: str, context_override: str | None) -> WorkloadProfile:
    """Load profile from name or path."""
    # Try built-in profile first
    profile = get_profile(profile_str)

    if profile is None:
        # Try loading from file
        try:
            profile = load_profile_from_json(profile_str)
        except FileNotFoundError:
            console.print(f"[red]Profile not found: {profile_str}[/red]")
            console.print(f"Available profiles: {', '.join(list_profiles())}")
            raise typer.Exit(1)

    # Override context lengths if specified
    if context_override:
        try:
            context_lengths = [int(x.strip()) for x in context_override.split(",")]
            profile = profile.model_copy(update={"context_lengths": context_lengths})
        except ValueError:
            console.print("[red]Invalid context format. Use comma-separated integers.[/red]")
            raise typer.Exit(1)

    return profile


def _parse_device(device_str: str) -> DeviceType:
    """Parse device string to DeviceType."""
    device_map = {
        "cuda": DeviceType.CUDA,
        "gpu": DeviceType.CUDA,
        "cpu": DeviceType.CPU,
        "mps": DeviceType.MPS,
    }

    device_lower = device_str.lower()
    if device_lower not in device_map:
        console.print(f"[red]Invalid device: {device_str}[/red]")
        console.print("Available: cuda, cpu, mps")
        raise typer.Exit(1)

    return device_map[device_lower]


def _display_results(result) -> None:
    """Display tuning results in a table."""
    console.print("\n[bold]Results[/bold]\n")

    # Best configuration
    best = result.best_config
    console.print(
        Panel(
            f"[bold green]Best Configuration[/bold green]\n\n"
            f"Cache Strategy: [cyan]{best.cache_strategy.value}[/cyan]\n"
            f"Attention Backend: [cyan]{best.attention_backend.value}[/cyan]\n"
            f"Data Type: [cyan]{best.dtype.value}[/cyan]\n"
            f"torch.compile: [cyan]{best.use_torch_compile}[/cyan]\n\n"
            f"[bold]Score: {result.best_score:.2f}[/bold]\n"
            f"Confidence: {result.confidence * 100:.0f}%",
            border_style="green",
        )
    )

    # Results table
    if len(result.all_results) > 1:
        table = Table(title="Top Configurations")
        table.add_column("#", style="dim")
        table.add_column("Config", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("TTFT (ms)", style="yellow")
        table.add_column("Throughput", style="blue")

        sorted_results = sorted(
            result.all_results,
            key=lambda r: r.score,
            reverse=True,
        )

        for i, r in enumerate(sorted_results[:5], 1):
            config_name = f"{r.candidate.cache_strategy.value}/{r.candidate.attention_backend.value}"
            table.add_row(
                str(i),
                config_name,
                f"{r.score:.2f}",
                f"{r.ttft_mean_ms:.2f}",
                f"{r.throughput_mean:.1f} tok/s",
            )

        console.print(table)


def cli() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    cli()
