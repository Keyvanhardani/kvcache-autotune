"""
Report generation for KVCache Auto-Tuner.

Generates human-readable reports in Markdown and HTML formats
from tuning results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from kvat.core.schema import BenchmarkResult, TuneResult


class ReportGenerator:
    """
    Generates reports from tuning results.

    Supports:
    - Markdown (lightweight, CI-friendly)
    - HTML (rich, visual)
    """

    def __init__(self, result: TuneResult) -> None:
        self.result = result

    def generate_markdown(self) -> str:
        """Generate Markdown report."""
        lines = [
            "# KVCache Auto-Tuner Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary",
            "",
            f"- **Model:** `{self.result.model_id}`",
            f"- **Device:** {self.result.device.value}",
            f"- **Profile:** {self.result.profile.name}",
            f"- **Tuning Duration:** {self.result.tuning_duration_seconds:.1f}s",
            f"- **Confidence:** {self.result.confidence * 100:.1f}%",
            "",
            "## Best Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Cache Strategy | {self.result.best_config.cache_strategy.value} |",
            f"| Attention Backend | {self.result.best_config.attention_backend.value} |",
            f"| Data Type | {self.result.best_config.dtype.value} |",
            f"| torch.compile | {self.result.best_config.use_torch_compile} |",
            f"| **Score** | **{self.result.best_score:.2f}** |",
            "",
            "## Performance Metrics",
            "",
        ]

        # Find best result metrics
        best_metrics = self._get_best_metrics()
        if best_metrics:
            lines.extend([
                "| Metric | Value |",
                "|--------|-------|",
                f"| TTFT (mean) | {best_metrics.ttft_mean_ms:.2f} ms |",
                f"| TTFT (std) | {best_metrics.ttft_std_ms:.2f} ms |",
                f"| Throughput (mean) | {best_metrics.throughput_mean:.2f} tok/s |",
            ])

            if best_metrics.peak_vram_mb:
                lines.append(
                    f"| Peak VRAM | {best_metrics.peak_vram_mb:.0f} MB |"
                )
            if best_metrics.peak_ram_mb:
                lines.append(
                    f"| Peak RAM | {best_metrics.peak_ram_mb:.0f} MB |"
                )

            lines.append(
                f"| Success Rate | {best_metrics.success_rate * 100:.0f}% |"
            )
            lines.append("")

        # Results table
        lines.extend([
            "## All Results",
            "",
            "| Configuration | Score | TTFT (ms) | Throughput (tok/s) | VRAM (MB) |",
            "|--------------|-------|-----------|-------------------|-----------|",
        ])

        sorted_results = sorted(
            self.result.all_results,
            key=lambda r: r.score,
            reverse=True,
        )

        for result in sorted_results[:10]:  # Top 10
            config_name = (
                f"{result.candidate.cache_strategy.value}/"
                f"{result.candidate.attention_backend.value}"
            )
            vram = f"{result.peak_vram_mb:.0f}" if result.peak_vram_mb else "-"

            is_best = result.candidate == self.result.best_config
            prefix = "**" if is_best else ""
            suffix = "**" if is_best else ""

            lines.append(
                f"| {prefix}{config_name}{suffix} | "
                f"{result.score:.2f} | "
                f"{result.ttft_mean_ms:.2f} | "
                f"{result.throughput_mean:.2f} | "
                f"{vram} |"
            )

        lines.append("")

        # System info
        if self.result.system_info:
            lines.extend([
                "## System Information",
                "",
            ])

            if "gpu" in self.result.system_info:
                gpu = self.result.system_info["gpu"]
                lines.extend([
                    f"- **GPU:** {gpu.get('name', 'Unknown')}",
                    f"- **GPU Memory:** {gpu.get('memory_mb', 0):.0f} MB",
                ])

            if "ram_total_mb" in self.result.system_info:
                lines.append(
                    f"- **RAM:** {self.result.system_info['ram_total_mb']:.0f} MB"
                )

            lines.extend([
                f"- **Platform:** {self.result.system_info.get('platform', 'Unknown')}",
                f"- **Python:** {self.result.system_info.get('python_version', 'Unknown')}",
                "",
            ])

        # Usage instructions
        lines.extend([
            "## Usage",
            "",
            "Apply the optimized configuration using:",
            "",
            "```bash",
            f"kvat apply --plan {self.result.model_id.replace('/', '_')}_plan.json",
            "```",
            "",
            "Or copy the generated `optimized_config.py` snippet.",
            "",
            "---",
            "",
            "*Generated by [KVCache Auto-Tuner](https://github.com/Keyvanhardani/kvcache-autotune)*",
            "",
            "**[Keyvan.ai](https://keyvan.ai)** | [LinkedIn](https://www.linkedin.com/in/keyvanhardani)",
            "",
            "*Made in Germany with dedication for the HuggingFace Community*",
        ])

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generate HTML report with styling."""
        best_metrics = self._get_best_metrics()

        sorted_results = sorted(
            self.result.all_results,
            key=lambda r: r.score,
            reverse=True,
        )

        # Build results table rows
        results_rows = []
        for i, result in enumerate(sorted_results[:15]):
            config_name = (
                f"{result.candidate.cache_strategy.value}/"
                f"{result.candidate.attention_backend.value}"
            )
            vram = f"{result.peak_vram_mb:.0f}" if result.peak_vram_mb else "-"

            is_best = result.candidate == self.result.best_config
            row_class = "best-row" if is_best else ""

            results_rows.append(f"""
                <tr class="{row_class}">
                    <td>{i + 1}</td>
                    <td>{config_name}</td>
                    <td>{result.candidate.dtype.value}</td>
                    <td class="score">{result.score:.2f}</td>
                    <td>{result.ttft_mean_ms:.2f}</td>
                    <td>{result.throughput_mean:.2f}</td>
                    <td>{vram}</td>
                    <td>{result.success_rate * 100:.0f}%</td>
                </tr>
            """)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KVCache Auto-Tuner Report</title>
    <style>
        :root {{
            --primary: #2563eb;
            --success: #16a34a;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}

        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .card h3 {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .card .value {{
            font-size: 1.75rem;
            font-weight: 600;
        }}

        .card .value.highlight {{
            color: var(--primary);
        }}

        .card .unit {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        .best-config {{
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
        }}

        .best-config h3 {{
            color: rgba(255,255,255,0.8);
        }}

        .best-config .value {{
            color: white;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: #f1f5f9;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .best-row {{
            background: #f0fdf4;
        }}

        .best-row td {{
            font-weight: 500;
        }}

        .score {{
            font-weight: 600;
            color: var(--primary);
        }}

        .section-title {{
            font-size: 1.25rem;
            margin: 2rem 0 1rem;
        }}

        .footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>KVCache Auto-Tuner Report</h1>
        <p class="subtitle">
            Model: <strong>{self.result.model_id}</strong> |
            Profile: <strong>{self.result.profile.name}</strong> |
            Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </p>

        <div class="grid">
            <div class="card best-config">
                <h3>Best Configuration</h3>
                <div class="value">{self.result.best_config.cache_strategy.value}</div>
                <div class="unit">
                    {self.result.best_config.attention_backend.value} /
                    {self.result.best_config.dtype.value}
                </div>
            </div>

            <div class="card">
                <h3>Score</h3>
                <div class="value highlight">{self.result.best_score:.2f}</div>
                <div class="unit">out of 100</div>
            </div>

            <div class="card">
                <h3>Confidence</h3>
                <div class="value">{self.result.confidence * 100:.0f}%</div>
                <div class="unit">recommendation confidence</div>
            </div>

            <div class="card">
                <h3>TTFT</h3>
                <div class="value">{best_metrics.ttft_mean_ms if best_metrics else 0:.1f}</div>
                <div class="unit">milliseconds (mean)</div>
            </div>

            <div class="card">
                <h3>Throughput</h3>
                <div class="value">{best_metrics.throughput_mean if best_metrics else 0:.1f}</div>
                <div class="unit">tokens/second</div>
            </div>

            <div class="card">
                <h3>Peak VRAM</h3>
                <div class="value">{best_metrics.peak_vram_mb if best_metrics and best_metrics.peak_vram_mb else 0:.0f}</div>
                <div class="unit">megabytes</div>
            </div>
        </div>

        <h2 class="section-title">All Results</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Configuration</th>
                    <th>DType</th>
                    <th>Score</th>
                    <th>TTFT (ms)</th>
                    <th>Throughput</th>
                    <th>VRAM (MB)</th>
                    <th>Success</th>
                </tr>
            </thead>
            <tbody>
                {''.join(results_rows)}
            </tbody>
        </table>

        <div class="footer">
            <p>
                Tuning completed in {self.result.tuning_duration_seconds:.1f} seconds |
                {len(self.result.all_results)} configurations tested
            </p>
            <p style="margin-top: 1rem;">
                Generated by <a href="https://github.com/Keyvanhardani/kvcache-autotune" style="color: var(--primary);">KVCache Auto-Tuner</a>
            </p>
            <p style="margin-top: 0.5rem;">
                <a href="https://keyvan.ai" style="color: var(--primary); font-weight: 600;">Keyvan.ai</a> |
                <a href="https://www.linkedin.com/in/keyvanhardani" style="color: var(--text-muted);">LinkedIn</a>
            </p>
            <p style="margin-top: 0.5rem; font-style: italic;">
                Made in Germany with dedication for the HuggingFace Community
            </p>
        </div>
    </div>
</body>
</html>"""

        return html

    def _get_best_metrics(self) -> BenchmarkResult | None:
        """Get metrics for best configuration."""
        for result in self.result.all_results:
            if result.candidate == self.result.best_config:
                return result
        return None

    def save(
        self,
        output_dir: str | Path,
        *,
        markdown: bool = True,
        html: bool = True,
    ) -> dict[str, Path]:
        """
        Save reports to files.

        Args:
            output_dir: Output directory
            markdown: Generate Markdown report
            html: Generate HTML report

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        if markdown:
            md_path = output_dir / "report.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.generate_markdown())
            saved["markdown"] = md_path

        if html:
            html_path = output_dir / "report.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.generate_html())
            saved["html"] = html_path

        return saved
