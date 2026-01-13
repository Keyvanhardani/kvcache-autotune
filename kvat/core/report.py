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
    <title>KVCache Auto-Tuner Report | Keyvan.ai</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --success: #16a34a;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --header-bg: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
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
        }}

        .header {{
            background: var(--header-bg);
            color: white;
            padding: 1.5rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-brand {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .header-logo {{
            width: 48px;
            height: 48px;
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 700;
        }}

        .header-title {{
            font-size: 1.5rem;
            font-weight: 600;
        }}

        .header-subtitle {{
            font-size: 0.875rem;
            opacity: 0.8;
        }}

        .header-links {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}

        .header-links a {{
            color: white;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
        }}

        .header-links a:hover {{
            background: rgba(255,255,255,0.2);
        }}

        .header-links svg {{
            width: 18px;
            height: 18px;
            fill: currentColor;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
            color: var(--text);
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
    <div class="header">
        <div class="header-brand">
            <div class="header-logo">KV</div>
            <div>
                <div class="header-title">KVCache Auto-Tuner</div>
                <div class="header-subtitle">by Keyvan Hardani</div>
            </div>
        </div>
        <div class="header-links">
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README.md" title="English">EN</a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README_DE.md" title="Deutsch">DE</a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README_FR.md" title="Francais">FR</a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README_ES.md" title="Espanol">ES</a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README_FA.md" title="Farsi">FA</a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/README_AR.md" title="Arabic">AR</a>
            <span style="border-left: 1px solid rgba(255,255,255,0.3); height: 24px; margin: 0 0.5rem;"></span>
            <a href="https://keyvan.ai" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
                Keyvan.ai
            </a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub
            </a>
            <a href="https://www.linkedin.com/in/keyvanhardani" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                LinkedIn
            </a>
        </div>
    </div>

    <div class="container">
        <h1>Performance Report</h1>
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
            <p style="margin-top: 1rem; font-style: italic;">
                Made in Germany with dedication for the HuggingFace Community
            </p>
            <p style="margin-top: 0.5rem;">
                <strong>&copy; {datetime.utcnow().year} <a href="https://keyvan.ai" style="color: var(--primary);">Keyvan.ai</a></strong> |
                <a href="https://github.com/Keyvanhardani/kvcache-autotune" style="color: var(--text-muted);">GitHub</a> |
                <a href="https://www.linkedin.com/in/keyvanhardani" style="color: var(--text-muted);">Keyvan Hardani</a>
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
