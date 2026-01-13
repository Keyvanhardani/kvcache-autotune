"""
Report generation for KVCache Auto-Tuner.

Generates human-readable reports in Markdown and HTML formats
from tuning results with multi-language support.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from kvat.core.schema import BenchmarkResult, TuneResult

# Supported languages
Language = Literal["en", "de", "fr", "es", "fa", "ar"]

# Translation dictionary
TRANSLATIONS: dict[str, dict[Language, str]] = {
    "report_title": {
        "en": "KVCache Auto-Tuner Report",
        "de": "KVCache Auto-Tuner Bericht",
        "fr": "Rapport KVCache Auto-Tuner",
        "es": "Informe KVCache Auto-Tuner",
        "fa": "گزارش KVCache Auto-Tuner",
        "ar": "تقرير KVCache Auto-Tuner",
    },
    "performance_report": {
        "en": "Performance Report",
        "de": "Leistungsbericht",
        "fr": "Rapport de Performance",
        "es": "Informe de Rendimiento",
        "fa": "گزارش عملکرد",
        "ar": "تقرير الأداء",
    },
    "generated": {
        "en": "Generated",
        "de": "Erstellt",
        "fr": "Genere",
        "es": "Generado",
        "fa": "تاریخ ایجاد",
        "ar": "تم الإنشاء",
    },
    "summary": {
        "en": "Summary",
        "de": "Zusammenfassung",
        "fr": "Resume",
        "es": "Resumen",
        "fa": "خلاصه",
        "ar": "ملخص",
    },
    "model": {
        "en": "Model",
        "de": "Modell",
        "fr": "Modele",
        "es": "Modelo",
        "fa": "مدل",
        "ar": "النموذج",
    },
    "device": {
        "en": "Device",
        "de": "Gerat",
        "fr": "Appareil",
        "es": "Dispositivo",
        "fa": "دستگاه",
        "ar": "الجهاز",
    },
    "profile": {
        "en": "Profile",
        "de": "Profil",
        "fr": "Profil",
        "es": "Perfil",
        "fa": "پروفایل",
        "ar": "الملف الشخصي",
    },
    "tuning_duration": {
        "en": "Tuning Duration",
        "de": "Optimierungsdauer",
        "fr": "Duree d'optimisation",
        "es": "Duracion de optimizacion",
        "fa": "مدت زمان بهینه‌سازی",
        "ar": "مدة التحسين",
    },
    "confidence": {
        "en": "Confidence",
        "de": "Konfidenz",
        "fr": "Confiance",
        "es": "Confianza",
        "fa": "اطمینان",
        "ar": "الثقة",
    },
    "best_configuration": {
        "en": "Best Configuration",
        "de": "Beste Konfiguration",
        "fr": "Meilleure Configuration",
        "es": "Mejor Configuracion",
        "fa": "بهترین پیکربندی",
        "ar": "أفضل تكوين",
    },
    "parameter": {
        "en": "Parameter",
        "de": "Parameter",
        "fr": "Parametre",
        "es": "Parametro",
        "fa": "پارامتر",
        "ar": "المعامل",
    },
    "value": {
        "en": "Value",
        "de": "Wert",
        "fr": "Valeur",
        "es": "Valor",
        "fa": "مقدار",
        "ar": "القيمة",
    },
    "cache_strategy": {
        "en": "Cache Strategy",
        "de": "Cache-Strategie",
        "fr": "Strategie de Cache",
        "es": "Estrategia de Cache",
        "fa": "استراتژی کش",
        "ar": "استراتيجية التخزين",
    },
    "attention_backend": {
        "en": "Attention Backend",
        "de": "Attention-Backend",
        "fr": "Backend Attention",
        "es": "Backend de Atencion",
        "fa": "بک‌اند توجه",
        "ar": "Backend Attention",
    },
    "data_type": {
        "en": "Data Type",
        "de": "Datentyp",
        "fr": "Type de Donnees",
        "es": "Tipo de Datos",
        "fa": "نوع داده",
        "ar": "نوع البيانات",
    },
    "score": {
        "en": "Score",
        "de": "Punktzahl",
        "fr": "Score",
        "es": "Puntuacion",
        "fa": "امتیاز",
        "ar": "النتيجة",
    },
    "performance_metrics": {
        "en": "Performance Metrics",
        "de": "Leistungsmetriken",
        "fr": "Metriques de Performance",
        "es": "Metricas de Rendimiento",
        "fa": "معیارهای عملکرد",
        "ar": "مقاييس الأداء",
    },
    "metric": {
        "en": "Metric",
        "de": "Metrik",
        "fr": "Metrique",
        "es": "Metrica",
        "fa": "معیار",
        "ar": "المقياس",
    },
    "ttft_mean": {
        "en": "TTFT (mean)",
        "de": "TTFT (Durchschnitt)",
        "fr": "TTFT (moyenne)",
        "es": "TTFT (promedio)",
        "fa": "TTFT (میانگین)",
        "ar": "TTFT (المتوسط)",
    },
    "ttft_std": {
        "en": "TTFT (std)",
        "de": "TTFT (Stdabw.)",
        "fr": "TTFT (ecart-type)",
        "es": "TTFT (desv. est.)",
        "fa": "TTFT (انحراف)",
        "ar": "TTFT (الانحراف)",
    },
    "throughput_mean": {
        "en": "Throughput (mean)",
        "de": "Durchsatz (Durchschnitt)",
        "fr": "Debit (moyenne)",
        "es": "Rendimiento (promedio)",
        "fa": "توان عملیاتی (میانگین)",
        "ar": "الإنتاجية (المتوسط)",
    },
    "peak_vram": {
        "en": "Peak VRAM",
        "de": "Max. VRAM",
        "fr": "VRAM Max",
        "es": "VRAM Maximo",
        "fa": "حداکثر VRAM",
        "ar": "ذروة VRAM",
    },
    "peak_ram": {
        "en": "Peak RAM",
        "de": "Max. RAM",
        "fr": "RAM Max",
        "es": "RAM Maximo",
        "fa": "حداکثر RAM",
        "ar": "ذروة RAM",
    },
    "success_rate": {
        "en": "Success Rate",
        "de": "Erfolgsrate",
        "fr": "Taux de Reussite",
        "es": "Tasa de Exito",
        "fa": "نرخ موفقیت",
        "ar": "معدل النجاح",
    },
    "all_results": {
        "en": "All Results",
        "de": "Alle Ergebnisse",
        "fr": "Tous les Resultats",
        "es": "Todos los Resultados",
        "fa": "همه نتایج",
        "ar": "جميع النتائج",
    },
    "configuration": {
        "en": "Configuration",
        "de": "Konfiguration",
        "fr": "Configuration",
        "es": "Configuracion",
        "fa": "پیکربندی",
        "ar": "التكوين",
    },
    "throughput": {
        "en": "Throughput",
        "de": "Durchsatz",
        "fr": "Debit",
        "es": "Rendimiento",
        "fa": "توان عملیاتی",
        "ar": "الإنتاجية",
    },
    "success": {
        "en": "Success",
        "de": "Erfolg",
        "fr": "Reussite",
        "es": "Exito",
        "fa": "موفقیت",
        "ar": "النجاح",
    },
    "system_information": {
        "en": "System Information",
        "de": "Systeminformationen",
        "fr": "Informations Systeme",
        "es": "Informacion del Sistema",
        "fa": "اطلاعات سیستم",
        "ar": "معلومات النظام",
    },
    "gpu": {
        "en": "GPU",
        "de": "GPU",
        "fr": "GPU",
        "es": "GPU",
        "fa": "GPU",
        "ar": "GPU",
    },
    "gpu_memory": {
        "en": "GPU Memory",
        "de": "GPU-Speicher",
        "fr": "Memoire GPU",
        "es": "Memoria GPU",
        "fa": "حافظه GPU",
        "ar": "ذاكرة GPU",
    },
    "ram": {
        "en": "RAM",
        "de": "RAM",
        "fr": "RAM",
        "es": "RAM",
        "fa": "RAM",
        "ar": "RAM",
    },
    "platform": {
        "en": "Platform",
        "de": "Plattform",
        "fr": "Plateforme",
        "es": "Plataforma",
        "fa": "پلتفرم",
        "ar": "المنصة",
    },
    "python": {
        "en": "Python",
        "de": "Python",
        "fr": "Python",
        "es": "Python",
        "fa": "Python",
        "ar": "Python",
    },
    "usage": {
        "en": "Usage",
        "de": "Verwendung",
        "fr": "Utilisation",
        "es": "Uso",
        "fa": "نحوه استفاده",
        "ar": "الاستخدام",
    },
    "apply_config": {
        "en": "Apply the optimized configuration using:",
        "de": "Die optimierte Konfiguration anwenden mit:",
        "fr": "Appliquer la configuration optimisee avec:",
        "es": "Aplicar la configuracion optimizada usando:",
        "fa": "اعمال پیکربندی بهینه با:",
        "ar": "تطبيق التكوين المحسّن باستخدام:",
    },
    "copy_snippet": {
        "en": "Or copy the generated `optimized_config.py` snippet.",
        "de": "Oder den generierten `optimized_config.py` Snippet kopieren.",
        "fr": "Ou copier le snippet `optimized_config.py` genere.",
        "es": "O copiar el snippet `optimized_config.py` generado.",
        "fa": "یا کد `optimized_config.py` تولید شده را کپی کنید.",
        "ar": "أو انسخ الكود `optimized_config.py` المُنشأ.",
    },
    "generated_by": {
        "en": "Generated by",
        "de": "Erstellt von",
        "fr": "Genere par",
        "es": "Generado por",
        "fa": "ایجاد شده توسط",
        "ar": "تم الإنشاء بواسطة",
    },
    "made_in_germany": {
        "en": "Made in Germany with dedication for the HuggingFace Community",
        "de": "Made in Germany mit Hingabe fur die HuggingFace Community",
        "fr": "Fabrique en Allemagne avec passion pour la communaute HuggingFace",
        "es": "Hecho en Alemania con dedicacion para la comunidad HuggingFace",
        "fa": "ساخته شده در آلمان با علاقه برای جامعه HuggingFace",
        "ar": "صنع في ألمانيا بإخلاص لمجتمع HuggingFace",
    },
    "out_of_100": {
        "en": "out of 100",
        "de": "von 100",
        "fr": "sur 100",
        "es": "de 100",
        "fa": "از 100",
        "ar": "من 100",
    },
    "recommendation_confidence": {
        "en": "recommendation confidence",
        "de": "Empfehlungskonfidenz",
        "fr": "confiance de recommandation",
        "es": "confianza de recomendacion",
        "fa": "اطمینان توصیه",
        "ar": "ثقة التوصية",
    },
    "milliseconds_mean": {
        "en": "milliseconds (mean)",
        "de": "Millisekunden (Durchschnitt)",
        "fr": "millisecondes (moyenne)",
        "es": "milisegundos (promedio)",
        "fa": "میلی‌ثانیه (میانگین)",
        "ar": "ميلي ثانية (المتوسط)",
    },
    "tokens_second": {
        "en": "tokens/second",
        "de": "Tokens/Sekunde",
        "fr": "tokens/seconde",
        "es": "tokens/segundo",
        "fa": "توکن/ثانیه",
        "ar": "رمز/ثانية",
    },
    "megabytes": {
        "en": "megabytes",
        "de": "Megabyte",
        "fr": "megaoctets",
        "es": "megabytes",
        "fa": "مگابایت",
        "ar": "ميجابايت",
    },
    "tuning_completed": {
        "en": "Tuning completed in",
        "de": "Optimierung abgeschlossen in",
        "fr": "Optimisation terminee en",
        "es": "Optimizacion completada en",
        "fa": "بهینه‌سازی تکمیل شد در",
        "ar": "اكتمل التحسين في",
    },
    "seconds": {
        "en": "seconds",
        "de": "Sekunden",
        "fr": "secondes",
        "es": "segundos",
        "fa": "ثانیه",
        "ar": "ثواني",
    },
    "configurations_tested": {
        "en": "configurations tested",
        "de": "Konfigurationen getestet",
        "fr": "configurations testees",
        "es": "configuraciones probadas",
        "fa": "پیکربندی تست شد",
        "ar": "تكوينات مختبرة",
    },
}

# Language names for display
LANGUAGE_NAMES: dict[Language, str] = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Francais",
    "es": "Espanol",
    "fa": "فارسی",
    "ar": "العربية",
}

# RTL languages
RTL_LANGUAGES: set[Language] = {"fa", "ar"}


class ReportGenerator:
    """
    Generates reports from tuning results.

    Supports:
    - Markdown (lightweight, CI-friendly)
    - HTML (rich, visual)
    - Multiple languages (EN, DE, FR, ES, FA, AR)
    """

    def __init__(
        self,
        result: TuneResult,
        language: Language = "en",
    ) -> None:
        self.result = result
        self.language = language
        self.is_rtl = language in RTL_LANGUAGES

    def t(self, key: str) -> str:
        """Get translated string for current language."""
        if key in TRANSLATIONS:
            return TRANSLATIONS[key].get(self.language, TRANSLATIONS[key]["en"])
        return key

    def generate_markdown(self) -> str:
        """Generate Markdown report."""
        rtl_start = '<div dir="rtl">\n\n' if self.is_rtl else ""
        rtl_end = "\n\n</div>" if self.is_rtl else ""

        lines = [
            rtl_start + f"# {self.t('report_title')}",
            "",
            f"**{self.t('generated')}:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            f"## {self.t('summary')}",
            "",
            f"- **{self.t('model')}:** `{self.result.model_id}`",
            f"- **{self.t('device')}:** {self.result.device.value}",
            f"- **{self.t('profile')}:** {self.result.profile.name}",
            f"- **{self.t('tuning_duration')}:** {self.result.tuning_duration_seconds:.1f}s",
            f"- **{self.t('confidence')}:** {self.result.confidence * 100:.1f}%",
            "",
            f"## {self.t('best_configuration')}",
            "",
            f"| {self.t('parameter')} | {self.t('value')} |",
            "|-----------|-------|",
            f"| {self.t('cache_strategy')} | {self.result.best_config.cache_strategy.value} |",
            f"| {self.t('attention_backend')} | {self.result.best_config.attention_backend.value} |",
            f"| {self.t('data_type')} | {self.result.best_config.dtype.value} |",
            f"| torch.compile | {self.result.best_config.use_torch_compile} |",
            f"| **{self.t('score')}** | **{self.result.best_score:.2f}** |",
            "",
            f"## {self.t('performance_metrics')}",
            "",
        ]

        best_metrics = self._get_best_metrics()
        if best_metrics:
            lines.extend([
                f"| {self.t('metric')} | {self.t('value')} |",
                "|--------|-------|",
                f"| {self.t('ttft_mean')} | {best_metrics.ttft_mean_ms:.2f} ms |",
                f"| {self.t('ttft_std')} | {best_metrics.ttft_std_ms:.2f} ms |",
                f"| {self.t('throughput_mean')} | {best_metrics.throughput_mean:.2f} tok/s |",
            ])

            if best_metrics.peak_vram_mb:
                lines.append(f"| {self.t('peak_vram')} | {best_metrics.peak_vram_mb:.0f} MB |")
            if best_metrics.peak_ram_mb:
                lines.append(f"| {self.t('peak_ram')} | {best_metrics.peak_ram_mb:.0f} MB |")

            lines.append(f"| {self.t('success_rate')} | {best_metrics.success_rate * 100:.0f}% |")
            lines.append("")

        lines.extend([
            f"## {self.t('all_results')}",
            "",
            f"| {self.t('configuration')} | {self.t('score')} | TTFT (ms) | {self.t('throughput')} (tok/s) | VRAM (MB) |",
            "|--------------|-------|-----------|-------------------|-----------|",
        ])

        sorted_results = sorted(self.result.all_results, key=lambda r: r.score, reverse=True)

        for result in sorted_results[:10]:
            config_name = f"{result.candidate.cache_strategy.value}/{result.candidate.attention_backend.value}"
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

        if self.result.system_info:
            lines.extend([f"## {self.t('system_information')}", ""])
            if "gpu" in self.result.system_info:
                gpu = self.result.system_info["gpu"]
                lines.extend([
                    f"- **{self.t('gpu')}:** {gpu.get('name', 'Unknown')}",
                    f"- **{self.t('gpu_memory')}:** {gpu.get('memory_mb', 0):.0f} MB",
                ])
            if "ram_total_mb" in self.result.system_info:
                lines.append(f"- **{self.t('ram')}:** {self.result.system_info['ram_total_mb']:.0f} MB")
            lines.extend([
                f"- **{self.t('platform')}:** {self.result.system_info.get('platform', 'Unknown')}",
                f"- **{self.t('python')}:** {self.result.system_info.get('python_version', 'Unknown')}",
                "",
            ])

        lines.extend([
            f"## {self.t('usage')}",
            "",
            self.t('apply_config'),
            "",
            "```bash",
            f"kvat apply --plan {self.result.model_id.replace('/', '_')}_plan.json",
            "```",
            "",
            self.t('copy_snippet'),
            "",
            "---",
            "",
            f"*{self.t('generated_by')} [KVCache Auto-Tuner](https://github.com/Keyvanhardani/kvcache-autotune)*",
            "",
            "**[Keyvan.ai](https://keyvan.ai)** | [LinkedIn](https://www.linkedin.com/in/keyvanhardani)",
            "",
            f"*{self.t('made_in_germany')}*" + rtl_end,
        ])

        return "\n".join(lines)

    def generate_html(self) -> str:
        """Generate HTML report with styling."""
        best_metrics = self._get_best_metrics()
        sorted_results = sorted(self.result.all_results, key=lambda r: r.score, reverse=True)

        results_rows = []
        for i, result in enumerate(sorted_results[:15]):
            config_name = f"{result.candidate.cache_strategy.value}/{result.candidate.attention_backend.value}"
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

        dir_attr = 'dir="rtl"' if self.is_rtl else ""
        lang_code = self.language

        html = f"""<!DOCTYPE html>
<html lang="{lang_code}" {dir_attr}>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.t('report_title')} | Keyvan.ai</title>
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

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

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
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .header-brand {{ display: flex; align-items: center; gap: 1rem; }}

        .header-logo {{
            width: 48px; height: 48px;
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem; font-weight: 700;
        }}

        .header-title {{ font-size: 1.5rem; font-weight: 600; }}
        .header-subtitle {{ font-size: 0.875rem; opacity: 0.8; }}

        .header-links {{ display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }}

        .header-links a {{
            color: white;
            text-decoration: none;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            background: rgba(255,255,255,0.1);
            transition: background 0.2s;
            display: flex; align-items: center; gap: 0.4rem;
            font-size: 0.8rem;
        }}

        .header-links a:hover {{ background: rgba(255,255,255,0.2); }}
        .header-links a.active {{ background: rgba(255,255,255,0.3); font-weight: 600; }}
        .header-links svg {{ width: 16px; height: 16px; fill: currentColor; }}
        .header-links .sep {{ border-left: 1px solid rgba(255,255,255,0.3); height: 20px; margin: 0 0.3rem; }}

        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        h1 {{ font-size: 1.75rem; margin-bottom: 0.5rem; color: var(--text); }}
        .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .card .value {{ font-size: 1.5rem; font-weight: 600; }}
        .card .value.highlight {{ color: var(--primary); }}
        .card .unit {{ font-size: 0.75rem; color: var(--text-muted); }}

        .best-config {{
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
        }}

        .best-config h3 {{ color: rgba(255,255,255,0.8); }}
        .best-config .value {{ color: white; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}

        th {{
            background: #f1f5f9;
            font-weight: 600;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }}

        tr:last-child td {{ border-bottom: none; }}
        .best-row {{ background: #f0fdf4; }}
        .best-row td {{ font-weight: 500; }}
        .score {{ font-weight: 600; color: var(--primary); }}
        .section-title {{ font-size: 1.25rem; margin: 2rem 0 1rem; }}

        .footer {{
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.875rem;
            text-align: center;
        }}

        @media (max-width: 768px) {{
            .header {{ flex-direction: column; text-align: center; }}
            .header-links {{ justify-content: center; }}
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
            <a href="?lang=en" {"class='active'" if self.language == "en" else ""}>EN</a>
            <a href="?lang=de" {"class='active'" if self.language == "de" else ""}>DE</a>
            <a href="?lang=fr" {"class='active'" if self.language == "fr" else ""}>FR</a>
            <a href="?lang=es" {"class='active'" if self.language == "es" else ""}>ES</a>
            <a href="?lang=fa" {"class='active'" if self.language == "fa" else ""}>FA</a>
            <a href="?lang=ar" {"class='active'" if self.language == "ar" else ""}>AR</a>
            <span class="sep"></span>
            <a href="https://keyvan.ai" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/></svg>
                Keyvan.ai
            </a>
            <a href="https://github.com/Keyvanhardani/kvcache-autotune" target="_blank">
                <svg viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                GitHub
            </a>
        </div>
    </div>

    <div class="container">
        <h1>{self.t('performance_report')}</h1>
        <p class="subtitle">
            {self.t('model')}: <strong>{self.result.model_id}</strong> |
            {self.t('profile')}: <strong>{self.result.profile.name}</strong> |
            {self.t('generated')}: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
        </p>

        <div class="grid">
            <div class="card best-config">
                <h3>{self.t('best_configuration')}</h3>
                <div class="value">{self.result.best_config.cache_strategy.value}</div>
                <div class="unit">{self.result.best_config.attention_backend.value} / {self.result.best_config.dtype.value}</div>
            </div>

            <div class="card">
                <h3>{self.t('score')}</h3>
                <div class="value highlight">{self.result.best_score:.2f}</div>
                <div class="unit">{self.t('out_of_100')}</div>
            </div>

            <div class="card">
                <h3>{self.t('confidence')}</h3>
                <div class="value">{self.result.confidence * 100:.0f}%</div>
                <div class="unit">{self.t('recommendation_confidence')}</div>
            </div>

            <div class="card">
                <h3>TTFT</h3>
                <div class="value">{best_metrics.ttft_mean_ms if best_metrics else 0:.1f}</div>
                <div class="unit">{self.t('milliseconds_mean')}</div>
            </div>

            <div class="card">
                <h3>{self.t('throughput')}</h3>
                <div class="value">{best_metrics.throughput_mean if best_metrics else 0:.1f}</div>
                <div class="unit">{self.t('tokens_second')}</div>
            </div>

            <div class="card">
                <h3>{self.t('peak_vram')}</h3>
                <div class="value">{best_metrics.peak_vram_mb if best_metrics and best_metrics.peak_vram_mb else 0:.0f}</div>
                <div class="unit">{self.t('megabytes')}</div>
            </div>
        </div>

        <h2 class="section-title">{self.t('all_results')}</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>{self.t('configuration')}</th>
                    <th>{self.t('data_type')}</th>
                    <th>{self.t('score')}</th>
                    <th>TTFT (ms)</th>
                    <th>{self.t('throughput')}</th>
                    <th>VRAM (MB)</th>
                    <th>{self.t('success')}</th>
                </tr>
            </thead>
            <tbody>
                {''.join(results_rows)}
            </tbody>
        </table>

        <div class="footer">
            <p>{self.t('tuning_completed')} {self.result.tuning_duration_seconds:.1f} {self.t('seconds')} | {len(self.result.all_results)} {self.t('configurations_tested')}</p>
            <p style="margin-top: 1rem; font-style: italic;">{self.t('made_in_germany')}</p>
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
        all_languages: bool = False,
    ) -> dict[str, Path]:
        """
        Save reports to files.

        Args:
            output_dir: Output directory
            markdown: Generate Markdown report
            html: Generate HTML report
            all_languages: Generate reports in all supported languages

        Returns:
            Dictionary of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = {}

        if all_languages:
            languages: list[Language] = ["en", "de", "fr", "es", "fa", "ar"]
        else:
            languages = [self.language]

        for lang in languages:
            self.language = lang
            self.is_rtl = lang in RTL_LANGUAGES

            suffix = f"_{lang}" if all_languages else ""

            if markdown:
                md_path = output_dir / f"report{suffix}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(self.generate_markdown())
                saved[f"markdown_{lang}"] = md_path

            if html:
                html_path = output_dir / f"report{suffix}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(self.generate_html())
                saved[f"html_{lang}"] = html_path

        return saved
