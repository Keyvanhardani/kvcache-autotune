# KVCache Auto-Tuner

<p align="center">
  <strong>Automatische KV-Cache Optimierung für HuggingFace Transformers</strong>
</p>

<p align="center">
  <a href="#funktionen">Funktionen</a> •
  <a href="#installation">Installation</a> •
  <a href="#schnellstart">Schnellstart</a> •
  <a href="#dokumentation">Dokumentation</a> •
  <a href="#mitwirken">Mitwirken</a>
</p>

---

**KVCache Auto-Tuner** findet automatisch die optimale KV-Cache-Konfiguration für Ihr Modell, Ihre Hardware und Ihren Anwendungsfall. Schluss mit dem Raten, welche Cache-Strategie, welches Attention-Backend oder welcher Datentyp am besten funktioniert – lassen Sie den Tuner die Arbeit erledigen.

## Warum KVCache Auto-Tuner?

Moderne LLM-Inferenz erfordert viele Konfigurationsentscheidungen:

- **Cache-Strategie**: Dynamic vs Static vs Sliding Window
- **Attention-Backend**: SDPA, Flash Attention, xFormers
- **Datentypen**: fp16, bf16, fp32
- **Kompilierung**: torch.compile Modi

Die optimale Kombination hängt von Ihrem spezifischen Modell, Ihrer Hardware und Ihrem Anwendungsfall ab. KVCache Auto-Tuner benchmarkt diese Kombinationen systematisch und liefert eine produktionsreife Konfiguration.

## Funktionen

- **Automatische Optimierung**: Beste Konfiguration ohne manuelle Experimente finden
- **Mehrere Profile**: Eingebaute Voreinstellungen für Chat/Agent, RAG und Langtext-Workflows
- **Benutzerdefinierte Workloads**: Eigene Profile mit spezifischen Kontext-/Ausgabelängen definieren
- **Produktionsreife Ausgabe**: Direkt verwendbare Python-Code-Snippets und JSON-Pläne
- **Schöne Berichte**: Markdown- und HTML-Berichte mit Leistungsvergleichen
- **Early Stopping**: Intelligentes Pruning dominierter Konfigurationen für schnellere Ergebnisse
- **Erweiterbare Architektur**: Adapter-basiertes Design für zukünftige vLLM/llama.cpp-Unterstützung

## Installation

```bash
# Basis-Installation (CLI + Core)
pip install kvcache-autotune

# Mit Transformers-Unterstützung (empfohlen)
pip install kvcache-autotune[transformers]

# Vollständige Installation mit allen optionalen Abhängigkeiten
pip install kvcache-autotune[full]
```

### Aus dem Quellcode

```bash
git clone https://github.com/your-org/kvcache-autotune.git
cd kvcache-autotune
pip install -e ".[full,dev]"
```

## Schnellstart

### CLI-Nutzung

```bash
# Basis-Tuning mit Chat-Agent-Profil
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# RAG-Workload mit benutzerdefinierten Kontextlängen
kvat tune mistralai/Mistral-7B-v0.1 --profile rag --context 8192,16384,32768

# Gespeicherten Plan anwenden
kvat apply ./kvat_results/best_plan.json --print-snippet

# Zwei Konfigurationen vergleichen
kvat compare baseline_plan.json new_plan.json

# Verfügbare Profile auflisten
kvat profiles
```

### Python-API

```python
from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import get_profile
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch
from kvat.core.planner import PlanBuilder

# Tuning konfigurieren
config = TuneConfig(
    model_id="meta-llama/Llama-3.2-1B",
    device=DeviceType.CUDA,
    profile=get_profile("chat-agent"),
    output_dir="./results",
)

# Optimierung ausführen
adapter = TransformersAdapter()
search = TuningSearch(config=config, adapter=adapter)
result = search.run()

# Produktionsreifen Code abrufen
planner = PlanBuilder(result)
print(planner.generate_code_snippet())
```

## Profile

KVCache Auto-Tuner enthält optimierte Profile für gängige Workloads:

| Profil | Kontext | Ausgabe | Optimierungsfokus |
|--------|---------|---------|-------------------|
| `chat-agent` | 2-8K | 64-256 | TTFT minimieren (50%) |
| `rag` | 8-32K | 256-512 | Alle Metriken ausbalancieren (35/35/30) |
| `longform` | 4-8K | 1-2K | Durchsatz maximieren (50%) |
| `ci-micro` | 512 | 32 | Schnelle CI-Validierung |

### Benutzerdefinierte Profile

```python
from kvat.core.profiles import create_custom_profile

profile = create_custom_profile(
    name="mein-workload",
    context_lengths=[4096, 8192],
    output_lengths=[256, 512],
    weight_ttft=0.4,
    weight_throughput=0.4,
    weight_memory=0.2,
)
```

## Ausgabe

KVCache Auto-Tuner generiert:

1. **JSON-Plan** (`best_plan.json`): Vollständige Konfiguration mit Metriken und Fallback-Regeln
2. **Code-Snippet** (`optimized_config.py`): Direkt verwendbarer Python-Code für Ihre Inferenz-Pipeline
3. **Markdown-Bericht** (`report.md`): Lesbare Zusammenfassung mit Rankings
4. **HTML-Bericht** (`report.html`): Visueller Bericht mit Diagrammen und Styling

### Beispielausgabe

```
Beste Konfiguration:
  Cache-Strategie: dynamic
  Attention-Backend: sdpa_flash
  Datentyp: float16
  Score: 87.32
  Konfidenz: 94%

Leistung:
  TTFT: 45.2ms (Mittel), 3.1ms (Std)
  Durchsatz: 78.5 Tok/s
  Max. VRAM: 4.521 MB
```

## Architektur

```
kvat/
├── core/
│   ├── schema.py      # Pydantic-Datenmodelle
│   ├── metrics.py     # TTFT, Durchsatz, Scoring
│   ├── profiles.py    # Workload-Profile
│   ├── search.py      # Grid-Search mit Pruning
│   ├── planner.py     # Plan-Generierung
│   └── report.py      # Markdown/HTML-Berichte
├── engines/
│   ├── base.py        # EngineAdapter-Interface
│   └── transformers.py # HuggingFace-Adapter
├── probes/
│   ├── gpu.py         # CUDA-Speicher-Tracking
│   └── cpu.py         # RAM-Überwachung
└── cli.py             # Typer-CLI
```

## Roadmap

### P0 (Aktuell)
- [x] Core-Tuning-Engine
- [x] Transformers-Adapter
- [x] CLI-Interface
- [x] Markdown/HTML-Berichte

### P1 (Nächste Schritte)
- [ ] Batch-Size-Sweeps
- [ ] CPU-Offload-Strategien
- [ ] CI-Micro-Benchmark-Suite
- [ ] `kvat watch` für kontinuierliches Monitoring

### P2 (Zukunft)
- [ ] vLLM-Adapter
- [ ] llama.cpp-Adapter
- [ ] Quantisierter KV-Cache-Support
- [ ] Multi-GPU-Konfigurationen

## Mitwirken

Beiträge sind willkommen! Bitte lesen Sie unseren [Contribution Guide](CONTRIBUTING.md) für Details.

```bash
# Entwicklungsumgebung einrichten
pip install -e ".[dev]"
pre-commit install

# Tests ausführen
pytest tests/ -v

# Linting ausführen
ruff check kvat/
```

## Lizenz

Apache 2.0 - Siehe [LICENSE](LICENSE) für Details.

## Zitierung

Wenn Sie KVCache Auto-Tuner in Ihrer Forschung verwenden, zitieren Sie bitte:

```bibtex
@software{kvcache_autotune,
  title = {KVCache Auto-Tuner: Automatic KV-Cache Optimization for Transformers},
  year = {2024},
  url = {https://github.com/your-org/kvcache-autotune}
}
```

---

<p align="center">
  Mit Hingabe für die HuggingFace-Community entwickelt
</p>
