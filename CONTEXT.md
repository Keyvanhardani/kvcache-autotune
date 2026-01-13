# KVCache Auto-Tuner - Projekt-Kontext

> **Wichtig**: Diese Datei dient der Session-Kontinuität. Bitte nach `/compact` lesen!

## Projekt-Status: P0 MVP - FUNKTIONSFÄHIG

Das Projekt ist vollständig implementiert und getestet.

### Letzte Session (2026-01-13 14:20 UTC)

- ✅ Vollständige Implementierung aller Module
- ✅ CLI funktioniert (`kvat --version`, `kvat info`, `kvat profiles`)
- ✅ 40/40 Unit-Tests bestanden
- ✅ PyPI Build erstellt (`dist/kvcache_autotune-0.1.0-py3-none-any.whl`)
- ✅ GitHub Actions Workflows erstellt (`.github/workflows/`)
- ✅ Git Repository initialisiert, erster Commit erstellt
- ✅ **Benchmark erfolgreich!** Metriken werden korrekt erfasst

### Benchmark-Ergebnis (GPT-2 auf RTX 4060)

```
Best Configuration: dynamic/sdpa_flash + bfloat16
TTFT: ~10ms
Throughput: ~123 tok/s
VRAM: 283 MB
Score: 100/100
```

### Behobene Probleme

1. **Triton/cache_implementation auf Windows**:
   - Problem: `cache_implementation` benötigt Triton, nicht auf Windows verfügbar
   - Lösung: Platform-Check hinzugefügt, auf Windows wird `use_cache=True` verwendet
   - Datei: `kvat/engines/transformers.py:269`

2. **Metriken zeigten 0.0**:
   - Problem: `mark_first_token()` wurde nie aufgerufen bei Batch-Generation
   - Lösung: Separate TTFT-Messung via `run_prefill()`, Throughput via `run_decode()`
   - Datei: `kvat/core/search.py:351-384`

3. **`__main__.py` fehlte**:
   - Problem: `python -m kvat` funktionierte nicht
   - Lösung: `kvat/__main__.py` erstellt

## Projekt-Übersicht

**Name**: `kvcache-autotune` (CLI: `kvat`)
**Ziel**: Automatische Optimierung der KV-Cache-Konfiguration für HuggingFace Transformers

### Was wurde implementiert

| Modul | Status | Beschreibung |
|-------|--------|--------------|
| `kvat/core/schema.py` | ✅ | Pydantic-Schemas für alle Datenmodelle |
| `kvat/core/metrics.py` | ✅ | TTFT, Throughput, Scoring-Logik |
| `kvat/core/profiles.py` | ✅ | Built-in Profile (chat-agent, rag, longform) |
| `kvat/core/search.py` | ✅ | Grid-Search mit TTFT/Throughput-Messung |
| `kvat/core/planner.py` | ✅ | Plan-Builder mit Code-Snippet-Generator |
| `kvat/core/report.py` | ✅ | Markdown + HTML Report-Generator |
| `kvat/engines/base.py` | ✅ | EngineAdapter Interface |
| `kvat/engines/transformers.py` | ✅ | TransformersAdapter Implementation |
| `kvat/probes/gpu.py` | ✅ | CUDA/NVML Memory Probes |
| `kvat/probes/cpu.py` | ✅ | RAM/Process Memory Probes |
| `kvat/cli.py` | ✅ | Typer CLI (tune, apply, compare, profiles) |
| `kvat/__main__.py` | ✅ | Entry-Point für `python -m kvat` |
| `tests/` | ✅ | Unit-Tests für Schema, Profiles, Metrics |
| `examples/` | ✅ | Nutzungsbeispiele |
| `pyproject.toml` | ✅ | Packaging-Konfiguration |
| `README.md` | ✅ | Englische Dokumentation |
| `README_DE.md` | ✅ | Deutsche Dokumentation |

## Wichtige Code-Stellen

### TransformersAdapter - Windows-Kompatibilität
```python
# kvat/engines/transformers.py:269
import platform
self._use_cache_implementation = platform.system() == "Linux"
```

### Benchmark - TTFT + Throughput Messung
```python
# kvat/core/search.py:351-384
# Measure TTFT using prefill
_, ttft_ms = self.adapter.run_prefill(prompt, max_new_tokens=1)

# Measure decode throughput with full generation
start_time = time.perf_counter()
output = self.adapter.run_decode(prompt, max_new_tokens=output_length)
total_time_ms = (time.perf_counter() - start_time) * 1000

# Calculate throughput (tokens/second)
throughput = (tokens_generated / decode_time_ms) * 1000
```

## Nächste Schritte

### 1. ✅ Completed - GPT-2 Benchmark

```bash
kvat tune gpt2 --profile ci-micro -o benchmark_v3 -v
# Ergebnis: 100/100 Score, ~10ms TTFT, ~123 tok/s
```

### 2. Noch zu tun - Größerer Benchmark (Optional)

```bash
# Chat-Agent Profil für realistischeren Test
kvat tune gpt2 --profile chat-agent -o benchmark_chat -v
```

### 3. Noch zu tun - GitHub Push

```bash
git remote add origin https://github.com/Keyvanhardani/kvcache-autotune.git
git push -u origin main
```

### 4. Noch zu tun - PyPI Release

Nach GitHub-Push wird automatisch via `publish.yml` auf PyPI veröffentlicht.

## P2 Roadmap

1. **Ollama Adapter**: Für lokale Ollama-Modelle
2. **llama.cpp Adapter**: Für GGUF-Modelle
3. **vLLM Adapter**: Für produktive Serving-Szenarien

## Architektur-Entscheidungen

1. **Adapter-Pattern**: `EngineAdapter` Interface ermöglicht spätere Erweiterung
2. **Pydantic v2**: Moderne Validierung mit `field_validator`
3. **Typer + Rich**: Moderne CLI mit schöner Ausgabe
4. **Separate TTFT-Messung**: `run_prefill()` für genaue TTFT-Werte
5. **Profile-basiertes Scoring**: Gewichtete Metriken je nach Workload

## Scoring-Gewichte

| Profil | TTFT | Throughput | Memory |
|--------|------|------------|--------|
| chat-agent | 50% | 35% | 15% |
| rag | 35% | 35% | 30% |
| longform | 25% | 50% | 25% |

## System-Info (lokale Testumgebung)

- **GPU**: NVIDIA GeForce RTX 4060 (8 GB)
- **RAM**: 64 GB
- **OS**: Windows 11
- **Python**: 3.12.10
- **Transformers**: 4.52.4+

---

**Letztes Update**: 2026-01-13 14:20 UTC
**Status**: Bereit für GitHub Push und PyPI Release
