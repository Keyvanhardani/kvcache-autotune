# KVCache Auto-Tuner

<p align="center">
  <img src="assets/benchmark_hero.png" alt="KVCache Auto-Tuner" width="700">
</p>

<h3 align="center">
  Automatic KV-Cache Optimization for HuggingFace Transformers
</h3>

<p align="center">
  <em>Find the optimal cache strategy, attention backend, and configuration for your model and hardware.</em>
</p>

<p align="center">
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/actions"><img src="https://github.com/Keyvanhardani/kvcache-autotune/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/v/kvat.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/pyversions/kvat.svg" alt="Python"></a>
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> |
  <a href="#-performance">Performance</a> |
  <a href="#-features">Features</a> |
  <a href="#-installation">Installation</a> |
  <a href="#-roadmap">Roadmap</a>
</p>

---

## What is KVCache Auto-Tuner?

**KVCache Auto-Tuner** (`kvat`) automatically benchmarks and optimizes your HuggingFace Transformers inference pipeline. Stop guessing which configuration works best - let the tuner find it for you.

```bash
# Install and optimize your model in seconds
pip install kvat[full]
kvat tune gpt2 --profile chat-agent
```

---

## Performance

### Baseline vs Optimized

See how **kvat** improves your Transformers inference:

<p align="center">
  <img src="assets/baseline_vs_optimized_hero.png" alt="Performance Improvement with KVCache Auto-Tuner" width="800">
</p>

| Model | Without kvat | With kvat | Improvement |
|-------|--------------|-----------|-------------|
| **GPT-2** (124M) | 118.1 tok/s | 120.2 tok/s | **+1.8%** |
| **Qwen2.5-0.5B** | 28.7 tok/s | 29.5 tok/s | **+2.7%** |
| **Phi-1.5** (1.3B) | 45.2 tok/s | 45.6 tok/s | **+0.9%** |

<details>
<summary><strong>View Detailed Comparison Charts</strong></summary>

<table>
<tr>
<td width="50%">
<img src="assets/baseline_vs_optimized_throughput.png" alt="Throughput: Baseline vs Optimized" width="100%">
<p align="center"><em>Throughput Comparison</em></p>
</td>
<td width="50%">
<img src="assets/baseline_vs_optimized_improvement.png" alt="Performance Improvement %" width="100%">
<p align="center"><em>Performance Gain</em></p>
</td>
</tr>
</table>

</details>

> **Note**: Results vary by model and hardware. Larger improvements are typical for models that benefit from Flash Attention and dynamic caching.

### Multi-Model Benchmarks

**Desktop (RTX 4060 - 8GB VRAM):**

| Model | TTFT | Throughput | VRAM | Best Config |
|-------|------|------------|------|-------------|
| GPT-2 | 9.1ms | 124.6 tok/s | 283MB | dynamic/sdpa_flash |
| Phi-1.5 | 40.9ms | 52.8 tok/s | 2.8GB | dynamic/sdpa_flash |
| Qwen2.5-0.5B | 33.9ms | 33.6 tok/s | 975MB | dynamic/eager |

**Server (RTX 4000 Ada - 20GB VRAM):**

| Model | TTFT | Throughput | VRAM | Best Config |
|-------|------|------------|------|-------------|
| GPT-2 | 4.2ms | **365.4 tok/s** | 264MB | dynamic/sdpa_flash |
| Qwen2.5-7B | 284ms | 3.3 tok/s | 13.6GB | dynamic/sdpa_flash |

> Server throughput is **3x faster** than desktop for the same model!

<details>
<summary><strong>View Multi-Model Charts</strong></summary>

<p align="center">
  <img src="assets/comparison_hero.png" alt="Multi-Model Performance Overview" width="800">
</p>

<table>
<tr>
<td width="50%">
<img src="assets/comparison_ttft.png" alt="Time to First Token by Model" width="100%">
<p align="center"><em>TTFT Comparison (lower is better)</em></p>
</td>
<td width="50%">
<img src="assets/comparison_throughput.png" alt="Throughput by Model" width="100%">
<p align="center"><em>Throughput Comparison (higher is better)</em></p>
</td>
</tr>
</table>

</details>

---

## Quick Start

### CLI Usage

```bash
# Optimize any HuggingFace model
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# Quick test
kvat tune gpt2 --profile ci-micro -v

# Show system info
kvat info
```

### Python API

```python
from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import get_profile
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch

# Configure and run optimization
config = TuneConfig(
    model_id="meta-llama/Llama-3.2-1B",
    device=DeviceType.CUDA,
    profile=get_profile("chat-agent"),
    output_dir="./results",
)

adapter = TransformersAdapter()
search = TuningSearch(config=config, adapter=adapter)
result = search.run()
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Automatic Optimization** | Find the best configuration without manual experimentation |
| **Multiple Profiles** | Built-in presets for Chat, RAG, and Longform workloads |
| **Production-Ready Output** | Get drop-in Python code snippets and JSON configs |
| **Beautiful Reports** | Markdown and HTML reports with performance comparisons |
| **Early Stopping** | Smart pruning of dominated configurations |
| **Extensible** | Adapter-based design for vLLM/llama.cpp/Ollama |

### Optimization Parameters

| Parameter | Options | Impact |
|-----------|---------|--------|
| **Cache Strategy** | Dynamic, Static, Sliding Window | Memory & prefill speed |
| **Attention Backend** | SDPA Flash, Memory Efficient, Math, Eager | Throughput & VRAM |
| **Data Type** | bfloat16, float16, float32 | Speed vs precision |
| **Compilation** | torch.compile modes | Startup vs runtime |

### Built-in Profiles

| Profile | Context | Output | Focus |
|---------|---------|--------|-------|
| `chat-agent` | 2-8K | 64-256 | TTFT (latency) |
| `rag` | 8-32K | 256-512 | Balanced |
| `longform` | 4-8K | 1-2K | Throughput |
| `ci-micro` | 512 | 32 | Quick testing |

---

## Installation

```bash
# Recommended: Full installation with all dependencies
pip install kvat[full]

# Basic installation
pip install kvat

# From source
git clone https://github.com/Keyvanhardani/kvcache-autotune.git
cd kvcache-autotune
pip install -e ".[full,dev]"
```

**Requirements**: Python 3.9+, PyTorch 2.0+, Transformers 4.35+

---

## Output Files

| File | Description |
|------|-------------|
| `best_plan.json` | Complete configuration with metrics |
| `optimized_config.py` | Drop-in Python code |
| `report.md` | Human-readable summary |
| `report.html` | Visual report with charts |

### Example Output

```
+-----------------------------------------------------------------------------+
| Best Configuration                                                          |
|                                                                             |
| Cache Strategy: dynamic                                                     |
| Attention Backend: sdpa_flash                                               |
| Data Type: bfloat16                                                         |
| Score: 100.00                                                               |
+-----------------------------------------------------------------------------+
```

---

## Roadmap

### v0.1.0 (Current)
- [x] Core tuning engine with grid search
- [x] HuggingFace Transformers adapter
- [x] CLI interface (`kvat tune`, `kvat apply`, `kvat compare`)
- [x] Built-in profiles (chat-agent, rag, longform)
- [x] CUDA/GPU memory tracking
- [x] Windows & Linux support

### v0.2.0 (Next)
- [ ] Batch size optimization
- [ ] CPU offload strategies
- [ ] `kvat watch` - Continuous monitoring
- [ ] Profile recommendations based on hardware

### v0.3.0 (Planned)
- [ ] **Ollama adapter** - Local model optimization
- [ ] **llama.cpp adapter** - GGUF model support
- [ ] **vLLM adapter** - Production serving
- [ ] Quantized KV-cache (INT8/INT4)

### v1.0.0 (Future)
- [ ] HuggingFace Hub integration
- [ ] Real-time inference monitoring
- [ ] A/B testing framework

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check kvat/
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{kvat,
  title = {KVCache Auto-Tuner: Automatic KV-Cache Optimization for Transformers},
  author = {Keyvanhardani},
  year = {2025},
  url = {https://github.com/Keyvanhardani/kvcache-autotune}
}
```

---

<p align="center">
  <a href="https://keyvan.ai"><strong>Keyvan.ai</strong></a> | <a href="https://linkedin.com/in/keyvanhardani">LinkedIn</a>
</p>
<p align="center">
  Made from Germany with dedication for the HuggingFace community
</p>
