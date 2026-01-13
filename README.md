# KVCache Auto-Tuner

<p align="center">
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/actions"><img src="https://github.com/Keyvanhardani/kvcache-autotune/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/v/kvat.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/kvat"><img src="https://img.shields.io/npm/v/kvat.svg" alt="npm"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/pyversions/kvat.svg" alt="Python"></a>
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>English</strong> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Francais</a> | <a href="README_ES.md">Espanol</a> | <a href="README_FA.md">فارسی</a> | <a href="README_AR.md">العربية</a>
</p>

---

## Why kvat?

When you run LLMs with HuggingFace Transformers, there are **dozens of configuration options** that affect performance:

| Setting | Options | What it affects |
|---------|---------|-----------------|
| Cache Strategy | dynamic, static, sliding_window | Memory usage, prefill speed |
| Attention Backend | sdpa_flash, eager, math, mem_efficient | Throughput, VRAM |
| Data Type | bfloat16, float16, float32 | Speed vs precision |

**The problem:** The optimal combination depends on YOUR specific model + YOUR GPU + YOUR use case. Nobody knows which config is best without testing.

**The solution:** `kvat` automatically benchmarks all combinations and tells you the fastest configuration.

```bash
# Before: Guessing and manual testing
model = AutoModelForCausalLM.from_pretrained("gpt2")  # Default config - slow

# After: Let kvat find the best config in 2 minutes
pip install kvat[full]
kvat tune gpt2 --profile ci-micro
# Output: "Best: dynamic/sdpa_flash/bfloat16 = 120 tok/s (+2.7% faster)"
```

---

## Installation

```bash
pip install kvat[full]
```

---

## Quick Start

```bash
# Tune any HuggingFace model
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# Quick test (recommended for first try)
kvat tune gpt2 --profile ci-micro

# Show your system info
kvat info
```

---

## Benchmark Results

<p align="center">
  <img src="assets/server_throughput.png" alt="Server Throughput" width="800">
</p>

### Server (RTX 4000 SFF Ada - 20GB VRAM)

| Model | Throughput | TTFT | Best Config |
|-------|------------|------|-------------|
| GPT-2 (124M) | **407.1 tok/s** | 4.0ms | dynamic/sdpa_flash |
| Qwen2.5-0.5B | **140.7 tok/s** | 10.9ms | dynamic/sdpa_flash |
| TinyLlama-1.1B | **93.0 tok/s** | 30.6ms | static/eager |
| Phi-1.5 (1.3B) | **78.8 tok/s** | 37.2ms | static/eager |

<p align="center">
  <img src="assets/server_dashboard.png" alt="Server Dashboard" width="800">
</p>

### Desktop (RTX 4060 - 8GB VRAM)

| Model | Baseline | With kvat | Improvement |
|-------|----------|-----------|-------------|
| GPT-2 (124M) | 118.1 tok/s | 120.2 tok/s | **+1.8%** |
| Qwen2.5-0.5B | 28.7 tok/s | 29.5 tok/s | **+2.7%** |
| Phi-1.5 (1.3B) | 45.2 tok/s | 45.6 tok/s | **+0.9%** |

<details>
<summary><strong>Desktop Benchmark Charts</strong></summary>

<p align="center">
  <img src="assets/baseline_vs_optimized_hero.png" alt="Baseline vs Optimized" width="800">
</p>

<table>
<tr>
<td width="50%">
<img src="assets/baseline_vs_optimized_throughput.png" alt="Throughput Comparison" width="100%">
<p align="center"><em>Throughput (tokens/second)</em></p>
</td>
<td width="50%">
<img src="assets/baseline_vs_optimized_improvement.png" alt="Improvement %" width="100%">
<p align="center"><em>Performance Gain %</em></p>
</td>
</tr>
</table>

</details>

---

## Profiles

| Profile | Context Length | Output Length | Best For |
|---------|---------------|---------------|----------|
| `ci-micro` | 512 | 32 | Quick testing |
| `chat-agent` | 2-8K | 64-256 | Chatbots, low latency |
| `rag` | 8-32K | 256-512 | RAG pipelines |
| `longform` | 4-8K | 1-2K | Long text generation |

---

## Output

After tuning, kvat generates:

```
results/
├── best_plan.json      # Full config as JSON
├── optimized_config.py # Ready-to-use Python code
├── report.md           # Human-readable report
└── report.html         # Visual report with charts
```

**Example optimized_config.py:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
# Cache strategy: dynamic (default in Transformers 4.35+)
# Measured: 120.2 tok/s, TTFT: 9.1ms
```

---

## Python API

```python
from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import get_profile
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch

config = TuneConfig(
    model_id="meta-llama/Llama-3.2-1B",
    device=DeviceType.CUDA,
    profile=get_profile("chat-agent"),
    output_dir="./results",
)

adapter = TransformersAdapter()
search = TuningSearch(config=config, adapter=adapter)
result = search.run()

print(f"Best config: {result.best_config}")
print(f"Throughput: {result.best_score} tok/s")
```

---

## npm Package (JavaScript/TypeScript)

```bash
npm install kvat
```

```javascript
const kvat = require('kvat');

// Run tuning
const result = await kvat.tune('gpt2', {
  profile: 'ci-micro',
  outputDir: './results'
});
```

---

## Roadmap

### v0.1.3 - Current
- [x] Auto context length limiting (fixes CUDA errors)
- [x] PyPI + npm + GitHub Packages
- [x] Baseline vs Optimized benchmarking
- [x] Multi-language READMEs (EN, DE, FR, ES, FA, AR)
- [x] Multi-language report generation (6 languages)
- [x] Server benchmarks (RTX 4000 SFF Ada)
- [x] Improved report branding

### v0.2.0 - Next
- [ ] Ollama adapter
- [ ] llama.cpp adapter (GGUF models)
- [ ] Batch size optimization

### v0.3.0 - Planned
- [ ] vLLM adapter
- [ ] Quantized KV-cache (INT8/INT4)

---

## Contributing

```bash
git clone https://github.com/Keyvanhardani/kvcache-autotune.git
cd kvcache-autotune
pip install -e ".[full,dev]"
pytest tests/ -v
```

---

## License

Apache 2.0

## Citation

```bibtex
@software{kvat,
  title = {KVCache Auto-Tuner: Automatic KV-Cache Optimization for Transformers},
  author = {Keyvanhardani},
  year = {2026},
  url = {https://github.com/Keyvanhardani/kvcache-autotune}
}
```

---

<p align="center">
  <a href="https://keyvan.ai"><strong>Keyvan.ai</strong></a> | <a href="https://www.linkedin.com/in/keyvanhardani">LinkedIn</a>
</p>
<p align="center">
  Made in Germany with dedication for the HuggingFace Community
</p>
