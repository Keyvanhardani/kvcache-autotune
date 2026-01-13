# KVCache Auto-Tuner

<p align="center">
  <strong>Automatic KV-Cache Optimization for HuggingFace Transformers</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

**KVCache Auto-Tuner** automatically finds the optimal KV-cache configuration for your model, hardware, and workload. Stop guessing which cache strategy, attention backend, or dtype works best—let the tuner find it for you.

## Why KVCache Auto-Tuner?

Modern LLM inference involves many configuration choices:

- **Cache Strategy**: Dynamic vs Static vs Sliding Window
- **Attention Backend**: SDPA, Flash Attention, xFormers
- **Data Types**: fp16, bf16, fp32
- **Compilation**: torch.compile modes

The optimal combination depends on your specific model, hardware, and use case. KVCache Auto-Tuner benchmarks these combinations systematically and provides a production-ready configuration.

## Features

- **Automatic Optimization**: Find the best configuration without manual experimentation
- **Multiple Profiles**: Built-in presets for Chat/Agent, RAG, and Longform workloads
- **Custom Workloads**: Define your own profiles with specific context/output lengths
- **Production-Ready Output**: Get drop-in Python code snippets and JSON plans
- **Beautiful Reports**: Markdown and HTML reports with performance comparisons
- **Early Stopping**: Smart pruning of dominated configurations for faster results
- **Extensible Architecture**: Adapter-based design for future vLLM/llama.cpp support

## Installation

```bash
# Basic installation (CLI + core)
pip install kvcache-autotune

# With Transformers support (recommended)
pip install kvcache-autotune[transformers]

# Full installation with all optional dependencies
pip install kvcache-autotune[full]
```

### From Source

```bash
git clone https://github.com/your-org/kvcache-autotune.git
cd kvcache-autotune
pip install -e ".[full,dev]"
```

## Quick Start

### CLI Usage

```bash
# Basic tuning with chat-agent profile
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# RAG workload with custom context lengths
kvat tune mistralai/Mistral-7B-v0.1 --profile rag --context 8192,16384,32768

# Apply a saved plan
kvat apply ./kvat_results/best_plan.json --print-snippet

# Compare two configurations
kvat compare baseline_plan.json new_plan.json

# List available profiles
kvat profiles
```

### Python API

```python
from kvat.core.schema import TuneConfig, DeviceType
from kvat.core.profiles import get_profile
from kvat.engines.transformers import TransformersAdapter
from kvat.core.search import TuningSearch
from kvat.core.planner import PlanBuilder

# Configure tuning
config = TuneConfig(
    model_id="meta-llama/Llama-3.2-1B",
    device=DeviceType.CUDA,
    profile=get_profile("chat-agent"),
    output_dir="./results",
)

# Run optimization
adapter = TransformersAdapter()
search = TuningSearch(config=config, adapter=adapter)
result = search.run()

# Get production-ready code
planner = PlanBuilder(result)
print(planner.generate_code_snippet())
```

## Profiles

KVCache Auto-Tuner includes optimized profiles for common workloads:

| Profile | Context | Output | Optimization Focus |
|---------|---------|--------|-------------------|
| `chat-agent` | 2-8K | 64-256 | Minimize TTFT (50%) |
| `rag` | 8-32K | 256-512 | Balance all metrics (35/35/30) |
| `longform` | 4-8K | 1-2K | Maximize throughput (50%) |
| `ci-micro` | 512 | 32 | Quick CI validation |

### Custom Profiles

```python
from kvat.core.profiles import create_custom_profile

profile = create_custom_profile(
    name="my-workload",
    context_lengths=[4096, 8192],
    output_lengths=[256, 512],
    weight_ttft=0.4,
    weight_throughput=0.4,
    weight_memory=0.2,
)
```

## Output

KVCache Auto-Tuner generates:

1. **JSON Plan** (`best_plan.json`): Complete configuration with metrics and fallback rules
2. **Code Snippet** (`optimized_config.py`): Drop-in Python code for your inference pipeline
3. **Markdown Report** (`report.md`): Human-readable summary with rankings
4. **HTML Report** (`report.html`): Visual report with charts and styling

### Example Output

```
Best Configuration:
  Cache Strategy: dynamic
  Attention Backend: sdpa_flash
  Data Type: float16
  Score: 87.32
  Confidence: 94%

Performance:
  TTFT: 45.2ms (mean), 3.1ms (std)
  Throughput: 78.5 tok/s
  Peak VRAM: 4,521 MB
```

## Architecture

```
kvat/
├── core/
│   ├── schema.py      # Pydantic data models
│   ├── metrics.py     # TTFT, throughput, scoring
│   ├── profiles.py    # Workload profiles
│   ├── search.py      # Grid search with pruning
│   ├── planner.py     # Plan generation
│   └── report.py      # Markdown/HTML reports
├── engines/
│   ├── base.py        # EngineAdapter interface
│   └── transformers.py # HuggingFace adapter
├── probes/
│   ├── gpu.py         # CUDA memory tracking
│   └── cpu.py         # RAM monitoring
└── cli.py             # Typer CLI
```

## Roadmap

### P0 (Current)
- [x] Core tuning engine
- [x] Transformers adapter
- [x] CLI interface
- [x] Markdown/HTML reports

### P1 (Next)
- [ ] Batch size sweeps
- [ ] CPU offload strategies
- [ ] CI micro-benchmark suite
- [ ] `kvat watch` for continuous monitoring

### P2 (Future)
- [ ] vLLM adapter
- [ ] llama.cpp adapter
- [ ] Quantized KV-cache support
- [ ] Multi-GPU configurations

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v

# Run linting
ruff check kvat/
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Citation

If you use KVCache Auto-Tuner in your research, please cite:

```bibtex
@software{kvcache_autotune,
  title = {KVCache Auto-Tuner: Automatic KV-Cache Optimization for Transformers},
  year = {2024},
  url = {https://github.com/your-org/kvcache-autotune}
}
```

---

<p align="center">
  Made with dedication for the HuggingFace community
</p>
