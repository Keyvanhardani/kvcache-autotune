# KVCache Auto-Tuner

<p align="center">
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/actions"><img src="https://github.com/Keyvanhardani/kvcache-autotune/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/v/kvat.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/kvat"><img src="https://img.shields.io/npm/v/kvat.svg" alt="npm"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/pyversions/kvat.svg" alt="Python"></a>
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Licencia"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Francais</a> | <strong>Espanol</strong> | <a href="README_FA.md">فارسی</a> | <a href="README_AR.md">العربية</a>
</p>

---

## Por que kvat?

Cuando ejecutas LLMs con HuggingFace Transformers, hay **docenas de opciones de configuracion** que afectan el rendimiento:

| Configuracion | Opciones | Impacto |
|---------------|----------|---------|
| Estrategia de cache | dynamic, static, sliding_window | Memoria, velocidad prefill |
| Backend Attention | sdpa_flash, eager, math, mem_efficient | Rendimiento, VRAM |
| Tipo de datos | bfloat16, float16, float32 | Velocidad vs precision |

**El problema:** La combinacion optima depende de TU modelo + TU GPU + TU caso de uso. Nadie sabe cual configuracion es la mejor sin probar.

**La solucion:** `kvat` benchmarkea automaticamente todas las combinaciones y te dice la configuracion mas rapida.

```bash
# Antes: Adivinar y probar manualmente
model = AutoModelForCausalLM.from_pretrained("gpt2")  # Config por defecto - lento

# Despues: Deja que kvat encuentre la mejor config en 2 minutos
pip install kvat[full]
kvat tune gpt2 --profile ci-micro
# Salida: "Best: dynamic/sdpa_flash/bfloat16 = 120 tok/s (+2.7% mas rapido)"
```

---

## Instalacion

```bash
pip install kvat[full]
```

---

## Inicio rapido

```bash
# Optimizar cualquier modelo HuggingFace
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# Prueba rapida (recomendado para el primer intento)
kvat tune gpt2 --profile ci-micro

# Mostrar informacion del sistema
kvat info
```

---

## Resultados de benchmark

### Desktop (RTX 4060 - 8GB VRAM)

| Modelo | Baseline | Con kvat | Mejora |
|--------|----------|----------|--------|
| GPT-2 (124M) | 118.1 tok/s | 120.2 tok/s | **+1.8%** |
| Qwen2.5-0.5B | 28.7 tok/s | 29.5 tok/s | **+2.7%** |
| Phi-1.5 (1.3B) | 45.2 tok/s | 45.6 tok/s | **+0.9%** |

---

## Perfiles

| Perfil | Longitud contexto | Longitud salida | Ideal para |
|--------|-------------------|-----------------|------------|
| `ci-micro` | 512 | 32 | Pruebas rapidas |
| `chat-agent` | 2-8K | 64-256 | Chatbots, baja latencia |
| `rag` | 8-32K | 256-512 | Pipelines RAG |
| `longform` | 4-8K | 1-2K | Generacion de texto largo |

---

## Salida

Despues de la optimizacion, kvat genera:

```
results/
├── best_plan.json      # Config completa en JSON
├── optimized_config.py # Codigo Python listo para usar
├── report.md           # Informe legible
└── report.html         # Informe visual con graficos
```

---

## Licencia

Apache 2.0

## Citacion

```bibtex
@software{kvat,
  title = {KVCache Auto-Tuner: Optimizacion automatica de KV-Cache para Transformers},
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
  Hecho en Alemania con dedicacion para la comunidad HuggingFace
</p>
