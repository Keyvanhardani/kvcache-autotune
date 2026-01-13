<div dir="rtl">

# KVCache Auto-Tuner

<p align="center">
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/actions"><img src="https://github.com/Keyvanhardani/kvcache-autotune/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/v/kvat.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/kvat"><img src="https://img.shields.io/npm/v/kvat.svg" alt="npm"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/pyversions/kvat.svg" alt="Python"></a>
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="مجوز"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Francais</a> | <a href="README_ES.md">Espanol</a> | <strong>فارسی</strong> | <a href="README_AR.md">العربية</a>
</p>

---

## چرا kvat؟

وقتی LLM ها را با HuggingFace Transformers اجرا می‌کنید، **ده‌ها گزینه پیکربندی** وجود دارد که بر عملکرد تأثیر می‌گذارد:

| تنظیمات | گزینه‌ها | تأثیر |
|---------|----------|-------|
| استراتژی کش | dynamic, static, sliding_window | حافظه، سرعت prefill |
| Backend Attention | sdpa_flash, eager, math, mem_efficient | توان عملیاتی، VRAM |
| نوع داده | bfloat16, float16, float32 | سرعت در مقابل دقت |

**مشکل:** ترکیب بهینه به مدل شما + GPU شما + مورد استفاده شما بستگی دارد. هیچ‌کس نمی‌داند کدام پیکربندی بهترین است بدون آزمایش.

**راه‌حل:** `kvat` به طور خودکار همه ترکیب‌ها را بنچمارک می‌کند و سریع‌ترین پیکربندی را به شما می‌گوید.

</div>

```bash
# قبل: حدس زدن و تست دستی
model = AutoModelForCausalLM.from_pretrained("gpt2")  # پیکربندی پیش‌فرض - کند

# بعد: بگذارید kvat بهترین پیکربندی را در 2 دقیقه پیدا کند
pip install kvat[full]
kvat tune gpt2 --profile ci-micro
# خروجی: "Best: dynamic/sdpa_flash/bfloat16 = 120 tok/s (+2.7% سریع‌تر)"
```

<div dir="rtl">

---

## نصب

</div>

```bash
pip install kvat[full]
```

<div dir="rtl">

---

## شروع سریع

</div>

```bash
# بهینه‌سازی هر مدل HuggingFace
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# تست سریع (پیشنهاد شده برای اولین بار)
kvat tune gpt2 --profile ci-micro

# نمایش اطلاعات سیستم
kvat info
```

<div dir="rtl">

---

## نتایج بنچمارک

### دسکتاپ (RTX 4060 - 8GB VRAM)

| مدل | Baseline | با kvat | بهبود |
|-----|----------|---------|-------|
| GPT-2 (124M) | 118.1 tok/s | 120.2 tok/s | **+1.8%** |
| Qwen2.5-0.5B | 28.7 tok/s | 29.5 tok/s | **+2.7%** |
| Phi-1.5 (1.3B) | 45.2 tok/s | 45.6 tok/s | **+0.9%** |

---

## پروفایل‌ها

| پروفایل | طول context | طول خروجی | مناسب برای |
|---------|-------------|-----------|------------|
| `ci-micro` | 512 | 32 | تست‌های سریع |
| `chat-agent` | 2-8K | 64-256 | چت‌بات‌ها، تأخیر کم |
| `rag` | 8-32K | 256-512 | پایپ‌لاین‌های RAG |
| `longform` | 4-8K | 1-2K | تولید متن طولانی |

---

## مجوز

Apache 2.0

## استناد

</div>

```bibtex
@software{kvat,
  title = {KVCache Auto-Tuner: بهینه‌سازی خودکار KV-Cache برای Transformers},
  author = {Keyvanhardani},
  year = {2026},
  url = {https://github.com/Keyvanhardani/kvcache-autotune}
}
```

<div dir="rtl">

---

<p align="center">
  <a href="https://keyvan.ai"><strong>Keyvan.ai</strong></a> | <a href="https://www.linkedin.com/in/keyvanhardani">LinkedIn</a>
</p>
<p align="center">
  ساخته شده در آلمان با علاقه برای جامعه HuggingFace
</p>

</div>
