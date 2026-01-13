<div dir="rtl">

# KVCache Auto-Tuner

<p align="center">
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/actions"><img src="https://github.com/Keyvanhardani/kvcache-autotune/workflows/Tests/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/v/kvat.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/kvat"><img src="https://img.shields.io/npm/v/kvat.svg" alt="npm"></a>
  <a href="https://pypi.org/project/kvat/"><img src="https://img.shields.io/pypi/pyversions/kvat.svg" alt="Python"></a>
  <a href="https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="الترخيص"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_DE.md">Deutsch</a> | <a href="README_FR.md">Francais</a> | <a href="README_ES.md">Espanol</a> | <a href="README_FA.md">فارسی</a> | <strong>العربية</strong>
</p>

---

## لماذا kvat؟

عند تشغيل نماذج اللغة الكبيرة مع HuggingFace Transformers، هناك **عشرات خيارات التكوين** التي تؤثر على الأداء:

| الإعداد | الخيارات | التأثير |
|---------|----------|---------|
| استراتيجية التخزين المؤقت | dynamic, static, sliding_window | الذاكرة، سرعة prefill |
| Backend Attention | sdpa_flash, eager, math, mem_efficient | الإنتاجية، VRAM |
| نوع البيانات | bfloat16, float16, float32 | السرعة مقابل الدقة |

**المشكلة:** يعتمد التركيب الأمثل على نموذجك + GPU الخاص بك + حالة الاستخدام الخاصة بك. لا أحد يعرف أي تكوين هو الأفضل بدون اختبار.

**الحل:** يقوم `kvat` تلقائيًا باختبار جميع التركيبات ويخبرك بأسرع تكوين.

</div>

```bash
# قبل: التخمين والاختبار اليدوي
model = AutoModelForCausalLM.from_pretrained("gpt2")  # التكوين الافتراضي - بطيء

# بعد: دع kvat يجد أفضل تكوين في دقيقتين
pip install kvat[full]
kvat tune gpt2 --profile ci-micro
# المخرج: "Best: dynamic/sdpa_flash/bfloat16 = 120 tok/s (+2.7% أسرع)"
```

<div dir="rtl">

---

## التثبيت

</div>

```bash
pip install kvat[full]
```

<div dir="rtl">

---

## البدء السريع

</div>

```bash
# تحسين أي نموذج HuggingFace
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# اختبار سريع (موصى به للمحاولة الأولى)
kvat tune gpt2 --profile ci-micro

# عرض معلومات النظام
kvat info
```

<div dir="rtl">

---

## نتائج الأداء

### سطح المكتب (RTX 4060 - 8GB VRAM)

| النموذج | Baseline | مع kvat | التحسن |
|---------|----------|---------|--------|
| GPT-2 (124M) | 118.1 tok/s | 120.2 tok/s | **+1.8%** |
| Qwen2.5-0.5B | 28.7 tok/s | 29.5 tok/s | **+2.7%** |
| Phi-1.5 (1.3B) | 45.2 tok/s | 45.6 tok/s | **+0.9%** |

---

## الملفات الشخصية

| الملف الشخصي | طول السياق | طول المخرج | مناسب لـ |
|--------------|------------|------------|----------|
| `ci-micro` | 512 | 32 | اختبارات سريعة |
| `chat-agent` | 2-8K | 64-256 | روبوتات الدردشة، زمن استجابة منخفض |
| `rag` | 8-32K | 256-512 | خطوط أنابيب RAG |
| `longform` | 4-8K | 1-2K | توليد نص طويل |

---

## الترخيص

Apache 2.0

## الاستشهاد

</div>

```bibtex
@software{kvat,
  title = {KVCache Auto-Tuner: تحسين تلقائي لـ KV-Cache لـ Transformers},
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
  صنع في ألمانيا بإخلاص لمجتمع HuggingFace
</p>

</div>
