# kvat - KVCache Auto-Tuner

[![npm version](https://img.shields.io/npm/v/kvat.svg)](https://www.npmjs.com/package/kvat)
[![PyPI](https://img.shields.io/pypi/v/kvat.svg)](https://pypi.org/project/kvat/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Keyvanhardani/kvcache-autotune/blob/main/LICENSE)

**Automatic KV-Cache Optimization for HuggingFace Transformers**

Find the optimal cache strategy, attention backend, and configuration for your model and hardware.

## Requirements

- **Node.js** 14.0+
- **Python** 3.9+
- **PyTorch** 2.0+
- **Transformers** 4.35+

## Installation

```bash
npm install kvat
```

Then install the Python package:

```bash
pip install kvat[full]
```

## CLI Usage

```bash
# Optimize any HuggingFace model
kvat tune meta-llama/Llama-3.2-1B --profile chat-agent

# Quick test
kvat tune gpt2 --profile ci-micro -v

# Show system info
kvat info
```

## JavaScript API

```javascript
const kvat = require('kvat');

// Check if kvat is installed
if (!kvat.isKvatInstalled()) {
  await kvat.installKvat();  // Install Python package
}

// Run tuning
const result = await kvat.tune('gpt2', {
  profile: 'chat-agent',
  outputDir: './results',
  verbose: true
});

console.log('Results saved to:', result.outputDir);

// Get system info
const info = await kvat.info();
console.log(info);

// Run arbitrary command
const { stdout, stderr, code } = await kvat.run(['profiles']);
```

## API Reference

### `isKvatInstalled()`

Check if the kvat Python package is installed.

Returns: `boolean`

### `installKvat(full = true)`

Install the kvat Python package.

- `full` (boolean): Install with full dependencies (default: true)

Returns: `Promise<void>`

### `tune(modelId, options)`

Run kvat tune command.

- `modelId` (string): HuggingFace model ID
- `options.profile` (string): Profile name (default: 'chat-agent')
- `options.device` (string): Device cuda/cpu/mps (default: 'cuda')
- `options.outputDir` (string): Output directory (default: './kvat_results')
- `options.verbose` (boolean): Verbose output (default: false)

Returns: `Promise<{success: boolean, outputDir: string, stdout: string, stderr: string}>`

### `info()`

Get system information.

Returns: `Promise<string>`

### `run(args)`

Run arbitrary kvat command.

- `args` (string[]): Command arguments

Returns: `Promise<{stdout: string, stderr: string, code: number}>`

## Available Profiles

| Profile | Context | Output | Focus |
|---------|---------|--------|-------|
| `chat-agent` | 2-8K | 64-256 | TTFT (latency) |
| `rag` | 8-32K | 256-512 | Balanced |
| `longform` | 4-8K | 1-2K | Throughput |
| `ci-micro` | 512 | 32 | Quick testing |

## Links

- **GitHub**: https://github.com/Keyvanhardani/kvcache-autotune
- **PyPI**: https://pypi.org/project/kvat/
- **Documentation**: https://github.com/Keyvanhardani/kvcache-autotune#readme

## License

Apache 2.0

---

<p align="center">
  <a href="https://keyvan.ai"><strong>Keyvan.ai</strong></a> | <a href="https://www.linkedin.com/in/keyvanhardani">LinkedIn</a>
</p>
<p align="center">
  Made in Germany with dedication for the HuggingFace Community
</p>
