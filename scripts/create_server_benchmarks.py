#!/usr/bin/env python3
"""Create server benchmark visualizations for README."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Server benchmark data (RTX 4000 SFF Ada 20GB)
models = ['GPT-2\n(124M)', 'Qwen2.5\n(0.5B)', 'TinyLlama\n(1.1B)', 'Phi-1.5\n(1.3B)']
throughput = [407.1, 140.7, 93.0, 78.8]
ttft = [3.96, 10.92, 30.57, 37.20]
vram = [284, 968, 2170, 2838]

# Color palette
PRIMARY = '#FF6F00'  # Orange
SECONDARY = '#1976D2'  # Blue
ACCENT = '#43A047'  # Green
BG_COLOR = '#FAFAFA'
GRID_COLOR = '#E0E0E0'

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# Figure 1: Hero comparison chart
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

x = np.arange(len(models))
width = 0.6

bars = ax.bar(x, throughput, width, color=PRIMARY, edgecolor='white', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, throughput):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom',
                fontweight='bold', fontsize=14, color='#333')

ax.set_ylabel('Throughput (tokens/second)', fontweight='bold', fontsize=12)
ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_title('Server Benchmark Results\nRTX 4000 SFF Ada (20GB VRAM)',
             fontweight='bold', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, max(throughput) * 1.15)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add kvat watermark
fig.text(0.99, 0.01, 'kvat.keyvan.ai', ha='right', va='bottom',
         fontsize=9, color='#888', style='italic')

plt.tight_layout()
plt.savefig('assets/server_throughput.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
plt.close()

# Figure 2: TTFT comparison
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars = ax.barh(models, ttft, color=SECONDARY, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, ttft):
    width = bar.get_width()
    ax.annotate(f'{val:.1f}ms',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center',
                fontweight='bold', fontsize=12, color='#333')

ax.set_xlabel('Time to First Token (ms)', fontweight='bold', fontsize=12)
ax.set_title('TTFT Latency Comparison', fontweight='bold', fontsize=14, pad=15)
ax.set_xlim(0, max(ttft) * 1.25)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.text(0.99, 0.01, 'kvat.keyvan.ai', ha='right', va='bottom',
         fontsize=9, color='#888', style='italic')

plt.tight_layout()
plt.savefig('assets/server_ttft.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
plt.close()

# Figure 3: VRAM usage
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

bars = ax.barh(models, vram, color=ACCENT, edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, vram):
    width = bar.get_width()
    ax.annotate(f'{val} MB',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha='left', va='center',
                fontweight='bold', fontsize=12, color='#333')

ax.set_xlabel('Peak VRAM Usage (MB)', fontweight='bold', fontsize=12)
ax.set_title('VRAM Memory Footprint', fontweight='bold', fontsize=14, pad=15)
ax.set_xlim(0, max(vram) * 1.2)
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.text(0.99, 0.01, 'kvat.keyvan.ai', ha='right', va='bottom',
         fontsize=9, color='#888', style='italic')

plt.tight_layout()
plt.savefig('assets/server_vram.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
plt.close()

# Figure 4: Combined dashboard
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG_COLOR)

# Throughput
ax = axes[0]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(range(len(models)), throughput, color=PRIMARY, edgecolor='white')
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace('\n', ' ') for m in models], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('tok/s', fontweight='bold')
ax.set_title('Throughput', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# TTFT
ax = axes[1]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(range(len(models)), ttft, color=SECONDARY, edgecolor='white')
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace('\n', ' ') for m in models], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('ms', fontweight='bold')
ax.set_title('TTFT Latency', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# VRAM
ax = axes[2]
ax.set_facecolor(BG_COLOR)
bars = ax.bar(range(len(models)), vram, color=ACCENT, edgecolor='white')
ax.set_xticks(range(len(models)))
ax.set_xticklabels([m.replace('\n', ' ') for m in models], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('MB', fontweight='bold')
ax.set_title('VRAM Usage', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.suptitle('Server Benchmark Dashboard - RTX 4000 SFF Ada (20GB)',
             fontweight='bold', fontsize=14, y=1.02)

fig.text(0.99, 0.01, 'kvat.keyvan.ai', ha='right', va='bottom',
         fontsize=9, color='#888', style='italic')

plt.tight_layout()
plt.savefig('assets/server_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
plt.close()

print("Server benchmark graphics created:")
print("  - assets/server_throughput.png")
print("  - assets/server_ttft.png")
print("  - assets/server_vram.png")
print("  - assets/server_dashboard.png")
