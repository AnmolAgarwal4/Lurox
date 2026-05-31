import numpy as np
import json
import matplotlib.pyplot as plt

np.random.seed(42)

# Real data load karo
with open('benchmarks/alpha_sweep_metrics.json') as f:
    metrics = json.load(f)

diversity_alpha02 = np.array(metrics["0.2"]["per_query_diversity"])
diversity_alpha10 = np.array(metrics["1.0"]["per_query_diversity"])
diversity_alpha00 = np.array(metrics["0.0"]["per_query_diversity"])

def bootstrap_dist(a, b, n=10000):
    diffs = []
    for _ in range(n):
        idx = np.random.randint(0, len(a), size=len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    return np.array(diffs)

diffs_bm25  = bootstrap_dist(diversity_alpha02, diversity_alpha10)
diffs_dense = bootstrap_dist(diversity_alpha02, diversity_alpha00)

print("=" * 55)
print("Lurox Bootstrap Significance Test (n=10,000)")
print("Real per-query diversity data — pure NumPy")
print("=" * 55)

for diffs, label in [(diffs_bm25, "α=0.2 vs α=1.0 (pure BM25)"),
                     (diffs_dense, "α=0.2 vs α=0.0 (pure Dense)")]:
    p  = np.mean(diffs <= 0)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)
    print(f"\n{label}:")
    print(f"  Observed diff : {diffs.mean():.4f}")
    print(f"  p-value       : {p:.4f} {'✓ significant' if p < 0.05 else '✗ not significant'}")
    print(f"  95% CI        : [{lo:.4f}, {hi:.4f}]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0d0d1a')

for ax, diffs, label, color in zip(
    axes,
    [diffs_bm25, diffs_dense],
    ['α=0.2 vs α=1.0 (pure BM25)', 'α=0.2 vs α=0.0 (pure Dense)'],
    ['#7c3aed', '#06b6d4']
):
    ax.set_facecolor('#0d0d1a')
    ax.hist(diffs, bins=60, color=color, alpha=0.8, edgecolor='none')
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Null (diff=0)')
    ax.axvline(np.percentile(diffs, 2.5),  color='white', linewidth=1.5, linestyle=':')
    ax.axvline(np.percentile(diffs, 97.5), color='white', linewidth=1.5, linestyle=':', label='95% CI')

    p  = np.mean(diffs <= 0)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)

    ax.set_title(label, color='white', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Diversity Difference', color='white', fontsize=10)
    ax.set_ylabel('Bootstrap Frequency', color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
    ax.text(0.97, 0.95,
            f'p = {p:.4f}\n95% CI [{lo:.3f}, {hi:.3f}]',
            transform=ax.transAxes,
            ha='right', va='top', color='white', fontsize=9,
            bbox=dict(facecolor='#1a1a2e', edgecolor=color, boxstyle='round,pad=0.4'))

fig.suptitle('Bootstrap Significance Test — Lurox Diversity Peak\n(n=10,000 resamples, real per-query data, pure NumPy)',
             color='white', fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('benchmarks/bootstrap_significance.png',
            dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
print("\nSaved: benchmarks/bootstrap_significance.png")