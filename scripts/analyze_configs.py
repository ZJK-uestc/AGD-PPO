"""
Compare all Drift configurations vs PPO baseline on ANYmal-C Rough.
Generates publication-quality reward and episode-length plots.
"""
import re, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/anymal_c_rough")
OUT = BASE / "reward_curve_comparison"
OUT.mkdir(parents=True, exist_ok=True)

RE_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
RE_REWARD = re.compile(r"Mean reward:\s+([-+]?\d+(?:\.\d+)?)")
RE_EP_LEN = re.compile(r"Mean episode length:\s+([-+]?\d+(?:\.\d+)?)")

# ── Academic paper style ───────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "FreeSerif"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
})

COLORS = {
    'Drift-default':      '#1f77b4',
    'Drift-baoshou':      '#ff7f0e',
    'Drift-baoshouseed':  '#2ca02c',
    'Drift-state_kernel': '#9467bd',
    'PPO-baseline':       '#d62728',
}

LABELS = {
    'Drift-default':      'AGD-PPO (default)',
    'Drift-baoshou':      'AGD-PPO (conservative)',
    'Drift-baoshouseed':  'AGD-PPO (ultra-cons.)',
    'Drift-state_kernel': 'AGD-PPO (state kernel)',
    'PPO-baseline':       'PPO (baseline)',
}

GROUPS_ORDER = ['PPO-baseline', 'Drift-default', 'Drift-state_kernel',
                'Drift-baoshou', 'Drift-baoshouseed']


def parse_log(log_path):
    reward_rows, ep_rows = [], []
    cur = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = RE_ITER.search(line)
        if m:
            cur = int(m.group(1))
            continue
        m = RE_REWARD.search(line)
        if m and cur is not None:
            reward_rows.append((cur, float(m.group(1))))
            continue
        m = RE_EP_LEN.search(line)
        if m and cur is not None:
            ep_rows.append((cur, float(m.group(1))))
    return dict(reward_rows), dict(ep_rows)


def load_runs(root, names):
    rewards, ep_lens = [], []
    for n in names:
        r, e = parse_log(root / n / "terminal.log")
        rewards.append(r)
        ep_lens.append(e)
    return rewards, ep_lens


def align_curves(curves):
    common = sorted(set.intersection(*(set(c) for c in curves)))
    if not common:
        raise ValueError("No common iterations")
    vals = np.array([[c[i] for i in common] for c in curves], dtype=np.float64)
    return np.array(common, dtype=np.int64), vals


def smooth_mean(values, window):
    if window <= 1:
        return values.copy()
    window = min(window, values.shape[0])
    kernel = np.ones(window, dtype=np.float64) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def summarize(values, window):
    raw_mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    return smooth_mean(raw_mean, window), raw_std


def style_ax(ax, ylabel):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=10)


# ══════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════
groups = {}
group_defs = [
    ('Drift-default',      BASE / 'Drift', ['1', '2', '3', '4', '5']),
    ('Drift-baoshou',      BASE / 'Drift', ['baoshou_seed1', 'baoshou_seed42']),
    ('Drift-baoshouseed',  BASE / 'Drift', ['baoshouseed1', 'baoshouseed42']),
    ('Drift-state_kernel', BASE / 'Drift', ['state_seed1', 'state_seed42']),
    ('PPO-baseline',       BASE / 'PPO',   ['1', '2', '3', '4', '5']),
]

for label, root, names in group_defs:
    r_curves, e_curves = load_runs(root, names)
    r_iters, r_vals = align_curves(r_curves)
    e_iters, e_vals = align_curves(e_curves)
    groups[label] = {
        'r_iters': r_iters, 'r_vals': r_vals,
        'e_iters': e_iters, 'e_vals': e_vals,
        'n_seeds': len(r_curves),
    }

SW = 25  # smoothing window

# ══════════════════════════════════════════════════════════════════
# Plot 1: Reward — all configs overlaid
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for label in GROUPS_ORDER:
    g = groups[label]
    m, s = summarize(g['r_vals'], SW)
    ax.plot(g['r_iters'], m, color=COLORS[label], linewidth=1.8, label=LABELS[label])
    ax.fill_between(g['r_iters'], m - s, m + s, color=COLORS[label], alpha=0.14, linewidth=0)
style_ax(ax, "Mean episode reward")
ax.set_title("Reward Comparison on ANYmal-C Rough", fontsize=13, pad=10)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "reward_all_configs.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "reward_all_configs.pdf", bbox_inches="tight")
plt.close(fig)
print("[1/7] reward_all_configs saved")

# ══════════════════════════════════════════════════════════════════
# Plot 2: Reward — Drift-default vs PPO baseline
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for label in ['Drift-default', 'PPO-baseline']:
    g = groups[label]
    m, s = summarize(g['r_vals'], SW)
    ax.plot(g['r_iters'], m, color=COLORS[label], linewidth=2.0, label=LABELS[label])
    ax.fill_between(g['r_iters'], m - s, m + s, color=COLORS[label], alpha=0.16, linewidth=0)
style_ax(ax, "Mean episode reward")
ax.set_title("AGD-PPO vs PPO Baseline", fontsize=13, pad=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "reward_drift_vs_ppo.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "reward_drift_vs_ppo.pdf", bbox_inches="tight")
plt.close(fig)
print("[2/7] reward_drift_vs_ppo saved")

# ══════════════════════════════════════════════════════════════════
# Plot 3: Episode Length — all configs
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.8, 3.8))
for label in GROUPS_ORDER:
    g = groups[label]
    m, s = summarize(g['e_vals'], SW)
    ax.plot(g['e_iters'], m, color=COLORS[label], linewidth=1.8, label=LABELS[label])
    ax.fill_between(g['e_iters'], m - s, m + s, color=COLORS[label], alpha=0.14, linewidth=0)
style_ax(ax, "Mean episode length")
ax.set_title("Episode Length Comparison", fontsize=13, pad=10)
ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "ep_len_all_configs.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ep_len_all_configs.pdf", bbox_inches="tight")
plt.close(fig)
print("[3/7] ep_len_all_configs saved")

# ══════════════════════════════════════════════════════════════════
# Plot 4: Episode Length — Drift-default vs PPO
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for label in ['Drift-default', 'PPO-baseline']:
    g = groups[label]
    m, s = summarize(g['e_vals'], SW)
    ax.plot(g['e_iters'], m, color=COLORS[label], linewidth=2.0, label=LABELS[label])
    ax.fill_between(g['e_iters'], m - s, m + s, color=COLORS[label], alpha=0.16, linewidth=0)
style_ax(ax, "Mean episode length")
ax.set_title("AGD-PPO vs PPO — Episode Length", fontsize=13, pad=10)
ax.legend(frameon=False, fontsize=10, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "ep_len_drift_vs_ppo.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "ep_len_drift_vs_ppo.pdf", bbox_inches="tight")
plt.close(fig)
print("[4/7] ep_len_drift_vs_ppo saved")

# ══════════════════════════════════════════════════════════════════
# Plots 5-7: Per-config reward curves with individual seeds
# ══════════════════════════════════════════════════════════════════
for label in GROUPS_ORDER:
    g = groups[label]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    # faint individual seed curves
    for i in range(g['r_vals'].shape[0]):
        ax.plot(g['r_iters'], g['r_vals'][i], color=COLORS[label],
                linewidth=0.5, alpha=0.35)
    m, s = summarize(g['r_vals'], SW)
    ax.plot(g['r_iters'], m, color=COLORS[label], linewidth=2.2,
            label=f"{LABELS[label]} (n={g['n_seeds']})")
    ax.fill_between(g['r_iters'], m - s, m + s, color=COLORS[label],
                    alpha=0.16, linewidth=0)
    style_ax(ax, "Mean episode reward")
    ax.set_title(f"{LABELS[label]} — Reward", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    slug = label.lower().replace('-', '_')
    fig.savefig(OUT / f"reward_{slug}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"reward_{slug}.pdf", bbox_inches="tight")
    plt.close(fig)
print("[5-7] per-config reward plots saved")

# ══════════════════════════════════════════════════════════════════
# Write manifest.md
# ══════════════════════════════════════════════════════════════════
lines = [
    "# Experiment Manifest — ANYmal-C Rough",
    "",
    f"**Smoothing window:** {SW} iterations (moving average on cross-seed mean)",
    "**Shaded band:** raw per-iteration population std (no smoothing)",
    "",
    "## Configurations",
    "",
    "| Label | Experiment | Drift | State Kern | Res.Drift | Top Frac | Max Act Dist | Step Size | Loss Coef | Warmup (M/A) |",
    "|-------|-----------|-------|------------|-----------|----------|-------------|-----------|-----------|--------------|",
    "| AGD-PPO (default) | stage3_small_positive | ✓ | | | 0.50 | 1.5 | 0.1 | 0.001 | 300/400 |",
    "| AGD-PPO (conservative) | stage3_baoshou | ✓ | | | 0.35 | 1.0 | 0.1 | 0.001 | 500/700 |",
    "| AGD-PPO (ultra-cons.) | stage3_baoshou | ✓ | | | 0.35 | 1.0 | 0.05 | 0.0005 | 500/700 |",
    "| AGD-PPO (state kernel) | stage3_state_kernel | ✓ | ✓ | ✓ | 0.35 | 1.5 | 0.1 | 0.001 | 300/400 |",
    "| PPO (baseline) | stage3_baseline | | | | — | — | — | — | — |",
    "",
    "## Final Performance (last 50 iterations)",
    "",
]

for label in GROUPS_ORDER:
    g = groups[label]
    r_final = g['r_vals'][:, -50:].mean(axis=1)
    e_final = g['e_vals'][:, -50:].mean(axis=1)
    lines.append(f"### {LABELS[label]} (n={g['n_seeds']})")
    lines.append(f"- **Reward:** {r_final.mean():.3f} ± {r_final.std(ddof=0):.3f}")
    lines.append(f"- **Episode Length:** {e_final.mean():.1f} ± {e_final.std(ddof=0):.1f}")
    lines.append("")

# Per-seed detail
lines.append("## Per-Seed Detail")
lines.append("")
for label in GROUPS_ORDER:
    g = groups[label]
    lines.append(f"### {LABELS[label]}")
    for i in range(g['r_vals'].shape[0]):
        rf = g['r_vals'][i, -50:].mean()
        rm = g['r_vals'][i].max()
        ef = g['e_vals'][i, -50:].mean()
        lines.append(f"- seed {i}: final_reward={rf:.3f}  max_reward={rm:.3f}  final_ep_len={ef:.1f}")
    lines.append("")

(OUT / "manifest.md").write_text("\n".join(lines), encoding="utf-8")
print("[✓] manifest.md written")

print(f"\nAll outputs saved to {OUT.resolve()}")
