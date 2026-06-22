import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RE_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
RE_REWARD = re.compile(r"Mean reward:\s+([-+]?\d+(?:\.\d+)?)")
RE_EPISODE_LENGTH = re.compile(r"Mean episode length:\s+([-+]?\d+(?:\.\d+)?)")


DEFAULT_DRIFT_RUNS = [
    "seed1",
    "seed2",
    "seed3",
    "seed4",
    "seed42",
]

DEFAULT_PPO_RUNS = [
    "1",
    "2",
    "3",
    "4",
    "5",
]


def parse_log(log_path):
    """Parse a terminal.log file and return reward and episode-length rows.

    Each row is (iteration, value).  Returns two lists.
    """
    reward_rows = []
    ep_len_rows = []
    current_iter = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        iter_match = RE_ITER.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            continue
        reward_match = RE_REWARD.search(line)
        if reward_match and current_iter is not None:
            reward_rows.append((current_iter, float(reward_match.group(1))))
            continue
        ep_len_match = RE_EPISODE_LENGTH.search(line)
        if ep_len_match and current_iter is not None:
            ep_len_rows.append((current_iter, float(ep_len_match.group(1))))
    if not reward_rows:
        raise ValueError(f"No reward points found in {log_path}")
    return reward_rows, ep_len_rows


def load_group(root, run_names):
    """Load a group of runs, returning (iters, reward_values, ep_len_values, paths)."""
    reward_curves = []
    ep_len_curves = []
    used_paths = []
    for run_name in run_names:
        run_dir = root / run_name
        log_path = run_dir / "terminal.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing terminal.log: {log_path}")
        reward_rows, ep_len_rows = parse_log(log_path)
        reward_curves.append(dict(reward_rows))
        ep_len_curves.append(dict(ep_len_rows))
        used_paths.append(run_dir)

    # --- reward ---
    common_iters = sorted(set.intersection(*(set(curve) for curve in reward_curves)))
    if not common_iters:
        raise ValueError(f"No common iterations found for group under {root}")
    reward_values = np.array([[curve[it] for it in common_iters] for curve in reward_curves], dtype=np.float64)

    # --- episode length ---
    ep_len_common_iters = sorted(set.intersection(*(set(curve) for curve in ep_len_curves)))
    if ep_len_common_iters:
        ep_len_values = np.array(
            [[curve[it] for it in ep_len_common_iters] for curve in ep_len_curves], dtype=np.float64
        )
    else:
        ep_len_values = None

    return np.array(common_iters, dtype=np.int64), reward_values, ep_len_values, np.array(ep_len_common_iters, dtype=np.int64) if ep_len_common_iters else None, used_paths


def smooth_mean(values, window):
    if window <= 1:
        return values.copy()
    window = min(window, values.shape[0])
    kernel = np.ones(window, dtype=np.float64) / window
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def summarize_group(values, smooth_window):
    raw_mean = values.mean(axis=0)
    raw_std = values.std(axis=0, ddof=0)
    return smooth_mean(raw_mean, smooth_window), raw_std


# ---------------------------------------------------------------------------
# Academic-paper styling
# ---------------------------------------------------------------------------
def set_academic_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "FreeSerif"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
        }
    )


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.set_xlabel("iteration")
    ax.tick_params(axis="both", labelsize=10)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------



def plot_combined(output_path, groups, smooth_window, ylabel="Reward"):
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for group in groups:
        mean_curve, std_curve = summarize_group(group["values"], smooth_window)
        ax.plot(group["iters"], mean_curve, color=group["color"], linewidth=1.8, label=group["label"])
        ax.fill_between(
            group["iters"],
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=group["color"],
            alpha=0.16,
            linewidth=0,
        )
    style_axes(ax)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(output_dir, groups, ep_len_groups, smooth_window):
    lines = [
        "# Reward & Episode-Length Curve Plot Manifest",
        "",
        f"smooth_window: {smooth_window}",
        "mean_smoothing: moving average on cross-seed mean only",
        "variance_smoothing: none; shaded band is raw per-iteration population std",
        "",
        "## Run directories",
        "",
    ]
    for group in groups:
        lines.append(f"### {group['label']}")
        for path in group["paths"]:
            lines.append(f"- {path}")
        lines.append("")

    lines.append("## Final statistics (last 50 iterations)")
    lines.append("")

    for group in groups:
        label = group["label"]
        final_reward_mean = group["values"][:, -50:].mean(axis=1).mean()
        final_reward_std = group["values"][:, -50:].mean(axis=1).std(ddof=0)
        lines.append(f"### {label} – Reward")
        lines.append(f"- final_reward_mean: {final_reward_mean:.3f}")
        lines.append(f"- final_reward_std:  {final_reward_std:.3f}")
        lines.append("")

    for group in ep_len_groups:
        if group["values"] is None:
            continue
        label = group["label"]
        final_ep_mean = group["values"][:, -50:].mean(axis=1).mean()
        final_ep_std = group["values"][:, -50:].mean(axis=1).std(ddof=0)
        lines.append(f"### {label} – Episode Length")
        lines.append(f"- final_episode_length_mean: {final_ep_mean:.3f}")
        lines.append(f"- final_episode_length_std:  {final_ep_std:.3f}")
        lines.append("")

    (output_dir / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_run_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot reward and episode-length curves for Drift and PPO seed groups.")
    parser.add_argument("--drift_root", type=Path, default=Path("results/anymal_c_rough/Drift"))
    parser.add_argument("--ppo_root", type=Path, default=Path("results/anymal_c_rough/PPO"))
    parser.add_argument("--drift_runs", type=parse_run_list, default=DEFAULT_DRIFT_RUNS)
    parser.add_argument("--ppo_runs", type=parse_run_list, default=DEFAULT_PPO_RUNS)
    parser.add_argument("--output_dir", type=Path, default=Path("results/anymal_c_rough/reward_curve_comparison"))
    parser.add_argument("--smooth_window", type=int, default=25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    drift_iters, drift_reward, drift_ep_len, drift_ep_iters, drift_paths = load_group(args.drift_root, args.drift_runs)
    ppo_iters, ppo_reward, ppo_ep_len, ppo_ep_iters, ppo_paths = load_group(args.ppo_root, args.ppo_runs)

    # --- reward groups ---
    reward_groups = [
        {
            "label": "AGD-PPO",
            "iters": drift_iters,
            "values": drift_reward,
            "paths": drift_paths,
            "color": "#1f77b4",
        },
        {
            "label": "PPO",
            "iters": ppo_iters,
            "values": ppo_reward,
            "paths": ppo_paths,
            "color": "#d62728",
        },
    ]

    # --- episode-length groups ---
    ep_len_groups = [
        {
            "label": "AGD-PPO",
            "iters": drift_ep_iters,
            "values": drift_ep_len,
            "paths": drift_paths,
            "color": "#1f77b4",
        },
        {
            "label": "PPO",
            "iters": ppo_ep_iters,
            "values": ppo_ep_len,
            "paths": ppo_paths,
            "color": "#d62728",
        },
    ]

    # Reward plots
    plot_combined(
        args.output_dir / "drift_vs_ppo_reward_curve",
        reward_groups,
        args.smooth_window,
    )

    # Episode-length plots
    if drift_ep_len is not None and ppo_ep_len is not None:
        plot_combined(
            args.output_dir / "drift_vs_ppo_episode_length_curve",
            ep_len_groups,
            args.smooth_window,
            ylabel="Mean episode length",
        )

    write_manifest(args.output_dir, reward_groups, ep_len_groups, args.smooth_window)

    print(f"Saved plots to {args.output_dir.resolve()}")
    print()
    for group in reward_groups:
        final_mean = group["values"][:, -50:].mean(axis=1).mean()
        final_std = group["values"][:, -50:].mean(axis=1).std(ddof=0)
        print(f"{group['label']}: last50 seed-mean reward = {final_mean:.3f} ± {final_std:.3f}")
    print()
    for group in ep_len_groups:
        if group["values"] is None:
            continue
        final_ep_mean = group["values"][:, -50:].mean(axis=1).mean()
        final_ep_std = group["values"][:, -50:].mean(axis=1).std(ddof=0)
        print(f"{group['label']}: last50 seed-mean episode length = {final_ep_mean:.3f} ± {final_ep_std:.3f}")


if __name__ == "__main__":
    main()
