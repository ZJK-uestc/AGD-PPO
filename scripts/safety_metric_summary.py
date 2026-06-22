#!/usr/bin/env python3
"""Summarize safety-related reward terms over the same 5-seed groups used by
plot_reward_groups.py, plot_reward_groups2.py, and plot_reward_groups3.py.

For each environment and algorithm, the script computes the mean over the last
50 iterations for each run, then reports group mean +- population std across
the five runs.
"""

from pathlib import Path
import re
import statistics


RE_ITER = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
RE_METRIC = re.compile(r"Mean episode (rew_[^:]+):\s+([-+]?\d+(?:\.\d+)?)")

LAST_N = 50
METRICS = [
    "rew_collision",
    "rew_lin_vel_z",
    "rew_dof_acc",
    "rew_tracking_lin_vel",
]

GROUPS = [
    {
        "env": "A1",
        "agd_label": "AGD-PPO",
        "agd_logs": [
            Path("results/a1/Drift/action_only_seed1/terminal.log"),
            Path("results/a1/Drift/action_only_seed3/terminal.log"),
            Path("results/a1/Drift/seed1/terminal.log"),
            Path("results/a1/Drift/action_only_seed5/terminal.log"),
            Path("results/a1/Drift/seed2/terminal.log"),
        ],
        "ppo_label": "PPO",
        "ppo_logs": [
            Path("results/a1/PPO/seed1/terminal.log"),
            Path("results/a1/PPO/seed2/terminal.log"),
            Path("results/a1/PPO/seed3/terminal.log"),
            Path("results/a1/PPO/seed6/terminal.log"),
            Path("results/a1/PPO/seed44/terminal.log"),
        ],
    },
    {
        "env": "ANYmal-C",
        "agd_label": "AGD-PPO",
        "agd_logs": [
            Path("results/anymal_c_rough/Drift/seed1/terminal.log"),
            Path("results/anymal_c_rough/Drift/seed2/terminal.log"),
            Path("results/anymal_c_rough/Drift/seed3/terminal.log"),
            Path("results/anymal_c_rough/Drift/seed4/terminal.log"),
            Path("results/anymal_c_rough/Drift/seed42/terminal.log"),
        ],
        "ppo_label": "PPO",
        "ppo_logs": [
            Path("results/anymal_c_rough/PPO/1/terminal.log"),
            Path("results/anymal_c_rough/PPO/2/terminal.log"),
            Path("results/anymal_c_rough/PPO/3/terminal.log"),
            Path("results/anymal_c_rough/PPO/4/terminal.log"),
            Path("results/anymal_c_rough/PPO/5/terminal.log"),
        ],
    },
    {
        "env": "Cassie",
        "agd_label": "AGD-PPO",
        "agd_logs": [
            Path("results/cassie/Drift/v1_seed1/terminal.log"),
            Path("results/cassie/Drift/v1seed4/terminal.log"),
            Path("results/cassie/Drift/v1seed5/terminal.log"),
            Path("results/cassie/Drift/v1seed6/terminal.log"),
            Path("results/cassie/Drift/v1seed8/terminal.log"),
        ],
        "ppo_label": "PPO",
        "ppo_logs": [
            Path("results/cassie/PPO/seed2/terminal.log"),
            Path("results/cassie/PPO/seed3/terminal.log"),
            Path("results/cassie/PPO/seed5/terminal.log"),
            Path("results/cassie/PPO/seed4/terminal.log"),
            Path("results/cassie/PPO/seed6/terminal.log"),
        ],
    },
]


def parse_terminal_log(path: Path):
    series = {metric: {} for metric in METRICS}
    current_iter = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        iter_match = RE_ITER.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            continue
        metric_match = RE_METRIC.search(line)
        if metric_match and current_iter is not None:
            metric_name = metric_match.group(1)
            if metric_name in series:
                series[metric_name][current_iter] = float(metric_match.group(2))
    return series


def last_n_mean(metric_series, last_n=LAST_N):
    iterations = sorted(metric_series.keys())
    if not iterations:
        return None
    selected = iterations[-last_n:]
    return sum(metric_series[it] for it in selected) / len(selected)


def summarize_logs(log_paths):
    per_metric = {metric: [] for metric in METRICS}
    for log_path in log_paths:
        parsed = parse_terminal_log(log_path)
        for metric in METRICS:
            value = last_n_mean(parsed[metric])
            if value is None:
                raise ValueError(f"Missing metric {metric} in {log_path}")
            per_metric[metric].append(value)

    summary = {}
    for metric, values in per_metric.items():
        summary[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
        }
    return summary


def fmt(mean, std):
    return f"{mean:.6f} ± {std:.6f}"


def main():
    out_dir = Path("results") / "figure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "safety_metrics_summary.md"

    lines = [
        "# Safety-Related Metric Summary",
        "",
        "Grouping: same 5-seed groups used by plot_reward_groups.py, plot_reward_groups2.py, and plot_reward_groups3.py",
        f"Aggregation: per run last {LAST_N} iterations mean, then group mean +- population std across 5 runs",
        "",
        "Interpretation:",
        "- `rew_collision`: less negative is better",
        "- `rew_lin_vel_z`: less negative is better",
        "- `rew_dof_acc`: less negative is better",
        "- `rew_tracking_lin_vel`: more positive is better",
        "",
    ]

    for group in GROUPS:
        agd = summarize_logs(group["agd_logs"])
        ppo = summarize_logs(group["ppo_logs"])

        lines.append(f"## {group['env']}")
        lines.append("")
        lines.append("| Metric | AGD-PPO (mean ± std) | PPO (mean ± std) | Preferred |")
        lines.append("|---|---:|---:|---|")
        for metric in METRICS:
            preferred = "higher" if metric == "rew_tracking_lin_vel" else "less negative"
            lines.append(
                f"| `{metric}` | {fmt(agd[metric]['mean'], agd[metric]['std'])} | "
                f"{fmt(ppo[metric]['mean'], ppo[metric]['std'])} | {preferred} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary to {out_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
