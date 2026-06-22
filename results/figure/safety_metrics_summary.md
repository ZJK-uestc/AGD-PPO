# Safety-Related Metric Summary

Grouping: same 5-seed groups used by plot_reward_groups.py, plot_reward_groups2.py, and plot_reward_groups3.py
Aggregation: per run last 50 iterations mean, then group mean +- population std across 5 runs

Interpretation:
- `rew_collision`: less negative is better
- `rew_lin_vel_z`: less negative is better
- `rew_dof_acc`: less negative is better
- `rew_tracking_lin_vel`: more positive is better

## A1

| Metric | AGD-PPO (mean ± std) | PPO (mean ± std) | Preferred |
|---|---:|---:|---|
| `rew_collision` | -0.027605 ± 0.002026 | -0.034571 ± 0.005409 | less negative |
| `rew_lin_vel_z` | -0.053895 ± 0.001479 | -0.062040 ± 0.004037 | less negative |
| `rew_dof_acc` | -0.136002 ± 0.005261 | -0.145440 ± 0.006091 | less negative |
| `rew_tracking_lin_vel` | 0.818047 ± 0.005633 | 0.776095 ± 0.016287 | higher |

## ANYmal-C

| Metric | AGD-PPO (mean ± std) | PPO (mean ± std) | Preferred |
|---|---:|---:|---|
| `rew_collision` | -0.026476 ± 0.002127 | -0.029050 ± 0.001458 | less negative |
| `rew_lin_vel_z` | -0.042405 ± 0.002134 | -0.043970 ± 0.001578 | less negative |
| `rew_dof_acc` | -0.065934 ± 0.003378 | -0.085080 ± 0.003640 | less negative |
| `rew_tracking_lin_vel` | 0.841995 ± 0.005572 | 0.855346 ± 0.003653 | higher |

## Cassie

| Metric | AGD-PPO (mean ± std) | PPO (mean ± std) | Preferred |
|---|---:|---:|---|
| `rew_collision` | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 | less negative |
| `rew_lin_vel_z` | -0.029948 ± 0.003763 | -0.032482 ± 0.003656 | less negative |
| `rew_dof_acc` | -0.066368 ± 0.003534 | -0.120230 ± 0.003902 | less negative |
| `rew_tracking_lin_vel` | 0.831829 ± 0.008881 | 0.826608 ± 0.012366 | higher |
