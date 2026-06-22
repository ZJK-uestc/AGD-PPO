# Reward & Episode-Length Curve Plot Manifest

smooth_window: 25
mean_smoothing: moving average on cross-seed mean only
variance_smoothing: none; shaded band is raw per-iteration population std

## Run directories

### AGD-PPO
- results/cassie/Drift/v1_seed1
- results/cassie/Drift/v1seed4
- results/cassie/Drift/v1seed5
- results/cassie/Drift/v1seed6
- results/cassie/Drift/v1seed8

### PPO
- results/cassie/PPO/seed2
- results/cassie/PPO/seed3
- results/cassie/PPO/seed5
- results/cassie/PPO/seed4
- results/cassie/PPO/seed6

## Final statistics (last 50 iterations)

### AGD-PPO – Reward
- final_reward_mean: 29.562
- final_reward_std:  0.718

### PPO – Reward
- final_reward_mean: 25.101
- final_reward_std:  1.388

### AGD-PPO – Episode Length
- final_episode_length_mean: 982.525
- final_episode_length_std:  1.805

### PPO – Episode Length
- final_episode_length_mean: 986.393
- final_episode_length_std:  4.631
