# Reward & Episode-Length Curve Plot Manifest

smooth_window: 25
mean_smoothing: moving average on cross-seed mean only
variance_smoothing: none; shaded band is raw per-iteration population std

## Run directories

### AGD-PPO
- results/a1/Drift/action_only_seed1
- results/a1/Drift/action_only_seed3
- results/a1/Drift/seed1
- results/a1/Drift/action_only_seed5
- results/a1/Drift/seed2

### PPO
- results/a1/PPO/seed1
- results/a1/PPO/seed2
- results/a1/PPO/seed3
- results/a1/PPO/seed6
- results/a1/PPO/seed44

## Final statistics (last 50 iterations)

### AGD-PPO – Reward
- final_reward_mean: 15.043
- final_reward_std:  0.198

### PPO – Reward
- final_reward_mean: 13.719
- final_reward_std:  0.571

### AGD-PPO – Episode Length
- final_episode_length_mean: 960.397
- final_episode_length_std:  6.275

### PPO – Episode Length
- final_episode_length_mean: 952.434
- final_episode_length_std:  5.079
