# Reward & Episode-Length Curve Plot Manifest

smooth_window: 25
mean_smoothing: moving average on cross-seed mean only
variance_smoothing: none; shaded band is raw per-iteration population std

## Run directories

### AGD-PPO
- results/anymal_c_rough/Drift/seed1
- results/anymal_c_rough/Drift/seed2
- results/anymal_c_rough/Drift/seed3
- results/anymal_c_rough/Drift/seed4
- results/anymal_c_rough/Drift/seed42

### PPO
- results/anymal_c_rough/PPO/1
- results/anymal_c_rough/PPO/2
- results/anymal_c_rough/PPO/3
- results/anymal_c_rough/PPO/4
- results/anymal_c_rough/PPO/5

## Final statistics (last 50 iterations)

### AGD-PPO – Reward
- final_reward_mean: 18.768
- final_reward_std:  0.279

### PPO – Reward
- final_reward_mean: 17.250
- final_reward_std:  0.133

### AGD-PPO – Episode Length
- final_episode_length_mean: 950.454
- final_episode_length_std:  6.215

### PPO – Episode Length
- final_episode_length_mean: 962.882
- final_episode_length_std:  3.088
