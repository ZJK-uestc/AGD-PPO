# Reward & Episode-Length Curve Plot Manifest

smooth_window: 25
mean_smoothing: moving average on cross-seed mean only
variance_smoothing: none; shaded band is raw per-iteration population std

## Run directories

### AGD-PPO
- results/anymal_c_flat/Drift/seed1
- results/anymal_c_flat/Drift/seed2
- results/anymal_c_flat/Drift/seed3
- results/anymal_c_flat/Drift/seed4
- results/anymal_c_flat/Drift/seed42

### PPO
- results/anymal_c_flat/PPO/seed1
- results/anymal_c_flat/PPO/seed2
- results/anymal_c_flat/PPO/seed3
- results/anymal_c_flat/PPO/seed4
- results/anymal_c_flat/PPO/seed42

## Final statistics (last 50 iterations)

### AGD-PPO – Reward
- final_reward_mean: 19.968
- final_reward_std:  0.180

### PPO – Reward
- final_reward_mean: 19.864
- final_reward_std:  0.172

### AGD-PPO – Episode Length
- final_episode_length_mean: 989.862
- final_episode_length_std:  2.363

### PPO – Episode Length
- final_episode_length_mean: 987.815
- final_episode_length_std:  3.055
