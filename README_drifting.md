# Drifting 模块说明

本文档总结当前仓库中 AGD-PPO / Drifting 模块的工作原理、目标函数、迭代更新方式和主要超参数。当前实现对应的是 **positive-only drifting field** 版本：它不训练额外的 DriftVelocityNet，也不引入单独的 drift optimizer，而是直接从 PPO mini-batch 中的正优势样本构造动作空间漂移目标。

相关源码位置：

- Drifting 计算模块：`/home/zjk/zjk/rsl_rl/rsl_rl/algorithms/drifting.py`
- PPO 集成位置：`/home/zjk/zjk/rsl_rl/rsl_rl/algorithms/ppo.py`
- 默认训练超参数：`legged_gym/envs/base/legged_robot_config.py`
- 命令行覆盖参数：`legged_gym/utils/helpers.py`

## 1. 核心思想

标准 PPO 通过 advantage 调整采样动作的概率：正 advantage 的动作概率被提高，负 advantage 的动作概率被压低。这个机制是有效的，但它主要作用在概率似然上，并没有显式告诉 actor mean 应该朝动作空间中的哪个方向移动。

Drifting 模块的作用是给 actor 提供一个额外的动作空间引导：

1. 从当前 PPO mini-batch 中选择 raw advantage 大于阈值的正样本。
2. 根据动作距离、可选状态距离和 advantage 权重，为每个当前 actor mean 找到一组相近且表现较好的正样本动作。
3. 用这些正样本构造一个 positive drifting field。
4. 把 actor mean 向该 drifting field 方向移动一小步，得到 stop-gradient 的 drift target。
5. 用 MSE 辅助损失把当前 actor mean 拉向这个 drift target。

因此，Drifting 并不替代 PPO 的 clipped objective，而是在 PPO loss 外增加一个较小权重的辅助项，使 actor 更新更有方向性。

## 2. 输入与正样本筛选

在每个 PPO mini-batch 中，Drifting 使用以下输入：

- `obs`：当前 mini-batch 的 actor observation。
- `actions`：rollout 时由旧策略采样并实际执行的动作。
- `raw_advantages`：当前实现中由 `returns_batch - target_values_batch` 得到。
- `actor_mean`：当前策略网络对 `obs` 输出的动作均值。
- `old_action_mean`：rollout 时旧策略的动作均值，用于可选 residual drift。

正样本集合定义为：

```text
B_pos = {i | raw_advantage_i > positive_adv_threshold}
```

如果正样本数量小于 `min_positive_samples`，当前 mini-batch 会跳过 drift，返回零 drift loss。若启用 `use_top_positive_filter`，则只保留正样本中 raw advantage 最高的一部分，比例由 `positive_top_fraction` 控制。

## 3. Drifting Field 的计算

对每个当前 actor mean `mu_i`，模块根据正样本动作构造一个加权中心或残差方向。

### 3.1 Advantage 权重

正样本 advantage 先截断为非负值，再按均值归一化，最后做上界裁剪：

```text
w_j = clamp(max(A_j, 0) / mean(max(A_pos, 0)), 0, advantage_clip)
```

这个权重会进入 softmax logits，使高 advantage 的正样本更容易影响 drift target。

### 3.2 Action-space kernel

默认情况下，kernel 只基于当前 actor mean 与正样本动作之间的欧氏距离：

```text
logit_ij = -||mu_i - a_j^+|| / T_a + w_j / T_A
alpha_ij = softmax_j(logit_ij)
```

其中：

- `T_a` 是 `action_kernel_temperature`。
- `T_A` 是 `advantage_temperature`。
- `alpha_ij` 是当前样本 `i` 对正样本 `j` 的归一化权重。

若启用温度调度，`T_a` 会从 `action_kernel_temperature_start` 线性变化到 `action_kernel_temperature_end`。若启用 multi-temperature，则会对多个 action temperature 分别计算 drifting field，再取平均。

### 3.3 State-conditioned kernel

如果 `use_state_kernel=True`，softmax logits 中会额外加入状态距离项：

```text
logit_ij =
    -||mu_i - a_j^+|| / T_a
    -||f(s_i) - f(s_j^+)|| / T_s
    + w_j / T_A
```

其中 `f(s)` 是 batch-normalized observation feature，`T_s` 是 `state_kernel_temperature`。这个项用于减少“动作相近但状态不相似”的错误匹配。

### 3.4 Absolute drift 与 residual drift

当前实现支持两种构造漂移方向的方式。

默认 absolute drift：

```text
c_i = sum_j alpha_ij a_j^+
v_i = c_i - mu_i
```

可选 residual drift：

```text
r_j = a_j^+ - old_mu_j
v_i = sum_j alpha_ij r_j
c_i = mu_i + v_i
```

当 `use_residual_drift=True` 且 `old_action_mean` 可用时，会使用 residual drift。它学习的是“旧策略均值到正样本动作的残差”，通常比直接追逐绝对动作更保守。

## 4. Drift Target 与目标函数

得到 drifting field `v_i` 后，模块先做范数裁剪：

```text
v_i <- clip_by_norm(v_i, max_drift_velocity_norm)
```

然后构造漂移步长：

```text
delta_i = drift_step_size * v_i
delta_i <- clip_by_norm(delta_i, max_drift_action_dist)
```

最终 drift target 为：

```text
mu_i^drift = stop_gradient(mu_i + delta_i)
```

注意：当前实现不会把 drift target clamp 到 `[-1, 1]`，只通过 `max_drift_velocity_norm` 和 `max_drift_action_dist` 限制局部漂移幅度。

Drifting 辅助损失为：

```text
L_drift = mean_i ||mu_i - mu_i^drift||_2^2
```

PPO 原始目标保持不变：

```text
L_PPO = L_surrogate + value_loss_coef * L_value - entropy_coef * H
```

最终优化目标为：

```text
L_total = L_PPO + drift_actor_loss_coef * L_drift
```

其中 drift target 是 stop-gradient 的，因此 `L_drift` 只会通过当前 actor mean 更新 actor 网络，不会反向传播到正样本动作、旧策略均值或 drifting field 的计算图中。

## 5. 更新与迭代流程

整体训练流程仍然遵循 PPO：

1. 使用当前 actor-critic 与环境交互，收集 rollout。
2. 根据 rollout 计算 returns 和 advantages。
3. 将 rollout 切成多个 mini-batch。
4. 对每个 mini-batch 计算 PPO surrogate loss、value loss 和 entropy loss。
5. 若启用 `use_drift` 且当前 update 达到 warmup 条件，则计算 `L_drift`。
6. 用 `L_total` 做一次反向传播，并使用 PPO 原有 optimizer 更新 actor-critic。
7. 一个 PPO update 内所有 mini-batch 完成后，清空 storage，并将 `update_counter` 加一。

当前实现有两个 warmup：

- `drift_model_warmup_updates`：达到该 update 后才开始计算 drift loss 和 drift logs。
- `drift_actor_warmup_updates`：达到该 update 后 drift loss 才真正乘以 `drift_actor_loss_coef` 加入总损失。

因为当前版本没有单独 drift model，所以 `drift_model_warmup_updates` 更准确地说是 drift computation/logging 的起始点，而不是额外网络的训练起点。

## 6. 主要超参数

默认设置在 `LeggedRobotCfgPPO.algorithm` 中。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `use_drift` | `True` | 是否启用 Drifting。PPO baseline 应设为 `False`。 |
| `drift_model_warmup_updates` | `300` | 开始计算 drift 的 update。当前实现没有 drift model，仅作为 drift 计算 warmup。 |
| `drift_actor_warmup_updates` | `400` | drift loss 开始影响 actor 的 update。 |
| `drift_actor_loss_coef` | `0.001` | drift loss 在总损失中的权重。 |
| `positive_adv_threshold` | `0.0` | raw advantage 大于该阈值才作为正样本。 |
| `min_positive_samples` | `64` | 每个 mini-batch 至少需要的正样本数量，不足则跳过。 |
| `use_top_positive_filter` | `False` | 是否只使用正样本中 advantage 较高的一部分。 |
| `positive_top_fraction` | `0.35` | top-positive filter 保留比例。 |
| `drift_step_size` | `0.1` | 沿 drifting field 移动的步长。 |
| `max_drift_velocity_norm` | `1.0` | drift field 的最大范数。 |
| `max_drift_action_dist` | `1.5` | drift target 相对当前 actor mean 的最大距离。 |
| `drift_chunk_size` | `1024` | 分块计算 kernel，降低显存峰值。 |
| `use_residual_drift` | `False` | 是否使用 `action - old_mu` 作为正样本残差方向。 |
| `action_kernel_temperature` | `0.3` | action-space kernel 温度。越小越偏向最近正样本。 |
| `use_temperature_schedule` | `True` | 是否线性调度 action kernel 温度。 |
| `action_kernel_temperature_start` | `0.5` | 温度调度起始值。 |
| `action_kernel_temperature_end` | `0.3` | 温度调度终止值。 |
| `action_kernel_temperature_schedule_start` | `400` | 温度调度开始 update。 |
| `action_kernel_temperature_schedule_end` | `1000` | 温度调度结束 update。 |
| `advantage_temperature` | `2.0` | advantage logits 温度。越小越强调高 advantage 样本。 |
| `advantage_clip` | `3.0` | advantage 权重上限，防止极端样本主导。 |
| `use_state_kernel` | `False` | 是否把状态距离加入 kernel。 |
| `state_kernel_temperature` | `0.5` | state kernel 温度。 |
| `state_feature_mode` | `"obs_norm"` | 状态特征模式，默认使用 batch-normalized 全观测。 |
| `use_multi_temperature` | `False` | 是否使用多个 action kernel 温度并平均。 |
| `action_kernel_temperatures` | `[0.3, 0.5, 1.0]` | multi-temperature 模式下使用的温度列表。 |
| `normalize_drift_field` | `False` | 旧配置兼容项；当前实现不再放大弱 drift field。 |
| `drift_field_norm_type` | `"batch"` | 旧配置兼容项；当前实现中不再实际使用。 |
| `log_drift_debug` | `True` | 是否记录详细 drift 调试指标。 |

## 7. 日志指标

PPO 更新时会对 mini-batch 内的 drift logs 做平均，常用指标包括：

| 指标 | 含义 |
| --- | --- |
| `loss` | 原始 drift MSE loss。 |
| `effective_loss` | `drift_actor_loss_coef * loss`，即实际加入总损失的大小。 |
| `positive_ratio` | raw advantage 大于阈值的样本比例。 |
| `num_positive_samples` | mini-batch 中正样本数量。 |
| `selected_positive_ratio` | top-positive filter 后实际选中样本比例。 |
| `num_selected_positive_samples` | 实际参与 drift 的正样本数量。 |
| `skip` | 是否因正样本不足跳过 drift。 |
| `action_kernel_temperature` | 当前 update 使用的 action kernel 温度。 |
| `field_norm` | 裁剪后 drift field 的平均范数。 |
| `field_norm_raw` | 裁剪前 drift field 的平均范数。 |
| `field_clip_ratio` | drift field 被范数裁剪的样本比例。 |
| `center_dist` | kernel 正样本中心到当前 actor mean 的距离。 |
| `action_dist` | 最终 drift delta 的平均范数。 |
| `state_dist` | state kernel 中的平均状态距离。 |
| `action_dist_to_pos` | 当前 actor mean 到正样本动作的平均距离。 |
| `max_weight` / `min_weight` | softmax 权重的最大值和最小值均值。 |
| `weight_entropy` | softmax 权重熵，用于判断 drift 是否被少数正样本主导。 |
| `mean_positive_adv` / `max_positive_adv` | 参与 drift 的正样本 advantage 统计。 |

## 8. 调参建议

通常先保证 PPO baseline 正常，再开启 Drifting。若训练不稳定，优先降低：

- `drift_actor_loss_coef`
- `drift_step_size`
- `max_drift_action_dist`

若 `field_clip_ratio` 长期很高，说明 drifting field 经常被裁剪，可能需要增大 `max_drift_velocity_norm`，或提高 action/state kernel 温度让权重更平滑。若 `weight_entropy` 长期很低，说明 drift target 由极少数正样本主导，可以提高 `action_kernel_temperature` 或 `advantage_temperature`，也可以关闭过强的 top-positive filter。

一个较保守的启动配置是：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --headless \
  --use_drift True \
  --drift_actor_loss_coef 0.001 \
  --drift_step_size 0.1 \
  --drift_model_warmup_updates 300 \
  --drift_actor_warmup_updates 400
```

如果显存不足，可以减小：

```bash
--drift_chunk_size 512
```

## 9. 与论文表述的对应关系

论文中可将当前模块概括为一种 advantage-guided drift-assisted PPO。需要注意的是，当前代码实现不是独立训练一个神经网络形式的 drift model，而是直接用当前 PPO mini-batch 的 positive-advantage samples 计算 kernel-weighted drifting field。因此，更准确的实现描述是：

> The drifting module computes a positive-advantage, kernel-weighted action-space drift target from each PPO mini-batch and applies it as a small stop-gradient auxiliary actor loss.

这句话既保留了 drifting 的核心思想，也与当前源码实现一致。
