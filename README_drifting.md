方案：Original-style positive-only drifting field** 

```text
不再训练 DriftVelocityNet。
不再使用 tau 路径插值。
不再计算 pred_velocity 与 target_velocity 的 MSE。
直接根据当前 PPO batch 中 advantage > 0 的样本计算 positive drifting field V_pos。
然后构造 stop-gradient drifted target：
    actor_mean_drifted = stopgrad(actor_mean + drift_step_size * V_pos)
最后：
    actor_drift_loss = MSE(actor_mean, actor_mean_drifted)
```

原始 Drifting Models 的训练伪代码是：先计算 drifting field `V = compute_V(x, y_pos, y_neg)`，再构造 `x_drifted = stopgrad(x + V)`，最后用 `mse_loss(x - x_drifted)` 训练网络输出靠近漂移后的冻结目标；原文也说明这种 stop-gradient 目标避免了直接对分布相关的 (V) 反传。([arXiv][1])

---

# Positive-only Original-style Drifting-PPO for legged_gym / rsl_rl

## 1. Goal

本项目在原始 PPO 基础上加入一个 **positive-only drifting field auxiliary loss**。

核心思想：

```text
PPO 原始 clipped objective 保持不变。
Drifting 不再训练单独的 velocity model。
Drifting 只使用当前 PPO minibatch 中 advantage > 0 的样本。
Drifting 根据 positive samples 直接计算漂移场 V_pos。
Actor mean 被推向 stop-gradient 的 drifted target。
第一版不使用 advantage < 0 的负样本。
```

总损失形式：

```text
PPO 原始损失:
    L_ppo = L_actor + value_loss_coef * L_value - entropy_coef * H

加入 positive-only drifting 后:
    L_total = L_ppo + drift_actor_loss_coef * L_drift
```

其中：

```text
L_drift = MSE(actor_mean, stopgrad(actor_mean + drift_step_size * V_pos))
```


---

## 2. File Structure

新增文件：

```text
rsl_rl/rsl_rl/algorithms/drifting.py
```

保持与 `ppo.py` 同级：

```text
rsl_rl/rsl_rl/algorithms/
├── ppo.py
├── drifting.py
└── __init__.py
```

---

## 3. Overall Algorithm

```python
for iteration in range(max_iterations):

    rollout = collect_rollout_with_current_ppo_policy()

    compute_returns_and_advantages(rollout)

    for epoch in ppo_epochs:
        for batch in rollout.minibatches:

            # 1. PPO 原始损失
            ppo_loss = compute_ppo_loss(batch)

            # 2. 当前 actor 输出动作均值
            actor_mean = actor_critic.get_action_mean(batch.obs)

            # 3. positive-only drifting loss
            if use_drift and iteration >= drift_actor_warmup_updates:
                drift_loss, drift_logs = drifting.compute_loss(
                    obs=batch.obs,
                    actions=batch.actions,
                    advantages=batch.advantages,
                    actor_mean=actor_mean
                )
            else:
                drift_loss = 0

            # 4. 总损失
            total_loss = ppo_loss + drift_actor_loss_coef * drift_loss

            update_actor_critic(total_loss)
```

---

# Stage 1: Minimal Original-style Positive-only Drifting

## 1.1 Goal

第一阶段只实现最小可运行版本：

```text
1. 新建 drifting.py。
2. 不实现 DriftVelocityNet。
3. 不训练 drifting model。
4. 只使用当前 PPO minibatch 中 advantage > 0 的样本。
5. 直接计算 positive drifting field V_pos。
6. 构造 stop-gradient drifted target。
7. PPO total loss 中加入 drift_actor_loss。
8. warmup 暂定为 300。
9. 暂时不使用负样本。
10. 暂时不使用 state-conditioned kernel，只用 action-space kernel。
```

---

## 1.2 Config

在 `legged_robot_config.py` 或具体任务 config 的 `class algorithm` 中新增：

```python
use_drift = True

drift_actor_warmup_updates = 300
drift_actor_loss_coef = 0.001

positive_adv_threshold = 0.0
min_positive_samples = 32

drift_step_size = 0.1
max_drift_velocity_norm = 1.0
max_drift_action_dist = 1.5

action_kernel_temperature = 0.5
advantage_temperature = 2.0
advantage_clip = 3.0

use_state_kernel = False
state_kernel_temperature = 1.0

log_drift_debug = True
```

---

## 1.3 drifting.py Structure

文件：

```text
rsl_rl/rsl_rl/algorithms/drifting.py
```

伪代码：

```python
class Drifting:

    def __init__(
        self,
        positive_adv_threshold,
        min_positive_samples,
        drift_step_size,
        max_drift_velocity_norm,
        max_drift_action_dist,
        action_kernel_temperature,
        advantage_temperature,
        advantage_clip,
        use_state_kernel=False,
        state_kernel_temperature=1.0,
    ):
        save_all_config()
```

本阶段 `Drifting` 不是神经网络，不继承 `nn.Module` 也可以。

---

## 1.4 Drifting.compute_loss()

### 核心公式

令：

```text
x_i = actor_mean_i = μθ(s_i)
positive actions = {a_j | A_j > threshold}
```

positive drifting field：

```text
V_pos_i = Σ_j w_ij * (a_j - x_i)
```

其中：

```text
w_ij = softmax(
    - ||x_i - a_j|| / action_temperature
    + clipped_adv_j / advantage_temperature
)
```

drifted target：

```text
x_drifted_i = stopgrad(x_i + drift_step_size * V_pos_i)
```

drift loss：

```text
L_drift = mean(||x_i - x_drifted_i||²)
```

---

### Pseudocode

```python
def compute_loss(obs, actions, advantages, actor_mean):

    # 1. detach rollout data
    obs_detached = detach(obs)
    actions_detached = detach(actions)
    advantages_detached = detach(advantages)

    # 2. normalize advantage
    adv = normalize(advantages_detached)

    # 3. select positive samples
    pos_mask = adv > positive_adv_threshold

    if num_positive_samples < min_positive_samples:
        return zero_loss, skip_logs

    obs_pos = obs_detached[pos_mask]
    act_pos = actions_detached[pos_mask]
    adv_pos = adv[pos_mask]

    # 4. advantage weight logits
    adv_weight = clamp(adv_pos, min=0, max=advantage_clip)

    # 5. compute action distance
    # actor_mean: [B, action_dim]
    # act_pos:    [P, action_dim]
    action_dist = cdist(actor_mean.detach(), act_pos)

    # 6. kernel logits
    logits = (
        - action_dist / action_kernel_temperature
        + adv_weight.unsqueeze(0) / advantage_temperature
    )

    # 7. softmax weights over positive samples
    weights = softmax(logits, dim=-1)

    # 8. positive center
    center_pos = weights @ act_pos

    # 9. positive drifting field
    V_pos = center_pos - actor_mean.detach()

    # 10. clip V_pos by norm
    V_pos = clip_by_norm(V_pos, max_drift_velocity_norm)

    # 11. drifted target
    actor_mean_drifted = actor_mean.detach() + drift_step_size * V_pos
    actor_mean_drifted = clip(actor_mean_drifted, -1, 1)

    # 12. limit target distance from current actor mean
    actor_mean_drifted = limit_distance(
        source=actor_mean.detach(),
        target=actor_mean_drifted,
        max_distance=max_drift_action_dist
    )

    # 13. original-style stop-gradient target
    actor_mean_drifted = stopgrad(actor_mean_drifted)

    # 14. drift loss
    drift_loss_per_sample = mse(actor_mean, actor_mean_drifted)
    drift_loss = mean(drift_loss_per_sample)

    # 15. logs
    logs = compute_logs(
        positive_ratio,
        drift_loss,
        V_pos,
        center_pos,
        actor_mean,
        actor_mean_drifted,
        weights
    )

    return drift_loss, logs
```

---

## 1.5 PPO.update() Integration

在 `ppo.py` 中：

```python
for minibatch in generator:

    # 1. 原始 PPO 计算
    new_log_prob, entropy, value, actor_mean, actor_std = evaluate_actions(batch)

    ratio = exp(new_log_prob - old_log_prob)

    surrogate_loss = compute_clipped_surrogate_loss(ratio, advantages)

    value_loss = compute_value_loss(value, returns)

    ppo_loss = (
        surrogate_loss
        + value_loss_coef * value_loss
        - entropy_coef * entropy
    )

    # 2. original-style positive-only drifting loss
    if use_drift and update_counter >= drift_actor_warmup_updates:
        drift_loss, drift_logs = drifting.compute_loss(
            obs=obs_batch,
            actions=actions_batch,
            advantages=advantages_batch,
            actor_mean=actor_mean
        )
    else:
        drift_loss = actor_mean.sum() * 0.0
        drift_logs = empty_drift_logs()

    # 3. total loss
    total_loss = ppo_loss + drift_actor_loss_coef * drift_loss

    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm(actor_critic.parameters)
    optimizer.step()
```

注意：

```text
不要创建 drift optimizer。
不要更新任何 drifting model。
不要把 drift_loss 单独 backward。
drift_loss 只通过 PPO optimizer 更新 actor。
```

---

## 1.6 Stage 1 Test Metrics

TensorBoard 至少输出：

```text
Drift/loss
Drift/effective_loss
Drift/positive_ratio
Drift/num_positive_samples
Drift/skip
Drift/field_norm
Drift/center_dist
Drift/action_dist
Drift/max_weight
Drift/weight_entropy
```

其中：

```text
Drift/effective_loss = drift_actor_loss_coef * Drift/loss
Drift/field_norm = mean(||V_pos||)
Drift/center_dist = mean(||center_pos - actor_mean||)
Drift/action_dist = mean(||actor_mean_drifted - actor_mean||)
Drift/weight_entropy = softmax 权重熵，用于判断是否只盯着极少数 positive samples
```

---

## 1.7 Stage 1 Acceptance Criteria

第一阶段通过标准：

```text
1. use_drift = False 时，PPO 训练行为与原始版本一致。
2. use_drift = True 时，训练能正常启动。
3. update < 300 时，Drift/loss = 0 或不参与 total loss。
4. update >= 300 后，Drift/loss 开始出现。
5. Drift/positive_ratio 不长期为 0。
6. Drift/loss 不出现 NaN。
7. Drift/effective_loss 明显小于 PPO surrogate loss。
8. PPO reward 不应在 warmup 后突然坍塌。
```

建议阈值：

```text
Drift/effective_loss < 0.1 * abs(Loss/surrogate_loss)

Drift/action_dist < max_drift_action_dist

Drift/field_norm 不应长期顶到 max_drift_velocity_norm

Drift/weight_entropy 不应过低，否则说明 drift target 只由少数极端样本决定
```

---

# Stage 2: State-conditioned Kernel and Stable Drift Field

## 2.1 Goal

第二阶段增强 positive drifting field 的质量，而不是重新引入 velocity network。

新增：

```text
1. state-conditioned kernel。
2. advantage 加权稳定化。
3. drift field normalization。
4. multi-temperature kernel 可选。
5. drift logs 做 batch 平均。
```

---

## 2.2 Additional Config

```python
use_state_kernel = True
state_kernel_temperature = 1.0
state_feature_mode = "obs_norm"

use_multi_temperature = False
action_kernel_temperatures = [0.3, 0.5, 1.0]

normalize_drift_field = True
drift_field_norm_type = "batch"

log_drift_debug = True
```

---

## 2.3 State-conditioned Kernel

第一阶段只用动作距离：

```text
logits = - ||actor_mean_i - act_pos_j|| / T_a + adv_j / T_adv
```

第二阶段加入状态距离：

```text
logits =
    - ||actor_mean_i - act_pos_j|| / T_a
    - ||f(obs_i) - f(obs_pos_j)|| / T_s
    + adv_j / T_adv
```

伪代码：

```python
def extract_state_feature(obs):

    if state_feature_mode == "obs_norm":
        feat = normalize_by_batch(obs)

    elif state_feature_mode == "selected":
        feat = select_low_dim_robot_features(obs)
        feat = normalize_by_batch(feat)

    return feat
```

然后：

```python
feat = extract_state_feature(obs)
feat_pos = extract_state_feature(obs_pos)

state_dist = cdist(feat.detach(), feat_pos.detach())

logits = (
    - action_dist / action_kernel_temperature
    - state_dist / state_kernel_temperature
    + adv_weight.unsqueeze(0) / advantage_temperature
)
```

---

## 2.4 Drift Field Normalization

为避免 `V_pos` 过大或长期被 hard clip，增加 batch-level normalization：

```python
def normalize_drift_field(V_pos):

    if normalize_drift_field:
        norm_mean = mean(norm(V_pos))
        V_pos = V_pos / (norm_mean + 1e-8)

    V_pos = clip_by_norm(V_pos, max_drift_velocity_norm)

    return V_pos
```

注意：

```text
normalize 不是替代 clip。
normalize 用来稳定尺度。
clip 用来做安全上限。
```

---

## 2.5 Multi-temperature Kernel

可选实现：

```python
def compute_multi_temperature_field(actor_mean, act_pos, adv_pos):

    fields = []

    for T_a in action_kernel_temperatures:
        logits = (
            - action_dist / T_a
            - state_dist / state_kernel_temperature
            + adv_weight / advantage_temperature
        )

        weights = softmax(logits, dim=-1)
        center_pos = weights @ act_pos
        V_pos = center_pos - actor_mean.detach()

        fields.append(V_pos)

    V_pos = mean(fields)

    return V_pos
```

第一版 Stage 2 可以先不打开：

```python
use_multi_temperature = False
```

---

## 2.6 Batch-average Drift Logs

PPO 一次 update 内有多个 minibatch，不要只记录最后一个 minibatch。

伪代码：

```python
drift_log_sum = {}
drift_log_count = 0

for minibatch in generator:

    drift_loss, drift_logs = drifting.compute_loss(...)

    for key, value in drift_logs.items():
        drift_log_sum[key] = drift_log_sum.get(key, 0.0) + float(value)

    drift_log_count += 1

for key in drift_log_sum:
    drift_log_sum[key] /= max(drift_log_count, 1)

self.last_drift_logs = drift_log_sum
```

---

## 2.7 Stage 2 Test Metrics

新增日志：

```text
Drift/state_dist
Drift/action_dist_to_pos
Drift/field_norm_raw
Drift/field_norm_normalized
Drift/field_clip_ratio
Drift/weight_entropy
Drift/max_weight
Drift/min_weight
Drift/mean_positive_adv
Drift/max_positive_adv
```

建议同时记录 PPO 损失：

```text
Loss/surrogate_loss
Loss/value_loss
Loss/entropy
Loss/total_loss
Loss/drift_effective_loss
```

---

## 2.8 Stage 2 Acceptance Criteria

第二阶段通过标准：

```text
1. Drift/loss 不爆炸。
2. Drift/field_clip_ratio 不应长期接近 1。
3. Drift/action_dist 不应长期顶到 max_drift_action_dist。
4. Drift/weight_entropy 不应过低。
5. warmup=300 后 PPO reward 不应明显低于 baseline。
6. state kernel 打开后，PPO reward 至少不比 action-only kernel 更差。
```

建议判断：

```text
如果 Drift/field_clip_ratio > 0.8：
    说明 V_pos 经常被裁剪，max_drift_velocity_norm 可能太小或 kernel target 太激进。

如果 Drift/action_dist 长期等于 max_drift_action_dist：
    说明 drift_step_size 过大或 V_pos 过强。

如果 Drift/weight_entropy 很低：
    说明 softmax 只关注少量极端正样本，可以提高 action_kernel_temperature 或 advantage_temperature。

如果 warmup 后 PPO reward 断崖下降：
    降低 drift_actor_loss_coef 或 drift_step_size。
```

---

# Stage 3: Stable Experiment Version

## 3.1 Goal

第三阶段把代码整理成可用于论文实验的稳定版本。

新增：

```text
1. 完整开关。
2. 完整日志。
3. 消融实验配置。
4. 安全限制。
5. 多 seed 实验准备。
6. 与原始 PPO 的可复现对比。
```

注意：

```text
由于方案 B 没有 drift model 参数，因此不需要保存 drift_model_state_dict。
只需要把 drift 配置写入日志或保存到 config。
```

---

## 3.2 Final Config

```python
# main switch
use_drift = True

# warmup
drift_actor_warmup_updates = 300

# loss weight
drift_actor_loss_coef = 0.001

# positive samples
positive_adv_threshold = 0.0
min_positive_samples = 64
advantage_clip = 3.0
advantage_temperature = 2.0

# kernel
use_state_kernel = True
state_feature_mode = "obs_norm"
state_kernel_temperature = 1.0
action_kernel_temperature = 0.5

# optional multi-temperature
use_multi_temperature = False
action_kernel_temperatures = [0.3, 0.5, 1.0]

# drift field
drift_step_size = 0.1
normalize_drift_field = True
max_drift_velocity_norm = 1.0
max_drift_action_dist = 1.5

# debug
log_drift_debug = True
```

---

## 3.3 Full Logging Metrics

PPO 侧：

```text
Train/mean_reward
Train/episode_length
Loss/surrogate_loss
Loss/value_loss
Loss/entropy
Loss/total_loss
Loss/learning_rate
Policy/mean_std
Policy/approx_kl
```

Drifting 侧：

```text
Drift/loss
Drift/effective_loss
Drift/positive_ratio
Drift/num_positive_samples
Drift/skip

Drift/field_norm
Drift/field_norm_raw
Drift/field_norm_normalized
Drift/field_clip_ratio

Drift/center_dist
Drift/action_dist
Drift/action_dist_to_pos

Drift/weight_entropy
Drift/max_weight
Drift/min_weight

Drift/mean_positive_adv
Drift/max_positive_adv
```

机器人任务侧建议额外记录：

```text
Task/forward_vel_error
Task/yaw_vel_error
Task/fall_rate
Task/episode_length
Task/terrain_level
Task/foot_slip
Task/torque_mean
Task/action_rate
```

---

## 3.4 Ablation Settings

### A. PPO Baseline

```python
use_drift = False
```

测试目的：

```text
确认原始 PPO baseline。
```

---

### B. Positive-only Drift, action kernel only

```python
use_drift = True
use_state_kernel = False
drift_actor_loss_coef = 0.001
drift_step_size = 0.1
drift_actor_warmup_updates = 300
```

测试目的：

```text
验证最基本的 action-space positive drifting 是否有效。
```

---

### C. Positive-only Drift, state-conditioned kernel

```python
use_drift = True
use_state_kernel = True
drift_actor_loss_coef = 0.001
drift_step_size = 0.1
drift_actor_warmup_updates = 300
```

测试目的：

```text
验证状态条件 kernel 是否减少错误动作匹配。
```

---

### D. Stronger Drift Weight

```python
use_drift = True
use_state_kernel = True
drift_actor_loss_coef = 0.005
drift_step_size = 0.1
drift_actor_warmup_updates = 300
```

测试目的：

```text
验证更强 drift 权重是否提升收敛或破坏 PPO 稳定性。
```

---

### E. Larger Drift Step

```python
use_drift = True
use_state_kernel = True
drift_actor_loss_coef = 0.001
drift_step_size = 0.2
drift_actor_warmup_updates = 300
```

测试目的：

```text
验证 drift_step_size 对动作漂移幅度和稳定性的影响。
```

---

### F. Multi-temperature Drift

```python
use_drift = True
use_state_kernel = True
use_multi_temperature = True
action_kernel_temperatures = [0.3, 0.5, 1.0]
drift_actor_loss_coef = 0.001
drift_step_size = 0.1
```

测试目的：

```text
验证多温度 kernel 是否提高 drift field 鲁棒性。
```

---

## 3.5 Stage 3 Acceptance Criteria

第三阶段通过标准：

```text
1. 所有 ablation 都能正常运行。
2. use_drift=False 时完全退化为原始 PPO。
3. Positive-only Drift 不出现 reward 断崖下降。
4. Drift/effective_loss 不主导 PPO loss。
5. Drift/field_clip_ratio 不长期接近 1。
6. Drift/action_dist 不长期顶到 max_drift_action_dist。
7. 至少 3 个 seed 可以稳定复现实验趋势。
8. state-conditioned kernel 的稳定性应优于 action-only kernel。
```

建议数值监控：

```text
Drift/effective_loss < 0.1 * abs(Loss/surrogate_loss)

Drift/action_dist < max_drift_action_dist

Drift/field_clip_ratio < 0.5 更理想

Drift/weight_entropy 不应长期过低

warmup 后 100 个 iteration 内 reward 不应持续下降
```

---

# Implementation Notes for Codex

## Do

```text
1. drifting 代码必须单独放在 drifting.py。
2. PPO 原始 loss 不删除。
3. 不创建 DriftVelocityNet。
4. 不创建 drift optimizer。
5. 只计算 positive-only drifting field。
6. 只把 drift_loss 加入 PPO total loss。
7. 第一版只使用 advantage > 0 的正样本。
8. warmup 默认 300。
9. 所有 drift 指标写入 TensorBoard。
10. drift logs 在一个 PPO update 内做 batch 平均。
```

---

## Do Not

```text
1. 不要使用 advantage < 0 的负样本。
5. 不要让 drifting 与环境交互。
8. 不要实现 tau 路径插值。
9. 不要把 drift 写进 PPO 原始 surrogate loss 内部。
```

---

# Test Commands

小规模调试：

```bash
cd /home/zjk/zjk/legged_gym   # 进入 legged_gym 项目目录
```

```bash
python legged_gym/scripts/train.py --task a1 --headless --num_envs 256 --max_iterations 50   # 小规模快速检查 drifting 代码是否能跑通
```

正式训练：

```bash
python legged_gym/scripts/train.py --task a1 --headless --num_envs 4096 --max_iterations 1500   # 正式训练 A1，并记录 PPO+Drift 曲线
```

查看日志：

```bash
tensorboard --logdir logs   # 打开 TensorBoard 查看 PPO 和 Drifting 指标
```

---

# Final Expected Behavior

最终期望结果：

```text
PPO 仍然是主训练算法。
Drifting 不再训练额外速度网络。
Drifting 直接从当前 PPO batch 中 advantage > 0 的样本计算 positive drifting field。
Actor mean 被推向 stop-gradient 的 drifted target。
整个方法不使用负样本，不改变 PPO clipped objective，只提供额外的 positive-only action-space policy guidance。
```

一句话总结：

```text
This implementation adds an original-style positive-only drifting field to PPO. It directly computes a kernel-weighted drift field from positive-advantage actions and applies a small auxiliary loss that moves the actor mean toward a stop-gradient drifted target after a warmup period.
```

[1]: https://arxiv.org/html/2602.04770v1 "Generative Modeling via Drifting"
