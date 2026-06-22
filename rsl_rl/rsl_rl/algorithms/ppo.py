# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.drifting import Drifting
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 use_drift=False,
                 drift_model_warmup_updates=300,
                 drift_actor_warmup_updates=400,
                 drift_actor_loss_coef=1e-3,
                 positive_adv_threshold=0.0,
                 min_positive_samples=32,
                 drift_step_size=0.1,
                 max_drift_velocity_norm=1.0,
                 max_drift_action_dist=1.5,
                 action_kernel_temperature=0.5,
                 advantage_temperature=2.0,
                 advantage_clip=3.0,
                 use_state_kernel=False,
                 state_kernel_temperature=1.0,
                 state_feature_mode="obs_norm",
                 use_multi_temperature=False,
                 action_kernel_temperatures=(0.3, 0.5, 1.0),
                 normalize_drift_field=False,
                 drift_field_norm_type="batch",
                 log_drift_debug=True,
                 drift_chunk_size=1024,
                 use_temperature_schedule=False,
                 action_kernel_temperature_start=0.5,
                 action_kernel_temperature_end=0.3,
                 action_kernel_temperature_schedule_start=400,
                 action_kernel_temperature_schedule_end=1000,
                 use_top_positive_filter=False,
                 positive_top_fraction=0.35,
                 use_residual_drift=False,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.use_drift = use_drift
        self.drift_model_warmup_updates = drift_model_warmup_updates
        self.drift_actor_warmup_updates = drift_actor_warmup_updates
        self.drift_actor_loss_coef = drift_actor_loss_coef
        self.drifting = None
        self.update_counter = 0
        self.drift_cfg = {
            "positive_adv_threshold": positive_adv_threshold,
            "min_positive_samples": min_positive_samples,
            "drift_step_size": drift_step_size,
            "max_drift_velocity_norm": max_drift_velocity_norm,
            "max_drift_action_dist": max_drift_action_dist,
            "action_kernel_temperature": action_kernel_temperature,
            "advantage_temperature": advantage_temperature,
            "advantage_clip": advantage_clip,
            "use_state_kernel": use_state_kernel,
            "state_kernel_temperature": state_kernel_temperature,
            "state_feature_mode": state_feature_mode,
            "use_multi_temperature": use_multi_temperature,
            "action_kernel_temperatures": list(action_kernel_temperatures),
            "normalize_drift_field": normalize_drift_field,
            "drift_field_norm_type": drift_field_norm_type,
            "log_drift_debug": log_drift_debug,
            "drift_chunk_size": drift_chunk_size,
            "use_temperature_schedule": use_temperature_schedule,
            "action_kernel_temperature_start": action_kernel_temperature_start,
            "action_kernel_temperature_end": action_kernel_temperature_end,
            "action_kernel_temperature_schedule_start": action_kernel_temperature_schedule_start,
            "action_kernel_temperature_schedule_end": action_kernel_temperature_schedule_end,
            "use_top_positive_filter": use_top_positive_filter,
            "positive_top_fraction": positive_top_fraction,
            "use_residual_drift": use_residual_drift,
        }

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)
        if self.use_drift:
            self.drifting = Drifting(
                **self.drift_cfg,
            )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_total_loss = 0
        mean_approx_kl = 0
        drift_sums = {
            "loss": 0.0,
            "effective_loss": 0.0,
            "positive_ratio": 0.0,
            "num_positive_samples": 0.0,
            "selected_positive_ratio": 0.0,
            "num_selected_positive_samples": 0.0,
            "skip": 0.0,
            "action_kernel_temperature": 0.0,
            "field_norm": 0.0,
            "field_norm_raw": 0.0,
            "field_norm_normalized": 0.0,
            "field_clip_ratio": 0.0,
            "center_dist": 0.0,
            "action_dist": 0.0,
            "state_dist": 0.0,
            "action_dist_to_pos": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
            "weight_entropy": 0.0,
            "mean_positive_adv": 0.0,
            "max_positive_adv": 0.0,
        }
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                entropy_loss = entropy_batch.mean()
                ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

                drift_logs = self._empty_drift_logs()
                drift_loss = mu_batch.sum() * 0.0
                drift_loss_coef = 0.0
                if self.use_drift and self.drifting is not None and self.update_counter >= self.drift_model_warmup_updates:
                    raw_advantages_batch = returns_batch - target_values_batch
                    drift_result = self.drifting.compute_loss(
                        obs=obs_batch,
                        actions=actions_batch,
                        raw_advantages=raw_advantages_batch,
                        actor_mean=mu_batch,
                        old_action_mean=old_mu_batch,
                        update_counter=self.update_counter,
                    )
                    drift_loss = drift_result["loss"]
                    drift_logs = drift_result["logs"]
                    if self.update_counter >= self.drift_actor_warmup_updates:
                        drift_loss_coef = self.drift_actor_loss_coef
                drift_logs["effective_loss"] = drift_loss_coef * drift_logs["loss"]

                loss = ppo_loss + drift_loss_coef * drift_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_entropy += entropy_loss.item()
                mean_total_loss += loss.item()
                mean_approx_kl += kl_mean.item()

                for key in drift_sums:
                    drift_sums[key] += drift_logs[key]

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_total_loss /= num_updates
        mean_approx_kl /= num_updates
        drift_logs = {key: value / num_updates for key, value in drift_sums.items()}
        self.storage.clear()
        self.update_counter += 1

        return mean_value_loss, mean_surrogate_loss, {
            "entropy": mean_entropy,
            "total_loss": mean_total_loss,
            "approx_kl": mean_approx_kl,
            "drift": drift_logs,
        }

    @staticmethod
    def _empty_drift_logs():
        return {
            "loss": 0.0,
            "effective_loss": 0.0,
            "positive_ratio": 0.0,
            "num_positive_samples": 0.0,
            "selected_positive_ratio": 0.0,
            "num_selected_positive_samples": 0.0,
            "skip": 0.0,
            "action_kernel_temperature": 0.0,
            "field_norm": 0.0,
            "field_norm_raw": 0.0,
            "field_norm_normalized": 0.0,
            "field_clip_ratio": 0.0,
            "center_dist": 0.0,
            "action_dist": 0.0,
            "state_dist": 0.0,
            "action_dist_to_pos": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
            "weight_entropy": 0.0,
            "mean_positive_adv": 0.0,
            "max_positive_adv": 0.0,
        }
