import math

import torch


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
        positive_top_fraction=0.5,
        use_residual_drift=False,
    ):
        self.positive_adv_threshold = positive_adv_threshold
        self.min_positive_samples = min_positive_samples
        self.drift_step_size = drift_step_size
        self.max_drift_velocity_norm = max_drift_velocity_norm
        self.max_drift_action_dist = max_drift_action_dist
        self.action_kernel_temperature = action_kernel_temperature
        self.advantage_temperature = advantage_temperature
        self.advantage_clip = advantage_clip
        self.use_state_kernel = use_state_kernel
        self.state_kernel_temperature = state_kernel_temperature
        self.state_feature_mode = state_feature_mode
        self.use_multi_temperature = use_multi_temperature
        self.action_kernel_temperatures = tuple(action_kernel_temperatures)
        self.normalize_drift_field = normalize_drift_field
        self.drift_field_norm_type = drift_field_norm_type
        self.log_drift_debug = log_drift_debug
        self.drift_chunk_size = drift_chunk_size
        self.use_temperature_schedule = use_temperature_schedule
        self.action_kernel_temperature_start = action_kernel_temperature_start
        self.action_kernel_temperature_end = action_kernel_temperature_end
        self.action_kernel_temperature_schedule_start = action_kernel_temperature_schedule_start
        self.action_kernel_temperature_schedule_end = action_kernel_temperature_schedule_end
        self.use_top_positive_filter = use_top_positive_filter
        self.positive_top_fraction = positive_top_fraction
        self.use_residual_drift = use_residual_drift

    def compute_loss(self, obs, actions, raw_advantages, actor_mean, old_action_mean=None, update_counter=None):
        obs_detached = self._flatten_batch(obs).detach()
        actions_detached = self._flatten_batch(actions).detach()
        raw_advantages_detached = self._flatten_advantages(raw_advantages).detach()
        actor_mean_flat = self._flatten_batch(actor_mean)
        actor_mean_detached = actor_mean_flat.detach()
        old_action_mean_detached = None
        if old_action_mean is not None:
            old_action_mean_detached = self._flatten_batch(old_action_mean).detach()

        pos_mask = raw_advantages_detached > self.positive_adv_threshold
        positive_ratio = pos_mask.float().mean().item() if pos_mask.numel() > 0 else 0.0
        num_positive = int(pos_mask.sum().item())

        zero_loss = actor_mean_flat.sum() * 0.0
        if num_positive < self.min_positive_samples:
            return {
                "loss": zero_loss,
                "logs": self._empty_logs(
                    positive_ratio=positive_ratio,
                    num_positive_samples=float(num_positive),
                    skip=1.0,
                ),
            }

        selected_mask = pos_mask.clone()
        if self.use_top_positive_filter:
            positive_indices = torch.nonzero(pos_mask, as_tuple=False).flatten()
            num_selected = self._num_top_positive_samples(num_positive)
            _, top_local_indices = torch.topk(raw_advantages_detached[positive_indices], k=num_selected)
            selected_mask = torch.zeros_like(pos_mask)
            selected_mask[positive_indices[top_local_indices]] = True

        num_selected_positive = int(selected_mask.sum().item())
        obs_pos = obs_detached[selected_mask]
        act_pos = actions_detached[selected_mask]
        adv_pos = raw_advantages_detached[selected_mask]
        old_mu_pos = old_action_mean_detached[selected_mask] if old_action_mean_detached is not None else None
        use_residual_drift = self.use_residual_drift and old_mu_pos is not None
        kernel_pos = old_mu_pos if use_residual_drift else act_pos
        residual_pos = act_pos - old_mu_pos if use_residual_drift else None
        adv_weight = torch.clamp(adv_pos, min=0.0)
        adv_weight = adv_weight / adv_weight.mean().clamp_min(1e-8)
        adv_weight = torch.clamp(adv_weight, min=0.0, max=self.advantage_clip)
        feat = None
        feat_pos = None
        if self.use_state_kernel:
            feat = self._extract_state_feature(obs_detached)
            feat_pos = feat[selected_mask]

        current_temperature = self._scheduled_action_temperature(update_counter)
        temperatures = self.action_kernel_temperatures if self.use_multi_temperature else (current_temperature,)
        chunk_size = max(1, int(self.drift_chunk_size))
        target_chunks = []
        log_sums = {
            "field_norm": 0.0,
            "field_norm_raw": 0.0,
            "field_clip_ratio": 0.0,
            "center_dist": 0.0,
            "action_dist": 0.0,
            "state_dist": 0.0,
            "action_dist_to_pos": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
            "weight_entropy": 0.0,
        }
        num_samples = actor_mean_detached.shape[0]

        with torch.no_grad():
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                actor_chunk = actor_mean_detached[start:end]
                action_dist = torch.cdist(actor_chunk, kernel_pos, p=2)
                action_dist_to_pos = action_dist.mean()

                state_dist = None
                state_dist_value = action_dist.new_tensor(0.0)
                if self.use_state_kernel:
                    state_dist = torch.cdist(feat[start:end].detach(), feat_pos.detach(), p=2)
                    state_dist_value = state_dist.mean()

                field_sum = torch.zeros_like(actor_chunk)
                center_sum = torch.zeros_like(actor_chunk)
                max_weight_sum = action_dist.new_tensor(0.0)
                min_weight_sum = action_dist.new_tensor(0.0)
                entropy_sum = action_dist.new_tensor(0.0)

                for temperature in temperatures:
                    logits = -action_dist / max(temperature, 1e-6)
                    if self.use_state_kernel:
                        logits = logits - state_dist / max(self.state_kernel_temperature, 1e-6)
                    logits = logits + adv_weight.unsqueeze(0) / max(self.advantage_temperature, 1e-6)

                    weights = torch.softmax(logits, dim=-1)
                    if use_residual_drift:
                        field_for_temp = weights @ residual_pos
                        center_for_temp = actor_chunk + field_for_temp
                    else:
                        center_for_temp = weights @ act_pos
                        field_for_temp = center_for_temp - actor_chunk
                    field_sum = field_sum + field_for_temp
                    center_sum = center_sum + center_for_temp
                    max_weight_sum = max_weight_sum + weights.max(dim=-1).values.mean()
                    min_weight_sum = min_weight_sum + weights.min(dim=-1).values.mean()
                    weight_entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
                    if weights.shape[-1] > 1:
                        weight_entropy = weight_entropy / math.log(weights.shape[-1])
                    entropy_sum = entropy_sum + weight_entropy.mean()

                num_temperatures = len(temperatures)
                v_pos_raw = field_sum / num_temperatures
                center_pos = center_sum / num_temperatures
                raw_field_norm = v_pos_raw.norm(dim=-1)
                v_pos, field_clip_ratio = self._clip_by_norm_with_ratio(v_pos_raw, self.max_drift_velocity_norm)
                drift_delta = self.drift_step_size * v_pos
                drift_delta = self._clip_by_norm(drift_delta, self.max_drift_action_dist)
                target_chunks.append(actor_chunk + drift_delta)

                chunk_count = end - start
                log_sums["field_norm"] += v_pos.norm(dim=-1).sum().item()
                log_sums["field_norm_raw"] += raw_field_norm.sum().item()
                log_sums["field_clip_ratio"] += field_clip_ratio * chunk_count
                log_sums["center_dist"] += (center_pos - actor_chunk).norm(dim=-1).sum().item()
                log_sums["action_dist"] += drift_delta.norm(dim=-1).sum().item()
                log_sums["state_dist"] += state_dist_value.item() * chunk_count
                log_sums["action_dist_to_pos"] += action_dist_to_pos.item() * chunk_count
                log_sums["max_weight"] += (max_weight_sum / num_temperatures).item() * chunk_count
                log_sums["min_weight"] += (min_weight_sum / num_temperatures).item() * chunk_count
                log_sums["weight_entropy"] += (entropy_sum / num_temperatures).item() * chunk_count

        actor_mean_drifted = torch.cat(target_chunks, dim=0).detach()

        drift_loss_per_sample = torch.mean((actor_mean_flat - actor_mean_drifted) ** 2, dim=-1)
        drift_loss = drift_loss_per_sample.mean()

        return {
            "loss": drift_loss,
            "logs": {
                "loss": drift_loss.detach().item(),
                "positive_ratio": positive_ratio,
                "num_positive_samples": float(num_positive),
                "selected_positive_ratio": float(num_selected_positive) / float(pos_mask.numel()),
                "num_selected_positive_samples": float(num_selected_positive),
                "skip": 0.0,
                "action_kernel_temperature": current_temperature,
                "field_norm": log_sums["field_norm"] / num_samples,
                "field_norm_raw": log_sums["field_norm_raw"] / num_samples,
                "field_norm_normalized": log_sums["field_norm"] / num_samples,
                "field_clip_ratio": log_sums["field_clip_ratio"] / num_samples,
                "center_dist": log_sums["center_dist"] / num_samples,
                "action_dist": log_sums["action_dist"] / num_samples,
                "state_dist": log_sums["state_dist"] / num_samples,
                "action_dist_to_pos": log_sums["action_dist_to_pos"] / num_samples,
                "max_weight": log_sums["max_weight"] / num_samples,
                "min_weight": log_sums["min_weight"] / num_samples,
                "weight_entropy": log_sums["weight_entropy"] / num_samples,
                "mean_positive_adv": adv_pos.mean().item(),
                "max_positive_adv": adv_pos.max().item(),
            },
        }

    def _extract_state_feature(self, obs):
        if self.state_feature_mode == "selected":
            feature_dim = min(obs.shape[-1], 48)
            feat = obs[:, :feature_dim]
        else:
            feat = obs
        return self._normalize_feature_by_batch(feat)

    @staticmethod
    def _normalize_feature_by_batch(feature):
        mean = feature.mean(dim=0, keepdim=True)
        std = feature.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (feature - mean) / std

    def _scheduled_action_temperature(self, update_counter):
        if not self.use_temperature_schedule:
            return self.action_kernel_temperature
        if update_counter is None:
            return self.action_kernel_temperature_start
        schedule_start = self.action_kernel_temperature_schedule_start
        schedule_end = max(schedule_start + 1, self.action_kernel_temperature_schedule_end)
        progress = (update_counter - schedule_start) / float(schedule_end - schedule_start)
        progress = min(1.0, max(0.0, progress))
        return (
            self.action_kernel_temperature_start
            + progress * (self.action_kernel_temperature_end - self.action_kernel_temperature_start)
        )

    def _num_top_positive_samples(self, num_positive):
        top_fraction = min(1.0, max(0.0, self.positive_top_fraction))
        num_selected = int(math.ceil(num_positive * top_fraction))
        num_selected = max(self.min_positive_samples, num_selected)
        return min(num_positive, max(1, num_selected))

    @staticmethod
    def _flatten_batch(tensor):
        return tensor.reshape(-1, tensor.shape[-1])

    @staticmethod
    def _flatten_advantages(advantages):
        return advantages.reshape(-1)

    @staticmethod
    def _clip_by_norm(tensor, max_norm):
        norms = tensor.norm(dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / norms.clamp_min(1e-8), max=1.0)
        return tensor * scale

    @staticmethod
    def _clip_by_norm_with_ratio(tensor, max_norm):
        norms = tensor.norm(dim=-1, keepdim=True)
        clip_mask = norms > max_norm
        scale = torch.clamp(max_norm / norms.clamp_min(1e-8), max=1.0)
        clipped = tensor * scale
        clip_ratio = clip_mask.float().mean().item() if clip_mask.numel() > 0 else 0.0
        return clipped, clip_ratio

    @staticmethod
    def _empty_logs(positive_ratio, num_positive_samples, skip):
        return {
            "loss": 0.0,
            "positive_ratio": positive_ratio,
            "num_positive_samples": num_positive_samples,
            "selected_positive_ratio": 0.0,
            "num_selected_positive_samples": 0.0,
            "skip": skip,
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
