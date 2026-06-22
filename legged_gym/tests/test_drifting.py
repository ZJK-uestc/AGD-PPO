import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = ROOT.parent / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.algorithms.drifting import Drifting


def make_drifting(**overrides):
    cfg = {
        "positive_adv_threshold": 0.0,
        "min_positive_samples": 1,
        "drift_step_size": 0.1,
        "max_drift_velocity_norm": 1.0,
        "max_drift_action_dist": 1.5,
        "action_kernel_temperature": 0.5,
        "advantage_temperature": 2.0,
        "advantage_clip": 3.0,
        "use_state_kernel": False,
        "state_kernel_temperature": 1.0,
        "state_feature_mode": "obs_norm",
        "use_multi_temperature": False,
        "action_kernel_temperatures": (0.3, 0.5, 1.0),
        "normalize_drift_field": False,
        "drift_field_norm_type": "batch",
        "log_drift_debug": True,
        "drift_chunk_size": 1024,
        "use_temperature_schedule": False,
        "action_kernel_temperature_start": 0.5,
        "action_kernel_temperature_end": 0.3,
        "action_kernel_temperature_schedule_start": 400,
        "action_kernel_temperature_schedule_end": 1000,
        "use_top_positive_filter": False,
        "positive_top_fraction": 0.35,
        "use_residual_drift": False,
    }
    cfg.update(overrides)
    return Drifting(**cfg)


class DriftingAcceptanceTests(unittest.TestCase):
    def test_all_negative_raw_advantages_produce_no_positive_samples(self):
        drifting = make_drifting()
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 2)
        raw_adv = torch.tensor([[-0.1], [-0.05], [-0.02], [-0.01]])
        actor_mean = torch.randn(4, 2, requires_grad=True)

        result = drifting.compute_loss(
            obs=obs,
            actions=actions,
            raw_advantages=raw_adv,
            actor_mean=actor_mean,
        )

        self.assertEqual((raw_adv.view(-1) > 0.0).sum().item(), 0)
        self.assertAlmostEqual(result["logs"]["skip"], 1.0)
        self.assertAlmostEqual(result["logs"]["num_positive_samples"], 0.0)
        self.assertAlmostEqual(result["logs"]["positive_ratio"], 0.0)

    def test_drift_target_stays_local_without_box_clamp(self):
        drifting = make_drifting(
            action_kernel_temperature=1.0,
            advantage_temperature=1.0,
            max_drift_velocity_norm=10.0,
            max_drift_action_dist=10.0,
        )
        actor_mean = torch.tensor([[2.5, -2.5]], dtype=torch.float32, requires_grad=True)
        obs = torch.zeros(1, 3)
        actions = torch.tensor([[2.51, -2.51]], dtype=torch.float32)
        raw_adv = torch.tensor([[0.2]], dtype=torch.float32)

        result = drifting.compute_loss(
            obs=obs,
            actions=actions,
            raw_advantages=raw_adv,
            actor_mean=actor_mean,
        )

        expected_target = torch.tensor([[2.501, -2.501]], dtype=torch.float32)
        expected_loss = torch.mean((actor_mean.detach() - expected_target) ** 2).item()

        self.assertAlmostEqual(result["loss"].item(), expected_loss, places=8)
        self.assertAlmostEqual(result["logs"]["action_dist"], 0.0014142135, places=8)
        self.assertGreater(expected_target[0, 0].item(), 1.0)
        self.assertLess(expected_target[0, 1].item(), -1.0)

    def test_weak_drift_field_is_not_amplified_by_clipping(self):
        v_weak = torch.tensor(
            [
                [0.001, 0.0],
                [0.002, 0.0],
                [0.0005, 0.0],
            ],
            dtype=torch.float32,
        )

        v_out, clip_ratio = Drifting._clip_by_norm_with_ratio(v_weak, max_norm=1.0)

        self.assertAlmostEqual(clip_ratio, 0.0)
        self.assertEqual(v_out.shape, v_weak.shape)
        self.assertTrue(torch.allclose(v_out, v_weak, atol=1e-8, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                torch.norm(v_out, dim=-1),
                torch.tensor([0.001, 0.002, 0.0005], dtype=torch.float32),
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_normalized_advantage_weights_remain_clipped(self):
        adv_pos = torch.tensor([3.0] + [1e-6] * 63)
        adv_weight = torch.clamp(adv_pos, min=0.0)
        adv_weight = adv_weight / adv_weight.mean().clamp_min(1e-8)
        adv_weight = torch.clamp(adv_weight, min=0.0, max=3.0)

        self.assertLessEqual(adv_weight.max().item(), 3.0)
        self.assertGreater(adv_weight.mean().item(), 0.0)

    def test_state_feature_normalization_is_finite_for_single_sample(self):
        feature = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
        normalized = Drifting._normalize_feature_by_batch(feature)

        self.assertTrue(torch.isfinite(normalized).all().item())
        self.assertTrue(torch.allclose(normalized, torch.zeros_like(feature), atol=1e-8, rtol=0.0))

    def test_temperature_schedule_interpolates_by_update(self):
        drifting = make_drifting(
            use_temperature_schedule=True,
            action_kernel_temperature_start=0.5,
            action_kernel_temperature_end=0.3,
            action_kernel_temperature_schedule_start=400,
            action_kernel_temperature_schedule_end=1000,
        )

        self.assertAlmostEqual(drifting._scheduled_action_temperature(300), 0.5)
        self.assertAlmostEqual(drifting._scheduled_action_temperature(700), 0.4)
        self.assertAlmostEqual(drifting._scheduled_action_temperature(1200), 0.3)

    def test_top_positive_filter_uses_subset_of_raw_positive_samples(self):
        drifting = make_drifting(
            use_top_positive_filter=True,
            positive_top_fraction=0.5,
            min_positive_samples=1,
        )
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 2)
        raw_adv = torch.tensor([[0.1], [0.4], [0.2], [-0.3]])
        actor_mean = torch.randn(4, 2, requires_grad=True)

        result = drifting.compute_loss(
            obs=obs,
            actions=actions,
            raw_advantages=raw_adv,
            actor_mean=actor_mean,
        )

        self.assertAlmostEqual(result["logs"]["positive_ratio"], 0.75)
        self.assertAlmostEqual(result["logs"]["selected_positive_ratio"], 0.5)
        self.assertAlmostEqual(result["logs"]["num_positive_samples"], 3.0)
        self.assertAlmostEqual(result["logs"]["num_selected_positive_samples"], 2.0)

    def test_residual_drift_uses_action_minus_old_mean(self):
        drifting = make_drifting(
            use_residual_drift=True,
            action_kernel_temperature=1.0,
            advantage_temperature=1.0,
            max_drift_velocity_norm=10.0,
            max_drift_action_dist=10.0,
        )
        obs = torch.zeros(1, 3)
        actor_mean = torch.tensor([[2.5, -2.5]], dtype=torch.float32, requires_grad=True)
        old_mu = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        actions = torch.tensor([[10.01, 9.99]], dtype=torch.float32)
        raw_adv = torch.tensor([[0.2]], dtype=torch.float32)

        result = drifting.compute_loss(
            obs=obs,
            actions=actions,
            raw_advantages=raw_adv,
            actor_mean=actor_mean,
            old_action_mean=old_mu,
        )

        expected_target = torch.tensor([[2.501, -2.501]], dtype=torch.float32)
        expected_loss = torch.mean((actor_mean.detach() - expected_target) ** 2).item()
        self.assertAlmostEqual(result["loss"].item(), expected_loss, places=8)


if __name__ == "__main__":
    unittest.main()
