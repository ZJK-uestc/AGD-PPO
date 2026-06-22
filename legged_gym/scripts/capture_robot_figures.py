#!/usr/bin/env python3
"""Interactive trained-policy figure capture.

Usage:
    conda run -n zjk python legged_gym/scripts/capture_robot_figures.py

Controls:
    P / Space: save current frame and continue to the next task
    ESC: close viewer and stop
"""

from pathlib import Path
import sys

import isaacgym
from isaacgym import gymapi
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


OUTPUT_DIR = Path(LEGGED_GYM_ROOT_DIR) / "results" / "figure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ROBOT_SETUPS = [
    {
        "task": "anymal_c_rough",
        "experiment_name": "stage3_roughB",
        "load_run": "Jun13_13-31-10_seed_42",
        "checkpoint": -1,
        "camera_pos": [7.8, 4.8, 1.25],
        "camera_lookat": [10.2, 7.7, 0.9],
        "num_envs": 9,
        "num_rows": 3,
        "num_cols": 3,
    },
    {
        "task": "a1",
        "experiment_name": "rough_a1",
        "load_run": "Jun10_21-33-58_",
        "checkpoint": -1,
        "camera_pos": [8.0, 5.0, 1.15],
        "camera_lookat": [10.3, 7.8, 0.78],
        "num_envs": 9,
        "num_rows": 3,
        "num_cols": 3,
    },

    {
        "task": "cassie",
        "experiment_name": "rough_cassie",
        "load_run": "Jun12_16-50-15_",
        "checkpoint": -1,
        "camera_pos": [8.1, 5.0, 1.35],
        "camera_lookat": [10.5, 7.9, 1.02],
        "num_envs": 9,
        "num_rows": 3,
        "num_cols": 3,
    },
]


def _clone_args(base_args, setup):
    base_args.task = setup["task"]
    base_args.headless = False
    base_args.resume = True
    base_args.experiment_name = setup["experiment_name"]
    base_args.load_run = setup["load_run"]
    base_args.checkpoint = setup["checkpoint"]
    base_args.num_envs = setup["num_envs"]
    return base_args


def _configure_env(env_cfg, setup):
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, setup["num_envs"])
    env_cfg.terrain.num_rows = setup["num_rows"]
    env_cfg.terrain.num_cols = setup["num_cols"]
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    if hasattr(env_cfg.domain_rand, "randomize_friction"):
        env_cfg.domain_rand.randomize_friction = False
    if hasattr(env_cfg.domain_rand, "randomize_base_mass"):
        env_cfg.domain_rand.randomize_base_mass = False
    return env_cfg


def capture_interactively(base_args, setup):
    print(f"\n{'=' * 60}")
    print(f"Interactive capture for {setup['task']}")
    print("Press P or Space to save and continue. Press ESC to quit.")
    print(f"{'=' * 60}")

    args = _clone_args(base_args, setup)
    env_cfg, train_cfg = task_registry.get_cfgs(name=setup["task"])
    env_cfg = _configure_env(env_cfg, setup)

    env, _ = task_registry.make_env(name=setup["task"], args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.experiment_name = setup["experiment_name"]
    train_cfg.runner.load_run = setup["load_run"]
    train_cfg.runner.checkpoint = setup["checkpoint"]

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=setup["task"],
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.set_camera(setup["camera_pos"], setup["camera_lookat"])
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "CAPTURE_AND_CONTINUE")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_SPACE, "CAPTURE_AND_CONTINUE")

    with torch.no_grad():
        while True:
            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

            if env.gym.query_viewer_has_closed(env.viewer):
                env.gym.destroy_viewer(env.viewer)
                env.gym.destroy_sim(env.sim)
                sys.exit(0)

            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
                    sys.exit(0)
                if evt.action == "toggle_viewer_sync" and evt.value > 0:
                    env.enable_viewer_sync = not env.enable_viewer_sync
                if evt.action == "CAPTURE_AND_CONTINUE" and evt.value > 0:
                    out_path = OUTPUT_DIR / f"{setup['task']}.png"
                    env.gym.write_viewer_image_to_file(env.viewer, str(out_path))
                    print(f"Saved figure to {out_path}")
                    env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
                    return

            if env.device != "cpu":
                env.gym.fetch_results(env.sim, True)
            if env.enable_viewer_sync:
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                env.gym.sync_frame_time(env.sim)
            else:
                env.gym.poll_viewer_events(env.viewer)


def main():
    args = get_args()
    for setup in ROBOT_SETUPS:
        capture_interactively(args, setup)
    print(f"\nDone. Figures saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
