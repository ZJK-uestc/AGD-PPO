#!/usr/bin/env bash

set -euo pipefail

TASK="${TASK:-cassie}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
SEEDS="${SEEDS:-9 10 11 12 13 14}"
ABLATIONS_TO_RUN="${ABLATIONS:-baseline cassie_v1}"
ENTROPY_COEF="${ENTROPY_COEF:-}"
# Cassie Drift ablations (round 2):
#   baseline  - PPO without Drift
#   cassie_v1 - state_kernel + moderate drift (keep best from round 1)
#   cassie_v4 - action-only kernel  (round 1 winner, +14.9%)
#   cassie_v5 - v4 refined: wider top_frac + higher entropy → reduce variance
#   cassie_v6 - v4 aggressive: tighter filter + earlier warmup → push ceiling
for ablation in ${ABLATIONS_TO_RUN}; do
  case "${ablation}" in
    baseline)
      EXTRA_ARGS=(--use_drift False)
      ;;
    cassie_v1)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel True
        --use_residual_drift True
        --drift_actor_loss_coef 0.003
        --drift_step_size 0.15
        --use_temperature_schedule True
        --state_kernel_temperature 0.5
        --positive_top_fraction 0.25
        --use_top_positive_filter True
        --drift_model_warmup_updates 300
        --drift_actor_warmup_updates 400
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.2
        --action_kernel_temperature_schedule_start 300
        --action_kernel_temperature_schedule_end 900
        --max_drift_action_dist 0.8
        --entropy_coef "${ENTROPY_COEF:-0.005}"
      )
      ;;
    cassie_v5)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel False
        --drift_actor_loss_coef 0.0025
        --drift_step_size 0.15
        --use_temperature_schedule True
        --positive_top_fraction 0.30
        --use_top_positive_filter True
        --drift_model_warmup_updates 250
        --drift_actor_warmup_updates 350
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.15
        --action_kernel_temperature_schedule_start 250
        --action_kernel_temperature_schedule_end 900
        --max_drift_action_dist 0.7
        --entropy_coef "${ENTROPY_COEF:-0.006}"
      )
      ;;
    cassie_v6)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel False
        --drift_actor_loss_coef 0.004
        --drift_step_size 0.18
        --use_temperature_schedule True
        --positive_top_fraction 0.20
        --use_top_positive_filter True
        --drift_model_warmup_updates 200
        --drift_actor_warmup_updates 300
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.1
        --action_kernel_temperature_schedule_start 200
        --action_kernel_temperature_schedule_end 700
        --max_drift_action_dist 0.9
        --entropy_coef "${ENTROPY_COEF:-0.004}"
      )
      ;;
    cassie_v4)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel False
        --drift_actor_loss_coef 0.003
        --drift_step_size 0.15
        --use_temperature_schedule True
        --positive_top_fraction 0.25
        --use_top_positive_filter True
        --drift_model_warmup_updates 300
        --drift_actor_warmup_updates 400
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.2
        --action_kernel_temperature_schedule_start 300
        --action_kernel_temperature_schedule_end 900
        --max_drift_action_dist 0.8
        --entropy_coef "${ENTROPY_COEF:-0.005}"
      )
      ;;
    *)
      echo "Unknown ablation: ${ablation}" >&2
      exit 1
      ;;
  esac

  for seed in ${SEEDS}; do
    python legged_gym/scripts/train.py \
      --task "${TASK}" \
      --headless \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${MAX_ITERATIONS}" \
      --seed "${seed}" \
      --experiment_name "stage3_${ablation}" \
      --run_name "seed_${seed}" \
      "${EXTRA_ARGS[@]}"
  done
done
