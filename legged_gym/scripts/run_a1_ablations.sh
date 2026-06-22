#!/usr/bin/env bash

set -euo pipefail

TASK="${TASK:-anymal_c_rough}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
SEEDS="${SEEDS:-1 42}"
ABLATIONS_TO_RUN="${ABLATIONS:-roughB}"
ENTROPY_COEF="${ENTROPY_COEF:-}"
#multi_temp state_kernel strong_weight  baseline action_only anymal_c_rough
for ablation in ${ABLATIONS_TO_RUN}; do
  case "${ablation}" in
    baseline)
      EXTRA_ARGS=(--use_drift False)
      ;;
    flat)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel False
        --drift_actor_loss_coef 0.003
        --drift_step_size 0.15
        --use_temperature_schedule True
        --max_drift_action_dist 1.0
        --drift_model_warmup_updates 500
        --drift_actor_warmup_updates 700
        --use_top_positive_filter True
        --positive_top_fraction 0.25
      )
      ;;
    rough)
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
        --drift_model_warmup_updates 200
        --drift_actor_warmup_updates 300
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.2
        --max_drift_action_dist 1.0          
        --action_kernel_temperature_schedule_start 200  
        --action_kernel_temperature_schedule_end 800 
        --entropy_coef "${ENTROPY_COEF:-0.005}"
      )
      ;;
    a1)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel False
        --drift_actor_loss_coef 0.001
        --drift_step_size 0.1
        --use_top_positive_filter True
        --use_temperature_schedule True
        --positive_top_fraction 0.5
      )
      ;;
    roughB)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel True
        --use_residual_drift True
        --drift_actor_loss_coef 0.008
        --drift_step_size 0.25
        --use_temperature_schedule True
        --max_drift_action_dist 1.5
        --action_kernel_temperature_end 0.05
        --use_top_positive_filter True
        --positive_top_fraction 0.1
        --state_kernel_temperature 0.2
        --entropy_coef "${ENTROPY_COEF:-0.0}"
        --use_temperature_schedule True       
        --drift_model_warmup_updates 200
        --drift_actor_warmup_updates 300
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_schedule_start 200  
        --action_kernel_temperature_schedule_end 800 

      )
      ;;
    roughA)
      EXTRA_ARGS=(
        --use_drift True
        --use_state_kernel True
        --use_residual_drift True
        --drift_actor_loss_coef 0.005
        --drift_step_size 0.2
        --use_temperature_schedule True
        --max_drift_action_dist 1.0
        --drift_model_warmup_updates 150
        --drift_actor_warmup_updates 250
        --action_kernel_temperature_schedule_start 150
        --action_kernel_temperature_schedule_end 600
        --action_kernel_temperature_start 0.5
        --action_kernel_temperature_end 0.2
        --use_top_positive_filter True
        --positive_top_fraction 0.15
        --state_kernel_temperature 0.3
        --entropy_coef "${ENTROPY_COEF:-0.002}"
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
