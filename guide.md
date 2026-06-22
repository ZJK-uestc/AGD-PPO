for seed in 1 2 3 4 5; do
  python legged_gym/scripts/train.py \
    --task anymal_c_rough \
    --headless \
    --num_envs 4096 \
    --max_iterations 1500 \
    --seed "$seed" \
    --experiment_name a1_multi_seed \
    --run_name "seed_${seed}"
done

bash legged_gym/scripts/run_stage3_ablations.sh

  python legged_gym/scripts/train.py \
    --task cassie \
    --headless \
    --num_envs 4096 \
    --max_iterations 1500 \
    --seed 4 \
    --use_drift False

python legged_gym/scripts/plot_reward_groups.py