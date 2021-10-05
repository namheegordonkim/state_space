# Train the agent
# With Car1DEnv-v1
python train.py \
--project_name car1d \
--run_name scratch \
--env envs:Car1DEnv-v1 \
--eval_every 1000 \
--log_std_init -1.0 \
--total_timesteps 240000 \
--n_steps 200 \
--policy_dims 256 256 \
--value_dims 256 256 \
--device auto

python enjoy.py \
--env envs:Car1DEnv-v1 \
--policy_path checkpoints/car1d_scratch/best.zip \
--stats_path checkpoints/car1d_scratch/best_stats.pth

python enjoy.py \
--env envs:Car1DEnv-v1 \
--policy_path checkpoints/car1d_scratch/latest.zip \
--stats_path checkpoints/car1d_scratch/latest_stats.pth

python encode_states.py \
--env envs:Car1DEnv-v0 \
--policy_path checkpoints/car1d_scratch/best.zip \
--stats_path checkpoints/car1d_scratch/best_stats.pth

python encode_states.py \
--env envs:Car1DEnv-v0 \
--policy_path checkpoints/car1d_scratch/latest.zip \
--stats_path checkpoints/car1d_scratch/latest_stats.pth