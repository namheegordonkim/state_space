# Train the agent
python train.py \
--project_name car1d \
--run_name scratch \
--env envs:Car1DEnv-v0 \
--eval_every 1000 \
--log_std_init -1.0 \
--total_timesteps 100000 \
--n_steps 1000 \
--policy_dims 2 2 \
--value_dims 16 16

# With Car1DEnv-v1
python train.py \
--project_name car1d \
--run_name scratch \
--env envs:Car1DEnv-v1 \
--eval_every 1000 \
--log_std_init -1.0 \
--total_timesteps 100000 \
--n_steps 200 \
--policy_dims 64 64 \
--value_dims 64 64 \
--device auto

# After training, appropriate directories are created under checkpoints
python enjoy.py \
--env envs:Car1DEnv-v0 \
--policy_path checkpoints/car1d_scratch/latest.zip \
--stats_path checkpoints/car1d_scratch/latest_stats.pth

python enjoy.py \
--env envs:Car1DEnv-v1 \
--policy_path checkpoints/car1d_scratch/latest.zip \
--stats_path checkpoints/car1d_scratch/latest_stats.pth

python encode_states.py \
--env envs:Car1DEnv-v0 \
--policy_path checkpoints/car1d_scratch/best.zip \
--stats_path checkpoints/car1d_scratch/best_stats.pth