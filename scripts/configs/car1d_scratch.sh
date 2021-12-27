export PROJECT_NAME="car1d"
export RUN_NAME="scratch"
export ENV="envs:Car1DEnv-v1"
export N_STEPS=200
export TOTAL_TIMESTEPS=1000000
export POLICY_DIMS="2 2"
export VALUE_DIMS="256 256"
export EVAL_EVERY=1
export PRETRAINED_PATH=""
export STATS_PATH=""
export DEBUG=0
export DEVICE="cpu"
export MODEL_NAME="ppo"
export SEED=0
export LOG_STD_INIT=-2.0

# results dir to be set in sbatch_setup script