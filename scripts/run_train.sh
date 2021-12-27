#!/bin/bash

# Assume that the arguments are set as variables via e.g. "export PROMPT=..."
set -e

OPTIONALS=""
if [[ -n ${PRETRAINED_PATH} ]]; then OPTIONALS="$OPTIONALS --pretrained_path $PRETRAINED_PATH"; fi
if [[ -n ${STATS_PATH} ]]; then OPTIONALS="$OPTIONALS --stats_path $STATS_PATH"; fi

python train.py \
--project_name "$PROJECT_NAME" \
--run_name "$RUN_NAME" \
--env "$ENV" \
--n_steps "$N_STEPS" \
--total_timesteps "$TOTAL_TIMESTEPS" \
--policy_dims ${POLICY_DIMS} \
--value_dims ${VALUE_DIMS} \
--eval_every "$EVAL_EVERY" \
--debug "$DEBUG" \
--device "$DEVICE" \
--model_name "$MODEL_NAME" \
--seed "$SEED" \
--results_dir "$RESULTS_DIR" \
--checkpoints_dir "$CHECKPOINTS_DIR" \
--log_std_init "$LOG_STD_INIT" \
${OPTIONALS}

wait

gifski "$RESULTS_DIR"/"$SEED"_[0-9][0-9][0-9].png --fps 10 --output "$RESULTS_DIR"/gifs/"$SEED"_animated.gif
