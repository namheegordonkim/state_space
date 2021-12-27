#!/bin/bash

set -e

source scripts/sbatch_setup.sh
source scripts/configs/car1d_scratch.sh

for SEED in $(seq 0 10)
do
    export SEED
    bash scripts/run_train.sh
#    sbatch \
#    --mem=16gb \
#    --time=4:0:0 \
#    --gres=gpu:1 \
#    --constraint="pascal|volta|a100" \
#    scripts/run_train.sh
done