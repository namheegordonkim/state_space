#!/bin/bash

set -e

source scripts/sbatch_setup.sh
source scripts/configs/car1d_scratch.sh

for SEED in $(seq 0 20)
do
    export SEED
#    bash scripts/run_train.sh
    sbatch \
    --mem=16gb \
    --time=4:0:0 \
    scripts/run_train.sh
done