#!/bin/bash

# to be run from another sweeping script

# to activate conda environment more consistently
set -e
echo $PATH
eval "$(conda shell.bash hook)"
conda activate py37

# for
export DATE=$(date +%Y%m%d_%H%M)
export RESULTS_DIR=runs/$DATE/results
export CHECKPOINTS_DIR=runs/$DATE/checkpoints
mkdir -p $RESULTS_DIR
mkdir -p $RESULTS_DIR/gifs
mkdir -p $CHECKPOINTS_DIR
