#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
echo "Running on seed $SEED..."

arg="imdb"

if [[ "$arg" == "imdb" ]]; then
    DATASET="imdb"
    MODEL="cnn_lstm"
    SGD_WEIGHT_DECAY="3."
    MFVI_WEIGHT_DECAY="5"
    SGLD_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--dataset_name=$DATASET --model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="3.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=20 --batch_size=80 --save_freq=500"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGLD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-5 --final_step_size=1e-5 --num_epochs=10000 --num_burnin_epochs=1000 --eval_freq=10 --batch_size=80 --save_freq=10 --momentum=0."
    SGD_STEP_DISPLAY="3e-07"
else
    echo "Please specify cifar10 or imdb"
    exit 1
fi


EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale/pretrain"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/miniforge3/envs/bnn"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Pretraining SGLD..."
# python bnn_hmc/run_sgmcmc.py \
#     --seed=$SEED \
#     --weight_decay=$SGLD_WEIGHT_DECAY \
#     --dir=$EXPERIMENT_DIR/sgld/cifar10/ \
#     --model_name=resnet20_frn_swish \
#     --dataset_name=cifar10 \
#     --subset_train_to=4080 \
#     --save_actual_dataset \
#     --init_step_size=1e-6 \
#     --final_step_size=1e-6 \
#     --num_epochs=10000 \
#     --num_burnin_epochs=1000 \
#     --eval_freq=10 \
#     --batch_size=80 \
#     --save_freq=10 \
#     --momentum=0.

python bnn_hmc/run_sgmcmc.py \
    --seed=$SEED \
    --weight_decay=$SGLD_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/sgld/$DATASET/ \
    --model_name=$MODEL \
    --dataset_name=$DATASET \
    --subset_train_to=4000 \
    --save_actual_dataset \
    $SGLD_HYPERPARAMS

