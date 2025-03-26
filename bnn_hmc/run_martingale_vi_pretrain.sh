#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
echo "Running on seed $SEED..."

arg="imdb"

if [[ "$arg" == "cifar10" ]]; then
    DATASET="cifar10"
    MODEL="resnet20_frn_swish"
    SGD_WEIGHT_DECAY="10"
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--dataset_name=$DATASET --model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="10.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=100"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGD_STEP_DISPLAY="3e-07"
elif [[ "$arg" == "cifar100" ]]; then
    DATASET="cifar100"
    MODEL="resnet20_frn_swish"
    SGD_WEIGHT_DECAY="10"
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--dataset_name=$DATASET --model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="10.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-6 --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=100"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGD_STEP_DISPLAY="1e-06"
elif [[ "$arg" == "imdb" ]]; then
    DATASET="imdb"
    MODEL="cnn_lstm"
    SGD_WEIGHT_DECAY="3."
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--dataset_name=$DATASET --model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="3.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=20 --batch_size=80 --save_freq=500"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
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

echo "Pretraining SGD..."
python bnn_hmc/run_sgd.py \
    --seed=$SEED \
    --weight_decay=$SGD_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/sgd/$DATASET/ \
    --subset_train_to=4000 \
    $SGD_HYPERPARAMS

echo "Pretraining VI..."
python bnn_hmc/run_vi.py \
    --seed=$SEED \
    --weight_decay=$MFVI_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/vi/$DATASET/ \
    --subset_train_to=4000 \
    --save_actual_dataset \
    --mean_init_checkpoint=$EXPERIMENT_DIR/sgd/$DATASET/sgd_mom_0.9__lr_sch_i_${SGD_STEP_DISPLAY}___epochs_500_wd_${SGD_WEIGHT_DECAY_DISPLAY}_batchsize_80_temp_1.0__seed_$SEED/model_step_499.pt \
    $VI_HYPERPARAMS

