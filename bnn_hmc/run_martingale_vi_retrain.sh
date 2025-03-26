#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1
PRETRAINED_SAMPLING_SEED="1"
echo "Running on sample seed $PRETRAINED_SAMPLING_SEED, retraining seed $SEED..."

arg="imdb"

if [[ "$arg" == "cifar10" ]]; then
    DATASET="cifar10"
    MODEL="resnet20_frn_swish"
    SGD_WEIGHT_DECAY="10"
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="10.0"
    MFVI_WEIGHT_DECAY_DISPLAY="5.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=100"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGD_STEP_DISPLAY="3e-07"
elif [[ "$arg" == "cifar100" ]]; then
    DATASET="cifar100"
    MODEL="resnet20_frn_swish"
    SGD_WEIGHT_DECAY="10"
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="10.0"
    MFVI_WEIGHT_DECAY_DISPLAY="5.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-6 --num_epochs=500 --eval_freq=10 --batch_size=80 --save_freq=100"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGD_STEP_DISPLAY="1e-06"
elif [[ "$arg" == "imdb" ]]; then
    DATASET="imdb"
    MODEL="cnn_lstm"
    SGD_WEIGHT_DECAY="3."
    MFVI_WEIGHT_DECAY="5"
    COMMON_HYPERPARAMS="--model_name=$MODEL"
    SGD_WEIGHT_DECAY_DISPLAY="3.0"
    MFVI_WEIGHT_DECAY_DISPLAY="5.0"
    SGD_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=3e-7 --num_epochs=500 --eval_freq=20 --batch_size=80 --save_freq=500"
    VI_HYPERPARAMS="$COMMON_HYPERPARAMS --init_step_size=1e-4 --num_epochs=300 --eval_freq=10 --batch_size=80 --save_freq=150 --optimizer=Adam --vi_sigma_init=0.01 --temperature=1. --vi_ensemble_size=50"
    SGD_STEP_DISPLAY="3e-07"
else
    echo "Please specify cifar10 or imdb"
    exit 1
fi


EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
BASE_POSTERIOR_DIR="/mnt/disks/checkpoints/martingale/pretrain/vi/$DATASET/mfvi_initsigma_0.01_meaninit__opt_adam__lr_sch_i_0.0001___epochs_300_wd_${MFVI_WEIGHT_DECAY_DISPLAY}_batchsize_80_temp_1.0__seed_$PRETRAINED_SAMPLING_SEED"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/miniforge3/envs/bnn"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Sampling synthetic labels..."
python bnn_hmc/sample_synthetic_labels.py \
    --seed=$SEED \
    --dir=$EXPERIMENT_DIR/synthdata/vi/$DATASET \
    --model_name=$MODEL \
    --dataset_name=$DATASET \
    --subset_train_to=8000 \
    --sequential_training \
    --num_sequential_training_folds=2 \
    --index_sequential_training_fold=1 \
    --vi_checkpoint=$BASE_POSTERIOR_DIR/model_step_299.pt \
    --append_synthetic_dataset_to=$BASE_POSTERIOR_DIR/data_subset_4000.npz

# Find the output file produced by SGD for the second half
SYNTHDATADIR=$(ls $EXPERIMENT_DIR/synthdata/vi/$DATASET 2>/dev/null | grep -E "^sample_from_vi_.*_subset_8000_split_2_of_2__seed_$SEED$")
if [ -z "$SYNTHDATADIR" ]; then
    echo "Error: No synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 1
elif [ $(echo "$SYNTHDATADIR" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 2
fi

SYNTHDATAFILE=$(ls $EXPERIMENT_DIR/synthdata/vi/$DATASET/$SYNTHDATADIR 2>/dev/null | grep -E "^synth_appended_.*.npz$")
if [ -z "$SYNTHDATAFILE" ]; then
    echo "Error: No synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 3
elif [ $(echo "$SYNTHDATAFILE" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 4
fi

echo "Retraining SGD..."
python bnn_hmc/run_sgd.py \
    --seed=$SEED \
    --weight_decay=$SGD_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/retrain/sgd_for_vi/$DATASET/ \
    --dataset_name=$EXPERIMENT_DIR/synthdata/vi/$DATASET/$SYNTHDATADIR/$SYNTHDATAFILE \
    --subset_train_to=8000 \
    $SGD_HYPERPARAMS

echo "Retraining VI..."
python bnn_hmc/run_vi.py \
    --seed=$SEED \
    --weight_decay=$MFVI_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/retrain/vi/$DATASET/ \
    --dataset_name=$EXPERIMENT_DIR/synthdata/vi/$DATASET/$SYNTHDATADIR/$SYNTHDATAFILE \
    --subset_train_to=8000 \
    --mean_init_checkpoint=$EXPERIMENT_DIR/retrain/sgd_for_vi/$DATASET/sgd_mom_0.9__lr_sch_i_${SGD_STEP_DISPLAY}___epochs_500_wd_${SGD_WEIGHT_DECAY_DISPLAY}_batchsize_80_temp_1.0__seed_$SEED/model_step_499.pt \
    $VI_HYPERPARAMS

