#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <sgld-sample>"
    exit 1
fi

SAMPLE=$1
# Re-using SGLD sample as seed for sampling
SEED=$SAMPLE
PRETRAINED_SAMPLING_SEED="1"
echo "Running on sample seed $PRETRAINED_SAMPLING_SEED, retraining seed $SEED..."

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
    SGLD_HYPERPARAMS="--model_name=$MODEL --init_step_size=1e-5 --final_step_size=1e-5 --num_epochs=10000 --num_burnin_epochs=1000 --eval_freq=10000 --ensemble_freq=20 --batch_size=80 --save_freq=500 --momentum=0."
    SGD_STEP_DISPLAY="3e-07"
else
    echo "Please specify cifar10 or imdb"
    exit 1
fi


EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
BASE_POSTERIOR_DIR="$EXPERIMENT_DIR/pretrain/sgld/$DATASET/sgld_mom_0.0_preconditioner_None__lr_sch_constant_i_1e-05_f_1e-05_c_50_bi_1000___epochs_10000_wd_5.0_batchsize_80_temp_1.0__seed_$PRETRAINED_SAMPLING_SEED"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/miniforge3/envs/bnn"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

echo "Sampling synthetic labels..."
python bnn_hmc/sample_synthetic_labels.py \
    --seed=$SEED \
    --dir=$EXPERIMENT_DIR/synthdata/sgld/$DATASET \
    --model_name=$MODEL \
    --dataset_name=$DATASET \
    --subset_train_to=8000 \
    --sequential_training \
    --num_sequential_training_folds=2 \
    --index_sequential_training_fold=1 \
    --params_checkpoint=$BASE_POSTERIOR_DIR/model_step_$SEED.pt \
    --append_synthetic_dataset_to=$BASE_POSTERIOR_DIR/data_subset_4000.npz

# Find the output file produced by SGD for the second half
SYNTHDATADIR=$(ls $EXPERIMENT_DIR/synthdata/sgld/$DATASET 2>/dev/null | grep -E "^sample_from_checkpoint_.*_subset_8000_split_2_of_2__seed_$SEED$")
if [ -z "$SYNTHDATADIR" ]; then
    echo "Error: No synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 1
elif [ $(echo "$SYNTHDATADIR" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling runs were found with seed $SEED:\n$SYNTHDATADIR"
    exit 2
fi

SYNTHDATAFILE=$(ls $EXPERIMENT_DIR/synthdata/sgld/$DATASET/$SYNTHDATADIR 2>/dev/null | grep -E "^synth_appended_.*.npz$")
if [ -z "$SYNTHDATAFILE" ]; then
    echo "Error: No synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 3
elif [ $(echo "$SYNTHDATAFILE" | wc -l) -gt 1 ]; then
    echo "Error: Multiple synthetic sampling run files were found with seed $SEED:\n$SYNTHDATAFILE"
    exit 4
fi

echo "Retraining SGLD..."
python bnn_hmc/run_sgmcmc.py \
    --seed=$SEED \
    --weight_decay=$SGLD_WEIGHT_DECAY \
    --dir=$EXPERIMENT_DIR/retrain/sgld/$DATASET/ \
    --dataset_name=$EXPERIMENT_DIR/synthdata/sgld/$DATASET/$SYNTHDATADIR/$SYNTHDATAFILE \
    --subset_train_to=8000 \
    $SGLD_HYPERPARAMS

