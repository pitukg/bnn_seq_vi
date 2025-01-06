#!/bin/bash

# Ensure a seed value is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi
MARTINGALE_SEED="$1"

# Define input filenames and number of bootstrap samples
EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/miniforge3/envs/bnn
# PYTHON=`which python`
# echo "Using python: $PYTHON"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"

input_checkpoints=()
for pt in $(seq 1000 10 9999); do
	input_checkpoints+=("$EXPERIMENT_DIR/retrain/sgld/cifar10/sgld_mom_0.0_preconditioner_None__lr_sch_constant_i_1e-06_f_1e-06_c_50_bi_1000___epochs_10000_wd_5.0_batchsize_80_temp_1.0__seed_$MARTINGALE_SEED/model_step_$pt.pt")
done
# readarray -d '' input_checkpoints < <(find $EXPERIMENT_DIR/retrain/sgld/cifar10 -wholename "*/sgld_mom_0.0_preconditioner_None__lr_sch_constant_i_1e-06_f_1e-06_c_50_bi_1000___epochs_10000_wd_5.0_batchsize_80_temp_1.0__seed_$MARTINGALE_SEED/model_step_*.pt" -print0)

# Get the number of input files
num_files="${#input_checkpoints[@]}"

echo "Ensembling checkpoints..."
python bnn_hmc/ensemble_checkpoints.py \
    --dir=$EXPERIMENT_DIR/ensemble/sgld/cifar10/preagg_sample_$MARTINGALE_SEED \
    --model_name=resnet20_frn_swish \
    --dataset_name=cifar10 \
    --subset_train_to=4080 \
    --save_ensembled_preds \
    --sgd_checkpoints \
    -- ${input_checkpoints[@]}

