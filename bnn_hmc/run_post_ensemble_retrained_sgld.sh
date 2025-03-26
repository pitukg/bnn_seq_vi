#!/bin/bash

# Ensure a seed value is provided
if [[ "$1" == "-bootstrap" ]]; then
	BOOTSTRAP=true
	if [ -n "$2" ]; then
		 RANDOM="$2"
	else
		$RANDOM=$(date '+%s')
	fi
	echo "Running on bootstrap shuffle seed $RANDOM"
elif [ -n "$1" ]; then
	echo "Usage: $0 <seed> [-pca]"
	exit 1
else
	BOOTSTRAP=false
fi

DATASET="imdb"
MODEL="cnn_lstm"

# Define input filenames and number of bootstrap samples
B=100  # Number of bootstrap samples
EXPERIMENT_DIR="/mnt/disks/checkpoints/martingale"
RUNWD="$HOME/bnn_hmc/"
cd $RUNWD
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/miniforge3/envs/bnn
# PYTHON=`which python`
# echo "Using python: $PYTHON"
export PYTHONPATH="$RUNWD/:$PYTHONPATH"


readarray -d '' input_checkpoints < <(find $EXPERIMENT_DIR/ensemble/sgld/$DATASET -wholename "*/preagg_sample_*/sgd_ensembled_*/ensembled_preds.npy" -print0)
# Get the number of input files
num_files="${#input_checkpoints[@]}"
echo $num_files

if [ "$BOOTSTRAP" = true ]; then
	# Perform resampling B times
	for ((i=1; i<=B; i++)); do
		# Generate a bootstrap sample with replacement
		bootstrap_sample=()
		for ((j=0; j<num_files; j++)); do
			random_index=$((RANDOM % num_files))
			bootstrap_sample+=("${input_checkpoints[random_index]}")
		done

		# Call your script with the resampled filenames
		echo "Bootstrap ensemble $i... length ${#bootstrap_sample[@]}"
		python bnn_hmc/ensemble_predictions.py \
		    --dir=$EXPERIMENT_DIR/ensemble/sgld/$DATASET/bootstrap_postagg \
		    --model_name=$MODEL \
		    --dataset_name=$DATASET \
		    --subset_train_to=4000 \
		    --save_ensembled_preds \
		    -- ${bootstrap_sample[@]}
	done
else
	echo "Ensembling checkpoints..."
	python bnn_hmc/ensemble_predictions.py \
	    --dir=$EXPERIMENT_DIR/ensemble/sgld/$DATASET/postagg \
	    --model_name=$MODEL \
	    --dataset_name=$DATASET \
	    --subset_train_to=4000 \
	    --save_ensembled_preds \
	    -- ${input_checkpoints[@]}
fi

