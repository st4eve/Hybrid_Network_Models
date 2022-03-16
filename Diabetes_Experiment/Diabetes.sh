#!/bin/env bash
#SBATCH --array=0-17
#SBATCH --job-name=test_sweep
#SBATCH --output=output.txt
#SBATCH --mem=32GB
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=23
#SBATCH --mail-type=ALL
#SBATCH --mail-user=18arth@queensu.ca

encoding_strategy_values=( None Sigmoid Sigmoid_BatchNorm )
cutoff_dimension_values=( 5 10 15 )
num_layers_values=( 1 2 )
trial=${SLURM_ARRAY_TASK_ID}
encoding_strategy=${encoding_strategy_values[$(( trial % ${#encoding_strategy_values[@]} ))]}
trial=$(( trial / ${#encoding_strategy_values[@]} ))
cutoff_dimension=${cutoff_dimension_values[$(( trial % ${#cutoff_dimension_values[@]} ))]}
trial=$(( trial / ${#cutoff_dimension_values[@]} ))
num_layers=${num_layers_values[$(( trial % ${#num_layers_values[@]} ))]}

python Diabetes_Experiment.py with encoding_strategy=${encoding_strategy} cutoff_dimension=${cutoff_dimension} num_layers=${num_layers} 