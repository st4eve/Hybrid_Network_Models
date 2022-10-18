#!/bin/env bash
#SBATCH --array=0-719
#SBATCH --job-name=WINE_Experiment_Sweep
#SBATCH --output=outputWINE/output%A%a.txt
#SBATCH --error=outputWINE/error%A%a.txt
#SBATCH --mem=16GB
#SBATCH --time=0-48:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=17tna@queensu.ca
#SBATCH --account=def-bshastri
input$SLURM_ARRAY_TASK_ID.dat

encoding_method_values=( Amplitude_Phase )
cutoff_dimension_values=( 5 )
num_layers_values=( 1 2 4 )
num_classical_values=( 1 2 4 )
classical_size_values=( 10 20 )
n_qumodes_values=( 2 4 )
n_circuits_values=( 1 )
regularizer_string_values=( L1=0.01 L1=0.1 L2=0.01 L2=0.1 None )
max_initial_weight_values=( None )
activation_values=( Sigmoid TanH )
norm_threshold_values=( 0.99 )
ff_activation_values=( relu None )
trial=${SLURM_ARRAY_TASK_ID}
encoding_method=${encoding_method_values[$(( trial % ${#encoding_method_values[@]} ))]}
trial=$(( trial / ${#encoding_method_values[@]} ))
cutoff_dimension=${cutoff_dimension_values[$(( trial % ${#cutoff_dimension_values[@]} ))]}
trial=$(( trial / ${#cutoff_dimension_values[@]} ))
num_layers=${num_layers_values[$(( trial % ${#num_layers_values[@]} ))]}
trial=$(( trial / ${#num_layers_values[@]} ))
num_classical=${num_classical_values[$(( trial % ${#num_classical_values[@]} ))]}
trial=$(( trial / ${#num_classical_values[@]} ))
classical_size=${classical_size_values[$(( trial % ${#classical_size_values[@]} ))]}
trial=$(( trial / ${#classical_size_values[@]} ))
n_qumodes=${n_qumodes_values[$(( trial % ${#n_qumodes_values[@]} ))]}
trial=$(( trial / ${#n_qumodes_values[@]} ))
n_circuits=${n_circuits_values[$(( trial % ${#n_circuits_values[@]} ))]}
trial=$(( trial / ${#n_circuits_values[@]} ))
regularizer_string=${regularizer_string_values[$(( trial % ${#regularizer_string_values[@]} ))]}
trial=$(( trial / ${#regularizer_string_values[@]} ))
max_initial_weight=${max_initial_weight_values[$(( trial % ${#max_initial_weight_values[@]} ))]}
trial=$(( trial / ${#max_initial_weight_values[@]} ))
activation=${activation_values[$(( trial % ${#activation_values[@]} ))]}
trial=$(( trial / ${#activation_values[@]} ))
norm_threshold=${norm_threshold_values[$(( trial % ${#norm_threshold_values[@]} ))]}
trial=$(( trial / ${#norm_threshold_values[@]} ))
ff_activation=${ff_activation_values[$(( trial % ${#ff_activation_values[@]} ))]}

python WINE_Experiment.py with encoding_method=${encoding_method} cutoff_dimension=${cutoff_dimension} num_layers=${num_layers} num_classical=${num_classical} classical_size=${classical_size} n_qumodes=${n_qumodes} n_circuits=${n_circuits} regularizer_string=${regularizer_string} max_initial_weight=${max_initial_weight} activation=${activation} norm_threshold=${norm_threshold} ff_activation=${ff_activation} 