#!/bin/env bash
#SBATCH --array=0-99
#SBATCH --job-name=CIFAR_Experiment_Test
#SBATCH --output=output%A%a.txt
#SBATCH --error=error%A%a.txt
#SBATCH --mem=16GB
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=17tna@queensu.ca
#SBATCH --account=def-bshastri
input$SLURM_ARRAY_TASK_ID.dat

activation_values=( Sigmoid )
cutoff_dimension_values=( 5 )
encoding_method_values=( Amplitude_Phase )
max_initial_weight_values=( 0.2 )
n_circuits_values=( 1 )
n_classes_values=( 4 )
n_qumodes_values=( 4 )
norm_threshold_values=( 0.99 )
num_layers_values=( 2 )
regularizer_string_values=( L1=0.1 )
seed_values=( 110101990 )
precision_values=( 1 2 4 8 16 32 64 128 256 512 )
shots_values=( 1 2 4 8 16 32 64 128 256 512 )
max_epoch_values=( 29.0 )
exp_train_values=( 14.0 )
trial=${SLURM_ARRAY_TASK_ID}
activation=${activation_values[$(( trial % ${#activation_values[@]} ))]}
trial=$(( trial / ${#activation_values[@]} ))
cutoff_dimension=${cutoff_dimension_values[$(( trial % ${#cutoff_dimension_values[@]} ))]}
trial=$(( trial / ${#cutoff_dimension_values[@]} ))
encoding_method=${encoding_method_values[$(( trial % ${#encoding_method_values[@]} ))]}
trial=$(( trial / ${#encoding_method_values[@]} ))
max_initial_weight=${max_initial_weight_values[$(( trial % ${#max_initial_weight_values[@]} ))]}
trial=$(( trial / ${#max_initial_weight_values[@]} ))
n_circuits=${n_circuits_values[$(( trial % ${#n_circuits_values[@]} ))]}
trial=$(( trial / ${#n_circuits_values[@]} ))
n_classes=${n_classes_values[$(( trial % ${#n_classes_values[@]} ))]}
trial=$(( trial / ${#n_classes_values[@]} ))
n_qumodes=${n_qumodes_values[$(( trial % ${#n_qumodes_values[@]} ))]}
trial=$(( trial / ${#n_qumodes_values[@]} ))
norm_threshold=${norm_threshold_values[$(( trial % ${#norm_threshold_values[@]} ))]}
trial=$(( trial / ${#norm_threshold_values[@]} ))
num_layers=${num_layers_values[$(( trial % ${#num_layers_values[@]} ))]}
trial=$(( trial / ${#num_layers_values[@]} ))
regularizer_string=${regularizer_string_values[$(( trial % ${#regularizer_string_values[@]} ))]}
trial=$(( trial / ${#regularizer_string_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
precision=${precision_values[$(( trial % ${#precision_values[@]} ))]}
trial=$(( trial / ${#precision_values[@]} ))
shots=${shots_values[$(( trial % ${#shots_values[@]} ))]}
trial=$(( trial / ${#shots_values[@]} ))
max_epoch=${max_epoch_values[$(( trial % ${#max_epoch_values[@]} ))]}
trial=$(( trial / ${#max_epoch_values[@]} ))
exp_train=${exp_train_values[$(( trial % ${#exp_train_values[@]} ))]}

python CIFAR_PWBTest_Experiment.py with activation=${activation} cutoff_dimension=${cutoff_dimension} encoding_method=${encoding_method} max_initial_weight=${max_initial_weight} n_circuits=${n_circuits} n_classes=${n_classes} n_qumodes=${n_qumodes} norm_threshold=${norm_threshold} num_layers=${num_layers} regularizer_string=${regularizer_string} seed=${seed} precision=${precision} shots=${shots} max_epoch=${max_epoch} exp_train=${exp_train} 