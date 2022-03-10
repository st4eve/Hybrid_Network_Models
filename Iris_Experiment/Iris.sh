#!/bin/env bash
#SBATCH --array=0-767
#SBATCH --job-name=test_sweep
#SBATCH --output=output.txt
#SBATCH --mem=16GB
#SBATCH --time=0-3:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=18arth@queensu.ca

initial_weight_amplitudes_values=( 0.1 0.25 0.5 0.75 1.0 1.25 1.5 2.0 )
initial_input_amplitude_values=( 0.1 0.25 0.5 0.75 1.0 1.25 1.5 2.0 )
loss_coefficient_values=( 0.01 )
cutoff_management_values=( None L1 L2 Loss )
cutoff_dimension_values=( 10 20 30 )
trial=${SLURM_ARRAY_TASK_ID}
initial_weight_amplitudes=${initial_weight_amplitudes_values[$(( trial % ${#initial_weight_amplitudes_values[@]} ))]}
trial=$(( trial / ${#initial_weight_amplitudes_values[@]} ))
initial_input_amplitude=${initial_input_amplitude_values[$(( trial % ${#initial_input_amplitude_values[@]} ))]}
trial=$(( trial / ${#initial_input_amplitude_values[@]} ))
loss_coefficient=${loss_coefficient_values[$(( trial % ${#loss_coefficient_values[@]} ))]}
trial=$(( trial / ${#loss_coefficient_values[@]} ))
cutoff_management=${cutoff_management_values[$(( trial % ${#cutoff_management_values[@]} ))]}
trial=$(( trial / ${#cutoff_management_values[@]} ))
cutoff_dimension=${cutoff_dimension_values[$(( trial % ${#cutoff_dimension_values[@]} ))]}

python Iris_Experiment.py with initial_weight_amplitudes=${initial_weight_amplitudes} initial_input_amplitude=${initial_input_amplitude} loss_coefficient=${loss_coefficient} cutoff_management=${cutoff_management} cutoff_dimension=${cutoff_dimension} 