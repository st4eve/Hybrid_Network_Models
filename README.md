<<<<<<< HEAD
# Hybrid_Network_Models
Working repo for ENPH455 Thesis on Hybrid Quantum-Classical Neural Networks
=======
# Hybrid_Network_Models
This repo uses the Sacred package to manage the parameter sweeps and experiment runs (https://sacred.readthedocs.io/en/stable/quickstart.html). 
We use PennyLane to build the CV networks (https://pennylane.readthedocs.io/en/stable/) and use the "strawberryfields.tf" backend.

Please see the Demo_Experiment for an up-to-date example of how to properly generate and run CV network experiments.
This README references this as the example. 

### Files
1. CV_quantum_layers: the backend for building CV Keras layers
2. SLURM_Script_Factory: builds the script to run the parameter sweeps on SLURM
3. Run_CIFAR_Experiment.py: defines the parameter grids and entry points to run the parameter sweep, or build the SLURM script
4. RUN_CIFAR_Experiment.sh: autogenerate bash script to run experiment on cluster build from Run_CIFAR_Experiment.py
5. CIFAR_Experiment.py: the main neural network and model run
6. CIFAR_Dataset.py: the pre-processing of the dataset we are training with

### Output
The output experiment data contains four files: 
1. config.json: the parameters used from the grid
2. cout.txt: the console output for the test, typically showing the training progress
3. metrics.json: the metrics we record for the run, such as accuracy at each step
4. run.json: run details to uniquely identify the run

### How to Run
To run the experiments directly: 
```
python Run_CIFAR_Experiment.py run
```
To build a SLURM script
```
python Run_CIFAR_Experiment.py slurm
```
The resulting Run_CIFAR_Experiment.sh should then be launched on the cluster. This can be done, for example, by 
cloning the git repo, and pushing the results when done. 

### Notable Points
The Demo_Experiment has been updated with enough documentation to make things clear. A few notable points: 
- The get_max_input() function goal and implementation are both unclear. We need to revise the cutoff dimension management strategy from the top down and carefully define each part. 
- The CV Keras interface has been updated to use classes to simplify the data encoding, measurement, and pre-quantum activation layer
- No post-process script has been added in this demo. It is recommended that we setup a MongoDB (i.e. with Atlas) and use a visualizer like omniboard (https://github.com/vivekratnavel/omniboard). For publication, we can always download the data, and make them prettier to our desired format
- Previously, the classical, qubit, and CV networks were all done in the same experiment. This resulted in a great deal of clutter. Each of qubit, CV, classical has their own parameters and should thus be divided into their own experiments (i.e. Demo_Experiment_CV, Demo_Experiment_qubit, Demo_Experiment_classical)
- The CV_quantum_layers.py and SLURM_Script_Factory.py should be placed in the main directory because they are common to all experiments 

## Next Steps
To make the code results publication-ready, we need the following: 
1. Check if we can push results directly to MongoDB - this will vastly speed up our post-processing and mean we don't need to track the experiment results in our repo
2. Refine the cutoff dimension management strategy and finalize our CV approach in CV_quantum_layers.py (this will include moving the get_max_input() and any other algorithms we come up with inside CV_quantum_layers.py)
3. Come up with a testing plan to carefully validate our cutoff dimension strategy (this may involve tracking normalization over epochs) 
4. Select better datasets and pre-process them 
>>>>>>> 6f14f7656b8a613a22383be9bc0b4d80fb2017b9
