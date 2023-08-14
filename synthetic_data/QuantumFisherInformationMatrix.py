"""Quantum Fisher Information Matrix
"""
#%%
%reload_ext autoreload
%autoreload 2
import sys
import numpy as np
import tensorflow as tf
import sys
sys.path.append("../")
import copy
from sacred import Experiment
import pennylane as qml
from common_packages.utilities import get_equivalent_classical_layer_size, get_num_parameters_per_quantum_layer
import pandas as pd
colors =   ["#5dd448",
            "#bfa900",
            "#ec742f",
            "#e9496f",
            "#b04ca4",]



from quantum_base import Net as Net_orig
from quantum_base import OPTIMIZER, LOSS_FUNCTION
from quantum_base_kerr import Net as Net_kerr
from data import generate_synthetic_dataset_easy
train_data, validate_data = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=8, n_classes=4)

train_data_kerr, validate_data_kerr = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=15, n_classes=4)

orig_ex_folder = '/home/st4eve/Mounts/graham/synthetic_data/Synthetic_Quantum_Base_Experiment_cutoff_sweep/'
kerr_ex_folder = '/home/st4eve/Mounts/graham/synthetic_data/Synthetic_Quantum_Base_Kerr/'

df_orig = pd.read_pickle('df_orig.pkl')
df_kerr = pd.read_pickle('df_kerr.pkl')
for metric in ['acc', 'loss', 'val_acc', 'val_loss']:
    df_orig[metric] = df_orig[metric].apply(lambda x: x[-1])
    df_kerr[metric] = df_kerr[metric].apply(lambda x: x[-1])
df_quantum = df_orig[df_orig['network_type']=='quantum']
df_classical = df_orig[df_orig['network_type']=='classical']

n = 2
c = 3
nl = 1
metric = 'acc'

exp_orig = {}
exp_orig['quantum'] = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c) & (df_quantum['n_layers']==nl)][metric]
exp_orig['classical'] = df_classical.loc[(df_classical['num_qumodes']==n)  & (df_classical['n_layers']==nl)][metric]

df_quantum = df_kerr[df_kerr['network_type']=='quantum']
df_classical = df_kerr[df_kerr['network_type']=='classical']
exp_kerr = {}

c_kerr = 5

exp_kerr['quantum'] = df_quantum.loc[(df_quantum['num_qumodes']==n) & (df_quantum['cutoff']==c_kerr) & (df_quantum['n_layers']==nl)][metric]
exp_kerr['classical'] = df_classical.loc[(df_classical['num_qumodes']==n)  & (df_classical['n_layers']==nl)][metric]
#%%
model = Net_orig(network_type='quantum',
                 num_qumodes=n,
                 cutoff=c,
                 n_layers=nl,
                 max_initial_weight=0.1)

exp_quantum = exp_orig['quantum'].idxmax()

model.load_weights(f'{orig_ex_folder}{exp_quantum}/weights/weight99.ckpt').expect_partial()
model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=["accuracy"])

model(validate_data[0][0:1])
#%%
inputs = model.quantum_preparation_layer(model.input_layer(validate_data[0][0:1]))
density_matrix = model.quantum_layer.get_density_matrix(inputs)
print(density_matrix)
# qnode = model.quantum_layer.qnodes[0]
# input_layer = model.input_layer
# quantum_prep_layer = model.quantum_preparation_layer
# weights = model.quantum_layer.weights
# params = tf.constant(quantum_prep_layer(input_layer(validate_data[0][0:1])))
# print(params[0][0])
# print(qnode(*params, *weights))
# cfim = qml.qinfo.classical_fisher(qnode)(*params, *weights)
#print(cfim)
# %%