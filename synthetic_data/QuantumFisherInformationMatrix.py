"""Quantum Fisher Information Matrix
"""
#%%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('pdf')
import sys
sys.path.append("../")
import copy
import pennylane as qml
from Plotting.generate_database import ResultsDatabaseGenerator
from common_packages.utilities import get_equivalent_classical_layer_size, get_num_parameters_per_quantum_layer
from Plotting.Plot import BasicPlot, MultiPlot
import pandas as pd
colors =   ["#5dd448",
            "#bfa900",
            "#ec742f",
            "#e9496f",
            "#b04ca4",]

colors = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 0.5

import numpy as np
from quantum_base import Net as Net_orig
from quantum_base import OPTIMIZER, LOSS_FUNCTION
from quantum_base_kerr import Net as Net_kerr
from data import generate_synthetic_dataset_easy
import seaborn as sns
from itertools import product
test_data, validate_data = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=8, n_classes=4)

test_data_kerr, validate_data_kerr = generate_synthetic_dataset_easy(num_datapoints=1000, n_features=15, n_classes=4)

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

#model(test_data[0])

print(model.quantum_layer.qnodes)

#qfim = qml.qinfo.tranforms.quantum_fisher(model.qnodes[0])
