import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

x = pd.read_pickle("real_amplitudes-batch-25.pkl")
n_qubits_nn = x['n_qubits'].to_numpy()
n_blocks_nn = x['n_blocks'].to_numpy()
times_nn = x['time'].to_numpy()

x = pd.read_pickle("real_amplitudes.pkl")
n_qubits_non_cv = x['n_qubits'].to_numpy()
n_blocks_non_cv = x['n_blocks'].to_numpy()
times_non_cv  = x['time'].to_numpy()

#%%
x = pd.read_pickle("cv_neural_net.pkl")
n_qubits = x['n_qubits'].to_numpy()
n_blocks = x['n_blocks'].to_numpy()
times = x['time'].to_numpy()
cutoff_dim = x['cutoff_dim'].to_numpy()
weights = x['n_weights'].to_numpy()

#%%
n_data_pts = 100

t = np.divide(times_nn,times_non_cv)
print(t)
print(np.average(t))
scale = 5*60000*(np.average(t)/100)

print(weights)

#%%
fig, axes = plt.subplots(figsize=(5,4))
for i in range(2, 6):
    axes.plot(weights[n_qubits==i], scale*times[n_qubits==i]/3600)
axes.set_ylabel("Time (h)")
axes.set_xlabel("Number of parameters")
labels = [str(i) + " Qubits" for i in range(2,6)]
fig.legend(labels=labels, loc="upper left", bbox_to_anchor=(0.2, 0.8), ncol=2)
plt.savefig('CV4.pdf', format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

#%%
fig, axes = plt.subplots(figsize=(5,4))
for i in range(1, 11):
    axes.plot(n_qubits[n_blocks==i], n_weights[n_blocks==i])
axes.set_ylabel("Number of Weights")
axes.set_xlabel("Number of Qubits")
plt.show()
