#%%
import strawberryfields as sf
from strawberryfields.ops import *
import pennylane as qml

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.patches as patches

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5

#%% PDF plot
def plot_pdf(X, P, Z, source_pt=None, dest_pt=None, line_curve=False, save_name=None):
    """CV_Plots the probability density function over all time steps."""
    fig, axes = plt.subplots(1, figsize=(2.8, 2.8))
    cs = axes.contourf(X, P, Z, cmap='Blues')
    axes.set_xlabel("x (a.u.)")
    axes.set_ylabel("p (a.u.)",labelpad=-8)
    axes.set_xlim(-10,10)
    axes.set_ylim(-10,10)


    axes.hlines(y=0, xmin=-10, xmax=10, color='black', lw=0.8)
    axes.vlines(x=0, ymin=-10, ymax = 10, color='black', lw=0.8)

    norm = mpl.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ticks=cs.levels)
    cbar.ax.set_ylabel("$|\Psi(x,t)|^2$ (a.u.)")
    cbar.ax.tick_params(size=0)

    if(source_pt is not None and dest_pt is not None):
        style = "Simple, tail_width=0.5, head_width=4, head_length=8"
        kw = dict(arrowstyle=style, color="k")
        if line_curve:
            a = patches.FancyArrowPatch(source_pt, dest_pt, connectionstyle="arc3,rad=0.416667", **kw)
        else:
            a = patches.FancyArrowPatch(source_pt, dest_pt, **kw)
        plt.gca().add_patch(a)

    if (save_name is not None):
        plt.savefig(save_name + '.svg',
                    format='svg',
                    dpi=300,
                    bbox_inches='tight')
    fig.tight_layout(pad=0.5)
    plt.show()
#%%
def plot_pdf_overlay(X, P, Z1, Z2, source_pt=None, dest_pt=None, line_curve=False, save_name=None):
    """CV_Plots the probability density function over all time steps."""
    fig, axes = plt.subplots(1, figsize=(2.8, 2.8))
    cs = axes.contourf(X, P, Z1, cmap='Blues', alpha=0.75)
    cs2 = axes.contourf(X, P, Z2, cmap='Reds', alpha=0.3)
    axes.set_xlabel("x (a.u.)")
    axes.set_ylabel("p (a.u.)",labelpad=-8)
    axes.set_xlim(-10,10)
    axes.set_ylim(-10,10)


    axes.hlines(y=0, xmin=-10, xmax=10, color='black', lw=0.8)
    axes.vlines(x=0, ymin=-10, ymax = 10, color='black', lw=0.8)

    if(source_pt is not None and dest_pt is not None):
        for i in range(len(source_pt)):
            style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            kw = dict(arrowstyle=style, color="k")
            if line_curve:
                a = patches.FancyArrowPatch(source_pt[i], dest_pt[i], connectionstyle="arc3,rad=0.416667", **kw)
            else:
                a = patches.FancyArrowPatch(source_pt[i], dest_pt[i], **kw)
            plt.gca().add_patch(a)

    if (save_name is not None):
        plt.savefig(save_name + '.svg',
                    format='svg',
                    dpi=300,
                    bbox_inches='tight')
    fig.tight_layout(pad=0.5)
    plt.show()

X = np.linspace(-10, 10, 1000)
P = np.linspace(-10, 10, 1000)

#%% Vacuum State
prog = sf.Program(1)
with prog.context as q:
    Vac | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="VacuumState")

#%% Rotation Gate
prog = sf.Program(1)
with prog.context as q:
    Dgate(4, np.pi/4) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z1 = state.wigner(0, X, P)
source_pt = (2*4*np.cos(np.pi/4), 2*4*np.sin(np.pi/4))

prog = sf.Program(1)
with prog.context as q:
    Dgate(4, np.pi/4) | q[0]
    Rgate((2/3)*np.pi ) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z2 = state.wigner(0, X, P)
dest_pt = (2*4*np.cos(np.pi/4+(2/3)*np.pi), 2*4*np.sin(np.pi/4+(2/3)*np.pi))

Z = Z1+Z2

plot_pdf(X, P, Z, source_pt, dest_pt, line_curve=True, save_name="RotationGate")

#%% Displacement Gate
prog = sf.Program(1)
with prog.context as q:
    Vac | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z1 = state.wigner(0, X, P)
source_pt = (0,0)

prog = sf.Program(1)
with prog.context as q:
    Dgate(4, np.pi/4) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z2 = state.wigner(0, X, P)
dest_pt = (6*np.cos((1/4)*np.pi), 6*np.sin((1/4)*np.pi))

Z = Z1+Z2

plot_pdf(X, P, Z, source_pt, dest_pt, line_curve=False, save_name="DisplacementGate")

#%% Squeezed State
prog = sf.Program(1)
with prog.context as q:
    Sgate(-1, 0) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="SqueezedVacuumState")

#%% Squeezed State
prog = sf.Program(1)
with prog.context as q:
    Dgate(1, 0) | q[0]
    Sgate(-1, 0) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="SqueezedState")

#%% Squeezed State
prog = sf.Program(1)
with prog.context as q:
    Sgate(-1, 0) | q[0]
    Dgate(2, np.pi/4) | q[0]
    eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="SqueezedRotatedState")

#%% Two-Mode Squeezed State
prog = sf.Program(2)
with prog.context as q:
    BSgate(np.pi/4, 0) | (q[0], q[1])
    Sgate(2,0) | (q[0])
    #Sgate(-2,0) | (q[1])
    BSgate(np.pi/4, 0) | (q[0], q[1])
    Rgate(np.pi/2) | q[0]
    eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z_1 = state.wigner(0, X, P)
Z_2 = state.wigner(1, X, P)

plot_pdf(X, P, Z_1, save_name="TwoModeSqueezedState1.1")
plot_pdf(X, P, Z_2, save_name="TwoModeSqueezedState1.2")

#%% Two-Mode Squeezed State
prog = sf.Program(2)
with prog.context as q:
    S2gate(1,0) | (q[0], q[1])
    BSgate() | (q[0], q[1])
    Rgate(np.pi/2) | q[0] 
    BSgate() | (q[0], q[1])
    eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z_1 = state.wigner(0, X, P)
Z_2 = state.wigner(1, X, P)

plot_pdf(X, P, Z_1, save_name="TwoModeSqueezedState2.1")
plot_pdf(X, P, Z_2, save_name="TwoModeSqueezedState2.2")

#

#%% Cubic Phase Gate
prog = sf.Program(1)
with prog.context as q:
    Vgate(-3) | q[0]
eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="CubicPhaseGate")

#%% Kerr Gate
prog = sf.Program(1)
with prog.context as q:
    Dgate(3) | q[0]
    Kgate(np.pi) | q[0]
eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="KerrGate")

#%% Fock Cutoff effect
cutoff_dim=30
dev = qml.device('strawberryfields.tf', wires=1, cutoff_dim=cutoff_dim, hbar=1)
@qml.qnode(dev, interface="tf")
def circuit(x, theta):
    qml.Displacement(x, theta, wires=0)
    qml.Squeezing(1, np.pi/8, wires=0)
    return qml.probs(wires=0)

x = np.arange(0,cutoff_dim,1)
outputs1 = circuit(3, np.pi)
outputs2 = circuit(5, 0)

integral1 = np.sum(outputs1)
print(integral1)

integral2 = np.sum(outputs2)
print(integral2)
#%% Original Coherent state
alpha = 1+0.5j
r = np.abs(alpha)
phi = np.angle(alpha)

prog = sf.Program(1)
with prog.context as q:
    Coherent(r, phi) | q[0]
    eng = sf.Engine('gaussian')
    
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="Coherent_Before")
#%% Teleportation
alpha = 1+0.5j
r = np.abs(alpha)
phi = np.angle(alpha)

prog = sf.Program(3)
with prog.context as q:
    # prepare initial states
    Coherent(r, phi) | q[0]
    Squeezed(-2) | q[1]
    Squeezed(2) | q[2]

    # apply gates
    BS = BSgate(np.pi/4, np.pi)
    BS | (q[1], q[2])
    BS | (q[0], q[1])

    # Perform homodyne measurements
    MeasureX | q[0]
    MeasureP | q[1]

    # Displacement gates conditioned on
    # the measurements
    Xgate(np.sqrt(2) * q[0].par) | q[2]
    Zgate(-np.sqrt(2) * q[1].par) | q[2]

    eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z_1 = state.wigner(0, X, P)
Z_2 = state.wigner(1, X, P)
Z_3 = state.wigner(2, X, P)
plot_pdf(X, P, Z_1, save_name="Tele_1")
plot_pdf(X, P, Z_2, save_name="Tele_2")
plot_pdf(X, P, Z_3, save_name="Tele_3")

#%%
# Create plot
fig, axes = plt.subplots(figsize=(3.2,2.4))

# Create settings
axes.plot(x, outputs1, 'r', linewidth=1.0)
axes.plot(x, outputs2, 'b', linewidth=1.0)
axes.vlines(x=cutoff_dim-1, ymin=0, ymax=0.16, color='black', lw=1, ls='--')
axes.xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
axes.yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
axes.set_xlabel("Fock State (n)")
axes.set_xlim(-3, 33)
axes.set_ylim(-0.01, 0.15)
axes.set_ylabel("Probability (a.u.)")
axes.grid(True, linestyle=':')
fig.tight_layout(pad=0.5)

plt.savefig("CV_Plots/Fock_Cutoff" + '.pdf',
            format='pdf',
            dpi=100,
            bbox_inches='tight')

plt.show()

#%% Beam Splitter
prog = sf.Program(2)
with prog.context as q:
    Dgate(4, np.pi/4) | q[0]
    Dgate(2, np.pi) | q[1]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z0_0 = state.wigner(1, X, P)
Z1_0 = state.wigner(0, X, P)

prog = sf.Program(2)
with prog.context as q:
    Dgate(4, np.pi/4) | q[0]
    Dgate(2, np.pi) | q[1]
    BSgate(9*np.pi/5, 0) | (q[0],q[1])
eng = sf.Engine('gaussian')
state = eng.run(prog).state

Z0_1 = state.wigner(1, X, P)
Z1_1 = state.wigner(0, X, P)
source_pt1 = (8*np.cos(np.pi/4), 8*np.sin(np.pi/4))
dest_pt1 = (-7,-4.8)

Z1 = Z0_0+Z0_1
Z2 = Z1_0+Z1_1
source_pt2 = (4*np.cos(np.pi), 4*np.sin(np.pi))
dest_pt2 = (0,-3.5)

plot_pdf_overlay(X, P, Z1, Z2,save_name="BeamSplitterGate")

#plot_pdf_overlay(X, P, Z1, Z2, source_pt=[source_pt1, source_pt2], dest_pt=[dest_pt1, dest_pt2], line_curve=False, save_name="CV_Plots/BeamSplitterGate")
#%% Cutoff Displacement Plot
def find_max_displacement(cutoff_dim, min_normalization):
    cutoff_dim = int(cutoff_dim)
    dev = qml.device("strawberryfields.tf", wires=1, cutoff_dim=cutoff_dim)
    @qml.qnode(dev, interface="tf")
    def qc(a):
        qml.Displacement(a, 0, wires=0)
        return qml.probs(wires=0)

    a = 0
    norm = 1
    while(norm>min_normalization):
        fock_dist = qc(a)
        norm = np.sum(fock_dist)
        a+=0.02

    return a

max_cutoff = 50
min_normalizations = np.array([0.90, 0.95, 0.99])

cutoff_dims = np.arange(1,max_cutoff+1,1, dtype='int')

max_displacement = np.zeros((len(min_normalizations),len(cutoff_dims)))
for i in range(len(cutoff_dims)):
    for j in range(len(min_normalizations)):
        max_displacement[j][i] = find_max_displacement(cutoff_dims[i], min_normalizations[j])

# Create plot
fig, axes = plt.subplots(figsize=(3.2,2.4))

# Create settings
axes.plot(cutoff_dims, max_displacement[0], 'r', linewidth=1.0)
axes.plot(cutoff_dims, max_displacement[1], 'b', linewidth=1.0)
axes.plot(cutoff_dims, max_displacement[2], 'g', linewidth=1.0)
#axes.vlines(x=cutoff_dim-1, ymin=0, ymax=0.16, color='black', lw=1, ls='--')
axes.xaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
axes.yaxis.set_tick_params(which='major', size=3, width=0.5, direction='in', right='on')
axes.xaxis.set_ticks(np.arange(0, max_cutoff+1, 5))
axes.set_xlabel("Cutoff Dimension")
axes.set_ylabel("Maximum Displacement \n Magnitude " r"($|\alpha|$)")
axes.grid(True, linestyle=':')
fig.legend(["0.99", "0.95", "0.90"], loc="upper left", bbox_to_anchor=(0.2,0.95))
fig.tight_layout(pad=0.5)

plt.savefig("Max_Displacement_Cutoff" + '.pdf',
            format='pdf',
            dpi=100,
            bbox_inches='tight')

plt.show()

# %%
