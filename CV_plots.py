import strawberryfields as sf
from strawberryfields.ops import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.patches as patches

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5


def plot_pdf(X, P, Z, source_pt=None, dest_pt=None, line_curve=False, save_name=None):
    """Plots the probability density function over all time steps."""
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
        plt.savefig(save_name + '.pdf',
                    format='pdf',
                    dpi=100,
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

plot_pdf(X, P, Z, save_name="Plots/VacuumState")

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

plot_pdf(X, P, Z, source_pt, dest_pt, line_curve=True, save_name="Plots/RotationGate")

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
    Dgate(3, 7*np.pi/4) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z2 = state.wigner(0, X, P)
dest_pt = (2*4*np.cos((1/4)*np.pi), 2*4*np.sin((1/4)*np.pi))

Z = Z1+Z2

plot_pdf(X, P, Z, source_pt, dest_pt, line_curve=False, save_name="Plots/DisplacementGate")

#%% Squeezed State
prog = sf.Program(1)
with prog.context as q:
    Dgate(2, 0) | q[0]
    Sgate(-0.7) | q[0]
eng = sf.Engine('gaussian')
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="Plots/SqueezedState")

#%% Cubic Phase Gate
prog = sf.Program(1)
with prog.context as q:
    Vgate(3) | q[0]
eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="Plots/CubicPhaseGate")

#%% Kerr Gate
prog = sf.Program(1)
with prog.context as q:
    Dgate(2) | q[0]
    Kgate(3) | q[0]
eng = sf.Engine('fock', backend_options={"cutoff_dim": 10})
state = eng.run(prog).state
Z = state.wigner(0, X, P)

plot_pdf(X, P, Z, save_name="Plots/KerrGate")

#%% Fock Cutoff effect
cutoff_dim=30
dev = qml.device('strawberryfields.tf', wires=1, cutoff_dim=cutoff_dim, hbar=1)
@qml.qnode(dev, interface="tf")
def circuit(x, theta):
    qml.Displacement(x, theta, wires=0)
    return qml.probs(wires=0)

x = np.arange(0,cutoff_dim,1)
outputs1 = circuit(3, 0)
outputs2 = circuit(5, 0)

integral1 = np.sum(outputs1)
print(integral1)

integral2 = np.sum(outputs2)
print(integral2)

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

plt.savefig("Plots/Fock_Cutoff" + '.pdf',
            format='pdf',
            dpi=100,
            bbox_inches='tight')

plt.show()
