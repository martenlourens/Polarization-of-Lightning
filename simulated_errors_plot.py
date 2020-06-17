#!/usr/bin/env python3

import numpy as np
from matplotlib.pyplot import figure, subplots_adjust, show
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM.main_plotter import gen_olaf_cmap

sim_err = np.loadtxt("simulated_errors.txt", comments='#', delimiter=' ')

#unpack the different variables
τ = sim_err[:, 0]
ε = sim_err[:, 1]
τ_err = sim_err[:, 2]
ε_err = sim_err[:, 3]

N = 100
#generate meshgrids
τ = τ.reshape(2*N, N) #y variable grid in radians
ε = ε.reshape(N, 2*N) #x variable grid in radians
τ_err = np.rad2deg(τ_err.reshape(2*N, N)) #τ_err grid in degrees
ε_err = np.rad2deg(ε_err.reshape(2*N, N)) #ε_err grid in degrees

xlimits = (np.rad2deg(np.min(ε)), np.rad2deg(np.max(ε)))
ylimits = (np.rad2deg(np.min(τ)), np.rad2deg(np.max(τ)))

cmap = gen_olaf_cmap()

fig = figure(figsize=(15, 10))
fig.subplots_adjust(wspace=0)

frame1 = fig.add_subplot(121)
frame1.set_title(r"$\tau_{error}$", fontsize=18, pad=9)
norm1 = colors.Normalize(np.min(τ_err), np.max(τ_err))
frame1.imshow(τ_err, cmap=cmap, norm=norm1, aspect='equal', origin='lower', extent=(xlimits[0], xlimits[1], ylimits[0], ylimits[1]))
frame1.set_xlabel(r"$\epsilon\ [^{\circ}]$", fontsize=16)
frame1.set_ylabel(r"$\tau\ [^{\circ}]$", fontsize=16)
divider1 = make_axes_locatable(frame1)
cax1 = divider1.append_axes("right", size="5%", pad=0.25)
cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm1, cmap=cmap), cax=cax1)
cbar1.set_label(label=r"$\tau_{err}\ [^{\circ}]$", fontsize=16)
frame1.grid()

frame2 = fig.add_subplot(122)
frame2.set_title(r"$\epsilon_{error}$", fontsize=18, pad=9)
norm2 = colors.Normalize(np.min(ε_err), np.max(ε_err))
frame2.imshow(ε_err, cmap=cmap, norm=norm2, aspect='equal', origin='lower', extent=(xlimits[0], xlimits[1], ylimits[0], ylimits[1]))
frame2.set_xlabel(r"$\epsilon\ [^{\circ}]$", fontsize=16)
frame2.set_ylabel(r"$\tau\ [^{\circ}]$", fontsize=16)
divider2 = make_axes_locatable(frame2)
cax2 = divider2.append_axes("right", size="5%", pad=0.25)
cbar2 = fig.colorbar(cm.ScalarMappable(norm=norm2, cmap=cmap), cax=cax2)
cbar2.set_label(label=r"$\epsilon_{err}\ [^{\circ}]$", fontsize=16)
frame2.grid()

#show()
fig.savefig("simulated_errors.pdf", dpi=fig.dpi, bbox_inches='tight')