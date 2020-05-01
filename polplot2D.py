#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import figure, show
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM.main_plotter import gen_olaf_cmap





def polPlot2D(station_loc, pulses_PE, source_info, ell_scale=2**50, ϕ_shift=False, cmap=None):
	fig = figure(figsize=(20,10)) #Az : [-90,90] and Z : [0:90] => 2:1 ratio ensures we fill canvas properly
	frame = fig.add_subplot(111, aspect='equal') #equal aspect ensures correct direction of polarization ellipse!

	#setting up colormap
	if cmap is None:
		cmap = gen_olaf_cmap()
	T = np.array([])
	for pulseID in pulses_PE.keys():
		T = np.append(T,source_info[pulseID]['XYZT'][3])
	vmin = np.min(T)
	vmax = np.max(T)
	T = 1/(vmax-vmin)*(T-vmin) #normalize T


	for i, pulseID in enumerate(pulses_PE.keys()):
		loc = source_info[pulseID]['XYZT'][:3] - station_loc
		Az = np.arctan(loc[1]/loc[0])
		Z = np.arctan(np.dot(loc[:2], loc[:2])**0.5/loc[2])
		Az = np.rad2deg(Az); Z = np.rad2deg(Z)

		#ensures clusterings at +90 and -90 degrees are brought to zero (reducing x scale)
		if ϕ_shift:
			if Az<0:
				Az += 90
			elif Az>0:
				Az -= 90

		#compute the ellipses
		width = 2*pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.cos(pulses_PE[pulseID][3])
		height = 2*pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])
		width *= ell_scale; height *= ell_scale
		angle = pulses_PE[pulseID][2]

		x, y = Ellipse((Az,Z),width,height,angle)

		frame.plot(x, y, color=cmap(T[i]))
		frame.scatter(Az, Z, s=3, marker='o', color=cmap(T[i]))

	#flip zenithal axis such that overhead it actually up!
	ylimits = frame.get_ylim()
	frame.set_ylim((ylimits[1],ylimits[0]))

	#setting up colorbar
	norm = mpl.colors.Normalize(vmin=vmin*1000, vmax=vmax*1000) #set time in ms
	divider = make_axes_locatable(frame)
	cax = divider.append_axes("right", size="1%", pad=0.03)
	cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

	#set axis labels
	frame.set_xlabel(r"$\phi\ [^\circ]$", fontsize=16)
	frame.set_ylabel(r"$\theta_z\ [^\circ]$", fontsize=16)
	cbar.set_label(label=r"$t\ [ms]$",fontsize=16)
	
	frame.grid()

	show()





#function that returns datapoints forming an ellipse (patches.Ellipse does not work properly for our purposes)
def Ellipse(pos,width,height,angle):
	t = np.linspace(0,2*np.pi,100)
	x = np.array([width/2*np.cos(t),height/2*np.sin(t)])

	#rotation matrix anti-clockwise
	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

	x = np.matmul(R,x)
	x = x + np.array([np.full_like(x[0],pos[0]),np.full_like(x[1],pos[1])])
	return x