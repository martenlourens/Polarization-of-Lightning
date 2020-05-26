#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import figure, show
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM.main_plotter import gen_olaf_cmap





def polPlot2D(station_loc, pulses_PE, source_info, ell_scale=2**50, ϕ_shift=False, cmap=None, errors=False, save=False, fig=None, frame=None):
	if fig is None:
		fig = figure(figsize=(20,10),dpi=108) #Az : [-90,90] and Z : [0:90] => 2:1 ratio ensures we fill canvas properly
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
		Az = np.arctan2(loc[1], loc[0])
		Z = np.arctan2(np.dot(loc[:2], loc[:2])**0.5, loc[2])
		Az = np.rad2deg(Az); Z = np.rad2deg(Z)

		#ensures clusterings at +90 and -90 degrees are brought to zero (reducing x scale)
		if ϕ_shift:
			if Az<0:
				Az += 180
			elif Az>0:
				Az -= 180


		#compute ellipse parameters
		width = 2*pulses_PE[pulseID][0]*np.cos(pulses_PE[pulseID][3])
		height = 2*pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])
		width *= ell_scale; height *= ell_scale
		angle = pulses_PE[pulseID][2]

		#alternative:
		#width = 2*pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.cos(pulses_PE[pulseID][3])
		#height = 2*pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.sin(pulses_PE[pulseID][3])
		#width *= ell_scale; height *= ell_scale
		#angle = pulses_PE[pulseID][2]

		if errors:
			#compute ellipse parameter errors
			width_err = 2*np.sqrt((np.cos(pulses_PE[pulseID][3])*pulses_PE[pulseID][4])**2 + (pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])*pulses_PE[pulseID][7])**2)
			height_err = 2*np.sqrt((np.sin(pulses_PE[pulseID][3])*pulses_PE[pulseID][4])**2 + (pulses_PE[pulseID][0]*np.cos(pulses_PE[pulseID][3])*pulses_PE[pulseID][7])**2)
			width_err *= ell_scale; height_err *= ell_scale
			angle_err = pulses_PE[pulseID][6]

			#alternative:
			#width_err = 2*np.sqrt((pulses_PE[pulseID][1]*np.cos(pulses_PE[pulseID][3])*pulses_PE[pulseID][4])**2 + (pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.sin(pulses_PE[pulseID][3])*pulses_PE[pulseID][7])**2 + (pulses_PE[pulseID][0]*np.cos(pulses_PE[pulseID][3])*pulses_PE[pulseID][5])**2)
			#height_err = 2*np.sqrt((pulses_PE[pulseID][1]*np.sin(pulses_PE[pulseID][3])*pulses_PE[pulseID][4])**2 + (pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.cos(pulses_PE[pulseID][3])*pulses_PE[pulseID][7])**2 + (pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])*pulses_PE[pulseID][5])**2)
			#width_err *= ell_scale; height_err *= ell_scale
			#angle_err = pulses_PE[pulseID][6]

			#compute error in position
			covXYZ = source_info[pulseID]['covXYZ']
			
			rsq = np.dot(loc[:2],loc[:2])
			#dϕ/dx
			part_Az_x = -loc[1]/rsq
			#dϕ/dy
			part_Az_y = loc[0]/rsq

			ρsq = np.dot(loc,loc)
			#dθ_z/dx
			part_Z_x = 1/ρsq*loc[0]*loc[2]/np.sqrt(rsq)
			#dθ_z/dy
			part_Z_y = 1/ρsq*loc[1]*loc[2]/np.sqrt(rsq)
			#dθ_z/dz
			part_Z_z = -np.sqrt(rsq)/ρsq

			pos_cov = np.array([[0,0],[0,0]])
			pos_cov[0][0] = (part_Az_x*covXYZ[0][0])**2 + (part_Az_y*covXYZ[1][1])**2 + part_Az_x*part_Az_y*covXYZ[0][1] + part_Az_x*part_Az_y*covXYZ[1][0]
			pos_cov[0][1] = part_Az_x*part_Z_x*covXYZ[0][0] + part_Az_y*part_Z_y*covXYZ[1][1] + part_Az_x*part_Z_y*covXYZ[0][1] + part_Az_x*part_Z_y*covXYZ[1][0] + part_Az_y*part_Z_x*covXYZ[0][1] + part_Az_y*part_Z_x*covXYZ[1][0] + part_Az_x*part_Z_z*covXYZ[0][2] + part_Az_x*part_Z_z*covXYZ[2][0] + part_Az_y*part_Z_z*covXYZ[0][2] + part_Az_y*part_Z_z*covXYZ[2][0]
			pos_cov[1][0] = pos_cov[0][1]
			pos_cov[1][1] = (part_Z_x*covXYZ[0][0])**2 + (part_Z_y*covXYZ[1][1])**2 + (part_Z_z*covXYZ[2][2])**2 + part_Z_x*part_Z_y*covXYZ[0][1] + part_Z_x*part_Z_y*covXYZ[1][0] + part_Z_x*part_Z_z*covXYZ[0][2] + part_Z_x*part_Z_z*covXYZ[2][0] + part_Z_y*part_Z_z*covXYZ[1][2] + part_Z_y*part_Z_z*covXYZ[2][1]
			

			#compute ellipse parametrization
			r, r_err = Ellipse((Az,Z), width, height,angle, pos_cov=pos_cov, width_err=width_err, height_err=height_err, angle_err=angle_err)
			
			#plot the ellipses with errorbars
			frame.errorbar(r[0], r[1], xerr=r_err[0], yerr=r_err[1], color=cmap(T[i]), linewidth=0.75, alpha=0.5, capsize=2, capthick=0.25, elinewidth=0.5, ecolor='k')
			
			#plot the pulses
			frame.errorbar(Az, Z, xerr=pos_cov[0][0], yerr=pos_cov[1][1], markersize=1, marker='s', color=cmap(T[i]), alpha=0.75, capsize=2, capthick=0.25, elinewidth=0.5, ecolor='k')

		else:
			#compute ellipse parametrization
			x, y = Ellipse((Az,Z),width,height,angle)

			#plot the ellipses
			frame.plot(x, y, color=cmap(T[i]), linewidth=0.75, alpha=0.75)
		
			#plot the pulses
			frame.scatter(Az, Z, s=4, marker='s', edgecolor='k', linewidths=0.4, color=cmap(T[i]), alpha=0.75)

	#flip zenithal axis such that overhead it actually up!
	ylimits = frame.get_ylim()
	frame.set_ylim((ylimits[1],ylimits[0]))

	#flip azimuthal axis such that left and right is actually left and right from the station's perspective
	xlimits = frame.get_xlim()
	frame.set_xlim((xlimits[1],xlimits[0]))

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

	if save:
		return fig
	
	show()





#function that returns datapoints forming an Ellipse. (patches.Ellipse does not work properly for our purposes)
def Ellipse(pos, width, height, angle, pos_cov=np.array([[0,0],[0,0]]), width_err=None, height_err=None, angle_err=None):
	t = np.linspace(0,2*np.pi,100)
	x = np.array([width/2*np.cos(t),height/2*np.sin(t)])

	#rotation matrix anti-clockwise
	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

	x = np.matmul(R,x)
	x = x + np.array([np.full_like(x[0],pos[0]),np.full_like(x[1],pos[1])])

	if not None in [width_err, height_err, angle_err]:

		σ = np.array([width_err, height_err, angle_err, pos_cov[0][0]**0.5, pos_cov[1][1]**0.5])
		
		#partials are initiated:
		#dx/dwidth
		part_width = np.matmul(R, np.array([1/2*np.cos(t),np.zeros(t.size)]))

		#dx/dheight
		part_height = np.matmul(R, np.array([np.zeros(t.size),1/2*np.sin(t)]))

		#dx/dangle
		#derivative of rotation matrix anti-clockwise
		part_R = np.array([[-np.sin(angle), -np.cos(angle)], [np.cos(angle), -np.sin(angle)]])
		part_angle = np.matmul(part_R, np.array([width/2*np.cos(t),height/2*np.sin(t)]))

		#dx/dpos0
		part_pos0 = np.array([np.ones(t.size),np.zeros(t.size)])

		#dx/dpos1
		part_pos1 = np.array([np.zeros(t.size),np.ones(t.size)])

		parts = np.array([part_width, part_height, part_angle, part_pos0, part_pos1])

		x_var = np.zeros((2, t.size))
		for i in range(5):
			x_var += np.multiply(parts[i]*σ[i], parts[i]*σ[i])

		return x, np.sqrt(x_var)

	return x