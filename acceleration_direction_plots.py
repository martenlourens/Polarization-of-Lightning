#!/usr/bin/env python3

import os

import numpy as np
import json
from matplotlib.pyplot import figure, show, setp
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_acceleration_vector

timeID = "D20190424T210306.154Z"

utilities.default_raw_data_loc = "/home/student4/Marten/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"
processed_data_folder = processed_data_dir(timeID)

station_delay_file = "station_delays.txt"
polarization_flips = "polarization_flips.txt"
bad_antennas = "bad_antennas.txt"
additional_antenna_delays = "ant_delays.txt"

polarization_flips = read_antenna_pol_flips(processed_data_folder + '/' + polarization_flips)
bad_antennas = read_bad_antennas(processed_data_folder + '/' + bad_antennas)
additional_antenna_delays = read_antenna_delays(processed_data_folder + '/' + additional_antenna_delays)
station_timing_offsets = read_station_delays(processed_data_folder + '/' + station_delay_file)

raw_fpaths = filePaths_by_stationName(timeID)

TBB_data = {sname : MultiFile_Dal1(fpath, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas,
additional_ant_delays=additional_antenna_delays, only_complete_pairs=True) for sname, fpath in raw_fpaths.items() if sname in station_timing_offsets}

#function that constructs the unit vector a from its fitted components in spherical coordinates
def construct_a(*params):
	x = np.cos(params[0])*np.sin(params[1])
	y = np.sin(params[0])*np.sin(params[1])
	z = np.cos(params[1])
	return np.array([x,y,z])

def forceEqualAspect(frame, shared_axis=None):
	xlim = frame.get_xlim()
	ylim = frame.get_ylim()
	x_w = abs(xlim[1]-xlim[0])
	y_w = abs(ylim[1]-ylim[0])
	f = y_w/x_w
	Δ = abs((x_w-y_w)/2)

	if shared_axis is None:
		if f < 1:
			frame.set_ylim((ylim[0]-Δ, ylim[1]+Δ))
		elif f > 1:
			frame.set_xlim((xlim[0]-Δ, xlim[1]+Δ))

	if shared_axis == 'x':
		frame.set_ylim((ylim[0]-Δ, ylim[1]+Δ))

	if shared_axis == 'y':
		frame.set_xlim((xlim[0]-Δ, xlim[1]+Δ))


if __name__ == "__main__":
	mode = 'stations' #set mode of plot (either 'stations' or 'a')
	#mode = 'a'

	station_names = natural_sort(station_timing_offsets.keys())
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC9" #phenomena name

	with open(data_folder + "/source_info_" + pName + ".json", 'r') as f:
		source_info = json.load(f)


	#sort station names according to zenith angle (Z) and impose a Z requirement
	Z = np.array([])
	for station in station_names:
		antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
		avg_station_loc = np.average(antenna_locations, axis=0)

		#average the source location
		avg_source_XYZ = np.zeros(3)
		N = 0
		for ID in source_info.keys():
			avg_source_XYZ += source_info[ID]['XYZT'][:3]
			N += 1
		avg_source_XYZ /= N

		Z = np.append(Z, zenith_to_src(avg_source_XYZ, avg_station_loc))

	sort_indices = np.argsort(Z)
	station_names = list(station_names)
	station_names = [station_names[i] for i in sort_indices]
	Z = Z[sort_indices]

	"""
	for i in range(sort_indices.size):
		print("{} : {} deg".format(station_names[i],Z[i]))
	"""

	Zlimit = 50
	station_names = [station_names[i] for i in np.where(Z<=Zlimit)[0]]

	a_data = read_acceleration_vector(timeID, fname="a_vectors_{}".format(pName), alt_loc=data_folder + '/' + "{}_data".format(pName))

	p_loc = np.empty((0,4))
	a_vec = np.empty((0,3))
	for pulseID in a_data.keys(): #change this for plotting separate pulses
		source_XYZ = source_info[pulseID]['XYZT']
		p_loc = np.append(p_loc, np.array([source_XYZ]), axis=0)
		a_vec = np.append(a_vec, np.array([construct_a(*a_data[pulseID]["params"])]), axis=0)
	
	s_loc = np.empty((0,3))
	for station in station_names:
		antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
		avg_station_XYZ = np.average(antenna_locations, axis=0)
		s_loc = np.append(s_loc, np.array([avg_station_XYZ]), axis=0)

	p_loc[:, :3] /= 1000 #convert to km
	s_loc /= 1000 #convert to km

	if mode == 'stations':
		##########################################
		##	PLOT WITH THE DIFFERENT STATIONS	##
		##########################################
		cmap = gen_olaf_cmap()
		norm = colors.Normalize(p_loc[:, 3].min(), p_loc[:, 3].max())

		fig = figure(figsize=(20,20))
		gs = fig.add_gridspec(2, 2, wspace=0, hspace=0) #width_ratios=(1, 1), height_ratios=(1, 1),
		
		p_scatter_setup = dict(s=10, marker='s', c=p_loc[:, 3], cmap=cmap, norm=norm, alpha=0.75, zorder=2) #alpha = 0.75, edgecolor='k', linewidths=0.01
		arrow_setup = dict(color='k', units='dots', angles='uv', width=0.5, headlength=0, headaxislength=0, pivot='mid', zorder=1)

		
		frame_XY = fig.add_subplot(gs[1, 0])
		frame_XY.set_xlabel(r"Distance West-East [km]", fontsize=16)
		frame_XY.set_ylabel(r"Distance South-North [km]", fontsize=16)
		
		frame_XY.scatter(s_loc[:, 0], s_loc[:, 1], s=4, marker='*', color='k')
		for i, s in enumerate(station_names):
			if Z[i]<=Zlimit:
				frame_XY.annotate(s, (s_loc[:, 0][i], s_loc[:, 1][i]), xytext=(6, 6), textcoords='offset pixels', fontsize=8, bbox=dict(boxstyle="round", fc='w', ec="0.5", alpha=0.9))
			else:
				frame_XY.annotate(s, (s_loc[:, 0][i], s_loc[:, 1][i]), xytext=(6, 6), textcoords='offset pixels', fontsize=8, color='w', bbox=dict(boxstyle="round", fc='k', ec="0.5", alpha=0.9))

		frame_XY.scatter(p_loc[:, 0], p_loc[:, 1], **p_scatter_setup)
		frame_XY.quiver(*p_loc[:, :2].T, *a_vec[:, :2].T, **arrow_setup)
		
		xlimits = frame_XY.get_xlim()
		ylimits = frame_XY.get_ylim()
		frame_XY.set_xlim((xlimits[0], xlimits[1]*1.1))
		frame_XY.set_ylim((ylimits[0], ylimits[1]*1.1))
		
		forceEqualAspect(frame_XY)

		frame_XY.grid()


		frame_XZ = fig.add_subplot(gs[0, 0], sharex=frame_XY)
		frame_XZ.set_ylabel(r"Altitude [km]", fontsize=16)
		setp(frame_XZ.get_xticklabels(), visible=False)

		frame_XZ.scatter(p_loc[:, 0], p_loc[:, 2], **p_scatter_setup)
		frame_XZ.quiver(*p_loc[:, ::2].T, *a_vec[:, ::2].T, **arrow_setup)

		forceEqualAspect(frame_XZ, shared_axis='x')

		frame_XZ.grid()

		
		frame_ZY = fig.add_subplot(gs[1, 1], sharey=frame_XY)
		frame_ZY.set_xlabel(r"Altitude [km]", fontsize=16)
		setp(frame_ZY.get_yticklabels(), visible=False)

		frame_ZY.scatter(p_loc[:, 2], p_loc[:, 1], **p_scatter_setup)
		frame_ZY.quiver(*p_loc[:, -2:-4:-1].T, *a_vec[:, -1:-3:-1].T, **arrow_setup)
		
		forceEqualAspect(frame_ZY, shared_axis='y')

		frame_ZY.grid()


		divider = make_axes_locatable(frame_ZY)
		cax = divider.append_axes("right", size="2%", pad=0.03)
		cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(p_loc[:, 3].min()*1E3, p_loc[:, 3].max()*1E3), cmap=cmap), cax=cax)
		cbar.set_label(label=r"$t\ [ms]$",fontsize=16)

		#extent = frame_XY.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "stations_versus_{}_plot.pdf".format(pName), dpi=fig.dpi, bbox_inches='tight') #extent
		#show()

	if mode == 'a':
		##########################################
		##	PLOT OF THE ACCELERATION VECTORS	##
		##########################################
		cmap = gen_olaf_cmap()
		norm = colors.Normalize(p_loc[:, 3].min(), p_loc[:, 3].max())

		fig = figure(figsize=(20,20))
		gs = fig.add_gridspec(2, 2, wspace=0, hspace=0)
				
		arrow_setup = dict(color='k', units='dots', angles='uv', width=0.5, headlength=0, headaxislength=0, pivot='mid', zorder=1)
		scatter_setup = dict(s=7.5, marker='s', c=p_loc[:, 3], cmap=cmap, norm=norm, alpha=0.75, zorder=2) #edgecolor='k', linewidths=0.3


		frame_XY = fig.add_subplot(gs[1, 0])
		frame_XY.set_xlabel(r"Distance West-East [km]", fontsize=16)
		frame_XY.set_ylabel(r"Distance South-North [km]", fontsize=16)

		frame_XY.scatter(p_loc[:, 0], p_loc[:, 1], **scatter_setup)

		frame_XY.quiver(*p_loc[:, :2].T, *a_vec[:, :2].T, **arrow_setup)

		forceEqualAspect(frame_XY)

		frame_XY.grid()
		
		
		frame_XZ = fig.add_subplot(gs[0, 0], sharex=frame_XY)
		frame_XZ.set_ylabel(r"Altitude [km]", fontsize=16)
		setp(frame_XZ.get_xticklabels(), visible=False)

		frame_XZ.scatter(p_loc[:, 0], p_loc[:, 2], **scatter_setup)

		frame_XZ.quiver(*p_loc[:, ::2].T, *a_vec[:, ::2].T, **arrow_setup)
		
		forceEqualAspect(frame_XZ, shared_axis='x')

		frame_XZ.grid()

		
		frame_ZY = fig.add_subplot(gs[1, 1], sharey=frame_XY)
		frame_ZY.set_xlabel(r"Altitude [km]", fontsize=16)
		setp(frame_ZY.get_yticklabels(), visible=False)

		frame_ZY.scatter(p_loc[:, 2], p_loc[:, 1], **scatter_setup)

		frame_ZY.quiver(*p_loc[:, -2:-4:-1].T, *a_vec[:, -1:-3:-1].T, **arrow_setup)

		forceEqualAspect(frame_ZY, shared_axis='y')

		frame_ZY.grid()

		divider = make_axes_locatable(frame_ZY)
		cax = divider.append_axes("right", size="2%", pad=0.03)
		cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(p_loc[:, 3].min()*1E3, p_loc[:, 3].max()*1E3), cmap=cmap), cax=cax)
		cbar.set_label(label=r"$t\ [ms]$",fontsize=16)

		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "a_vectors_plot_{}.pdf".format(pName), dpi=fig.dpi, bbox_inches='tight')
		#show()
