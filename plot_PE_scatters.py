#!/usr/bin/env python3

import numpy as np
import json
import pickle
from matplotlib.pyplot import figure, show, close
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data

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

def plot_PE_scatter(pulses_PE, save=False):
	cmap = gen_olaf_cmap()

	errorbar_setup = dict(fmt='.k', markersize=0.5, marker='s', capsize=2, capthick=0.25, elinewidth=0.5, ecolor='k', zorder=1)
	scatter_setup = dict(s=10, marker='s', cmap=cmap, edgecolor='k', linewidths=0.4, zorder=2)

	fig = figure(figsize=(20,20))
	gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.15)

	fig.suptitle(r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z[i]), fontsize=32, position=(0.5, 0.95))
			
	frame1 = fig.add_subplot(gs[0])
	norm1 = colors.Normalize(min(pulses_PE['ε']), max(pulses_PE['ε']))
	frame1.scatter(pulses_PE['δ'], pulses_PE['τ'], c=pulses_PE['ε'], norm=norm1, **scatter_setup)
	frame1.errorbar(pulses_PE['δ'], pulses_PE['τ'], xerr=pulses_PE["σ_δ"], yerr=pulses_PE["σ_τ"], **errorbar_setup)
	frame1.set_xlabel(r"$δ$", fontsize=16)
	frame1.set_ylabel(r"$τ\ [^{\circ}]$", fontsize=16)

	divider1 = make_axes_locatable(frame1)
	cax1 = divider1.append_axes("right", size="2%", pad=0.03)
	cbar1 = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm1), cax=cax1)
	cbar1.set_label(label=r"$ε\ [^{\circ}]$",fontsize=16)

	frame1.set_xlim((0, 1))
	frame1.set_ylim(-90, 90)
	frame1.grid()

	frame2 = fig.add_subplot(gs[1])
	norm2 = colors.Normalize(min(pulses_PE['τ']), max(pulses_PE['τ']))
	frame2.scatter(pulses_PE['δ'], pulses_PE['ε'], c=pulses_PE['τ'], norm=norm2, **scatter_setup)
	frame2.errorbar(pulses_PE['δ'], pulses_PE['ε'], xerr=pulses_PE["σ_δ"], yerr=pulses_PE["σ_ε"], **errorbar_setup)
	frame2.set_xlabel(r"$δ$", fontsize=16)
	frame2.set_ylabel(r"$ε\ [^{\circ}]$", fontsize=16)

	divider2 = make_axes_locatable(frame2)
	cax2 = divider2.append_axes("right", size="2%", pad=0.03)
	cbar2 = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm2), cax=cax2)
	cbar2.set_label(label=r"$τ\ [^{\circ}]$",fontsize=16)

	frame2.set_xlim((0, 1))
	frame2.set_ylim(-45, 45)
	frame2.grid()

	frame3 = fig.add_subplot(gs[2])
	norm3 = colors.Normalize(min(pulses_PE['δ']), max(pulses_PE['δ']))
	frame3.scatter(pulses_PE['τ'], pulses_PE['ε'], c=pulses_PE['δ'], norm=norm3, **scatter_setup)
	frame3.errorbar(pulses_PE['τ'], pulses_PE['ε'], xerr=pulses_PE["σ_τ"], yerr=pulses_PE["σ_ε"], **errorbar_setup)
	frame3.set_xlabel(r"$τ\ [^{\circ}]$", fontsize=16)
	frame3.set_ylabel(r"$ε\ [^{\circ}]$", fontsize=16)

	divider3 = make_axes_locatable(frame3)
	cax3 = divider3.append_axes("right", size="2%", pad=0.03)
	cbar3 = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm3), cax=cax3)
	cbar3.set_label(label=r"$δ$",fontsize=16)

	frame3.set_xlim(-90, 90)
	frame3.set_ylim(-45, 45)
	frame3.grid()

	if save:
		return fig
	else:
		show()

if __name__ == "__main__":
	station_names = natural_sort(station_timing_offsets.keys())
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC9" #phenomena name

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
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

	with open(data_folder + '/' + "pulseIDs_{}.pkl".format(pName), 'rb') as f:
		pulseIDs = pickle.load(f)

	rd = read_polarization_data(timeID, alt_loc=data_folder + '/' + "{}_data".format(pName))

	for i, station in enumerate(station_names):
		print(station)

		pulses_PE = {'δ' : [], "σ_δ" : [], 'τ' : [], "σ_τ" : [], 'ε' : [], "σ_ε" : []}
		
		for pulseID in pulseIDs:

			df = rd.read_PE(pulseID)

			if df.empty:
				continue

			if station in df.index:
				pulses_PE['δ'].append(df.loc[station]['δ'])
				pulses_PE["σ_δ"].append(df.loc[station]["σ_δ"])
				pulses_PE['τ'].append(np.rad2deg(df.loc[station]['τ']))
				pulses_PE["σ_τ"].append(np.rad2deg(df.loc[station]["σ_τ"]))
				pulses_PE['ε'].append(np.rad2deg(df.loc[station]['ε']))
				pulses_PE["σ_ε"].append(np.rad2deg(df.loc[station]["σ_ε"]))

		if pulses_PE == {'δ' : [], "σ_δ" : [], 'τ' : [], "σ_τ" : [], 'ε' : [], "σ_ε" : []}:
			continue

		fig = plot_PE_scatter(pulses_PE, save=True)
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "PE_scatters" + '/' + "{}_PE_scatter.png".format(station), dpi=fig.dpi)
		close(fig)