#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import pickle
import json
from matplotlib.pyplot import figure, close

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data
from polplot2D import polPlot2D

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


if __name__ == "__main__":
	station_names = natural_sort(station_timing_offsets.keys())
	#station_names = ["CS006"] #for KC9
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/Positive Leader"
	pName = "PL2" #phenomena name

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
		source_info = json.load(f)


	#sort station names according to Z and impose a Z requirement
	Z = np.array([])
	for station in station_names:
		antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
		avg_station_loc = np.average(antenna_locations, axis=0)

		#average the source location
		N = 0
		avg_source_XYZ = np.zeros(3)
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

	for station in station_names:
		print(station)

		pulses_PE = {}

		for filename in os.listdir(data_folder + '/' + pName + "_data/Polarization_Ellipse"):
			pulseID = os.path.splitext(filename)[0]

			if int(pulseID) in pulseIDs:
				df = rd.read_PE(pulseID)

				#if the dataframe is empty go back to top
				if df.empty:
					continue

				#if there is no vector for the station go back to top
				if station not in df.index:
					continue
				
				pulses_PE[pulseID] = df.loc[station].values

		#if the dictionary is still empty continue to the next iteration
		if not pulses_PE:
			continue

		#get the average location of the station so we can find the location of the pulse relative to the station
		antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
		avg_station_loc = np.average(antenna_locations, axis=0)

		#uncomment to  get info on circularly polarized pulses only
		"""
		ε = []
		circID = []
		atol = np.deg2rad(30)
		for pulseID in pulses_PE.keys():
			if abs(pulses_PE[pulseID][3]) > atol:
				circID.append(pulseID)
				ε.append(abs(pulses_PE[pulseID][3]))

		print(circID, max(ε))
		circ_pulses_PE = {}
		circ_source_info = {}
		for ID in circID:
			circ_source_info["{}".format(ID)] = source_info[ID]
			circ_pulses_PE["{}".format(ID)] = pulses_PE[ID]

		polPlot2D(avg_station_loc, circ_pulses_PE, circ_source_info, ell_scale=2**50, ϕ_shift=False, errors=False)
		"""

		scale = 2**51
		ϕ_shift = False

		fig = polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=scale, ϕ_shift=ϕ_shift, errors=False, save=True)
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "{}_result.png".format(station), dpi=fig.dpi)
		close(fig=fig)

		fig = polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=scale, ϕ_shift=ϕ_shift, errors=True, save=True)
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "{}_result_withErrorbars.png".format(station), dpi=fig.dpi)
		close(fig=fig)

		"""
		fig = figure(figsize=(25,10),dpi=108) #Az : [-90,90] and Z : [0:90] => 2:1 ratio ensures we fill canvas properly
		
		frame1 = fig.add_subplot(141, aspect='equal') #equal aspect ensures correct direction of polarization ellipse!
		frame1.set_title(r"$S_0$",fontsize=16)
		polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=2**50, ϕ_shift=False, errors=False, save=True, fig=fig, frame=frame1)

		avg = 0
		N = 0
		for ID in pulses_PE.keys():
			avg+=pulses_PE[ID][0]
			N+=1
			pulses_PE[ID][0] *= pulses_PE[ID][1]
		avg /= N
		scale = avg*2**50
		frame2 = fig.add_subplot(142, aspect='equal') #equal aspect ensures correct direction of polarization ellipse!
		frame2.set_title(r"$S_0\cdot\delta$",fontsize=16)
		polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=2**50, ϕ_shift=False, errors=False, save=True, fig=fig, frame=frame2)

		for ID in pulses_PE.keys():
			pulses_PE[ID][0] = pulses_PE[ID][1]
		frame3 = fig.add_subplot(143, aspect='equal') #equal aspect ensures correct direction of polarization ellipse!
		frame3.set_title(r"$\delta$",fontsize=16)
		polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=scale, ϕ_shift=False, errors=False, save=True, fig=fig, frame=frame3)

		for ID in pulses_PE.keys():
			pulses_PE[ID][0] = 1
		frame4 = fig.add_subplot(144, aspect='equal') #equal aspect ensures correct direction of polarization ellipse!
		frame4.set_title(r"$1$",fontsize=16)
		polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=scale, ϕ_shift=False, errors=False, save=True, fig=fig, frame=frame4)

		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "{}_result_comp.png".format(station), dpi=fig.dpi)
		close(fig=fig)
		"""