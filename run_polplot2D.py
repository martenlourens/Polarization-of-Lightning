#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import pickle
import json

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1

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
	station = "RS205"

	with open(processed_data_folder+"/polarization_data/pulseIDs_2.pkl", 'rb') as f:
		pulseIDs = pickle.load(f)

	rd = read_polarization_data(timeID)

	pulses_PE = {}

	for filename in os.listdir(processed_data_folder+"/polarization_data/Polarization_Ellipse"):
		pulseID = os.path.splitext(filename)[0]
		
		if int(pulseID) in pulseIDs:
			df = rd.read_PE(pulseID)

			#if the dataframe is empty go back to top
			if df.empty:
				continue

			#if dataframe wasn't empty then look for data of the particular station and save it in the dictionary
			pulses_PE[pulseID] = df.loc[station].values

	with open(processed_data_folder+"/polarization_data/source_info_2.json", 'r') as f:
		source_info = json.load(f)

	#get the average location of the station so we can find the location of the pulse relative to the station
	antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
	avg_station_loc = np.average(antenna_locations, axis=0)

	#uncomment to  get info on circularly polarized pulses only
	"""
	ε = []
	circID = []
	atol = np.deg2rad(24)
	for pulseID in pulses_PE.keys():
		if abs(pulses_PE[pulseID][3]) > atol:
			circID.append(pulseID)
			ε.append(abs(pulses_PE[pulseID][3]))

	#print(circID, max_ε)
	circ_pulses_PE = {}
	circ_source_info = {}
	for ID in circID:
		circ_source_info["{}".format(ID)] = source_info[ID]
		circ_pulses_PE["{}".format(ID)] = pulses_PE[ID]
	print(circ_source_info,"\n",circ_pulses_PE)
	
	polPlot2D(avg_station_loc, circ_pulses_PE, circ_source_info, ell_scale=5E13, ϕ_shift=True, errors=True)
	"""

	polPlot2D(avg_station_loc, pulses_PE, source_info, ell_scale=2**52, ϕ_shift=True, errors=True) #5E13