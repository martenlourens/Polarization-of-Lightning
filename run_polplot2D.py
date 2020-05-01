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
	station = "CS302"

	with open(processed_data_folder+"/polarization_data/pulseIDs.pkl", 'rb') as f:
		pulseIDs = pickle.load(f)

	rd = read_polarization_data(timeID)

	pulses_PE = {}

	for filename in os.listdir(processed_data_folder+"/polarization_data/Polarization_Ellipse"):
		pulseID = os.path.splitext(filename)[0]
		df = rd.read_PE(pulseID)

		#if the dataframe is empty go back to top
		if df.empty:
			continue

		#if dataframe wasn't empty and pulseID is in the pulseIDs list then look for data of the particular station and save it in the dictionary
		if int(pulseID) in pulseIDs:
			pulses_PE[pulseID] = df.loc[station].values

	with open(processed_data_folder+"/polarization_data/source_info.json", 'r') as f:
		source_info = json.load(f)

	#get the average location of the station so we can find the location of the pulse relative to the station
	antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
	avg_station_loc = np.average(antenna_locations, axis=0)

	polPlot2D(avg_station_loc, pulses_PE, source_info, ϕ_shift=True)