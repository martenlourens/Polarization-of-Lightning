#!/usr/bin/env python3

import numpy as np
from numpy.linalg import norm

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1

#function for computing horizontal and vertical distance to an average source location from an average station location
def dist_to_src(avg_source_XYZ, avg_station_XYZ):
	XYZ = avg_source_XYZ - avg_station_XYZ
	vert = XYZ[2]
	horiz = norm(XYZ[:2])
	return horiz, vert

#function for computing the zenith angle to an average source location from an average station location
def zenith_to_src(avg_source_XYZ, avg_station_XYZ, deg=True):
	horiz, vert = dist_to_src(avg_source_XYZ, avg_station_XYZ)
	Z = np.arctan(horiz/vert)
	if deg:
		return np.rad2deg(Z)
	return Z

#for testing purposes only
if __name__ == "__main__":
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

	sNames = station_timing_offsets.keys()

	#test location
	source_info = {'1477200': {'XYZT' : [-743.41, -3174.29, 4377.67, 1.1350323478]}}

	Z = np.array([])
	for station in sNames:
		antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
		avg_station_loc = np.average(antenna_locations, axis=0)

		N = 0
		loc_sum = np.zeros(3)
		for ID in source_info:
			loc_sum += source_info[ID]['XYZT'][:3]
			N += 1
		avg_source_XYZ = loc_sum/N
		Z = np.append(Z, zenith_to_src(avg_source_XYZ, avg_station_loc)) 

	sort_indices = np.argsort(Z)

	#sort Z an station names
	#Z = Z[sort_indices]

	sNames = list(sNames)
	for i in sort_indices:
		print("{} : {} deg".format(sNames[i], Z[i]))