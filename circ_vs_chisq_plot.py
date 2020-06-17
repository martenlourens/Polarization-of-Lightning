#!/usr/bin/env python3

import numpy as np
import json
from matplotlib.pyplot import figure, show, close
from matplotlib import colors, cm
from matplotlib.ticker import LogLocator, LogFormatter, Locator

"""
	Class to fix the tickmarks for symlog. Source: https://stackoverflow.com/questions/20470892/how-to-place-minor-ticks-on-symlog-scale.
"""
class MinorSymLogLocator(Locator):
	"""
	Dynamically find minor tick positions based on the positions of
	major ticks for a symlog scaling.
	"""
	def __init__(self, linthresh, nints=10):
		"""
		Ticks will be placed between the major ticks.
		The placement is linear for x between -linthresh and linthresh,
		otherwise its logarithmically. nints gives the number of
		intervals that will be bounded by the minor ticks.
		"""
		self.linthresh = linthresh
		self.nintervals = nints

	def __call__(self):
		# Return the locations of the ticks
		majorlocs = self.axis.get_majorticklocs()

		if len(majorlocs) == 1:
			return self.raise_if_exceeds(np.array([]))

		# add temporary major tick locs at either end of the current range
		# to fill in minor tick gaps
		dmlower = majorlocs[1] - majorlocs[0]    # major tick difference at lower end
		dmupper = majorlocs[-1] - majorlocs[-2]  # major tick difference at upper end

		# add temporary major tick location at the lower end
		if majorlocs[0] != 0. and ((majorlocs[0] != self.linthresh and dmlower > self.linthresh) or (dmlower == self.linthresh and majorlocs[0] < 0)):
			majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
		else:
			majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

		# add temporary major tick location at the upper end
		if majorlocs[-1] != 0. and ((np.abs(majorlocs[-1]) != self.linthresh and dmupper > self.linthresh) or (dmupper == self.linthresh and majorlocs[-1] > 0)):
			majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
		else:
			majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

		# iterate through minor locs
		minorlocs = []

		# handle the lowest part
		for i in range(1, len(majorlocs)):
			majorstep = majorlocs[i] - majorlocs[i-1]
			if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
				ndivs = self.nintervals
			else:
				ndivs = self.nintervals - 1.

			minorstep = majorstep / ndivs
			locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
			minorlocs.extend(locs)

		return self.raise_if_exceeds(np.array(minorlocs))

import seaborn as sns
from tqdm import tqdm

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data, read_acceleration_vector
from acceleration_direction import residuals


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
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC9" #phenomena name

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
		source_info = json.load(f)

	rd = read_polarization_data(timeID, alt_loc=data_folder + '/' + "{}_data".format(pName))
	a_data = read_acceleration_vector(timeID, fname="a_vectors_{}".format(pName), alt_loc=data_folder + '/' + "{}_data".format(pName))

	ε = np.array([]) #open angle of the polarization ellipse
	ε_err = np.array([]) #error in open angle of the polarization ellipse
	χ2 = np.array([]) #χ2 values for each station at the model τ

	pbar = tqdm(a_data.keys(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
	for pulseID in pbar:
		pbar.set_description("Processing pulse {}".format(pulseID))

		#sort station names according to zenith angle (Z) and impose a Z requirement
		station_names = natural_sort(station_timing_offsets.keys())
		Z = np.array([])
		for station in station_names:
			#station location is determined from the average antenna locations
			antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
			avg_station_loc = np.average(antenna_locations, axis=0)

			#source location
			source_XYZ = source_info[pulseID]['XYZT'][:3]

			Z = np.append(Z, zenith_to_src(source_XYZ, avg_station_loc))

		sort_indices = np.argsort(Z)
		Z = Z[sort_indices]
		station_names = np.array(station_names)[sort_indices]

		Zlimit = 50
		constrain_indices = np.where(Z<=Zlimit)[0]
		station_names = station_names[constrain_indices]
		Z = Z[constrain_indices]

		df = rd.read_PE(pulseID)

		#del_indices = []
		for i, station in np.ndenumerate(station_names):
			if station in df.index:
				if df.loc[station]["δ"] < 0.8:
					continue
				ε = np.append(ε, np.rad2deg(df.loc[station]["ε"]))
				ε_err = np.append(ε_err, np.rad2deg(df.loc[station]["σ_ε"]))

				τ = np.array([df.loc[station]["τ"]])
				τ_err = np.array([df.loc[station]["σ_τ"]])

				#compute r (i.e. radial vector pointing from station to source)
				antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
				avg_station_XYZ = np.average(antenna_locations, axis=0)
				source_XYZ = source_info[pulseID]['XYZT'][:3]
				r = np.array([source_XYZ - avg_station_XYZ])

				data = (r, τ, τ_err)

				#compute chisq for each station
				χ2 = np.append(χ2, residuals(a_data[pulseID]['params'], data)**2)
		
	fig = figure(figsize=(10, 10))
	frame = fig.add_subplot(111)
	frame.scatter(ε, χ2, marker='.', color='k', s=4) #NOTE HIGH |ε| DOES NOT IMPLY HIGH χ^2!!!
	frame.set_xlabel(r"$\epsilon\ [^{\circ}]$", fontsize=16)
	frame.set_ylabel(r"$\chi^2$", fontsize=16)
	frame.set_xlim((-45, 45))
	frame.set_ylim((0, None))
	frame.set_yscale("symlog")
	#frame.yaxis.set_minor_formatter(LogFormatter(linthresh=1e-1))
	frame.yaxis.set_minor_locator(MinorSymLogLocator(linthresh=1e-1))
	frame.grid(True, which='both')
	fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "circular_polarization_versus_chisq_{}.pdf".format(pName), dpi=fig.dpi, bbox_inches='tight')
	#show()
	close(fig)

	pbar.close()