#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import pickle
import json
from kapteyn import kmpfit
from matplotlib.pyplot import figure, show, setp
from matplotlib import cm, colors
#from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
#import corner
from dynesty import NestedSampler, plotting

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

def model(p, r):
	ϕ = np.arctan2(r[:,1], r[:,0])
	ϕ_hat = np.array([-np.sin(ϕ), np.cos(ϕ), np.zeros(ϕ.size)]).T #unit azimuth vector
	
	r_hat = r/np.linalg.norm(r, axis=1)[:, np.newaxis] #radial unit vector to source
	
	a = np.array([np.sin(p[1])*np.cos(p[0]), np.sin(p[1])*np.sin(p[0]), np.cos(p[1])])
	a_perp = a - np.einsum('j,ij,ik->ik', a, r_hat, r_hat)
	a_perp_hat = a_perp/np.linalg.norm(a_perp, axis=1)[:, np.newaxis]
	
	τ = np.arccos(np.einsum('ij,ij->i', a_perp_hat, ϕ_hat))
	return τ

def residuals(p, data):
	r, τ, τ_err = data
	return (τ - model(p, r))/τ_err


if __name__ == "__main__":
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

	#

	Zlimit = 50
	station_names = [station_names[i] for i in np.where(Z<=Zlimit)[0]]


	with open(data_folder + "/pulseIDs_" + pName + ".pkl", 'rb') as f:
		pulseIDs = pickle.load(f)

	rd = read_polarization_data(timeID, alt_loc="polarization_data/Lightning Phenomena/K changes/" + pName + "_data")

	for pulseID in pulseIDs:
		if pulseID == 1481534:
			τ = np.array([])
			τ_err = np.array([])
			r = np.empty((0,3))

			df = rd.read_PE(pulseID)

			if df.empty:
				continue

			#max_stations = 3
			#station_select = 21
			#i = 0
			for station in station_names:
				if station in df.index:
					#i += 1
					#if i == station_select:
					τ = np.append(τ, df.loc[station]["τ"])
					τ_err = np.append(τ_err, df.loc[station]["σ_τ"])

						#compute r (i.e. radial vector pointing from station to source)
					antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
					avg_station_XYZ = np.average(antenna_locations, axis=0)
					source_XYZ = source_info[str(pulseID)]['XYZT'][:3]
					r = np.append(r, np.array([source_XYZ - avg_station_XYZ]), axis=0)
					print(station)
						
						#break

					#if i == max_stations:
					#	break

					data = (r, τ, τ_err)

					######################################
					##	PLOT OF CHISQUARE DISTRIBUTION	##
					######################################
					N = 200
					ϕ = np.linspace(-np.pi, np.pi, 2*N)
					θ = np.linspace(0, np.pi, N)
					χ2 = np.zeros((N, 2*N))

					for i in range(2*N):
					    for j in range(N):
					        χ2[j,i] = np.sum(residuals(np.array([ϕ[i], θ[j]]), data)**2)
					
					ϕθ = np.meshgrid(ϕ,θ)

					level = np.logspace(np.log10(χ2.min()),np.log10(χ2.max()), 30)

					cmap = gen_olaf_cmap()
					norm = colors.LogNorm(χ2.min(), χ2.max())

					#plot without marginalization
					fig = figure(figsize=(20,10))
					frame = fig.add_subplot(111, aspect='equal')
					cont = frame.contour(ϕθ[0], ϕθ[1], χ2, level, cmap=cmap, norm=norm)
					frame.clabel(cont, level, fontsize=8, colors='k')
					frame.set_xlabel(r"$ϕ$", fontsize=16)
					frame.set_ylabel(r"$θ$", fontsize=16)
					frame.set_ylim((np.pi, 0))
					show()

					τ = np.array([])
					τ_err = np.array([])
					r = np.empty((0,3)) 

			break