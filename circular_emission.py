#!/usr/bin/env python3

import numpy as np
import json
from tqdm import tqdm
from scipy.interpolate import interp2d, griddata
from matplotlib.pyplot import figure, show, close
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data, read_acceleration_vector
from acceleration_direction_plots import construct_a

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

#TEST CODE
"""
a = np.array([1, 0, 0])
ϕ_a = np.arctan2(a[1], a[0])
θ_a = np.arctan2(np.linalg.norm(a[:2]), a[2])

print("a = {}\nϕ_a = {} deg\nθ_a = {} deg".format(a, np.rad2deg(ϕ_a), np.rad2deg(θ_a)))


r = np.array([1, 0, 0])
ϕ_r = np.arctan2(r[1], r[0])
θ_r = np.arctan2(np.linalg.norm(r[:2]), r[2])

print("r = {}\nϕ_r = {} deg\nθ_r = {} deg".format(r, np.rad2deg(ϕ_r), np.rad2deg(θ_r)))


Δϕ = ϕ_a - ϕ_r
if Δϕ > np.pi:
	Δϕ -= 2*np.pi
elif Δϕ < -np.pi:
	Δϕ += 2*np.pi
Δθ = abs(θ_a - θ_r)

print("Δϕ = {} deg\nΔθ = {} deg".format(np.rad2deg(Δϕ), np.rad2deg(Δθ)))
"""

if __name__ == "__main__":
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC9" #phenomena name

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
		source_info = json.load(f)

	rd = read_polarization_data(timeID, alt_loc=data_folder + '/' + "{}_data".format(pName))
	a_data = read_acceleration_vector(timeID, fname="a_vectors_{}".format(pName), alt_loc=data_folder + '/' + "{}_data".format(pName))

	station_names = natural_sort(station_timing_offsets.keys())
	Zlimit = 50

	"""	
	fig = figure(figsize=(10, 10))
	frame = fig.add_subplot(111)

	x = np.linspace(0, np.pi, 100)
	frame.plot(np.rad2deg(x), 45*np.cos(x), color='k')
	frame.plot(np.rad2deg(x), -45*np.cos(x), color='r')

	frame.set_xlabel(r"$\Delta\theta\ [^{\circ}]$", fontsize=16)
	frame.set_ylabel(r"$\epsilon\ [^{\circ}]$", fontsize=16)

	frame.set_xlim((0, 180))
	frame.set_ylim((-45, 45))

	frame.grid()
	"""

	pbar = tqdm(a_data.keys(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
	for pulseID in pbar:
		pbar.set_description("Processing pulse {}".format(pulseID))

		
		#uncomment for plot per pulse
		#Δϕ = np.array([]) DEPRECATED!!!
		Δθ = np.array([])
		ε = np.array([])
		ε_err = np.array([])
		
		Z = np.array([])

		df = rd.read_PE(pulseID)

		for station in station_names:
			if station in df.index:
				if df.loc[station]['δ'] <= 0.8:
					continue

				#compute r (i.e. radial vector pointing from station to source)
				antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
				avg_station_XYZ = np.average(antenna_locations, axis=0)
				source_XYZ = source_info[pulseID]['XYZT'][:3]

				#if zenith_to_src(source_XYZ, avg_station_XYZ) > Zlimit:
				#	continue
				Z = np.append(Z, zenith_to_src(source_XYZ, avg_station_XYZ))

				r = source_XYZ - avg_station_XYZ

				#DEPRECATED!!!
				"""
				ϕ_r = np.arctan2(r[1], r[0])
				θ_r = np.arctan2(np.linalg.norm(r[:2]), r[2])

				ϕ_a, θ_a = a_data[pulseID]['params']

				diff_ϕ = ϕ_a - ϕ_r
				if diff_ϕ > np.pi:
					diff_ϕ -= 2*np.pi
				elif diff_ϕ < -np.pi:
					diff_ϕ += 2*np.pi
				Δϕ = np.append(Δϕ, np.rad2deg(diff_ϕ))
				
				diff_θ = np.abs(θ_a - θ_r)
				Δθ = np.append(Δθ, np.rad2deg(diff_θ))
				"""

				ϕ_a, θ_a = a_data[pulseID]['params']

				"""
				if ϕ_a < -np.pi/2:
					ϕ_a += np.pi
					θ_a = np.pi - θ_a
				elif ϕ_a > np.pi/2:
					ϕ_a -= np.pi
					θ_a = np.pi - θ_a
				"""
				
				a = construct_a(ϕ_a, θ_a) #a is a unit vector along the large scale acceleration direction of the charge


				Δθ = np.append(Δθ, np.rad2deg(np.arccos(np.dot(r/np.linalg.norm(r), a))))
				ε = np.append(ε, np.rad2deg(df.loc[station]["ε"]))
				ε_err = np.append(ε_err, np.rad2deg(df.loc[station]["σ_ε"]))


		#uncomment for plot per pulse
		fig = figure(figsize=(10, 10))
		frame = fig.add_subplot(111)

		#x = np.linspace(0, np.pi, 100)
		#frame.plot(np.rad2deg(x), 45*np.cos(x), color='k')
		#frame.plot(np.rad2deg(x), -45*np.cos(x), color='r')

		frame.set_xlabel(r"$\Delta\theta\ [^{\circ}]$", fontsize=16)
		frame.set_ylabel(r"$\epsilon\ [^{\circ}]$", fontsize=16)

		frame.set_xlim((0, 180))
		frame.set_ylim((-45, 45))

		frame.grid()

		color = 'k'
		for i, θ_z in np.ndenumerate(Z):
			if θ_z <= Zlimit:
				color = 'k'
			else:
				color = 'r'

			frame.scatter(Δθ[i], ε[i], marker='s', s=10, zorder=2, color=color)
			frame.errorbar(Δθ[i], ε[i], yerr=ε_err[i], fmt=".{}".format(color), markersize=1, elinewidth=1, capsize=2, capthick=1, ecolor="{}".format(color), zorder=1)

		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "circular_emission_patterns" + '/' + "{}_circems.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		close(fig)
		#show()

		#DEPRECATED!!!
		"""
		#stack the data coordinates as follows (n, D) where n is the number of datapoints and D is the dimensionality of the data
		Pi = np.vstack((Δϕ, Δθ)).T

		cmap = gen_olaf_cmap()
		fig = figure(figsize=(20, 10))
		frame = fig.add_subplot(111)

		N = 1000
		Δϕ_new = np.linspace(-180, 180, 2*N)
		Δθ_new = np.linspace(0, 180, N)
		ΔΦ, ΔΘ = np.meshgrid(Δϕ_new, Δθ_new)
		P = np.array([ΔΦ.flatten(), ΔΘ.flatten()]).T
		ε_linear = griddata(Pi, ε, P, method="cubic").reshape([N, 2*N])
		
		frame.contourf(Δϕ_new, Δθ_new, ε_linear, 50, cmap=cmap)

		norm = colors.Normalize(np.min(ε), np.max(ε))
		scatter_setup = dict(s=10, marker='s', cmap=cmap, norm=norm, edgecolor='k', linewidths=0.4)
		frame.scatter(Δϕ, Δθ, c=ε, **scatter_setup)
		
		divider = make_axes_locatable(frame)
		cax = divider.append_axes("right", size="2%", pad=0.03)
		cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
		
		frame.set_xlabel(r"$\Delta\phi\ [^{\circ}]$", fontsize=16)
		frame.set_ylabel(r"$\Delta\theta\ [^{\circ}]$", fontsize=16)
		cbar.set_label(label=r"$ε\ [^{\circ}]$", fontsize=16)

		frame.set_xlim((-180, 180))
		frame.set_ylim((180, 0))

		frame.grid()
		show()
		"""

		#break

	pbar.close()

	"""
	#comment for plot per pulse
	#stack the data coordinates as follows (n, D) where n is the number of datapoints and D is the dimensionality of the data
	Pi = np.vstack((Δϕ, Δθ)).T
	print(Pi)

	cmap = gen_olaf_cmap()
	fig = figure(figsize=(20, 10))
	frame = fig.add_subplot(111, aspect='equal')

	N = 1000
	Δϕ_new = np.linspace(-180, 180, 2*N)
	Δθ_new = np.linspace(0, 180, N)
	ΔΦ, ΔΘ = np.meshgrid(Δϕ_new, Δθ_new)
	P = np.array([ΔΦ.flatten(), ΔΘ.flatten()]).T
	ε_linear = griddata(Pi, ε, P, method="cubic").reshape([N, 2*N])
	
	levels = np.linspace(-45, 45, 100)	
	frame.contourf(Δϕ_new, Δθ_new, ε_linear, levels, cmap=cmap)

	norm = colors.Normalize(np.min(ε), np.max(ε))
	scatter_setup = dict(s=10, marker='s', cmap=cmap, norm=norm, edgecolor='k', linewidths=0.4)
	frame.scatter(Δϕ, Δθ, c=ε, **scatter_setup)
		
	divider = make_axes_locatable(frame)
	cax = divider.append_axes("right", size="2%", pad=0.03)
	cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax)
		
	frame.set_xlabel(r"$\Delta\phi\ [^{\circ}]$", fontsize=16)
	frame.set_ylabel(r"$\Delta\theta\ [^{\circ}]$", fontsize=16)
	cbar.set_label(label=r"$ε\ [^{\circ}]$", fontsize=16)

	frame.set_xlim((-180, 180))
	frame.set_ylim((180, 0))

	frame.grid()
	"""


	#show()