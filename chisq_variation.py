#!/usr/bin/env python3

import numpy as np
import json
from matplotlib.pyplot import figure, setp, show, close
from matplotlib import colors, cm
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

def compute_RMS(absdev, dof, Z, Zlimit=None, dop=None):
	tmp = np.copy(absdev)
	
	if Zlimit is not None and dop is not None:
		tmp = tmp[Z<Zlimit]
		dop = dop[Z<Zlimit]
		tmp = tmp[dop>=0.8]
	elif Zlimit is not None:
		tmp = tmp[Z<Zlimit]
	elif dop is not None:
		tmp = tmp[dop>=0.8]

	return np.sqrt(1/dof*np.sum(tmp**2))

def plot_chisq_variation(x, y, yerr, dof, rchi2, c, cmode='ε', Zlimit=None, dop=None, save=False):
	#'cmode' : possible modes are 'ε' and 'dop' (degree of polarization)
	#'x' should be a tuple containing station names and zenith angles (in degrees) for each station, i.e. (station_names, zenith_angles)
	#'y' should contain chi-square values and absolute deviations from the model (in radians) for each station, i.e. (chi-square, abs_dev)
	#'yerr' should contain the errors in the data fitted to the model (τ_err) in radians
	#'dof' should contain the degrees of freedom of the fitted data
	#'rchi2' should contain the reduced chi-square value corresponding to the fit
	#'c' should contain a data array defining the colors for each scatter point according to 'cmode'
	#'Zlimit' is the zenithal limit for which the fit of the acceleration direction was made
	station_names = x[0]
	Z = x[1]
	χ2 = y[0]
	absdev = np.rad2deg(y[1])
	absdev_err = np.rad2deg(yerr)

	fig  = figure(figsize=(20,10))
	gs = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.05)

	errorbar_setup = dict(fmt='.k', markersize=0.5, marker='s', capsize=2, capthick=1.5, elinewidth=1.5, ecolor='k', zorder=1)

	cmap = gen_olaf_cmap()
	if cmode == 'ε':
		c = np.rad2deg(c)	
	norm = colors.Normalize(np.min(c), np.max(c))
	scatter_setup = dict(s=150, marker='*', c=c, cmap=cmap, norm=norm, edgecolor='k', linewidths=0.4, zorder=2)

	frame2 = fig.add_subplot(gs[1])
	frame1 = fig.add_subplot(gs[0], sharex=frame2)


	frame1.scatter(station_names, χ2, **scatter_setup)
	frame1.axhline(y=rchi2, linestyle='dashed', linewidth=1, color='r')
	ytrans1 = frame1.get_yaxis_transform()
	frame1.annotate(r"$\chi^2_{\nu}$", [1.01, rchi2], xycoords=ytrans1, fontsize=16)
	setp(frame1.get_xticklabels(), visible=False)
	frame1.set_ylabel(r"$\chi^2$", fontsize=16)
	frame1.grid()

	RMS = compute_RMS(absdev, dof, Z, Zlimit=Zlimit, dop=dop)

	frame2.scatter(station_names, absdev, **scatter_setup)
	frame2.errorbar(station_names, absdev, yerr=absdev_err, **errorbar_setup)
	frame2.axhline(y=RMS, linestyle='dashed', linewidth=1, color='r')
	ytrans2 = frame2.get_yaxis_transform()
	frame2.annotate(r"RMS", [1.01, RMS], xycoords=ytrans2, fontsize=16)
	xticklabels = ["{}\n".format(s) + r"${:.1f}^{{\circ}}$".format(z) for s, z in zip(station_names, Z)]
	frame2.set_xticklabels(xticklabels, rotation=45)
	frame2.set_xlabel("station\n" + r"$\theta_z$", fontsize=16) #r"stations (ordered lowest $\rightarrow$ highest zenith angle)"
	frame2.set_ylabel(r"$\left|\tau_i - \tau\left(\vec{p}\right)\right|\ [^{\circ}]$", fontsize=16)
	frame2.set_ylim((0, None))
	frame2.grid()

	cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), orientation='vertical', ax=[frame1, frame2])
	if cmode == 'ε':
		cbar.set_label(label=r"$\epsilon\ [^{\circ}]$", fontsize=16)
	elif cmode == 'dop':
		cbar.set_label(label=r"$\delta$", fontsize=16)
	
	if save:
		return fig
	else:
		show()

if __name__ == "__main__":
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC9" #phenomena name

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
		source_info = json.load(f)

	rd = read_polarization_data(timeID, alt_loc=data_folder + '/' + "{}_data".format(pName))
	a_data = read_acceleration_vector(timeID, fname="a_vectors_{}".format(pName), alt_loc=data_folder + '/' + "{}_data".format(pName))

	pbar = tqdm(a_data.keys(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0) #a_data.keys()
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


		χ2 = np.array([]) #χ2 values for each station at the model τ
		absdev = np.array([]) #absolute deviations from the modelled τ for each station
		absdev_err = np.array([])

		#additional parameters
		dop = np.array([])
		ε = np.array([])
		ε_err = np.array([])


		df = rd.read_PE(pulseID)


		epsilon = np.array([])
		for station in station_names[constrain_indices]:
			if station in df.index:
				if df.loc[station]['δ'] <= 0.8:
					continue
				epsilon = np.append(epsilon, np.rad2deg(df.loc[station]['ε']))
		avg_epsilon = np.mean(epsilon)
		std_epsilon = 1/np.sqrt(epsilon.size)*np.std(epsilon, ddof=1)

		case1 = avg_epsilon + std_epsilon >= 0 and avg_epsilon <= 0
		case2 = avg_epsilon - std_epsilon <= 0 and avg_epsilon >= 0

		if case1 or case2:
			tqdm.write("{}".format(pulseID))
			tqdm.write("{} +- {} deg".format(avg_epsilon, std_epsilon))
		else:
			continue


		"""
		station_names = station_names[constrain_indices]
		Z = Z[constrain_indices]
		"""


		del_indices = []
		for i, station in np.ndenumerate(station_names):
			if station in df.index:
				if df.loc[station]['δ'] <= 0.8:
					del_indices.append(i)
					continue
				τ = np.array([df.loc[station]["τ"]])
				τ_err = np.array([df.loc[station]["σ_τ"]])
				
				dop = np.append(dop, df.loc[station]["δ"])
				ε = np.append(ε, df.loc[station]["ε"])
				ε_err = np.append(ε_err, df.loc[station]["σ_ε"])

				#compute r (i.e. radial vector pointing from station to source)
				antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
				avg_station_XYZ = np.average(antenna_locations, axis=0)
				source_XYZ = source_info[pulseID]['XYZT'][:3]
				r = np.array([source_XYZ - avg_station_XYZ])

				data = (r, τ, τ_err)

				#compute chisq for each station
				χ2 = np.append(χ2, residuals(a_data[pulseID]['params'], data)**2)
				
				#compute absolute deviations in τ from the model for each station
				absdev = np.append(absdev, τ_err*abs(residuals(a_data[pulseID]['params'], data)))
				absdev_err = np.append(absdev_err, τ_err)
			else:
				del_indices.append(i)

		station_names = np.delete(station_names, del_indices)
		Z = np.delete(Z, del_indices)
		
		#plot with ε cmap		
		fig = plot_chisq_variation((station_names, Z), (χ2, absdev), absdev_err, a_data[pulseID]['dof'], a_data[pulseID]['rchi2_min'], ε, cmode='ε', Zlimit=Zlimit, save=True)
		#fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "chisq_plots" + '/' + "{}_chisq_scatter_epsilon.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "chisq_variation_above_Zlim" + '/' + "{}_chisq_scatter_epsilon.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		#show()
		close(fig)

		#plot with dop cmap
		fig = plot_chisq_variation((station_names, Z), (χ2, absdev), absdev_err, a_data[pulseID]['dof'], a_data[pulseID]['rchi2_min'], dop, cmode='dop', Zlimit=Zlimit, save=True)
		#fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "chisq_plots" + '/' + "{}_chisq_scatter_dop.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "chisq_variation_above_Zlim" + '/' + "{}_chisq_scatter_dop.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		#show()
		close(fig)

	pbar.close()