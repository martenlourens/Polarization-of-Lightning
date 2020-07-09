#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import pickle
import json
from kapteyn import kmpfit
from matplotlib.pyplot import figure, show, setp, close
from matplotlib import colors
from scipy.optimize import brute
#from dynesty import NestedSampler, plotting
from tqdm import tqdm

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data, save_acceleration_vector

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

	τ = np.arccos(np.around(np.einsum('ij,ij->i', a_perp_hat, ϕ_hat), decimals=15))

	
	#θ = np.arctan2(np.linalg.norm(r[:, :2], axis=1), r[:, 2])
	#θ_hat = np.array([np.cos(θ)*np.cos(ϕ), np.cos(θ)*np.sin(ϕ), np.sin(θ)]).T #unit zenith vector

	#x = a_perp_hat - np.einsum('i,ij->ij', np.einsum('ij,ij->i', a_perp_hat, ϕ_hat), ϕ_hat)
	#α = np.einsum('ij,ij->i', x, θ_hat)
	#n = α/np.abs(α)

	#τ = np.einsum('i,i->i', n, abs_τ)

	return τ

def residuals(p, data):
	r, τ, τ_err = data

	#shift tau to align with model output
	for i in range(τ.size):
		if τ[i] < 0:
			τ[i] += np.pi

	res = τ - model(p, r)

	#there are two possible angles for 'res' the loop below ensures we always take the one for which res < 90 deg
	for i in range(res.size):
		if np.abs(res[i]) > np.pi/2:
			if res[i] < 0:
				res[i] += np.pi
			elif res[i] > 0:
				res[i] -= np.pi			

	return res/τ_err


if __name__ == "__main__":
	#station_names = natural_sort(station_timing_offsets.keys())
	data_folder = processed_data_folder + "/polarization_data/Lightning Phenomena/K changes"
	pName = "KC13" #phenomena name
	Zlimit = 50 #max zenith angle for which antenna model should hold!
	δlimit = 0.8 #lower limit for the degree of polarization (pulses below this limit are observed to have large errorbars due to local interference or pulses are mixed together in time)

	with open(data_folder + '/' + "source_info_{}.json".format(pName), 'r') as f:
		source_info = json.load(f)

	with open(data_folder + '/' + "pulseIDs_{}.pkl".format(pName), 'rb') as f:
		pulseIDs = pickle.load(f)

	rd = read_polarization_data(timeID, alt_loc=data_folder + '/' + "{}_data".format(pName))
	sv = save_acceleration_vector(timeID, fname="a_vectors_{}".format(pName), alt_loc=data_folder + '/' + "{}_data".format(pName))

	pbar = tqdm(pulseIDs, ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
	for pulseID in pbar:
		pbar.set_description("Processing pulse {}".format(pulseID))

		"""
			sort station names according to zenith angle (Z) and impose a Z requirement
		"""
		station_names = natural_sort(station_timing_offsets.keys())
		Z = np.array([])
		for station in station_names:
			#station location is determined from the average antenna locations
			antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
			avg_station_loc = np.average(antenna_locations, axis=0)

			#source location
			source_XYZ = source_info[str(pulseID)]['XYZT'][:3]

			Z = np.append(Z, zenith_to_src(source_XYZ, avg_station_loc))

		sort_indices = np.argsort(Z)
		Z = Z[sort_indices]
		station_names = np.array(station_names)[sort_indices]

		constrain_indices = np.where(Z<=Zlimit)[0]
		station_names = station_names[constrain_indices]
		Z = Z[constrain_indices]


		τ = np.array([])
		τ_err = np.array([])
		r = np.empty((0,3))

		df = rd.read_PE(pulseID)

		if df.empty:
			continue

		for station in station_names:
			if station in df.index:
				#station measurements yielding a degree of polarization below 'δlimit' are NOT accepted
				if df.loc[station]["δ"] < δlimit:
					continue

				τ = np.append(τ, df.loc[station]["τ"])
				τ_err = np.append(τ_err, df.loc[station]["σ_τ"])

				#compute r (i.e. radial vector pointing from station to source)
				antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
				avg_station_XYZ = np.average(antenna_locations, axis=0)
				source_XYZ = source_info[str(pulseID)]['XYZT'][:3]
				r = np.append(r, np.array([source_XYZ - avg_station_XYZ]), axis=0)

		"""Check whether dataset provides the proper degrees of freedom: ν = N_data - N_params, for a fit.
		If ν < 0 we can't do a fit, if ν == 0 we find an exact solution and if ν > 0 we can do a fit."""
		if τ.size - 2 <= 0:
			continue




		#THIS METHOD IS DEPRECATED AS IT DIDN'T WORK WELL!!!
		#Initial guess of 'a' ('a0')
		"""For each station we can accurately determine the direction of 'a' w.r.t. the azimuthal and zenithal axes.
		Averaging the vectors of each station supplies us with an initial guess of 'a' ('a0')."""
		#ϕ = np.arctan2(r[:, 1], r[:, 0])
		#ϕ_hat = np.array([-np.sin(ϕ), np.cos(ϕ), np.zeros(ϕ.size)]).T #unit azimuth vector
		#θ_z = np.arctan2(np.linalg.norm(r, axis=1), r[:, 2])
		#θ_z_hat = np.array([np.cos(θ_z)*np.cos(ϕ), np.cos(θ_z)*np.sin(ϕ), -np.sin(θ_z)]).T
		#a0 = np.empty((0, 3))
		#for i in range(τ.size):
		#	a0 = np.append(a0, np.array([np.cos(τ[i])*ϕ_hat[i] + np.sin(τ[i])*θ_z_hat[i]]), axis=0) #note: this vector is automatically normalized
		#a0 = np.average(a0, axis=0) #compute the average of all the 'a0' vectors making that our initial guess
		#a0 /= np.linalg.norm(a0) #re-normalize 'a0'

		"""
			For our fit of 'a' we want the magnitude of 'a' to remain fixed.
			I.e., 'a' has 2 degrees of freedom parametrized in polar coordinates by p=(ϕ, θ) (ϕ and θ are respectively the azimuthal and polar angles of 'a').
			We make an initial guess of these below called 'p0'.
		"""
		#p0 = np.zeros(2)
		#p0[0] = np.arctan2(a0[1], a0[0]) #ϕ = arctan(a_y/a_x)
		#p0[1] = np.arccos(a0[2]) #θ = arccos(a_z/|a|), since 'a' is normalized, |a| = 1.




		data = (r, τ, τ_err)

		#NEW METHOD!!!
		"""
			Before fitting we will sample the chisquare plane to make a first guess of 'p0'. This will be the sample for which χ2 is minimal.
		"""
		Ns = 200
		def chi2(p, *data):
			return np.sum(residuals(p, data)**2)
		prange = ((-np.pi, np.pi), (0, np.pi))
		p0, fval, ϕθ, χ2 = brute(chi2, ranges=prange, args=data, Ns=200, full_output=True, workers=-1)


		fitobj = kmpfit.Fitter(residuals)
		fitobj.data = data
		fitobj.params0 = p0
		#we don't impose any limits on the parameter space as that would cause branch cuts
		fitobj.fit()

		#Below we phase shift the results such that -π < ϕ < π and 0 < θ < π.
		#phase shift ϕ
		if fitobj.params[0] < 0:
			N = fitobj.params[0]//-np.pi
			fitobj.params = np.array([fitobj.params[0] + 2*np.pi*N, fitobj.params[1]])
		elif fitobj.params[0] > 0:
			N = fitobj.params[0]//np.pi
			fitobj.params = np.array([fitobj.params[0] - 2*np.pi*N, fitobj.params[1]])

		#phase shift θ
		if fitobj.params[1] < 0:
			N = fitobj.params[1]//-np.pi
			fitobj.params = np.array([fitobj.params[0], fitobj.params[1] + np.pi*(N+1)])
		elif fitobj.params[1] > 0:
			N = fitobj.params[1]//np.pi
			fitobj.params = np.array([fitobj.params[0], fitobj.params[1] - np.pi*N])

		#print the results		
		tqdm.write("\n#################### KMPFIT RESULTS PULSE {} ####################\n".format(pulseID))
		tqdm.write("p⃗:                               {}".format(fitobj.params))
		tqdm.write("σp⃗:                              {}".format(fitobj.xerror))
		tqdm.write("Uncertainties assuming χν^2=1:   {}".format(fitobj.stderr))
		tqdm.write("χ^2 min:                         {}".format(fitobj.chi2_min))
		tqdm.write("ν:                               {}".format(fitobj.dof))
		tqdm.write("χν^2:                            {}".format(fitobj.rchi2_min))
		tqdm.write("Cov:\n{}".format(fitobj.covar))
		tqdm.write("\n")

		sv.save_a_vector(pulseID, fitobj)

		
		######################################
		##	PLOT OF CHISQUARE DISTRIBUTION	##
		######################################
		ϕ = np.linspace(-np.pi, np.pi, Ns)
		θ = np.linspace(0, np.pi, Ns)
		
		level = np.logspace(np.log10(χ2.min()),np.log10(χ2.max()), 30)

		cmap = gen_olaf_cmap()
		norm = colors.LogNorm(χ2.min(), χ2.max())

		#plot without marginalization
		#fig = figure(figsize=(20,10))
		#frame = fig.add_subplot(111, aspect='equal')
		#cont = frame.contour(ϕθ[0], ϕθ[1], χ2, level, cmap=cmap, norm=norm)
		#frame.clabel(cont, level, fontsize=8, colors='k')
		#frame.plot(*fitobj.params, marker="*", color="k")
		#frame.set_xlabel(r"$ϕ$", fontsize=16)
		#frame.set_ylabel(r"$θ$", fontsize=16)
		#show()
		
		ϕ_lims = [-np.pi, np.pi]
		θ_lims = [np.pi, 0]

		fig = figure(figsize=(30, 20))
		gs = fig.add_gridspec(2, 2, width_ratios=(10, 1), height_ratios=(1, 5), wspace=0, hspace=0)

		frame_contour = fig.add_subplot(gs[1, 0])
		frame_contour.set_xlim(ϕ_lims)
		frame_contour.set_ylim(θ_lims)
		frame_contour.set_xlabel(r"$ϕ\ [rad]$", fontsize=16)
		frame_contour.set_ylabel(r"$θ\ [rad]$", fontsize=16)
		frame_contour.grid()

		cont = frame_contour.contour(ϕθ[0], ϕθ[1], χ2, level, cmap=cmap, norm=norm)
		frame_contour.clabel(cont, level, fontsize=8, inline_spacing=10, colors='k')
		frame_contour.plot(*fitobj.params, marker="*", color="k")
		frame_contour.axhline(y=fitobj.params[1], linestyle='dashed', linewidth=1, color='r')
		frame_contour.axvline(x=fitobj.params[0], linestyle='dashed', linewidth=1, color='r')


		frame_marg1 = fig.add_subplot(gs[0, 0], sharex=frame_contour)
		setp(frame_marg1.get_xticklabels(), visible=False)
		frame_marg1.set_yticks([])
		frame_marg1.set_ylabel(r"$\chi^2(\phi)$", fontsize=16)

		χ2ϕ = np.sum(χ2, axis=0)
		frame_marg1.plot(ϕ, χ2ϕ, color='k')
		frame_marg1.fill_between(ϕ, χ2ϕ.min(), χ2ϕ, color='k', alpha=0.25)
		frame_marg1.axvline(x=fitobj.params[0], linestyle='dashed', linewidth=1, color='r')
		frame_marg1.set_ylim((χ2ϕ.min(), None))

		frame_marg2 = fig.add_subplot(gs[1, 1], sharey=frame_contour)
		setp(frame_marg2.get_yticklabels(), visible=False)
		frame_marg2.set_xticks([])
		frame_marg2.set_xlabel(r"$\chi^2(\theta)$", fontsize=16)

		χ2θ = np.sum(χ2, axis=1)
		frame_marg2.plot(χ2θ, θ, color='k')
		frame_marg2.fill_betweenx(θ, χ2θ.min(), χ2θ, color='k', alpha=0.25)
		frame_marg2.axhline(y=fitobj.params[1], linestyle='dashed', linewidth=1, color='r')
		frame_marg2.set_xlim((χ2θ.min(), None))

		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "chisq_plots" + '/' + "{}.pdf".format(pulseID), dpi=fig.dpi, bbox_inches='tight')
		close(fig)
		

		##############################################
		##	NESTED SAMPLING TO FIND BEST ESTIMATE	##
		##############################################
		"""
		def loglikelihood(p):
			return np.sum( -1/2*np.log(2*np.pi) - np.log(data[2]) - 1/2*residuals(p, data)**2 )

		def prior_transform(u):
			u[0] *= 2*np.pi
			#u[0] -= np.pi
			u[1] *= np.pi
			return u

		ndim = 2
		nlive = 1000

		sampler = NestedSampler(loglikelihood, prior_transform, ndim, bound='single', nlive=nlive)
		sampler.run_nested(dlogz=0.01)
		res = sampler.results

		fig, axes = plotting.cornerplot(res, color='black', truths=np.zeros(ndim),
                           span=[(-np.pi, np.pi), (0, np.pi)],
                           show_titles=True, quantiles=None)
		#fig, axes = plotting.traceplot(res, truths=np.zeros(ndim),
        #                     truth_color='black', show_titles=True,
        #                     trace_cmap='viridis', connect=True,
        #                     connect_highlight=range(5))
		#fig, axes = plotting.runplot(res, color='black', mark_final_live=False,
        #                   logplot=True)  # static run

		show()
		"""
	pbar.close()