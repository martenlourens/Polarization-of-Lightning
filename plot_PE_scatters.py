#!/usr/bin/env python3

import numpy as np
from scipy.stats import poisson
from scipy.optimize import root_scalar
import json
import pickle
from matplotlib.pyplot import figure, show, close
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

from stokes_utils import zenith_to_src
from stokesIO import read_polarization_data


import matplotlib.pyplot as plt


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

def plot_PE_scatter(pulses_PE, station, Z, save=False):
	cmap = gen_olaf_cmap()

	errorbar_setup = dict(fmt='.k', markersize=0.5, marker='s', capsize=2, capthick=0.25, elinewidth=0.5, ecolor='k', zorder=1)
	scatter_setup = dict(s=10, marker='s', cmap=cmap, edgecolor='k', linewidths=0.4, zorder=2)

	fig = figure(figsize=(20,20))
	gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.15)

	fig.suptitle(r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z), fontsize=32, position=(0.5, 0.95))
			
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
		return fig, (frame1, frame2, frame3)
	else:
		show()

def capped_weights(σ, verbose=False):
	median = np.median(σ)
	mean = np.mean(σ)
	size = σ.size
	minimum = σ.min()
	if verbose:	print("median: {}\nmean: {}\nsize: {}\nminimum: {}".format(median, mean, size, minimum))
	σ_new = np.where(σ < median/size**0.5, median/size**0.5, σ)
	return 1/σ_new**2

#DEPRECATED!!!
"""
def find_hist_errorbars(count, p=1-1/3):
	#function for computing the errorbars of histograms
	#'count' is the bin count (it can be thought of as an average number of counts for a single bin)
	#'p' is the p-value

	#cumulative distribution function of a poissonian where L is the upper limit of the integral
	def CDF(L):
		return 1 - poisson.cdf(count, L)

	#lower root function
	def rf_lower(L):
		return (1.0-CDF(L)) - p

	#upper root function
	def rf_upper(L):
		return CDF(L) - p

	#derivative of lower root function
	def deriv_rf_lower(L):
		ΔL = 0.00001
		if L - ΔL < 0:
			deriv = (rf_lower(L + ΔL) - rf_lower(L))/ΔL
		else:
			deriv = (rf_lower(L + ΔL) - rf_lower(L - ΔL))/(2*ΔL)
		return deriv

	#derivative of upper root function
	def deriv_rf_upper(L):
		ΔL = 0.00001
		if L - ΔL < 0:
			deriv = (rf_upper(L + ΔL) - rf_upper(L))/ΔL
		else:
			deriv = (rf_upper(L + ΔL) - rf_upper(L - ΔL))/(2*ΔL)
		return deriv

	#Find the roots of both the upper and lower root functions using the Newton-Raphson method.
	lower_result = root_scalar(rf_lower, x0=count, fprime=deriv_rf_lower, method='newton')
	upper_result = root_scalar(rf_upper, x0=count, fprime=deriv_rf_upper, method='newton')

	#compute the size of the errorbars
	σ_lower = count - lower_result.root
	σ_upper = upper_result.root - count


	#L = np.linspace(0, 2*max(lower_result.root, upper_result.root), 100)
	#plt.plot(L, rf_lower(L))
	#plt.plot(L, rf_upper(L))
	#plt.plot(L, CDF(L))
	#plt.scatter(lower_result.root, rf_lower(lower_result.root))
	#plt.scatter(upper_result.root, rf_upper(upper_result.root))
	#plt.show()


	if σ_lower < 0: σ_lower = count
	return σ_lower, σ_upper
"""

def plot_PE_histograms(pulses_PE, station, Z, save=False):
	fig = figure(figsize=(20,20))
	gs = fig.add_gridspec(2, 2, wspace=0.2, hspace=0.2)

	bins = 35
	bar_setup = dict(color="yellowgreen", align="edge", edgecolor='k', linewidth=1, zorder=0)
	#errorbar_setup = dict(ecolor='k', capsize=4, capthick=0.5, elinewidth=1, zorder=2)
	hist_setup = dict(density=True, edgecolor="firebrick", linewidth=3, fc='none', histtype='step', zorder=1)

	fig.suptitle(r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z), fontsize=32, position=(0.5, 0.96))


	frame1 = fig.add_subplot(gs[0])
	
	#τ unweighted hist with errorbars
	bin_counts, bin_edges = np.histogram(pulses_PE['τ'], bins=bins, range=(-90, 90))
	#errorbars = np.array([find_hist_errorbars(count) for count in bin_counts]).T
	frame1.bar(x=bin_edges[:-1], height=bin_counts, width=bin_edges[1:]-bin_edges[:-1], **bar_setup)#, yerr=errorbars, **bar_setup, error_kw=errorbar_setup)
	
	#τ weighted histogram
	frame1_twin = frame1.twinx()
	weights_τ = capped_weights(np.array(pulses_PE["σ_τ"]))
	frame1_twin.hist(pulses_PE['τ'], weights=weights_τ, bins=bins, range=(-90, 90), **hist_setup)
	frame1_twin.set_ylabel(r"weighted hist", fontsize=16, color="firebrick")


	#DEPRECATED!!!
	#weights_τ = 1/np.array(pulses_PE["σ_τ"])**2
	#sns.distplot(pulses_PE['τ'], kde=True, norm_hist=True, bins=range(-90, 90, 10), ax=frame1, color='k', kde_kws={"shade" : True}, hist_kws={"weights" : weights_τ, **hist_setup})
	
	frame1.set_xlabel(r"$τ\ [^{\circ}]$", fontsize=16)
	frame1.set_ylabel(r"unweighted hist", fontsize=16)
	frame1.set_xlim(-90, 90)
	frame1.grid()


	frame2 = fig.add_subplot(gs[1])
	
	#ε unweighted hist with errorbars
	bin_counts, bin_edges = np.histogram(pulses_PE['ε'], bins=bins, range=(-45, 45))
	#errorbars = np.array([find_hist_errorbars(count) for count in bin_counts]).T
	frame2.bar(x=bin_edges[:-1], height=bin_counts, width=bin_edges[1:]-bin_edges[:-1], **bar_setup)#, yerr=errorbars, **bar_setup, error_kw=errorbar_setup)

	#ε weighted histogram
	frame2_twin = frame2.twinx()
	weights_ε = capped_weights(np.array(pulses_PE["σ_ε"]))
	frame2_twin.hist(pulses_PE['ε'], weights=weights_ε, bins=bins, range=(-45, 45), **hist_setup)
	frame2_twin.set_ylabel(r"weighted hist", fontsize=16, color="firebrick")

	#DEPRECATED!!!
	#weights_ε = 1/np.array(pulses_PE["σ_ε"])**2
	#sns.distplot(pulses_PE['ε'], kde=True, norm_hist=True, bins=range(-45, 45, 5), ax=frame2, color='k', kde_kws={"shade" : True}, hist_kws={"weights" : weights_ε, **hist_setup})
	
	frame2.set_xlabel(r"$ε\ [^{\circ}]$", fontsize=16)
	frame2.set_ylabel(r"unweighted hist", fontsize=16)
	frame2.set_xlim(-45, 45)
	frame2.grid()


	frame3 = fig.add_subplot(gs[2])
	
	#δ unweighted histogram with errorbars
	bin_counts, bin_edges = np.histogram(pulses_PE['δ'], bins=bins, range=(0.8, 1))
	#errorbars = np.array([find_hist_errorbars(count) for count in bin_counts]).T
	frame3.bar(x=bin_edges[:-1], height=bin_counts, width=bin_edges[1:]-bin_edges[:-1], **bar_setup)#, yerr=errorbars, **bar_setup, error_kw=errorbar_setup)

	#δ weighted histogram
	frame3_twin = frame3.twinx()
	weights_δ = capped_weights(np.array(pulses_PE["σ_δ"]))
	frame3_twin.hist(pulses_PE['δ'], weights=weights_δ, bins=bins, range=(0.8, 1), **hist_setup)
	frame3_twin.set_ylabel(r"weighted hist", fontsize=16, color="firebrick")

	#DEPRECATED!!!
	#weights_δ = capped_weights(np.array(pulses_PE["σ_δ"])) #1/np.array(pulses_PE["σ_δ"])**2
	#sns.kdeplot(pulses_PE['δ'], shade=True, color='k', ax=frame3)
	#frame3.hist(pulses_PE['δ'], **hist_setup)#, weights=weights_δ)
	#sns.distplot(pulses_PE['δ'], kde=True, ax=frame3, color='k', kde_kws={"shade" : True}, hist_kws={"weights" : weights_δ, **hist_setup})
	
	frame3.set_xlabel(r"$δ$", fontsize=16)
	frame3.set_ylabel(r"unweighted hist", fontsize=16)
	frame3.set_xlim(0.8, 1)
	frame3.grid()

	if save:
		return fig, (frame1, frame2, frame3)
	else:
		show()

"""
def weighted_KDE(x, μ, σ, density=False):
	def Gaussian(μ_i, σ_i):
		return 1/np.sqrt(2*np.pi*σ_i**2)*np.exp(-1/2*((x - μ_i)/σ_i)**2)

	KDE = np.zeros(x.size)
	for i in range(μ.size):
		KDE += Gaussian(μ[i], σ[i])

	if density:
		KDE /= x.size

	return KDE
"""

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


	#figure initialization for full hist
	"""
	total_hist_setup = dict(density=True, lw=1.5, alpha=0.5, histtype='stepfilled', zorder=-1)
	fig_total_hist = figure(figsize=(20, 20))
	gs = fig_total_hist.add_gridspec(2, 2, wspace=0.2, hspace=0.2)
	
	frame1 = fig_total_hist.add_subplot(gs[0])
	frame1.set_xlabel(r"$τ\ [^{\circ}]$", fontsize=16)
	frame1.set_ylabel(r"kernel density", fontsize=16)
	frame1.set_xlim(-90, 90)
	frame1.grid()
	
	frame2 = fig_total_hist.add_subplot(gs[1])
	frame2.set_xlabel(r"$ε\ [^{\circ}]$", fontsize=16)
	frame2.set_ylabel(r"kernel density", fontsize=16)
	frame2.set_xlim(-45, 45)
	frame2.grid()

	frame3 = fig_total_hist.add_subplot(gs[2])
	frame3.set_xlabel(r"$δ$", fontsize=16)
	frame3.set_ylabel(r"kernel density", fontsize=16)
	frame3.set_xlim(0, 1)
	frame3.grid()
	"""

	#initialization for avg_ε versus station scatter
	fig_ε = figure(figsize=(20,10))
	frame4 = fig_ε.add_subplot(111)
	errorbar_setup = dict(fmt='sk', markersize=0.5, capsize=6, capthick=0.75, elinewidth=1.5, ecolor='k', zorder=1)
	scatter_setup = dict(s=75, marker='*', color='r', edgecolor='k', linewidths=0.4, zorder=2)
	xtick_labels = []
	frame4.set_xlabel("station", fontsize=16)
	frame4.set_ylabel(r"$\overline{\epsilon}$", fontsize=16)
	frame4.grid()

	#N1 = 0
	#N2 = 0
	#station_names = ['CS001']
	for i, station in enumerate(station_names):
		print(station)
		pulses_PE = {'δ' : [], "σ_δ" : [], 'τ' : [], "σ_τ" : [], 'ε' : [], "σ_ε" : []}
		
		for pulseID in pulseIDs:

			df = rd.read_PE(pulseID)

			if df.empty:
				continue

			if station in df.index:
				if df.loc[station]['δ'] <= 0.8:
					continue
				pulses_PE['δ'].append(df.loc[station]['δ'])
				pulses_PE["σ_δ"].append(df.loc[station]["σ_δ"])
				pulses_PE['τ'].append(np.rad2deg(df.loc[station]['τ']))
				pulses_PE["σ_τ"].append(np.rad2deg(df.loc[station]["σ_τ"]))
				pulses_PE['ε'].append(np.rad2deg(df.loc[station]['ε']))
				pulses_PE["σ_ε"].append(np.rad2deg(df.loc[station]["σ_ε"]))

		if pulses_PE == {'δ' : [], "σ_δ" : [], 'τ' : [], "σ_τ" : [], 'ε' : [], "σ_ε" : []}:
			continue

		#N1 += 1
		#N2 += len(pulses_PE['δ'])

		fig, frames = plot_PE_scatter(pulses_PE, station, Z[i], save=True)
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "PE_scatters" + '/' + "{}_PE_scatter.pdf".format(station), dpi=fig.dpi, bbox_inches='tight')
		close(fig)

		fig, frames = plot_PE_histograms(pulses_PE, station, Z[i], save=True)
		#extent = frames[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
		fig.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "PE_scatters" + '/' + "{}_PE_hist.pdf".format(station), dpi=fig.dpi, bbox_inches='tight') #extent
		#show()
		close(fig)
		
		#plot histograms per station on total histogram canvas
		#τ = np.linspace(-90, 90, 1000)
		#frame1.plot(τ, weighted_KDE(τ, np.array(pulses_PE['τ']), np.array(pulses_PE["σ_τ"]), density=True), label=r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z[i]))

		#DEPRECATED!!!
		#weights_τ = capped_weights(np.array(pulses_PE["σ_τ"]))
		#frame1.hist(pulses_PE['τ'], bins=range(-90, 90, 10), range=(-90, 90), weights=weights_τ, **total_hist_setup, label=r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z[i]))
		#weights_ε = capped_weights(np.array(pulses_PE["σ_ε"]))
		#frame2.hist(pulses_PE['ε'], bins=range(-45, 45, 5), range=(-45, 45), weights=weights_ε, **total_hist_setup)
		#weights_δ = capped_weights(np.array(pulses_PE["σ_δ"]))
		#binSize_δ = 0.01
		#frame3.hist(pulses_PE['δ'], bins=np.arange(0.8, 1+binSize_δ, binSize_δ), range=(0.8, 1), weights=weights_δ, **total_hist_setup)

		#DEPRECATED!!!
		#sns.kdeplot(pulses_PE['τ'], shade=True, ax=frame1, label=r"{} ($\overline{{\theta}}_z = {:.2f}^{{\circ}}$)".format(station, Z[i]))
		#sns.kdeplot(pulses_PE['ε'], shade=True, ax=frame2)
		#sns.kdeplot(pulses_PE['δ'], shade=True, ax=frame3)

		#avg_ε versus station scatter plot
		#weights = 1/np.array(pulses_PE["σ_ε"])**2
		weights = capped_weights(np.array(pulses_PE["σ_ε"]))
		partition = np.sum(weights)
		avg_ε = np.sum(np.array(pulses_PE['ε'])*weights)/partition
		s_avg_ε = np.sqrt(partition**(-1))
		#print(avg_ε, s_avg_ε)

		#DEPRECATED!!!
		#avg_ε = np.mean(np.array(pulses_PE['ε']))
		#s_avg_ε = 1/np.sqrt(len(pulses_PE['ε']))*np.std(np.array(pulses_PE['ε']), ddof=1)

		xtick_labels.append(station)
		frame4.errorbar([station], [avg_ε], yerr=[s_avg_ε], **errorbar_setup)
		frame4.scatter([station], [avg_ε], **scatter_setup)

		#print data in LaTeX format
		#print("		{} & {:.2f} & {:.2f}".format(station, avg_ε, s_avg_ε), r"\\")

	#print(N2/N1)
	#make a legend for fig_total_hist
	"""
	frame1.legend().set_visible(False)
	handles, labels = frame1.get_legend_handles_labels()
	legend_frame = fig_total_hist.add_subplot(gs[3])
	legend_frame.axis('off')
	legend_frame.legend(handles, labels, loc='center', ncol=2, fontsize=16)

	fig_total_hist.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "PE_scatters" + '/' + "total_PE_hist.pdf", dpi=fig_total_hist.dpi, bbox_inches='tight')
	close(fig_total_hist)
	"""

	#set ticklabels and save fig_ε
	frame4.set_xticklabels(xtick_labels, rotation=45)
	fig_ε.savefig(data_folder + '/' + "{}_data".format(pName) + '/' + "PE_scatters" + '/' + "avg_epsilon_scatter.pdf", dpi=fig_ε.dpi, bbox_inches='tight')
	close(fig_ε)