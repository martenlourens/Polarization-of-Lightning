#!/usr/bin/env python3

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.getTrace_fromLoc import getTrace_fromLoc

from LoLIM.antenna_response import LBA_ant_calibrator

from LoLIM.signal_processing import half_hann_window

import numpy as np
import matplotlib.pyplot as plt

from StokesParams import SP

#####################
##    SET PATHS    ##
#####################
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


#################################
##    LOAD DATA AND FILTERS    ##
#################################
#implement station timing offsets to data later!
TBB_data = {sname : MultiFile_Dal1(fpath, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas,
additional_ant_delays=additional_antenna_delays, only_complete_pairs=True) for sname, fpath in raw_fpaths.items() if sname in station_timing_offsets}

data_filters = {sname : window_and_filter(timeID=timeID,sname=sname) for sname in station_timing_offsets}
	
sorted_snames = natural_sort(station_timing_offsets.keys())

trace_locator = getTrace_fromLoc(TBB_data, data_filters, station_timing_offsets)


###########################
##    SOURCE LOCATION    ##
###########################
source_XYZT = np.array([-743.41, -3174.29, 4377.67, 1.1350323478]) #1477200
#source_XYZT = np.array([247.86, -3555.45, 4779.61, 1.1051278583]) #1409503
#source_XYZT = np.array([-1040.17, -2877.24, 4405.51, 1.1518796563]) #1515137
#source_XYZT = np.array([-1594.76, -3307.02, 4339.72, 1.1386715757]) #1485269
#source_XYZT = np.array([-1804.64, -2937.45, 4062.84, 1.1452615564]) #1500111

AC = LBA_ant_calibrator(timeID=timeID)

width = 40

#set parameters for padding array with zeros
padSize = 2**16
pad_width = (padSize-width)//2

plt.figure(figsize=(20, 10))

for sname in ['CS302']: #sorted_snames
	sTBB_data = TBB_data[sname]
	antenna_names_E = sTBB_data.get_antenna_names()[::2]
	antenna_names_O = sTBB_data.get_antenna_names()[1::2]
	antenna_locations = sTBB_data.get_LOFAR_centered_positions()[::2]

	#stokes = SP(plot=['stokes','polarization_ellipse','poincare'])

	for n in range(len(antenna_names_E)):
		start_sample_E, total_time_offset_E, arrival_time_E, data_E = trace_locator.get_trace_fromLoc(source_XYZT, antenna_names_E[n], width, do_remove_RFI=True, do_remove_saturation=True)
		start_sample_O, total_time_offset_O, arrival_time_O, data_O = trace_locator.get_trace_fromIndex(start_sample_E, antenna_names_O[n], width, do_remove_RFI=True, do_remove_saturation=True)

		#uncomment to plot Hilbert envelope of uncalibrated data
	
		plt.plot(5E-9*np.arange(data_E.size)*10**(9), np.real(data_E), color='k', label=r"Real Signal")
		plt.plot(5E-9*np.arange(data_E.size)*10**(9), np.abs(data_E), color='r', label=r"Hilbert Envelope")
		#plt.plot(5E-9*np.arange(data_O.size), np.real(data_O), label=r"{}".format(antenna_names_O[n]))
		#plt.plot(5E-9*np.arange(data_O.size)+total_time_offset_E-total_time_offset_O, np.abs(data_O), label=r"{}".format(antenna_names_O[n]))
		


		#apply half Hann window to surpress sharp cutoff at the edges of dataset
		window = half_hann_window(width, half_percent=0.1)
		data_E *= window; data_O *= window

		#pad edges of dataset with zeros such that the total size becomes 2**16 samples
		data_E = np.pad(data_E,(pad_width,pad_width),mode='constant',constant_values=(0,0))
		data_O = np.pad(data_O,(pad_width,pad_width),mode='constant',constant_values=(0,0))
		break
		#plot spectrum to check whether frequency space is well sampled (solved by padding data) and whether nonzero cutoffs cause high freq. peaks (solved with Hann window)
		"""
		data_Ek = np.fft.fftshift(np.fft.fft(np.real(data_E)))
		Efk = np.fft.fftshift(np.fft.fftfreq(data_E.size,d=5E-9))
		data_Ok = np.fft.fftshift(np.fft.fft(np.real(data_O)))
		Ofk = np.fft.fftshift(np.fft.fftfreq(data_O.size,d=5E-9))
		plt.plot(Efk,2*np.abs(data_Ek), label=r"{}".format(antenna_names_E[n]))
		plt.plot(Ofk,2*np.abs(data_Ok), label=r"{}".format(antenna_names_O[n]))
		plt.xlim((0,None))
		"""

		#prepare datasets for calibration by performing a fast Fourier transform
		AC.FFT_prep(antenna_names_E[n], data_E, data_O)

		#shift in time with respect to even antenna to ensure they are aligned in frequency space
		AC.apply_time_shift(0,total_time_offset_E-total_time_offset_O)

		#apply a calibration correcting the Hilbert envelope using the known Galactic spectrum and perform a frequency response calibration
		AC.apply_GalaxyCal()

		#uncomment to plot Hilbert envelope of calibrated Even and Odd antenna data
		"""
		cal_data_E, cal_data_O = AC.getResult()
		cal_data_E = cal_data_E[pad_width:pad_width+width]; cal_data_O = cal_data_O[pad_width:pad_width+width]
		plt.plot(5E-9*np.arange(cal_data_E.size),np.abs(cal_data_E), label=r"{}".format(antenna_names_E[n]))
		plt.plot(5E-9*np.arange(cal_data_O.size),np.abs(cal_data_O), label=r"{}".format(antenna_names_O[n]))
		"""

		#uncomment for frequency spectra after Galaxy and frequency response calibration
		"""
		cal_data_E, cal_data_O = AC.getResult()
		cal_data_Ek = np.fft.fftshift(np.fft.fft(np.real(cal_data_E)))
		cal_Efk = np.fft.fftshift(np.fft.fftfreq(cal_data_E.size,d=5E-9))
		cal_data_Ok = np.fft.fftshift(np.fft.fft(np.real(cal_data_O)))
		cal_Ofk = np.fft.fftshift(np.fft.fftfreq(cal_data_O.size,d=5E-9))
		plt.plot(cal_Efk,np.abs(cal_data_Ek), label=r"{}".format(antenna_names_E[n]))
		plt.plot(cal_Ofk,np.abs(cal_data_Ok), label=r"{}".format(antenna_names_O[n]))
		"""

		#get the location of the source in zenith and azimuth angles
		LFA = source_XYZT[:3] - antenna_locations[n] #LFA == Location From Antenna
		Az = np.arctan2(LFA[1], LFA[0])
		Z = np.arctan2(np.dot(LFA[:2],LFA[:2])**0.5, LFA[2])
		
		#convert angles to degrees
		Az = np.rad2deg(Az); Z = np.rad2deg(Z)

		#multiply the voltage data in frequency space with the inverse Jones' matrix to get the Zenithal and Azimuthal electric fields per frequency
		AC.unravelAntennaResponce(Z, Az)

		#perform an inverse fft to get the electric fields in time
		cal_data_z, cal_data_az = AC.getResult()

		#uncomment for frequency spectra after antenna response calibration
		"""
		cal_data_zk = np.fft.fftshift(np.fft.fft(np.real(cal_data_z)))
		cal_zfk = np.fft.fftshift(np.fft.fftfreq(cal_data_z.size,d=5E-9))
		cal_data_azk = np.fft.fftshift(np.fft.fft(np.real(cal_data_az)))
		cal_azfk = np.fft.fftshift(np.fft.fftfreq(cal_data_az.size,d=5E-9))
		plt.plot(cal_zfk,np.abs(cal_data_zk), label=r"Z : {}/{}".format(antenna_names_E[n], antenna_names_O[n]))
		plt.plot(cal_azfk,np.abs(cal_data_azk), label=r"Az : {}/{}".format(antenna_names_E[n], antenna_names_O[n]))
		"""

		#slice the dataset to remove the pads
		cal_data_z = cal_data_z[pad_width:pad_width+width]; cal_data_az = cal_data_az[pad_width:pad_width+width]

		#uncomment to plot Hilbert envelope of calibrated Zenithal and Azimuthal electric field
		"""
		plt.plot(5E-9*np.arange(cal_data_z.size),np.abs(cal_data_z), label=r"Z : {}/{}".format(antenna_names_E[n], antenna_names_O[n]))
		plt.plot(5E-9*np.arange(cal_data_az.size),np.abs(cal_data_az), label=r"Az : {}/{}".format(antenna_names_E[n], antenna_names_O[n]))
		"""
		
		#compute the Stokes parameters (antenna names are loaded for plotting purposes)
		#stokes.get_stokes_parameters(cal_data_az,cal_data_z,antenna_names=[antenna_names_E[n],antenna_names_O[n]])
		
		#get the width of the pulse(s) in the dataset
		#stokes.get_pulseWidth()

		#average the Stokes parameters over the pulse(s)
		#stokes.average_stokes_parameters(output=True)

		#compute degree of polarization
		#stokes.get_dop()

		#compute parameters for the polarization ellipse model
		#stokes.polarization_ellipse_parameters()

		#fill canvas with plots
		#stokes.plot_stokes_parameters(plotWidth=True)
		#stokes.plot_polarization_ellipse()
		#stokes.plot_poincare()

		#stokes.next_antenna()
		
	#plot a legend for the Stokes parameters plot
	#stokes.plotlegend(figure='stokes')

	#stokes.print_data_dicts()

	#uncomment for raw data plots
	plt.xlabel(r"$t\ [ns]$",fontsize=16)
	#plt.ylabel(r"Hilbert envelope [V/m]",fontsize=16)
	plt.ylabel(r"Signal [arbitrary units]", fontsize=16)

	#uncomment for frequency plots
	#plt.xlabel(r"$\nu\ [s^{-1}]$",fontsize=16)
	#plt.ylabel(r"Amplitude [arbitrary units]",fontsize=16)
	
	plt.legend()
	plt.grid()

plt.savefig("RD_HE.pdf", dpi='figure', bbox_inches='tight')
#plt.show()
