#!/usr/bin/env python3

import numpy as np
from LoLIM.antenna_response import LBA_ant_calibrator
from LoLIM.signal_processing import half_hann_window

class calibrate:
	def __init__(self, timeID, width, verbose=False):
		self.AC = LBA_ant_calibrator(timeID=timeID)
		self.width = width
		self.padSize = 2**16
		self.pad_width = (self.padSize - self.width)//2

		self.verbose = verbose

	def run_calibration(self, source_XYZ, data_E, data_O, ant_name_E, ant_XYZ, time_shift):
		#apply half Hann window to surpress sharp cutoff at the edges of dataset
		if self.verbose: print("Windowing data...")
		window = half_hann_window(self.width, half_percent=0.1)
		data_E *= window; data_O *= window

		#pad edges of dataset with zeros such that the total size becomes 2**16 samples
		if self.verbose: print("Zero padding data...")
		data_E = np.pad(data_E,(self.pad_width, self.pad_width), mode='constant', constant_values=(0,0))
		data_O = np.pad(data_O,(self.pad_width, self.pad_width), mode='constant', constant_values=(0,0))

		#prepare datasets for calibration by performing a fast Fourier transform
		if self.verbose: print("Performing FFT...")
		self.AC.FFT_prep(ant_name_E, data_E, data_O)

		#shift in time with respect to even antenna to ensure they are aligned in frequency space
		if self.verbose: print("Applying timeshift...")
		self.AC.apply_time_shift(0, time_shift)

		#apply a calibration removing the spectrum of the Galaxy and perform a frequency response calibration
		if self.verbose: print("Applying Galaxy calibration and frequency response calibration...")
		self.AC.apply_GalaxyCal()

		if self.verbose: print("Applying antenna response calibration...")
		#get the location of the source in zenith and azimuth angles
		LFA = source_XYZ - ant_XYZ #LFA == Location From Antenna
		Az = np.arctan2(LFA[1], LFA[0])
		Z = np.arctan2(np.dot(LFA[:2], LFA[:2])**0.5, LFA[2])
		#convert angles to degrees
		Az = np.rad2deg(Az); Z = np.rad2deg(Z)
		#multiply voltage data in frequency space with the inverse Jones' matrix obtaining Zenithal and Azimuthal electric fields in frequency space
		self.AC.unravelAntennaResponce(Z, Az)

		#perform an inverse fft to get the electric fields in time
		if self.verbose: print("Performing IFFT...")
		cal_data_z, cal_data_az = self.AC.getResult()

		#slice the dataset to remove the pads
		if self.verbose: print("Removing pads...")
		cal_data_z = cal_data_z[self.pad_width:self.pad_width+self.width]; cal_data_az = cal_data_az[self.pad_width:self.pad_width+self.width]

		if self.verbose: print("Done.")

		return cal_data_az, cal_data_z