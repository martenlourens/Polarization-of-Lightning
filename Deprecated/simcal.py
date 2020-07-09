#!/usr/bin/env python3

import numpy as np

from LoLIM.signal_processing import half_hann_window
from LoLIM.antenna_response import LBA_antenna_model, invert_2X2_matrix_list

def simCal(N, ν_s, data_E, data_O):
	"""
		Now we perform the calibration as we do with any pulse.
		Note: We skip applying the timeshift and applying the Galaxy calibration as these are not included in our model for the data.
	"""
	#apply a Tukey window (half Hann window):
	window = half_hann_window(N, half_percent=0.1)
	data_E *= window; data_O *= window

	#pad edges of the dataset with zeros such that the total size becomes 2**16 samples
	padSize = 2**16
	pad_width = (padSize - N)//2
	data_E = np.pad(data_E, (pad_width, pad_width), mode='constant', constant_values=(0,0))
	data_O = np.pad(data_O, (pad_width, pad_width), mode='constant', constant_values=(0,0))

	#prepare for calibration by performing a FFT
	data_E_FFT = np.fft.fft(data_E)
	data_O_FFT =  np.fft.fft(data_O)
	freq = np.fft.fftfreq(data_E.size, 1/ν_s)

	#make a bandpass filter of 10 MHz around resonance frequency of antenna (i.e. 58 MHz)
	bandpass_filter = np.ones(freq.size)
	for i, n in np.ndenumerate(freq):
		if n < 58E6-5E6 or n > 58E6+5E6:
			bandpass_filter[i[0]] = 0

	#apply bandpass filter
	data_E_FFT *= bandpass_filter
	data_O_FFT *= bandpass_filter

	#unravel the antenna response
	antenna_model = LBA_antenna_model()
	jones_matrices = antenna_model.JonesMatrix_MultiFreq(freq, 0, 0)
	inverse_jones_matrix = invert_2X2_matrix_list(jones_matrices)
	zenith_component = data_O_FFT*inverse_jones_matrix[:, 0,0] + data_E_FFT*inverse_jones_matrix[:, 0,1]
	azimuth_component = data_O_FFT*inverse_jones_matrix[:, 1,0] + data_E_FFT*inverse_jones_matrix[:, 1,1]
	data_E_FFT = zenith_component
	data_O_FFT = azimuth_component

	#get the results by performing an IFFT
	cal_data_z, cal_data_az = np.fft.ifft(data_E_FFT), np.fft.ifft(data_O_FFT)

	#slice the dataset to remove the pads
	cal_data_z = cal_data_z[pad_width:pad_width + N]; cal_data_az = cal_data_az[pad_width:pad_width + N]

	return cal_data_z, cal_data_az