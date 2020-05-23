#!/usr/bin/env python3

import numpy as np
from matplotlib.pyplot import figure, subplots_adjust, show
from tqdm import tqdm

from stokes import stokes_params, stokes_plotter

from LoLIM.signal_processing import half_hann_window
from LoLIM.antenna_response import LBA_antenna_model, invert_2X2_matrix_list

#function returning an amplitude envelope for the pulse
def envelope(t):
	sampling_period = t[1]-t[0]
	FWHM = 5*sampling_period #observed full width half maximum of a typical VHF pulse in lightning is ~25 ns
	σ = FWHM/(2*np.sqrt(2*np.log(2)))
	μ = t.size//2*sampling_period
	return np.exp(-1/2*((t - μ) / σ)**2)

#function returning pulse data in the azimuthal and zenithal direction
def E(t, ω, τ, ε, δ):
	R = np.array([[np.cos(τ), -np.sin(τ)], [np.sin(τ), np.cos(τ)]]) #counter clockwise rotation matrix
	complex_amplitude = np.array([np.cos(ε), np.sin(ε)*np.exp(1j*δ)]) #the complex amplitude can cause circular polarization by adjusting δ and ε
	E_field = np.einsum('i,j->ij', complex_amplitude, np.exp(1j*ω*t))
	E_field = np.matmul(R, E_field)
	E_field = envelope(t)*E_field #modulate the amplitude of the E_field using an envelope
	return E_field

#function mapping the azimuthal and zenithal electric fields coming from directly overhead (ϕ, θ) = (0, 0) to theoretical voltages measured by the X and Y antennas
"""
	Note1:	The antenna spherical coordinate system breaks down directly overhead so we set ϕ = 0 as a convention.
			Doing the projection for a linearly polarized wave we find:
				E_x = E*sin(τ - ϕ) and,
				E_y = E*cos(τ - ϕ)
			Thus setting ϕ = 0 is equivalent to absorbing ϕ into τ for a linearly polarized wave.
	Note2:	The positive x-axis is pointed to the East, whilst the positive X antenna is pointed in the SW direction.
			The positive y-axis is pointed to the North, whilst the positive Y antenna is pointed in the SE direction.
"""
def E_to_V(E_field):
	ϕ = 0
	θ = 0
	P = np.array([[-np.sin(ϕ), np.cos(θ)*np.cos(ϕ)], [np.cos(ϕ), np.cos(θ)*np.sin(ϕ)]]) #matrix that projects the E_field on the x-axis and y-axis
	E_field_xy = np.matmul(P, E_field)
	R = 1/np.sqrt(2)*np.array([[1, 1], [-1, 1]]) #rotation matrix that rotates the x,y projection by 45 degrees clockwise (i.e. the basis is rotated by 45 degrees counter-clockwise)
	E_field_XY = -np.matmul(R, E_field_xy) #the -1 is to flip the E_field components such that they align with the +X and +Y directions of the antennae
	antenna_length = np.sqrt(2)*1.380 + 0.090 #compute the antenna length parallel to the ground
	V_XY = E_field_XY*antenna_length #voltages are computed by multiplying by the length of the antenna parallel to the ground
	return V_XY

#function that plots the simulated pulse
def plot_simulated_pulse(t, E_field, V, cal_data):
	fig = figure(figsize=(10,10))
	subplots_adjust(wspace=0.2, hspace=0.3)

	frame1 = fig.add_subplot(221)
	frame1.set_title(r"Simulated $\vec{E}$ Field Data", fontsize=11)
	frame1.plot(t, np.real(E_field[0]), label=r"$E_{\phi}$", color='k')
	frame1.plot(t, np.real(E_field[1]), label=r"$E_{\theta}$", color='r')
	frame1.plot(t, np.abs(E_field[0]), label=r"$|E_{\phi}|$", color='k')
	frame1.plot(t, np.abs(E_field[1]), label=r"$|E_{\theta}|$", color='r')
	frame1.set_xlabel(r"$t\ [s]$", fontsize=11)
	frame1.set_ylabel(r"$V/m$ (arbitrary units)", fontsize=11)
	frame1.legend()
	frame1.grid()

	frame2 = fig.add_subplot(222)
	frame2.set_title(r"Simulated Received $V$ Data", fontsize=11)
	frame2.plot(t, np.real(V[0]), label=r"$V_X$ (odd)", color='k')
	frame2.plot(t, np.real(V[1]), label=r"$V_Y$ (even)", color='r')
	frame2.plot(t, np.abs(V[0]), label=r"$|V_X| (odd)$", color='k')
	frame2.plot(t, np.abs(V[1]), label=r"$|V_Y| (even)$", color='r')
	frame2.set_xlabel(r"$t\ [s]$", fontsize=11)
	frame2.set_ylabel(r"$V$ (arbitrary units)", fontsize=11)
	frame2.legend()
	frame2.grid()

	frame3  = fig.add_subplot(223)
	frame3.set_title(r"Reconstructed $\vec{E}$ Field Data", fontsize=11)
	frame3.plot(t, np.real(cal_data[0]), label=r"$E_{\phi}$", color='k')
	frame3.plot(t, np.real(cal_data[1]), label=r"$E_{\theta}$", color='r')
	frame3.plot(t, np.abs(cal_data[0]), label=r"$|E_{\phi}|$", color='k')
	frame3.plot(t, np.abs(cal_data[1]), label=r"$|E_{\theta}|$", color='r')
	frame3.set_xlabel(r"$t\ [s]$", fontsize=11)
	frame3.set_ylabel(r"$V/m$ (arbitrary units)", fontsize=11)
	frame3.legend()
	frame3.grid()

	show()

if __name__ == "__main__":
	N = 40 #number of samples we want
	ν_s = 200E6 #sampling frequency in Hz
	t = np.linspace(0, (N-1)/ν_s, N)

	ν = 58E6 #resonance frequency of a LOFAR LBA antenna
	ω = 2*np.pi*ν

	δ = np.pi/2 #the most general case is obtained setting δ = π/2

	stokes = stokes_params()

	##############################
	##	CALIBRATION VARIABLES	##
	##############################
	window = half_hann_window(N, half_percent=0.1)

	padSize = 2**16
	pad_width = (padSize - N)//2
	freq = np.fft.fftfreq(2*pad_width + N, 1/ν_s)

	#make a bandpass filter of 10 MHz around resonance frequency of antenna (i.e. 58 MHz)
	bandpass_filter = np.ones(freq.size)
	for i, n in np.ndenumerate(freq):
		if n < 58E6-5E6 or n > 58E6+5E6:
			bandpass_filter[i[0]] = 0

	antenna_model = LBA_antenna_model()


	#we use different number of samples to ensure an equal aspect ratio for the pixels
	τ_range = np.linspace(-np.pi/2, np.pi/2, 200) #τ can only lie between -90 deg and +90 deg (note: the -90 deg and +90 deg case are exactly the same for a polarization ellipse!)
	ε_range = np.linspace(-np.pi/4, np.pi/4, 100) #ε can only lie between -45 deg and +45 deg (note: the -45 deg and +45 deg case are not the same (ε expresses the chirality of the polarization ellipse)! However, in these cases τ can't be determined!
	
	results = np.empty((0,4))

	#running τ before ε ensures τ is the y axis and ε is the x axis
	pbar0 = tqdm(τ_range, ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
	for τ in pbar0:
		pbar1 = tqdm(ε_range, leave=False, ascii=True, unit_scale=True, dynamic_ncols=True, position=1)
		for ε in pbar1:
			
			E_field = E(t, ω, τ, ε, δ) #E waves in the azimuthal and zenithal direction
			V = E_to_V(E_field) #obtain the voltage data obtained by the antennae
			data_E = V[1]; data_O = V[0] #We assume LBA_OUTER antenna set in which case the Y antenna is the "even" antenna and the X antenna is the "odd" antenna

			"""
				Now we perform the calibration as we do with any pulse.
				Note: We skip applying the timeshift and applying the Galaxy calibration as these are not included in our model for the data.
			"""
			#apply a Tukey window (half Hann window):
			data_E *= window; data_O *= window

			#pad edges of the dataset with zeros such that the total size becomes 2**16 samples
			data_E = np.pad(data_E, (pad_width, pad_width), mode='constant', constant_values=(0,0))
			data_O = np.pad(data_O, (pad_width, pad_width), mode='constant', constant_values=(0,0))

			#prepare for calibration by performing a FFT
			data_E_FFT = np.fft.fft(data_E)
			data_O_FFT =  np.fft.fft(data_O)

			#apply bandpass filter
			data_E_FFT *= bandpass_filter
			data_O_FFT *= bandpass_filter

			#unravel the antenna response
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

			"""
			##########################################
			##	PLOT THE RESULT OF THE SIMULATION	##
			##########################################
			#apply bandpass filter to input signal for plotting purposes
			E_field_FFT = np.fft.fft(E_field)
			V_FFT = np.fft.fft(V)
			ν = np.fft.fftfreq(E_field[0].size, 1/ν_s)
			bandpass_filter_2 = np.ones(ν.size)
			for i, n in np.ndenumerate(ν):
				if n < 58E6-5E6 or n > 58E6+5E6:
					bandpass_filter_2[i[0]] = 0
			E_field_FFT *= bandpass_filter_2
			V_FFT *= bandpass_filter_2
			E_field = np.fft.ifft(E_field_FFT)
			V = np.fft.ifft(V_FFT)

			#function that will show what was simulated
			plot_simulated_pulse(t, E_field, V, (cal_data_az, cal_data_z))
			"""

			##########################
			##	RUN STOKES ANALYSIS	##
			##########################
			S = stokes.get_stokes_vector(cal_data_az, cal_data_z)
			pulseWidth, t_l, t_r = stokes.get_pulseWidth() #remember to comment out lines 52-56 in the stokes.py module otherwise it can't find the pulsewidth
			S_avg = stokes.average_stokes_parameters(S)
			dop = stokes.get_dop()
			pol_ell_params = stokes.polarization_ellipse_parameters()
			pol_ell_params.reshape((1, 4))
			
			#print the results
			#print("####	RESULTS	####")
			#print("τ = {} deg\nε = {} deg\ndop = {}".format(np.rad2deg(pol_ell_params[0][2]), np.rad2deg(pol_ell_params[0][3]), pol_ell_params[0][1]))

			#tqdm.write("####	RESULTS	####")
			#tqdm.write("τ = {} deg\nε = {} deg\ndop = {}".format(np.rad2deg(pol_ell_params[0][2]), np.rad2deg(pol_ell_params[0][3]), pol_ell_params[0][1]))

			tqdm.write("τ = {} & {}\nε = {} & {}".format(np.rad2deg(τ), np.rad2deg(pol_ell_params[0][2]), np.rad2deg(ε), np.rad2deg(pol_ell_params[0][3])))
			tqdm.write("τ_err = {} rad\nε_err = {} rad".format(np.rad2deg(abs(τ - pol_ell_params[0][2])), np.rad2deg(abs(ε - pol_ell_params[0][3]))))
			
			τ_err = abs(τ - pol_ell_params[0][2])
			ε_err = abs(ε - pol_ell_params[0][3])
			results = np.append(results, np.array([[τ, ε, τ_err, ε_err]]), axis=0)

			"""
			stokes_plot = stokes_plotter(plot=['stokes', 'polarization_ellipse'])
			stokes_plot.plot_stokes_parameters(S, antenna_names=[0,1], width=[t_l,t_r])
			stokes_plot.plot_polarization_ellipse(pol_ell_params)
			stokes_plot.showPlots(legend=True)
			"""

		pbar1.close()
	pbar0.close()

	np.savetxt("simulated_errors.txt", results, delimiter=' ', header="τ ε τ_err ε_err", comments='#')