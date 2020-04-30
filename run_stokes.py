#!/usr/bin/env python3

#LoLIM imports
from LoLIM import utilities
from LoLIM.utilities import processed_data_dir, natural_sort
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.findRFI import window_and_filter
from LoLIM.getTrace_fromLoc import getTrace_fromLoc

#Python module imports
import numpy as np
from tqdm import tqdm, trange
import json

#Own module imports
from pulse_calibration import calibrate
from stokes import stokes_params, stokes_plotter
from stokesIO import save_polarization_data

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
tqdm.write("Loading TBB data:")
pbar1 = tqdm(raw_fpaths.items(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
TBB_data = {sname : MultiFile_Dal1(fpath, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas,
additional_ant_delays=additional_antenna_delays, only_complete_pairs=True) for sname, fpath in pbar1 if sname in station_timing_offsets}
pbar1.close()

tqdm.write("Loading data filters:")
pbar1 = tqdm(station_timing_offsets, ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
data_filters = {sname : window_and_filter(timeID=timeID,sname=sname) for sname in pbar1}
pbar1.close()
	
sorted_snames = natural_sort(station_timing_offsets.keys())

trace_locator = getTrace_fromLoc(TBB_data, data_filters, station_timing_offsets)


#############################################
##    IMPORT SOURCE INFO FROM JSON FILE    ##
#############################################

width = 40 #set width of datablock

cal = calibrate(timeID,width,verbose=False)

stokes = stokes_params()
stokes_plot = stokes_plotter(plot=['stokes'])

with open(processed_data_dir(timeID)+"/polarization_data/source_info.json", 'r') as f:
	source_info = json.load(f)

#SIMPLE TEST PULSES
#source_info = {'1477200': {'XYZT' : [-743.41, -3174.29, 4377.67, 1.1350323478]}}
#source_info = {'1409503': {'XYZT' : [247.86, -3555.45, 4779.61, 1.1051278583]}}
#source_info = {'1515137': {'XYZT' : [-1040.17, -2877.24, 4405.51, 1.1518796563]}}
#source_info = {'1485269': {'XYZT' : [-1594.76, -3307.02, 4339.72, 1.1386715757]}}
#source_info = {'1500111': {'XYZT' : [-1804.64, -2937.45, 4062.84, 1.1452615564]}}

#COMPLEX TEST PULSES
#source_info = {'1422213': {'XYZT' : [246.96, -3558.95, 4785.46, 1.1107716958]}}
source_info = {'1422214': {'XYZT' : [247.14, -3558.79, 4785.24, 1.1107717917]}}

pbar = tqdm(source_info.keys(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0)
for ID in pbar:
	source_XYZT = np.array(source_info[ID]['XYZT']); srcName = int(ID)

	pbar.set_description("Processing pulse {}".format(srcName))

	sIO = save_polarization_data(timeID,srcName)

	pbar1 = tqdm(["CS301"], leave=False, ascii=True, unit_scale=True, dynamic_ncols=True, position=1) 
	for sname in pbar1: #sorted_snames
		pbar1.set_description("Processing station {}".format(sname))

		sTBB_data = TBB_data[sname]
		antenna_names_E = sTBB_data.get_antenna_names()[::2]
		antenna_names_O = sTBB_data.get_antenna_names()[1::2]
		antenna_locations = sTBB_data.get_LOFAR_centered_positions()[::2]

		sStokesVector = [] #station Stokes vector
		PEVector = [] #station polarization ellipse vector

		pbar2 = trange(len(antenna_names_E), leave=False, ascii=True, unit_scale=True, dynamic_ncols=True, position=2) 
		for n in pbar2:
			pbar2.set_description("Processing antenna set {}/{}".format(antenna_names_E[n],antenna_names_O[n]))

			start_sample_E, total_time_offset_E, arrival_time_E, data_E = trace_locator.get_trace_fromLoc(source_XYZT, antenna_names_E[n], width, do_remove_RFI=True, do_remove_saturation=True)
			start_sample_O, total_time_offset_O, arrival_time_O, data_O = trace_locator.get_trace_fromIndex(start_sample_E, antenna_names_O[n], width, do_remove_RFI=True, do_remove_saturation=True)

			cal_data_az, cal_data_z = cal.run_calibration(source_XYZT[:3], data_E,data_O, antenna_names_E[n], antenna_locations[n], total_time_offset_E - total_time_offset_O)

			S = stokes.get_stokes_vector(cal_data_az,cal_data_z)
			pulseWidth, t_l, t_r = stokes.get_pulseWidth()
			if pulseWidth==0:
				continue

			S_avg = stokes.average_stokes_parameters(S)
			sStokesVector.append(S_avg)

			stokes.get_dop()

			pol_ell_params = stokes.polarization_ellipse_parameters()
			PEVector.append(pol_ell_params)

			stokes_plot.plot_stokes_parameters(S,antenna_names=[antenna_names_E[n],antenna_names_O[n]], width=[t_l,t_r])

		pbar2.close()

		if sStokesVector:
			sStokesVector = np.array(sStokesVector)
			PEVector = np.array(PEVector)
			if sStokesVector.shape[0]>1: #we need more than one sample for a good estimate of the standard deviation
				tqdm.write("Saving polarization data for station {}...".format(sname))
				sIO.save_S(sname,np.average(sStokesVector,axis=0)[0],np.std(sStokesVector,axis=0,ddof=1)[0])
				sIO.save_PE(sname,np.average(PEVector,axis=0)[0],np.std(PEVector,axis=0,ddof=1)[0])
		else:
			tqdm.write("Polarization data  of station {} will not be saved as less than two antennas have received a measurable signal.".format(sname))

	pbar1.close()
	tqdm.write("Done.")

	stokes_plot.showPlots(legend=True)
	break

pbar.close()