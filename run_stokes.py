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
from stokes_utils import zenith_to_src
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

pName = "KC13" #phenomena name
source_info_loc = processed_data_dir(timeID) + "/polarization_data/Lightning Phenomena/K changes"
with open(source_info_loc + '/' + "source_info_{}.json".format(pName), 'r') as f:
	source_info = json.load(f)


#SIMPLE TEST PULSES
#source_info = {'1477200': {'XYZT' : [-743.41, -3174.29, 4377.67, 1.1350323478]}}
#source_info = {'1409503': {'XYZT' : [247.86, -3555.45, 4779.61, 1.1051278583]}}
#source_info = {'1515137': {'XYZT' : [-1040.17, -2877.24, 4405.51, 1.1518796563]}}
#source_info = {'1485269': {'XYZT' : [-1594.76, -3307.02, 4339.72, 1.1386715757]}}
#source_info = {'1500111': {'XYZT' : [-1804.64, -2937.45, 4062.84, 1.1452615564]}}

#COMPLEX TEST PULSES
#source_info = {'1422213': {'XYZT' : [246.96, -3558.95, 4785.46, 1.1107716958]}}
#source_info = {'1422214': {'XYZT' : [247.14, -3558.79, 4785.24, 1.1107717917]}}


#TEST
"""paste here any sources you want to test"""
#source_info = {'673142': {'XYZT': [518.3462239330252, -8046.281136970598, 1325.8904768440038, 0.7794669167761877], 'covXYZ': [[0.1317841460918076, 0.02490589272650902, 0.2590605178220296], [0.02490589272650902, 0.012351375315124712, 0.09332010610219753], [0.2590605178220296, 0.09332010610219753, 1.4206201496288422]], 'rmsT': 1.865154995834813e-09}, '669714': {'XYZT': [593.3310092183877, -8293.584380922252, 1298.061568679833, 0.7779925076940286], 'covXYZ': [[0.3326047735020519, -0.8701458108672928, 0.3092000033747412], [-0.8701458108672928, 3.224135369934415, -0.39229011663509017], [0.3092000033747412, -0.39229011663509017, 1.471403985122024]], 'rmsT': 2.9851182996823564e-09}, '670215': {'XYZT': [578.8219050262953, -8181.435889163621, 1313.3178734680193, 0.7782138717053353], 'covXYZ': [[0.013283992633874496, 8.282648658564857e-05, 0.023676757021448523], [8.282648658564857e-05, 0.00656106579101438, 0.03496776331783239], [0.023676757021448523, 0.03496776331783239, 0.7547878775689175]], 'rmsT': 2.299808843086895e-09}, '669212': {'XYZT': [576.2784274247553, -8222.39637990222, 1348.4849671004492, 0.7777726638803316], 'covXYZ': [[0.01169246573140886, -0.0005644081360223011, 0.01904176194439015], [-0.0005644081360223011, 0.006761741374154555, 0.03726408132698664], [0.01904176194439015, 0.03726408132698664, 0.7903190925426798]], 'rmsT': 2.3226741967504635e-09}}

#sort station names by distance to average source in source_info
Z = np.array([])
for station in sorted_snames:
	antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]
	avg_station_loc = np.average(antenna_locations, axis=0)

	N = 0
	loc_sum = np.zeros(3)
	for ID in source_info.keys():
		loc_sum += source_info[ID]['XYZT'][:3]
		N += 1
	avg_source_XYZ = loc_sum/N

	Z = np.append(Z, zenith_to_src(avg_source_XYZ, avg_station_loc)) 

sort_indices = np.argsort(Z)
sorted_snames = list(sorted_snames)
sorted_snames = [sorted_snames[i] for i in sort_indices]
Z = Z[sort_indices]

#create empty Z file or flush it
Z_file = open(source_info_loc + '/' + "{}_data".format(pName) + '/' + "Z", 'w')
Z_file.write('')
Z_file.close()

Z_file = open(source_info_loc + '/' + "{}_data".format(pName) + '/' + "Z", 'a')
for i in range(sort_indices.size):
	print("{} : {} deg".format(sorted_snames[i],Z[i]))
	print("{} : {} deg".format(sorted_snames[i],Z[i]), file=Z_file) #also print the results to Z
Z_file.close()

"""
Zlimit = 50
sorted_snames = [sorted_snames[i] for i in np.where(Z<=Zlimit)[0]]
"""

pbar = tqdm(source_info.keys(), ascii=True, unit_scale=True, dynamic_ncols=True, position=0) #source_info.keys()
for ID in pbar:
	tqdm.write("{}".format(ID))
	source_XYZT = np.array(source_info[ID]['XYZT']); srcName = int(ID)

	pbar.set_description("Processing pulse {}".format(srcName))

	sIO = save_polarization_data(timeID,srcName,alt_loc=source_info_loc + '/' + "{}_data".format(pName))

	pbar1 = tqdm(sorted_snames, leave=False, ascii=True, unit_scale=True, dynamic_ncols=True, position=1) #sorted_snames
	for sname in pbar1:
		pbar1.set_description("Processing station {}".format(sname))

		#stokes_plot = stokes_plotter(plot=['stokes']) #uncomment for plots

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

			cal_data_az, cal_data_z = cal.run_calibration(source_XYZT[:3], data_E, data_O, antenna_names_E[n], antenna_locations[n], total_time_offset_E - total_time_offset_O)

			S = stokes.get_stokes_vector(cal_data_az,cal_data_z)
			pulseWidth, t_l, t_r = stokes.get_pulseWidth()
			if pulseWidth==0:
				continue

			S_avg = stokes.average_stokes_parameters(S)
			sStokesVector.append(S_avg)

			stokes.get_dop()

			pol_ell_params = stokes.polarization_ellipse_parameters()
			PEVector.append(pol_ell_params)

			#stokes_plot.plot_stokes_parameters(S,antenna_names=[antenna_names_E[n],antenna_names_O[n]], width=[t_l,t_r]) #uncomment for plots

		pbar2.close()
		tqdm.write("{} : {}".format(sname, len(sStokesVector)))

		if sStokesVector:
			sStokesVector = np.array(sStokesVector)
			PEVector = np.array(PEVector)
			if sStokesVector.shape[0]>1: #we need more than one sample for a good estimate of the standard deviation
				tqdm.write("Saving polarization data for station {}...".format(sname))
				sIO.save_S(sname,np.average(sStokesVector,axis=0)[0],np.std(sStokesVector,axis=0,ddof=1)[0])
				sIO.save_PE(sname,np.average(PEVector,axis=0)[0],np.std(PEVector,axis=0,ddof=1)[0])
			else:
				tqdm.write("Polarization data of station {} will not be saved as less than two antennas have received a measurable signal.".format(sname))
		else:
			tqdm.write("Polarization data of station {} will not be saved as less than two antennas have received a measurable signal.".format(sname))

		#stokes_plot.showPlots(legend=True, sname=sname) #uncomment for plots
	pbar1.close()
	tqdm.write("Done.")

pbar.close()