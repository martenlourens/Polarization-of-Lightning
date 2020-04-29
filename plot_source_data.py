#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import sys

from LoLIM.IO.raw_tbb_IO import filePaths_by_stationName, MultiFile_Dal1, read_station_delays, read_antenna_pol_flips, read_bad_antennas, read_antenna_delays
from LoLIM.utilities import processed_data_dir, natural_sort, RTD
from LoLIM.findRFI import window_and_filter
from LoLIM.getTrace_fromLoc import getTrace_fromLoc

from LoLIM import utilities
utilities.default_raw_data_loc = "/home/student4/Marten/KAP_data_link/lightning_data"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

if __name__ == "__main__":
    
    timeID = "D20190424T210306.154Z"
    
    station_delay_file = "station_delays.txt"
    polarization_flips = "polarization_flips.txt"
    bad_antennas = "bad_antennas.txt"
    additional_antenna_delays = "ant_delays.txt"

    source_XYZT = np.array([-743.41, -3174.29, 4377.67, 1.1350323478]); srcName = 1477200
    #source_XYZT = np.array([247.86, -3555.45, 4779.61, 1.1051278583]); srcName = 1409503
    #source_XYZT = np.array([-1040.17, -2877.24, 4405.51, 1.1518796563]); srcName = 1515137
    #source_XYZT = np.array([-1594.76, -3307.02, 4339.72, 1.1386715757]); srcName = 1485269
    #source_XYZT = np.array([-1804.64, -2937.45, 4062.84, 1.1452615564]); srcName = 1500111

    #source_XYZT = np.array([246.96, -3558.95, 4785.46, 1.1107716958]); srcName = 1422213
    #source_XYZT = np.array([247.14, -3558.79, 4785.24, 1.1107717917]); srcName = 1422214
    
    pulse_length = 1000
    
    processed_data_folder = processed_data_dir(timeID)
    
    polarization_flips = read_antenna_pol_flips( processed_data_folder + '/' + polarization_flips )
    bad_antennas = read_bad_antennas( processed_data_folder + '/' + bad_antennas )
    additional_antenna_delays = read_antenna_delays(  processed_data_folder + '/' + additional_antenna_delays )
    station_timing_offsets = read_station_delays( processed_data_folder+'/'+station_delay_file )
    
    raw_fpaths = filePaths_by_stationName(timeID)
    raw_data_files = {sname:MultiFile_Dal1(fpaths, force_metadata_ant_pos=True, polarization_flips=polarization_flips, bad_antennas=bad_antennas, additional_ant_delays=additional_antenna_delays) \
                      for sname,fpaths in raw_fpaths.items() if sname in station_timing_offsets}
    
    sorted_station_names = natural_sort( raw_data_files.keys() )
    
    data_filters = {sname:window_and_filter(timeID=timeID,sname=sname) for sname in station_timing_offsets}
    
    
    trace_locator = getTrace_fromLoc( raw_data_files, data_filters, station_timing_offsets )


    H = 0
    for sname in sorted_station_names:
        input_file = raw_data_files[ sname ]
        antenna_names = input_file.get_antenna_names()[::2] ## even antennas
        antenna_locations = input_file.get_LOFAR_centered_positions()[::2]
        station_location_difference = source_XYZT[:3] - np.average(antenna_locations, axis=0)
        zenith = np.arctan2( station_location_difference[2],  np.linalg.norm(station_location_difference[:2]) )

        sys.stdout = open(processed_data_folder + "/polarization_data/Pulse_Plots/" + "{}.log".format(srcName),'a')
        print(sname, 90 - zenith*RTD)
        sys.stdout.close()
        
        max_H = 0
        plt.annotate(sname, (0,H))
        for an in antenna_names:
            start_sample, total_time_offset, arrival_time, extracted_trace = trace_locator.get_trace_fromLoc(source_XYZT, an, pulse_length, do_remove_RFI=True, do_remove_saturation=True)
    
            HE = np.abs(extracted_trace)
    
            plt.plot(HE+H)
            
            max_H = max( np.max(HE), max_H )

            plt.plot([pulse_length/2,pulse_length/2],[H,H+max_H],linestyle='dashed',linewidth=1,color='grey')
            
        H += max_H
    plt.savefig(processed_data_folder + "/polarization_data/Pulse_Plots/" + "{}.svg".format(srcName),dpi=70)