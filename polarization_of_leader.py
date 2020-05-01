#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import pickle
from matplotlib.pyplot import figure, show

import os

from stokesIO import read_polarization_data

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir
from LoLIM.IO.raw_tbb_IO import read_antenna_pol_flips, read_bad_antennas, read_antenna_delays, read_station_delays, filePaths_by_stationName, MultiFile_Dal1
from LoLIM.main_plotter import gen_olaf_cmap

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


#THIS IS WHERE THE MAIN CODE STARTS

station = "CS302"

antenna_locations = TBB_data[station].get_LOFAR_centered_positions()[::2]

avg_station_loc = np.average(antenna_locations, axis=0)

with open(processed_data_folder+"/polarization_data/pulseIDs.pkl", 'rb') as f:
	pulseIDs = pickle.load(f)

rd = read_polarization_data(timeID)

pulses_PE = {}

with open(processed_data_folder+"/polarization_data/source_info.json", 'r') as f:
	source_info = json.load(f)

for filename in os.listdir(processed_data_folder+"/polarization_data/Polarization_Ellipse"):
	pulseID = os.path.splitext(filename)[0]
	df = rd.read_PE(pulseID)
	if df.empty:
		continue
	if int(pulseID) in pulseIDs:
		pulses_PE[pulseID] = df.loc[station].values

#function that returns datapoints forming an ellipse (patches.Ellipse does not work properly for our purposes)
def Ellipse(pos,width,height,angle):
	t = np.linspace(0,2*np.pi,100)
	x = np.array([width/2*np.cos(t),height/2*np.sin(t)])

	#rotation matrix anti-clockwise
	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])

	x = np.matmul(R,x)
	x = x + np.array([np.full_like(x[0],pos[0]),np.full_like(x[1],pos[1])])
	return x


fig = figure(figsize=(10,10))
#['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
frame = fig.add_subplot(111, aspect='equal')

#colormap
cmap = gen_olaf_cmap()
T = np.array([])
for pulseID in pulses_PE.keys():
	T = np.append(T,source_info[pulseID]['XYZT'][3])
vmin = np.min(T)
vmax = np.max(T)
T = 1/(vmax-vmin)*(T-vmin) #normalize T

for i, pulseID in enumerate(pulses_PE.keys()):
	loc = source_info[pulseID]['XYZT'][:3] - avg_station_loc
	Az = np.arctan(loc[1]/loc[0])
	Z = np.arctan(np.dot(loc[:2], loc[:2])**0.5/loc[2])
	Az = np.rad2deg(Az); Z = np.rad2deg(Z)
	if Az<0:
		Az += 90
	elif Az>0:
		Az -= 90

	scale = 10**15

	width = scale*2*pulses_PE[pulseID][0]*pulses_PE[pulseID][1]*np.cos(pulses_PE[pulseID][3])
	height = scale*2*pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])
	angle = pulses_PE[pulseID][2]

	x, y = Ellipse((Az,Z),width,height,angle)

	frame.plot(x, y, color=cmap(T[i]))
	frame.scatter(Az, Z, s=3, marker='o', color=cmap(T[i]))

frame.set_xlabel(r"$\phi\ [^\circ]$", fontsize=16)
frame.set_ylabel(r"$\theta_z\ [^\circ]$", fontsize=16)
frame.grid()

show()