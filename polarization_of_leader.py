#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
from matplotlib.pyplot import figure, show

import os

from stokesIO import read_polarization_data

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

station = "CS302"

rd = read_polarization_data(timeID)

pulses_PE = {}

with open(processed_data_dir(timeID)+"/polarization_data/source_info.json", 'r') as f:
	source_info = json.load(f)

for filename in os.listdir(processed_data_dir(timeID)+"/polarization_data/Polarization_Ellipse"):
	pulseID = os.path.splitext(filename)[0]
	df = rd.read_PE(pulseID)
	if df.empty:
		continue
	
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
frame = fig.add_subplot(1,1,1) #aspect = 'equal'

for pulseID in pulses_PE.keys():
	loc = source_info[pulseID]['XYZT']
	Az = np.arctan(loc[1]/loc[0])
	Z = np.arctan(np.dot(loc[:2], loc[:2])/loc[2])
	Az = np.rad2deg(Az); Z = np.rad2deg(Z)

	s = 10**12

	width = s*2*pulses_PE[pulseID][0]*np.cos(pulses_PE[pulseID][3])
	height = s*2*pulses_PE[pulseID][0]*np.sin(pulses_PE[pulseID][3])
	angle = pulses_PE[pulseID][2]

	x, y = Ellipse((Az,Z),width,height,angle)
	
	frame.plot(x,y,color='r')
	frame.scatter(Az,Z,s=3,marker='.',color='k')

frame.set_xlabel(r"$\phi\ [^\circ]$", fontsize=16)
frame.set_ylabel(r"$\theta_z\ [^\circ]$", fontsize=16)
frame.grid()

show()