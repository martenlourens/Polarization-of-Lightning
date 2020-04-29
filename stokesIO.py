#!/usr/bin/env python3

from LoLIM.utilities import processed_data_dir

from os import mkdir
from os.path import isdir, isfile

import numpy as np
import pandas as pd





class save_polarization_data:
	def __init__(self,timeID,pulseID):
		self.data_loc = processed_data_dir(timeID) + '/' + "polarization_data"

		#if the directory doesn't exist create it!
		if not isdir(self.data_loc):
			mkdir(self.data_loc)

		if not isdir(self.data_loc + '/' + "Stokes_Vectors"):
			mkdir(self.data_loc + '/' + "Stokes_Vectors")

		if not isdir(self.data_loc + '/' + "Polarization_Ellipse"):
			mkdir(self.data_loc + '/' + "Polarization_Ellipse")

		self.filename_S = "{}.txt".format(pulseID) #filename for stokesvectors
		self.filename_PE = "{}.txt".format(pulseID) #filename for polarization ellipse data

		#create empty files with header for S and PE for a particular pulse
		with open(self.data_loc + "/Stokes_Vectors/" + self.filename_S, 'w') as f:
			f.write("# Station I Q U V Ierr Qerr Uerr Verr\n")

		with open(self.data_loc + "/Polarization_Ellipse/" + self.filename_PE, 'w') as f:
			f.write("# Station S_0 δ τ ε σ_S_0 σ_δ σ_τ σ_ε\n")

	def save_S(self,stationName,data,σ):
		data = [np.concatenate((data,σ))]

		with open(self.data_loc + "/Stokes_Vectors/" + self.filename_S, 'a') as f:
			f.write(stationName+' ')
			np.savetxt(f, data, delimiter=' ', newline='\n')

	def save_PE(self,stationName,data,σ):
		data = [np.concatenate((data,σ))]
		
		with open(self.data_loc + "/Polarization_Ellipse/" + self.filename_PE, 'a') as f:
			f.write(stationName+" ")
			np.savetxt(f, data, delimiter=' ', newline='\n')





class read_polarization_data:
	def __init__(self,timeID):
		self.data_loc = processed_data_dir(timeID)

		if not isdir(self.data_loc + '/' + "polarization_data"):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + '/' + "polarization_data"))

	def read_S(self,pulseID):
		if not isdir(self.data_loc + "/polarization_data/Stokes_Vectors"):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + "/polarization_data/Stokes_Vectors"))

		if not isfile(self.data_loc + "/polarization_data/Stokes_Vectors/" + "{}.txt".format(pulseID)):
			print("No datafile for pulse {}.".format(pulseID))
			return

		file_loc = self.data_loc + "/polarization_data/Stokes_Vectors/" + "{}.txt".format(pulseID)

		df = pd.read_csv(file_loc, sep=' ', index_col=0, skiprows=1, names=["I","Q","U","V","Ierr","Qerr","Uerr","Verr"])
		
		return df

	def read_PE(self,pulseID):
		if not isdir(self.data_loc + "/polarization_data/Polarization_Ellipse"):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + "/polarization_data/Polarization_Ellipse"))

		if not isfile(self.data_loc + "/polarization_data/Polarization_Ellipse/" + "{}.txt".format(pulseID)):
			print("No datafile for pulse {}.".format(pulseID))
			return

		file_loc = self.data_loc + "/polarization_data/Polarization_Ellipse/" + "{}.txt".format(pulseID)

		df = pd.read_csv(file_loc, sep=' ', index_col=0, skiprows=1, names=["S_0", "δ", "τ", "ε", "σ_S_0", "σ_δ", "σ_τ", "σ_ε"])
		
		return df






if __name__ == "__main__":
	from LoLIM import utilities

	timeID = "D20190424T210306.154Z"
	utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

	rd = read_polarization_data(timeID)
	print(rd.read_S('1477200'))
	print(rd.read_PE('1477200'))