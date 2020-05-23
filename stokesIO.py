#!/usr/bin/env python3

from LoLIM.utilities import processed_data_dir

from os import mkdir
from os.path import isdir, isfile
import errno

import numpy as np
import pandas as pd
import json





class save_polarization_data:
	def __init__(self,timeID,pulseID,alt_loc=None):
		self.data_loc = processed_data_dir(timeID) + '/' + "polarization_data"

		if alt_loc is not None:
			self.data_loc = alt_loc

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
	def __init__(self,timeID,alt_loc=None):
		self.data_loc = processed_data_dir(timeID)

		if alt_loc is not None:
			self.data_loc = alt_loc
		else:
			if not isdir(self.data_loc + '/' + "polarization_data"):
				raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + '/' + "polarization_data"))
			self.data_loc += "/polarization_data"

	def read_S(self,pulseID):
		if not isdir(self.data_loc + "/Stokes_Vectors"):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + "/Stokes_Vectors"))

		if not isfile(self.data_loc + "/Stokes_Vectors/" + "{}.txt".format(pulseID)):
			print("No datafile for pulse {}.".format(pulseID))
			return

		file_loc = self.data_loc + "/Stokes_Vectors/" + "{}.txt".format(pulseID)

		df = pd.read_csv(file_loc, sep=' ', index_col=0, skiprows=1, names=["I","Q","U","V","Ierr","Qerr","Uerr","Verr"])
		
		return df

	def read_PE(self,pulseID):
		if not isdir(self.data_loc + "/Polarization_Ellipse"):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + "/Polarization_Ellipse"))

		if not isfile(self.data_loc + "/Polarization_Ellipse/" + "{}.txt".format(pulseID)):
			print("No datafile for pulse {}.".format(pulseID))
			return

		file_loc = self.data_loc + "/Polarization_Ellipse/" + "{}.txt".format(pulseID)

		df = pd.read_csv(file_loc, sep=' ', index_col=0, skiprows=1, names=["S_0", "δ", "τ", "ε", "σ_S_0", "σ_δ", "σ_τ", "σ_ε"])
		
		return df





class save_acceleration_vector:
	def __init__(self, timeID, fname=None, alt_loc=None):
		self.data_loc = processed_data_dir(timeID)

		if alt_loc is not None:
			self.data_loc = alt_loc
		else:
			if not isdir(self.data_loc + '/' + "polarization_data"):
				raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(self.data_loc + '/' + "polarization_data"))
			self.data_loc += "/polarization_data"

		if fname is not None:
			self.save_file = fname + ".json"
		else:
			self.save_file = "a_vectors.json"

		#create the file
		with open(self.data_loc + '/' + self.save_file, 'w') as f:
			json.dump(dict(), f, indent=4)

	def save_a_vector(self, pulseID, fitobj):
		"""
			'fitobj' should be a python object containing the fit results.
			In should contain at least the  following results:
				fitobj	.params:		best estimate (ndarray)
						.xerror:		standard deviation in the best estimate (ndarray)
						.stderr:		standard deviation assuming χν^2=1 (ndarray)
						.chi2_min:		minimized χ^2 value
						.dof:			degrees of freedom of the fit
						.rchi2_min:		reduced χ^2 (χν^2 = χ^2/ν)
						.covar:			covariance matrix corresponding to the best estimate (ndarray)
		"""
		save_dict = {}
		save_dict["{}".format(pulseID)] = {
											'params' : fitobj.params.tolist(),
											'xerror' : fitobj.xerror.tolist(),
											'stderr' : fitobj.stderr.tolist(),
											'chi2_min' : fitobj.chi2_min,
											'dof' : fitobj.dof,
											'rchi2_min' : fitobj.rchi2_min,
											'covar' : fitobj.covar.tolist()
										}

		with open(self.data_loc +  '/' + self.save_file, 'r') as f:
			data = json.load(f)
			data.update(save_dict)

		with open(self.data_loc +  '/' + self.save_file, 'w') as f:
			json.dump(data, f, indent=4)
			




def read_acceleration_vector(timeID, fname, alt_loc=None):
	data_loc = processed_data_dir(timeID)

	if alt_loc is not None:
		data_loc = alt_loc
	else:
		if not isdir(data_loc):
			raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(data_loc))
		data_loc += "/polarization_data"

	if fname is not None:
		save_file = fname + ".json"
	else:
		save_file = "a_vectors.json"

	if not isfile(data_loc + '/' + save_file):
		raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT),"{}".format(data_loc + '/' + save_file))

	with open(data_loc + '/' + save_file, 'r') as f:
		data = json.load(f)

	return data







if __name__ == "__main__":
	from LoLIM import utilities

	timeID = "D20190424T210306.154Z"
	utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

	rd = read_polarization_data(timeID)
	print(rd.read_S('1477200'))
	print(rd.read_PE('1477200'))