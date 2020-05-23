#!/usr/bin/env python3

from os import listdir
from os.path import splitext

from LoLIM.iterativeMapper.iterative_mapper import read_header
from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

import pickle
import json

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

processed_data_folder = processed_data_dir(timeID)

#user defined variables
pkl_data_folder = "polarization_data/Lightning Phenomena/Positive Leader"

pkl_filename = [f for f in listdir(processed_data_folder + '/' + pkl_data_folder) if f.endswith(".pkl")] 

header = read_header('iterMapper_50_CS002',  timeID)
data = header.load_data_as_sources(maxRMS=3E-9) #DON'T FORGET TO CHECK THIS!!!

for file in pkl_filename:

	with open(processed_data_folder + '/' + pkl_data_folder + '/' + file, 'rb') as f:
		pulseIDs = pickle.load(f)


	sourceDict = {}

	for info in data:
		for ID in pulseIDs:
			if info.uniqueID == ID:
				sourceDict["{}".format(info.uniqueID)] = {'XYZT':info.XYZT.tolist(), 'covXYZ':info.cov_matrix.tolist(), 'rmsT':info.RMS}
				break
	
	if splitext(file)[0][:8] == 'pulseIDs':
		outfile = "source_info" + splitext(file)[0][8:]
	else:
		outfile = "source_info" + splitext(file)[0]

	with open(processed_data_folder + '/' + pkl_data_folder + '/' + outfile + ".json", 'w') as f:
		json.dump(sourceDict,f)
