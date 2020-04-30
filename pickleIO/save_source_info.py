#!/usr/bin/env python3

from LoLIM.iterativeMapper.iterative_mapper import read_header
from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

#import numpy as np
import pickle
import json

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

header = read_header( 'iterMapper_50_CS002',  timeID)
data = header.load_data_as_sources(maxRMS=2.5E-9)

with open(processed_data_dir(timeID)+"/polarization_data/pulseIDs.pkl", 'rb') as f:
	pulseIDs = pickle.load(f)


sourceDict = {}

for info in data:
	for ID in pulseIDs:
		if info.uniqueID == ID:
			sourceDict["{}".format(info.uniqueID)] = {'XYZT':info.XYZT.tolist(), 'covXYZ':info.cov_matrix.tolist(), 'rmsT':info.RMS}
			break

with open(processed_data_dir(timeID)+"/polarization_data/source_info.json", 'w') as f:
	json.dump(sourceDict,f)
