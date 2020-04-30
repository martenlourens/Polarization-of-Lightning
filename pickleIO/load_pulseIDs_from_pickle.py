#!/usr/bin/env python3

import pickle

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

with open(processed_data_dir(timeID)+"/polarization_data/pulseIDs.pkl", 'rb') as f:
	data = pickle.load(f)

print(data)