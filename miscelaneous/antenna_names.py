#!/usr/bin/env python3

import numpy as np
from LoLIM.IO.raw_tbb_IO import MultiFile_Dal1, filePaths_by_stationName

## these lines are anachronistic and should be fixed at some point
from LoLIM import utilities
#utilities.default_raw_data_loc = "/exp_app2/appexp1/lightning_data"
utilities.default_raw_data_loc = "/home/student4/Marten/KAP_data_link/lightning_data"
#utilities.default_processed_data_loc = "/home/brian/processed_files"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

timeID = "D20190424T210306.154Z"
raw_fpaths = filePaths_by_stationName(timeID)

for station in ['RS508']:
	TBB_data = MultiFile_Dal1( raw_fpaths[station] )
	print(TBB_data.get_antenna_names())