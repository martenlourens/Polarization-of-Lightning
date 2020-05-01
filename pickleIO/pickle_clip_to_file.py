#!/usr/bin/env python3

import tkinter as tk
import pickle

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"
processed_data_folder = processed_data_dir(timeID)

ui = tk.Tk()
ui.withdraw()
clip = bytes.fromhex(ui.clipboard_get())
ui.destroy()

data = pickle.loads(clip)

with open(processed_data_folder + "/polarization_data/pulseIDs.pkl", 'wb') as f:
	pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)