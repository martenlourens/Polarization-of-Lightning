#!/usr/bin/env python3

import pickle
import tkinter as tk

from LoLIM import utilities
from LoLIM.utilities import processed_data_dir

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

with open(processed_data_dir(timeID)+"/polarization_data/Lightning Phenomena/Positive Leader/pulseIDs_PL3.pkl", 'rb') as f:
	data = pickle.load(f)

dump_data = pickle.dumps(data).hex()
del data

ui = tk.Tk()
ui.withdraw()
ui.clipboard_append(dump_data)
ui.mainloop()