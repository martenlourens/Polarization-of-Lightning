#!/usr/bin/env python3

import sys
sys.path.append('../')

from LoLIM import utilities

import numpy as np
import pandas as pd

from stokesIO import read_polarization_data
from stokes import stokes_plotter

timeID = "D20190424T210306.154Z"
utilities.default_processed_data_loc = "/home/student4/Marten/processed_files"

rd = read_polarization_data(timeID)

pulseID = 1477200

data = rd.read_PE(pulseID)
data = data.values

stokes_plot = stokes_plotter(plot=['polarization_ellipse'])
stokes_plot.plot_polarization_ellipse(data, errors=True)
stokes_plot.showPlots()