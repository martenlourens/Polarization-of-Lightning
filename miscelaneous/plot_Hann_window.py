#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

from LoLIM.signal_processing import half_hann_window

window = half_hann_window(1000, half_percent=0.1)

plt.plot(window,'k')
plt.grid()
plt.show()