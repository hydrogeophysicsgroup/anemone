# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:16:44 2025

@author: au516205
"""

from timeit import default_timer as tictoc
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.forwards import ForwardCentralCircle
import numpy as np
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    times = np.geomspace(2.19e-6, 909e-6, 30)
    radius = 5.
    resistivities = torch.ones(2)
    thicknesses = torch.tensor([20.])
    resistivities = resistivities[:, None]*np.logspace(0, 2, 20)[None, :]
    from time import time as tictoc

else:
    radius = 5.


a = tictoc()

times = torch.tensor(times, dtype=torch.double)
resist0 = np.array(resistivities)  # resistivity in ohm.m
thick0 = np.array(thicknesses)

resist = torch.tensor(resist0,dtype=torch.double)  # resistivity in ohm.m
thick = torch.tensor(thick0, dtype=torch.double)

fwr = ForwardCentralCircle(radius, times)

b = tictoc()
dbdt = fwr(resist, thick)
c = tictoc()

int_time = c-b
imp_time = c-a

time_calc_int = int_time
time_calc_imp = imp_time


dpred = dbdt.detach().numpy()

if __name__ == "__main__":
    plt.figure()
    plt.loglog(times, np.abs(dbdt).T)
    plt.xlabel('Time (s)')
    plt.ylabel('|dBz/dt| (T/s)')
    plt.title('Central Circular Loop Receiver')
    plt.legend()
    plt.show()
    print(f'Integral time: {time_calc_int:.3f} s, Impulse time: {time_calc_imp:.3f} s')