# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:16:44 2025

@author: au516205
"""

from timeit import default_timer as tictoc
from anemone.system import Loop, Receiver
from anemone.forwards import Forward
import numpy as np
import torch
import matplotlib.pyplot as plt

#torch.set_num_threads(1)

a = tictoc()

times = torch.tensor(times, dtype=torch.double)
resist0 = np.array(resistivities)  # resistivity in ohm.m
thick0 = np.array(thicknesses)

resist = torch.tensor(resist0,dtype=torch.double)  # resistivity in ohm.m
thick = torch.tensor(thick0, dtype=torch.double)

radius = 1.#/np.sqrt(np.pi)
tx = Loop.circle(radius)
rx = Receiver()
fwr = Forward(tx, rx, times, tolerance=1e-6)

b = tictoc()
dbdt = fwr(resist, thick).real
c = tictoc()

int_time = c-b
imp_time = c-a

time_calc_int = int_time
time_calc_imp = imp_time

dpred = dbdt.numpy()/np.pi

if __name__ == "__main__":
    plt.figure()
    plt.loglog(times, np.abs(dbdt).T)
    plt.xlabel('Time (s)')
    plt.ylabel('|dBz/dt| (T/s)')
    plt.title('Central Circular Loop Receiver')
    plt.legend()
    plt.show()
    print(f'Integral time: {time_calc_int:.3f} s, Impulse time: {time_calc_imp:.3f} s')