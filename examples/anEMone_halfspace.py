# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:16:44 2025

@author: au516205
"""

from timeit import default_timer as tictoc
from anemone.experiment import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.forwards import Forward
import numpy as np
import torch

import matplotlib.pyplot as plt

a = tictoc()

times = torch.tensor(times, dtype=torch.double)
waveform_times = torch.tensor(waveform_times, dtype=torch.double)
waveform_current = torch.tensor(waveform_current, dtype=torch.double)
receiver = torch.tensor(receiver, dtype=torch.double)

#usestep=1
if usestep==1:
  wv = None
else:
  wv = Waveform(waveform_times, waveform_current)
    
receiverfilters = None

rec_location = receiver
rx = Receiver(x=rec_location[0], y=rec_location[1], z=rec_location[2])# height is positive above ground.
#print(rec_location)
#print(usestep)
#Loop.circle for offset receiver
#ForwardCentralCircle for concentric receiver

resist0 = np.array(resistivities)  # resistivity in ohm.m
thick0 = np.array(thicknesses)

resist = torch.tensor(resist0,dtype=torch.double)  # resistivity in ohm.m
thick = torch.tensor(thick0, dtype=torch.double)

tx = Loop.square(torch.tensor(1))
#tx = Loop.circle(torch.tensor(1))

Nlay = 2

t_width = None
fwr = Forward(tx, rx, times, waveform=wv, filterfunc=receiverfilters, t_gatewidth=t_width)

b = tictoc()
dbdt = fwr(resist, thick)
c = tictoc()

int_time = c-b
imp_time = c-a

time_calc_int = int_time
time_calc_imp = imp_time


dpred = dbdt.detach().numpy()
