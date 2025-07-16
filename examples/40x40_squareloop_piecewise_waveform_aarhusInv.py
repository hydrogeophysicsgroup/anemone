from anemone.experiment import Loop, Receiver, Waveform
from anemone.forwards import Forward
import matplotlib.pyplot as plt
import numpy as np
import torch


ainv_dbdt = np.array([3.853627e-04, 2.309149e-04, 1.255715e-04, 7.168451e-05,
       4.196206e-05, 2.297534e-05, 1.302996e-05, 7.591422e-06,
       4.179841e-06, 2.394241e-06, 1.452859e-06, 8.455657e-07,
       4.466630e-07, 2.468966e-07, 1.423754e-07, 7.665493e-08,
       4.189503e-08, 2.367454e-08, 1.263738e-08, 6.809730e-09])

times = np.array([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
       3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
       1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
       5.575e-04, 7.035e-04])

t = np.array([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
wv = np.array([0, 1, 1, 0])

waveform = Waveform(t, wv)


lp = Loop.square(40)
rx = Receiver()


rho = torch.ones(2)*40.
thick = torch.tensor([10])

fwr = Forward(lp, rx, times, waveform=waveform)
dbdt = fwr(rho, thick);

fig,ax = plt.subplots(1,2, sharex=True)
ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
ax[0].legend()
ax[0].set_aspect('equal')

ax[1].semilogx(times, 100*np.abs(dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
ax[1].set_ylabel('relative difference [%]')

fig.suptitle('Comparison with AarhusInv with Piecewise Waveform')

plt.show()