from anemone.experiment import Loop, Receiver, Forward, Waveform, ButterworthFilter, FilterChain
import matplotlib.pyplot as plt
import numpy as np
import torch


ainv_dbdt = np.array([ 6.145711e-04, 3.319648e-04, 1.650304e-04, 8.868765e-05, 4.967526e-05, 2.618016e-05, 1.444254e-05, 8.242962e-06, 4.457639e-06, 2.520354e-06, 1.515021e-06, 8.744991e-07, 4.584869e-07, 2.520399e-07, 1.447630e-07, 7.766509e-08, 4.233407e-08, 2.387553e-08, 1.272288e-08, 6.846980e-09])
times = np.array([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
       3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
       1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
       5.575e-04, 7.035e-04])

t = np.array([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
wv = np.array([0, 1, 1, 0])

waveform = Waveform(t, wv)

recfilter = FilterChain([
    ButterworthFilter(2.4e5, 1),
    ButterworthFilter(6.2e5, 1),
    ButterworthFilter(4.5e5, 2)
])


lp = Loop.square(40)
rx = Receiver()

rho = 40
sigma = 1/torch.tensor([rho,rho])
thick = torch.tensor([10])

fwr = Forward(lp, rx, torch.from_numpy(times), waveform=waveform, filterfunc=recfilter)
dbdt = fwr(sigma, thick);

fig,ax = plt.subplots(1,2, sharex=True)
ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
ax[0].legend()
ax[0].set_aspect('equal')

ax[1].semilogx(times, 100*np.abs(dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
ax[1].set_ylabel('relative difference [%]')

fig.suptitle('Comparison with AarhusInv with Piecewise Waveform')

plt.show()