from em1d.experiment import *
import numpy as np

t0 = np.geomspace(1e-6, 1e-2, 31)
sq = Loop.square(40)
rx = Receiver()
t = np.array([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
wv = np.array([0, 1, 1, 0])
waveform = Waveform(t, wv)
recfilter = FilterChain([
    ButterworthFilter(2.4e5, 1),
    ButterworthFilter(6.2e5, 1),
    ButterworthFilter(4.5e5, 2)
])

fwr0 = Forward(sq, rx, torch.from_numpy(t0), waveform=waveform, filterfunc=recfilter)
fwr1 = Forward(sq, rx, torch.from_numpy(t0), waveform=waveform, filterfunc=recfilter, tolerance=1e-4)

rho   = 10**(np.random.randn(6)+1)
thick = np.geomspace(2, 50, len(rho)-1)
sigma = torch.from_numpy(1./rho)
thick = torch.from_numpy(thick)

dbdt = fwr0(sigma, thick)
dbdt1 = fwr1(sigma, thick)

import matplotlib.pyplot as plt
plt.figure()
plt.loglog(t0, dbdt.numpy(), label="dbdt")
plt.loglog(t0, dbdt1.numpy(), '--', label="dbdt1")
plt.show()

assert torch.allclose(dbdt, dbdt1, atol=1e-4), "Forward transform is not stable."