from em1d.experiment import Loop, Receiver, Forward
import matplotlib.pyplot as plt
import numpy as np
import torch
from em1d import constants as c


ainv_dbdt=np.array([2.749861e-04, 1.753167e-04, 1.011688e-04, 6.031561e-05,
       3.650625e-05, 2.059544e-05, 1.194428e-05, 7.079678e-06,
       3.957627e-06, 2.293402e-06, 1.403765e-06, 8.236817e-07,
       4.389777e-07, 2.446409e-07, 1.422868e-07, 7.752075e-08,
       4.302391e-08, 2.479088e-08, 1.362022e-08, 7.621867e-09])

times = np.array([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
       3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
       1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
       5.575e-04, 7.035e-04])

lp = Loop.square(40)
rx = Receiver()

rho = 40
sigma = 1/torch.tensor([rho,rho])
thick = torch.tensor([10])

fwr = Forward(lp, rx, times)
dbdt = fwr(sigma, thick);

fig,ax = plt.subplots(1,2, sharex=True)
ax[0].loglog(fwr.times.numpy(), c.mu0*dbdt.real, label='mforward')
ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
ax[0].legend()
ax[0].set_aspect('equal')

ax[1].semilogx(times, 100*np.abs(c.mu0*dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
ax[1].set_ylabel('relative difference [%]')

plt.show()