from em1d.inversion import *

from em1d.experiment import Loop, Receiver, Waveform
from em1d.forwards import Forward, ForwardLaplace
tx = Loop.square(25)
rx = Receiver()
t = torch.logspace(-5, -3.5, 16, dtype=torch.double)



fwr = ForwardLaplace(tx, rx, t, waveform=Waveform.example(), tolerance=1e-5)

n_layers = 10
thick = torch.logspace(-1, 1, n_layers-1)

tot_list = []
dt_list  = []
n_models = []

from time import perf_counter as timer
for n_mod in np.logspace(0, 3, 10):

    z = np.arange(n_layers)[:,None]
    x = np.arange(int(n_mod))[None,:]

    rho = 300*np.ones_like(z+x)
    rho[(0.03*x-0.75*z+10)>0] = 10
    rho[(0.075*x-1.25*z+5)>0] = 40

    sigma = torch.from_numpy(1./rho)
    thick = torch.logspace(-1, 1, len(z)-1)
    m_true = SmoothModels(sigma, thick)

    bt = timer()
    data = fwr(*m_true.mod)
    et = timer()
    print(f"Forward time: {et-bt:.2f} s")
    print(f"Forward time per forward: {(et-bt)/int(n_mod):.2f} s")

    tot_list.append(et-bt)
    dt_list.append((et-bt)/int(n_mod))
    n_models.append(int(n_mod))

import matplotlib.pyplot as plt
import numpy as np

fig,axs = plt.subplots(1,2)
axs[0].loglog(n_models, tot_list, 'o-')
axs[1].loglog(n_models,  dt_list, 'o-')
for ax in axs:
    ax.set_xlabel("Number of models")
    ax.set_ylabel("Time [s]")
plt.show()
