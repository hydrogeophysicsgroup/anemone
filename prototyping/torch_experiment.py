# import warnings
# warnings.filterwarnings('error')

global torch
import torch

import numpy as np
import matplotlib.pyplot as plt
from anemone.experiment import Loop, Receiver
from anemone.transforms import Hankel, plot_loglog
from anemone.physics import refl_transverse_electric as rte, freespace_u0 as u0

two = torch.tensor([2], dtype=torch.complex64)

lp = Loop.square(50)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)


def integrand(wave_length):
    sigma = torch.tensor([10, 1., 10], dtype=torch.complex64)[:,None,None]
    thick = torch.tensor([25, 50],     dtype=torch.complex64)[:,None,None]
    omega = torch.tensor([100.],       dtype=torch.complex64)[None,:,None]*two*torch.pi
    lmb = wave_length[None, None, :]
    one = torch.ones(1, dtype=torch.complex64)
    intg = (one+rte(sigma, thick, omega, lmb))*lmb**2 / u0(omega, lmb)
    return intg

hkl = Hankel(integrand, 1)


r = torch.tensor([1.], dtype=torch.complex64)

r,hr = hkl.range(rmin, rmax)
r=torch.real(r).numpy()
np_hr = np.array(hr)

fig,axs = plt.subplots()
plt.plot(r, np_hr.real)
plt.plot(r, np.abs(np_hr.imag), '--')
plt.xscale('log')
plt.yscale('log')
plt.show()

