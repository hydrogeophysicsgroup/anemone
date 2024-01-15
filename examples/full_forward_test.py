import torch
from em1d.experiment import Loop, Receiver
from em1d.transforms import Hankel, Fourier, plot_loglog
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0
import matplotlib.pyplot as plt
import numpy as np


# first lets compute Hz(R)
lp = Loop.square(50)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

sigma = torch.tensor([100,200], dtype=torch.complex128)[:,None,None]
thick = torch.tensor([50], dtype=torch.complex128)[:,None,None]



def frq_integrand(omega):
    print(len(omega))
    f = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        intg = (1.+rte(sigma, thick, f, lmb))*lmb**2 / u0(f, lmb)
        return torch.squeeze(intg)
    hkl = Hankel(integrand, 1)
    r,hr = hkl.range(rmin, rmax)
    s = -1*torch.tensor(lp.s_of_r(rx, r.numpy()), dtype=torch.complex128)
    res = hr.T@s
    return torch.squeeze(res)*omega


from time import perf_counter as timer
bt = timer()
f_t = Fourier(frq_integrand)
t,dbdt = f_t.range(1e-5, 1e-1)
print(f'dipole time: {timer()-bt} s')