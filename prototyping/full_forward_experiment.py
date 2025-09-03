import torch

if __name__ == "__main__": 
    print(f'default number of threads {torch.get_num_threads()}')
    print(f'default number of interop threads {torch.get_num_interop_threads()}')
# torch.set_num_interop_threads(16)

from anemone.experiment import Loop, Receiver
from anemone.transforms import Hankel, Fourier
from anemone.physics import refl_transverse_electric as rte, freespace_u0 as u0
import matplotlib.pyplot as plt
import numpy as np


# first lets compute Hz(R)
lp = Loop.square(50)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

sigma = torch.tensor([100,200], dtype=torch.complex128)[:,None,None]
thick = torch.tensor([50], dtype=torch.complex128)[:,None,None]


# @torch.jit.script
def frq_integrand(omega):
    f = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        intg = (1.+rte(sigma, thick, f, lmb))*lmb**2 / u0(f, lmb)
        return torch.squeeze(intg)
    hkl = Hankel(integrand, 1)
    r,hr = hkl.range(rmin, rmax)
    s = 0j-1*lp.geomz(rx, r)
    res = hr@s.T
    return torch.squeeze(res)*omega

from time import perf_counter as timer

for i in range(5):
    num_threads = 2**i
    torch.set_num_threads(num_threads)

    f_t = Fourier(frq_integrand)
    t,dbdt = f_t.range(1e-5, 1e-1)

    bt = timer()
    f_t = Fourier(frq_integrand)
    t,dbdt = f_t.range(1e-5, 1e-1)
    delta_t = timer() - bt
    if __name__ == "__main__": 
        print(f'{num_threads} threads, dipole time: {delta_t} s')
