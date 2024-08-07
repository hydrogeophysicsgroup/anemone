import torch
import matplotlib.pyplot as plt
from em1d.experiment import Loop, Receiver, getRz
from em1d.transforms import Hankel
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0
from numpy.linalg import norm

# first lets compute Hz(R)
lp = Loop.square(50)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

def integrand(wave_length):
    sigma = torch.tensor([10, 1., 10])[:,None,None]
    thick = torch.tensor([25, 50])[:,None,None]
    omega = torch.logspace(-1, 5, 120)[None,:,None]*2*torch.pi
    #omega = torch.tensor([100])*2*torch.pi
    lmb = wave_length[None, None, :]
    intg = (1.+rte(sigma, thick, omega, lmb))*lmb**2 / u0(omega, lmb)
    return intg


def hr_interp(r):
    return int_hr(np.log10(r))



def integrate_around_loop(lp, rx, hr):
    lp_xyz = lp.xyz - rx.xyz[None, ...]

    HR = 0+0j
    n_dipoles = 0
    for i in range(len(lp)-1):
        Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
        p1 = torch.einsum('...i,...i->...', Rz, lp_xyz[i])
        p2 = torch.einsum('...i,...i->...', Rz, lp_xyz[i+1])

        p12 = p2-p1
        tdist = norm(p12, axis=-1)
        p12 /= tdist
        cp = p1
        cdist = 0

        while cdist < tdist:
            cR = norm(cp)
            ds = cR/15
            if cdist+ds > tdist:
                ds = tdist-cdist
                cdist += cR/15
            else:
                cdist += ds
            HR += ds*p1[1]*hr(cR)/cR
            cp += ds*p12
            n_dipoles +=1
    return HR

from time import perf_counter as timer
from scipy.interpolate import interp1d

bt = timer()
hkl = Hankel(integrand, 1)
r, hr = hkl.range(rmin, rmax)
# int_hr = interp1d(r, hr.T)
print(f'dipole time: {timer()-bt} s')

# hz_loop = integrate_around_loop(lp, Receiver(), int_hr)
# print(f'loop int time: {timer()-bt} s')
