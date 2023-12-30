import numpy as np
import matplotlib.pyplot as plt
from em1d.experiment import Loop, Receiver, getRz
from em1d.transforms import Hankel, Fourier, plot_loglog
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0
from numpy.linalg import norm

# first lets compute Hz(R)
lp = Loop.square(50)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

sigma = np.array([100,200])[:,None,None]
thick = np.array([50])[:,None,None]



def frq_integrand(omega):
    print(len(omega))
    f = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        intg = (1.+rte(sigma, thick, f, lmb))*lmb**2 / u0(f, lmb)
        return np.squeeze(intg)
    hkl = Hankel(integrand, 1)
    r,hr = hkl.range(rmin, rmax)
    s = -1*lp.s_of_r(rx, r)
    res = hr.T.dot(s)#/f
    return np.squeeze(res)*omega


from time import perf_counter as timer
bt = timer()
f_t = Fourier(frq_integrand)
t,dbdt = f_t.range(1e-5, 1e-1)
print(f'dipole time: {timer()-bt} s')

fig,axs = plt.subplots()
plot_loglog(t, dbdt.real, ax=axs)
axs.loglog(t, t**(-5/2), 'k--')
axs.set_aspect('equal')
plt.show()