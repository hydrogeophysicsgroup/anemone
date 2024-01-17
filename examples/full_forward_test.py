import numpy as np
import matplotlib.pyplot as plt
from em1d.experiment import Loop, Receiver, getRz
from em1d.transforms import *
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0
from numpy.linalg import norm

# first lets compute Hz(R)
lp = Loop.square(10)
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

sigma = np.array([0.1, 1, 0.1])[:,None,None]
thick = np.array([40, 40])[:,None,None]



def frq_integrand(omega):
    f = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        tmp_u0 = u0(f, lmb)
        intg = (np.exp(-tmp_u0*40)*rte(sigma, thick, f, lmb))*lmb**2 / tmp_u0
        return np.squeeze(intg/(1j*f))
    hkl = OptimizedHankel(integrand, 1, 5, 7)
    r,hr = hkl.range(5, 7)

    s = np.zeros_like(r)
    s[np.searchsorted(r, 6)] = 2*np.pi*6
    res = hr.T.dot(s)#/f
    return -np.squeeze(res)


# from time import perf_counter as timer
# bt = timer()

def prof_script():
    f_t = Fourier(frq_integrand)#, 1e-5, 1e-2)
    t,dbdt = f_t.range(1e-5, 1e-2)
    return t, dbdt

import cProfile
cProfile.run('prof_script()', 'restats')
import pstats
from pstats import SortKey
p = pstats.Stats('restats')
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats()
# print(len(t))
# print(f'dipole time: {timer()-bt} s')

from time import perf_counter as timer
bt = timer()
t,dbdt=prof_script()
print(f'dipole optimized time: {timer()-bt} s')



hkl = defHankel1
f_t = defFourier

def frq_integrand(omega):
    f = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        tmp_u0 = u0(f, lmb)
        intg = (np.exp(-tmp_u0*40)*rte(sigma, thick, f, lmb))*lmb**2 / tmp_u0
        return np.squeeze(intg/(1j*f))
    hkl.func = integrand
    hkl.F = integrand(hkl.lamda[hkl.subset])
    r,hr = hkl.range(5, 7)

    #hr /= r[:,None]
    s = np.zeros_like(r)
    s[np.searchsorted(r, 6)] = 2*np.pi*6
    res = hr.T.dot(s)#/f
    return -np.squeeze(res)


from time import perf_counter as timer
bt = timer()
f_t.F = frq_integrand(f_t.lamda[f_t.subset])*np.sqrt(f_t.lamda[f_t.subset])
t1,dbdt1 = f_t.range(1e-5, 1e-2)
print(len(t))
print(f'dipole default time: {timer()-bt} s')



fig,axs = plt.subplots()
plot_loglog(t, dbdt.real, ax=axs)
plot_loglog(t1, dbdt1.real, ax=axs, color='r')

axs.set_aspect('equal')
plt.show()