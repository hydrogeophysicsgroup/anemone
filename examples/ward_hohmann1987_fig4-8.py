import matplotlib.pyplot as plt
import numpy as np

from anemone.transforms import BaseHankel, BaseFourier, CubicSpline
from anemone.physics import refl_transverse_electric as rte, freespace_u0 as u0, mu0, hzdt_halfspace_center_circle_loop, calc_wavenumber
from anemone.experiment import Loop, Receiver
import torch
twopi = torch.pi*2
fourpi = 2*twopi

class Forward:
    def __init__(self, source, receiver, times, radius=1.):
        self.source = source
        self.receiver = receiver
        self.times = torch.tensor(times)

        self.rmin = self.source.min_horizontal_distance(self.receiver)
        self.rmax = self.source.max_horizontal_distance(self.receiver)
        # self.sofr = torch.tensor([0., 0., 2*np.pi*a], dtype=torch.complex128)
        self.dz = torch.tensor(self.receiver.z+self.source.z[0])
        self.a = torch.tensor(radius)

    def forward(self):
        def frq_integrand(omega):
            omega = omega[None,:,None]
            def integrand(wave_length):
                lmb = wave_length[None, None, :]
                tmp_u0 = u0(omega, lmb)
                intg = torch.squeeze(torch.exp(-tmp_u0*self.dz)*rte(self.sigma, self.thick, omega, lmb)*lmb**2 / tmp_u0)
                return intg
            
            hkl1 = BaseHankel(integrand, 1, smp_per_decade=30)
            r, hr = hkl1.range(self.rmin, self.rmax)
            hr *= r
            cbhr = CubicSpline(r.real, hr.T)
            res = cbhr(self.a)
            return res
        f_t=BaseFourier(frq_integrand, smp_per_decade=30, log10min=-10, log10max=10)
        t,dbdt = f_t.range(self.times[0], self.times[-1])
        # cbs = CubicSpline(t, dbdt, transform=torch.log)
        return t, dbdt/fourpi

    def __call__(self, sigma, thickness):
        self.sigma = torch.tensor(sigma,     dtype=torch.complex128)[...,None,None]
        self.thick = torch.tensor(thickness, dtype=torch.complex128)[...,None,None]
        return self.forward()
    
times = torch.logspace(-6, -1, 51)
    
lp = Loop.circle(50)
rx=Receiver()

rho = 100
sigma = 1./torch.tensor([rho, rho])
thick = torch.tensor([40])

fwr = Forward(lp, rx, times, radius=50)
t,dhdt = fwr(sigma, thick)

dhdt_analytic = hzdt_halfspace_center_circle_loop(t, sigma[0], 50)


fig,ax = plt.subplots()
ax.plot(t.numpy(), dhdt_analytic.numpy(), 'gray', label='analytic')
ax.plot(t.numpy(), dhdt.real.numpy(), '--', label='numerical')

ax.legend()
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_aspect('equal')
ax.set_ylim(1e-7, 1e4)
plt.show()