from em1d.experiment import *
from em1d.transforms import invLaplace

class ForwardLaplace:
    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.srcz = torch.tensor([source.z[0]], dtype=torch.float64)
        self.rz = self.receiver.z
        self.pfield_dz = torch.abs(self.srcz-self.rz)
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)

        self.waveform = waveform
        if t_gatewidth is not None:
            if isinstance(t_gatewidth, float): t_gatewidth = torch.ones_like(self.times)*t_gatewidth
            elif not isinstance(t_gatewidth, torch.Tensor) and t_gatewidth is not None: t_gatewidth = torch.tensor(t_gatewidth, device=self.times.device)
            self.t_gatewidth = t_gatewidth
            self.t_st = self.times-t_gatewidth/2
            self.t_end= self.times+t_gatewidth/2
        else:
            self.t_gatewidth=None
        if self.t_gatewidth is not None:
            self.tmin = (self.times-t_gatewidth/2).min()/1.1
            self.tmax = (self.times+t_gatewidth/2).max()
        else:
            self.tmin = self.times.min()
            self.tmax = self.times.max()

        self.filterfunc = filterfunc
        self.device = times.device
        if waveform is not None:
            if self.waveform.time.max()<self.tmin:
                self.tmin += -self.waveform.time.max()
            self.tmax += -self.waveform.time.min()
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
        self.tolerance=tolerance

    def forward(self):
        def wvlngth_integrand(wave_length):
            lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                omega = -1j*s
                tmp_u0 = u0(omega, lmb)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+rte(self.sigma, self.thick, omega, lmb)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= lmb**2 / tmp_u0
            
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                if self.waveform is not None:
                    intg /= -1j*omega.squeeze()                    
                return torch.squeeze(intg/c.fourpi).real
            cbft = invLaplace(s_integrand, n_coefficients=16, device=self.device)
            return cbft(self.times)

        hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
        res = integrate_around_loop(self.source, self.receiver, hkl1)
        cbft=CubicSpline(self.times, res, transform=torch.log10)

        if self.t_gatewidth is None:
            t = self.times

        if self.waveform is not None:
            t_eval = t[:, None] - self.waveform.time[None, :]
            tshp = t_eval.shape
            t_eval = t_eval.reshape(-1)
            resp_step = cbft(t_eval).real.reshape(tshp)
            dbdt = resp_step @ self.waveform.d2wdt2
            if self.t_gatewidth is not None:
                cbft = CubicSpline(t, dbdt, transform=torch.log10)
            else:
                return dbdt
        if self.t_gatewidth is not None:
            return cbft.integrate(self.t_st, self.t_end, normalize=True)
        else:
            return cbft(self.times)

    def __call__(self, sigma, thickness):
        if isinstance(sigma, np.ndarray): sigma = torch.from_numpy(sigma)
        if isinstance(thickness, np.ndarray): thickness = torch.from_numpy(thickness)
        self.sigma = (sigma     + 0j).to(self.device)[...,None,None]
        self.thick = (thickness + 0j).to(self.device)[...,None,None]
        return self.forward().real*c.mu0


tx = Loop.circle(40)
rx = Receiver()
wv = Waveform.example()
t = torch.logspace(-5, -3, 16, dtype=torch.double)
# t = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05, 3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
#         1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
#         5.575e-04, 7.035e-04], dtype=torch.float64)

f0 = Forward(tx, rx, t, wv)
f1 = ForwardLaplace(tx, rx, t, tolerance=1e-5)
f2 = FastForward(tx, rx, t, tolerance=1e-5)

rho = torch.tensor([40, 200, 70], dtype=torch.double)
thk = torch.tensor([10, 35], dtype=torch.double)
sigma = 1./rho

dbdt0 = f0(sigma, thk)
dbdt1 = f1(sigma, thk)
dbdt2 = f2(sigma, thk)


from time import time as timer
bt = timer()
n_times = 10
for i in range(n_times):
    dbdt2=f2(sigma, thk)
print(f'run time: {1e3*(timer()-bt)/n_times:1.2f} ms')

import matplotlib.pyplot as plt

plt.figure()
plt.loglog(t.numpy(), dbdt0.numpy())
plt.loglog(t.numpy(), dbdt1.numpy())
plt.show()