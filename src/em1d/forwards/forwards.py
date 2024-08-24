from em1d.transforms import Hankel, invFourier, CubicSpline, invLaplace
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0, colecole
from em1d import constants as c
from em1d.experiment import integrate_around_loop
import torch
from numpy import ndarray

class Forward:
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
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                lmb = wave_length[None, None, :]
                
                tmp_u0 = u0(omega, lmb)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+rte(self.sigma, self.thick, omega, lmb)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= lmb**2 / tmp_u0
            
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                return torch.squeeze(intg/c.fourpi)
            
            hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
            res = integrate_around_loop(self.source, self.receiver, hkl1)
            if self.waveform is not None:
                res /= -1j*omega.squeeze()
            return res
        
        cbft=invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device, tolerance=self.tolerance)

        if self.t_gatewidth is None:
            t = self.times


        if self.waveform is not None:
            t_eval = t[:, None] - self.waveform.time[None, :]
            resp_step = cbft(t_eval).real
            dbdt = resp_step @ self.waveform.d2wdt2
            if self.t_gatewidth is not None:
                cbft = CubicSpline(t, dbdt, transform=torch.log10)
            else:
                return dbdt
        if self.t_gatewidth is not None:
            return cbft.integrate(self.t_st, self.t_end, normalize=True)
        else:
            return cbft(self.times)

    def compute_relstd(self, sigma, thickness, noise_slope_exponent=-0.5, noise_level_at_1ms=1e-8, moment=1, relative_noise=0.03):
        moment = self.source.area
        t_gates, data_val = self.times, self(sigma, thickness)
        b = noise_level_at_1ms/(1e-3**noise_slope_exponent)
        abs_noise = b*(t_gates**noise_slope_exponent)
        y = data_val*moment
        std_rel = abs_noise/y
        std_tot = torch.sqrt(std_rel**2 + relative_noise**2)
        return std_tot

    def __call__(self, sigma, thickness):
        if isinstance(sigma, ndarray): sigma = torch.from_numpy(sigma)
        if isinstance(thickness, ndarray): thickness = torch.from_numpy(thickness)
        self.sigma = ((sigma+0j)[..., None, None]).to(self.device)
        if sigma.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward().real*c.mu0

class DualMoment(Forward):
    def __init__(self, source, receiver, times, filterfunc, waveforms):
        self.source = source
        self.receiver = receiver
        self.times = times
        self.device = times[0].device

        self.rmin = self.source.min_horizontal_distance(self.receiver).to(self.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(self.device)
        self.srcz = torch.tensor([source.z[0]], dtype=torch.float64)
        self.rz = self.receiver.z
        self.pfield_dz = torch.abs(self.srcz-self.rz).to(self.device)
        self.sfield_dz = (torch.abs(self.srcz)+torch.abs(self.rz)).to(self.device)
                
        self.dz = torch.tensor(self.receiver.z+self.source.z[0], device=self.device)
        self.waveforms = waveforms
        self.tmin = min(self.times[0].min(), self.times[1].min())
        self.tmax = max(self.times[0].max(), self.times[1].max())

        self.filterfunc = filterfunc
        self.waveforms = []
        for waveform in waveforms:
            if waveform is not None:
                if waveform.time.max()<self.tmin:
                    self.tmin += -waveform.time.max()
                self.tmax += -waveform.time.min()
                waveform.time = waveform.time.to(self.device)
                waveform.amplitude = waveform.amplitude.to(self.device)
                waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            self.waveforms.append(waveform)

    def forward(self):
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                lmb = wave_length[None, None, :]
                tmp_u0 = u0(omega, lmb)
                cRTE = rte(self.sigma, self.thick, omega, lmb)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+cRTE*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= lmb**2 / tmp_u0

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                return torch.squeeze(intg/c.fourpi)

            hkl1 = Hankel(wvlngth_integrand, 1, smp_per_decade=10, range=(self.rmin, self.rmax), device=self.device, tolerance=1e-5)
            return -integrate_around_loop(self.source, self.receiver, hkl1)/(1j*omega.squeeze())
        
        cbft = invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device, tolerance=1e-6)
        
        dbdts = []
        for i in range(2):
            t = self.times[i]
            t_eval = t[:, None] - self.waveforms[i].time[None, :]
            resp_step = cbft(t_eval)
            dbdt = resp_step.real @ self.waveforms[i].d2wdt2
            dbdts.append(dbdt.T)
        return torch.concatenate(dbdts).T    

class IPForward(Forward):
    def forward(self):
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]
            self.sigma = 1./colecole(self.rho, self.m, self.tau, self.c, omega)

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                lmb = wave_length[None, None, :]
                
                tmp_u0 = u0(omega, lmb)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+rte(self.sigma, self.thick, omega, lmb)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= lmb**2 / tmp_u0
            
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                return torch.squeeze(intg/c.fourpi)
            
            hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
            res = integrate_around_loop(self.source, self.receiver, hkl1)
            if self.waveform is not None:
                res /= -1j*omega.squeeze()
            return res
        
        cbft=invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device, tolerance=self.tolerance)

        if self.t_gatewidth is None:
            t = self.times


        if self.waveform is not None:
            t_eval = t[:, None] - self.waveform.time[None, :]
            resp_step = cbft(t_eval).real
            dbdt = resp_step @ self.waveform.d2wdt2
            if self.t_gatewidth is not None:
                cbft = CubicSpline(t, dbdt, transform=torch.log10)
            else:
                return dbdt
        if self.t_gatewidth is not None:
            return cbft.integrate(self.t_st, self.t_end, normalize=True)
        else:
            return cbft(self.times)

    def __call__(self, rho, m, tau, C, thickness):
        self.rho   =       rho.to(self.device).to(torch.complex128)[..., None,None]
        self.thick = thickness.to(self.device).to(torch.complex128)[..., None,None]
        self.m     =         m.to(self.device).to(torch.complex128)[..., None,None]
        self.tau   =       tau.to(self.device).to(torch.complex128)[..., None,None]
        self.c     =         C.to(self.device).to(torch.complex128)[..., None,None]
        return self.forward()*c.mu0
    
class ForwardLaplace(Forward):
    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=None, t_smp_per_decade=6):
        super().__init__(source, receiver, times, waveform, filterfunc, t_gatewidth, tolerance)
        if waveform is not None:
            self.tmin += -self.waveform.time.max()
            self.tmax += -self.waveform.time.min()
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            self.t_smp_per_decade = t_smp_per_decade
            self.t_eval = torch.logspace(torch.log10(self.tmin), torch.log10(self.tmax), int(t_smp_per_decade*torch.log10(self.tmax/self.tmin)), dtype=torch.double)

        else:
            self.t_eval = self.times
        self.tolerance=tolerance
        self.save={}


    def forward(self):
        def wvlngth_integrand(wave_length):
            lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                omega = -1j*s
                
                if 'tmp_u0' not in self.save: 
                    self.save['tmp_u0'] = u0(omega, lmb)
                tmp_u0 = self.save['tmp_u0']

                if 'pfield' not in self.save:
                    self.save['pfield'] = (torch.exp(-tmp_u0*self.pfield_dz)*lmb**2 / tmp_u0 / c.fourpi)#.real
                pfield = self.save['pfield']     

                if 'sfield/rte' not in self.save:
                    self.save['sfield/rte'] = torch.exp(-tmp_u0*self.sfield_dz)*lmb**2 / tmp_u0 / c.fourpi
                sfield_rte = self.save['sfield/rte']

                intg = pfield + rte(self.sigma, self.thick, omega, lmb)*sfield_rte

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                if self.waveform is not None:
                    intg /= -1j*omega.squeeze()                    
                return torch.squeeze(intg).real
            cbft = invLaplace(s_integrand, n_coefficients=16, device=self.device)
            return cbft(self.t_eval)

        hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
        res = integrate_around_loop(self.source, self.receiver, hkl1)
        cbft=CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)

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
