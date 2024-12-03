from anemone.transforms import Hankel, invFourier, CubicSpline, invLaplace
from anemone.physics import refl_transverse_electric as rte, freespace_u0 as u0, colecole, mpa2colecole, colecole2mpa
from anemone import constants as c
from anemone.experiment import integrate_around_loop
import torch
from numpy import ndarray

class Forward:
    omega = None
    lamda = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.srcz = torch.tensor([source.z[0]], dtype=torch.float64)
        if receiver.z.numel()>1:
            self.rz = receiver.z.flatten()[0]
            assert torch.all(receiver.z==self.rz), "Receiver z coordinates are not the same"
        else:
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
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))
    @property
    def sigma(self):
        return 1./self.rho
    
    def calc_ft(self):
        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lamda = wave_length[None, None, :]
                
                tmp_u0 = u0(self.omega, self.lamda)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+rte(self.sigma, self.thick, self.omega, self.lamda)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= self.lamda**2 / tmp_u0
                
                if self.waveform is not None:
                    intg /= -1j*self.omega
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(self.omega)
                return intg/c.fourpi
            
            hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
            res = integrate_around_loop(self.source, self.receiver, hkl1)
            return res.T
        
        return invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device, tolerance=self.tolerance)

    def forward(self):
        self._ft = self.calc_ft()

        if self.t_gatewidth is None:
            t = self.times


        if self.waveform is not None:
            t_eval = t[:, None] - self.waveform.time[None, :]
            resp_step = self._ft(t_eval).real
            dbdt = resp_step @ self.waveform.d2wdt2
            if self.t_gatewidth is not None:
                self._ft = CubicSpline(t, dbdt, transform=torch.log10)
            else:
                return dbdt
        if self.t_gatewidth is not None:
            return self._ft.integrate(self.t_st, self.t_end, normalize=True).real
        else:
            return self._ft(self.times).real
        
    @classmethod
    def example(cls, **kwargs):
        from anemone.experiment import Loop, Receiver, Waveform
        kwargs.setdefault('source',  Loop.square(40))
        kwargs.setdefault('receiver',  Receiver())
        kwargs.setdefault('times', torch.logspace(-6, -3, 32))
        kwargs.setdefault('waveform', Waveform.example())
        return cls(**kwargs)
        

    def compute_relstd(self, sigma, thickness, noise_slope_exponent=-0.5, noise_level_at_1ms=1e-8, moment=1, relative_noise=0.03):
        moment *= self.source.area
        t_gates, data_val = self.times, self(sigma, thickness)
        b = noise_level_at_1ms/(1e-3**noise_slope_exponent)
        abs_noise = b*(t_gates**noise_slope_exponent)
        y = data_val*moment
        std_rel = abs_noise/y
        relstd_tot = torch.sqrt(std_rel**2 + relative_noise**2)
        return relstd_tot

    def __call__(self, rho, thickness):
        if isinstance(rho, ndarray): rho = torch.from_numpy(rho)
        if isinstance(thickness, ndarray): thickness = torch.from_numpy(thickness)
        self.rho = ((rho+0j)[..., None, None]).to(self.device)
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward()*c.mu0

    @classmethod
    def colecole(cls, source, receiver, times, **kwargs):

        class ColeCole(cls):
            @property
            def sigma(self):
                return 1./colecole(self.rho, self.m, self.tau, self.c, self.omega)

            def __call__(self, rho, m, tau,  C, thickness):
                self.rho   =       rho.to(self.device).to(torch.complex128)[..., None,None]
                self.m     =         m.to(self.device).to(torch.complex128)[..., None,None]
                self.tau   =       tau.to(self.device).to(torch.complex128)[..., None,None]
                self.c     =         C.to(self.device).to(torch.complex128)[..., None,None]
                if rho.ndim > thickness.ndim:
                    thickness = thickness[:, None]
                self.thick = ((thickness+0j)[..., None, None]).to(self.device)

                return self.forward()*c.mu0
        
        return ColeCole(source, receiver, times, **kwargs)
    
    @classmethod
    def mpa(cls, source, receiver, times, **kwargs):
        class MPA(cls):
            @property
            def sigma(self):
                rho, tau_cc, m_new, C = mpa2colecole(self.rho, self.phi_max, self.tau_phi, self.c)
                return 1./colecole(rho, m_new, tau_cc, C, self.omega)
            
            @staticmethod
            def colecole2mpa(rho, tau_cc, m, C):
                from anemone.physics import colecole2mpa # not used much, so imported here to keep things clean.
                rho, phi_max, tau_phi, C = colecole2mpa(rho, tau_cc, m, C)
                return rho, phi_max, tau_phi, C

            def __call__(self, rho, phi_max, tau_phi, C, thickness):
                self.rho     =     rho.to(self.device)[..., None,None]
                self.phi_max = phi_max.to(self.device)[..., None,None]
                self.tau_phi = tau_phi.to(self.device)[..., None,None]
                self.c       =       C.to(self.device)[..., None,None]
                if rho.ndim > thickness.ndim:
                    thickness = thickness[:, None]
                self.thick = ((thickness+0j)[..., None, None]).to(self.device)
                return self.forward()*c.mu0
        return MPA(source, receiver, times, **kwargs)
    
    @classmethod
    def cpa(cls, source, receiver, times, drake_model=True, **kwargs):
        class CPA(cls):
            omega_L = c.twopi*1e-5 if drake_model else 0.
            @property
            def sigma(self):
                b = -2.*self.phi / c.pi
                k = self.rho * self.omega_L**b
                return k*(1j*self.omega + self.omega_L)**(-b)

            def __call__(self, rho, phi, thickness):
                self.rho   =       rho.to(self.device).to(torch.complex128)[..., None,None]
                self.phi     =     phi.to(self.device).to(torch.complex128)[..., None,None]
                if rho.ndim > thickness.ndim:
                    thickness = thickness[:, None]
                self.thick = ((thickness+0j)[..., None, None]).to(self.device)
                return self.forward()*c.mu0

        return CPA(source, receiver, times, **kwargs)

class DualMoment(Forward):
    def __init__(self, source, receiver, times, waveforms, filterfunc=None, tolerance=None):
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
        self.waveform = []
        for waveform in waveforms:
            if waveform is not None:
                if waveform.time.max()<self.tmin:
                    self.tmin += -waveform.time.max()
                self.tmax += -waveform.time.min()
                waveform.time = waveform.time.to(self.device)
                waveform.amplitude = waveform.amplitude.to(self.device)
                waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            self.waveform.append(waveform)
        self.tolerance=tolerance

    @classmethod
    def example(cls, tx=None, rx=None, times=None, waveforms=None):
        from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter
        if tx is None: tx = Loop.square(40)
        if rx is None: rx = Receiver()
        if times is None: times = [torch.logspace(-6, -4, 32), torch.logspace(-5, -3, 32)]
        if waveforms is None: waveforms = [Waveform.example(), Waveform.example()]
        return cls(tx, rx, times, ButterworthFilter(3e5), waveforms)

    def forward(self):
        self._ft = self.calc_ft()
        
        dbdts = []
        for i in range(2):
            t = self.times[i]
            t_eval = t[:, None] - self.waveforms[i].time[None, :]
            resp_step = self._ft(t_eval)
            dbdt = resp_step.real @ self.waveforms[i].d2wdt2
            dbdts.append(dbdt.T)
        
        self.dpred = torch.concatenate(dbdts).T    
        return self.dpred

    @property
    def lm(self):
        return self.dpred[:len(self.times[0])]

    @property
    def hm(self):
        return self.dpred[len(self.times[0]):]
    
class ForwardLaplace(Forward):
    sfield_rte = None
    pfield = None
    tmp_u0 = None
    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=None, t_smp_per_decade=6):
        super().__init__(source, receiver, times, waveform, filterfunc, t_gatewidth, tolerance)
        if waveform is not None:
            self.tmin += -self.waveform.time.max()
            self.tmax += -self.waveform.time.min()
            if self.tmin<0: 
                self.tmin = torch.tensor([0.1e-6])
                Warning('tmin is less than 0, setting tmin to 0.25e-6')
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            self.t_smp_per_decade = t_smp_per_decade
            
            self.t_eval = torch.logspace(torch.log10(self.tmin).item(), torch.log10(self.tmax).item(), int(t_smp_per_decade*torch.log10(self.tmax/self.tmin)), dtype=torch.double)

        else:
            self.t_eval = self.times
        self.tolerance=tolerance
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))

    def calc_ft(self):
        def wvlngth_integrand(wave_length):
            lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                omega = -1j*s[None, None, :]
                
                if self.tmp_u0     is None: self.tmp_u0 = u0(omega, lmb)
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-self.tmp_u0*self.sfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi
                if self.pfield     is None: self.pfield = torch.exp(-self.tmp_u0*self.pfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi

                intg = self.pfield +  rte(self.sigma, self.thick, omega, lmb)*self.sfield_rte

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                if self.waveform is not None:
                    intg /= -1j*omega
                return torch.squeeze(intg).real
            cbft = invLaplace(s_integrand, device=self.device, n_coefficients=8)
            return cbft(self.t_eval)

        hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
        res = integrate_around_loop(self.source, self.receiver, hkl1)
        return CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)


class DualMomentLaplace(DualMoment, ForwardLaplace):
    def __init__(self, source, receiver, times, filterfunc, waveforms, tolerance=None, t_smp_per_decade=6):
        super().__init__(source, receiver, times, filterfunc, waveforms)
        if waveforms is not None:
            self.t_smp_per_decade = t_smp_per_decade
            self.t_eval = torch.logspace(torch.log10(self.tmin), torch.log10(self.tmax), int(t_smp_per_decade*torch.log10(self.tmax/self.tmin)), dtype=torch.double)
        else:
            self.t_eval = torch.concatenate(self.times)
        self.tolerance=tolerance
        self.save={}
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))              

    def calc_ft(self):
        return ForwardLaplace.__dict__['calc_ft'].__get__(self)()

class ForwardFD(Forward):
    def __init__(self, source, receiver, freqs, waveform=None, filterfunc=None, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.freqs = freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs)
        self.device = freqs.device
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(freqs.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(freqs.device)
        self.srcz = torch.tensor([source.z[0]], dtype=torch.float64)
        self.rz = self.receiver.z.flatten()[0]
        self.pfield_dz = torch.abs(self.srcz-self.rz)#[:, None, None]
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)#[:, None, None]

        self.filterfunc = filterfunc
        self.waveform = waveform
        self.tolerance=tolerance
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))      

    def forward(self):
        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]
            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lmb = wave_length[None, None, :]
                
                tmp_u0 = u0(self.omega, self.lmb)
                intg = torch.exp(-tmp_u0*self.pfield_dz)+rte(self.sigma, self.thick, self.omega, self.lmb)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= self.lmb**2 / tmp_u0
            
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(self.omega)
                return torch.squeeze(intg/c.fourpi)
            
            hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
            res = integrate_around_loop(self.source, self.receiver, hkl1)
            if self.waveform is not None:
                res /= -1j*self.omega.squeeze()
            return res
        
        H = frq_integrand(self.freqs)

        if self.waveform is not None:
            return H*self.waveform.fourier_d2wdt2(self.freqs)
        else:
            return H
        

class ForwardGeomOpLaplace(ForwardLaplace):
    geomz=None

    def calc_ft(self):
        def wvlngth_integrand(wave_length):
            lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                omega = -1j*s[None, None, :]
                
                if self.tmp_u0     is None: self.tmp_u0 = u0(omega, lmb)
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-self.tmp_u0*self.sfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi
                if self.pfield     is None: self.pfield = torch.exp(-self.tmp_u0*self.pfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi

                intg = self.pfield +  rte(self.sigma, self.thick, omega, lmb)*self.sfield_rte

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                if self.waveform is not None:
                    intg /= -1j*omega
                return torch.squeeze(intg).real
            cbft = invLaplace(s_integrand, device=self.device, n_coefficients=8)
            return cbft(self.t_eval)

        hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, tolerance=self.tolerance)
        r,HR = hkl1._range(self.rmin, self.rmax)
        if self.geomz is None:
            self.geomz = self.source.geomz(self.receiver, r)
            self.r = r
        else:
            assert torch.all(self.r==r)
        res = torch.einsum('...ij,...j->...i', HR, self.geomz).squeeze().T       
        return CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)
        
        
class ForwardGeomOp(Forward):
    geomz=None
    def calc_ft(self):
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
            
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, tolerance=self.tolerance)
            r,HR = hkl1.range(self.rmin, self.rmax)
            if self.geomz is None:
                self.geomz = self.source.geomz(self.receiver, r).to(torch.complex128)
                self.r=r
            else:
                assert torch.all(self.r==r)
            
            res = torch.einsum('...ik,...jk',  self.geomz, HR)
            if self.waveform is not None:
                res /= -1j*omega.squeeze()
            return res
        return invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device, tolerance=self.tolerance)



from anemone.experiment import lamontagne

def wv_to_t(t_eval, t_sampled, waveform=None):
    t_eval = t_eval.to(torch.double)
    t_sampled = t_sampled.to(torch.double)

    logT = torch.log(t_sampled)#.cpu()
    dT = logT[1]-logT[0]
    if waveform is None:
        t2t = lamontagne((torch.log(t_eval)[:,None]-logT[None,:])/dT)
    else:
        t2t = torch.zeros(t_eval.shape+t_sampled.shape, device=t_sampled.device, dtype=t_eval.dtype)

        for t_wv,k_wv in zip(waveform.time, waveform.d2wdt2):
            t2t += k_wv*lamontagne((torch.log(t_eval-t_wv)[:,None]-logT[None,:])/dT)
    return t2t


class Forward3(Forward):
    geomz=None
    t2t =None
    
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
            
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, tolerance=self.tolerance, smp_per_decade=10)
            r,HR = hkl1.range(self.rmin, self.rmax)
            if self.geomz is None:
                self.geomz = self.source.geomz(self.receiver, r).to(torch.complex128)
                self.r = r
            else:
                assert torch.all(self.r==r)
            
            res = torch.einsum('...ij,...j->...i', HR, self.geomz)
            if self.waveform is not None:
                res /= -1j*omega.squeeze()
            return res
        
        cbft=invFourier(frq_integrand, smp_per_decade=10, device=self.device)
        self.t_sampled,Ht = cbft.range(self.tmin, self.tmax)



        if self.t2t is None:
            self.t2t = wv_to_t(self.times, self.t_sampled, waveform=self.waveform)
        if len(Ht.shape)==1:
            return torch.einsum('...i,...i', Ht.real, self.t2t).squeeze()
        else:
            return torch.einsum('...ik,...jk->...ij', Ht.real, self.t2t).squeeze()


class DualMoment3(DualMoment):
    geomz=None
    t2t =None

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
            
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device)
            r,HR = hkl1.range(self.rmin, self.rmax)
            if self.geomz is None:
                self.geomz = self.source.geomz(self.receiver, r).to(torch.complex128)
                self.r = r
            else:
                assert torch.all(self.r==r)
            
            res = torch.einsum('...ij,...j->...i', HR, self.geomz)
            res /= -1j*omega.squeeze()
            return res
        
        cbft=invFourier(frq_integrand, smp_per_decade=10, device=self.device)
        self.t_sampled,Ht = cbft.range(self.tmin, self.tmax)

        if self.t2t is None:
            self.t2t = torch.vstack([wv_to_t(self.times[0], self.t_sampled, waveform=self.waveforms[0]),
                                     wv_to_t(self.times[1], self.t_sampled, waveform=self.waveforms[1])])
        if len(Ht.shape)==1:
            return torch.einsum('...i,...i', Ht.real, self.t2t).squeeze()
        else:
            return torch.einsum('...ik,...jk->...ij', Ht.real, self.t2t).squeeze()

class ForwardLaplace3(ForwardLaplace):
    geomz = None
    t2t   = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None, tolerance=None, t_smp_per_decade=8):
        super().__init__(source, receiver, times, waveform, filterfunc, t_gatewidth, tolerance, t_smp_per_decade)
        self.t2t = wv_to_t(self.times, self.t_eval, waveform=self.waveform)#.to_sparse()

    def forward(self):
        def wvlngth_integrand(wave_length):
            lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                omega = -1j*s[None, None, :]
                
                if self.tmp_u0 is None: self.tmp_u0 = u0(omega, lmb)
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-self.tmp_u0*self.sfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi
                if self.pfield is None: self.pfield = (torch.exp(-self.tmp_u0*self.pfield_dz)*lmb**2 / self.tmp_u0 / c.fourpi)
                intg = self.pfield +  (rte(self.sigma, self.thick, omega, lmb)*self.sfield_rte)

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                if self.waveform is not None:
                    intg /= -1j*omega
                return torch.squeeze(intg).real

            cbft = invLaplace(s_integrand, device=self.device, n_coefficients=8)
            return cbft(self.t_eval).real.squeeze()

        hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, tolerance=self.tolerance, smp_per_decade=10)
        r,HR = hkl1.range(self.rmin, self.rmax)
        if self.geomz is None:
            self.geomz = self.source.geomz(self.receiver, r).to(torch.double)
            self.r=r
        else:
            assert torch.all(self.r==r)
        Ht = torch.einsum('...ik,...jk',  self.geomz, HR)

        if len(Ht.shape)==1:
            return torch.einsum('...i,...i', Ht, self.t2t).mT.squeeze()
        elif len(Ht.shape)==3:
            return torch.einsum('...ik,...jk->...ij', Ht.squeeze().T, self.t2t).squeeze()
        else:
            return torch.einsum('...ik,...jk->...ij', Ht, self.t2t).squeeze()        