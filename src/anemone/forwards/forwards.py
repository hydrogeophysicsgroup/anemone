from anemone.transforms import Hankel, invFourier, CubicSpline, invLaplace, coefficient_bounds
from anemone.physics import rte_raiche1999 as rte, freespace_u0 as u0, colecole, mpa2colecole, colecole2mpa
from anemone.physics import rte_nbc_laplace as rtelap
from anemone.physics import drte_dsigma, calc_dpropmat
from anemone import constants as c, rdtype, cdtype

from anemone.experiment import integrate_around_loop
import torch
from numpy import ndarray
import warnings

from anemone.experiment import lamontagne
from copy import deepcopy

def wv_to_t(t_eval, t_sampled, waveform=None):
    t_eval = t_eval.to(rdtype)
    t_sampled = t_sampled.to(rdtype)

    logT = torch.log(t_sampled)#.cpu()
    dT = logT[1]-logT[0]
    if waveform is None:
        t2t = lamontagne((torch.log(t_eval)[:,None]-logT[None,:])/dT)
    else:
        t2t = torch.zeros(t_eval.shape+t_sampled.shape, device=t_sampled.device, dtype=t_eval.dtype)

        for t_wv,k_wv in zip(waveform.time, waveform.d2wdt2):
            t2t += k_wv*torch.nan_to_num(lamontagne((torch.log(t_eval-t_wv)[:,None]-logT[None,:])/dT))
    return t2t

def intMat(t_in, t_out):
    t_out = t_out.to(t_in.dtype)
    logT = torch.log(t_in)#.cpu()
    dT = logT[1]-logT[0]
    t2t = torch.nan_to_num(lamontagne((torch.log(t_out)[:,None]-logT[None,:])/dT))
    return t2t

def gating_matrix(t_sampled, t_st, t_end, n_eval=1000):
    logT = torch.log(t_sampled)#.cpu()
    dT = logT[1]-logT[0]

    dt = (t_end - t_st)/n_eval

    gate_matrix = torch.zeros(t_st.shape+t_sampled.shape, dtype=t_sampled.dtype, device=t_sampled.device)

    for i in range(n_eval):
        t_eval = torch.log(t_st + dt*i)
        gate_matrix += lamontagne((t_eval[:, None]-logT[None, :])/dT)
    return gate_matrix/n_eval

def logify_forward(cls):
    class LogFwr(cls):
        def __call__(self, *args, **kwargs):
            if not kwargs.get('jac', False):
                return torch.abs(super().__call__(*args, **kwargs)).log()
            else:
                return super().__call__(*args, **kwargs)
    return LogFwr

class BaseForward:
    """ Has everything that the Forward needs, but is very slow. 
        Good for understanding the physics, but not for production use.
    """
    omega = None
    lamda = None
    kind = 'res-only'
    t_end = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=None):
        warnings.warn("BaseForward is very slow, use Forward instead.", DeprecationWarning)
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)

        if t_end is not None:
            self.t_st = self.times
            self.t_end= t_end
            self.times = (self.t_st * self.t_end)**0.5
            self.tmin = self.t_st.min()
            self.tmax = self.t_end.max()
        else:
            self.tmin = self.times.min()
            self.tmax = self.times.max()

        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype)
        if receiver.z.numel()>1:
            self.rz = receiver.z.flatten()[0]
            assert torch.all(receiver.z==self.rz), "Receiver z coordinates are not the same"
        else:
            self.rz = self.receiver.z

        self.pfield_dz = torch.abs(self.srcz-self.rz)
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)

        self.waveform = waveform

        
        

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
        
        self.tmin /= 2.
        self.tmax *= 2.

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
                intg   = rte(self.sigma, self.thick, self.omega, self.lamda)*torch.exp(-tmp_u0*self.sfield_dz)
                intg  *= self.lamda

                return intg/c.fourpi
            
            hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device)
            res = integrate_around_loop(self.source, self.receiver, hkl1).squeeze().T
            if self.waveform is not None:
                res /= -1j*self.omega.squeeze()
            if self.filterfunc is not None:
                res *= self.filterfunc.omega_filter(self.omega).squeeze()
            return res
        ift = invFourier(frq_integrand, smp_per_decade=16, range=(self.tmin, self.tmax), device=self.device)
        _, _, self.t_eval = ift.calc_tvals(self.tmin, self.tmax)
        return ift

    def forward(self):
        self._ft = self.calc_ft()
        if self.waveform is not None:
            t_eval = self.t_eval[:, None] - self.waveform.time[None, :]
            resp_step = torch.nan_to_num(self._ft(t_eval).real)
            dbdt = resp_step @ self.waveform.d2wdt2
        else:
            dbdt = self._ft(self.t_eval).real
        if self.t_end is not None:
            tmp_ft = CubicSpline(self.t_eval, dbdt, transform=torch.log10)
            return tmp_ft.integrate(self.t_st, self.t_end, normalize=True).real
        else:
            tmp_ft = CubicSpline(self.t_eval, dbdt, transform=torch.log10, integrate=False)
            return tmp_ft(self.times).real
        
    @classmethod
    def example(cls, **kwargs):
        from anemone.experiment import Loop, Receiver, Waveform
        kwargs.setdefault('source',  Loop.square(40))
        kwargs.setdefault('receiver',  Receiver())
        kwargs.setdefault('times', torch.logspace(-6, -3, 32))
        kwargs.setdefault('waveform', Waveform.example())
        return cls(**kwargs)

    def __call__(self, rho, thickness):
        if not isinstance(rho, torch.Tensor):       rho       = torch.tensor(rho)
        if not isinstance(thickness, torch.Tensor): thickness = torch.tensor(thickness)
        self.rho = ((rho+0j)[..., None, None]).to(self.device).to(cdtype)
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None, ...]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device).to(cdtype)
        return self.forward().squeeze().real*c.mu0

    @classmethod
    def colecole(cls, source, receiver, times, **kwargs):
        

        class ColeCole(cls):
            kind = 'colecole'
            @property
            def sigma(self):
                return 1./colecole(self.rho, self.m, self.tau, self.c, self.omega)

            def __call__(self, rho, m, tau,  C, thickness):
                self.rho   =       rho.to(self.device).to(cdtype)[..., None,None]
                self.m     =         m.to(self.device).to(cdtype)[..., None,None]
                self.tau   =       tau.to(self.device).to(cdtype)[..., None,None]
                self.c     =         C.to(self.device).to(cdtype)[..., None,None]
                if rho.ndim > thickness.ndim:
                    thickness = thickness[:, None]
                self.thick = ((thickness+0j)[..., None, None]).to(self.device)

                return self.forward()*c.mu0
        
        return ColeCole(source, receiver, times, **kwargs)
    
    @classmethod
    def mpa(cls, source, receiver, times, **kwargs):

        class MPA(cls):
            kind = 'mpa'
            @property
            def sigma(self):
                rho, m_new, tau_cc, C = mpa2colecole(self.rho, self.phi_max, self.tau_phi, self.c)
                return 1./colecole(rho, m_new, tau_cc, C, self.omega)
            
            @staticmethod
            def colecole2mpa(rho, m, tau, C):
                from anemone.physics import colecole2mpa # not used much, so imported here to keep things clean.
                rho, phi_max, tau_phi, C = colecole2mpa(rho, m, tau, C)
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
            kind = 'cpa'
            omega_L = c.twopi*1e-5 if drake_model else 0.
            @property
            def sigma(self):
                b = -2.*self.phi / c.pi
                k = self.rho * self.omega_L**b
                return k*(1j*self.omega + self.omega_L)**(-b)

            def __call__(self, rho, phi, thickness):
                self.rho   =       rho.to(self.device).to(cdtype)[..., None,None]
                self.phi     =     phi.to(self.device).to(cdtype)[..., None,None]
                if rho.ndim > thickness.ndim:
                    thickness = thickness[:, None]
                self.thick = ((thickness+0j)[..., None, None]).to(self.device)
                return self.forward()*c.mu0

        return CPA(source, receiver, times, **kwargs)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)                    

class ForwardLaplace(BaseForward):
    """ Essentially the same as the BaseForward, but uses invserse Laplace transforms instead of inverse Fourier transforms."""
    sfield_rte = None
    pfield = None
    tmp_u0 = None
    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=None, t_smp_per_decade=12):
        super().__init__(source, receiver, times, waveform, filterfunc, t_end, tolerance)
        if waveform is not None:
            if self.tmin<0: 
                self.tmin = torch.tensor([1e-9])
                Warning('tmin is less than 0, setting tmin to 0.25e-6')
            self.tmin /= 2.2
            self.tmax *= 2.2

            self.t_smp_per_decade = t_smp_per_decade
            self.t_eval = torch.logspace(torch.log10(self.tmin).item(), torch.log10(self.tmax).item(), int(t_smp_per_decade*torch.log10(self.tmax/self.tmin)), dtype=rdtype)
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
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-self.tmp_u0*self.sfield_dz)# / self.tmp_u0
                if self.pfield     is None: self.pfield     = torch.exp(-self.tmp_u0*self.pfield_dz)# / self.tmp_u0
                cRTE = torch.nan_to_num(rte(self.sigma, self.thick, omega, lmb))
                intg = cRTE*self.sfield_rte

                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)

                if self.waveform is not None:
                    intg /= -1j*omega

                return intg
            cbft = invLaplace(s_integrand, device=self.device, n_coefficients=16)
            res =  cbft(self.t_eval)
            res *= lmb / c.fourpi
            return res.squeeze().swapaxes(-2,-1)

        hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
        res = integrate_around_loop(self.source, self.receiver, hkl1)
        ft = CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)
        return ft

class DualMoment(BaseForward):
    def __init__(self, source, receiver, times, waveforms, filterfunc=None, tolerance=None):
        fwr1 = Forward(source, receiver, times[0], waveforms[0], filterfunc, tolerance=tolerance)
        fwr2 = Forward(source, receiver, times[1], waveforms[1], filterfunc, tolerance=tolerance)
        self = fwr1 + fwr2

    @classmethod
    def example(cls, tx=None, rx=None, times=None, waveforms=None):
        from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter
        if tx is None: tx = Loop.square(40)
        if rx is None: rx = Receiver()
        if times is None: times = [torch.logspace(-6, -4, 32), torch.logspace(-5, -3, 32)]
        if waveforms is None: waveforms = [Waveform.example(), Waveform.example()]
        return cls(tx, rx, times, waveforms, ButterworthFilter(3e5))

    @property
    def lm(self):
        return self.dpred[:len(self.times[0])]

    @property
    def hm(self):
        return self.dpred[len(self.times[0]):]

# Experimental forwards
# These are not used in production, but are here for reference and testing purposes.

class ForwardFD(BaseForward):
    def __init__(self, source, receiver, freqs, waveform=None, filterfunc=None, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.freqs = freqs if isinstance(freqs, torch.Tensor) else torch.tensor(freqs)
        self.device = freqs.device
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(freqs.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(freqs.device)
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype)
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
            return H.T

    
def waveform_matrix_with_primary(t_in, t_out, waveform):
    t_out = t_out.to(rdtype)
    t_in  = t_in.to(rdtype)

    logT = torch.log(t_in)
    dT = torch.diff(logT, prepend=logT[1:2]-2*logT[:1])
    # dT = logT[1]-logT[0]
    #dT = torch.roll(dT, 1)  # Shift to align with t_in

    t2t = torch.zeros(t_out.shape+(len(t_in)+1,), device=t_in.device, dtype=t_out.dtype)

    for t_wv,k_wv in zip(waveform.time, waveform.d2wdt2):
        c_t_eval = t_out - t_wv
        positive_time = c_t_eval > 0.
        t2t[~positive_time , 0] -= k_wv
        t2t[ positive_time, 1:] += k_wv*lamontagne((torch.log(c_t_eval[positive_time])[:,None]-logT[None,:])/dT[None,:])
        
        # t_negative = -1.*c_t_eval
        # positive_time = t_negative >= 0.
        # t2t[positive_time, 1:] -= k_wv*torch.nan_to_num(lamontagne((torch.log(t_negative[positive_time])[:,None]-logT[None,:])/dT[None,:]))
        # t2t[~positive_time, 0] -= k_wv
    return t2t

class BstepLaplace(ForwardLaplace):
    geomz = None
    t2t   = None

    def __init__(self, source, receiver, times, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
        self.t_eval = self.times
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype)
        if receiver.z.numel()>1:
            self.rz = receiver.z.flatten()[0]
            assert torch.all(receiver.z==self.rz), "Receiver z coordinates are not the same"
        else:
            self.rz = self.receiver.z
        self.pfield_dz = torch.abs(self.srcz-self.rz)
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)    
        self.device = times.device
        self.tolerance = tolerance

    def forward(self):
        def wvlngth_integrand(wave_length):
            self.lmb = wave_length[None, :, None]
            def s_integrand(s):
                """ Hz from horizontal electric dipole. """
                self.s_lap = s[None, None, :]
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-self.lmb*self.sfield_dz)
                intg = rtelap(self.sigma, self.thick, self.s_lap, self.lmb)*self.sfield_rte
                intg /= -self.s_lap
                return torch.squeeze(intg).real
            res = invLaplace(s_integrand, device=self.device, n_coefficients=16)(self.t_eval)
            res[None, ...] *= self.lmb / c.fourpi
            return res.swapaxes(-2,-1)
        if self.geomz is None:
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, smp_per_decade=16, tolerance=self.tolerance)
            def geomz_r(r):
                return self.source.geomz(self.receiver, r).to(rdtype).to(self.device)
            Hstep, self.geomz, self.lmb = hkl1.preintegrate(geomz_r, self.rmin, self.rmax)
            self.sfield_rte = None
        else:
            Hstep = torch.einsum('...i,i->...', wvlngth_integrand(self.lmb.squeeze()), self.geomz)
        return Hstep
        
    def __call__(self, rho, thickness):
        self.rho = rho
        self.thick = thickness
        return self.forward().squeeze()*c.mu0
    
class BstepHankel(BaseForward):
    geomz=None
    sfield_rte=None
    lmb=None
    tout=None
    def __init__(self, source, receiver, tmin, tmax, tolerance=None):
        self.source = source
        self.receiver = receiver
        self.tmin = tmin
        self.tmax = tmax
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(tmin.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(tmin.device)
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype)
        self.device = tmin.device
        if receiver.z.numel()>1:
            self.rz = receiver.z.flatten()[0]
            assert torch.all(receiver.z==self.rz), "Receiver z coordinates are not the same"
        else:
            self.rz = self.receiver.z
        self.pfield_dz = torch.abs(self.srcz-self.rz)
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)
        self.tolerance = tolerance
    
    def calc_bstep(self):
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                if self.sfield_rte is None: self.sfield_rte = torch.exp(-wave_length*self.sfield_dz)
                intg = rte(self.sigma, self.thick, omega, wave_length[None,None,:])*self.sfield_rte*wave_length
                return intg
            
            if self.geomz is None:
                hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, smp_per_decade=10, tolerance=self.tolerance)
                def geomz_r(r):
                    return self.source.geomz(self.receiver, r).to(rdtype).to(self.device)
                Hstep, self.geomz, self.lmb = hkl1.preintegrate(geomz_r, self.rmin, self.rmax)
                self.sfield_rte = None
            else:
                Hstep = (wvlngth_integrand(self.lmb.squeeze())@self.geomz).squeeze()
            Hstep /= -1j*omega.squeeze()
            return Hstep.squeeze()
        ift = invFourier(frq_integrand, smp_per_decade=16, device=self.device)
        # ift._recenter_function((self.tmax/self.tmin, 1e12*self.tmax/self.tmin))
        # print(f"Recentered function to {torch.sqrt((self.tmax/self.tmin)*(self.tmax/self.tmin*1e12))}")
        self.tout, Hstep = ift.range(self.tmin, self.tmax)
        return Hstep/c.fourpi

    # def calc_bstep(self):

    #     def wvlngth_integrand(wave_length):
    #         lmb = wave_length[None, :, None]
    #         def frq_integrand(f):
    #             omega = c.twopi*f[None,None, :]
    #             intg = rte(self.sigma, self.thick, omega, lmb)
    #             return intg / (-1j*omega.squeeze())
            
    #         ift = invFourier(frq_integrand, device=self.device, smp_per_decade=20)
    #         ift._recenter_function((self.tmin/10, self.tmin*10))
    #         self.tout, Hstep = ift.range(self.tmin, self.tmax)


    #         if self.sfield_rte is None: self.sfield_rte = torch.exp(-lmb*self.sfield_dz)
    #         Hstep *= self.sfield_rte*lmb/c.fourpi
    #         return Hstep.swapaxes(-2,-1).squeeze()

    #     if self.geomz is None:
    #         hkl1 = Hankel(wvlngth_integrand, 1, device=self.device, smp_per_decade=10)
    #         def geomz_r(r):
    #             return self.source.geomz(self.receiver, r).to(rdtype).to(self.device)
    #         Hstep, self.geomz, self.lmb = hkl1.preintegrate(geomz_r, self.rmin, self.rmax, tolerance=self.tolerance)
    #         self.sfield_rte = None
    #     else:
    #         Hstep = (wvlngth_integrand(self.lmb.squeeze())@self.geomz).squeeze()
    #     return Hstep.squeeze()
    
    def get_bstep(self, rho, thickness):
        self.rho = rho
        self.thick = thickness
        # t,bstep = self.calc_bstep()
        # return t, bstep.squeeze()*c.mu0
        bstep = self.calc_bstep()
        return self.tout, bstep.squeeze()*c.mu0
    
class OnTimeForward:
    hankel = None
    laplace = None
    t2t = None

    def __init__(self, source, receiver, times, waveform, tolerance=None, t_smp_per_decade=10):
        self.source = source
        self.receiver = receiver
        self.times = times
        self.tmin = times.min().to(rdtype)
        self.tmax = times.max().to(rdtype)
        self.device = times[0].device
        from biotsavart import Loop, ObsPoint, calc_bfield
        _loop = Loop(self.source.x.numpy(), self.source.y.numpy(), self.source.z.numpy())
        _obs = ObsPoint(self.receiver.x.numpy(), self.receiver.y.numpy(), self.receiver.z.numpy())
        self.pfield = torch.from_numpy(calc_bfield(_loop.xyz, _obs.xyz)[..., 2]).reshape([1])# / c.mu0

        self.rmin = self.source.min_horizontal_distance(self.receiver).to(self.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(self.device)
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype)
        self.rz = self.receiver.z.flatten()[0]
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)

        self.waveform = waveform
        self.tmin += -self.waveform.time.max()
        self.tmax += -self.waveform.time.min()
        waveform.time = waveform.time.to(self.device)
        waveform.amplitude = waveform.amplitude.to(self.device)
        waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
        self.t_smp_per_decade = t_smp_per_decade
        self.t_switch = torch.tensor([1e-3])
        
        

        self.tolerance=tolerance
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))
        
    def forward(self):
        if self.hankel is None:
            self.hankel = BstepHankel(self.source, self.receiver, self.t_switch, self.tmax, tolerance=self.tolerance)

        t, bstep_hankel = self.hankel.get_bstep(self.rho, self.thick)
        dt = t[1]/t[0]
        Ndec = int(torch.log10(t[0]/1e-11)/torch.log10(dt))+1
        # print(f"Number of decades in Laplace: {Ndec}, dt={dt}, t[0]={t[0]}")
        self.t_laplace = t[0]*dt**torch.arange(-Ndec, 0)

        if self.laplace is None:
            self.laplace = BstepLaplace(self.source, self.receiver, self.t_laplace, tolerance=self.tolerance)
        bstep_laplace = self.laplace(self.rho, self.thick)
        if self.t2t is None:
            self.t_in = torch.concat([self.t_laplace, t])
            self.t2t = waveform_matrix_with_primary(self.t_in, self.times, self.waveform)
            self.t2t[:,1] = 0.
        self.bstep = torch.concat([self.pfield, bstep_laplace, bstep_hankel]).real
        return self.t2t @ self.bstep

    def __call__(self, rho, thickness):
        self.rho = rho[..., None, None]
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None, ...]
        self.thick = thickness[..., None, None]
        return self.forward()

class VMDForward(BaseForward):
    geomz = None
    t2t   = None

    def __init__(self, source_height, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=None):
        from anemone.system import Loop
        tx = Loop.square(1., z=source_height)
        self.rx_dist = torch.sqrt(receiver.x**2 + receiver.y**2)
        if self.rx_dist==0.: self.rx_dist = torch.tensor([1e-2])
        self.t2t = None
        super().__init__(tx, receiver, times, waveform, filterfunc, t_end, tolerance)
        

    def forward(self):
        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]
            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lmb = wave_length[None, None, :]          
                self.tmp_u0 = u0(self.omega, self.lmb)
                self.sfield_rte = torch.exp(-self.tmp_u0*self.sfield_dz)*self.lmb**2 / c.fourpi
                intg = rte(self.sigma, self.thick, self.omega, self.lmb)*self.sfield_rte
                return intg.squeeze()
            hkl1 = Hankel(wvlngth_integrand, 0, smp_per_decade=10, device=self.device)
            hkl1._recenter_function(self.rx_dist)
            Hr = ((hkl1.func(hkl1.lamda))@(hkl1.fc+0j))/self.rx_dist
            if self.filterfunc is not None:
                Hr *= self.filterfunc.omega_filter(self.omega.squeeze())
            if self.waveform is not None:
                Hr /= -1j*self.omega.squeeze()
            return Hr.squeeze()
        
        if self.t2t is None:
            self.cbft=invFourier(frq_integrand, device=self.device, tolerance=self.tolerance)
            def t2t_func(t_in):
                if self.t_end is not None:
                    waveform_matrix = wv_to_t(t_in, t_in, waveform=self.waveform)
                    gate_matrix = gating_matrix(t_in, self.t_st, self.t_end, n_eval=1000)
                    return gate_matrix @ waveform_matrix
                else:
                    return wv_to_t(self.times, t_in, waveform=self.waveform)

            
            Ht, self.t2t, self.omega1 = self.cbft.preintegrate(t2t_func, self.tmin, self.tmax)
        else:
            Ht = torch.einsum('ij,...j->...i', self.t2t, self.cbft.f1(self.omega1.squeeze())*self.omega1.squeeze())
        self.dpred = Ht.real
        return self.dpred

def make_H2t(t2t, geomz, omega, lmb, tolerance=1e-6, sigma_ref=0.1):
    import numpy as np

    t_indices = np.arange(t2t.shape[0])
    t1 = (t2t[0, :, None]*geomz.real[None, :]).numpy()
    
    from anemone.physics import rte_halfspace
    rte_ref = rte_halfspace(torch.tensor(sigma_ref), omega[:,None], lmb[None,:]).squeeze().cpu().numpy()

    if tolerance is not None:
        full_mask = np.zeros_like(t1, dtype=bool)
        for i in t_indices:
            t1 = (t2t[i, :, None]*geomz.real[None, :]).numpy()*rte_ref
            t1_shape = t1.shape
            t1sort = np.abs(t1).flatten()
            # indices = np.arange(len(t1sort))

            sort_idx = np.argsort(t1sort)
            t1sort = t1sort[sort_idx]
            t1cumsum = np.cumsum(t1sort)
            t1cumsum /= t1cumsum[-1]
            idx = np.searchsorted(t1cumsum, tolerance)#-1

            mask = np.zeros_like(t1sort, dtype=bool)
            mask[idx:] = True


            inv_sort = np.argsort(sort_idx)
            mask = mask[inv_sort].reshape(t1_shape)
            full_mask += mask
    else:
        full_mask = np.ones_like(t1, dtype=bool)

    omega_i = np.arange(omega.squeeze().shape[0])
    lamda_i = np.arange(lmb.squeeze().shape[0])
    omega_i, lamda_i = np.meshgrid(omega_i, lamda_i, indexing='ij')

    omega_i = omega_i[full_mask]
    lamda_i = lamda_i[full_mask]


    H2t = np.zeros(t2t.shape[:1]+(full_mask.sum(),), dtype=complex)

    for i in range(t2t.shape[0]):
        t1 = (t2t[i, :, None]*geomz.real[None, :]).numpy()
        H2t[i] = t1[full_mask].flatten()

    H2tweight = np.abs(H2t).sum(0)#.numpy()

    idx_H2t = np.argsort(H2tweight)[::-1]
    H2t = H2t[:, idx_H2t]

    Omega,Lambda = np.meshgrid(omega.squeeze().cpu().numpy(), lmb.squeeze().cpu().numpy(), indexing='ij')
    Omega = Omega[full_mask]
    Lambda = Lambda[full_mask]
    return torch.tensor(H2t), torch.tensor(Omega[idx_H2t])+0j, torch.tensor(Lambda[idx_H2t])+0j


class Forward(BaseForward):
    geomz = None
    t2t   = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=1e-7, do_setup=True):
        device = times.device

        super().__init__(source, receiver, times, waveform, filterfunc, t_end, tolerance)
        self.t2t = None
        if do_setup: 
            self.setup()
            self.device=device
            self.H2t = self.H2t.to(self.device).to(cdtype)
            self.omega = self.omega.to(self.device).to(cdtype)
            self.lmb = self.lmb.to(self.device).to(cdtype)

    def setup(self):
        self.rho   = 40*torch.ones(2, dtype=cdtype)[:, None, None]
        self.thick = 20*torch.ones(1, dtype=cdtype)[:, None, None]

        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]
            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lmb = wave_length[None, None, :]          
                self.sfield_rte = torch.exp(-self.lmb*self.sfield_dz)*self.lmb / c.fourpi
                intg = rte(1./self.rho, self.thick, self.omega, self.lmb)*self.sfield_rte
                return intg.squeeze()
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device)
            def geomz_r(r):
                return self.source.geomz(self.receiver, r).to(rdtype).to(self.device)
            Hr, self.geomz, self.lmb1 = hkl1.preintegrate(geomz_r, self.rmin, self.rmax)
            
            self.geomz *= torch.exp(-self.lmb1*self.sfield_dz)*self.lmb1 / c.fourpi
            return Hr.squeeze()
        

        self.cbft=invFourier(frq_integrand, device=self.device)
        def t2t_func(t_in):
            if self.t_end is not None:
                waveform_matrix = wv_to_t(t_in, t_in, waveform=self.waveform)
                gate_matrix = gating_matrix(t_in, self.t_st, self.t_end, n_eval=1000)
                return gate_matrix @ waveform_matrix
            else:
                return wv_to_t(self.times, t_in, waveform=self.waveform)

        
        Ht, self.t2t, self.omega1 = self.cbft.preintegrate(t2t_func, self.tmin, self.tmax)

        if self.filterfunc is not None:
            self.t2t *= self.filterfunc.omega_filter(self.omega1*c.sqrt_twopi)#.squeeze()
        if self.waveform is not None:
            self.t2t /= -1j*c.sqrt_twopi
        else:
            self.t2t *= self.omega1.squeeze()

        self.H2t, self.omega, self.lmb = make_H2t(self.t2t, self.geomz, self.omega1, self.lmb1, tolerance=self.tolerance)

        self.H2t    = self.H2t.to(self.device).to(cdtype)
        self.H2t   /= torch.sqrt(self.omega)
        self.omega *= c.sqrt_twopi
        self.rho    = None
        self.thick  = None
        self.H2tjac = self.H2t#.to(torch.complex64)
        self.omegajac = self.omega#.to(torch.complex64)
        self.lmbjac = self.lmb#.to(torch.complex64)


    def forward(self):
        if self.H2t is None: self.setup()
        with torch.no_grad():
            cRTE = rte(self.sigma, self.thick, self.omega, self.lmb).squeeze()
            return torch.einsum('ik,...k->i...', self.H2t, cRTE).real

    def jacobian(self):
        # dRTE = drte_dsigma(self.sigma.to(torch.complex64), self.thick.to(torch.complex64), self.omegajac, self.lmbjac).squeeze()
        dRTE = drte_dsigma(self.sigma, self.thick, self.omegajac, self.lmbjac).squeeze()
        
        if len(dRTE.shape)>2:
            Gr = torch.einsum('ijk,lk->jil', dRTE, self.H2tjac).real
            Gr /= (self.rho.real.squeeze().T**2)[:,:,None]
        else:
            Gr = torch.einsum('ik,...k->i...', dRTE.real, self.H2tjac.real) - torch.einsum('ik,jk->ij', dRTE.imag, self.H2tjac.imag)
            Gr /= (self.rho.real.squeeze()**2)[:,None]
        return Gr
    
    def __call__(self, rho, thickness, jac=False):
        with torch.no_grad():
            if not isinstance(rho, torch.Tensor):       rho       = torch.tensor(rho)
            if not isinstance(thickness, torch.Tensor): thickness = torch.tensor(thickness)        
            self.rho = ((rho+0j)[..., None, None]).to(cdtype)
            if rho.ndim > thickness.ndim:
                thickness = thickness[:, None, ...]
            self.thick = ((thickness+0j)[..., None, None]).to(cdtype)
            if jac:
                G = self.jacobian()
                return G*c.mu0
            else:            
                return self.forward()*c.mu0      
        
    def __add__(self, other):
        newfwr = self.copy()
        tmpother = other.copy()
        assert other.kind       == self.kind,       "Forwards must be the same type"
        assert other.source     == self.source,     "Current implementation requires that Forwards have the same source"
        assert other.receiver   == self.receiver,   "Current implementation requires that Forwards have the same receiver"
        assert other.filterfunc == self.filterfunc, "Current implementation requires that Forwards have the same filters"

        newfwr.tmin     , tmpother.tmin      = [min(newfwr.tmin, other.tmin)]*2
        newfwr.tmax     , tmpother.tmax      = [max(newfwr.tmax, other.tmax)]*2
        
        
        tol =  min(filter(None, [newfwr.tolerance, tmpother.tolerance]), default=None)
        newfwr.tolerance, tmpother.tolerance = [tol]*2
        
        newfwr.tolerance, tmpother.tolerance = None, None
        newfwr.lmb,       tmpother.lmb       = None, None
        newfwr.omega,     tmpother.omega     = None, None

        newfwr.setup()
        tmp_lmb = newfwr.lmb1.squeeze().clone()
        tmp_omega = newfwr.omega1.squeeze().clone()
        tmpother.setup()
        t2t = torch.vstack([newfwr.t2t, tmpother.t2t])
        newfwr.H2t, newfwr.omega, newfwr.lmb = make_H2t(t2t, newfwr.geomz, tmp_omega, tmp_lmb, tolerance=tol)

        newfwr.H2t       = newfwr.H2t.to(newfwr.device).to(cdtype)
        newfwr.H2t      /= torch.sqrt(newfwr.omega)
        newfwr.omega    *= c.sqrt_twopi
        newfwr.rho       = None
        newfwr.thick     = None
        newfwr.H2tjac    = newfwr.H2t#[:, :newfwr.H2t.shape[1]//2]
        newfwr.omegajac  = newfwr.omega#[:newfwr.omega.shape[0]//2]
        newfwr.lmbjac    = newfwr.lmb#[:newfwr.lmb.shape[0]//2]

        newfwr.t1slc = slice(None, len(newfwr.times))
        newfwr.t2slc = slice(len(newfwr.times), None)
        newfwr.times = torch.cat([newfwr.times, other.times])

        return newfwr
    
    def save(self, filename):
        if not filename.endswith('.em1'):
            filename = filename+'.em1'
        torch.save({'H2t': self.H2t.cpu(),
                    'omega': self.omega.cpu(),
                    'lmb': self.lmb.cpu()}, filename)
        
    @classmethod
    def load(cls, filename):
        data = torch.load(filename)
        fwr = cls.example(do_setup=False)
        fwr.H2t = data['H2t'].to(cdtype)
        fwr.omega = data['omega'].to(cdtype)
        fwr.lmb = data['lmb'].to(cdtype)
        return fwr