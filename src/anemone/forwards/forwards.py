from anemone.transforms import Hankel, invFourier, CubicSpline, invLaplace, coefficient_bounds
from anemone.physics import rte_raiche1999 as rte, freespace_u0 as u0, colecole, mpa2colecole, colecole2mpa
from anemone.physics import rte_nbc_laplace as rtelap
from anemone.physics import drte_dsigma, calc_dpropmat
from anemone import constants as c, rdtype, cdtype

from anemone.system import integrate_around_loop
import torch
from numpy import ndarray
import warnings

from anemone.system import lamontagne
from copy import deepcopy

def waveform_matrix(t_eval, t_sampled, waveform=None):
    t_eval = t_eval.to(rdtype)
    t_sampled = t_sampled.to(rdtype)

    if waveform is None:
        t2t = interpolation_matrix(t_sampled, t_eval)
    else:
        logT = torch.log(t_sampled)#.cpu()
        dT = logT[1]-logT[0]
        t2t = torch.zeros(t_eval.shape+t_sampled.shape, device=t_sampled.device, dtype=t_eval.dtype)

        for t_wv,k_wv in zip(waveform.time, waveform.d2wdt2):
            t2t += k_wv*torch.nan_to_num(lamontagne((torch.log(t_eval-t_wv)[:,None]-logT[None,:])/dT))
    return t2t

def interpolation_matrix(t_in, t_out):
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

def stepoff_response(t):
    amp = torch.zeros_like(t)
    amp[t<0.] = 1.
    return amp

def logify_forward(cls):
    class LogFwr(cls):
        def __call__(self, *args, **kwargs):
            if not kwargs.get('jac', False):
                return torch.abs(super().__call__(*args, **kwargs)).log()
            else:
                return super().__call__(*args, **kwargs)
    return LogFwr

from nosj import nosj
from anemone.system import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
import biotsavart

@nosj
class BaseForward:
    """ Has everything that the Forward needs, but is very slow. 
        Good for understanding the physics, but not for production use.
    """
    source     : Loop
    receiver   : Receiver
    times      : torch.Tensor
    waveform   : Waveform = None
    filterfunc : FilterChain = None
    t_end      : torch.Tensor = None
    tolerance  : float = None
    omega      : torch.Tensor = None
    lamda      : torch.Tensor = None
    primary    : torch.Tensor = None
    kind       : str          = 'res'
    log_scale : bool         = False
    description: str          = 'Base Forward Model'


    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=None, **kwargs):
        warnings.warn("BaseForward is very slow, use Forward instead.", DeprecationWarning)
        self.source   = source
        self.receiver = receiver
        self.times    = times if isinstance(times, torch.Tensor) else torch.tensor(times)

        if t_end is not None:
            self.t_st = self.times
            self.t_end= t_end
            self.times = (self.t_st * self.t_end)**0.5
            self.tmin = self.t_st.min()
            self.tmax = self.t_end.max()
        else:
            self.tmin = self.times.min()
            self.tmax = self.times.max()
            self.t_end = None

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
            else:
                self.tmin /= 100.
            self.tmax += -self.waveform.time.min()
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)


        self.tolerance=tolerance
        
        self.tmin /= 2.
        self.tmax *= 2.

        bfield = torch.from_numpy(biotsavart.calc_bfield(source.xyz, receiver.xyz)[..., -1])

        if self.t_end is not None:
            t_primary = torch.logspace(self.tmin.log10(), self.tmax.log10(), 16*(int((self.tmax/self.tmin).log10())+1))
        else:
            t_primary = self.times*1.
        
        if waveform is not None:
            if self.filterfunc is None:
                primary =  stepoff_response(t_primary[:, None] - self.waveform.time[None, :]) @ self.waveform.d2wdt2
            else:
                primary =  self.filterfunc.stepoff_response(t_primary[:, None] - self.waveform.time[None, :]) @ self.waveform.d2wdt2
        else:
            primary  =  stepoff_response(t_primary) if self.filterfunc is None else self.filterfunc.stepoff_response(t_primary)

        if self.t_end is not None:
            primary_ft = CubicSpline(t_primary, primary, torch.log10)
            primary = primary_ft.integrate(self.t_st, self.t_end, normalize=True)

        if getattr(self.receiver, 'weighting', None) is not None:
            bfield = bfield @ self.receiver.weighting
        
        if bfield.numel()==1:
            self.primary = bfield*primary            
        else:
            self.primary = (bfield[:,None]*primary[None,:]).mean(0)
            
       

        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))

        
        self.omega = kwargs.get('omega', None)
        self.lamda = kwargs.get('lamda', None)
        self.kind  = kwargs.get('kind', 'res')
        self.description = kwargs.get('description', 'Base Forward Model')
        self.log_scale = kwargs.get('log_scale', False)

    # @property
    # def sigma(self):
    #     return 1./self.rho
    
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
        kwargs.setdefault('times', torch.logspace(-6, -3, 32, dtype=rdtype))
        kwargs.setdefault('waveform', Waveform.example())
        return cls(**kwargs)

    def _res_call(self, rho, thickness):
        if not isinstance(rho, torch.Tensor):       rho       = torch.tensor(rho)
        if not isinstance(thickness, torch.Tensor): thickness = torch.tensor(thickness)        
        self.rho = ((rho+0j)[..., None, None]).to(self.device).to(cdtype)
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device).to(cdtype)
        return self.forward().squeeze().real*c.mu0

    def _colecole_call(self, rho, m, tau,  C, thickness):
        self.rho   =       rho.to(self.device).to(cdtype)[..., None,None]
        self.m     =         m.to(self.device).to(cdtype)[..., None,None]
        self.tau   =       tau.to(self.device).to(cdtype)[..., None,None]
        self.c     =         C.to(self.device).to(cdtype)[..., None,None]
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device).to(cdtype)   
    
        return self.forward()*c.mu0        

    def _mpa_call(self, rho, phi_max, tau_phi, C, thickness):
        self.rho     =     rho.to(self.device)[..., None,None]
        self.phi_max = phi_max.to(self.device)[..., None,None]
        self.tau_phi = tau_phi.to(self.device)[..., None,None]
        self.c       =       C.to(self.device)[..., None,None]
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward()*c.mu0
    
    def _cpa_call(self, rho, phi, thickness):
        self.rho   =       rho.to(self.device).to(cdtype)[..., None,None]
        self.phi     =     phi.to(self.device).to(cdtype)[..., None,None]
        if rho.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward()*c.mu0

    def __call__(self, *args, **kwargs):
        if   self.kind == 'res':      data = self._res_call(*args, **kwargs)
        elif self.kind == 'mpa':      data = self._mpa_call(*args, **kwargs)
        elif self.kind == 'colecole': data = self._colecole_call(*args, **kwargs)
        elif self.kind == 'cpa':      data = self._cpa_call(*args, **kwargs)
        data += self.primary      
        if self.log_scale and not kwargs.get('jac', False): data = torch.log(torch.abs(data))
        return data

    @staticmethod
    def colecole2mpa(rho, m, tau, C):
        from anemone.physics import colecole2mpa # not used much, so imported here to keep things clean.
        rho, phi_max, tau_phi, C = colecole2mpa(rho, m, tau, C)
        return rho, phi_max, tau_phi, C

    @property
    def sigma(self):
        if   self.kind == 'res':      return 1./self.rho
        elif self.kind == 'colecole': return 1./colecole(self.rho, self.m, self.tau, self.c, self.omega)    
        elif self.kind == 'mpa':      
            rho, m_new, tau_cc, C = mpa2colecole(self.rho, self.phi_max, self.tau_phi, self.c)
            return 1./colecole(rho, m_new, tau_cc, C, self.omega)
        elif self.kind == 'cpa':    
            b = -2.*self.phi / c.pi
            k = self.rho * self.omega_L**b
            return k*(1j*self.omega + self.omega_L)**(-b)            
        
    @classmethod
    def colecole(cls, source, receiver, times, **kwargs):
        kwargs.setdefault('kind', 'colecole')
        return cls(source, receiver, times, **kwargs)
    
    @classmethod
    def mpa(cls, source, receiver, times, **kwargs):
        kwargs.setdefault('kind', 'mpa')
        return cls(source, receiver, times, **kwargs)
    
    @classmethod
    def cpa(cls, source, receiver, times, drake_model=True, **kwargs):
        kwargs.setdefault('kind', 'cpa')
        self = cls(source, receiver, times, **kwargs)
        self.omega_L = c.twopi*1e-5 if drake_model else 0.
        return self

    def logify(self):
        self.log_scale = True
        return self

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
        return CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)


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
                self.sfield_rte = torch.exp(-self.lmb*self.sfield_dz)*self.lmb**2 / c.fourpi
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
                    wv_mat = waveform_matrix(t_in, t_in, waveform=self.waveform)
                    gate_matrix = gating_matrix(t_in, self.t_st, self.t_end, n_eval=1000)
                    return gate_matrix @ wv_mat
                else:
                    return waveform_matrix(self.times, t_in, waveform=self.waveform)
            
            Ht, self.t2t, self.omega1 = self.cbft.preintegrate(t2t_func, self.tmin, self.tmax)
        else:
            Ht = torch.einsum('ij,...j->...i', self.t2t, self.cbft.f1(self.omega1.squeeze())*self.omega1.squeeze())
        self.dpred = Ht.real
        return self.dpred

def make_H2t(t2t, geomz, omega, lmb, tolerance=1e-6, sigma_ref=1e-2):
    import numpy as np
    t_indices = np.arange(t2t.shape[0])

    if geomz.ndim > 1:
        g_indices = np.arange(geomz.shape[0])
    else:
        g_indices = np.array([0])
        geomz = geomz[None, :]

    t1 = (t2t[0, :, None]*geomz[0].real[None, :]).numpy()

    from anemone.physics import rte_halfspace
    rte_ref = rte_halfspace(torch.tensor(sigma_ref), omega[:,None], lmb[None,:]).squeeze().cpu().numpy()

    if tolerance is not None:
        full_mask = np.zeros_like(t1, dtype=bool)
        for i_g in g_indices:
            c_geomz = geomz[i_g]
            for i_t in t_indices:
                t1 = (t2t[i_t, :, None].real*c_geomz.real[None, :]).numpy()*rte_ref
                t1_shape = t1.shape
                t1sort = np.abs(t1).flatten()

                sort_idx = np.argsort(t1sort)
                t1sort = t1sort[sort_idx]
                t1cumsum = np.cumsum(t1sort)
                t1cumsum /= t1cumsum[-1]
                idx = np.searchsorted(t1cumsum, tolerance)+1

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


    H2t = np.zeros(t2t.shape[:1]+geomz.shape[:-1]+(full_mask.sum(),), dtype=complex)

    for i_t in range(t2t.shape[0]):
        for i_g in g_indices:
            t1 = (t2t[i_t, :, None]*geomz[i_g].real[None, :]).numpy()
            H2t[i_t, i_g] = t1[full_mask].flatten()

    H2t = H2t.squeeze()
    Omega,Lambda = np.meshgrid(omega.squeeze().cpu().numpy(), lmb.squeeze().cpu().numpy(), indexing='ij')
    Omega = Omega[full_mask]
    Lambda = Lambda[full_mask]
    return torch.tensor(H2t), torch.from_numpy(Omega), torch.from_numpy(Lambda)

@nosj
class Forward(BaseForward): 
    H2t                 : torch.Tensor = None
    omega               : torch.Tensor = None
    lamda               : torch.Tensor = None

    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_end=None, tolerance=1e-6, do_setup=True, **kwargs):
        device = times.device

        super().__init__(source, receiver, times, waveform, filterfunc, t_end, tolerance, **kwargs)
        self.t2t = kwargs.get('t2t', None)
        self.H2t = kwargs.get('H2t', None)
        self.omega = kwargs.get('omega', None)
        self.lamda = kwargs.get('lamda', None)

        if self.H2t is None and do_setup: 
            self.setup()
            self.device=device
            # self.H2t = self.H2t.to(self.device).to(rdtype)
            self.omega = self.omega.to(self.device).to(rdtype)
            self.lamda = self.lamda.to(self.device).to(rdtype)

    def setup(self):
        self.rho   = 40*torch.ones(2, dtype=cdtype)[:, None, None]
        self.thick = 20*torch.ones(1, dtype=cdtype)[:, None, None]

        def frq_integrand(f):
            self.omega = c.twopi*f[None,:,None]
            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                self.lamda = wave_length[None, None, :]          
                self.sfield_rte = torch.exp(-self.lamda*self.sfield_dz)*self.lamda / c.fourpi
                intg = rte(1./self.rho, self.thick, self.omega, self.lamda)*self.sfield_rte
                return intg.squeeze()
            hkl1 = Hankel(wvlngth_integrand, 1, device=self.device)
            def geomz_r(r):
                return self.source.geomz(self.receiver, r).to(rdtype).to(self.device)
            Hr, self.geomz, self.lamda1 = hkl1.preintegrate(geomz_r, self.rmin, self.rmax)
            
            self.geomz *= torch.exp(-self.lamda1*self.sfield_dz)*self.lamda1 / c.fourpi
            return Hr.squeeze()
        

        self.cbft=invFourier(frq_integrand, device=self.device)
        def t2t_func(t_in):
            if self.t_end is not None:
                wv_mat = waveform_matrix(t_in, t_in, waveform=self.waveform)
                self.gate_mat = gating_matrix(t_in, self.t_st, self.t_end, n_eval=1000)
                
                return self.gate_mat @ wv_mat
            else:
                self.gate_mat = None
                return waveform_matrix(self.times, t_in, waveform=self.waveform)

        
        Ht, self.t2t, self.omega1 = self.cbft.preintegrate(t2t_func, self.tmin, self.tmax)



        if self.filterfunc is not None:
            self.t2t *= self.filterfunc.omega_filter(self.omega1*c.sqrt_twopi)#.squeeze()
        if self.waveform is not None:
            self.t2t /= -1j*c.sqrt_twopi
        else:
            self.t2t *= self.omega1.squeeze()
        self.t2t /= torch.sqrt(self.omega1.squeeze())
        self.H2t, self.omega, self.lamda = make_H2t(self.t2t, self.geomz.real, self.omega1.real, self.lamda1.real, tolerance=self.tolerance)

        self.H2t    = self.H2t#.real.to(self.device).to(rdtype)
        # self.H2t   /= torch.sqrt(self.omega)
        self.omega *= c.sqrt_twopi
        self.rho    = None
        self.thick  = None


    def forward(self):
        if self.H2t is None: 
            self.setup()
            # self.H2t = self.H2t.to(self.device).to(rdtype)

        with torch.no_grad():
            cRTE = rte(self.sigma, self.thick, self.omega, self.lamda).squeeze()#.to(rdtype)
            res = torch.einsum('ik,...k->i...', self.H2t, cRTE)
            if len(res.shape)>1: res = res.mT
            return res

    def jacobian(self):
        # dRTE = drte_dsigma(self.sigma.to(torch.complex64), self.thick.to(torch.complex64), self.omegajac, self.lamdajac).squeeze()
        dRTE = drte_dsigma(self.sigma, self.thick, self.omega, self.lamda).squeeze().real.T
        
        if len(dRTE.shape)>2:
            Gr = torch.einsum('ijk,lk->jil', dRTE.real, self.H2t.real).real
            Gr /= (self.rho.real.squeeze().T**2)[:,:,None]
        else:
            Gr = torch.einsum('ik,...k->i...', self.H2t.real, dRTE).T
            Gr /= (self.rho.real.squeeze()**2)[:,None]
        return Gr
    
    def _res_call(self, rho, thickness, jac=False):
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
        newfwr.lamda,     tmpother.lamda     = None, None
        newfwr.omega,     tmpother.omega     = None, None

        newfwr.setup()
        tmp_lamda = newfwr.lamda1.squeeze().clone()
        tmp_omega = newfwr.omega1.squeeze().clone()
        tmpother.setup()
        t2t = torch.vstack([newfwr.t2t, tmpother.t2t])
        newfwr.H2t, newfwr.omega, newfwr.lamda = make_H2t(t2t, newfwr.geomz, tmp_omega, tmp_lamda, tolerance=tol)

        newfwr.H2t       = newfwr.H2t.to(newfwr.device).to(cdtype)
        newfwr.H2t      /= torch.sqrt(newfwr.omega)
        newfwr.omega    *= c.sqrt_twopi
        newfwr.rho       = None
        newfwr.thick     = None

        newfwr.t1slc = slice(None, len(newfwr.times))
        newfwr.t2slc = slice(len(newfwr.times), None)
        newfwr.times = torch.cat([newfwr.times, other.times])

        return newfwr