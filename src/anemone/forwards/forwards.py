from anemone.transforms import Hankel, invFourier, CubicSpline, invLaplace

from anemone.system import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
from anemone.system import lamontagne, stepoff_response, integrate_around_loop
from anemone.system import waveform_matrix, interpolation_matrix, gating_matrix, make_H2t

from anemone.physics import rte, freespace_u0 as u0, colecole, mpa2colecole, colecole2mpa, drte_dsigma

from anemone import constants as c, rdtype, cdtype

import torch
import warnings

from nosj import nosj
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
    log_scale  : bool         = False
    description: str          = 'Base Forward Model'

    kind = 'res' # tag of the earth model, set by the earth model classes; not a serialised field


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
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype())
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

            wv_tmax = self.waveform.time.max() if isinstance(self.waveform, Waveform) else max([wv.time.max() for wv in self.waveform])
            wv_tmin = self.waveform.time.min() if isinstance(self.waveform, Waveform) else min([wv.time.min() for wv in self.waveform])

            if wv_tmax <self.tmin:
                self.tmin += -wv_tmax
            else:
                self.tmin /= 100.
            self.tmax += -wv_tmin
            # waveform.time = waveform.time.to(self.device)
            # waveform.amplitude = waveform.amplitude.to(self.device)
            # waveform.d2wdt2 = waveform.d2wdt2.to(self.device)


        self.tolerance=tolerance
        
        self.tmin /= 2.
        self.tmax *= 2.

        bfield = torch.from_numpy(biotsavart.calc_bfield(source.xyz, receiver.xyz)[..., -1])

        if self.t_end is not None:
            t_primary = torch.logspace(self.tmin.log10(), self.tmax.log10(), 16*(int((self.tmax/self.tmin).log10())+1))
        else:
            t_primary = self.times*1.

        if isinstance(self.waveform, Waveform):
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
        else:
            self.primary = 0.
       

        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))

        
        self.omega = kwargs.get('omega', None)
        self.lamda = kwargs.get('lamda', None)
        self.description = kwargs.get('description', 'Base Forward Model')
        self.log_scale = kwargs.get('log_scale', False)

    
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
            return tmp_ft.integrate(self.t_st, self.t_end, normalize=True).real*c.mu0
        else:
            tmp_ft = CubicSpline(self.t_eval, dbdt, transform=torch.log10, integrate=False)
            return tmp_ft(self.times).real*c.mu0
        
    @classmethod
    def example(cls, **kwargs):
        from anemone.experiment import Loop, Receiver, Waveform
        kwargs.setdefault('source',  Loop.square(40))
        kwargs.setdefault('receiver',  Receiver())
        kwargs.setdefault('times', torch.logspace(-6, -3, 32, dtype=rdtype()))
        kwargs.setdefault('waveform', Waveform.example())
        return cls(**kwargs)

    def _set_param(self, value, complex=False):
        if not isinstance(value, torch.Tensor): value = torch.tensor(value)
        if complex: value = value.to(cdtype())
        return value[..., None, None].to(self.device)

    def _set_thickness(self, thickness):
        if not isinstance(thickness, torch.Tensor): thickness = torch.tensor(thickness)
        if self.rho.ndim-2 > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device).to(cdtype())

    def _evaluate(self, jac=False):
        if jac: raise NotImplementedError('{} does not provide an analytic jacobian'.format(type(self).__name__))
        return self.forward().squeeze().real

    def _finalize(self, data, jac=False):
        if jac: return data
        data = data + self.primary
        if self.log_scale: data = torch.log(torch.abs(data))
        return data

    def __call__(self, rho, thickness=None, jac=False):
        self.rho = self._set_param(rho)
        if thickness is not None: self._set_thickness(thickness)
        return self._finalize(self._evaluate(jac), jac)

    @property
    def sigma(self):
        return 1./self.rho

    @classmethod
    def colecole(cls, source, receiver, times, **kwargs) -> 'ForwardColeCole':
        return specialize(cls, ColeColeEarth)(source, receiver, times, **kwargs)
    
    @classmethod
    def mpa(cls, source, receiver, times, **kwargs) -> 'ForwardMPA':
        return specialize(cls, MPAEarth)(source, receiver, times, **kwargs)
    
    @classmethod
    def cpa(cls, source, receiver, times, drake_model=True, **kwargs) -> 'ForwardCPA':
        self = specialize(cls, CPAEarth)(source, receiver, times, **kwargs)
        self.omega_L = c.twopi*1e-5 if drake_model else 0.
        return self

    def logify(self):
        self.log_scale = True
        return self


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
            self.omega = self.omega.to(self.device).to(cdtype())
            self.lamda = self.lamda.to(self.device).to(cdtype())

    def setup(self):
        self.rho   = 40*torch.ones(2, dtype=cdtype())[:, None, None]
        self.thick = 20*torch.ones(1, dtype=cdtype())[:, None, None]
        with torch.no_grad():
            self.rho   = 40*torch.ones(2, dtype=cdtype())[:, None, None]
            self.thick = 20*torch.ones(1, dtype=cdtype())[:, None, None]

            def frq_integrand(f):
                self.omega = c.twopi*f[None,:,None]
                def wvlngth_integrand(wave_length):
                    """ Hz from horizontal electric dipole. """
                    self.lamda = wave_length[None, None, :]          
                    sfield_rte = torch.exp(-self.lamda*self.sfield_dz)*self.lamda / c.fourpi
                    intg = rte(1./self.rho, self.thick, self.omega, self.lamda)*sfield_rte
                    return intg.squeeze()
                hkl1 = Hankel(wvlngth_integrand, 1, device=self.device)
                def geomz_r(r):
                    return self.source.geomz(self.receiver, r).to(rdtype()).to(self.device)
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

            if self.filterfunc is None:
                self.H2t    = 2.*self.H2t.real.to(self.device).to(rdtype())

            self.H2t *= c.mu0
            self.omega *= c.sqrt_twopi
            self.rho    = None
            self.thick  = None

    def forward(self):
        if self.H2t is None: 
            self.setup()

        with torch.no_grad():
            cRTE = rte(self.sigma, self.thick, self.omega, self.lamda).squeeze()
            if self.H2t.dtype==rdtype():
                cRTE = cRTE.real.to(rdtype())
            res = torch.einsum('ik,...k->i...', self.H2t, cRTE)
            if len(res.shape)>1: res = res.mT
            # dbdt is a physically real quantity; any residual imaginary part
            # is numerical noise from the Hankel/Fourier transforms. This
            # matches BaseForward._evaluate, which always discards it via .real.
            return res.real.to(rdtype())

    def jacobian(self):
        with torch.no_grad():
            dRTE = drte_dsigma(self.sigma, self.thick, self.omega, self.lamda)
            if self.H2t.dtype==rdtype():
                dRTE = dRTE.real.to(rdtype())
            
            if len(dRTE.shape)>2:
                Gr = torch.einsum('ijk,lk->jil', dRTE, self.H2t)
                Gr /= -(self.rho.real.squeeze().T**2)[:,:,None]
            else:
                Gr = torch.einsum('ik,...k->i...', self.H2t, dRTE).T
                Gr /= -(self.rho.real.squeeze()**2)[:,None]
            # The jacobian of a real quantity (dbdt) w.r.t. a real parameter
            # (resistivity) is real. When filterfunc is set, H2t stays complex
            # and Gr would otherwise stay complex too, which breaks downstream
            # linear algebra (e.g. G.T @ Cdi @ G) that assumes real dtype.
            return Gr.real.to(rdtype())
    
    def _evaluate(self, jac=False):
        with torch.no_grad():
            return self.jacobian().cpu() if jac else self.forward()
        
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
        # t2t /= torch.sqrt(tmp_omega)
        newfwr.H2t, newfwr.omega, newfwr.lamda = make_H2t(t2t, newfwr.geomz, tmp_omega, tmp_lamda, tolerance=tol)

        if newfwr.filterfunc is None:
            newfwr.H2t       = 2.*newfwr.H2t.real.to(newfwr.device).to(rdtype())
        # newfwr.H2t      /= torch.sqrt(newfwr.omega.real)
        # setup() applies this mu0 scaling to H2t for individual forwards; since
        # we recompute H2t here from the combined t2t rather than reusing
        # newfwr/tmpother's already-scaled H2t, it must be (re)applied here too.
        newfwr.H2t      *= c.mu0
        newfwr.omega    *= c.sqrt_twopi
        newfwr.rho       = None
        newfwr.thick     = None

        newfwr.t1slc = slice(None, len(newfwr.times))
        newfwr.t2slc = slice(len(newfwr.times), None)
        newfwr.times = torch.cat([newfwr.times, other.times])
        newfwr.primary = torch.cat([newfwr.primary, other.primary])
        newfwr.waveform = [self.waveform, other.waveform]
        return newfwr
    
    def set_device(self, device):
        self.H2t   = self.H2t.to(device)
        self.omega = self.omega.to(device)
        self.lamda = self.lamda.to(device)
        self.device=device

    @classmethod
    def from_gex(cls, fname, **kwargs):
        from libaarhusxyz import GEX
        import numpy as np
        gex = GEX(fname)
        x,y = gex.General['TxLoopPoint'][:, 0], gex.General['TxLoopPoint'][:, 1]
        x,y = np.append(x, x[0]),               np.append(y, y[0])
        z   = np.ones_like(x)*abs(gex.General['TxCoilPosition'][2])
        tx  = Loop(x,y,z)
        rx  = Receiver(*abs(gex.General['RxCoilPosition']))        
        lmgates = torch.from_numpy(gex.gate_times('Channel1')[1:,0])
        hmgates = torch.from_numpy(gex.gate_times('Channel2')[1:,0])
        order, fcut = gex.Channel1['TiBLowPassFilter']
        rx_filter = ButterworthFilter(fcut,order)

        LMwv = Waveform(gex.General['WaveformLMPoint'][:, 0], gex.General['WaveformLMPoint'][:, 1])
        HMwv = Waveform(gex.General['WaveformHMPoint'][:, 0], gex.General['WaveformHMPoint'][:, 1])

        lmfwr = cls(tx, rx, lmgates, waveform=LMwv, filterfunc=rx_filter, do_setup=False, **kwargs)
        hmfwr = cls(tx, rx, hmgates, waveform=HMwv, filterfunc=rx_filter, do_setup=False, **kwargs)
        return lmfwr + hmfwr

    @classmethod
    def from_temfile(cls, fname, **kwargs):
        raise NotImplementedError

class ColeColeEarth:
    """ Dispersive earth parameterised with the Cole-Cole model. """
    kind = 'colecole'

    @property
    def sigma(self):
        return 1./colecole(self.rho, self.m, self.tau, self.c, self.omega)

    def jacobian(self):
        rho,m,tau,c,thick,w = self.rho.squeeze(), self.m.squeeze(), self.tau.squeeze(), self.c.squeeze(), self.thick.squeeze(), self.omega.squeeze()
        with torch.no_grad():

            drds   = drte_dsigma(self.sigma, self.thick, self.omega, self.lamda).T
            drdrho = - drds / rho[None,:]**2
            iwtp1  = 1+1j*w[:,None]*tau[None,:]
            drhodm = -rho*(1-iwtp1**(-c[None,:]))
            drhodt = -c*rho[None,:]*m[None,:]*(iwtp1**(-c-1))*(1j*w[:,None])
            drhodc = -rho[None,:]*m[None,:]*torch.log(iwtp1)*(iwtp1**(-c[None,:]))
            G_rte  = torch.concatenate([drdrho, drdrho*drhodm, drdrho*drhodt, drdrho*drhodc], dim=1).real
            
            if self.H2t.dtype == rdtype():
               G_rte           = G_rte.real.to(rdtype())

            return torch.einsum('ij,jk->ik', self.H2t, G_rte)
    
    def __call__(self, rho, m, tau, C, thickness=None, jac=False):
        self.rho = self._set_param(rho)
        self.m   = self._set_param(m)
        self.tau = self._set_param(tau)
        self.c   = self._set_param(C)
        if thickness is None:
            self.thick = None
        else:
            self._set_thickness(thickness)
        return self._finalize(self._evaluate(jac), jac)


class MPAEarth:
    """ Dispersive earth parameterised with the maximum phase angle re-parameterisation of Cole-Cole. """
    kind = 'mpa'

    @property
    def sigma(self):
        rho, m_new, tau_cc, C = mpa2colecole(self.rho, self.phi_max, self.tau_phi, self.c)
        return 1./colecole(rho, m_new, tau_cc, C, self.omega)

    def __call__(self, rho, phi_max, tau_phi, C, thickness=None, jac=False):
        self.rho     = self._set_param(rho,     complex=False)
        self.phi_max = self._set_param(phi_max, complex=False)
        self.tau_phi = self._set_param(tau_phi, complex=False)
        self.c       = self._set_param(C,       complex=False)
        if thickness is None:
            self.thick = None
        else:
            self._set_thickness(thickness)
        return self._finalize(self._evaluate(jac), jac)

    @staticmethod
    def colecole2mpa(rho, m, tau, C):
        from anemone.physics import colecole2mpa # not used much, so imported here to keep things clean.
        rho, phi_max, tau_phi, C = colecole2mpa(rho, m, tau, C)
        return rho, phi_max, tau_phi, C


class CPAEarth:
    """ Dispersive earth with a constant phase angle. """
    kind = 'cpa'
    omega_L = c.twopi*1e-5

    @property
    def sigma(self):
        b = -2.*self.phi / c.pi
        k = self.rho * self.omega_L**b
        return k*(1j*self.omega + self.omega_L)**(-b)

    def __call__(self, rho, phi, thickness=None, jac=False):
        self.rho = self._set_param(rho)
        self.phi = self._set_param(phi)
        if thickness is None:
            self.thick = None
        else:
            self._set_thickness(thickness)
        return self._finalize(self._evaluate(jac), jac)


_specialized_forwards = {}

def _make_specialized(cls, base):
    # __init__ is set explicitly so that dataclass() does not shadow it with a generated one
    cls.__init__ = base.__init__
    annotations = {}
    for klass in reversed(cls.__mro__):
        annotations.update(vars(klass).get('__annotations__', {}))
    cls.__annotations__ = annotations
    return nosj(cls)

def specialized(base, earth):
    """ Register an explicitly written `earth`+`base` class as the specialization of that pair. """
    def decorator(cls):
        cls = _make_specialized(cls, base)
        _specialized_forwards[(base, earth)] = cls
        return cls
    return decorator

def specialize(base, earth):
    """ Build (and cache) the forward class combining a forward `base` with an `earth` model. """
    if issubclass(base, earth):
        return base
    key = (base, earth)
    if key not in _specialized_forwards:
        name = base.__name__ + earth.__name__.replace('Earth', '')
        _specialized_forwards[key] = _make_specialized(type(name, (earth, base), {}), base)
    return _specialized_forwards[key]


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
            self.t_eval = torch.logspace(torch.log10(self.tmin).item(), torch.log10(self.tmax).item(), int(t_smp_per_decade*torch.log10(self.tmax/self.tmin)), dtype=rdtype())
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
        self.srcz = torch.tensor([source.z[0]], dtype=rdtype())
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

# written out explicitly (rather than built by specialize) so editors can resolve their __call__ signatures
@specialized(Forward, ColeColeEarth)
class ForwardColeCole(ColeColeEarth, Forward):
    pass

@specialized(Forward, MPAEarth)
class ForwardMPA(MPAEarth, Forward):
    pass

@specialized(Forward, CPAEarth)
class ForwardCPA(CPAEarth, Forward):
    pass