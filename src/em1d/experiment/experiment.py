import numpy as np
from dataclasses import dataclass
import torch as tc

class Receiver:
    def __init__(self, x=0., y=0., z=0., grid=True) -> None:
        self.x = tc.tensor(x, dtype=tc.float64)
        self.y = tc.tensor(y, dtype=tc.float64)
        self.z = tc.tensor(z, dtype=tc.float64)
        
        if grid:
            self.grid()
        else:
            npoints = max(len(self.x), len(self.y),len(self.z))
            if len(self.x)==1: self.x = tc.ones(npoints)*self.x
            if len(self.y)==1: self.y = tc.ones(npoints)*self.y
            if len(self.z)==1: self.z = tc.ones(npoints)*self.z

    def grid(self):
        X,Y,Z = tc.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.shp = X.shape+(3,)
        self.x = X.squeeze()
        self.y = Y.squeeze()
        self.z = Z.squeeze()
        return self
    
    @property
    def xyz(self):
        return tc.squeeze(tc.moveaxis(tc.stack((
                            self.x.flatten(), 
                            self.y.flatten(), 
                            self.z.flatten())), 0, -1))

    def __len__(self):
        return self.x.numel()

@dataclass
class Loop:
    def __init__(self, x, y, z):
        if not isinstance(x, tc.Tensor):
            self.x = tc.tensor(x, dtype=tc.float64)
            self.y = tc.tensor(y, dtype=tc.float64)
            self.z = tc.tensor(z, dtype=tc.float64)
        else:
            self.x = x
            self.y = y
            self.z = z
        if not tc.all(self.z == self.z[0]): raise NotImplemented('current implementation only supports horizontal transmitter loops; all z values must be equal.')

    @classmethod
    def circle(cls, radius, n=40, **kwargs):
        t = tc.linspace(0, 2*tc.pi, n)
        x = radius*tc.cos(t)
        y = radius*tc.sin(t)
        z = tc.ones_like(x)*kwargs.get('z', 0.)
        lp = cls(x, y, z)
        lp.area = tc.pi*radius**2
        lp.radius = radius
        return lp

    @classmethod
    def square(cls, side, **kwargs):
        half = side / 2.
        x = tc.tensor([-half,  half, half, -half, -half])
        y = tc.tensor([-half, -half, half,  half, -half])
        z = tc.ones_like(x)*kwargs.get('z', 0.)
        lp = cls(x, y, z)
        lp.area = side*side
        return lp

    @classmethod
    def fig8(cls, side, **kwargs):
        half = side / 2.
        x = tc.tensor([-side, 0, 0, side, side, 0, 0, -side, -side])
        y = tc.tensor([-half, -half, half, half, -half, -half, half, half, -half])
        z = tc.ones_like(x)*kwargs.get('z', 0.)
        return cls(x, y, z)
    
    @classmethod
    def differential(cls, length, width, seperation=0.4, height=0.2):
        hl = length/2
        hw =  width/2
        x = [-hl, hl, hl, -hl, -hl]
        y = [-hw, -hw, hw, hw, -hw ]
        z = [height]*5
        bottom_coil = cls(x,y,z)
        z = [height+seperation]*5
        top_coil = cls(x[::-1], y[::-1], z)
        return bottom_coil, top_coil


    @property
    def xyz(self):
        return tc.moveaxis(tc.stack((self.x,self.y,self.z)), 0, -1)
    
    def min_horizontal_distance(self, rx):
        overall_min = tc.inf
        for i in range(len(self)-1):
            v1x,v1y = rx.x-self.x[i], rx.y-self.y[i]
            v2x,v2y = self.x[i+1]-self.x[i], self.y[i+1]-self.y[i]
            v2r = v2x**2 + v2y**2
            line_dist = v1x*v2x + v1y*v2y
            
            # conditional to check to ensure rx points lies between the two end points.
            if ((0 < line_dist)*(line_dist<v2r)).any():
                numerator = tc.abs(v1x*v2y-v2x*v1y)
                cmin = tc.min(numerator/tc.sqrt(v2r))
                if cmin < overall_min: overall_min = cmin*1
        return tc.tensor(overall_min)
    
    def max_horizontal_distance(self, rx):
        overall_max = 0
        for i in range(len(self)):
            cdist2 = tc.max((self.x[i]-rx.x)**2 + (self.y[i]-rx.y)**2)
            if cdist2 > overall_max: overall_max = cdist2*1
        return tc.sqrt(cdist2)

    def __len__(self):
        return self.x.numel()
    
    def s_of_r(self, rx, R):
        # dR = tc.gradient(R)
        
        # s = tc.zeros_like(R)

        # lp_xyz = self.xyz - rx.xyz[None, ...]

        # for i in range(len(self)-1):
        #     Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
        #     p1 = tc.einsum('...i,...i->...', Rz, lp_xyz[i])
        #     p2 = tc.einsum('...i,...i->...', Rz, lp_xyz[i+1])

        #     r_perp = p1[1]
        #     interval = box_filter(R, tc.abs(r_perp), p2[0])-box_filter(R, tc.abs(r_perp), p1[0])

        #     ds = tc.nan_to_num(interval*R*dR/tc.sqrt(R**2 - r_perp**2))
        #     ds *= tc.abs(p2[0]-p1[0])/tc.abs(tc.nansum(ds))
        #     ds *= r_perp
        #     s +=  ds
        # return tc.nan_to_num(s)
        dR_alpha = tc.sqrt(R[1]/R[0])
        R_edge1 = (R/dR_alpha)**2
        R_edge2 = (R*dR_alpha)**2

        rx_xyz = rx.xyz
        if len(rx_xyz.shape)==1:
            s = tc.zeros(R.shape, dtype=torch.complex128)
        else:
            s = tc.zeros(rx_xyz.shape[:-1]+R.shape, dtype=torch.complex128)

        lp_xyz = self.xyz[:, None] - rx_xyz[None, :]
        for i in range(len(self)-1):
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
            p1 = tc.einsum('...i,...i->...', Rz, lp_xyz[i])
            p2 = tc.einsum('...i,...i->...', Rz, lp_xyz[i+1])
            y2 = p1[1]**2

            x1,x2 = min(tc.abs(p1[0]), tc.abs(p2[0])), max(tc.abs(p1[0]), tc.abs(p2[0]))
            r1 = x1**2 + y2
            r2 = x2**2 + y2

            ds = tc.nan_to_num(tc.sqrt(R_edge2 - y2)) - tc.nan_to_num(tc.sqrt(R_edge1 - y2))

            ds1,ds2 = ds*1., ds*1.
            
            i_s = tc.searchsorted(R_edge1, r1)-1
            ds1[i_s] = tc.nan_to_num(tc.sqrt(r1 -y2)) - tc.nan_to_num(tc.sqrt(R_edge1[i_s] -y2))
            
            ds1[i_s+1:] = 0

            i_s = tc.searchsorted(R_edge1, r2)-1
            ds2[i_s] = tc.nan_to_num(tc.sqrt(r2 -y2)) - tc.nan_to_num(tc.sqrt(R_edge1[i_s] -y2))
            ds2[i_s+1:] = 0
            
            sign = tc.sign(p1[0])*tc.sign(p2[0])
            # if tc.sign(p1[0])==tc.sign(p2[0]): multiplier = 1
            # else: multiplier = 2
            
            ds = tc.nan_to_num(ds2) - tc.nan_to_num(sign*ds1)
            ds *= tc.abs(p2[0]-p1[0])/tc.nansum(ds)
            
            s += p1[1]*ds/R

        return s        
    
def perp_distance_to_line(x,y, line_points):
	((x1,y1), (x2,y2)) = line_points
	numerator = tc.abs((x2-x1)*(y1-y) - (x1-x)*(y2-y1))
	denominator = tc.sqrt((x2-x1)**2 + (y2-y1)**2)
	return numerator / denominator


def box_filter(r, r_min, r_par_max):
    r_max = tc.sqrt(r_min**2 + r_par_max**2)
    y = tc.ones_like(r)
    y[r<r_min]=0
    y[r>r_max]=0
    # if tc.all(y==0):
    #      ir = tc.searchsorted(r, r_min)
    #      y[ir] = 0.5
    #      y[ir+1] = 0.5
    y *= tc.sign(r_par_max)
    return y

def norm(tensor, axis=-1):
    return torch.sqrt(torch.sum(tensor**2, axis=axis))


def getRz(P,Q):
    QP = Q-P
    mQP = norm(QP)
    cos_theta = QP[..., 0] / mQP
    sin_theta = QP[..., 1] / mQP

    Rz = torch.zeros(QP.shape+(3,), dtype=torch.float64, device=P.device)

    Rz[..., 0,0] = cos_theta
    Rz[..., 1,1] = cos_theta
    Rz[..., 0,1] = sin_theta
    Rz[..., 1,0] = -sin_theta
    Rz[..., 2,2] = 1.
    
    return Rz, torch.swapaxes(Rz, -1, -2)


def integrate_around_loop(lp, rx, hr):
    """ this works, but it is not proper. 
        there is a little bit of physics embedded here, and it should only be geometry...
    """

    if 'radius' in dir(lp):
        return 2*c.pi*lp.radius*hr(lp.radius)
    else:

        lp_xyz = lp.xyz - rx.xyz[None, ...]
        
        HR = torch.zeros_like(hr(0.))
        n_dipoles = 0
        wire_length=0
        for i in range(len(lp)-1):
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
            p1 = torch.einsum('...i,...i->...', Rz, lp_xyz[i])
            p2 = torch.einsum('...i,...i->...', Rz, lp_xyz[i+1])

            p12 = p2-p1
            tdist = norm(p12)
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
                wire_length+=ds
                resp = hr(cR)
                HR += -1*ds*cp[1]*resp/cR
                cp += ds*p12
                n_dipoles +=1
        return HR

from ..transforms import Hankel, invFourier, CubicSpline, invLaplace
from ..physics import refl_transverse_electric as rte, freespace_u0 as u0
from .. import constants as c
import torch

class Waveform:
    def __init__(self, time, amplitude):
        self.time      = time      if isinstance(time, torch.Tensor)      else torch.tensor(time)
        self.amplitude = amplitude if isinstance(amplitude, torch.Tensor) else torch.tensor(amplitude)
        self.calc_d2wdt2()

    def calc_d2wdt2(self):
        self.d2wdt2       = torch.zeros_like(self.time)
        slope             = torch.diff(self.amplitude)/torch.diff(self.time)
        self.d2wdt2[1:-1] = torch.diff(slope)
        self.d2wdt2[0]    = slope[0]
        self.d2wdt2[-1]   = -slope[-1]
        self.d2wdt2 = self.d2wdt2.type(torch.DoubleTensor)

    def fourier_d2wdt2(self, f):
        FT = torch.exp(2j*torch.pi*f[..., :, None]*self.time[..., None, :])*self.d2wdt2[..., None, :]
        return FT.sum(axis=-1)

    @classmethod
    def example(cls):
        t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06], dtype=torch.double)
        wv = torch.tensor([0, 1, 1, 0], dtype=torch.double)
        return cls(t, wv)

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
        if self.device is not None:
            for attr_name in self.__dict__:
                attr_value = getattr(self, attr_name)
                if torch.is_tensor(attr_value):
                    setattr(self, attr_name, attr_value.to(self.device))



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
#            tshp = t_eval.shape
#           t_eval = t_eval.reshape(-1)
            resp_step = cbft(t_eval).real#.reshape(tshp)
            dbdt = resp_step @ self.waveform.d2wdt2
            if self.t_gatewidth is not None:
                cbft = CubicSpline(t, dbdt, transform=torch.log10)
            else:
                return dbdt
        if self.t_gatewidth is not None:
            return cbft.integrate(self.t_st, self.t_end, normalize=True)
        else:
            return cbft(self.times).T

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
        if isinstance(sigma, np.ndarray): sigma = torch.from_numpy(sigma)
        if isinstance(thickness, np.ndarray): thickness = torch.from_numpy(thickness)
        self.sigma = ((sigma+0j)[..., None, None]).to(self.device)
        if sigma.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward().real*c.mu0

class DualMoment:
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

    def __call__(self, sigma, thickness):
        self.sigma = ((sigma+0j)[..., None, None]).to(self.device)
        if sigma.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward().real*c.mu0
    

class ButterworthFilter:
    def __init__(self, fcut, order=1):
        self.fcut = fcut
        self.order = order

    def omega_filter(self, omega):
        return self.__call__(omega/c.twopi)
    
    def __call__(self, f):
        s = 1j*f/self.fcut
        if   self.order==1: recip_resp = (s+1)
        elif self.order==2: recip_resp = (s**2 + s*2**(1/2) + 1)
        elif self.order==3: recip_resp = (s**2 + s + 1)*(1+s)
        elif self.order==4: recip_resp = (s**2 + 1.84776*s+1)*(s**2 + 0.76537*s+1)
        return 1./recip_resp
    
class FilterChain:
    """filter based on frequency in Hz (f), not radians/s (Omega)"""
    def __init__(self, filters):
        self.filters = filters
    
    def omega_filter(self, omega):
        return self.__call__(omega/c.twopi)
    
    def __call__(self, f):
        resp = self.filters[0](f)
        for i in range(1, len(self.filters)):
            resp *= self.filters[i](f)
        return resp
    
    def __len__(self):
        return len(self.filters)
    
    def __iter__(self):
        return iter(self.filters)

def compute_relstd(t_gates, data_val, noise_slope_exponent=-0.5, noise_level_at_1ms=1e-8, moment=1, relative_noise=0.03):
    b = noise_level_at_1ms/(1e-3**noise_slope_exponent)
    abs_noise = b*(t_gates**noise_slope_exponent)
    y = data_val*moment
    std_rel = abs_noise/y
    relstd_tot = torch.sqrt(std_rel**2 + relative_noise**2)
    return relstd_tot.to(data_val.device)

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
            self.tmin += -self.waveform.time.max()
            self.tmax += -self.waveform.time.min()
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            # defaults to 10 samples per decade...
            self.t_eval = torch.logspace(torch.log10(self.tmin), torch.log10(self.tmax), int(6*np.log10(self.tmax/self.tmin)), dtype=torch.double)

        else:
            self.t_eval = self.times
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
            return cbft(self.t_eval)

        hkl1 = Hankel(wvlngth_integrand, 1, range=(self.rmin, self.rmax), device=self.device, tolerance=self.tolerance)
        res = integrate_around_loop(self.source, self.receiver, hkl1)
        cbft=CubicSpline(self.t_eval, res, transform=torch.log10, integrate=False)

        if self.t_gatewidth is None:
            t = self.times

        if self.waveform is not None:
            t_eval = t[:, None] - self.waveform.time[None, :]
            # tshp = t_eval.shape
            # t_eval = t_eval.reshape(-1)

            resp_step = cbft(t_eval).real#.reshape(tshp)
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
        self.sigma = ((sigma+0j)[..., None, None]).to(self.device)
        if sigma.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward().real*c.mu0

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
            self.tmin += -self.waveform.time.max()
            self.tmax += -self.waveform.time.min()
            waveform.time = waveform.time.to(self.device)
            waveform.amplitude = waveform.amplitude.to(self.device)
            waveform.d2wdt2 = waveform.d2wdt2.to(self.device)
            # defaults to 10 samples per decade...
            self.t_eval = torch.logspace(torch.log10(self.tmin), torch.log10(self.tmax), int(6*np.log10(self.tmax/self.tmin)), dtype=torch.double)

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

    def __call__(self, sigma, thickness):
        if isinstance(sigma, np.ndarray): sigma = torch.from_numpy(sigma)
        if isinstance(thickness, np.ndarray): thickness = torch.from_numpy(thickness)
        self.sigma = ((sigma+0j)[..., None, None]).to(self.device)
        if sigma.ndim > thickness.ndim:
            thickness = thickness[:, None]
        self.thick = ((thickness+0j)[..., None, None]).to(self.device)
        return self.forward().real*c.mu0
