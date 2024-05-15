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
        self.x = tc.squeeze(X)
        self.y = tc.squeeze(Y)
        self.z = tc.squeeze(Z)
        return self
    
    @property
    def xyz(self):
        return tc.squeeze(tc.moveaxis(tc.stack((self.x, self.y, self.z)), 0, -1))

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
        return overall_min
    
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

        # lp_xyz = self.xyz - xyz[None, ...]

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

        
        s = tc.zeros_like(R, dtype=torch.complex128)

        lp_xyz = self.xyz - rx.xyz[None, :]
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

from ..transforms import Hankel, invFourier, CubicSpline
from ..physics import refl_transverse_electric as rte, freespace_u0 as u0
from .. import constants as c
import torch

class Waveform:
    def __init__(self, time, amplitude):
        self.time      = time      if isinstance(time, torch.Tensor)      else torch.tensor(time)
        self.amplitude = amplitude if isinstance(amplitude, torch.Tensor) else torch.tensor(amplitude)
        self.calc_d2wdt2()

    def calc_d2wdt2(self):
        self.d2wdt2 = torch.zeros_like(self.time)
        slope = torch.diff(self.amplitude)/torch.diff(self.time)
        self.d2wdt2[1:-1] = torch.diff(slope)
        self.d2wdt2[0] = slope[0]
        self.d2wdt2[-1] = -slope[-1]

    @classmethod
    def example(cls):
        t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
        wv = torch.tensor([0, 1, 1, 0])
        cls(t, wv)

class Forward:
    def __init__(self, source, receiver, times, waveform=None, filterfunc=None, t_gatewidth=None):
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.dz = torch.tensor(self.receiver.z+self.source.z[0], device=times.device)
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


    def forward(self):
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                lmb = wave_length[None, None, :]
                tmp_u0 = u0(omega, lmb)
                if self.waveform is not None:
                    tmp_u0 *= -1j*omega
                intg = torch.exp(-tmp_u0*self.dz)*rte(self.sigma, self.thick, omega, lmb)*lmb**2 / tmp_u0
                if self.filterfunc is not None:
                    intg *= self.filterfunc.omega_filter(omega)
                return torch.squeeze(intg/c.fourpi)
            
            hkl1 = Hankel(wvlngth_integrand, 1, smp_per_decade=10, range=(self.rmin, self.rmax), device=self.device)
            return integrate_around_loop(self.source, self.receiver, hkl1)
        
        ift=invFourier(frq_integrand, smp_per_decade=16, device=self.device)
        t,dbdt=ift.range(self.tmin, self.tmax)
        cbft = CubicSpline(t,dbdt, transform=torch.log10)

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


    def __call__(self, sigma, thickness):
        self.sigma = torch.tensor(sigma,     device=self.device, dtype=torch.complex128)[...,None,None]
        self.thick = torch.tensor(thickness, device=self.device, dtype=torch.complex128)[...,None,None]
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