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
            print(cR)
            ds = cR/25
            if cdist+ds > tdist:
                ds = tdist-cdist
                cdist += cR/25
            else:
                cdist += ds
            wire_length+=ds
            resp = hr(cR)
            HR += -1*ds*cp[1]*resp/cR
            cp += ds*p12
            n_dipoles +=1
    print('number of dipoles: ', n_dipoles)
    print('wire length: ', wire_length)
    return HR

from ..transforms import Hankel, Fourier, CubicSpline
from ..physics import refl_transverse_electric as rte, freespace_u0 as u0
import torch

class Forward:
    def __init__(self, source, receiver, times=tc.logspace(1e-5, 1e-2, 31)):
        self.source = source
        self.receiver = receiver
        self.times = torch.tensor(times)
        self.lgtimes = torch.log(self.times)

        self.rmin = self.source.min_horizontal_distance(self.receiver)
        self.rmax = self.source.max_horizontal_distance(self.receiver)
        self.r = precalcHankel1.range(self.rmin, self.rmax)[0].numpy()
        self.sofr = torch.tensor(self.source.s_of_r(self.receiver, self.r), dtype=torch.complex128)
        self.dz = torch.tensor(self.receiver.z+self.source.z[0])# tentative solution

    def forward(self):
        def frq_integrand(omega):
            omega = omega[None,:,None]
            def integrand(wave_length):
                lmb = wave_length[None, None, :]
                tmp_u0 = u0(omega, lmb)
                intg = (torch.exp(-tmp_u0*self.dz)*rte(self.sigma, self.thick, omega, lmb))*lmb**2 / tmp_u0
                return intg
            hkl = Hankel(integrand, 1)
            r, hr = hkl.range(self.rmin, self.rmax)
            hr /= r[:,None,None]
            return hr.T@self.sofr
        f_t=Fourier(frq_integrand)
        t,dbdt = f_t.range(self.times[0], self.times[-1])
        cbs = CubicSpline(t, dbdt)
        return cbs(self.times)


    def __call__(self, sigma, thickness):
        self.sigma = torch.tensor(sigma, dtype=torch.complex128)[...,None,None]
        self.thick = torch.tensor(thickness, dtype=torch.complex128)[...,None,None]
        return self.forward()