import numpy as np
import torch
from anemone import constants as c, rdtype, cdtype
from nosj import nosj


S1 = lambda x: 1 - (9/4)*x**2 * ( 1 - 2./9*x*(1 + 5/2*x*(1 - 2/5 * x)))
S2 = lambda x: -1/2*x*(1 - 11/6*x*(1 - 2/11*x*(1 + 5/2*x*(1-2/5*x))))

def lamontagne(x):
    phi   = x*0.
    xm2m1 = (-2 <= x)*(x < -1)
    xm1m0 = (-1 <= x)*(x < 0)
    x0p1  = (0 <= x)*(x <= 1)
    x1p2  = (1 < x)*(x <= 2)
    
    phi[xm2m1] = S2(-x[xm2m1]-1)
    phi[xm1m0] = S1(-x[xm1m0])
    phi[x0p1] = S1(x[x0p1])
    phi[x1p2] = S2(x[x1p2]-1)
    
    return phi

from torch import Tensor

@nosj
class Receiver: 
    x       : Tensor
    y       : Tensor
    z       : Tensor
    grid    : bool = True

    def __init__(self, x=0., y=0., z=0., grid=False) -> None:
        self.x = torch.tensor(x, dtype=rdtype) if not isinstance(x, Tensor) else x.to(rdtype)
        self.y = torch.tensor(y, dtype=rdtype) if not isinstance(y, Tensor) else y.to(rdtype)
        self.z = torch.tensor(z, dtype=rdtype) if not isinstance(z, Tensor) else z.to(rdtype)
        self.grid = grid
        
        if self.grid:
            self._grid()
        else:
            all_points = [coord.numel() for coord in (self.x, self.y, self.z)]
            npoints = max(all_points)
            self.X = self.x if self.x.numel()!=npoints else torch.ones(npoints)*self.x
            self.Y = self.y if self.y.numel()!=npoints else torch.ones(npoints)*self.y
            self.Z = self.z if self.z.numel()!=npoints else torch.ones(npoints)*self.z

    def _grid(self):
        X,Y,Z = torch.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.shp = X.shape+(3,)
        self.X = X.squeeze()
        self.Y = Y.squeeze()
        self.Z = Z.squeeze()
        return self
    
    @property
    def xyz(self):
        return torch.squeeze(torch.moveaxis(torch.stack((
                            self.X.flatten(), 
                            self.Y.flatten(), 
                            self.Z.flatten())), 0, -1))



    def __len__(self):
        return self.X.numel()

@nosj
class Loop:
    x : Tensor
    y : Tensor
    z : Tensor

    def __init__(self, x, y, z, **kwargs) -> None:
        self.x = torch.tensor(x, dtype=rdtype) if not isinstance(x, Tensor) else x.to(rdtype)
        self.y = torch.tensor(y, dtype=rdtype) if not isinstance(y, Tensor) else y.to(rdtype)
        self.z = torch.tensor(z, dtype=rdtype) if not isinstance(z, Tensor) else z.to(rdtype)
        
        if not torch.all(self.z == self.z[0]): raise NotImplemented('current implementation only supports horizontal transmitter loops; all z values must be equal.')
        r2 = self.x[0]**2 + self.y[0]**2
        if torch.allclose(self.x**2 + self.y**2, r2) and len(self.x)>=20:
            self.area = torch.pi*r2
            self.radius = torch.sqrt(r2)

    @classmethod
    def circle(cls, radius, n=40, **kwargs):
        r = radius
        
        t = torch.linspace(0, 2*torch.pi, n)
        x = r*torch.cos(t)
        y = r*torch.sin(t)
        z = torch.ones_like(x)*kwargs.get('z', 0.)
        lp = cls(x, y, z, **kwargs)
        lp.area = torch.pi*radius**2
        lp.radius = r
        return lp

    @classmethod
    def square(cls, side, **kwargs):
        half = side / 2.
        x = torch.tensor([-half,  half, half, -half, -half], dtype=rdtype)
        y = torch.tensor([-half, -half, half,  half, -half], dtype=rdtype)
        z = torch.ones_like(x, dtype=rdtype)*kwargs.get('z', 0.)
        lp = cls(x, y, z)
        lp.area = side*side
        return lp

    @classmethod
    def fig8(cls, side, **kwargs):
        half = side / 2.
        x = torch.tensor([-side, 0, 0, side, side, 0, 0, -side, -side])
        y = torch.tensor([-half, -half, half, half, -half, -half, half, half, -half])
        z = torch.ones_like(x)*kwargs.get('z', 0.)
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
        return torch.moveaxis(torch.stack((self.x,self.y,self.z)), 0, -1)
    
    def min_horizontal_distance(self, rx):
        overall_min = torch.inf
        for i in range(len(self)-1):
            v1x,v1y = rx.x-self.x[i], rx.y-self.y[i]
            v2x,v2y = self.x[i+1]-self.x[i], self.y[i+1]-self.y[i]
            v2r = v2x**2 + v2y**2
            line_dist = v1x*v2x + v1y*v2y
            
            # conditional to check to ensure rx points lies between the two end points.
            if ((0 <= line_dist)*(line_dist<=v2r)).any():
                numerator = torch.abs(v1x*v2y-v2x*v1y)
                cmin = torch.min(numerator/torch.sqrt(v2r))
                if cmin < overall_min: overall_min = cmin*1
        return overall_min
    
    def max_horizontal_distance(self, rx):
        overall_max = 0
        for i in range(len(self)):
            cdist2 = torch.max((self.x[i]-rx.x)**2 + (self.y[i]-rx.y)**2)
            if cdist2 > overall_max: overall_max = cdist2*1
        return torch.sqrt(overall_max)

    def __len__(self):
        return self.x.numel()
    
    def geomz(self, rx, R):
        if hasattr(self, 'radius') and rx.xyz.numel()==3 and rx.x==0. and rx.y==0.:
            return self._geomz_circle(rx, R)
        logR = torch.log(R).cpu()
        dR = logR[1]-logR[0]

        rx_xyz = rx.xyz
        n_rec  = len(rx)

        geomz      = torch.zeros((n_rec, )+R.shape, dtype=rdtype)

        R      = R [None, None, :]
        
        lp_xyz = self.xyz[:, None] - rx_xyz[None, :]
        
        for i in range(len(self)-1):
        
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
            p1 = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i])
            p2 = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])
            x1,y1 = p1[..., 0], p1[..., 1]
            x2,y2 = p2[..., 0], p2[..., 1]
            ix = int(100*torch.abs(x2-x1).max())
            tmpx = torch.linspace(-1, 1, ix)
            x_samp = tmpx
            x_samp += 1.
            length = torch.abs(x2-x1)/2.


            x_ = (x1[:,None] + length[:,None]*x_samp[None, :])
            dx = (x_[..., 1] - x_[..., 0])[:,None]
            
            x1 = x1[:, None]
            x2 = x2[:, None]
            y1 = y1[:, None, None]
            x_ = x_[:, :, None]
            dx = dx[:, None]


            cr = torch.sqrt(y1**2 + x_**2)
            phi = lamontagne((torch.log(cr)-logR)/dR)
            cs =  -y1 * phi * dx / cr
            geomz += cs.sum(axis=1, keepdims=False)
        return geomz.to(R.device)

    def _geomz_circle(self, rx, R):
        if not isinstance(self.radius, torch.Tensor):
            radius = torch.ones(1, dtype=R.dtype)*self.radius
        else:
            radius = self.radius
        logR = torch.log(R).cpu()
        dR = logR[1]-logR[0]
        phi = lamontagne((torch.log(radius)-logR)/dR)
        geomz = 2*torch.pi*phi*radius
        return geomz.to(R.device)
     
def perp_distance_to_line(x,y, line_points):
	((x1,y1), (x2,y2)) = line_points
	numerator = torch.abs((x2-x1)*(y1-y) - (x1-x)*(y2-y1))
	denominator = torch.sqrt((x2-x1)**2 + (y2-y1)**2)
	return numerator / denominator


def box_filter(r, r_min, r_par_max):
    r_max = torch.sqrt(r_min**2 + r_par_max**2)
    y = torch.ones_like(r)
    y[r<r_min]=0
    y[r>r_max]=0
    # if torch.all(y==0):
    #      ir = torch.searchsorted(r, r_min)
    #      y[ir] = 0.5
    #      y[ir+1] = 0.5
    y *= torch.sign(r_par_max)
    return y

def norm(tensor, axis=-1, keepdim=False):
    return torch.sqrt(torch.sum(tensor**2, axis=axis, keepdim=keepdim))


def getRz(P,Q):
    QP = Q-P
    mQP = norm(QP)
    cos_theta = QP[..., 0] / mQP
    sin_theta = QP[..., 1] / mQP

    Rz = torch.zeros(QP.shape+(3,), dtype=rdtype, device=P.device)

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
        
        
        if rx.xyz.numel()==3:
            lp_xyz = lp.xyz - rx.xyz
            hr_example = hr(0.1)
            HR = torch.zeros_like(hr_example).reshape((1,)+hr_example.shape)
        else:
            lp_xyz = lp.xyz[:, None, :]-rx.xyz[None,:,:]
            hr_example = hr(torch.ones(len(rx.xyz)))
            HR = torch.zeros_like(hr_example)
            

        n_dipoles = 0
        wire_length= torch.zeros(len(rx.xyz))
        for i in range(len(lp)-1):
            Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])

            p1 = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i])
            p2 = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])

            p12 = p2-p1
            tdist = norm(p12, keepdim=True)
            p12 /= tdist
            cp = p1

            tdist = tdist.squeeze()
            cdist = torch.zeros_like(tdist)

            while torch.any(cdist < tdist):
                still_to_go = cdist < tdist
                cR = norm(cp[still_to_go])
                ds = cR/15.
                if torch.any(cdist[still_to_go]+ds > tdist[still_to_go]):
                    go_back = cdist[still_to_go]+ds > tdist[still_to_go]
                    ds[go_back] = (tdist-cdist)[still_to_go][go_back]
                cdist[still_to_go]       += ds
                wire_length[still_to_go] += ds
                resp                      = hr(cR)
                HR[..., still_to_go]     += -1.*ds*cp[still_to_go, 1]*resp/cR
                cp[still_to_go]          += (ds[:, None]*p12[still_to_go])
                n_dipoles                += 1
        return HR.T.squeeze()


@nosj
class Waveform:
    time      : Tensor
    amplitude : Tensor
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
        # self.d2wdt2 = self.d2wdt2.to(rdtype)

    def fourier_d2wdt2(self, f):
        FT = torch.exp(2j*torch.pi*f[..., :, None]*self.time[..., None, :])*self.d2wdt2[..., None, :]
        return FT.sum(axis=-1)

    @classmethod
    def example(cls):
        t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06], dtype=rdtype)
        wv = torch.tensor([0, 1, 1, 0], dtype=rdtype)
        return cls(t, wv)

# @nosj
class SystemResponse:
    time      : Tensor
    dSRdt     : Tensor
    def __init__(self, time, dSRdt):
        np_time = time.numpy()
        dtime = torch.from_numpy(np.diff(np_time, prepend=np_time[0]))
        amplitude = dSRdt * dtime
        super().__init__(time, dSRdt)
        self.d2wdt2 = amplitude
        self.time = time
        self.amplitude = amplitude

@nosj
class ButterworthFilter:
    fcut : float
    order : int = 1

    def omega_filter(self, omega):
        return self.__call__(omega/c.twopi)
    
    def __call__(self, f):
        s = 1j*f/self.fcut
        if   self.order==1: recip_resp = (s+1)
        elif self.order==2: recip_resp = (s**2 + s*2**(1/2) + 1)
        elif self.order==3: recip_resp = (s**2 + s + 1)*(1+s)
        elif self.order==4: recip_resp = (s**2 + 1.84776*s+1)*(s**2 + 0.76537*s+1)
        return 1./recip_resp

from typing import List

@nosj
class FilterChain:
    filters : List[ButterworthFilter]
    
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
    

if __name__ == "__main__":
    f1 = ButterworthFilter(1000., order=2)
    f2 = ButterworthFilter(5000., order=1)
    fc = FilterChain([f1, f2])

    fc.save('test_filterchain.json')
    fc2 = FilterChain.load('test_filterchain.json', load_subclasses=True)
    print(fc)
    print(fc2)
    assert len(fc)==len(fc2)
    print('is fc and fc2 equal?', fc==fc2)