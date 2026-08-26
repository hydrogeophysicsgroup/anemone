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
    weighting  : Tensor = None

    def __init__(self, x=0., y=0., z=0., grid=False, weighting=None) -> None:
        self.x = torch.tensor(x, dtype=rdtype()) if not isinstance(x, Tensor) else x.to(rdtype())
        self.y = torch.tensor(y, dtype=rdtype()) if not isinstance(y, Tensor) else y.to(rdtype())
        self.z = torch.tensor(z, dtype=rdtype()) if not isinstance(z, Tensor) else z.to(rdtype())
        self.grid = grid
        self.weighting = torch.tensor(weighting, dtype=rdtype()) if weighting is not None else None

        if self.grid:
            self._grid()
        else:
            all_points = [coord.numel() for coord in (self.x, self.y, self.z)]
            npoints = max(all_points)

            self.X = self.x if self.x.numel()==npoints else torch.ones(npoints)*self.x
            self.Y = self.y if self.y.numel()==npoints else torch.ones(npoints)*self.y
            self.Z = self.z if self.z.numel()==npoints else torch.ones(npoints)*self.z

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
        self.x = torch.tensor(x, dtype=rdtype()) if not isinstance(x, Tensor) else x.to(rdtype())
        self.y = torch.tensor(y, dtype=rdtype()) if not isinstance(y, Tensor) else y.to(rdtype())
        self.z = torch.tensor(z, dtype=rdtype()) if not isinstance(z, Tensor) else z.to(rdtype())
        
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
        z = torch.ones_like(x)*kwargs.pop('z', 0.)
        lp = cls(x, y, z, **kwargs)
        lp.area = torch.pi*radius**2
        lp.radius = r
        return lp

    @classmethod
    def square(cls, side, **kwargs):
        half = side / 2.
        x = torch.tensor([-half,  half, half, -half, -half], dtype=rdtype())
        y = torch.tensor([-half, -half, half,  half, -half], dtype=rdtype())
        z = torch.ones_like(x, dtype=rdtype())*kwargs.pop('z', 0.)
        lp = cls(x, y, z)
        lp.area = side*side
        return lp

    @classmethod
    def fig8(cls, side, **kwargs):
        half = side / 2.
        x = torch.tensor([-side, 0, 0, side, side, 0, 0, -side, -side])
        y = torch.tensor([-half, -half, half, half, -half, -half, half, half, -half])
        z = torch.ones_like(x)*kwargs.pop('z', 0.)
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
        if hasattr(self, 'radius'):
            return self._geomz_circle(rx, R)        
        from numpy.polynomial.legendre import leggauss

        R = R.cpu().numpy()

        logR = np.log(R)
        dR   = float(logR[1] - logR[0])
        n_R  = len(R)

        rx_xyz = rx.xyz.numpy()
        n_rec  = len(rx)

        geomz_out = np.zeros((n_rec, n_R))

        lp_xyz = self.xyz.numpy()[:, None] - rx_xyz[None, :]  # (n_pts, n_rec, 3)

        # 3-pt GL is exact for degree ≤ 5 polynomials; phi is piecewise degree-5.
        # Splitting at each phi piece boundary means the polynomial part is
        # integrated exactly and only the smooth ln(cosh) composition incurs error.
        gp, gw = leggauss(3)  # (3,)
        u_breaks = np.array([-2., -1., 0., 1., 2.])

        for i in range(len(self) - 1):
            Rz, _ = getRz(lp_xyz[i], lp_xyz[i+1])
            p1    = np.einsum('...ij,...j->...i', Rz, lp_xyz[i])    # (n_rec, 3)
            p2    = np.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])  # (n_rec, 3)
            x1, y1 = p1[..., 0].flatten(), p1[..., 1].flatten()     # (n_rec,)
            x2     = p2[..., 0].flatten()                            # (n_rec,)

            # Skip receivers closer than 1 cm to the wire (y1 is perp. distance)
            safe = np.abs(y1) > 1e-3
            if not safe.any():
                continue

            y1_s    = y1[safe];  abs_y1_s = np.abs(y1_s)  # (n_safe,)
            x1_s    = x1[safe];  x2_s     = x2[safe]

            # Change of variable  x = y1·sinh(α)
            #   r = |y1|·cosh(α),   dx = y1·cosh(α) dα
            #   ∫ −y1·phi/r dx  =  ∫ −y1²/|y1|·phi dα  =  ∫ −|y1|·phi dα
            a1 = np.arcsinh(x1_s / y1_s)  # (n_safe,)
            a2 = np.arcsinh(x2_s / y1_s)  # (n_safe,)

            # beta[j,k] = (log|y1_j| − logR_k) / dR
            beta = (np.log(abs_y1_s)[:, None] - logR[None, :]) / dR  # (n_safe, n_R)

            # Alpha breakpoints for phi piece boundaries, per (receiver, R).
            # alpha_break corresponds to u_break via cosh(alpha) = exp((u−beta)·dR).
            # Set to 0 (lower bound of positive-alpha domain) when breakpoint < 0.
            with np.errstate(over='ignore'):
                cosh_b = np.exp(np.minimum(
                    (u_breaks[None, None, :] - beta[:, :, None]) * dR, 500.
                ))  # (n_safe, n_R, 5)
            alpha_breaks = np.where(
                cosh_b >= 1., np.arccosh(np.maximum(cosh_b, 1.)), 0.
            )  # (n_safe, n_R, 5)

            def F_pos(a_pos):
                """∫_0^{a_pos} −|y1|·phi(beta + ln(cosh α)/dR) dα → (n_safe, n_R)"""
                out = np.zeros((int(safe.sum()), n_R))
                cap = a_pos[:, None]  # (n_safe, 1)
                for k in range(4):
                    lo = np.minimum(alpha_breaks[:, :, k  ], cap)
                    hi = np.minimum(alpha_breaks[:, :, k+1], cap)
                    if not (lo < hi).any():
                        continue
                    ah = np.where(lo < hi, (hi - lo) / 2., 0.)  # (n_safe, n_R)
                    am = np.where(lo < hi, (hi + lo) / 2., 0.)  # (n_safe, n_R)
                    alpha_gl = am[:, :, None] + ah[:, :, None] * gp[None, None, :]
                    r_gl     = abs_y1_s[:, None, None] * np.cosh(alpha_gl)
                    u_gl     = (np.log(r_gl) - logR[None, :, None]) / dR
                    # ah=0 for inactive cells → zero contribution without explicit mask
                    out += (-abs_y1_s[:, None]
                            * (lamontagne(u_gl) * ah[:, :, None] * gw[None, None, :]).sum(-1))
                return out

            # Fundamental theorem: ∫_{a1}^{a2} = sgn(a2)·F_pos(|a2|) − sgn(a1)·F_pos(|a1|)
            contrib = (  np.sign(a2)[:, None] * F_pos(np.abs(a2))
                       - np.sign(a1)[:, None] * F_pos(np.abs(a1)))
            geomz_out[safe] += contrib
            
        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz_out = np.einsum('i,ij...->j...', rx.weighting, geomz_out)
        return torch.from_numpy(geomz_out)

    def _geomz_circle(self, rx, R):
        radius = self.radius if isinstance(self.radius, torch.Tensor) else torch.tensor(self.radius, dtype=R.dtype)

        d = torch.sqrt(rx.X**2 + rx.Y**2).flatten()   # (n_rec,)

        logR = torch.log(R).cpu()
        dR = logR[1] - logR[0]

        N = 1000
        theta = torch.linspace(0, 2 * torch.pi, N + 1, dtype=rdtype())[:-1]  # (N,)
        dtheta = 2 * torch.pi / N

        # r: (n_rec, N)
        r = torch.sqrt(radius**2 + d[:, None]**2 - 2 * radius * d[:, None] * torch.cos(theta[None, :]))
        integrand_factor = radius * (radius - d[:, None] * torch.cos(theta[None, :])) / r  # (n_rec, N)

        log_r = torch.log(r)  # (n_rec, N)
        # phi: (n_rec, N, n_R)
        phi = lamontagne((log_r[:, :, None] - logR[None, None, :]) / dR)

        # geomz: (n_rec, n_R)
        geomz = (integrand_factor[:, :, None] * phi * dtheta).sum(axis=1)

        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz = torch.einsum('i,ij->j', rx.weighting, geomz)

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

def norm(tensor, axis=-1, keepdims=False):
    if isinstance(tensor, np.ndarray):
        return np.sum(tensor**2, axis=axis, keepdims=keepdims)**0.5
    else: 
        return torch.sum(tensor**2, axis=axis, keepdim=keepdims)**0.5



def getRz(P,Q):
    QP = Q-P
    mQP = norm(QP)
    cos_theta = QP[..., 0] / mQP
    sin_theta = QP[..., 1] / mQP

    Rz = np.zeros(QP.shape+(3,))

    Rz[..., 0,0] = cos_theta
    Rz[..., 1,1] = cos_theta
    Rz[..., 0,1] = sin_theta
    Rz[..., 1,0] = -sin_theta
    Rz[..., 2,2] = 1.
    
    return Rz, np.swapaxes(Rz, -1, -2)


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
        # self.d2wdt2 = self.d2wdt2.to(rdtype())

    def fourier_d2wdt2(self, f):
        FT = torch.exp(2j*torch.pi*f[..., :, None]*self.time[..., None, :])*self.d2wdt2[..., None, :]
        return FT.sum(axis=-1)

    @classmethod
    def example(cls):
        t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06], dtype=rdtype())
        wv = torch.tensor([0, 1, 1, 0], dtype=rdtype())
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

    @property
    def tau(self):
        return 1./(c.twopi*self.fcut)

    def omega_filter(self, omega):
        return self.__call__(omega/c.twopi)
    
    def __call__(self, f):
        s = 1j*f/self.fcut
        if   self.order==1: recip_resp = (s+1)
        elif self.order==2: recip_resp = (s**2 + s*2**(1/2) + 1)
        elif self.order==3: recip_resp = (s**2 + s + 1)*(1+s)
        elif self.order==4: recip_resp = (s**2 + 1.84776*s+1)*(s**2 + 0.76537*s+1)
        return (1./recip_resp)
    
    def stepoff_response(self, t):
        amp = torch.exp(-t/self.tau)
        amp[t<0.] = 1.
        return amp**self.order
    

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
    
    def stepoff_response(self, t):
        N = len(self)
        omegas = c.twopi*torch.tensor([cfilter.fcut for cfilter in self.filters])
        if len(np.unique(omegas.numpy())) != N:
            raise ValueError("All cutoff frequencies in 'omegas' must be distinct.")
        
        # Initialize the summation term
        summation_term = torch.zeros_like(t, dtype=float)
        summation_term[t<=0.] = 1.
        i_step = t>0
        
        
        for i in range(N):
            w_i = omegas[i]
            
            # Isolate all w_j where j != i
            w_j = omegas[omegas!=w_i]
            
            # Calculate the product coefficient for this specific filter
            C_i = torch.prod(w_j / (w_j - w_i))
            
            # Add this filter's exponential decay curve to the total summation
            summation_term[i_step] += C_i * torch.exp(-w_i * t[i_step])
        return summation_term

    

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