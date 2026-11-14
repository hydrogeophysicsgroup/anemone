import numpy as np
import torch
from anemone import constants as c, rdtype, cdtype
from nosj import nosj



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

        R = R.cpu()#.numpy()

        logR = torch.log(R)
        dR   = float(logR[1] - logR[0])
        n_R  = len(R)

        rx_xyz = rx.xyz
        n_rec  = len(rx)

        geomz_out = torch.zeros((n_rec, n_R))

        lp_xyz = self.xyz[:, None] - rx_xyz[None, :]  # (n_pts, n_rec, 3)

        # 3-pt GL is exact for degree ≤ 5 polynomials; phi is piecewise degree-5.
        # Splitting at each phi piece boundary means the polynomial part is
        # integrated exactly and only the smooth ln(cosh) composition incurs error.
        gp, gw = leggauss(3)  # (3,)
        u_breaks = torch.tensor([-2., -1., 0., 1., 2.])

        for i in range(len(self) - 1):
            Rz, _ = getRz(lp_xyz[i], lp_xyz[i+1])
            p1    = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i])    # (n_rec, 3)
            p2    = torch.einsum('...ij,...j->...i', Rz, lp_xyz[i+1])  # (n_rec, 3)
            x1, y1 = p1[..., 0].flatten(), p1[..., 1].flatten()     # (n_rec,)
            x2     = p2[..., 0].flatten()                            # (n_rec,)

            # Skip receivers closer than 1 cm to the wire (y1 is perp. distance)
            safe = torch.abs(y1) > 1e-3
            if not safe.any():
                continue

            y1_s    = y1[safe];  abs_y1_s = torch.abs(y1_s)  # (n_safe,)
            x1_s    = x1[safe];  x2_s     = x2[safe]

            # Change of variable  x = y1·sinh(α)
            #   r = |y1|·cosh(α),   dx = y1·cosh(α) dα
            #   ∫ −y1·phi/r dx  =  ∫ −y1²/|y1|·phi dα  =  ∫ −|y1|·phi dα
            a1 = torch.arcsinh(x1_s / y1_s)  # (n_safe,)
            a2 = torch.arcsinh(x2_s / y1_s)  # (n_safe,)

            # beta[j,k] = (log|y1_j| − logR_k) / dR
            beta = (torch.log(abs_y1_s)[:, None] - logR[None, :]) / dR  # (n_safe, n_R)

            # Alpha breakpoints for phi piece boundaries, per (receiver, R).
            # alpha_break corresponds to u_break via cosh(alpha) = exp((u−beta)·dR).
            # Set to 0 (lower bound of positive-alpha domain) when breakpoint < 0.
            # with np.errstate(over='ignore'):
            cosh_b = torch.exp(torch.minimum(
                (u_breaks[None, None, :] - beta[:, :, None]) * dR, torch.tensor([500.])
            ))  # (n_safe, n_R, 5)

            alpha_breaks = torch.where(
                cosh_b >= 1., torch.arccosh(torch.maximum(cosh_b, torch.ones(1))), 0.
            )  # (n_safe, n_R, 5)

            def F_pos(a_pos):
                """∫_0^{a_pos} −|y1|·phi(beta + ln(cosh α)/dR) dα → (n_safe, n_R)"""
                out = torch.zeros((int(safe.sum()), n_R))
                cap = a_pos[:, None]  # (n_safe, 1)
                for k in range(4):
                    lo = torch.minimum(alpha_breaks[:, :, k  ], cap)
                    hi = torch.minimum(alpha_breaks[:, :, k+1], cap)
                    if not (lo < hi).any():
                        continue
                    ah = torch.where(lo < hi, (hi - lo) / 2., 0.)  # (n_safe, n_R)
                    am = torch.where(lo < hi, (hi + lo) / 2., 0.)  # (n_safe, n_R)
                    alpha_gl = am[:, :, None] + ah[:, :, None] * gp[None, None, :]
                    r_gl     = abs_y1_s[:, None, None] * torch.cosh(alpha_gl)
                    u_gl     = (torch.log(r_gl) - logR[None, :, None]) / dR
                    # ah=0 for inactive cells → zero contribution without explicit mask
                    out += (-abs_y1_s[:, None]
                            * (lamontagne(u_gl) * ah[:, :, None] * gw[None, None, :]).sum(-1))
                return out

            # Fundamental theorem: ∫_{a1}^{a2} = sgn(a2)·F_pos(|a2|) − sgn(a1)·F_pos(|a1|)
            contrib = (  torch.sign(a2)[:, None] * F_pos(torch.abs(a2))
                       - torch.sign(a1)[:, None] * F_pos(torch.abs(a1)))
            geomz_out[safe] += contrib
            
        if hasattr(rx, 'weighting') and rx.weighting is not None:
            geomz_out = torch.einsum('i,ij...->j...', rx.weighting, geomz_out)
        return geomz_out

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

    Rz = torch.zeros(QP.shape+(3,))

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
            tdist = norm(p12, keepdims=True)
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

def allpole_stepoff_response(poles, t):
    """Step-off response of an all-pole low-pass filter with unit DC gain.

    The filter is ``H(s) = prod(-p_k) / prod(s - p_k)`` with ``poles`` p_k in
    rad/s (Re(p_k) < 0). The input is 1 for t < 0 and 0 for t >= 0, so the
    response is 1 for t <= 0 and ``1 - L^-1[H(s)/s](t)`` for t > 0.

    The partial fractions are expanded with multiplicity, so repeated poles
    (e.g. two identical first-order filters in a chain) and complex pole
    pairs (Butterworth filters of order >= 2) are both handled.
    """
    from scipy.signal import residue
    poles = np.atleast_1d(np.asarray(poles, dtype=complex))
    w0    = np.exp(np.mean(np.log(np.abs(poles))))   # normalise, so the poles are O(1)
    a     = np.real(np.poly(poles/w0))
    # step-on response: Y(s) = H(s)/s, which adds a pole at s=0
    r, p, _ = residue([a[-1]], np.append(a, 0.), tol=1e-6)

    amp = torch.ones_like(t, dtype=rdtype())
    i_step = t > 0
    tau = (t[i_step]*w0).to(cdtype())
    step_on = torch.zeros_like(tau)
    m = 0
    for i in range(len(r)):
        # residue() lists a pole of multiplicity M as M consecutive entries,
        # for (s-p)^-1, (s-p)^-2, ..., (s-p)^-M
        m = m+1 if (i > 0 and np.isclose(p[i], p[i-1], rtol=0., atol=1e-12)) else 1
        step_on += complex(r[i]) * tau**(m-1) / float(np.prod(np.arange(1, m))) * torch.exp(complex(p[i])*tau)
    amp[i_step] = 1. - step_on.real.to(amp.dtype)
    return amp


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
        else: raise NotImplementedError(f"filter of order {self.order} has not been implemented")
        return (1./recip_resp)
    
    def poles(self):
        """Poles of the filter in rad/s."""
        n = int(self.order)
        k = np.arange(n)
        return c.twopi*self.fcut*np.exp(1j*np.pi*(2*k + n + 1)/(2*n))

    def stepoff_response(self, t):
        return allpole_stepoff_response(self.poles(), t)
    

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
        if N==1: return self.filters[0].stepoff_response(t)
        # The chain is the product of the filters, so its poles are the union
        # of the filters' poles (with multiplicity). Equal cutoff frequencies
        # give repeated poles, which allpole_stepoff_response handles.
        return allpole_stepoff_response(np.concatenate([np.atleast_1d(cfilter.poles()) for cfilter in self.filters]), t)


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

def waveform_matrix(t_eval, t_sampled, waveform=None):
    t_eval = t_eval.to(rdtype())
    t_sampled = t_sampled.to(rdtype())

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

def gating_matrix(t_sampled, t_st, t_end, n_eval=10000):
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

def make_H2t(t2t, geomz, omega, lmb, tolerance=1e-6, sigma_ref=1e-2):
    import numpy as np

    if geomz.ndim > 1:
        n_g = geomz.shape[0]
    else:
        n_g = 1
        geomz = geomz[None, :]

    t2t_real = t2t.real.cpu().numpy()
    t2t_np   = t2t.cpu().numpy()
    geomz_np = geomz.real.cpu().numpy()
    omega_np = omega.squeeze().cpu().numpy()
    lmb_np   = lmb.squeeze().cpu().numpy()
    n_t      = t2t.shape[0]
    n_omega  = omega_np.shape[0]
    n_lmb    = lmb_np.shape[0]
    n_ol     = n_omega * n_lmb

    from anemone.physics import rte_halfspace
    rte_ref = rte_halfspace(torch.tensor(sigma_ref), omega[:, None], lmb[None, :]).squeeze().cpu().numpy()

    if tolerance is not None:
        full_mask  = np.zeros((n_omega, n_lmb), dtype=bool)
        row_arange = np.arange(n_t)

        for i_g in range(n_g):
            t_g    = np.abs(t2t_real[:, :, np.newaxis] *
                            geomz_np[i_g][np.newaxis, np.newaxis, :] *
                            rte_ref[np.newaxis, :, :])
            t_flat   = t_g.reshape(n_t, n_ol)
            # sort values only — no argsort needed since threshold is applied by value
            t_sorted = np.sort(t_flat, axis=1)
            t_cumsum = np.cumsum(t_sorted, axis=1)
            t_cumsum /= t_cumsum[:, -1:]

            threshold_idx = (t_cumsum < tolerance).sum(axis=1) + 1   # (n_t,)
            safe_idx      = np.minimum(threshold_idx, n_ol - 1)
            threshold_val = t_sorted[row_arange, safe_idx]            # (n_t,)
            threshold_val[threshold_idx >= n_ol] = np.inf

            mask = t_flat >= threshold_val[:, np.newaxis]             # (n_t, n_ol)
            full_mask |= mask.any(axis=0).reshape(n_omega, n_lmb)
    else:
        full_mask = np.ones((n_omega, n_lmb), dtype=bool)

    n_selected = full_mask.sum()
    H2t = np.zeros((n_t, n_g, n_selected), dtype=complex)
    for i_g in range(n_g):
        t_g = t2t_np[:, :, np.newaxis] * geomz_np[i_g][np.newaxis, np.newaxis, :]
        H2t[:, i_g, :] = t_g[:, full_mask]

    H2t = H2t.squeeze()
    Omega, Lambda = np.meshgrid(omega_np, lmb_np, indexing='ij')
    Omega  = Omega[full_mask]
    Lambda = Lambda[full_mask]
    return torch.tensor(H2t), torch.from_numpy(Omega), torch.from_numpy(Lambda)

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