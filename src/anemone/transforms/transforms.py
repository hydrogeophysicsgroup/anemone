from functools import lru_cache
from .hankel_coefficients import hankel_coefficients, coefficient_bounds
import torch
import numpy as np

from anemone import constants as c, rdtype, cdtype

def factorial(s):
    x = s*1
    y = 1
    while x>0:
        y *= x
        x -= 1
    return y

@lru_cache
def gaver_stehfest_coefficients(N):
    """ Inverse laplace transform using Gaver-Stehfest method.
        see https://dl.acm.org/doi/10.1145/361953.361969
        Algorithm 368: Numerical inversion of laplace transforms
    """
    idx = np.arange(N)+1
    k_lower = (idx+1)//2
    k_upper = np.clip(idx, 0, N/2)+1
    #v = np.zeros(N, dtype=float)
    v = []

    for i in idx:
        tmpv = 0
        for k in np.arange(k_lower[i-1], k_upper[i-1]):
            numerator = factorial(2*k)
            numerator *= k**(N/2)# niels code values
            # numerator *= k**(1+N/2)# what the paper says. its wrong
            denominator = factorial(N/2 - k)
            denominator *= factorial(k)
            denominator *= factorial(k-1)
            denominator *= factorial(i-k)
            denominator *= factorial(2*k-i)
            tmpv += numerator/denominator
        v.append((-1)**(i + N/2.)*tmpv)
    return torch.tensor(v, dtype=rdtype())

class invLaplace:
    """ Inverse laplace transform using Gaver-Stehfest method.
        see https://dl.acm.org/doi/10.1145/361953.361969
        Algorithm 368: Numerical inversion of laplace transforms
    """
    def __init__(self, func, n_coefficients=16, device=None):
        #8 coefficient is more stable than 16 for current implementation. not sure why this is.

        assert n_coefficients % 2 == 0, "Gaver-stehfest method requires an even number of coefficients"
        self.func = func
        self.device = device
        self.v = gaver_stehfest_coefficients(n_coefficients)
        if device is not None:
            self.v = self.v.to(device)
        else:
            self.device = self.v.device
        self.i = (torch.arange(n_coefficients, device=self.device)+1).to(rdtype())
        self.n_coefficients = n_coefficients
        

    def __call__(self, t):
        q = (c.log2 / t)[:, None]
        qi = (q*self.i).flatten()
        respt1 = self.func(qi)
        respt1 = respt1.reshape(respt1.shape[:-1]+t.shape+self.i.shape)
        respt1[..., :,:] *= q
        if respt1.dtype != self.v.dtype:
            self.v = self.v.to(respt1.dtype)
        res = respt1@self.v            
        return res
        

import matplotlib.pyplot as plt
import numpy as np

def passthrough(x):
    return x

class CubicSpline:
    def __init__(self, x, y, transform=passthrough, integrate=True):
        self.x = transform(x.to(y.device))
        self.y = y
        self.integrated=None
        self.compute_coefficients(integrate=integrate)
        self.transform = transform


    def compute_coefficients(self, integrate=True):
        x = self.x
        y = self.y

        n = len(x)-1

        b = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
        c = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
        d = torch.zeros(y.shape, dtype=y.dtype, device=y.device)

        dx = torch.diff(x, prepend=torch.zeros(1, device=y.device), axis=0)
        # dx[0] = dx[1]

        d[..., :-1] = dx[1:]
        b[..., 1:]  = 2*(d[..., :-1]+d[..., 1:])
        c[..., 1:]  = torch.diff(y, axis=-1)/d[..., :-1]
        c[..., :-1] = torch.diff(c, axis=-1)

        b[..., 0]=-d[..., 0]
        b[..., n]=-d[..., n-1]
        c[..., 0]=0
        c[..., n]=0
        if n>3:
            c[..., 0]=c[..., 2]/(x[..., 3]-x[..., 1])-c[..., 1]/(x[..., 2]-x[..., 0])
            c[..., n]=c[..., n-1]/(x[..., n]-x[..., n-2])-c[..., n-2]/(x[..., n-1]-x[..., n-3])
            c[..., 0]=c[..., 0]*d[..., 0]*d[..., 0]/(x[..., 3]-x[..., 0])
            c[..., n]=-c[..., n]*d[..., n-1]*d[..., n-1]/(x[..., n]-x[..., n-3])

        for k in range(1, n+1):
            t=d[..., k-1]/b[..., k-1]
            b[..., k]=b[..., k]-t*d[..., k-1]
            c[..., k]=c[..., k]-t*c[..., k-1]

        c[..., n]=c[..., n]/b[..., n]
        for k in range(n-1, -1, -1):
            c[..., k]=(c[..., k]-d[..., k]*c[..., k+1])/b[..., k]

        b[..., n]=(y[..., n]-y[..., n-1])/d[..., n-1]+d[..., n-1]*(c[..., n-1]+c[..., n]+c[..., n])
        for k in range(n):
            b[..., k]=(y[..., k+1]-y[..., k])/d[..., k]-d[..., k]*(c[..., k+1]+c[..., k]+c[..., k])
            d[..., k]=(c[..., k+1]-c[..., k])/d[..., k]
            c[..., k]=3*c[..., k]
        c[..., n]=3*c[..., n]
        d[..., n]=d[..., n-1]

    
        self.a = y
        self.b = b
        self.c = c
        self.d = d
        if integrate:
            self.integrated = torch.cumsum(self.a*dx + 0.5*self.b*dx**2 + (1./3.)*self.c*dx**3 + 0.25*self.d*dx**4, axis=-1)



    def integrate(self, x_lower, x_upper, normalize=False):

        xt1 = self.transform(x_lower)
        ix1 = torch.searchsorted(self.x, xt1, side='right')-1
        dx = xt1-self.x[ix1]
        res_lb = self.a[..., ix1]*dx + 0.5*self.b[..., ix1]*dx**2 + (1./3.)*self.c[..., ix1]*dx**3 + 0.25*self.d[..., ix1]*dx**4
        
        xt2 = self.transform(x_upper)
        ix2 = torch.searchsorted(self.x, xt2, side='right')-1
        dx = xt2-self.x[ix2]
        res_ub = self.a[..., ix2]*dx + 0.5*self.b[..., ix2]*dx**2 + (1./3.)*self.c[..., ix2]*dx**3 + 0.25*self.d[..., ix2]*dx**4
        integrand = (res_ub+self.integrated[..., ix2-1]-self.integrated[..., ix1-1]-res_lb)
        if normalize:
            integrand /= (xt2-xt1)
        return integrand
    
    def __call__(self, xnew):
        if isinstance(xnew, float): xtmp = torch.tensor(xnew, device=self.y.device)
        else: xtmp = xnew.to(self.y.device)
        xt = self.transform(xtmp)
        ix = torch.searchsorted(self.x, xt, side='right')-1
        dx = (xt-self.x[ix])#[:, None]
        return self.a[..., ix] + self.b[..., ix]*dx + self.c[..., ix]*dx**2 + self.d[..., ix]*dx**3
    

class Hankel:
    def __init__(self, func, order, log10min=-20, log10max=20, smp_per_decade=10, device=None, range=None, tolerance=None, lambda_included=True, omega_0=np.pi/4.):
        self.func    = func
        self.nu    = order
        self.smp_per_decade = smp_per_decade
        self.n_decades = log10max-log10min
        a = None if order == 0 else 0.1
        hkl = hankel_coefficients(order, log10min, log10max, smp_per_decade, omega_0, a=a)
        # if tolerance is not None:
        #     idxmin, idxmax = coefficient_bounds(hkl[1].cpu().numpy(), tolerance=tolerance)
        #     hkl = hkl[0][idxmin:idxmax], hkl[1][idxmin:idxmax]
        self.tolerance = tolerance

        self.func_results = None       

        self.lamda = torch.clone(hkl[0])#.to(rdtype())
        self.fc = torch.clone(hkl[1])#.to(rdtype())

        if device is not None:
            self.fc=self.fc.to(device)
            self.lamda=self.lamda.to(device)
        
        self.lambda_included = lambda_included
        self.device=device

        self.rc = 1.
        self.rcidx = 0
        if range is not None:
            self._recenter_function(range)
            self.create_interpolator(range[0], range[1])
    
    def _recenter_function(self, r_center):
        assert self.rcidx == 0, "recentering is only allowed once"
        if isinstance(r_center, tuple): r_center = torch.sqrt(r_center[0]*r_center[1]).to(self.device)
        self.rc = r_center#.to(rdtype())
        self.lamda = self.lamda/self.rc
        self.rcidx = self.smp_per_decade*torch.log10(self.rc)
        self.func_results = None

    def create_interpolator(self, rmin, rmax):
        if self.func_results is None:
            self.func_results = self.func(self.lamda)
            if not self.lambda_included:
                self.func_results *= self.lamda
        self.r, self.resp = self.range(rmin, rmax)
        self.interpfunc = CubicSpline(self.r, self.resp, transform=torch.log10, integrate=False)
    
    
    def r_to_idx(self, r):
        return self.smp_per_decade*torch.log10(r)-self.rcidx
    
    def idx_to_r(self, idx):
        return 10**((idx + self.rcidx)*self.delta)
    

    @property
    def delta(self):
        """
        Calculate the delta value.

        Returns:
        - delta (float): The delta value.

        """
        return 1. / self.smp_per_decade

    
    def range(self, rmin, rmax):
        """ the reason for this is to make it reusable for the fourier transform"""
        return self._range(rmin, rmax)

    def _range(self, rmin, rmax):
        """
        Compute the range of values between `rmin` and `rmax`.

        Parameters:
        - rmin (float): The minimum value of the range.
        - rmax (float): The maximum value of the range.

        Returns:
        - rout (torch.Tensor): The range of values between `rmin` and `rmax`.
        - HR (torch.Tensor): The computed values based on the range.

        """
        if self.func_results is None:
            self.func_results = self.func(self.lamda)
            if not self.lambda_included: 
                self.func_results *= self.lamda
        if not isinstance(rmin, torch.Tensor): rmin = torch.tensor(rmin, device=self.device)
        if not isinstance(rmax, torch.Tensor): rmax = torch.tensor(rmax, device=self.device)
        imin = int(torch.floor(self.r_to_idx(rmin)))-2
        imax = int(torch.ceil(self.r_to_idx(rmax)))+2
        i_offsets = torch.arange(imin, imax, device=self.device)
        rout = self.idx_to_r(i_offsets).to(self.lamda.dtype)
        HR = torch.zeros(self.func_results.shape[:-1] + rout.shape, dtype=self.func_results.dtype, device=self.device)
        if self.device is not None:
            rout = rout.to(self.device)
            HR = HR.to(self.device)
            i_offsets = i_offsets.to(self.device)
        if self.fc.dtype != self.func_results.dtype:
            self.fc = self.fc.to(self.func_results.dtype)   

         


        for i, i_offset in enumerate(i_offsets):
            if i_offset == 0:
                HR[..., i] = torch.einsum('...i,...i->...', self.func_results, self.fc)
            elif i_offset > 0:
                HR[..., i] = torch.einsum('...i,...i->...', self.func_results[..., :-i_offset], self.fc[..., i_offset:])
            else:
                i_offset *= -1
                HR[..., i] = torch.einsum('...i,...i->...', self.fc[..., :-i_offset], self.func_results[..., i_offset:])
        
        return rout, HR/rout
        
    def __call__(self, r):
        if isinstance(r, float): r = torch.tensor(r, device=self.device)
        rmax = r.max()
        rmin = r.min()
        if not 'interpfunc' in dir(self):
            self.r, self.resp = self.range(rmin, rmax)
            self.interpfunc = CubicSpline(self.r, self.resp, transform=torch.log10)
        r = r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
        return self.interpfunc(r)#.to(self.device)
    
    def preintegrate(self, GeomOperator, rmin, rmax):
        if self.func_results is None:
            self.func_results = self.func(self.lamda)
            if not self.lambda_included: 
                self.func_results *= self.lamda
        if not isinstance(rmin, torch.Tensor): rmin = torch.tensor(rmin, device=self.device)
        if not isinstance(rmax, torch.Tensor): rmax = torch.tensor(rmax, device=self.device)
        imin = int(torch.floor(self.r_to_idx(rmin)))-2
        imax = int(torch.ceil(self.r_to_idx(rmax)))+2
        i_offsets = torch.arange(imin, imax, device=self.device)
        rout = self.idx_to_r(i_offsets).to(self.lamda.dtype)
        geomz = GeomOperator(rout).squeeze()

        if self.device is not None:
            rout = rout.to(self.device)
            geomz = geomz.to(self.device)
            i_offsets = i_offsets.to(self.device)


        FC = torch.zeros_like(self.fc, device=self.device)
        
        for i, i_offset in enumerate(i_offsets):
            if i_offset == 0:
                FC += geomz[i]*self.fc / rout[i]
            elif i_offset > 0:
                FC[:-i_offset] += geomz[i]*self.fc[..., i_offset:] / rout[i]
            else:
                i_offset *= -1
                FC[i_offset:] += geomz[i]*self.fc[..., :-i_offset] / rout[i]
        
        if self.tolerance is not None:
            idxmin, idxmax = coefficient_bounds(FC.cpu().numpy(), tolerance=self.tolerance)
            idx_slice = slice(idxmin, idxmax)
        else:
            idx_slice = slice(None, None)
        if FC.dtype != self.func_results.dtype:
            FC = FC.to(self.func_results.dtype)   
        return torch.einsum('...i,i->...', self.func_results, FC), FC[idx_slice], self.lamda[..., idx_slice]

class Fourier(Hankel):
    """ A fourier transform for logarithmically sampled data. The implementation uses hankel transforms at the base.
        See Johansen and Sorensen 1979 - Fast Hankel Transforms eq.  87 + 88
    """
        
    def __init__(self, func, log10min=-20, log10max=20, smp_per_decade=16, device=None, range=None, tolerance=None):
        def f1(lamda):
            return func(lamda/c.sqrt_twopi)/torch.sqrt(lamda)
        super().__init__(f1, -0.5, log10min, log10max, smp_per_decade, device, None)
        self.lambda_included = False
        _, realfc = hankel_coefficients(-0.5, log10min, log10max, smp_per_decade, omega_0=np.pi/5.25, a=0.1)
        _, imagfc = hankel_coefficients( 0.5, log10min, log10max, smp_per_decade, omega_0=np.pi/5.25, a=0.1)
        # imagfc = imagfc
        self.fc = torch.complex(realfc, -1*imagfc).to(self.fc.device)
        self.f1 = f1
       

        if range is not None:
            self._recenter_function(range)
            self.create_interpolator(range[0], range[1])

    
    def f2r(self, f):
        """frequency to r"""
        return f*c.sqrt_twopi
    def r2f(self, r):
        """r to frequency"""
        return r/c.sqrt_twopi

    def range(self, fmin, fmax):
        rout, Resp = self._range(self.f2r(fmin), self.f2r(fmax))
        return self.r2f(rout), Resp*torch.sqrt(rout)
    
    def __call__(self, r):
        if isinstance(r, float): r = torch.tensor(r, device=self.device)
        rmax = r.max()
        rmin = r.min()
        if not 'interpfunc' in dir(self):
            self.r, self.resp = self.range(rmin, rmax)
            self.interpfunc = CubicSpline(self.r, self.resp, integrate=False, transform=torch.log10)
        r = r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
        return self.interpfunc(r)


    def calc_tvals(self, tmin, tmax):
        rmin,rmax = self.f2r(tmin), self.f2r(tmax)
        if not isinstance(rmin, torch.Tensor): rmin = torch.tensor(rmin, device=self.device)
        if not isinstance(rmax, torch.Tensor): rmax = torch.tensor(rmax, device=self.device)
        imin = int(torch.floor(self.r_to_idx(rmin)))-2
        imax = int(torch.ceil( self.r_to_idx(rmax)))+2
        i_offsets = torch.arange(imin, imax, device=self.device)
        rout = self.idx_to_r(i_offsets).to(self.lamda.dtype)
        return i_offsets, rout, self.r2f(rout)

    def preintegrate(self, timeOperator, tmin, tmax):
        i_offsets, rout, tout = self.calc_tvals(tmin, tmax)
        Wmat = timeOperator(tout).to(cdtype())

        if self.func_results is None:
            self.func_results = self.func(self.lamda)
            if not self.lambda_included: 
                self.func_results *= self.lamda


        if self.device is not None:
            rout = rout.to(self.device)
            Wmat = Wmat.to(self.device)
            i_offsets = i_offsets.to(self.device)
        if self.fc.dtype != self.func_results.dtype:
            self.fc = self.fc.to(self.func_results.dtype)   

        FC = torch.zeros(Wmat.shape[:-1]+self.fc.shape, dtype=cdtype(), device=self.device)
        sqRout = torch.sqrt(rout)
        
        for i, i_offset in enumerate(i_offsets):
            if i_offset == 0:
                FC += Wmat[..., i, None]*self.fc*sqRout[i]/rout[i]
            elif i_offset > 0:
                FC[..., :-i_offset] += Wmat[..., i, None]*self.fc[i_offset:]*sqRout[i]/rout[i]
            else:
                i_offset *= -1
                FC[..., i_offset:] += Wmat[..., i, None]*self.fc[:-i_offset]*sqRout[i]/rout[i]
        
        if FC.dtype != self.func_results.dtype:
            FC = FC.to(self.func_results.dtype)   
                    
        if self.tolerance is not None:
            idxmin, idxmax = FC.shape[-1]-1, 0
            for i in range(len(FC)):
                cimin,cimax = coefficient_bounds(FC[i].cpu().numpy(), tolerance=self.tolerance)
                idxmin = min(idxmin, cimin)
                idxmax = max(idxmax, cimax)

            return torch.einsum('ij,...j->...i', FC, self.func_results), FC[..., idxmin:idxmax], self.lamda.squeeze()[idxmin:idxmax]
        else:
            return torch.einsum('ij,...j->...i', FC, self.func_results), FC, self.lamda.squeeze()



class invFourier(Fourier):
    def __init__(self, func, log10min=-20, log10max=20, smp_per_decade=16, device=None, range=None, tolerance=None):
        super().__init__(func, log10min, log10max, smp_per_decade, device, None)
        self.lambda_included = False        
        self.fc = self.fc.real - 1j*self.fc.imag
        self.tolerance = tolerance
        if range is not None:
            self._recenter_function(range)
            self.create_interpolator(range[0], range[1])
    def __repr__(self) -> str:
        return 'X(f)->x(t)'
    