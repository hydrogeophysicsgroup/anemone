from scipy.special import gamma
import numpy as np
import torch

pi = np.pi

def christensen1990_calculate_smoothing(omega_0, sc):
    """calculate the parameter a.

    Args:
        omega_0 (float): angle opening angle in radians
        sc (float): cutoff frequency
    """
    return 1. / (2*sc*omega_0)

def christensen1990_p_hat(s, a):
    """
    Calculate the Christensen 1990 p-hat function.

    Parameters:
    s (float): The input value.
    a (float): The parameter value.

    Returns:
    float: The calculated p-hat value.
    """
    p1 = np.tanh(pi*(s+0.5)/a)
    p2 = np.tanh(pi*(s-0.5)/a)
    return 0.5*(p1-p2)


def christensen1990_z_n(n, nu=0):
    return -1j*(n+0.5*(nu+1))/pi


def johansen1979_Hhat_nu(s, nu=0):
    """equation 8 Johansen Sorensen 1979"""
    Hhat_nu = 2**(-2j*pi*s)
    Hhat_nu *= gamma(0.5*(nu+1) - 1j*np.pi*s)
    Hhat_nu /= gamma(0.5*(nu+1) + 1j*np.pi*s)
    return Hhat_nu


def christensen1990_eq16(v, sc, nu=0, a=0.1, term_max=50, convergence_threshold=1e-9):
    """this equation is for v<ln(2)"""
    # assert np.all(v<np.log(2)), "positive value of v will not converge"
    delta = 1./(2*sc)

    n = 0
    zn = christensen1990_z_n(n, nu=nu)
    en_delta_pn = np.exp(2*n*v)*delta*christensen1990_p_hat(delta*zn, a)
    H_nu_sum1 = en_delta_pn/(gamma(n+nu+1))
    negOne_pow_n = 1
    den4 = 1
    for n in range(1, term_max):
        negOne_pow_n *= -1
        den4 *= 4
        denominator = den4*gamma(n+1)*gamma(n+nu+1)

        zn = christensen1990_z_n(n, nu=nu)
        numerator = np.exp(2*n*v)*delta*christensen1990_p_hat(delta*zn, a)
        dHnu = numerator / denominator
        H_nu_sum1 += negOne_pow_n*dHnu

        if np.all(dHnu < convergence_threshold*H_nu_sum1):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    H_nu_sum1 *= np.exp(v*(nu+1))*2**(-nu)

    n = 0
    H_nu_partial_sum2 = np.imag(
        np.exp(1j*2*pi*sc*v)*np.exp(4*np.pi*a*sc*n*v)*johansen1979_Hhat_nu(sc-1j*2*a*sc*(n+0.5)))

    for n in range(1, term_max):
        xi_minus = sc-1j*2*a*sc*(n+0.5)
        dHnu = np.imag(np.exp(1j*2*pi*sc*v)*np.exp(4*np.pi *
                       a*sc*n*v)*johansen1979_Hhat_nu(xi_minus, nu=nu))
        H_nu_partial_sum2 += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(H_nu_partial_sum2)):
            break
    H_nu_sum2 = 2*a*np.exp(2*pi*a*sc*v)*H_nu_partial_sum2
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    return H_nu_sum1 - H_nu_sum2.astype(complex)

def christensen1990_eq17(v, sc, nu=0, a=0.1, term_max=50, convergence_threshold=1e-9):
    n = 0
    xi_plus = sc+1j*2*a*sc*(n+0.5)
    H_nu_partial_sum = np.exp(-4*pi*a*sc*n*v)*johansen1979_Hhat_nu(xi_plus, nu=nu)

    for n in range(1, term_max):
        xi_plus = sc+1j*2*a*sc*(n+0.5)
        dHnu = np.exp(-4*pi*a*sc*n*v)*johansen1979_Hhat_nu(xi_plus, nu=nu)
        H_nu_partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(H_nu_partial_sum)):
            break
    H_nu = 2*a*np.exp(-2.*pi*a*sc*v)*np.imag(np.exp(1j*2*pi*sc*v)*H_nu_partial_sum)

    return H_nu.astype(complex)


def christensen1990_eq41(v, delta, Delta1, a, sc, nu=0, smp_per_decade=10, term_max=50, convergence_threshold=1e-9):
    n=0
    partial_sum = delta*christensen1990_p_hat(n*delta*Delta1, a)*johansen1979_Hhat_nu(n*Delta1, nu=nu)*np.exp(1j*2*pi*n*v*Delta1)

    for n in range(1, term_max):
        dHnu = delta*christensen1990_p_hat(n*delta*Delta1, a)*johansen1979_Hhat_nu(n*Delta1, nu=nu)*np.exp(1j*2*pi*n*v*Delta1)
        partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(partial_sum)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    for n in range(-1, -term_max, -1):
        dHnu = delta*christensen1990_p_hat(n*delta*Delta1, a) * johansen1979_Hhat_nu(n *
                                                        Delta1, nu=nu)*np.exp(1j*2*pi*n*v*Delta1)
        partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(partial_sum)):
            break

    partial_sum *= Delta1

    leftside_removal = christensen1990_eq16(v-1/Delta1, sc, nu=nu, a=a)
    for n in range(2, term_max):
        dleft = christensen1990_eq16(v-n/Delta1, sc, nu=nu, a=a)
        leftside_removal += dleft
        if np.all(np.abs(dleft) < convergence_threshold*np.abs(leftside_removal)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    rightside_removal = christensen1990_eq17(v+1./Delta1, sc, nu=nu, a=a)
    for n in range(2, term_max):
        dright = christensen1990_eq17(v+n/Delta1, sc, nu=nu, a=a)
        rightside_removal += dright
        if np.all(np.abs(dright) < convergence_threshold*np.abs(rightside_removal)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    return partial_sum-leftside_removal-rightside_removal


def optimized_coefficients_christensen1990(v, omega_0=0., nu=0, smp_per_decade=10,  term_max=50, convergence_threshold=1e-12):
    """
    Calculate the optimized coefficients using the Christensen (1990) method.

    Parameters:
    v (array-like): Frequencies at which to calculate the coefficients.
    omega_0 (float, optional): Reference frequency. Default is 0.
    nu (int, optional): Smoothing parameter. Default is 0.
    smp_per_decade (int, optional): Number of samples per decade. Default is 10.
    term_max (int, optional): Maximum number of terms in the series. Default is 50.
    convergence_threshold (float, optional): Threshold for convergence. Default is 1e-16.

    Returns:
    array-like: Optimized coefficients calculated using the Christensen (1990) method.
    """
    delta = np.log(10)/smp_per_decade
    sc = 1./(2*delta)

    v_minus = np.log(2) # from Christensen 1990;
    v_plus = 4.+0.1*sc   # from Christensen 1990; this will make the 
    #    error integral < 10^-16

    Delta1 = 1./(v_plus-v_minus)# from Christensen 1990

    a = christensen1990_calculate_smoothing(omega_0, sc)

    Hnustar = np.zeros_like(v, dtype=np.complex128)
    Hnustar[:] = np.nan

    Hnustar[v < v_minus] = christensen1990_eq16(v[v < v_minus], sc, nu=nu, a=a, term_max=term_max, 
                                                convergence_threshold=convergence_threshold)
    Hnustar[v > v_plus] = christensen1990_eq17(v[v > v_plus], sc,  nu=nu, a=a, term_max=term_max,
                                               convergence_threshold=convergence_threshold)
    
    v_middle = (v>=v_minus)*(v<=v_plus)
    Hnustar[v_middle] = christensen1990_eq41(v[v_middle], delta, Delta1, a, sc, nu=nu)

    return np.real(Hnustar)


def coefficient_bounds(coefficients, tolerance=1e-6):
    """a function to find an interval that makes up most of the signal."""
    fcabs = np.abs(coefficients)
    bigdogs = np.argsort(fcabs)[::-1]
    fcsort = fcabs[bigdogs]*1
    signal_fraction = np.cumsum(fcsort)/fcsort.sum()
    idx = np.searchsorted(signal_fraction, 1.-tolerance)
    bigdogs = bigdogs[:idx+1]
    return bigdogs.min(), bigdogs.max()


from functools import lru_cache
@lru_cache
def hankel_coefficients(order, log10min, log10max, smp_per_decade):
    """the reason from wrapping this is to make the inputs hashable and therefore cachable"""
    lamda = 10**(np.arange(log10min*smp_per_decade, log10max*smp_per_decade+1).astype(np.float64)/smp_per_decade)
    v = np.log(lamda)
    fc = optimized_coefficients_christensen1990(v, omega_0=np.pi/4, nu=order, smp_per_decade=smp_per_decade)
    return torch.from_numpy(lamda), torch.from_numpy(fc)

class Hankel:
    def __init__(self, func, order, log10min=-20, log10max=10, smp_per_decade=10, device=None, range=None, tolerance=None):
        self.func    = func
        self.nu    = order
        self.smp_per_decade = smp_per_decade
        self.n_decades = log10max-log10min
        hkl = hankel_coefficients(order, log10min, log10max, smp_per_decade)
        if tolerance is not None:
            idxmin, idxmax = coefficient_bounds(hkl[1].cpu().numpy(), tolerance=tolerance)
            hkl = hkl[0][idxmin:idxmax], hkl[1][idxmin:idxmax]
        self.tolerance = tolerance

        self.func_results = None       

        self.lamda = torch.clone(hkl[0])#.to(torch.double)
        self.fc = torch.clone(hkl[1])#.to(torch.double)
        if device is not None:
            self.fc=self.fc.to(device)
            self.lamda=self.lamda.to(device)
        self.device=device

        self.rc = 1.
        self.rcidx = 0
        if range is not None:
            self._recenter_function(range)
            self.create_interpolator(range[0], range[1])
    
    def _recenter_function(self, r_center):
        assert self.rcidx == 0, "recentering is only allowed once"
        if isinstance(r_center, tuple): r_center = torch.sqrt(r_center[0]*r_center[1]).to(self.device)
        self.rc = r_center
        self.lamda = self.lamda/self.rc
        self.rcidx = self.smp_per_decade*torch.log10(self.rc)
        self.func_results = None

    def create_interpolator(self, rmin, rmax):
        if self.func_results is None:
            self.func_results = self.func(self.lamda)
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
        if not isinstance(rmin, torch.Tensor): rmin = torch.tensor(rmin, device=self.device)
        if not isinstance(rmax, torch.Tensor): rmax = torch.tensor(rmax, device=self.device)
        imin = int(torch.floor(self.r_to_idx(rmin)) - 2)
        imax = int(torch.ceil(self.r_to_idx(rmax))  + 2)
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
        if not 'interpfunc' in dir(self):
            rmax = r*(self.delta)**2
            rmin = r/(self.delta**2)
            self.r, self.resp = self.range(rmin, rmax)
            self.interpfunc = CubicSpline(self.r, self.resp, transform=torch.log)
        r = r if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
        return self.interpfunc(r)#.to(self.device)
            
     
        
# instead of giving hz(t) from Hz(f) this is giving d Hz / dt (t)
twopi = torch.pi*2.
twopi_sqrt = twopi**(1/2)
from anemone import constants as c

class Fourier(Hankel):
    """ A fourier transform for logarithmically sampled data. The implementation uses hankel transforms at the base.
        See Johansen and Sorensen 1979 - Fast Hankel Transforms eq.  87 + 88
    """
        
    def __init__(self, func, log10min=-20, log10max=10, smp_per_decade=16, device=None, range=None, tolerance=None):
        def f1(lamda):
            return func(lamda/twopi_sqrt)*torch.sqrt(lamda)
        super().__init__(f1, -0.5, log10min, log10max, smp_per_decade, device)
        _, imagfc = hankel_coefficients(0.5, log10min, log10max, smp_per_decade)
        imagfc = imagfc.to(self.fc.device)
        self.fc = torch.complex(self.fc, -1*imagfc)
        if tolerance is not None:
            idxmin, idxmax = coefficient_bounds(self.fc.cpu().numpy(), tolerance=tolerance)
            self.fc = self.fc[idxmin:idxmax]
            self.lamda = self.lamda[idxmin:idxmax]

        if range is not None:
            self._recenter_function(range)
            self.create_interpolator(range[0], range[1])
        

    def range(self, fmin, fmax):
        rout, Resp = self._range(fmin*c.sqrt_twopi, fmax*c.sqrt_twopi)
        return rout/c.sqrt_twopi, Resp*torch.sqrt(rout)       

    def __repr__(self) -> str:
        return 'x(t)->X(f)'

class invFourier(Fourier):
    def __init__(self, func, log10min=-20, log10max=10, smp_per_decade=20, device=None, range=None, tolerance=None):
        super().__init__(func, log10min, log10max, smp_per_decade, device, range, tolerance)
        self.fc = self.fc.real - 1j*self.fc.imag
        if range is not None:
            self.create_interpolator(range[0], range[1])
    def __repr__(self) -> str:
        return 'X(f)->x(t)'

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
    return torch.tensor(v, dtype=torch.float64)

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
        self.i = (torch.arange(n_coefficients, device=self.device)+1).to(torch.float64)
        self.n_coefficients = n_coefficients
        

    # def __call__(self, t):
    #     # ntimes = t.shape[-1]

    #     q = (c.log2 / t)[:, None]
    #     respt1 = self.func(q[0,:]*self.i)
    #     if respt1.dtype != self.v.dtype:
    #         self.v = self.v.to(respt1.dtype)
            
    #     res1 = q[0,:]*respt1@self.v
    #     res = torch.zeros(t.shape+res1.shape, dtype=respt1.dtype, device=self.device)
    #     res[0] = res1
    #     for i_t in range(1, t.shape[-1]):
    #         res[i_t] = q[i_t,:]*self.func(q[i_t,:]*self.i)@self.v
    #     # print('the shape of res is ', res.shape)
    #     return res.T

    def __call__(self, t):
        q = (c.log2 / t)[:, None]
        qi = (q*self.i).flatten()
        # print('the shape of qi is ', qi.shape)
        respt1 = self.func(qi)
        # print('the shape of respt1 is ', respt1.shape)
        respt1 = respt1.reshape(respt1.shape[:-1]+t.shape+self.i.shape)
        respt1[..., :,:] *= q
        # print('the shape of respt1 is ', respt1.shape)
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
    