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


def christensen1990_eq41(v, delta, Delta1, a, sc, nu=0, smp_per_decade=10,term_max=50, convergence_threshold=1e-9):
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



def optimized_coefficients_christensen1990(v, omega_0=0., nu=0, smp_per_decade=10,  term_max=50, convergence_threshold=1e-16):
    delta = np.log(10)/smp_per_decade
    sc = 1./(2*delta)
    # sc = 1.25
    # delta = 1./(2*sc)
    #if nu==0:
    v_minus = np.log(2) # from Christensen 1990;
    v_plus = 4.+0.1*sc   # from Christensen 1990; this will make the 
#    error integral < 10^-16
    # else:
    #     v_minus=0
    #     v_plus = 8

    Delta1 = 1./(v_plus-v_minus)# from Christensen 1990



    a = christensen1990_calculate_smoothing(omega_0, sc)

    Hnustar = np.zeros_like(v, dtype=complex)
    Hnustar[:] = np.nan

    Hnustar[v < v_minus] = christensen1990_eq16(v[v < v_minus], sc, nu=nu, a=a, term_max=term_max, 
                                                convergence_threshold=convergence_threshold)
    Hnustar[v > v_plus] = christensen1990_eq17(v[v > v_plus], sc,  nu=nu, a=a, term_max=term_max,
                                               convergence_threshold=convergence_threshold)
    
    v_middle = (v>=v_minus)*(v<=v_plus)
    Hnustar[v_middle] = christensen1990_eq41(v[v_middle], delta, Delta1, a, sc, nu=nu)

    return np.real(Hnustar)


class BaseHankel:
    def __init__(self, func, order, log10min=-20, log10max=10, smp_per_decade=10, device=None):
        self.func    = func
        self.nu    = order
        self.smp_per_decade = smp_per_decade
        self.n_decades = log10max-log10min
        self.lamda = 10**(np.arange(log10min*self.smp_per_decade, log10max*self.smp_per_decade+1)*self.delta)
        self.v     = np.log(self.lamda)
        self.fc=torch.tensor(self.calc_coefficients(), dtype=torch.complex128)
        self.filter_i0 = np.searchsorted(self.v, 0)
        self.lamda = torch.tensor(self.lamda, dtype=torch.complex128)
        if device is not None:
            self.fc=self.fc.to(device)
            self.lamda=self.lamda.to(device)
        self.device=device
        self.F = self.func(self.lamda)
    
    @property
    def delta(self):
        return 1./self.smp_per_decade
    
    def calc_coefficients(self):
        return optimized_coefficients_christensen1990(self.v, omega_0=np.pi/4, nu=self.nu, smp_per_decade=self.smp_per_decade)
           
    def range(self, rmin, rmax):
        imin = int(np.floor(self.smp_per_decade*np.log10(rmin))-1)
        imax = int(np.ceil(self.smp_per_decade*np.log10(rmax))+1)
        i_offsets = torch.tensor(np.arange(imin, imax))
        rout = 10**(i_offsets/self.smp_per_decade)
        HR = torch.zeros(self.F.shape[:-1]+rout.shape, dtype=torch.complex128)
        if self.device is not None:
            rout=rout.to(self.device)
            HR=HR.to(self.device)
            i_offsets=i_offsets.to(self.device)


        for i,i_offset in enumerate(i_offsets):
            if i_offset==0:
                HR[..., i] = torch.einsum('...i,...i->...', self.F, self.fc)
            elif i_offset>0:
                HR[..., i] = torch.einsum('...i,...i->...', self.F[..., :-i_offset], self.fc[..., i_offset:])
            else:
                i_offset *= -1
                HR[..., i] = torch.einsum('...i,...i->...', self.fc[..., :-i_offset], self.F[..., i_offset:])
        
        return rout, HR/rout

        
        
# instead of giving hz(t) from Hz(f) this is giving d Hz / dt (t)
twopi = torch.pi*2.
twopi_sqrt = twopi**(1/2)

class Fourier(BaseHankel):
    def __init__(self, func, log10min=-20, log10max=10, smp_per_decade=16, device=None):
        def f1(lamda):
            return func(lamda/twopi_sqrt)*torch.sqrt(lamda)
        super().__init__(f1, -0.5, log10min, log10max, smp_per_decade, device)
        self.fc += -1j*torch.from_numpy(optimized_coefficients_christensen1990(self.v, omega_0=np.pi/4, nu=0.5, smp_per_decade=self.smp_per_decade))
       
    def range(self, fmin, fmax):
        rout, Resp = super().range(fmin*twopi_sqrt, fmax*twopi_sqrt)
        return rout/twopi_sqrt, Resp*torch.sqrt(rout)

    def __repr__(self) -> str:
        return 'x(t)->X(f)'

class invFourier(Fourier):
    def __init__(self, func, log10min=-20, log10max=10, smp_per_decade=20, device=None):
        super().__init__(func, log10min, log10max, smp_per_decade, device)
        self.fc = self.fc.real -1j*self.fc.imag
    
    def __repr__(self) -> str:
        return 'X(f)->x(t)'

import copy
precalcHankel0 =  BaseHankel(lambda x:x, 0, smp_per_decade=10)
precalcHankel1 =  BaseHankel(lambda x:x, 1, smp_per_decade=10)
# precalcFourier =  BaseFourier(lambda x:x, smp_per_decade=30)

def Hankel(func, order):
    if order==0:
        self = copy.deepcopy(precalcHankel0)
    elif order==1:
        self = copy.deepcopy(precalcHankel1)
    self.func = func
    self.F = self.func(self.lamda)
    return self

# def Fourier(func):
#     self = copy.deepcopy(precalcFourier)
#     self.func = func
#     self.F = self.func(self.lamda).T*torch.sqrt(self.lamda)
#     return self

import matplotlib.pyplot as plt
import numpy as np

def passthrough(x):
    return x

class CubicSpline:
    def __init__(self, x, y, transform=passthrough):
        self.x = transform(x)
        self.y = y
        self.compute_coefficients()
        self.transform = transform


    def compute_coefficients(self):
        x = self.x
        y = self.y

        n = len(x)-1

        b = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
        c = torch.zeros(y.shape, dtype=y.dtype, device=y.device)
        d = torch.zeros(y.shape, dtype=y.dtype, device=y.device)

        dx = torch.diff(x, axis=0)

        d[..., :-1] = dx
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

    def integrate(self, x_lower, x_upper):
        # pseudocode

        res = self.a*dx + 0.25*self.d*dx**4 + (1/3)*self.c*dx**3 + (1/2)*self.b*dx**2

        raise NotImplemented
    
    def __call__(self, xnew):
        xt = self.transform(xnew)
        ix = torch.searchsorted(self.x, xt, side='right')-1
        dx = (xt-self.x[ix])#[:, None]
        return self.a[..., ix] + self.b[..., ix]*dx + self.c[..., ix]*dx*dx + self.d[..., ix]*dx**3
    