from scipy.special import gamma
import numpy as np

pi = np.pi

def plot_loglog(x, *args, color='C0', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if len(args) > 0:
        t = x
        x = args[0]
    else:
        t = np.arange(len(x))

    if x.dtype == complex:

        xp = x.real*1.
        xn = x.real*1.
        xp[x < 0] = np.nan
        xn[x > 0] = np.nan

        yp = x.imag*1.
        yn = x.imag*1.
        yp[yp < 0] = np.nan
        yn[yp > 0] = np.nan
        ax.loglog(t, xp, 'C0', label='Real')
        ax.loglog(t, np.abs(xn), 'C0--')
        ax.loglog(t, yp, 'C1', label='Imag')
        ax.loglog(t, np.abs(yn), 'C1--')
        ax.legend()
    else:
        xp = x*1.
        xn = x*1.
        xp[x < 0] = np.nan
        xn[x > 0] = np.nan
        ax.loglog(t, xp, color)
        ax.loglog(t, np.abs(xn), color+'--')
    return ax

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
    assert np.all(v<np.log(2)), "positive value of v will not converge"
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

    rightside_removal = christensen1990_eq17(v+1./Delta1, sc, nu=nu, a=a)
    for n in range(2, term_max):
        dright = christensen1990_eq17(v+n/Delta1, sc, nu=nu, a=a)
        rightside_removal += dright
        if np.all(np.abs(dright) < convergence_threshold*np.abs(rightside_removal)):
            break

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


class Hankel:
    def __init__(self, func, order, log10min=-20, log10max=10, smp_per_decade=10):
        self.func    = func
        self.nu    = order
        self.smp_per_decade = smp_per_decade
        self.n_decades = log10max-log10min
        self.lamda = 10**(np.arange(log10min*self.smp_per_decade, log10max*self.smp_per_decade+1)*self.delta)
        self.v     = np.log(self.lamda)
        self.fc=self.calc_coefficients()
        self.filter_i0 = np.searchsorted(self.v, 0)
        self.F = self.func(self.lamda)
    
    @property
    def delta(self):
        return 1./self.smp_per_decade
    
    def calc_coefficients(self):
        return optimized_coefficients_christensen1990(self.v, omega_0=np.pi/4., nu=self.nu, smp_per_decade=self.smp_per_decade)
    
    def __call__(self, r):
        i_offset = int(np.round(self.smp_per_decade*np.log10(r)))
        if i_offset==0:
            return np.einsum('...i,...i->...', self.F, self.fc)/r
        elif i_offset>0:
            return np.einsum('...i,...i->...', self.F[..., :-i_offset], self.fc[..., i_offset:])/r
        else:
            i_offset *= -1
            return np.einsum('...i,...i->...', self.fc[..., :-i_offset], self.F[..., i_offset:])/r
        
    def range(self, rmin, rmax):
        imin = int(np.floor(self.smp_per_decade*np.log10(rmin))-1)
        imax = int(np.ceil(self.smp_per_decade*np.log10(rmax))+1)
        i_offsets = np.arange(imin, imax)
        rout = 10**(i_offsets/self.smp_per_decade)
        HR = np.zeros(rout.shape+self.F.shape[:-1], dtype=complex)
        for i,i_offset in enumerate(i_offsets):
            if i_offset==0:
                HR[i] = np.einsum('...i,...i->...', self.F, self.fc)
            elif i_offset>0:
                HR[i] = np.einsum('...i,...i->...', self.F[..., :-i_offset], self.fc[..., i_offset:])
            else:
                i_offset *= -1
                HR[i] = np.einsum('...i,...i->...', self.fc[..., :-i_offset], self.F[..., i_offset:])
        return rout, HR/rout[:,None]

        
        
# instead of giving hz(t) from Hz(f) this is giving d Hz / dt (t)
twopi = np.pi*2.
twopi_sqrt = np.sqrt(2*np.pi)


class Fourier:
    def __init__(self, func, log10min=-20, log10max=10, smp_per_decade=20):
        self.func = func
        self.smp_per_decade = smp_per_decade
        self.n_decades = log10max-log10min
        self.lamda = 10**(np.arange(log10min*smp_per_decade,
                          log10max*smp_per_decade+1)*self.delta)
        self.v = np.log(self.lamda)
        self.fc = optimized_coefficients_christensen1990(self.v, omega_0=np.pi/4, nu=-0.5, smp_per_decade=self.smp_per_decade) \
                  +1j*optimized_coefficients_christensen1990(self.v, omega_0=np.pi/4, nu=0.5, smp_per_decade=self.smp_per_decade)
        self.F = self.func(self.lamda)*np.sqrt(self.lamda)

    @property
    def delta(self):
        return 1./self.smp_per_decade

    def __call__(self, r):
        i_offset = int(np.round(self.smp_per_decade*np.log10(r)))
        r_offset = 10**(i_offset*self.delta)
        if i_offset == 0:
            return self.F.dot(self.fc)/np.sqrt(r_offset)/twopi_sqrt
        else:
            return self.F[-i_offset:].dot(self.fc[:i_offset])/np.sqrt(r_offset)/twopi_sqrt
        
    def range(self, tmin, tmax):
        imin = int(np.floor(self.smp_per_decade*np.log10(tmin))-1)
        imax = int(np.ceil(self.smp_per_decade*np.log10(tmax))+1)
        i_offsets = np.arange(imin, imax)
        tout = 10**(i_offsets/self.smp_per_decade)
        Ht = np.zeros(tout.shape, dtype=complex)
        for i,i_offset in enumerate(i_offsets):
            if i_offset==0:
                Ht[i] = np.einsum('...i,...i->...', self.F, self.fc)
            elif i_offset>0:
                Ht[i] = np.einsum('...i,...i->...', self.F[..., :-i_offset], self.fc[..., i_offset:])
            else:
                i_offset *= -1
                Ht[i] = np.einsum('...i,...i->...', self.fc[..., :-i_offset], self.F[..., i_offset:])
        return tout, Ht/np.sqrt(tout)/twopi_sqrt