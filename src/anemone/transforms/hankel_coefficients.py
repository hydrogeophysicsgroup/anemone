from scipy.special import gamma
import numpy as np
import torch

from anemone import rdtype

pi = np.pi

def christensen1990_calculate_smoothing(omega_0, sc):
    """calculate the parameter a.
    from equation 32 in Christensen 1990

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


def christensen1990_z_n(n, nu):
    return -1j*(n+0.5*(nu+1))/pi


def johansen1979_Hhat_nu(s, nu):
    """equation 8 Johansen Sorensen 1979"""
    Hhat_nu = 2**(-2j*pi*s)
    Hhat_nu *= gamma(0.5*(nu+1) - 1j*np.pi*s)
    Hhat_nu /= gamma(0.5*(nu+1) + 1j*np.pi*s)
    return Hhat_nu


def christensen1990_eq16(v, sc, nu, a, term_max, convergence_threshold):
    """this equation is for v<ln(2)"""
    # assert np.all(v<np.log(2)), "positive value of v will not converge"
    delta = 1./(2*sc)

    n = 0
    zn = christensen1990_z_n(n, nu)
    en_delta_pn = np.exp(2*n*v)*delta*christensen1990_p_hat(delta*zn, a)
    H_nu_sum1 = en_delta_pn/(gamma(n+nu+1))
    negOne_pow_n = 1
    den4 = 1
    for n in range(1, term_max):
        negOne_pow_n *= -1
        den4 *= 4
        denominator = den4*gamma(n+1)*gamma(n+nu+1)

        zn = christensen1990_z_n(n, nu)
        numerator = np.exp(2*n*v)*delta*christensen1990_p_hat(delta*zn, a)
        dHnu = numerator / denominator
        H_nu_sum1 += negOne_pow_n*dHnu

        if np.all(dHnu < convergence_threshold):#*H_nu_sum1):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    H_nu_sum1 *= np.exp(v*(nu+1))*2**(-nu)

    n = 0
    H_nu_partial_sum2 = np.imag(
        np.exp(1j*2*pi*sc*v)*np.exp(4*np.pi*a*sc*n*v)*johansen1979_Hhat_nu(sc-1j*2*a*sc*(n+0.5), nu))

    for n in range(1, term_max):
        xi_minus = sc-1j*2*a*sc*(n+0.5)
        dHnu = np.imag(np.exp(1j*2*pi*sc*v)*np.exp(4*np.pi *
                       a*sc*n*v)*johansen1979_Hhat_nu(xi_minus, nu))
        H_nu_partial_sum2 += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold):#*np.abs(H_nu_partial_sum2)):
            break
    H_nu_sum2 = 2*a*np.exp(2*pi*a*sc*v)*H_nu_partial_sum2
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    return H_nu_sum1 - H_nu_sum2.astype(complex)

def christensen1990_eq17(v, sc, nu, a, term_max, convergence_threshold):
    n = 0
    xi_plus = sc+1j*2*a*sc*(n+0.5)
    H_nu_partial_sum = np.exp(-4*pi*a*sc*n*v)*johansen1979_Hhat_nu(xi_plus, nu)

    for n in range(1, term_max):
        xi_plus = sc+1j*2*a*sc*(n+0.5)
        dHnu = np.exp(-4*pi*a*sc*n*v)*johansen1979_Hhat_nu(xi_plus, nu)
        H_nu_partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(H_nu_partial_sum)):
            break
    H_nu = 2*a*np.exp(-2.*pi*a*sc*v)*np.imag(np.exp(1j*2*pi*sc*v)*H_nu_partial_sum)

    return H_nu.astype(complex)


def christensen1990_eq41(v, delta, Delta1, a, sc, nu, term_max=100, convergence_threshold=1e-12):
    n=0
    partial_sum = delta*christensen1990_p_hat(n*delta*Delta1, a)*johansen1979_Hhat_nu(n*Delta1, nu)*np.exp(1j*2*pi*n*v*Delta1)

    for n in range(1, term_max):
        dHnu = delta*christensen1990_p_hat(n*delta*Delta1, a)*johansen1979_Hhat_nu(n*Delta1, nu)*np.exp(1j*2*pi*n*v*Delta1)
        partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(partial_sum)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    for n in range(-1, -term_max, -1):
        dHnu = delta*christensen1990_p_hat(n*delta*Delta1, a) * johansen1979_Hhat_nu(n * Delta1, nu)*np.exp(1j*2*pi*n*v*Delta1)
        partial_sum += dHnu
        if np.all(np.abs(dHnu) < convergence_threshold*np.abs(partial_sum)):
            break

    partial_sum *= Delta1

    leftside_removal = christensen1990_eq16(v-1/Delta1, sc, nu, a, term_max, convergence_threshold)
    for n in range(2, term_max):
        dleft = christensen1990_eq16(v-n/Delta1, sc, nu, a, term_max, convergence_threshold)
        leftside_removal += dleft
        if np.all(np.abs(dleft) < convergence_threshold*np.abs(leftside_removal)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    rightside_removal = christensen1990_eq17(v+1./Delta1, sc, nu, a, term_max, convergence_threshold)
    for n in range(2, term_max):
        dright = christensen1990_eq17(v+n/Delta1, sc, nu, a, term_max, convergence_threshold)
        rightside_removal += dright
        if np.all(np.abs(dright) < convergence_threshold*np.abs(rightside_removal)):
            break
    if n==term_max-1:
        print('hankel coefficients didnt converge')

    return partial_sum-leftside_removal-rightside_removal


def optimized_coefficients_christensen1990(v, omega_0=0., nu=0, smp_per_decade=10,  term_max=100, convergence_threshold=1e-16, a=None):
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
    if a is None:
        sc = 1./(2*delta)
        a = christensen1990_calculate_smoothing(omega_0, sc)
    else:
        sc = 1./(2*delta)



    v_minus = np.log(2) # from Christensen 1990;
    v_plus = 4.+0.1*sc   # from Christensen 1990; this will make the 
    #    error integral < 10^-16

    Delta1 = 1./(v_plus-v_minus)# from Christensen 1990

        

    Hnustar = np.zeros_like(v, dtype=np.complex128)
    Hnustar[:] = np.nan

    Hnustar[v < v_minus] = christensen1990_eq16(v[v < v_minus], sc, nu, a, term_max, convergence_threshold)
    Hnustar[v > v_plus] = christensen1990_eq17(v[v > v_plus], sc,  nu, a=a, term_max=term_max,
                                               convergence_threshold=convergence_threshold)
    
    v_middle = (v>=v_minus)*(v<=v_plus)
    Hnustar[v_middle] = christensen1990_eq41(v[v_middle], delta, Delta1, a, sc, nu)

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
def hankel_coefficients(order, log10min, log10max, smp_per_decade, omega_0, a=None):
    """the reason from wrapping this is to make the inputs hashable and therefore cachable"""
    lamda = 10**(np.arange(log10min*smp_per_decade, log10max*smp_per_decade+1).astype(np.float64)/smp_per_decade)
    v = np.log(lamda)
    fc = optimized_coefficients_christensen1990(v, omega_0=omega_0, nu=order, smp_per_decade=smp_per_decade, a=a)
    return torch.from_numpy(lamda).to(rdtype), torch.from_numpy(fc)#.to(rdtype)