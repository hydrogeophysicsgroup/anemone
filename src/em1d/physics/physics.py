# so there are a few dimensions of parameters required to compute the TEM forward model.
# 1. depth       (z)
# 2. frequency   (omega)
# 3. wave length (lambda)
# we assume the physical properties of the earth only change with depth, i.e. the earth is 1D.
# the order of arguments will be sigma, omega, wavelength

#import numpy as nt
import torch as nt

# print('this is the physics submodule, the next print has the globals')
# print(globals().keys())
# print(globals()['__spec__'])

mu0 = nt.pi*4e-7
epsilon0 = 8.854187817e-12



def calc_zy(sigma, omega, epsilon=epsilon0, mu=mu0):
    """compute the impedivity (zhat) and admittivity (yhat)
       see W&H between eq. 1.15->1.16, reference to harrington, 1961
    """
    z = 1j*mu*omega
    y = sigma + 1j*epsilon*omega
    return z,y


def calc_wavenumber(sigma, omega, mu=mu0, epsilon=epsilon0, sqrt=True):
    """get the intrinsic complex wavenumber 'k' 
       see equation 4.30 in ward in hohmann (pg 207)
    """
    k = mu*omega*(epsilon*omega - 1j*sigma)
    if sqrt: k = nt.sqrt(k)
    return k

def halfspace_vmd(sigma, omega, r):
    """get the analytic response for a vertical magnetic dipole at the surface of the earth
       see equation 4.56 of ward and hohmann (pg.211)

        Args:
            omega (1d-array): the angular frequency
            sigma    (float): halfspace conductivity
            r        (float): horizontal distance from the dipole

        Returns:
            z-component of the magnetic field
    """
    k = calc_wavenumber(sigma, omega)
    Hz = 9 - (9 + 9j*k*r - 4*k**2 * r**2 -1j*k**3*r**3)*nt.exp(-1j*k*r)
    Hz /= (2*nt.pi*k**2 * r**5)
    return Hz

def halfspace_center_circle_loop(omega, sigma, r, I=1.):
    """ get the vertical component of the response at the center of a circular loop on the earth's surface.
        see equation 4.94 of ward and hohmann (pg. 220)
        Args:
            omega (1d-array): set of angular frequencies
            sigma    (float): halfspace conductivity
            r        (float): radius of the loop
        Returns:
            Hz    (1d-array): magnetic field strength
    """
    k = calc_wavenumber(omega, sigma)

    Hz = 3-(3+3j*k*r - (r*k)**2)*nt.exp(-1j*k*r)
    Hz *= -I / (k**2 * r**3)
    return Hz



def freespace_u0(omega, wave_length, sqrt=True):
    """compute the vertical component of the wavenumber in freespace.
       this function automatically handles the broadcasting to get all combinations of wave_lengths and frequencies.     
    Args:
        omega       (nd-array): angular frequency
        wave_length (nd-array): wavelength

    Returns:
        u0: vertical wavenumber with shape = omega.shape+wave_length.shape
    """
    k2 = mu0*epsilon0*omega**2
    u02 = wave_length**2 - k2
    if sqrt: u02=nt.sqrt(u02+0j)
    return u02


def refl_transverse_electric(sigma, thickness, omega, wave_length, mu=mu0, epsilon=epsilon0):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance

    Args:
        sigma       (1d-array): conductivity of each layer
        thickness   (1d-array): thickness of each layer
        wave_length (1d-array): wave_length (lambda)
        omega       (1d-array): angular frequency
        mu             (float): _description_. Defaults to mu0.
        epsilon        (float): _description_. Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    if isinstance(mu, float):
        mu = nt.ones_like(sigma, dtype=nt.complex128, device=sigma.device)*mu
    if isinstance(epsilon, float):
        epsilon = nt.ones_like(sigma, dtype=nt.complex128, device=sigma.device)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    adm_freespace = nt.sqrt(lambda2 - omega**2 * mu0*epsilon0 + 0j)/(1j*omega*mu0)
    
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    zn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[0]

    adm_intrinsic = nt.sqrt(lambda2 - k2)
    tanh_un_thick = nt.tanh(adm_intrinsic[:-1]*thickness)
    adm_intrinsic/= zn
    
    adm_boundary = adm_intrinsic[-1]*1.
    
    for i in range(n_layers-2, -1, -1):
        adm_boundary = adm_intrinsic[i] \
                        *(adm_boundary + adm_intrinsic[i]*tanh_un_thick[i]) \
                        /(adm_intrinsic[i]  + adm_boundary*tanh_un_thick[i])

    rte = (adm_freespace - adm_boundary) / (adm_freespace + adm_boundary)
    return rte

def refl_transverse_magnetic(sigma, thickness, omega, wave_length, mu=mu0, epsilon=epsilon0):
    """ rtm = (Z_0 - hat Z_1)/(Z_0 + hat Z_1) - W&H eq 4.20 (pg 206)
        where Z_0 is the intrinsic impedance
        and hat Z_1 is the surface or boundary impedance

    Args:
        sigma (1d-array): conductivity of each layer
        thickness (1d-array): thickness of each layer
        wave_length (1d-array): wave_length (lambda)
        omega (1d-array): angular frequency
        mu (_type_, optional): _description_. Defaults to mu0.
        epsilon (_type_, optional): _description_. Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse magnetic mode
    """
    if isinstance(mu, float):
        mu = nt.ones_like(sigma, dtype=nt.complex128)*mu
    if isinstance(epsilon, float):
        epsilon = nt.ones_like(sigma, dtype=nt.complex128)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    

    
    imp_freespace = nt.sqrt(lambda2 - omega**2 * mu0*epsilon0)/(1j*omega*epsilon0)
    
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    yn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[1]

    imp_intrinsic  = nt.sqrt(lambda2 - k2)
    tanh_un_thick  = nt.tanh(imp_intrinsic[:-1]*thickness)
    imp_intrinsic /=  yn

    imp_boundary  = nt.zeros_like(imp_intrinsic)
    imp_boundary[-1] = imp_intrinsic[-1]*1.

        
    
    for i in range(n_layers-2, -1, -1):
        imp_boundary[i] = imp_intrinsic[i] \
                        *(imp_boundary[i+1] + imp_intrinsic[i]*tanh_un_thick[i]) \
                        /(imp_intrinsic[i]  + imp_boundary[i+1]*tanh_un_thick[i])

    rtm = (imp_freespace - imp_boundary[0]) / (imp_freespace + imp_boundary[0])
    return rtm
