from typing import Any
import numpy as np

mu0 = np.pi*4e-7
epsilon0 = 8.854187817e-12


def calc_wavenumber(omega, sigma, mu=mu0, epsilon=epsilon0):
    """get the intrinsic complex wavenumber 'k' 
       see equation 4.30 in ward in hohmann (pg 207)
    """
    return np.sqrt(mu*omega*(epsilon*omega - 1j*sigma))

def halfspace_vmd(omega, sigma, r):
    """get the analytic response for a vertical magnetic dipole at the surface of the earth
       see equation 4.56 of ward and hohmann (pg.211)

        Args:
            omega (1d-array): the angular frequency
            sigma (float):    halfspace conductivity
            r (float):        horizontal distance from the dipole

        Returns:
            z-component of the magnetic field
    """
    k = calc_wavenumber(omega, sigma)
    Hz = 9 - (9 + 9j*k*r - 4*k**2 * r**2 -1j*k**3*r**3)*np.exp(-1j*k*r)
    Hz /= (2*np.pi*k**2 * r**5)
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

    Hz = 3-(3+3j*k*r - (r*k)**2)*np.exp(-1j*k*r)
    Hz *= -I / (k**2 * r**3)
    return Hz



def freespace_u0(omega, wave_length):
    """compute the vertical component of the wavenumber in freespace.
     
    Args:
        omega (_type_): _description_
        wave_length (_type_): _description_

    Returns:
        _type_: _description_
    """
    k2 = mu0*epsilon0*omega**2
    return np.squeeze(np.sqrt(wave_length[None,:]**2 - k2[:,None]))


def refl_transverse_electric(sigma, thickness, wave_length, omega, mu=mu0, epsilon=epsilon0):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance

    Args:
        sigma (1d-array): conductivity of each layer
        thickness (1d-array): thickness of each layer
        wave_length (1d-array): wave_length (lambda)
        omega (1d-array): angular frequency
        mu (_type_, optional): _description_. Defaults to mu0.
        epsilon (_type_, optional): _description_. Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    if isinstance(mu, float):
        mu = np.ones_like(sigma)*mu
    if isinstance(epsilon, float):
        epsilon = np.ones_like(sigma)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    adm_boundary  = np.zeros(sigma.shape+omega.shape+wave_length.shape, dtype=complex)
    adm_intrinsic = np.zeros(sigma.shape+omega.shape+wave_length.shape, dtype=complex)
    
    adm_freespace = np.sqrt(lambda2[None, :] - omega[:,None]**2 * mu0*epsilon0)/(1j*omega[:, None]*mu0)

    
    for i in range(n_layers):
        adm_intrinsic[i] = np.sqrt(lambda2[None, :] - (calc_wavenumber(omega, sigma[i])**2)[:, None])/(1j*omega[:, None]*mu[i])
    
    
    adm_boundary[-1] = adm_intrinsic[-1]*1.
    tanh_un_thick = np.tanh(adm_intrinsic[:-1]*thickness[:, None, None]*(1j*omega[None, :, None]*mu[:-1, None, None]))
    for i in range(n_layers-2, -1, -1):
        adm_boundary[i] = adm_intrinsic[i] \
                        *(adm_boundary[i+1] + adm_intrinsic[i]*tanh_un_thick[i]) \
                        /(adm_intrinsic[i]  + adm_boundary[i+1]*tanh_un_thick[i])
    rte = (adm_freespace - adm_boundary[0]) / (adm_freespace + adm_boundary[0])
    return rte

# def refl_transverse_electric(sigma, thickness, wave_length, omega, mu=mu0, epsilon=epsilon0):
#     """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
#         where Y_0 is the intrinsic admittance
#         and hat Y_1 is the surface or boundary admittance

#     Args:
#         sigma (1d-array): conductivity of each layer
#         thickness (1d-array): thickness of each layer
#         wave_length (1d-array): wave_length (lambda)
#         omega (1d-array): angular frequency
#         mu (_type_, optional): _description_. Defaults to mu0.
#         epsilon (_type_, optional): _description_. Defaults to epsilon0.

#     Returns:
#         reflection coefficients for transverse electric mode
#     """

#     n_layers=len(sigma)

#     lambda2 = wave_length**2
    
#     adm_boundary  = np.zeros(sigma.shape+wave_length.shape, dtype=complex)
#     adm_intrinsic = np.zeros(sigma.shape+wave_length.shape, dtype=complex)
    
#     adm_freespace = np.sqrt(lambda2 - omega**2 * mu0*epsilon0)/(1j*omega*mu)
    
#     for i in range(n_layers):
#         adm_intrinsic[i] = np.sqrt(lambda2 - (calc_wavenumber(omega, sigma[i])**2))/(1j*omega*mu)
    
    
#     adm_boundary[-1] = adm_intrinsic[-1]*1.
#     tanh_un_thick = np.tanh(adm_intrinsic[:-1]*thickness[:, None]*(1j*omega*mu))
#     for i in range(n_layers-2, -1, -1):
#         adm_boundary[i] = adm_intrinsic[i] \
#                         *(adm_boundary[i+1] + adm_intrinsic[i]*tanh_un_thick[i]) \
#                         /(adm_intrinsic[i]  + adm_boundary[i+1]*tanh_un_thick[i])
        
#     rte = (adm_freespace - adm_boundary[0]) / (adm_freespace + adm_boundary[0])
#     return rte


def refl_transverse_magnetic(sigma, thickness, wave_length, omega, mu=mu0, epsilon=epsilon0):
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
        mu = np.ones_like(sigma)*mu
    if isinstance(epsilon, float):
        epsilon = np.ones_like(sigma)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    imp_boundary  = np.zeros(sigma.shape+omega.shape+wave_length.shape, dtype=complex)
    imp_intrinsic = np.zeros(sigma.shape+omega.shape+wave_length.shape, dtype=complex)

    
    for i in range(n_layers):
        imp_intrinsic[i] = np.sqrt(lambda2[None, :] - calc_wavenumber(omega, sigma[i])[:, None])/(1j*omega*epsilon[i])
    
    imp_boundary[-1] = imp_intrinsic[-1]
    tanh_un_thick = np.tanh(imp_intrinsic*thickness[:, None, None]*(1j*omega[None, :, None]*epsilon[:, None, None]))
    for i in range(n_layers-1, -1, -1):
        imp_boundary[i] = imp_intrinsic[i] \
                        *(imp_boundary[i+1] + imp_intrinsic[i]*tanh_un_thick[i]) \
                        /(imp_intrinsic[i]  + imp_boundary[i+1]*tanh_un_thick[i])
    rtm = (wave_length[None, None, :] - imp_boundary[0]) / (wave_length[None, None, :] + imp_boundary[0])
    return rtm



class Dipole:
    def __init__(self, rte_func, rtm_func):
        pass

    def __call__(self, sigma, thickness, h=0.):
        pass