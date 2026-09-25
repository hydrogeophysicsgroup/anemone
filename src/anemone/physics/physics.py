# so there are a few dimensions of parameters required to compute the TEM forward model.
# 1. depth       (z)
# 2. frequency   (omega)
# 3. wave length (lambda)
# we assume the physical properties of the earth only change with depth, i.e. the earth is 1D.
# the order of arguments will be sigma, omega, wavelength

import torch
from anemone import constants as c, rdtype, cdtype

def calc_zy(sigma, omega, epsilon=c.epsilon0, mu=c.mu0):
    """compute the impedivity (zhat) and admittivity (yhat)
       see W&H between eq. 1.15->1.16, reference to harrington, 1961
    """
    z = 1j*mu*omega
    y = sigma + 1j*epsilon*omega
    return z,y


def calc_wavenumber(sigma, omega, mu=c.mu0, epsilon=c.epsilon0, sqrt=True, low_frequency=True):
    """ get the intrinsic complex wavenumber 'k' 
        see equation 4.30 in Ward and Hohmann (pg 207)
    """
    if low_frequency: k = -1j*sigma*mu*omega
    else:             k = mu*omega*(epsilon*omega - 1j*sigma)
    
    if sqrt: k = torch.sqrt(k)
    return k

def rte_raiche1999(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    """ raiche 1999 eq 7 and 8
    """
    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    Fj = 0.
    sj1 = (lambda2 + 1j*omega*c.mu0*sigma[-1])**0.5

    for i in range(n_layers-2, -1, -1):
        sj = (lambda2 +  1j*omega*c.mu0*sigma[i])**0.5
        RTEj = 1j*omega*c.mu0*(sigma[i] - sigma[i+1])/(sj + sj1)**2
        exp2uh = torch.exp(-2.*sj*thickness[i])
        Fj = exp2uh * (Fj + RTEj) / (1. + Fj * RTEj)
        sj1 = sj*1.
    RTEj = 1j*omega*c.mu0*(-sigma[0])/(sj + wave_length)**2
    rte = (RTEj+Fj) / (1.+RTEj*Fj)
    del RTEj, sj1, sj, exp2uh, lambda2
    return torch.nan_to_num(rte)


def rte_halfspace(sigma, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*mu
    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*epsilon
    
    u = torch.sqrt(1. + 1j*sigma*mu*omega/wave_length**2)
    rte = (1 - u) / (1 + u)
    return torch.nan_to_num(rte)

def rte(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    if thickness is None: 
        return rte_halfspace(sigma, omega, wave_length, mu=mu, epsilon=epsilon)
    else:
        return rte_raiche1999(sigma, thickness, omega, wave_length, mu=mu, epsilon=epsilon)


def freespace_u0(omega, wave_length, sqrt=True):
    """compute the vertical component of the wavenumber in freespace.
       this function automatically handles the broadcasting to get all combinations of wave_lengths and frequencies.     
    Args:
        omega       (nd-array): angular frequency
        wave_length (nd-array): wavelength

    Returns:
        u0: vertical wavenumber with shape = omega.shape+wave_length.shape
    """
    k2 = c.mu0*c.epsilon0*omega**2
    u02 = wave_length**2 - k2
    if sqrt: u02=torch.sqrt(u02+0j)
    return u02

def colecole(rho_o, m, tau, C, omega):
    """equation 1 from Pelton et al. 1978
    """
    tmp = 1.+(1j*omega*tau)**C
    tmp = 1. - 1./tmp
    tmp = 1. - m*tmp
    rho = rho_o*tmp
    return rho

def mpa2colecole(rho, phi_max, tau_phi, C):
    n = 0
    m0 = torch.zeros_like(phi_max)
    m_new = torch.ones_like(phi_max)
    max_m0 = 0.99
    
    tan_phi = torch.tan(-phi_max).real

    while torch.any(torch.abs(m_new - m0)/m_new > 1e-3) and n<100:
        if n>0: m0 = m_new*1.
        tau_cc = tau_phi*((1.-m0)**(-0.5/C))
        ab = 1./(1.+(1j*tau_cc/tau_phi)**C)
        m_new = tan_phi / ((1.-ab.real)*tan_phi + ab.imag)
        m_new[m_new>max_m0] = max_m0
        m_new[m_new<0] = 0.
        n += 1
    return rho, m_new, tau_cc,  C

def colecole2mpa(rho, m, tau_cc, C):
    tau_phi = tau_cc * (1-m)**(0.5/C)
    rho_cmplx = colecole(rho, m, tau_cc, C, 1./tau_phi)
    phi_max =  -torch.arctan(rho_cmplx.imag/rho_cmplx.real)
    return rho, phi_max, tau_phi, C    

def drte_dsigma(sigma, thickness, omega, wave_length):
    """
    Analytic drte/dsigma via Fortran TEM1DRESP.for.
    https://github.com/hydrogeophysicsgroup/TEM1D/blob/main/TEM1DRESP.for#L860C1-L942C12

    Index convention (Fortran -> Python):
      W(0..NLAY): W(0)=wave_length (air), W(1..NLAY)=u for conductors+halfspace.
      IPARM=1..NLAY, IPP=IPARM used directly as array index into W/G/R/E.
      sigma[0..NLAY-1] = Fortran SIGS(1..NLAY), SIGS(0)=0 (air).
    Layer axis is 0.
    """
    NLAY = len(sigma)

    # --- Build extended W(0..NLAY): prepend air wavenumber W(0)=wave_length ---
    u_cond = (wave_length**2 + 1j*omega*c.mu0*sigma)**0.5         
    u = torch.cat([wave_length.expand_as(u_cond[:1]), u_cond], dim=0)

    G = torch.zeros_like(u)
    R = torch.zeros_like(u)
    E = torch.zeros_like(u)

    # --- Forward pass: vectorise R and E, keep G sequential (Möbius chain) ---
    H_fwd = u[:-1] + u[1:]                                             
    dsigs = torch.cat([-sigma[:1], sigma[:-1] - sigma[1:]], dim=0)     # SIGS(i)-SIGS(i+1)
    R[1:] = c.mu0*1j*omega * dsigs / H_fwd**2

    thks = torch.cat([torch.zeros_like(thickness[:1]), thickness], dim=0)  
    E[:-1] = torch.exp(-2.*u[:-1]*thks)

    for i in range(NLAY - 1, -1, -1):
        G[i] = E[i] * (G[i+1] + R[i+1]) / (1. + G[i+1] * R[i+1])


    denom_k  = (1. + G[1:-1]*R[1:-1])**2

    # P[k] = E[k]*(1-R[k+1]^2) / (1+G[k+1]*R[k+1])^2
    P  = E[:-2] * (1. - R[1:-1]**2) / denom_k
    CP = torch.cat([torch.ones_like(P[:1]),
                    torch.cumprod(P, dim=0)], dim=0)               # (NLAY,   ...)

    # B[k] = DT_neg[k]*(1-G[k+1]^2)*E[k] / (1+G[k+1]*R[k+1])^2
    DT_neg = -2.*u[:-2] / (u[:-2] + u[1:-1])**2
    B = DT_neg * (1. - G[1:-1]**2) * E[:-2] / denom_k

    # DG_IPP[j] for IPP=j+1 in 1..NLAY-1
    H_ipp   = u[1:-1] + u[2:]
    DG_IPP  = (-2.*thks[1:]*G[1:-1]
               + 2.*u[2:]/H_ipp**2 * E[1:-1]*(1. - G[2:]**2)
                                              / (1. + G[2:]*R[2:])**2)

    # Combine: drte_du[IPP] = DG_IPP * CP[IPP] + B * CP[IPP-1]
    drte_du = torch.zeros_like(u)
    drte_du[1:NLAY] = DG_IPP * CP[1:] + B * CP[:-1]

    # Halfspace IPP=NLAY: E[NLAY-1] * DT_neg_half * CP[NLAY-1]
    drte_du[NLAY] = E[-2] * (-2.*u[-2] / (u[-1] + u[-2])**2) * CP[-1]

    drte_dsigma = (drte_du[1:] * 0.5*c.mu0*1j*omega / u_cond).squeeze()

    return drte_dsigma

def refl_transverse_electric(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance
        
        assumes nd-arrays have been appropriately broadcasted with axis ordered as:
        sigma, omega, wave_length

    Args:
        sigma         (nd-array): conductivity of each layer
        thickness     (nd-array): thickness of each layer
        omega         (nd-array): angular frequency
        wave_length   (nd-array): wave_length (lambda)
        mu      (float/nd-array): Defaults to mu0.
        epsilon (float/nd-array): Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*mu#.to(sigma.device)
        zn = 1.
    else:
        zn = None
    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*epsilon#.to(sigma.device)

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    adm_freespace = torch.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0 + 0j)/(1j*omega*c.mu0)
    
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    if zn is None:
        zn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[0]
    else:
        adm_freespace *= (1j*omega*c.mu0)

    # this is u_n (misnamed temporarily, will divide by zn soon.)
    adm_intrinsic = torch.sqrt(lambda2 - k2)
    tanh_un_thick = torch.tanh(adm_intrinsic[:-1]*thickness)
    adm_intrinsic/= zn
    
    adm_boundary = adm_intrinsic[-1]*1.
    
    for i in range(n_layers-2, -1, -1):
        adm_boundary = adm_intrinsic[i] \
                        *(adm_boundary + adm_intrinsic[i]*tanh_un_thick[i]) \
                        /(adm_intrinsic[i]  + adm_boundary*tanh_un_thick[i])

    rte = (adm_freespace - adm_boundary) / (adm_freespace + adm_boundary)
    return rte

def refl_transverse_electric_memopt(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance
        
        assumes nd-arrays have been appropriately broadcasted with axis ordered as:
        sigma, omega, wave_length

    Args:
        sigma         (nd-array): conductivity of each layer
        thickness     (nd-array): thickness of each layer
        omega         (nd-array): angular frequency
        wave_length   (nd-array): wave_length (lambda)
        mu      (float/nd-array): Defaults to mu0.
        epsilon (float/nd-array): Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*mu

    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    adm_freespace = torch.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0 + 0j)/(1j*omega*c.mu0)
    
    # k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    # zn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[0]

    # # this is u_n (misnamed temporarily, will divide by zn soon.)
    # adm_intrinsic = torch.sqrt(lambda2 - k2)
    # tanh_un_thick = torch.tanh(adm_intrinsic[:-1]*thickness)
    # adm_intrinsic/= zn

    c_k2 = calc_wavenumber(sigma[-1], omega[-1], mu=mu[-1], epsilon=epsilon[-1], sqrt=False)
    c_zn = calc_zy(sigma[-1], omega[-1], mu=mu[-1], epsilon=epsilon[-1])[0]
    c_adm_instrinsic = torch.sqrt(lambda2[-1]-c_k2) / c_zn

    adm_boundary = c_adm_instrinsic*1.
    
    for i in range(n_layers-2, -1, -1):
        c_k2 = calc_wavenumber(sigma[i], omega[-1], mu=mu[i], epsilon=epsilon[i], sqrt=False)
        c_zn = calc_zy(sigma[i], omega[-1], mu=mu[i], epsilon=epsilon[i])[0]
        c_adm_instrinsic = torch.sqrt(lambda2[-1]-c_k2)
        c_tanh_un_thick = torch.tanh(c_adm_instrinsic*thickness[i])
        c_adm_instrinsic /= c_zn        
        adm_boundary = c_adm_instrinsic \
                        *(adm_boundary + c_adm_instrinsic*c_tanh_un_thick) \
                        /(c_adm_instrinsic  + adm_boundary*c_tanh_un_thick)

    rte = (adm_freespace - adm_boundary) / (adm_freespace + adm_boundary)
    return rte


def refl_transverse_magnetic(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    """ rtm = (Z_0 - hat Z_1)/(Z_0 + hat Z_1) - W&H eq 4.20 (pg 206)
        where Z_0 is the intrinsic impedance
        and hat Z_1 is the surface or boundary impedance

        assumes nd-arrays have been appropriately broadcasted with axis ordered as:
        sigma, omega, wave_length
        
    Args:
        sigma         (nd-array): conductivity of each layer
        thickness     (nd-array): thickness of each layer
        omega         (nd-array): angular frequency
        wave_length   (nd-array): wave_length (lambda)
        mu      (float/nd-array): Defaults to mu0.
        epsilon (float/nd-array): Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse magnetic mode
    """
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype())*mu
    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype())*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    

    
    imp_freespace = torch.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0)/(1j*omega*c.epsilon0)
    
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    yn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[1]

    imp_intrinsic  = torch.sqrt(lambda2 - k2)
    tanh_un_thick  = torch.tanh(imp_intrinsic[:-1]*thickness)
    imp_intrinsic /=  yn

    imp_boundary  = torch.zeros_like(imp_intrinsic)
    imp_boundary[-1] = imp_intrinsic[-1]*1.

        
    
    for i in range(n_layers-2, -1, -1):
        imp_boundary[i] = imp_intrinsic[i] \
                        *(imp_boundary[i+1] + imp_intrinsic[i]*tanh_un_thick[i]) \
                        /(imp_intrinsic[i]  + imp_boundary[i+1]*tanh_un_thick[i])

    rtm = (imp_freespace - imp_boundary[0]) / (imp_freespace + imp_boundary[0])
    return rtm

      

def transverse_electric_field_coefficients(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0, return_un=False):
    import numpy as np

    sigma       = sigma.numpy()
    thickness   = thickness.numpy()
    omega       = omega.numpy()
    wave_length = wave_length.numpy()

    if isinstance(mu, float):
        mu = np.ones_like(sigma)*np.array(mu)
    if isinstance(epsilon, float):
        epsilon = np.ones_like(sigma)*np.array(epsilon)

    n_layers=len(sigma)+1

    lambda2 = wave_length**2
    
    u0 = np.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0 + 0j)
    
    k2 = mu*omega*(epsilon*omega - 1j*sigma)

    u_n   = np.sqrt(lambda2 - k2)
    euh_p = np.exp( u_n[:-1]*thickness)
    euh_m = np.exp(-u_n[:-1]*thickness)
    
    Amat = np.zeros((omega.size, wave_length.size, 2*n_layers, 2*n_layers), dtype=complex)
    Bmat = np.zeros((omega.size, wave_length.size, 2*n_layers,), dtype=complex)


    Amat[..., 0,0], Amat[..., 0,1], Amat[..., 0,2], Amat[..., 0,3] =  1.,  1.,    -1., -1.
    Amat[..., 1,0], Amat[..., 1,1], Amat[..., 1,2], Amat[..., 1,3] = -u0,  u0, u_n[0], -u_n[0]
    
    for i in range(1, n_layers-1):
        Amat[..., i*2,i*2]   = euh_m[i-1]
        Amat[..., i*2,i*2+1] = euh_p[i-1]
        Amat[..., i*2,i*2+2] = -1.
        Amat[..., i*2,i*2+3] = -1.

        Amat[..., i*2+1,i*2  ] = -u_n[i-1]*euh_m[i-1]
        Amat[..., i*2+1,i*2+1] =  euh_p[i-1]*u_n[i-1]

        Amat[..., i*2+1,i*2+2] =  u_n[i]
        Amat[..., i*2+1,i*2+3] = -u_n[i]

    Amat[..., -2,-1] = 1.
    Amat[..., -1, 0] = 1.

    Bmat[..., -1] = 1.
    Amat = np.nan_to_num(Amat)

    AB = np.linalg.solve(Amat, Bmat)
    res = (torch.from_numpy(np.moveaxis(AB[..., ::2], -1,0)), torch.from_numpy(np.moveaxis(AB[..., 1::2], -1,0)))
    if return_un:
        res = res + (torch.from_numpy(np.vstack((u0[None, ...], u_n))),)
    return res

def transverse_magnetic_field_coefficients(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0, return_un=False):
    import numpy as np

    sigma       = sigma.numpy()
    thickness   = thickness.numpy()
    omega       = omega.numpy()
    wave_length = wave_length.numpy()

    if isinstance(mu, float):
        mu = np.ones_like(sigma)*np.array(mu)
    if isinstance(epsilon, float):
        epsilon = np.ones_like(sigma)*np.array(epsilon)

    n_layers=len(sigma)+1

    lambda2 = wave_length**2
    
    u0 = np.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0 + 0j)
    
    k2 = mu*omega*(epsilon*omega - 1j*sigma)
    zhat,yhat = calc_zy(sigma, omega)
    yhat0 = yhat[0]-yhat[0].real

    u_n   = np.sqrt(lambda2 - k2)
    euh_p = np.exp( u_n[:-1]*thickness)
    euh_m = np.exp(-u_n[:-1]*thickness)
    
    Amat = np.zeros((omega.size, wave_length.size, 2*n_layers, 2*n_layers), dtype=complex)
    Bmat = np.zeros((omega.size, wave_length.size, 2*n_layers,), dtype=complex)


    Amat[..., 0,0], Amat[..., 0,1], Amat[..., 0,2], Amat[..., 0,3] =  1.,  1.,    -1., -1.
    Amat[..., 1,0], Amat[..., 1,1], Amat[..., 1,2], Amat[..., 1,3] = -u0/yhat0,  u0/yhat0, u_n[0]/yhat[0], -u_n[0]/yhat[0]
    
    for i in range(1, n_layers-1):
        Amat[..., i*2,i*2]   = euh_m[i-1]
        Amat[..., i*2,i*2+1] = euh_p[i-1]
        Amat[..., i*2,i*2+2] = -1.
        Amat[..., i*2,i*2+3] = -1.

        Amat[..., i*2+1,i*2  ] = -u_n[i-1]*euh_m[i-1]/yhat[i-1]
        Amat[..., i*2+1,i*2+1] =  euh_p[i-1]*u_n[i-1]/yhat[i-1]
        Amat[..., i*2+1,i*2+2] =  u_n[i]/yhat[i]
        Amat[..., i*2+1,i*2+3] = -u_n[i]/yhat[i]

    Amat[..., -2,-1] = 1.
    Amat[..., -1, 0] = 1.

    Bmat[..., -1] = 1.
    Amat = np.nan_to_num(Amat)

    AB = np.linalg.solve(Amat, Bmat)
    res = (torch.from_numpy(np.moveaxis(AB[..., ::2], -1,0)), torch.from_numpy(np.moveaxis(AB[..., 1::2], -1,0)))
    if return_un:
        res = res + (torch.from_numpy(np.vstack((u0[None, ...], u_n))),)
    return res

def rte_nbc(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance
        
        assumes nd-arrays have been appropriately broadcasted with axis ordered as:
        sigma, omega, wave_length

    Args:
        sigma         (nd-array): conductivity of each layer
        thickness     (nd-array): thickness of each layer
        omega         (nd-array): angular frequency
        wave_length   (nd-array): wave_length (lambda)
        mu      (float/nd-array): Defaults to mu0.
        epsilon (float/nd-array): Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*mu
        zn = 1.
    else:
        zn = None
    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*epsilon

    n_layers=len(sigma)

    lambda2 = wave_length**2
    
    adm_freespace = torch.sqrt(lambda2 - omega**2 * c.mu0*c.epsilon0 + 0j)/(1j*omega*c.mu0)
    
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    if zn is None:
        zn = calc_zy(sigma, omega, mu=mu, epsilon=epsilon)[0]
    else:
        adm_freespace *= (1j*omega*c.mu0)

    adm_intrinsic = torch.sqrt(lambda2 - k2)
    exp2uh = torch.exp(-2*adm_intrinsic[:-1]*thickness)
    adm_intrinsic/= zn
    
    gamma_n = 0.
    for i in range(n_layers-2, -1, -1):
        phi_n = (adm_intrinsic[i] - adm_intrinsic[i+1])/ (adm_intrinsic[i] + adm_intrinsic[i+1])
        gamma_n = exp2uh[i]*(gamma_n + phi_n)/(1. + gamma_n*phi_n)
    phi_n = (adm_freespace - adm_intrinsic[0])/(adm_freespace + adm_intrinsic[0])
    return phi_n

def rte_nbc_laplace(sigma, thickness, s, wave_length):
    """ rte = (Y_0 - hat Y_1)/(Y_0 + hat Y_1) - W&H eq 4.19 (pg 205)
        where Y_0 is the intrinsic admittance
        and hat Y_1 is the surface or boundary admittance
        
        assumes nd-arrays have been appropriately broadcasted with axis ordered as:
        sigma, omega, wave_length

    Args:
        sigma         (nd-array): conductivity of each layer
        thickness     (nd-array): thickness of each layer
        omega         (nd-array): angular frequency
        wave_length   (nd-array): wave_length (lambda)
        mu      (float/nd-array): Defaults to mu0.
        epsilon (float/nd-array): Defaults to epsilon0.

    Returns:
        reflection coefficients for transverse electric mode
    """
    n_layers=len(sigma)

    lambda2 = wave_length**2

    adm_intrinsic = torch.sqrt(lambda2 + c.mu0*s*sigma)
    exp2uh = torch.exp(-2.*adm_intrinsic[:-1]*thickness)
    
    gamma_n = 0.
    for i in range(n_layers-2, -1, -1):
        H = adm_intrinsic[i] + adm_intrinsic[i+1]
        R = c.mu0*s*(sigma[i] - sigma[i+1])/(H*H)
        gamma_n = exp2uh[i]*(gamma_n + R)/(1. + gamma_n*R)
    H = wave_length + adm_intrinsic[0]
    R = -c.mu0*s*sigma[0]/(H*H)
    return (R+gamma_n)/(1.+R*gamma_n)

def rte_lap(sigma, thickness, s, wave_length):
    sigma *= c.mu0

    n_layers=len(sigma)

    lambda2 = wave_length**2

    un = torch.sqrt(lambda2 + s*sigma[-1])
    gamma_n = 0.
    for i in range(n_layers-2, -1, -1):
        un1 = (lambda2 + s*sigma[i])**0.5
        exp2uh = torch.exp(-2.*un1*thickness[i])
        H = un+un1
        R = s*(sigma[i-1] - sigma[i])/(H*H)
        gamma_n = exp2uh*(gamma_n + R)/(1. + gamma_n*R)
        un = un1*1.
    H = wave_length + un1
    R = -s*sigma[0]/(H*H)
    return (R+gamma_n)/(1.+R*gamma_n)

def calc_prop_matrix(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0, pad=0):
    thick = torch.vstack([torch.zeros_like(thickness[:1]), thickness])

    lambda2 = wave_length**2
    k2 = calc_wavenumber(sigma, omega, mu=mu, epsilon=epsilon, sqrt=False)
    un2 = torch.sqrt(lambda2 - k2)
    # u0 = freespace_u0(omega, wave_length)
    un1 = torch.roll(un2, 1, dims=0)
    un1[0] = wave_length
    # un2 = un*1.

    euh_p = torch.exp(un1*thick)

    euh_m   = 1./euh_p
    unratio = un2/un1
    d_un    = 1.-unratio
    m_un    = 1.+unratio

    propmat = torch.zeros((2,2, m_un.shape[0]+pad)+m_un.shape[1:], dtype=cdtype(), device=sigma.device)
    if pad>0:
        propmat[0,0,:pad] = 1.
        propmat[1,1,:pad] = 1.

        propmat[0, 0, pad:] = m_un*euh_p
        propmat[0, 1, pad:] = d_un*euh_p
        propmat[1, 0, pad:] = d_un*euh_m
        propmat[1, 1, pad:] = m_un*euh_m
    else:
        propmat[0, 0] = m_un*euh_p
        propmat[0, 1] = d_un*euh_p
        propmat[1, 0] = d_un*euh_m
        propmat[1, 1] = m_un*euh_m
    return propmat

def calc_dpropmat(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):

    pad = 2

    propmat = calc_prop_matrix(sigma, thickness, omega, wave_length, mu=mu, epsilon=epsilon, pad=pad)#.to(sigma.device)
    propmat = torch.roll(propmat, -pad//2, dims=2)

    answer = torch.zeros((2,)+propmat.shape[2:], dtype=cdtype(), device=thickness.device)
    
    answer[0, :] = 1.
    for i in range(answer.shape[1]-2, -1, -1):
        answer[:, i] = torch.einsum('ij...,j...->i...', propmat[:,:,i], answer[:, i+1])

    for i in range(1, propmat.shape[2]):
        propmat[:,:,i] = torch.einsum('ij...,jk...->ik...', propmat[:,:, i-1], propmat[:,:, i])

    rte = answer[1, 0] / answer[0, 0]

    answer = torch.roll(answer, -1, dims=1)
    answer[0,-1:] = 1.
    answer[1,-1:] = 0.

    # rho = 1./sigma
    # drho = (0.02*rho.log()).exp()
    # dsig = 1./drho
    # # print(dsig)
        
    dsigma = sigma.clone()
    sigma_pert = 0.02*dsigma
    dsigma[::2] -= sigma_pert[::2]
    dpropmat = calc_prop_matrix(dsigma, thickness, omega, wave_length, mu=mu, epsilon=epsilon, pad=pad)
    dpropmat = torch.roll(dpropmat, -pad//2, dims=2)

    drte = torch.einsum('ij...,j...->i...', dpropmat, answer)
    drte = torch.roll(drte, -1, dims=1)
    drte[0,-1] = 1.
    drte[1,-1] = 0.
    drte = torch.einsum('ij...,j...->i...', dpropmat, drte)
    drte = torch.roll(drte, -1, dims=1)
    drte[0,-1] = 1.
    drte[1,-1] = 0.
    drte = torch.einsum('ij...,j...->i...', propmat, drte)

    dsigma = sigma.clone()
    dsigma[1::2] -= sigma_pert[1::2]
    dpropmat = calc_prop_matrix(dsigma, thickness, omega, wave_length, mu=mu, epsilon=epsilon, pad=pad)
    dpropmat = torch.roll(dpropmat, -pad//2, dims=2)
    
    drte2 = torch.einsum('ij...,j...->i...', dpropmat, answer)
    drte2 = torch.roll(drte2, -1, dims=1)
    drte2[0,-1] = 1.
    drte2[1,-1] = 0.
    drte2 = torch.einsum('ij...,j...->i...', dpropmat, drte2)
    drte2 = torch.roll(drte2, -1, dims=1)
    drte2[0,-1] = 1.
    drte2[1,-1] = 0.
    drte2 = torch.einsum('ij...,j...->i...', propmat,  drte2)
    drte[:, 1::2] = drte2[:, 1::2]

    mid = slice(None,len(sigma))
    return (drte[1, mid]/drte[0, mid] - rte)/sigma_pert, rte

def rte_matrix(sigma, thickness, omega, wave_length, mu=c.mu0, epsilon=c.epsilon0):
    if isinstance(mu, float):
        mu = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*mu
    if isinstance(epsilon, float):
        epsilon = torch.ones_like(sigma, dtype=cdtype(), device=sigma.device)*epsilon
    _,rte = calc_dpropmat(sigma, thickness, omega, wave_length, mu=mu, epsilon=epsilon)
    return torch.nan_to_num(rte)


def halfspace_vmd(rho, omega, r):
    """get the analytic response for a vertical magnetic dipole at the surface of the earth
       see equation 4.56 of ward and hohmann (pg.211)

        Args:
            rho    (float): halfspace resistivity
            omega (1d-array): the angular frequency
            r        (float): horizontal distance from the dipole

        Returns:
            z-component of the magnetic field
    """
    sigma = 1./rho
    k = calc_wavenumber(sigma, omega)
    Hz = 9 - (9 + 9j*k*r - 4*k**2 * r**2 -1j*k**3*r**3)*torch.exp(-1j*k*r)
    Hz /= (2*torch.pi*k**2 * r**5)
    return Hz

def halfspace_horizontal_electric_dipole(rho, omega, r):
    """this needs to be confirmed.

        Args:
            rho    (float): halfspace resistivity
            omega (1d-array): the angular frequency
            r        (float): horizontal distance from the dipole

        Returns:
            z-component of the magnetic field
    """
    UserWarning("Hz from a horizontal electric dipole over a halfspace should be confirmed.\nUse at your own risk.")
    sigma = 1./rho
    k = calc_wavenumber(sigma, omega)

    Hz = 3 - (3 + 3j*k*r - (r*k)**2)*torch.exp(-1j*k*r)

    Hz /= -(2*torch.pi*k**2 * r**5)
    return Hz

def hz_halfspace_center_circle_loop(rho, omega, r, I=1.):
    """ get the vertical component of the response at the center of a circular loop on the earth's surface.
        see equation 4.94 of ward and hohmann (pg. 220)
        Args:
            rho    (float): halfspace resistivity
            omega (1d-array): set of angular frequencies
            r        (float): radius of the loop
        Returns:
            Hz    (1d-array): magnetic field strength
    """
    sigma = 1./rho
    k = calc_wavenumber(sigma, omega)

    Hz = 3-(3+3j*k*r - (r*k)**2)*torch.exp(-1j*k*r)
    Hz *= -I / (k**2 * r**3)
    return Hz

def hz_halfspace_center_circle_loop_time(times, rho, r, I=1.):
    sigma = 1./rho
    theta = torch.sqrt(0.25*c.mu0*sigma / times)
    theta_r = theta*r

    hz = 3*torch.exp(-theta_r**2)/(r*theta*c.sqrt_pi) + (1 - 3 / (2*theta_r**2))*torch.erf(theta_r)
    hz *= I/(2*r)
    return hz


def hzdt_halfspace_center_circle_loop(times, rho, r, I=1.):
    sigma = 1./rho
    theta = torch.sqrt(0.25*c.mu0*sigma / times)
    dhdt = 1./(c.mu0*sigma*r**3)*(3*torch.erf(theta*r)-2*theta*r*(3+2*theta**2*r**2)*torch.exp(-theta**2*r**2)/c.sqrt_pi)
    return dhdt

def dbdt_center_of_circular_loop_halfspace(time, rho, r):
    if max(time)>1e-3: 
        return dbdt_center_of_circular_loop_halfspace_extended(time, rho, r)
    else:
        sigma = 1./rho
        tau = r*torch.sqrt(0.25*sigma*c.mu0/time)
        term1 = 3.0*torch.erf(tau)
        term2 = (2.0/c.sqrt_pi)*tau*(3.0 + 2.0*tau**2)*torch.exp(-tau**2)
        dbdt = (term1 - term2)/(sigma*r**3)
        return dbdt

def dhdt_center_of_circular_loop_late_time_approx(times, rho, radius):
    sigma = 1./rho
    return sigma**(3/2)*c.mu0**(3/2) * radius**2 * times**(-5/2.) / 20. / c.sqrt_pi


def dbdt_center_of_circular_loop_halfspace_extended(times, rhos, r, late_time=1e-3):
    import math
    sqrt_pi = float(torch.pi**0.5)
    mu0 = float(torch.pi*4e-7)    
    dbdt = torch.empty(rhos.shape+times.shape)
    for i_rho, rho in enumerate(rhos):
        sigma = 1./rho
        for i_t,time in enumerate(times[times<late_time]):    
            sigma            = 1./rho
            theta            = math.sqrt(0.25*mu0*sigma / time)
            dhdt             = 1./(mu0*sigma*r**3)*(3*math.erf(theta*r)-2*theta*r*(3+2*theta**2*r**2)*math.exp(-theta**2*r**2)/sqrt_pi)
            dbdt[i_rho, i_t] = dhdt
    
    # analytic solution breaks down around 1 ms, 
    # so we use the known asymptotic behavior to extend it to later times
    
    i_t = torch.searchsorted(times, late_time)
    dbdt[:, i_t:] = dhdt_center_of_circular_loop_late_time_approx(times[None,i_t:], rhos[:, None], r)

    return dbdt*mu0    

def dbzdt_halfspace_coincident_circle_loop(times, rho, r, I=1., max_terms=2000):
    """
    W&H Eq. 4.101
    """
    sigma = 1. / rho
    # Use float64 throughout for extended range
    times64 = times.double()
    theta = torch.sqrt(0.25 * c.mu0 * sigma[:, None] / times64[None,:])
    tr = theta * r
    x2 = tr ** 2

    term = torch.ones_like(x2) / 5.
    summation = term.clone()
    for m in range(max_terms - 1):
        ratio = -(2*m+4) * (2*m+3) * (2*m+5) * x2 / ((m+1) * (m+2) * (m+3) * (2*m+7))
        term = term * ratio
        summation = summation + term
        if torch.all(torch.abs(term / summation) < 1e-12):
            break

    dbdt = 2 * I * r * c.mu0 * c.sqrt_pi * tr**3 * summation / times64[None,:]
    return dbdt.to(times.dtype)

from mpmath import mp, mpf, sqrt, fabs

def dbzdt_halfspace_coincident_circle_loop(times, rho, r, I=1., max_terms=2000, prec=100):
    """W&H Eq. 4.101 — mpmath arbitrary precision."""
    print('arbitrary precision floats')

    mp.dps = prec  # decimal digits of precision

    mu0     = mpf('1.25663706212e-6')      # or mpf(c.mu0)
    sqrt_pi = sqrt(mp.pi)
    r_mp    = mpf(r)
    I_mp    = mpf(I)

    result = [[mpf(0)] * len(times) for _ in range(len(rho))]

    for i, p in enumerate(rho):
        sigma = mpf(1) / mpf(float(p))
        for j, t in enumerate(times):
            t_mp  = mpf(float(t))
            theta = sqrt(mpf('0.25') * mu0 * sigma / t_mp)
            tr    = theta * r_mp
            x2    = tr * tr

            term = mpf(1) / 5
            summation = term
            for m in range(max_terms - 1):
                ratio = (-(2*m + 4) * (2*m + 3) * (2*m + 5) * x2
                         / ((m + 1) * (m + 2) * (m + 3) * (2*m + 7)))
                term = term * ratio
                summation = summation + term
                if fabs(term / summation) < mpf(10) ** (-prec + 2):
                    print(f'broke at {m}', flush=True)
                    break

            result[i][j] = 2 * I_mp * r_mp * mu0 * sqrt_pi * tr**3 * summation / t_mp

    # cast back to float / tensor at the end
    return torch.tensor([[float(v) for v in row] for row in result])