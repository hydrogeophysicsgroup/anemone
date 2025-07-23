from anemone import transforms
from anemone import physics
import numpy as np
import torch

import pytest


def calc_rte_halfspace(sigma, omega, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(sigma, omega)**2
    u1          = torch.sqrt(lambda_sqrd-K1_sqrd)
    rte         = 1./(wave_length+u1)
    return rte


def test_frequency_response():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = torch.logspace(-6, 5, 501, dtype=torch.complex128)  # Hz
    omega = 2*torch.pi*frqs
    
    hz_analytical = physics.halfspace_vmd(1/sigma, omega, rho)
    omega = omega[:, None]

    def rte1(wave_length):
        wave_length = wave_length[None, :]
        return calc_rte_halfspace(sigma, omega, wave_length)*wave_length**3
    
    Hkl_rte1 = transforms.Hankel(rte1, 0)
    r,resp = Hkl_rte1.range(90, 110)
    cbResp = transforms.CubicSpline(r, resp)
    hz = (cbResp(rho)/(2*torch.pi))
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.plot(frqs, hz_analytical, 'k-')
        plt.plot(frqs, hz, 'r--')
        plt.xlabel('r (m)')
        plt.ylabel('H(r)')
        plt.show()

    assert torch.allclose(hz, hz_analytical), "hankel transform results do not match analytical solution"

def check_rte_convergence(rte_func):
    from anemone import constants as c

    print('it would be wise to expand the range the frequencies or adjust the conductivity so that it does not converge so fast')
    sigma = torch.tensor([100, 0.5, 0.1], dtype=torch.complex128)
    omega = torch.tensor([1000], dtype=torch.complex128)*2*torch.pi


    wave_length   = torch.logspace(-3, 3, 1001, dtype=torch.complex128)
    lambda_sqrd   = wave_length*wave_length

    K0_sqrd       = c.epsilon0*c.mu0*omega**2
    K1_sqrd       = c.mu0*omega*(c.epsilon0*omega-1j*sigma[0])

    u0            = torch.sqrt(lambda_sqrd-K0_sqrd)
    u1            = torch.sqrt(lambda_sqrd-K1_sqrd)

    z0,y0 = physics.calc_zy(0., omega)
    z1,y1 = physics.calc_zy(sigma[0], omega)

    rte_halfspace = (u0-u1)/(u0+u1)

    for iz,thickness in enumerate(np.geomspace(1e-4, 1e1, 15)[:-1]):
        thick = torch.tensor([thickness, 10], dtype=torch.complex128)
        c_rte = torch.squeeze(physics.refl_transverse_electric(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:]))
        assert not np.allclose(rte_halfspace.numpy(), c_rte.numpy()), f'unexpected convergence at {thickness} m'
    
    thick = torch.tensor([100, 10], dtype=torch.complex128)
    c_rte = torch.squeeze(rte_func(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:]))        

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.semilogx(wave_length.numpy(), rte_halfspace.numpy(), 'k-', label='Halfspace')
        plt.semilogx(wave_length.numpy(), c_rte.numpy(), 'r--', label=rte_func.__name__)
        plt.xlabel('Wavelength (m)')
        plt.ylabel('RTE')
        plt.legend()
        plt.show()

    assert np.allclose(rte_halfspace.numpy(), c_rte.numpy()), f'{rte_func.__name__} did not converge as expected'

def test_rte_convergence():
    check_rte_convergence(physics.refl_transverse_electric)

def test_rte_memop_convergence():
    check_rte_convergence(physics.refl_transverse_electric_memopt)

def test_rte_convergence_nbc():
    check_rte_convergence(physics.rte_nbc)

@pytest.mark.skipif(reason="not used in main codebase")
def test_rte_convergence_nbc_laplace():
    check_rte_convergence(physics.rte_nbc_laplace)

def test_rtm_convergence():
    from anemone import constants as c
    sigma = torch.tensor([100, 0.5, 0.1], dtype=torch.complex128)
    omega = torch.tensor([1000], dtype=torch.complex128)*2*torch.pi


    wave_length   = torch.logspace(-3, 3, 1001, dtype=torch.complex128)
    lambda_sqrd   = wave_length*wave_length

    K0_sqrd       = c.epsilon0*c.mu0*omega**2
    K1_sqrd       = c.mu0*omega*(c.epsilon0*omega-1j*sigma[0])

    u0            = torch.sqrt(lambda_sqrd-K0_sqrd)
    u1            = torch.sqrt(lambda_sqrd-K1_sqrd)

    z0,y0 = physics.calc_zy(0., omega)
    z1,y1 = physics.calc_zy(sigma[0], omega)

    rtm_halfspace = ((u0 - y0*u1/y1) / (u0 + y0*u1/y1))#.numpy()

    for iz,thickness in enumerate(np.geomspace(1e-4, 1e-2, 15)):
        thick = torch.tensor([thickness, 10], dtype=torch.complex128)
        c_rte = torch.squeeze(physics.refl_transverse_magnetic(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:]))
        assert not np.allclose(rtm_halfspace.numpy(), c_rte.numpy()), f'unexpected convergence at {thickness} m'
    
    thick = torch.tensor([100, 10], dtype=torch.complex128)
    c_rte = torch.squeeze(physics.refl_transverse_magnetic(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:]))        
    assert np.allclose(rtm_halfspace.numpy(), c_rte.numpy()), 'rtm did not converge as expected'

if __name__ == '__main__':
    # test_frequency_response()
    test_rte_convergence()
    # test_rte_convergence_nbc_laplace()