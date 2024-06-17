from em1d import transforms
from em1d import physics
import numpy as np
import torch

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
    
    hz_analytical = physics.halfspace_vmd(sigma, omega, rho).numpy()
    omega = omega[:, None]

    def rte1(wave_length):
        wave_length = wave_length[None, :]
        return calc_rte_halfspace(sigma, omega, wave_length)*wave_length**3
    Hkl_rte1 = transforms.Hankel(rte1, 0)
    r,resp = Hkl_rte1.range(90, 110)
    cbResp = transforms.CubicSpline(r, resp)
    hz = (cbResp(rho)/(2*torch.pi)).numpy()

    assert np.allclose(hz, hz_analytical), "hankel transform results do not match analytical solution"

def test_rte_convergence():
    from em1d import constants as c

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
    c_rte = torch.squeeze(physics.refl_transverse_electric(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:]))        
    assert np.allclose(rte_halfspace.numpy(), c_rte.numpy()), 'rte did not converge as expected'

def test_rtm_convergence():
    from em1d import constants as c
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
    test_rtm_convergence()