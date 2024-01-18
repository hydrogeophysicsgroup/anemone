from em1d import transforms
from em1d import physics
from em1d import experiment
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
    hz = (Hkl_rte1(rho)/(2*torch.pi)).numpy()

    assert np.allclose(hz, hz_analytical), "hankel transform results do not match analytical solution"


def test_transforms_varying_samples_per_decades():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = torch.logspace(-6, 5, 501, dtype=torch.complex128)  # Hz
    omega = 2*torch.pi*frqs[:, None]

    def rte1(wave_length):
        wave_length = wave_length[None, :]
        return calc_rte_halfspace(sigma, omega, wave_length)*wave_length**3

    Hkl_rte1 = transforms.Hankel(rte1, 0)
    hz1 = Hkl_rte1(rho)/(2*torch.pi)

    Hkl_rte2 = transforms.Hankel(rte1, 0, smp_per_decade=20)
    hz2 = Hkl_rte2(rho)/(2*torch.pi)


    assert np.allclose(hz1.numpy(), hz2.numpy()), "changing the number of decades is changing the results of the transform"


if __name__ == '__main__':
    test_frequency_response()
    test_transforms_varying_samples_per_decades()