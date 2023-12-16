from em1d import transforms
from em1d import physics
import numpy as np

def calc_rte_halfspace(omega, sigma, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(omega, sigma)**2
    u1          = np.sqrt(lambda_sqrd-K1_sqrd)
    rte         = 1./(wave_length+u1)
    return rte


def test_frequency_response():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = np.logspace(-6, 5, 501)  # Hz
    omega = 2*np.pi*frqs           # radians
    
    hz_analytical = physics.halfspace_vmd(sigma, omega, rho)

    hz = np.zeros(len(frqs), dtype=complex)
    for i_frq, freq in enumerate(frqs):
        freq *= 2*np.pi
        def rte1(wave_length):
            return calc_rte_halfspace(freq, sigma, wave_length)*wave_length**3
        Hkl_rte1 = transforms.Hankel(rte1, 0)
        hz[i_frq] = Hkl_rte1(rho)
    hz /= (2*np.pi)
    assert np.allclose(hz, hz_analytical), "hankel transform results do not match analytical solution"

if __name__ == '__main__':
    test_frequency_response()