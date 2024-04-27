from em1d import transforms
from em1d import physics
from em1d import experiment

import numpy as np

def calc_rte_halfspace(omega, sigma, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(omega, sigma)**2
    u1          = np.sqrt(lambda_sqrd[None,:]-K1_sqrd[:,None])
    rte         = 1./(wave_length+u1)
    return np.squeeze(rte)


def test_frequency_response():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = np.logspace(-6, 5, 501)  # Hz
    omega = 2*np.pi*frqs           # radians
    
    hz_analytical = physics.halfspace_vmd(omega, sigma, rho)

    def rte1(wave_length):
        return calc_rte_halfspace(omega, sigma, wave_length)*wave_length[None,:]**3
    Hkl_rte1 = transforms.Hankel(rte1, 0)
    hz = Hkl_rte1(rho)/(2*np.pi)

    assert np.allclose(hz, hz_analytical), "hankel transform results do not match analytical solution"

def test_transforms_varying_samples_per_decades():
    rho = 100.                   # meters
    sigma = 1./100.            # 1/ohm-m
    frqs = np.logspace(-6, 5, 501)  # Hz
    omega = 2*np.pi*frqs           # radians
    
    def rte1(wave_length):
        return calc_rte_halfspace(omega, sigma, wave_length)*wave_length[None,:]**3
    Hkl_rte1 = transforms.Hankel(rte1, 0)
    hz1 = Hkl_rte1(rho)/(2*np.pi)

    Hkl_rte2 = transforms.Hankel(rte1, 0, smp_per_decade=20)
    hz2 = Hkl_rte2(rho)/(2*np.pi)


    assert np.allclose(hz1, hz2), "changing the number of decades is changing the results of the transform"

def test_spline():
    x = np.linspace(0, 10, 4*80+1)
    xs = x[::4]

    f = lambda x: 1+x**2 + 2*x**3

    ys = f(xs)
    ys = ys[None, :] + np.arange(650)[:, None]

    y = f(x)
    y = y[None, :] + np.arange(650)[:, None]

    cspl = transforms.CubicSpline(xs,ys.T)
    y2 = cspl(x).T
    assert np.allclose(y, y2), "cubic spline interpolation on cubic function is giving non-exact results"


if __name__ == '__main__':
    test_frequency_response()
    test_spline()
