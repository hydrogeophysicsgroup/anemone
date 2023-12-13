from em1d import transforms
from em1d import physics
import numpy as np
import matplotlib.pyplot as plt

mu0 = np.pi*4e-7
epsilon0 = 8.854187817e-12

# using values from the example in Fig 4.2
rho = 100.                   # meters
sigma = 1./100.            # 1/ohm-m
frqs = np.logspace(-6, 5, 501)  # Hz
omega = 2*np.pi*frqs           # radians

def calc_rte_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = epsilon0*np.ones_like(sigma)
    if mu is None:
        mu = mu0*np.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = np.sqrt(lambda_sqrd-K1_sqrd)
    rte         = 1./(wave_length+u1)
    return rte


m=1.
def calc_wavenumber(omega, sigma, mu=mu0, epsilon=epsilon0):
    return np.sqrt(mu*omega*(epsilon*omega - 1j*sigma))

def freq_response(omega):
    k = calc_wavenumber(omega, sigma)
    Hz = 9 - (9 + 9j*k*rho - 4*k**2 * rho**2 -
            1j*k**3*rho**3)*np.exp(-1j*k*rho)
    Hz *= m/(2*np.pi*k**2 * rho**5)
    return Hz

hz_analytical = freq_response(omega)
hz = np.zeros(len(frqs), dtype=complex)
for i_frq, freq in enumerate(frqs):
    freq *= 2*np.pi
    def rte1(wave_length):
        return calc_rte_halfspace(freq, sigma, wave_length)*wave_length**3
    Hkl_rte1 = transforms.Hankel(rte1, 0)
    hz[i_frq] = Hkl_rte1(100.)
hz /= (2*np.pi)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = transforms.plot_loglog(frqs, hz_analytical, ax=axs[0])
ax.set_ylabel('$H_z$ [A/m]')
ax.set_xlabel('frequency [Hz]')
ax.set_ylim(1e-12, 1e-6)
# ax.set_xlim(1e-1, 1e5)
ax.set_title('analytic')


ax = transforms.plot_loglog(frqs, hz, ax=axs[1])
ax.set_ylabel('$H_z$ [A/m]')
ax.set_xlabel('frequency [Hz]')
ax.set_ylim(1e-12, 1e-6)
# ax.set_xlim(1e-1, 1e5)
ax.set_title('numerical')
plt.tight_layout()
plt.show()