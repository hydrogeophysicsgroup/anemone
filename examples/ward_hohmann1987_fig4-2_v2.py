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


hz_analytical = physics.halfspace_vmd(omega, sigma, rho)

wave_length = np.logspace(-5, 5, 501)

rte = physics.refl_transverse_electric(np.ones(31)*sigma+np.random.randn(31), np.geomspace(0.5, 300., 30), wave_length, omega[250:251])[0]

def calc_rte_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = epsilon0*np.ones_like(sigma)
    if mu is None:
        mu = mu0*np.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = np.sqrt(lambda_sqrd-K1_sqrd)
    rte         = (wave_length-u1)/(wave_length+u1)
    return rte

rte2 = calc_rte_halfspace(omega[250], sigma, wave_length)


hz_analytical = physics.halfspace_vmd(omega, sigma, rho)

cond_z = np.abs(np.ones(31)*sigma+np.random.randn(31))
z = np.geomspace(0.5, 300, 31)
thick = np.diff(z)

plt.figure()
plt.plot(cond_z, z)
plt.show()

def hank_integrand(wave_length):
    rte = physics.refl_transverse_electric(cond_z, thick, wave_length, omega)
    u0  = physics.freespace_u0(omega, wave_length)
    return np.nan_to_num(((1+rte)*wave_length**3)/u0)

from time import perf_counter_ns as timer

bt = timer()
Hkl_rte1 = transforms.Hankel(hank_integrand, 0)
hz = Hkl_rte1(100)/(2*np.pi)
print(f'forward time: {(timer()-bt)/1e6} ms')



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

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# ax = transforms.plot_loglog(wave_length, rte, ax=axs[0])
# ax.set_ylabel('$H_z$ [A/m]')
# ax.set_xlabel('frequency [Hz]')
# # ax.set_ylim(1e-12, 1e-6)
# # ax.set_xlim(1e-1, 1e5)
# ax.set_title('analytic')


# ax = transforms.plot_loglog(wave_length, rte2, ax=axs[1])
# ax.set_ylabel('$H_z$ [A/m]')
# ax.set_xlabel('frequency [Hz]')
# # ax.set_ylim(1e-12, 1e-6)
# # ax.set_xlim(1e-1, 1e5)
# ax.set_title('numerical')
# plt.tight_layout()
# plt.show()