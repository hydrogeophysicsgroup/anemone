from anemone import transforms
from anemone import physics
import torch
import matplotlib.pyplot as plt

mu0 = torch.pi*4e-7
epsilon0 = 8.854187817e-12

# using values from the example in Fig 4.2
rho = 100.                   # meters
sigma = 1./100.            # 1/ohm-m
frqs = torch.logspace(-6, 5, 501)  # Hz
omega = 2*torch.pi*frqs           # radians


hz_analytical = physics.halfspace_vmd(omega, sigma, rho)

wave_length = torch.logspace(-5, 5, 501)[None, :, None]

rho = torch.ones(31)*rho+torch.randn(31)
thk = torch.logspace(0, 1, 30)



rte = physics.refl_transverse_electric(rho[:, None, None], thk[:, None, None], wave_length, omega[250:251])[0]

def calc_rte_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = epsilon0*torch.ones_like(sigma)
    if mu is None:
        mu = mu0*torch.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = torch.sqrt(lambda_sqrd-K1_sqrd)
    rte         = (wave_length-u1)/(wave_length+u1)
    return rte

rte2 = calc_rte_halfspace(omega[250], 1./rho, wave_length)


hz_analytical = physics.halfspace_vmd(rho[:, None], omega, 100.)

# cond_z = torch.abs(torch.ones(31)*sigma+torch.randn(31))
# thick = thk

# plt.figure()
# plt.plot(cond_z, torch.cumsum(thick, dim=0))

if __name__ == "__main__":
    plt.show()

def hank_integrand(wave_length):
    rte = physics.refl_transverse_electric(cond_z, thick, wave_length, omega)
    u0  = physics.freespace_u0(omega, wave_length)
    return torch.nan_to_num(((1+rte)*wave_length**3)/u0)

from time import perf_counter_ns as timer

bt = timer()
Hkl_rte1 = transforms.Hankel(hank_integrand, 0)
hz = Hkl_rte1(torch.tensor100)/(2*torch.pi)
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

if __name__ == "__main__":
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