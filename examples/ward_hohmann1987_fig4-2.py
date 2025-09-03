from anemone import transforms
from anemone import physics

import torch
import matplotlib.pyplot as plt


def plot_loglog(x, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(x,  y.real, 'C0', label='real')
    ax.loglog(x, -y.real, 'C0--')
    ax.loglog(x,  y.imag, 'C1', label='imag')
    ax.loglog(x, -y.imag, 'C1--')
    return ax


def calc_rte_halfspace(sigma, omega, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(sigma, omega)**2
    u1          = torch.sqrt(lambda_sqrd-K1_sqrd)
    rte         = 1./(wave_length+u1)
    return rte


rho = 100.                   # meters
radius = 100.            # 1/ohm-m
frqs = torch.logspace(-6, 5, 501, dtype=torch.complex128)  # Hz
omega = 2*torch.pi*frqs

hz_analytical = physics.halfspace_vmd(rho, omega, radius).numpy()
omega = omega[:, None]

def rte1(wave_length):
    wave_length = wave_length[None, :]
    return calc_rte_halfspace(1/rho, omega, wave_length)*wave_length**3

Hkl_rte1 = transforms.Hankel(rte1, 0)
hz = (Hkl_rte1(rho)/(2*torch.pi)).numpy()
frqs = frqs.real
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = plot_loglog(frqs.numpy(), hz_analytical, ax=axs[0])
ax.set_ylabel('$H_z$ [A/m]')
ax.set_xlabel('frequency [Hz]')
ax.set_ylim(1e-12, 1e-6)
# ax.set_xlim(1e-1, 1e5)
ax.set_title('analytic')


ax = plot_loglog(frqs.numpy(), hz, ax=axs[1])
ax.set_ylabel('$H_z$ [A/m]')
ax.set_xlabel('frequency [Hz]')
ax.set_ylim(1e-12, 1e-6)
# ax.set_xlim(1e-1, 1e5)
ax.set_title('numerical')
plt.tight_layout()

if __name__ == "__main__":
    plt.show()