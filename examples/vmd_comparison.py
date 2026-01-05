from anemone import transforms
from anemone import physics

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_loglog(x, y, ax=None, color=None, line_not_marker=True):
    if ax is None:
        fig, ax = plt.subplots()
    if line_not_marker: pos,neg = '-', '--'
    else: pos,neg = '.', 'x'
    if color is None:
        real = 'C0'
        imag = 'C1'
    else:
        real = color
        imag = color

    ax.loglog(x,  y.real, real+pos, label='real')
    ax.loglog(x, -y.real, real+neg)
    ax.loglog(x,  y.imag, imag+pos, label='imag')
    ax.loglog(x, -y.imag, imag+neg)
    return ax

def calc_rte_halfspace(sigma, omega, wave_length):
    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = physics.calc_wavenumber(sigma, omega)**2
    # u1          = torch.sqrt(lambda_sqrd-K1_sqrd)
    u1 = torch.sqrt(1.-K1_sqrd/lambda_sqrd)*wave_length
    rte         = 1./(wave_length+u1)
    return rte


rho = 100.                   # meters
radius = 50.            # 1/ohm-m
frqs = torch.logspace(-6, 5, 101, dtype=torch.complex128)  # Hz
omega = 2*torch.pi*frqs

hz_analytical = physics.halfspace_vmd(rho, omega, radius).numpy()
omega = omega[:, None]

def rte1(wave_length):
    wave_length = wave_length[None, :]
    return calc_rte_halfspace(1/rho, omega, wave_length)*wave_length**3


Hkl_rte1 = transforms.Hankel(rte1, 0, smp_per_decade=10)
hz = (Hkl_rte1(radius)/(2*torch.pi)).numpy()
frqs = frqs.real

plt.figure()
plt.plot(torch.log10(torch.abs(rte1(Hkl_rte1.lamda))).T)
plt.plot(torch.log10(torch.abs(Hkl_rte1.fc)), 'r--')

frqs = frqs.real
fig, axs = plt.subplots(2, 1, figsize=(4,6), sharex=True)

ax = plot_loglog(frqs.numpy(), hz_analytical, ax=axs[0], color='k')
ax = plot_loglog(frqs.numpy(), hz, ax=axs[0], line_not_marker=False)
ax.set_ylabel('$H_z$ [A/m]')
ax.set_ylim(1e-12, 1e-6)

difference = (hz - hz_analytical)
rel_diff = 100.*np.abs(difference.real)/np.abs(hz_analytical.real) + 1j*100.*np.abs(difference.imag)/np.abs(hz_analytical.imag)

ax = plot_loglog(frqs.numpy(), rel_diff, ax=axs[1])
axs[1].axhline(1., color='k', ls='--')
ax.set_ylabel('relative error [%]')
ax.set_xlabel('frequency [Hz]')

plt.tight_layout()

if __name__ == "__main__":
    plt.show()