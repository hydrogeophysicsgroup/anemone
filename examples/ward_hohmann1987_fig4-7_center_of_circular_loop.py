from anemone import transforms
from anemone import physics
from anemone import constants as c
import torch
import matplotlib.pyplot as plt

# using values from the example in Fig 4.7
radius = 50.                   # meters
rho = torch.tensor([100])           # 1/ohm-m
frqs = torch.logspace(-1, 6, 501, dtype=torch.complex128)  # Hz
omega = 2*torch.pi*frqs           # radians

def plot_loglog(x, y, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.loglog(x,  y.real, 'C0', label='real')
    ax.loglog(x, -y.real, 'C0--')
    ax.loglog(x,  y.imag, 'C1', label='imag')
    ax.loglog(x, -y.imag, 'C1--')
    return ax


def rte_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = c.epsilon0*torch.ones_like(sigma)
    if mu is None:
        mu = c.mu0*torch.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = torch.sqrt(lambda_sqrd[None,:]-K1_sqrd[:, None])
    rte         = (wave_length[None,:]-u1)/(wave_length[None,:]+u1)
    return rte



fig,axs = plt.subplots(1,3, sharex=True, sharey=True)

hz_analytical = physics.hz_halfspace_center_circle_loop(rho, omega, 50)
plot_loglog(frqs.real, hz_analytical, ax=axs[0])
axs[0].set_title('analytic solution')

omega = omega[:, None]


def circle_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = c.epsilon0*torch.ones_like(sigma)
    if mu is None:
        mu = c.mu0*torch.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = torch.sqrt(lambda_sqrd[None,:]-K1_sqrd[:, None])
    rte         = lambda_sqrd[None,:]/(wave_length[None,:]+u1)
    return torch.squeeze(rte)

def hankel_integrand(wave_length):
    lmb = wave_length[None,:]
    return torch.squeeze(circle_halfspace(omega, 1./rho, lmb))

hkl_rte1 = transforms.Hankel(hankel_integrand, 1, smp_per_decade=20)
hz_circle = rho*hkl_rte1(rho)
plot_loglog(frqs.real, hz_circle, ax=axs[1])
axs[1].set_title('numerical integral - simplified integrand')

def hankel_integrand(wave_length):
    rte = rte_halfspace(omega, 1/rho, wave_length)
    u0  = physics.freespace_u0(omega, wave_length)
    return torch.nan_to_num(((1. + rte)*wave_length**2)/u0)

hkl_rte1 = transforms.Hankel(hankel_integrand, 1, smp_per_decade=20)
hz_circle = 2*torch.pi*rho*hkl_rte1(rho)/(4*torch.pi)

plot_loglog(frqs.real, hz_circle[-1], ax=axs[2])
axs[2].set_title('numerical integral - full integrand')

for ax in axs:
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('$H_z$ [A/m]')

fig.suptitle('W&H - Fig4.7: sign of imaginary component does not match')

if __name__ == "__main__":
    plt.show()
