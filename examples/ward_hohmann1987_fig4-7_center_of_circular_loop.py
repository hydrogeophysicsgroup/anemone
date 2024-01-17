from em1d import transforms
from em1d import physics
import numpy as np
import matplotlib.pyplot as plt

# using values from the example in Fig 4.7
rho = 50.                   # meters
sigma = 1./100.            # 1/ohm-m
frqs = np.logspace(-1, 6, 501)  # Hz
omega = 2*np.pi*frqs           # radians

def rte_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = physics.epsilon0*np.ones_like(sigma)
    if mu is None:
        mu = physics.mu0*np.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = np.sqrt(lambda_sqrd[None,:]-K1_sqrd[:, None])
    rte         = (wave_length[None,:]-u1)/(wave_length[None,:]+u1)
    return rte



fig,axs = plt.subplots(1,3, sharex=True, sharey=True)

hz_analytical = physics.halfspace_center_circle_loop(omega, sigma, 50)
transforms.plot_loglog(frqs, hz_analytical, ax=axs[0])
axs[0].set_title('analytic solution')

omega = omega[:, None]


def circle_halfspace(omega, sigma, wave_length, epsilon=None, mu=None):
    if epsilon is None:
        epsilon = physics.epsilon0*np.ones_like(sigma)
    if mu is None:
        mu = physics.mu0*np.ones_like(sigma)

    lambda_sqrd = wave_length*wave_length
    K1_sqrd     = mu*omega*(epsilon*omega-1j*sigma)
    u1          = np.sqrt(lambda_sqrd[None,:]-K1_sqrd[:, None])
    rte         = lambda_sqrd[None,:]/(wave_length[None,:]+u1)
    return np.squeeze(rte)

def hankel_integrand(wave_length):
    lmb = wave_length[None,:]
    return np.squeeze(circle_halfspace(omega, sigma, lmb))

hkl_rte1 = transforms.Hankel(hankel_integrand, 1, smp_per_decade=20)
hz_circle = rho*hkl_rte1(rho)
transforms.plot_loglog(frqs, hz_circle, ax=axs[1])
axs[1].set_title('numerical integral - simplified integrand')

def hankel_integrand(wave_length):
    rte = rte_halfspace(omega, sigma, wave_length)
    u0  = physics.freespace_u0(omega, wave_length)
    # u0[u0==0.] = np.inf
    return np.nan_to_num(((1. + rte)*wave_length**2)/u0)

hkl_rte1 = transforms.Hankel(hankel_integrand, 1, smp_per_decade=20)
hz_circle = 2*np.pi*rho*hkl_rte1(rho)/(4*np.pi)
transforms.plot_loglog(frqs, hz_circle[-1], ax=axs[2])
axs[2].set_title('numerical integral - full integrand')

for ax in axs:
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('$H_z$ [A/m]')

fig.suptitle('W&H - Fig4.7: sign of imaginary component does not match')

plt.show()
