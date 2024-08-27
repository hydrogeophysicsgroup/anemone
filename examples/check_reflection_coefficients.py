from anemone import physics
import matplotlib.pyplot as plt
import numpy as np
import torch

sigma = torch.tensor([100, 0.5, 0.1], dtype=torch.complex128)
omega = torch.tensor([1000], dtype=torch.complex128)*2*torch.pi


wave_length   = torch.logspace(-3, 3, 1001, dtype=torch.complex128)
lambda_sqrd   = wave_length*wave_length

K0_sqrd       = physics.epsilon0*physics.mu0*omega**2
K1_sqrd       = physics.mu0*omega*(physics.epsilon0*omega-1j*sigma[0])

u0            = torch.sqrt(lambda_sqrd-K0_sqrd)
u1            = torch.sqrt(lambda_sqrd-K1_sqrd)

z0,y0 = physics.calc_zy(0., omega)
z1,y1 = physics.calc_zy(sigma[0], omega)


rte_halfspace = (u0-u1)/(u0+u1)
# rtm_halfspace = (u1*K0_sqrd - u0*K1_sqrd)/(u1*K0_sqrd + u0*K1_sqrd)
rtm_halfspace = ((u0 - y0*u1/y1) / (u0 + y0*u1/y1)).numpy()


fig,axs = plt.subplots(2,1, sharex=True)

labels=[]
for iz,thickness in enumerate(np.geomspace(1e-4, 1e3, 5)):
    thick = torch.tensor([thickness, 10], dtype=torch.complex128)
    c_rte = torch.squeeze(physics.refl_transverse_electric(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:])).numpy()
    axs[0].loglog(wave_length, np.abs(c_rte.real), color=f'C{iz}')
    axs[1].loglog(wave_length, np.abs(c_rte.imag), color=f'C{iz}')
    labels.append('{:1.1e}'.format(thickness))
labels.append('halfspace')

axs[0].loglog(wave_length, np.abs(rte_halfspace.real), 'k--')
axs[1].loglog(wave_length, np.abs(rte_halfspace.imag), 'k--')
axs[0].legend(labels)

axs[0].set_ylabel('$r_{TE}$ real')
axs[1].set_ylabel('$r_{TE}$ imag')
axs[-1].set_xlabel('$\\lambda$ [m]')
axs[0].set_title('reflection coefficients transverse electric')


fig,axs = plt.subplots(2,1, sharex=True)

labels=[]
for iz,thickness in enumerate(np.geomspace(1e-4, 1e2, 5)):
    thick = torch.tensor([thickness, 10], dtype=torch.complex128)
    c_rtm = torch.squeeze(physics.refl_transverse_magnetic(sigma[:,None,None], thick[:, None,None], omega[None,:,None], wave_length[None,None,:])).numpy()
    axs[0].loglog(wave_length, np.abs(c_rtm.real), color=f'C{iz}')
    axs[1].loglog(wave_length, np.abs(c_rtm.imag), color=f'C{iz}')
    labels.append('{:1.1e}'.format(thickness))
labels.append('halfspace')

axs[0].loglog(wave_length, np.abs(rtm_halfspace.real), 'k--')
axs[1].loglog(wave_length, np.abs(rtm_halfspace.imag), 'k--')
axs[0].legend(labels)

axs[0].set_ylabel('$r_{TM}$ real')
axs[1].set_ylabel('$r_{TM}$ imag')
axs[-1].set_xlabel('$\\lambda$ [m]')
axs[0].set_title('reflection coefficients transverse magnetic')



plt.show()