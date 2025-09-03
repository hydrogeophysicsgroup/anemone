from anemone.transforms import optimized_coefficients_christensen1990
import matplotlib.pyplot as plt
import numpy as np
pi = np.pi


lmda = np.linspace(-6, 6, 1001)
Hnustar = optimized_coefficients_christensen1990(lmda, omega_0=pi/4, smp_per_decade=10)#, -1, 1)

fig,axs = plt.subplots(3,1, sharex=True, sharey=True, figsize=(4,7))

for ax,omega,name in zip(axs, [pi, pi/2, pi/4], ['$\pi$', '$\pi/2$', '$\pi/4$']):
    Hnustar = optimized_coefficients_christensen1990(lmda, omega_0=omega)
    ax.plot(lmda, Hnustar)
    ax.set_ylabel('$H_o^*(v)$')
    ax.text(4, 0.3, f'$\omega_o$={name}')
ax.set_xlabel('$v$')
ax.set_ylim(-1,1)
axs[0].set_title('Optimized $J_o$-filters $s_c$=1.25')

if __name__ == "__main__":
    plt.show()