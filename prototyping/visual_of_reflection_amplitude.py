import sys, os
import matplotlib.pyplot as plt
import numpy as np
outname = sys.argv[0][:-3]

from plog import Log

sigma = np.array([0.5, 0.2, 1])
lg = Log.standard(np.log10(sigma), [10, 20], 'conductivity')
lg.depth_bot[-1] = 30

z = np.linspace(0, 10, 101)
z_sky = np.linspace(-5, 0, 101)

fig,axs = plt.subplots(1,3, sharey=True, sharex=True)
for ax in axs:
    ax,pcm = lg.plot(ax, cmap='Greys', vmin=-2, vmax=2)

i=-1
sgm=5e-2
w = 0.45
u_align_pos = -0.4
tx_amp = 0.35
y_dn = tx_amp*np.exp(-sgm*np.abs(z_sky+5))#*A
y_up = np.exp(sgm*z_sky)*y_dn.min()
axs[0].plot(y_dn, z_sky, f'C5')
axs[0].plot(y_up, z_sky, f'C5--')
axs[0].annotate('$u_0$', (u_align_pos, -2), ha='left', va='bottom', color=f'C5')

uparrow, dnarrow = '\\uparrow', '\\downarrow'
axs[0].annotate('$A_{{air}}^{{{:}}}$'.format(uparrow), (y_up.min(), -1), ha='right', va='bottom', color=f'C5')
axs[0].annotate('$A_{{air}}^{{{:}}}$'.format(dnarrow), (0.95, -3), ha='center', va='center', color=f'C5')

for i,sgm in enumerate(sigma):
    y_dn = w*np.exp(-sgm*z)#*A
    y_up = w*np.exp(sgm*(z-10))#*B
    axs[0].plot(y_dn, z+i*10, f'C{i}')
    if i!=2: axs[0].plot(y_up, z+i*10, f'C{i}--')
    for ax in axs[::2]: ax.annotate(f'$u_{i+1}, h_{i+1}$', (u_align_pos, 5+i*10), ha='left', va='center', color=f'C{i}')
    xpos = y_dn.min()# if i==1 else 0.05
    uparrow, dnarrow = '\\uparrow', '\\downarrow'
    
    axs[0].annotate('$A_{:}^{{{:}}}$'.format(i+1, dnarrow), (w-0.025, i*10+2), ha='center', va='center', color=f'C{i}')
    if i!=2: axs[0].annotate('$A_{:}^{{{:}}}$'.format(i+1, uparrow), (w-0.025, (i+1)*10-2), ha='center', va='center', color=f'C{i}')

ax = axs[1]
bgcolor = "#EBEBEB"
ax.annotate('$X_0$', (0, 0), ha='center', va='center', fontsize=7, backgroundcolor=bgcolor)
ax.annotate('$X_1$', (0,10), ha='center', va='center', fontsize=7, backgroundcolor=bgcolor)
ax.annotate('$X_2$', (0,20), ha='center', va='center', fontsize=7, backgroundcolor=bgcolor)

def alab(i, arrow='up'):
    if arrow=='up': arrow=uparrow
    else: arrow=dnarrow
    return 'A_{:}^{{{:}}}'.format(i, arrow)

l_align = -0.4
r_align = 0.45
i = 0
eq = "$\\binom{"+alab(i,'dn')+"}{"+alab(i,'up')+"}$"
eq = eq.replace(alab(i, 'up'), 'A_{air}^{{\\uparrow}}').replace(alab(i, 'dn'), '1')
ax.annotate(eq, (l_align, 5+(i-1)*10), ha='left', va='center', color=f'C5', fontsize=12)
eq = "$\\binom{"+alab(i+1,'dn')+"}{"+alab(i+1, 'up')+"}$"
ax.annotate(eq, (r_align, 5+(i-1)*10), ha='right', va='center', color=f'C0', fontsize=12)

i = 1
eq = "$\\binom{"+alab(i,'dn')+"}{"+alab(i,'up')+"}$"
ax.annotate(eq, (l_align, 5+(i-1)*10), ha='left', va='center', color=f'C{i-1}', fontsize=12)
eq = "$\\binom{"+alab(i+1,'dn')+"}{"+alab(i+1, 'up')+"}$"
ax.annotate(eq, (r_align, 5+(i-1)*10), ha='right', va='center', color=f'C{i}', fontsize=12)

i = 2
eq = "$\\binom{"+alab(i,'dn')+"}{"+alab(i,'up')+"}$"
ax.annotate(eq, (l_align, 5+(i-1)*10), ha='left', va='center', color=f'C{i-1}', fontsize=12)
eq = "$\\binom{" + alab(i+1, 'dn') + "}{0}$"
ax.annotate(eq, (r_align, 5+(i-1)*10), ha='right', va='center', color=f'C{i}', fontsize=12)
ax.annotate(eq, (r_align, 5+i*10), ha='right', va='center', color=f'C{i}', fontsize=12)

for i in range(3):
    eq = f"$ = X_{{{i}}} \cdot$"
    ax.annotate(eq, (0, (i-1)*10+5), ha='center', va='center', fontsize=12, color='k')

ax = axs[2]

from scipy.linalg import solve
rho = [20, 8, 40]
sigma = [1./x for x in rho]
sgm_air = 1e-2
tx_height = 5.
A = np.array([
    [1., 0., 0., 0., 0., 0., 0., 0.],
    [np.exp(-sgm_air*tx_height), np.exp(sgm_air*tx_height), -1., -1., 0., 0., 0., 0.],
    [-sgm_air*np.exp(-sgm_air*tx_height), sgm_air*np.exp(sgm_air*tx_height), sigma[0], -sigma[0], 0., 0., 0., 0.],
    [0., 0., np.exp(-sigma[0]*10), np.exp(sigma[0]*10), -1., -1., 0., 0.],
    [0., 0., -sigma[0]*np.exp(-sigma[0]*10), sigma[0]*np.exp(sigma[0]*10), sigma[1], -sigma[1], 0., 0.],
    [0., 0., 0., 0., np.exp(-sigma[1]*10), np.exp(sigma[1]*10), -1., -1.],
    [0., 0., 0., 0., -sigma[1]*np.exp(-sigma[1]*10), sigma[1]*np.exp(sigma[1]*10), sigma[2], -sigma[2]],
    [0., 0., 0., 0., 0., 0., 0., 1.]
])
b = np.array([1, 0, 0, 0, 0, 0, 0., 0.])
x = solve(A, b)
from plog import Log
thickness = np.array([10, 20])
# lg = Log.standard(np.log10(np.array([0.5, 0.1, 1])), thickness, 'conductivity')
lg.depth_bot[-1] = 30
A_up = x[1::2]
A_dn = x[::2]

z = np.linspace(-10, 20, 100)
z_air = np.linspace(0, tx_height, 100)

ax,pcm = lg.plot(ax=ax, cmap='Greys', vmin=-2, vmax=2)


for i in [-1, 0, 1, 2]:
    if i == -1:
        sgm = sgm_air
        y1 = np.exp(-sgm*z_air)*A_dn[0]
        y2 = np.exp(sgm*z_air)*A_up[0]
        ax.plot(y1 + y2, z_air+i*tx_height, f'C5')
        tx_amp = A_dn[0]+A_up[0]
        # ax.plot(y2, z_air+i*10, f'C5--')
    else:
        sgm = sigma[i]
        y = A_up[i+1] * np.exp(sgm * z) + A_dn[i+1] * np.exp(-sgm * z)
        ax.plot(y, z+i*10, f'C{i}--', linewidth=0.5)
        in_layer = (z>0) & (z<10)
        ax.plot(y[in_layer], z[in_layer]+i*10, f'C{i}')

theta = np.linspace(0, 2.*np.pi, 100)
y = 0.5*np.cos(theta)-tx_height
x = np.sin(theta)*tx_amp
for ax in axs:
    ax.set_aspect(0.05)
    ax.set_ylim(30, -10)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([]);
    ax.set_xlabel('field strength')
    ax.plot([-0.5,0.5], [0,0], 'k-', linewidth=0.75)

for ax in axs[::2]:
    ax.plot(x, y, 'k-')
    ax.plot([0,0], [-10, 30], 'k-', linewidth=0.75)

axs[0].set_ylabel('(+)$\\leftarrow$ z $\\rightarrow$ (-) ')

axs[0].set_yticks([-5, 0, 10, 20])
axs[0].set_yticklabels(['$h_{tx}$', 'surface', '$z_1$', '$z_2$'])
plt.tight_layout()
plt.show()


if 'save' in sys.argv:
    fig.savefig(outname, dpi=600)