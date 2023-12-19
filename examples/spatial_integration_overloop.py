import numpy as np
import matplotlib.pyplot as plt
from em1d.experiment import Loop, Receiver
from em1d.transforms import Hankel
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0
# first lets compute Hz(R)
lp = Loop.square(50)
lp.x += 0.005156154881
rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)

def integrand(wave_length):
    sigma = np.array([10, 1., 10])[:,None,None]
    thick = np.array([25, 50])[:,None,None]
    omega = np.array([100.])[None,:,None]*2*np.pi
    lmb = wave_length[None, None, :]
    intg = (1.+rte(sigma, thick, omega, lmb))*lmb**2 / u0(omega, lmb)
    print(intg.shape)
    return intg


hkl = Hankel(integrand, 1)
r, hr = hkl.range(rmin, rmax)
from scipy.interpolate import interp1d

int_hr = interp1d(np.log10(r), hr)

def hr_interp(r):
    return int_hr(np.log10(r))
from em1d.transforms import plot_loglog
from em1d.experiment import getRz
fig,axs = plt.subplots()
axs = plot_loglog(r, hr, ax=axs)
axs.set_xscale('linear')
from numpy.linalg import norm


def integrate_around_loop(lp, rx, hr):
    lp_xyz = lp.xyz - rx.xyz[None, ...]

    HR = 0+0j
    n_dipoles = 0
    for i in range(len(lp)-1):
        Rz,iRz = getRz(lp_xyz[i], lp_xyz[i+1])
        p1 = np.einsum('...i,...i->...', Rz, lp_xyz[i])
        p2 = np.einsum('...i,...i->...', Rz, lp_xyz[i+1])

        p12 = p2-p1
        tdist = norm(p12, axis=-1)
        p12 /= tdist
        cp = p1
        cdist = 0

        while cdist < tdist:
            cR = norm(cp)
            ds = cR/15.
            if cdist+ds > tdist:
                ds = tdist-cdist
                cdist += cR/15
            else:
                cdist += ds
            HR += ds*p1[1]*hr(cR)/cR
            cp += ds*p12
            n_dipoles +=1
    return HR

hz_loop = integrate_around_loop(lp, Receiver(), hr_interp)
print(r)
lp = Loop.square(50)
s = lp.s_of_r(Receiver(), r)
print(s)
hz_sr = (hr/r).dot(s+0j)
fig,axs=plt.subplots()
# plot_loglog(r, hr*(s+0j) / r, ax=axs)
plot_loglog(r, s, ax=axs)
print(hz_loop)
print(hz_sr)
lp = Loop.square(50)

xy = np.linspace(-100, 100, 51)
HZ2 = np.zeros((len(xy), len(xy)), dtype=complex)

rxall = Receiver(x=xy, y=xy)
rmin = lp.min_horizontal_distance(rxall)
rmax = lp.max_horizontal_distance(rxall)
print(rmin, rmax)
hkl = Hankel(integrand, 1, smp_per_decade=10)
r, hr = hkl.range(rmin, rmax)
int_hr = interp1d(r, hr)

def hr_interp(r):
    return int_hr(r)


for i in range(len(xy)):
    for j in range(len(xy)):
        rx = Receiver(x=xy[i], y=xy[j])
        hz_loop = integrate_around_loop(lp, rx, hr_interp)
        HZ2[i,j] = hz_loop
    print(f'{i}of{len(xy)}', flush=True, end='\r')
HZ = np.zeros((len(xy), len(xy)), dtype=complex)

rxall = Receiver(x=xy, y=xy)
rmin = lp.min_horizontal_distance(rxall)
rmax = lp.max_horizontal_distance(rxall)
r, hr = hkl.range(rmin, rmax)

dr = np.gradient(r)

for i in range(len(xy)):
    for j in range(len(xy)):
        rx = Receiver(x=xy[i], y=xy[j])
        s = lp.s_of_r(rx, r)
        s = np.convolve(s, np.array([2, 8, 1])/11, mode='same')+0j
        #hrs = np.convolve(hr*s, np.array([2, 8, 1])/11, mode='same'); HZ[i,j] = np.sum(hrs)
        HZ[i,j] = hr.dot(s)
fig,ax = plt.subplots()
# plt.plot(r, s, '.-')
# plt.plot(r, hr.real, '.-')
# plt.plot(r, hr.imag, '.-')
plt.plot(r, (s).real)
plt.plot(r, (s).imag)
plt.xscale('log')
fig,axs = plt.subplots(1,2, sharex=True, sharey=True)
pcm = axs[0].pcolor(xy, xy, HZ.real, vmin=-3, vmax=3, cmap='seismic')
pcm = axs[1].pcolor(xy, xy, HZ2.real, vmin=-3, vmax=3, cmap='seismic')
plt.colorbar(pcm)
fig,axs = plt.subplots(1,3, sharex=True, sharey=True, figsize=(9,4))
pcm = axs[0].pcolor(xy, xy, np.log10(np.abs(HZ)), vmin=-6, vmax=2)
pcm = axs[1].pcolor(xy, xy, np.log10(np.abs(HZ2)), vmin=-6, vmax=2)
pcm=axs[2].pcolor(xy, xy, np.abs((HZ2-HZ)/HZ2), vmin=0, vmax=1)
plt.colorbar(pcm)

fig,axs = plt.subplots(1,2)
axs[0].plot(np.abs(HZ2.flatten().real), np.abs(HZ.flatten().real), '.')
axs[0].plot(np.abs(HZ2.flatten().imag), np.abs(HZ.flatten().imag), '.')
axs[0].plot([0,20], [0, 20], 'k--')
axs[0].set_yscale('log')
axs[0].set_xscale('log')

axs[1].hist(np.abs((HZ2-HZ)/HZ2).flatten())


plt.show()