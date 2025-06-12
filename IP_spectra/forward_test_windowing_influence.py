import matplotlib.pyplot as plt
import numpy as np
import torch
import anemone
from copy import deepcopy
from anemone.experiment import *
from anemone.forwards import *

tx = Loop.square(3)
rx = Receiver(x=13)
waveform = Waveform.example()
recfilter = FilterChain([
    ButterworthFilter(2.4e5, 1),
    ButterworthFilter(6.2e5, 1),
    ButterworthFilter(4.5e5, 2)
])

rho = torch.tensor([1000, 1000, 1000], dtype=torch.double)
thickness = torch.tensor([20, 30], dtype=torch.double)
tau = torch.ones(3, dtype=torch.double)*1e-4
c = torch.ones(3, dtype=torch.double)*0.6
m = torch.tensor([0.01, 0.01, 0.01], dtype=torch.double)


times = torch.logspace(-5, -3, 32)
fwrTD = IPForward(tx, rx, times, filterfunc=recfilter)
dbdt = fwrTD(rho, tau,  m, c, thickness).real

frqs = torch.logspace(3, 6, 51)
fwrFD = IPForwardFD(tx, rx, frqs, filterfunc=recfilter)
Bfrq = fwrFD(rho, tau,  m, c, thickness)

sampling_rate = 1e-9

fig,axs = plt.subplots(1,2)

scale = np.array([1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5])
colors = plt.cm.viridis(np.linspace(0, 1, len(scale)))

for i, fac in enumerate(scale): 
    
    print(fac*sampling_rate)
    times2 = torch.arange(sampling_rate*fac, times.max(), sampling_rate)
    fwrTD2 = IPForward(tx, rx, times2, filterfunc=recfilter)
    n = len(times2)
    dt = times2[1] - times2[0]
    frqs2= torch.fft.fftfreq(n, float(dt))

    dbdt2 = fwrTD2(rho, tau,  m, c, thickness).real
    Bfrq2 = torch.fft.fft(dbdt2)*dt

    axs[1].loglog(frqs2[0:len(frqs2)//2], torch.abs(Bfrq2[:len(frqs2)//2]), color=colors[i]) 
    if i==0:
        axs[0].loglog(times2, dbdt2, 'k')
        axs[0].loglog(times2, -dbdt2, 'k--')
    
    axs[0].loglog([fac*sampling_rate, fac*sampling_rate], [1e-9, 1e-3], color=colors[i])


axs[0].set_aspect('equal')
axs[1].loglog(frqs, torch.abs(Bfrq), 'r')

axs[1].legend()
axs[1].set_title(str(int(rho[0].item()))+ ' Ohmm halfspace')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('dB/dt (V/m/s)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('B (T)')

plt.tight_layout()

fig.savefig(f"Window_test_{int(rho[0].item())}_Ohmm.png")

plt.show()