import matplotlib.pyplot as plt
import numpy as np
import torch
import anemone
from copy import deepcopy
from anemone.experiment import *
from anemone.forwards import *
from anemone.inversion import SmoothModelColeCole, FewLayerColeCole


#set up the loop geometries and filters
tx = Loop.square(3)
rx = Receiver(x=13)
waveform = Waveform.example()
recfilter = FilterChain([
    ButterworthFilter(2.4e5, 1),
    ButterworthFilter(6.2e5, 1),
    ButterworthFilter(4.5e5, 2)
])

#setup the times, freqs, and forward models
times = torch.logspace(-5, -3, 32)
fwrTD = IPForward(tx, rx, times, filterfunc=recfilter)

frqs = torch.logspace(3, 6, 51)
fwrFD = IPForwardFD(tx, rx, frqs, filterfunc=recfilter)

sampling_rate=1e-9
times2 = torch.arange(times.min(), times.max(), sampling_rate)
fwrTD2 = IPForward(tx, rx, times2, filterfunc=recfilter)
n = len(times2)
dt2 = times2[1] - times2[0]
frqs2= torch.fft.fftfreq(n, float(dt2))

# sampling_rate=1e-9
# times3 = torch.arange(times.min(), 2e-4, sampling_rate)
# fwrTD3 = IPForward(tx, rx, times2, filterfunc=recfilter)
# n = len(times3)
# dt3 = times3[1] - times3[0]
# frqs3= torch.fft.fftfreq(n, float(dt3))



fig,axs = plt.subplots(2,4, figsize=(12,6))
axmod = axs[:,:-2].flatten()


name = 'm0p5_var_res'

res_ref = 1000
tau_ref = 1e-4
m_ref = 0.5
C_ref = 0.5

res_sweep = np.logspace(1, 3, 10)
m_sweep = np.logspace(np.log10(0.05), np.log10(0.5), 10)
tau_sweep = np.logspace(-6, -2, 10)
C_sweep = np.linspace(0.1, 1,10)


sweep_params = C_sweep

col = plt.cm.viridis(np.linspace(0, 1, len(m_sweep)))


for i,res_ref in enumerate(res_sweep):
# for i,res_ref in enumerate(res_sweep):
# for i,m_ref in enumerate(m_sweep):
# for i,tau_ref in enumerate(tau_sweep):
# for i,C_ref in enumerate(C_sweep):
   # for j,tau_ref in enumerate(tau_sweep):
    #     for k,C_ref in enumerate(C_sweep):
        
    #make the model 
    cmod = FewLayerColeCole([res_ref, res_ref], [tau_ref,tau_ref], [m_ref,m_ref], [C_ref, C_ref], [50])

    #forward models - freq and time domain
    Bf = fwrFD(*cmod.mod)

    dbdt = fwrTD(*cmod.mod).real
    dbdt2 = fwrTD2(*cmod.mod).real
    Bf2 = torch.fft.fft(dbdt2)*dt2
    # dbdt3 = fwrTD2(*cmod.mod).real
    # Bf3 = torch.fft.fft(dbdt3)*dt3

    # axs[0,-2].loglog(frqs, torch.abs(Bf), color=col[i])
    axs[0,-2].loglog(frqs2[0:len(frqs2)//2], torch.abs(Bf2[:len(frqs2)//2]), color=col[i])
    # axs[0,-2].loglog(frqs3[0:len(frqs3)//2], torch.abs(Bf3[:len(frqs3)//2]), '--', color=col[i])
    axs[1,-2].loglog(times, dbdt, color=col[i])
    axs[1,-2].loglog(times, -dbdt, color=col[i],linestyle='--')

    dbdt = fwrTD(*cmod.mod).real
    axs[1,-2].loglog(times, dbdt,color=col[i])
    axs[1,-2].loglog(times,-dbdt,color=col[i],linestyle='--')

    if i == 0:  #grab the first data for references for comparisons with other models
        dbdt_ref = deepcopy(dbdt)
        Bf_ref = deepcopy(Bf)
        Bf2_ref = deepcopy(Bf2)
        # Bf3_ref = deepcopy(Bf3)


    #difference plots
    axs[0,-1].plot(frqs2[0:len(frqs2)//2], (torch.abs(Bf2[0:len(frqs2)//2])-torch.abs(Bf2_ref[0:len(frqs2)//2]))/torch.abs(Bf2_ref[0:len(frqs2)//2])*100,color=col[i])
    # axs[0,-1].plot(frqs3[0:len(frqs3)//2], np.abs((torch.abs(Bf3[0:len(frqs3)//2])-torch.abs(Bf3_ref[0:len(frqs3)//2]))/torch.abs(Bf3_ref[0:len(frqs3)//2])*100),'--',color=col[i])
    axs[1,-1].plot(times, (dbdt-dbdt_ref)/dbdt_ref*100, color=col[i])

    axs[0,-1].set_ylim([-100,100])
    axs[1,-1].set_ylim([-100,100])

    cmod.plot(axs=axmod, color=col[i])



for ax in axmod:
    ax.set_ylim(70,0)
axs[0,-1].set_title('FD % diff')
axs[0,-1].set_title('FD % diff')
axs[1,-1].set_title('TD % diff')
axs[0,-2].set_title('FD data')
axs[1,-2].set_title('TD data')
axs[0,0].set_xscale('log')
axs[0,0].set_xlim([5e-1,1100])
axs[0,1].set_xscale('log')
axs[0,1].set_xlim([1e-6,1e-2])
axs[1,0].set_xscale('log')
axs[1,0].set_xlim([1e-3,1])
axs[1,1].set_xlim([0,1])

axs[0,-1].set_xscale('log')
axs[0,-1].set_xlim([1e3,1e6])
axs[0,-2].set_xlim([1e3,1e6])
axs[0,-2].set_ylim([1e-17,1e-9])
axs[1,-1].set_xscale('log')
axs[1,-2].set_ylim([1e-12,1e-4])


axs[0,0].set_xlabel('res (ohm-m)')
axs[1,0].set_xlabel('m (V/V)')
axs[0,1].set_xlabel('tau (s)')
axs[1,1].set_xlabel('C (1)')
axs[1,2].set_xlabel('t (s)')
axs[1,3].set_xlabel('t (s)')
axs[0,2].set_xlabel('f (Hz)')
axs[0,3].set_xlabel('f (Hz)')
axs[0,3].grid(True)
axs[1,3].grid(True)



plt.tight_layout()

plt.savefig(name + '.png')

plt.show()