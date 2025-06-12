
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

off_sweep = np.linspace(10, 100, 21)
col = plt.cm.viridis(np.linspace(0, 1, len(off_sweep)))

fig,axs = plt.subplots(2,4, figsize=(12,6))
axmod = axs[:,:-2].flatten()

fig2,axs2 = plt.subplots(1,3, figsize=(8,4))

for j,off_tmp in enumerate(off_sweep):

    rx = Receiver(x=off_tmp)
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



    name = '1000Ohmm_var_m'

    res_fix = 50
    tau_fix = 1e-4
    m_fix = 1e-6
    C_fix = 0.5

    thk1 = 30
    thk2 = 10

            
    #make the model 
    # cmod = FewLayerColeCole([res_fix, res_tmp, res_tmp], [tau_fix,tau_fix,tau_fix], [m_fix,m_fix,m_fix], [C_fix,C_fix, C_fix], [thk1,thk2])

    cmod = FewLayerColeCole([res_fix, res_fix, res_fix], [tau_fix,tau_fix,tau_fix], [m_fix,m_fix,m_fix], [C_fix,C_fix, C_fix], [thk1,thk2])


    #forward models - freq and time domain
    Bf = fwrFD(*cmod.mod)

    dbdt = fwrTD(*cmod.mod).real
    dbdt2 = fwrTD2(*cmod.mod).real
    Bf2 = torch.fft.fft(dbdt2)*dt2
    # dbdt3 = fwrTD2(*cmod.mod).real
    # Bf3 = torch.fft.fft(dbdt3)*dt3

    # axs[0,-2].loglog(frqs, torch.abs(Bf), color=col[i])
    axs[0,-2].loglog(frqs2[0:len(frqs2)//2], torch.abs(Bf2[:len(frqs2)//2]), color=col[j])
    # axs[0,-2].loglog(frqs3[0:len(frqs3)//2], torch.abs(Bf3[:len(frqs3)//2]), '--', color=col[i])
    axs[1,-2].loglog(times, dbdt, color=col[j])
    axs[1,-2].loglog(times, -dbdt, color=col[j],linestyle='--')

    dbdt = fwrTD(*cmod.mod).real
    axs[1,-2].loglog(times, dbdt,color=col[j])
    axs[1,-2].loglog(times,-dbdt,color=col[j],linestyle='--')

    if j == 0:  #grab the first data for references for comparisons with other models
        dbdt_ref = deepcopy(dbdt)
        Bf_ref = deepcopy(Bf)
        Bf2_ref = deepcopy(Bf2)
        # Bf3_ref = deepcopy(Bf3)


    #difference plots
    axs[0,-1].plot(frqs2[0:len(frqs2)//2], (torch.abs(Bf2[0:len(frqs2)//2])-torch.abs(Bf2_ref[0:len(frqs2)//2]))/torch.abs(Bf2_ref[0:len(frqs2)//2])*100,color=col[j])
    # axs[0,-1].plot(frqs3[0:len(frqs3)//2], np.abs((torch.abs(Bf3[0:len(frqs3)//2])-torch.abs(Bf3_ref[0:len(frqs3)//2]))/torch.abs(Bf3_ref[0:len(frqs3)//2])*100),'--',color=col[i])
    axs[1,-1].plot(times, (dbdt-dbdt_ref)/dbdt_ref*100, color=col[j])

    axs[0,-1].set_ylim([-100,100])
    axs[1,-1].set_ylim([-100,100])

    cmod.plot(axs=axmod, color=col[j])

    axs2[0].loglog(frqs2[0:len(frqs2)//2], torch.abs(Bf2[:len(frqs2)//2]), color=col[j])
    axs2[1].plot(frqs2[0:len(frqs2)//2], torch.angle(Bf2[:len(frqs2)//2]), color=col[j])
    axs2[2].plot(off_tmp, 0, color=col[j],marker = '*')
    if j == 0:  
        axs2[2].plot(np.array([-1.5, -1.5, 1.5, 1.5, -1.5]), np.array([-1.5, 1.5, 1.5, -1.5, -1.5]), color='k')
        axs2[2].axis('equal')

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


axs2[0].set_title('Abs spectra')
axs2[1].set_title('Phase spectra')
axs2[0].set_xscale('log')
axs2[1].set_xscale('log')
axs2[0].set_xlim([1e3,1e6])
axs2[1].set_xlim([1e3,1e6])
# axs2[1].set_ylim([-np.pi,np.pi])
axs2[1].set_xlabel('f (Hz)')
axs2[0].set_xlabel('f (Hz)')
axs2[1].set_ylabel('phase (rad)')
axs2[0].set_ylabel('B (T)')

axs2[2].set_xlabel('offset (m)')
axs2[2].set_ylabel('offset (m)')

plt.tight_layout()

# plt.savefig(name + '.png')

plt.show()