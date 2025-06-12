import matplotlib.pyplot as plt
import numpy as np
import torch
from anemone.experiment import *
from anemone.forwards import *
from anemone.inversion import SmoothModelColeCole, FewLayerColeCole
from anemone.inversion import *

#define the loops and filters
tx = Loop.square(3)
rx = Receiver(x=13)
waveform = Waveform.example()
recfilter = FilterChain([
    ButterworthFilter(2.4e5, 1),
    ButterworthFilter(6.2e5, 1),
    ButterworthFilter(4.5e5, 2)
])

# make the forward operators
times = torch.logspace(-5, -3, 32)
frqs = (torch.arange(32)+1)*4e3

#IP forwards
fwrTDIP = IPForward(tx, rx, times, waveform=waveform, filterfunc=recfilter)
fwrFDIP = IPForwardFD(tx, rx, frqs)

#res only forwards
fwrTD = Forward(tx, rx, times, waveform=waveform, filterfunc=recfilter, tolerance=1e-7)
fwrFD = ForwardFD(tx, rx, frqs)

#make the true model
res = np.array([50,50])
m   = np.array([1e-6,1e-6])
tau = np.array([1e-4,1e-4])
C   = np.array([0.5,0.5])
thk = np.array([30])

cmod_true = FewLayerColeCole(res, m, tau, C, thk)

#forward the true data
dbdt_true = fwrTDIP(*cmod_true.mod).real
Bfrq_true = fwrFDIP(*cmod_true.mod).real

dbdt_err = dbdt_true*0.03
Bfrq_err = Bfrq_true*1e-3

#make the starting model
res_st = res[0]
n_layers = 31

mod_st = SmoothModel.starting_model(n_layers, depth_bottom=100,rho=res_st)


#define the data transform - this is to work with absolute data
def transformFD(data):
    return torch.abs(data)

def transformTD(data):
    return data


dbdt_test = fwrTD(*mod_st.mod).real
print('shape of dbdt: ', dbdt_test.shape)
Bfrq_test = fwrFD(*mod_st.mod).real

invTD   = Inversion(dbdt_true, dbdt_err, mod_st, fwrTD, data_transform=transformTD)

mod_stFD = SmoothModel.starting_model(n_layers, depth_bottom=100,rho=res_st)
mod_stFD.thick = torch.logspace(0, 1, mod_stFD.n_layers-1)
mod_stFD.thick *= 100./mod_stFD.thick.sum()

invFD   = Inversion(Bfrq_true, torch.abs(Bfrq_err), mod_stFD, fwrFD, data_transform=transformFD)


res_mod_FD = invFD.invert(smoothness=1e2,iter_cap=1)
res_mod_TD = invTD.invert(smoothness=1e2,iter_cap=1)

Bfrq_est = fwrFD(*res_mod_FD.mod).real
dbdt_est = fwrTD(*res_mod_TD.mod).real


fig,axs = plt.subplots(2,4, figsize=(12,6))
ax_IP = axs[:,:-2].flatten()
ax_res = axs[0,0]

cmod_true.plot(axs=ax_IP, color='k')

axs[0,-2].plot(frqs, torch.abs(Bfrq_true),'k')
axs[1,-2].loglog(times, dbdt_true,'k')
axs[1,-2].loglog(times,-dbdt_true,'k',linestyle='--')

axs[0,-2].plot(frqs, torch.abs(Bfrq_est),'r')
axs[1,-2].loglog(times, dbdt_est,'r')
axs[1,-2].loglog(times,-dbdt_est,'r',linestyle='--')

axs[0,-2].errorbar(frqs, torch.abs(Bfrq_true), yerr=torch.abs(Bfrq_err), fmt='k', capsize=5)
axs[1,-2].errorbar(times, dbdt_true, yerr=torch.abs(dbdt_err), fmt='k', capsize=5)
axs[1,-2].errorbar(times,-dbdt_true, yerr=torch.abs(dbdt_err), fmt='k', capsize=5)


axs[0,-1].plot(frqs, torch.abs(Bfrq_est-Bfrq_true)/torch.abs(Bfrq_true)*100,'k')
axs[1,-1].plot(times, torch.abs(dbdt_est-dbdt_true)//torch.abs(dbdt_true)*100,'k')


res_mod_FD.plot(ax=ax_res, color='b',label='FD')
res_mod_TD.plot(ax=ax_res, color='r',label = 'TD')
ax_res.legend()

plt.show()

# fig,axs = plt.subplots(1,2)
# axs[0].loglog(times, dbdt_true.T, 'C0')
# axs[0].loglog(times,-dbdt_true.T, 'C0--')
# axs[0].set_aspect('equal')
# axs[1].loglog(frqs, torch.abs(Bfrq_true).T, 'C0');

# plt.show()