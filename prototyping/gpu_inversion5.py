# %%
from anemone.forwards import *
import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy as sp

from anemone.experiment import compute_relstd
from anemone.inversion import *
from anemone import rdtype

gpu = torch.device('cpu')
print('number of threads: ', torch.get_num_threads())
logForward = logify_forward(Forward)
t = torch.logspace(-5, -3, 31, device=gpu, dtype=rdtype)
fwr0 = logForward.example(times=t, tolerance=1e-7)
fwr = logForward.example(times=t, tolerance=1e-3)
fastJac = fastjacobian(dolphinv.Inversion)



z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
z /= z.max()

z *= 150

rho_true = torch.tensor([75., 10., 60.], device=gpu, dtype=rdtype)
thk_true = torch.tensor([50, 30], device=gpu, dtype=rdtype)


res_start = LogResistivity.smooth(torch.ones(len(z)+1)*40., z, device=gpu)
res_start.set_device(gpu)
thick1 = torch.linspace(0, 20, 100).to(rdtype)
inv_list0 = []
inv_list1 = []
m_trues=[]

for i,dthick in enumerate(thick1):
    this_fwr = fwr.copy()
    thk = thk_true.clone()
    thk[0] += dthick
    crho = rho_true.clone()
    crho[1] += abs(i-len(thick1)//2)*2.
    crho[2] -= abs((i-5)*70/len(thick1))
    crho[2] = max(crho[2], 20.)
    m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
    m_trues.append(m_true)
    
    dbdt = this_fwr(crho, thk).exp()
    # plt.figure()
    # plt.loglog(fwr.times.cpu(), dbdt.cpu(), label='data')
    # plt.show()
    d_err = compute_relstd(fwr.times, dbdt.cpu(), current=10.).to(gpu)
    d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log().to(gpu)
    tmp_inv = fastJac(d_obs, (1. + d_err).log(), res_start.copy(), this_fwr)

    # tmp_inv.setup_matrices()
    inv_list0.append(tmp_inv.copy())    

    tmp_inv = fastJac(d_obs, (1. + d_err).log(), res_start.copy(), fwr.copy())

    inv_list1.append(tmp_inv.copy())

print(f"Created {len(inv_list0)} inversions")
from time import time as timer
bt = timer()
lci = LCI(inv_list0)
lci.set_device(gpu)
res_mods = lci.invert(iter_cap=25)
# print('lci._GtG.shape: ', lci._GtG.shape)
# print('lci.CtC.shape: ', lci.CtC.shape)

print(f'time for normal inversion: {timer()-bt:2.2f} s')

# plt.figure()
# plt.matshow(lci.CtC.cpu().numpy())
# plt.show()

# bt = timer()
# lci = specialLCI(inv_list1)
# lci.set_device(gpu)
# res_mods = lci.invert(iter_cap=10)
# print(f'time for special inversion: {timer()-bt:2.2f} s')
# # lci.smoothness_lat

# # lci.smoothness_vert = 1e3
# # lci.smoothness_horiz = 1e2
# from time import time as timer
# import cProfile
# import pstats
# profiler = cProfile.Profile()
# profiler.enable()

# res_mods = lci.invert(iter_cap=10)

# print(f'NUMBER OF ITERATIONS: {lci.iteration}')
# print(f'NUMBER OF MODEL UPDATES: {lci.n_model_updates}')

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.sort_stats('tottime').print_stats(20)  # Print top 20 by cumulative time

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plog import Log, Section, PieceWiseLine

    fig,ax = plt.subplots(2,1)
    result = []
    actual = []
    _,thick = res_mods[0].mod
    z = torch.cumsum(thick, 0)
    for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
        result.append(Log.standard(res_mod.values[0].cpu().log10().numpy(), z.cpu().numpy(), 'rho', xy=(i,0)))
        cres,thick = mtrue.mod
        actual.append(Log.standard(cres.cpu().log10().numpy(), np.cumsum(thick.cpu().numpy()), 'rho', xy=(i,0)))
    result = Section(result, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
    actual = Section(actual, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
    result.plot(axs=ax[0], cmap='turbo',  vmin=0, vmax=2)
    actual.plot(axs=ax[1], cmap='turbo',  vmin=0, vmax=2)


    plt.figure()
    plt.semilogy(lci.misfits, label='misfits')
    plt.semilogy(lci.dampf, label='damp_fs')
    plt.semilogy(lci.target_step, label='target_step')
    plt.semilogy(lci.actual_step, label='actual_step')
    plt.semilogy(lci.data_misfits, label='data_misfits')
    plt.legend()    
    plt.show()

    # %%



