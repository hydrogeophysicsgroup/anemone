from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd

gpu = torch.device('cpu')
print(torch.get_num_threads())
# breakpoint()
logForward = logify_forward(Forward)
t = torch.logspace(-5.5, -2, 51, device=gpu, dtype=torch.double)
fwr = logForward.example(times=t, tolerance=1e-5)
fastJac = fastjacobian(Tikhonov)

z = np.cumsum(np.logspace(0, np.log10(50.), 30))
z /= z.max()
z *= 150

rho_true = torch.tensor([75., 10., 60.], device=gpu, dtype=torch.double)
thk_true = torch.tensor([50, 30], device=gpu, dtype=torch.double)


res_start = LogResistivity.smooth(np.ones(len(z)+1)*50., z)
thick1 = torch.linspace(0, 20, 100)
inv_list = []
m_trues=[]

for i,dthick in enumerate(thick1):
    thk = thk_true.clone()
    thk[0] += dthick
    crho = rho_true.clone()
    crho[1] += abs(i-len(thick1)//2)*2.
    crho[2] = max(2, abs((i-5)*70/len(thick1)))
    m_true = LogResistivity.few_layers(crho, thk)
    m_trues.append(m_true)
    
    dbdt = 10**fwr(crho, thk)#.exp()
    d_err = compute_relstd(fwr.times, dbdt, current=25.)
    d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log10()
    tmp_inv = fastJac(d_obs.numpy(), (1. + d_err).log10().numpy(), res_start.copy(), fwr)

    tmp_inv.setup_matrices()
    inv_list.append(tmp_inv)    
print(f"Created {len(inv_list)} inversions")
from time import time as timer
bt = timer()
lci = LCI(inv_list)
lci.smoothness_vert = 1e3
lci.smoothness_horiz = 1e2

import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

res_mods = lci.invert(iter_cap=5)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)  # Show top 20 time-consuming calls


print(f"LCI inversion took {timer()-bt:.2f} seconds")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from plog import Log, Section, PieceWiseLine

    fig,ax = plt.subplots(2,1)
    result = []
    actual = []
    _,thick = res_mods[0].mod
    z = np.cumsum(thick)
    for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
        result.append(Log.standard(res_mod.values[0].log10(), z, 'rho', bottom=200, xy=(i,0)))
        cres,thick = mtrue.mod
        actual.append(Log.standard(np.log10(cres.numpy()), np.cumsum(thick.numpy()), 'rho', bottom=200, xy=(i,0)))
    profile = ([-1,0], [len(thick1)+1, 0])
    result = Section(result, proj_pnts=profile)
    actual = Section(actual, proj_pnts=profile)
    result.plot(axs=ax[0], cmap='turbo', vmin=0, vmax=2)
    actual.plot(axs=ax[1], cmap='turbo', vmin=0, vmax=2)
    plt.show()