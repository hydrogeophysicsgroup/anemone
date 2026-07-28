from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd

#def test_tikhonov_smooth():
logForward = logify_forward(Forward)

times = torch.logspace(-5.5,-2, 52)
fwrTD = logForward.example(times=times, tolerance=1e-6)

rho = torch.tensor([20, 80, 40])
thk = torch.tensor([30,40.])
mod_true = LogResistivity.few_layers(rho, thk)


dbdt_true = fwrTD(rho, thk).exp()
d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=1.)
d_err = torch.log1p(d_relerr)
d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

d_err[d_obs<0.] = torch.inf

d_obs = torch.abs(d_obs).log()

res_start = LogResistivity.smooth(torch.ones(31)*50, torch.logspace(0,1,30).cumsum(0))
# res_start.plot()
# plt.show()

inv = Inversion(d_obs, d_err, res_start, fwrTD)
inv.data_misfit_acceptance_threshold = 0.9
res_mod = inv.invert(iter_cap=10)    
misfit_final = max(1, inv.misfits[-1].mean().item()*1.1)
print(misfit_final)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Original fit plot ---
    fig, axs = plt.subplots(1, 2)
    ax = res_start.plot(axs=axs[:1], color='black', linestyle='--')
    ax = res_mod.plot(axs=axs[:1])
    mod_true.plot(axs=axs[:1], color='red', label='True model')


    d_pred = fwrTD(*res_mod.mod)
    axs[1].loglog(fwrTD.times, d_obs.exp(), 'o', label='Observed')
    axs[1].loglog(fwrTD.times, d_pred.exp(), '-', label='Predicted')
    axs[1].legend()

    # --- DOI sweep ---
    rho_mod, thk_mod = res_mod.mod
    n_layers = len(rho_mod)
    rho_sweep = torch.logspace(0, 3, 15)   # 1 … 1000 Ω·m
    valid = ~torch.isinf(d_err)

    rho_grid = rho_mod[:, None] + torch.zeros_like(rho_sweep)[None, :]
    rho_grid[-1,:] = rho_sweep

    hi_res = rho_mod*1.
    lo_res = rho_mod*1.

    tmp_hi_res = rho_mod*1.
    tmp_lo_res = rho_mod*1.

    cmap='GnBu_r'
    n_lines=25
    cm = plt.get_cmap(cmap)
    axs[0].set_prop_cycle(color=[cm(x) for x in np.linspace(0.3, 1., n_lines)])


    d_pred_sweep = fwrTD(rho_grid, thk_mod)
    d_misfit = ((d_obs[None,:] - d_pred_sweep)**2 / d_err[None, :]**2).mean(-1)**0.5
    if (d_misfit<misfit_final).any():

        c_lo = rho_sweep[d_misfit<misfit_final].min()
        c_hi = rho_sweep[d_misfit<misfit_final].max()

        hi_res[-1] = c_hi
        lo_res[-1] = c_lo

    for k in range(2, n_layers + 1):
        rho_grid[-k:,:] = rho_sweep[None,:]

        d_pred_sweep = fwrTD(rho_grid, thk_mod)
        d_misfit = ((d_obs[None,:] - d_pred_sweep)**2 / d_err[None, :]**2).mean(-1)**0.5
        if (d_misfit<misfit_final).any():
            c_lo = rho_sweep[d_misfit<misfit_final].min()
            c_hi = rho_sweep[d_misfit<misfit_final].max()

            hi_res[-k] = c_hi
            lo_res[-k] = c_lo

            rho_sweep = torch.logspace(c_lo.log10(), c_hi.log10(), 15)   # 1 … 1000 Ω·m

            tmp_hi_res[-k:] = c_hi
            tmp_lo_res[-k:] = c_lo

            tmp_hi_resmod = LogResistivity.smooth(tmp_hi_res, torch.logspace(0,1,30).cumsum(0))#.value_z_plot('resistivity')
            tmp_lo_resmod = LogResistivity.smooth(tmp_lo_res, torch.logspace(0,1,30).cumsum(0))#.value_z_plot('resistivity')

            tmp_hi_resmod.plot(axs=axs[:1], linewidth=1.)
            tmp_lo_resmod.plot(axs=axs[:1], linewidth=1.)
           
        else:
            break
    res_mod.plot(axs=axs[:1], color='C0')
    mod_true.plot(axs=axs[:1], color='red', label='True model')
    # hi_resmod,z = LogResistivity.smooth(hi_res, torch.logspace(0,1,30).cumsum(0)).value_z_plot('resistivity')
    # lo_resmod,z = LogResistivity.smooth(lo_res, torch.logspace(0,1,30).cumsum(0)).value_z_plot('resistivity')
    # i_z = np.argsort(z)

    # axs[0].fill_betweenx(z[i_z], lo_resmod[i_z], hi_resmod[i_z], alpha=0.5)
    axs[0].set_xscale('log')
    plt.tight_layout()
    plt.show()