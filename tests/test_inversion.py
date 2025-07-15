from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd

def test_tikhonov_few_layer():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-2, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-7)

    rho = torch.tensor([20, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = 10**fwrTD(rho, thk)
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.)
    d_err = torch.log10(1. + d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    # if __name__ == '__main__':
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.loglog(fwrTD.times, dbdt_true, '-', label='Observed')
    #     plt.loglog(fwrTD.times, d_obs, 'o', label='Observed')
    #     plt.show()

    d_obs = torch.abs(d_obs).log10()

    res_start = LogResistivity.few_layers([2*torch.tensor([20,21,20], dtype=torch.double)], torch.tensor([30,30]))
    inv = BaseInversion(d_obs, d_err, res_start, fwrTD)
    inv.data_misfit_acceptance_threshold = 0.9
    res_mod = inv.invert(iter_cap=10)        

    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        fig,axs = plt.subplots(1,2)
        
        ax = res_start.plot(axs=axs[:1], color='black', linestyle='--')
        ax = res_mod.plot(axs=axs[:1])
        mod_true.plot(axs=axs[:1], color='red', label='True model')

        d_pred = fwrTD(*res_mod.mod)
        axs[1].loglog(fwrTD.times, 10**d_obs, 'o', label='Observed')
        axs[1].loglog(fwrTD.times, 10**d_pred, '-', label='Predicted')
        plt.show()

def test_tikhonov_smooth():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-2, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-5)

    rho = torch.tensor([20, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = 10**fwrTD(rho, thk)
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.)
    d_err = torch.log10(1. + d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    d_obs = torch.abs(d_obs).log10()

    res_start = LogResistivity.smooth(torch.ones(31, dtype=torch.double)*50, torch.cumsum(torch.logspace(0,1.5,30), 0))

    inv = Tikhonov(d_obs, d_err, res_start, fwrTD)
    inv.data_misfit_acceptance_threshold = 0.9
    res_mod = inv.invert(iter_cap=10)        

    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        fig,axs = plt.subplots(1,2)
        
        ax = res_start.plot(axs=axs[:1], color='black', linestyle='--')
        ax = res_mod.plot(axs=axs[:1])
        mod_true.plot(axs=axs[:1], color='red', label='True model')

        d_pred = fwrTD(*res_mod.mod)
        axs[1].loglog(fwrTD.times, 10**d_obs, 'o', label='Observed')
        axs[1].loglog(fwrTD.times, 10**d_pred, '-', label='Predicted')
        plt.show()

def test_fastjacobian():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-2, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-4)

    rho = torch.tensor([20, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = 10**fwrTD(rho, thk)
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.)
    d_err = torch.log10(1. + d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    d_obs = torch.abs(d_obs).log10()

    res_start = LogResistivity.smooth(torch.ones(31, dtype=torch.double)*50, torch.cumsum(torch.logspace(0,1.5,30), 0))

    fastJac = fastjacobian(Tikhonov)

    inv = fastJac(d_obs, d_err, res_start, fwrTD)
    inv.data_misfit_acceptance_threshold = 0.9
    res_mod = inv.invert(iter_cap=10)        

    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        fig,axs = plt.subplots(1,2)
        
        ax = res_start.plot(axs=axs[:1], color='black', linestyle='--')
        ax = res_mod.plot(axs=axs[:1])
        mod_true.plot(axs=axs[:1], color='red', label='True model')

        d_pred = fwrTD(*res_mod.mod)
        axs[1].loglog(fwrTD.times, 10**d_obs, 'o', label='Observed')
        axs[1].loglog(fwrTD.times, 10**d_pred, '-', label='Predicted')
        plt.show()

def test_lci():
    gpu = torch.device('cpu')
    logForward = logify_forward(Forward)
    t = torch.logspace(-5.5, -2, 51, device=gpu, dtype=torch.double)
    fwr = logForward.example(times=t, tolerance=1e-3)
    fastJac = fastjacobian(Inversion)

    z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
    z /= z.max()
    z *= 150

    rho_true = torch.tensor([75., 10., 60.], device=gpu, dtype=torch.double)
    thk_true = torch.tensor([50, 30], device=gpu, dtype=torch.double)


    res_start = LogResistivity.smooth(torch.ones(21)*50., z)
    thick1 = torch.linspace(0, 20, 15).to(torch.double)#[:3]
    inv_list = []
    m_trues=[]

    for i,dthick in enumerate(thick1):
        thk = thk_true.clone()
        thk[0] += dthick
        crho = rho_true.clone()
        crho[1] += abs(i-len(thick1)//2)*2.
        crho[2] -= abs((i-5)*70/len(thick1))
        m_true = LogResistivity.few_layers(crho, thk)
        m_trues.append(m_true)
        
        dbdt = 10**fwr(crho, thk)#.exp()
        d_err = compute_relstd(fwr.times, dbdt, current=5.)
        d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log10()
        tmp_inv = fastJac(d_obs, (1. + d_err).log10(), res_start.copy(), fwr)
        # tmp_inv.vertical_smoothness = 5e-3
        # tmp_inv.max_step = 50.
        tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions")
    lci = LCI(inv_list)
    lci.smoothness_vert = 1e3
    lci.smoothness_horiz = 1e2
    res_mods = lci.invert(iter_cap=10)

    import matplotlib.pyplot as plt
    from plog import Log, Section

    fig,ax = plt.subplots(2,1)
    result = []
    actual = []
    _,thick = res_mods[0].mod
    z = torch.cumsum(thick, 0)
    for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
        result.append(Log.standard(res_mod.values[0].log10(), z, 'rho', xy=(i,0)))
        cres,thick = mtrue.mod
        actual.append(Log.standard(cres.log10(), np.cumsum(thick), 'rho', bottom=160., xy=(i,0)))
    result = Section(result)
    actual = Section(actual)
    result.plot(axs=ax[0], cmap='turbo', vmin=0, vmax=2)
    actual.plot(axs=ax[1], cmap='turbo', vmin=0, vmax=2)
    plt.show()



if __name__ == '__main__':
    # test_tikhonov_few_layer()
    # test_tikhonov_smooth()
    # test_fastjacobian()
    test_lci()
    pass
