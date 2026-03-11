from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd

def test_tikhonov_few_layer():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-3, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-7)

    rho = torch.tensor([20, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = fwrTD(rho, thk).exp()
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=1.)
    d_err = torch.log1p(d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(fwrTD.times, dbdt_true, '-', label='Observed')
        plt.loglog(fwrTD.times, d_obs, 'o', label='Observed')
        plt.show()

    d_obs = torch.abs(d_obs).log()

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
        axs[1].loglog(fwrTD.times, d_obs.exp(), 'o', label='Observed')
        axs[1].loglog(fwrTD.times, d_pred.exp(), '-', label='Predicted')
        plt.show()

def test_tikhonov_smooth():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-2, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-5)

    rho = torch.tensor([20, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = fwrTD(rho, thk).exp()
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.)
    d_err = torch.log1p(d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    d_obs = torch.abs(d_obs).log()

    res_start = LogResistivity.smooth(torch.ones(31)*50, torch.logspace(-1,2.2,30)+1)


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
        axs[1].loglog(fwrTD.times, d_obs.exp(), 'o', label='Observed')
        axs[1].loglog(fwrTD.times, d_pred.exp(), '-', label='Predicted')
        plt.show()

def test_fastjacobian():
    logForward = logify_forward(Forward)

    times = torch.logspace(-5,-2, 32)
    fwrTD = logForward.example(times=times, tolerance=1e-6)

    rho = torch.tensor([40, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = fwrTD(rho, thk).exp()
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.,relative_noise=0.03,noise_level_at_1ms=1e-8)
    d_err = torch.log1p(d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    d_obs = torch.abs(d_obs).log()


    fastJac = fastjacobian(Inversion)
    res_start = LogResistivity.smooth(torch.ones(31)*40, torch.tensor(np.cumsum(np.logspace(0, 1.3, 30))))

    # print(fwrTD)
    inv = fastJac(d_obs, d_err, res_start, fwrTD)
    # inv = fastJac(dbdt_true.log(), d_err, res_start, fwrTD)
    inv.setup_matrices()
    doi=inv.calc_doi()
    inv.data_misfit_acceptance_threshold = 0.9
    inv.vertical_smoothness=0.4
    res_mod= inv.invert(iter_cap=10)
    doi = inv.doi        

    if __name__ == '__main__':
        import matplotlib.pyplot as plt

        fig,axs = plt.subplots(1,2)
        
        ax = res_start.plot(axs=axs[:1], color='black', linestyle='--')
        ax = res_mod.plot(axs=axs[:1])
        axs[0].axhline(y=doi, color='k', linestyle='--', label='DOI')
        mod_true.plot(axs=axs[:1], color='red', label='True model')

        d_pred = fwrTD(*res_mod.mod)
        axs[1].loglog(fwrTD.times, d_obs.exp(), 'o', label='Observed')
        axs[1].loglog(fwrTD.times, d_pred.exp(), '-', label='Predicted')
        plt.show()

def test_lci():
    gpu = torch.device('cpu')
    print('number of threads: ', torch.get_num_threads())
    logForward = logify_forward(Forward)

    t1 = torch.logspace(-6, -4, 15)
    t2 = torch.logspace(-5, -3, 31)
    tx = Loop.square(3)
    rx = Receiver(x=10.)
    wv1 = Waveform(torch.tensor([-0.5e-3, -0.9e-3, 0., 0.5e-6]), torch.tensor([0, 1, 1, 0]))
    wv2 = Waveform(torch.tensor([-1e-3, -1.5e-3, 0., 0.5e-5]), torch.tensor([0, 1, 1, 0]))

    fwr1 = logForward(tx, rx, t1, waveform=wv1, tolerance=1e-6)
    fwr2 = logForward(tx, rx, t2, waveform=wv2, tolerance=1e-6)

    fwr = fwr1 + fwr2


    z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
    z /= z.max()

    z *= 80.

    rho_true = torch.tensor([75., 5., 60.], device=gpu, dtype=rdtype)
    thk_true = torch.tensor([20, 30], device=gpu, dtype=rdtype)


    res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z)
    thick1 = torch.linspace(0, 20, 25).to(rdtype)#[:3]
    inv_list = []
    m_trues=[]

    for i,dthick in enumerate(thick1):
        this_fwr = fwr.copy()
        thk = thk_true.clone()
        thk[0] += dthick
        crho = rho_true.clone()
        crho[1] += abs(i-len(thick1)//2)*2.
        crho[2] -= abs((i-5)*70/len(thick1))
        m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
        m_trues.append(m_true)
        
        dbdt = this_fwr(crho, thk).exp()
        d_err = compute_relstd(fwr.times, dbdt, current=10.)
        d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log()
        tmp_inv = fastJac(d_obs, torch.log1p(d_err), res_start.copy(), this_fwr)

        # tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions")
    from time import time as timer
    bt = timer()
    lci = LCI(inv_list)
    lci.horizontal_smoothness=0.3
    lci.vertical_smoothness=1

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2
    from time import time as timer
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    res_mods= lci.invert(iter_cap=2)
    dois = lci.doi
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from plog import Log, Section, PieceWiseLine

        fig,ax = plt.subplots(2,1)
        result = []
        actual = []
        _,thick = res_mods[0].values
        z = np.cumsum(thick)
        for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
            result.append(Log.standard(res_mod.values[0].cpu().log10(), z, 'rho', xy=(i,0)))
            cres,thick = mtrue.mod
            actual.append(Log.standard(cres.cpu().log10(), np.cumsum(thick.cpu()), 'rho', xy=(i,0)))
        result = Section(result, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
        actual = Section(actual, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
        result.plot(axs=ax[0], cmap='turbo', vmin=0, vmax=2)
        actual.plot(axs=ax[1], cmap='turbo', vmin=0, vmax=2)
        ax[0].plot(range(len(res_mods)),dois,'k')
        plt.show()

def test_sci():
    gpu = torch.device('cpu')
    print('number of threads: ', torch.get_num_threads())
    logForward = logify_forward(Forward)

    t1 = torch.logspace(-6, -4, 15)
    t2 = torch.logspace(-5, -3, 31)
    tx = Loop.square(3)
    rx = Receiver(x=10.)
    wv1 = Waveform(torch.tensor([-0.5e-3, -0.9e-3, 0., 0.5e-6]), torch.tensor([0, 1, 1, 0]))
    wv2 = Waveform(torch.tensor([-1e-3, -1.5e-3, 0., 0.5e-5]), torch.tensor([0, 1, 1, 0]))

    fwr1 = logForward(tx, rx, t1, waveform=wv1, tolerance=1e-6)
    fwr2 = logForward(tx, rx, t2, waveform=wv2, tolerance=1e-6)

    fwr = fwr1 + fwr2


    z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
    z /= z.max()

    z *= 80.

    rho_true = torch.tensor([75., 5., 60.], device=gpu, dtype=rdtype)
    thk_true = torch.tensor([20, 30], device=gpu, dtype=rdtype)


    res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z)
    thick1 = torch.linspace(0, 20, 25).to(rdtype)#[:3]
    inv_list = []
    m_trues=[]

    for i,dthick in enumerate(thick1):
        this_fwr = fwr.copy()
        thk = thk_true.clone()
        thk[0] += dthick
        crho = rho_true.clone()
        crho[1] += abs(i-len(thick1)//2)*2.
        crho[2] -= abs((i-5)*70/len(thick1))
        m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
        m_trues.append(m_true)
        
        dbdt = this_fwr(crho, thk).exp()
        d_err = compute_relstd(fwr.times, dbdt, current=10.)
        d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log()
        tmp_inv = fastJac(d_obs, torch.log1p(d_err), res_start.copy(), this_fwr)

        # tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions")
    n_pts = len(thick1)

    # 1 km radius (in meters)
    radius = 1000.0

    # Random angles [0, 2π)
    angles = np.random.uniform(0, 2*np.pi, n_pts)

    # Random radii with uniform spatial density
    r = radius * np.sqrt(np.random.uniform(0, 1, n_pts))

    # Convert to Cartesian coordinates
    x_positions = r * np.cos(angles)
    y_positions = r * np.sin(angles)
    from time import time as timer
    bt = timer()
    sci = SCI(inv_list, x = np.array(x_positions), y = np.array(y_positions))
    sci.horizontal_smoothness=0.7
    sci.vertical_smoothness=1

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2
    from time import time as timer
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    res_mods = sci.invert(iter_cap=10)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from plog import Log, Section, PieceWiseLine

        fig,ax = plt.subplots(2,1)
        result = []
        actual = []
        _,thick = res_mods[0].mod
        z = np.cumsum(thick)
        for i, (mtrue, res_mod) in enumerate(zip(m_trues, res_mods)):
            result.append(Log.standard(res_mod.values[0].cpu().log10(), z, 'rho', xy=(i,0)))
            cres,thick = mtrue.mod
            actual.append(Log.standard(cres.cpu().log10(), np.cumsum(thick.cpu()), 'rho', xy=(i,0)))
        result = Section(result, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
        actual = Section(actual, proj_pnts=[(-1, 0), (i/2, 0), (i+1, 0)])
        result.plot(axs=ax[0], cmap='turbo', vmin=0, vmax=2)
        actual.plot(axs=ax[1], cmap='turbo', vmin=0, vmax=2)
        plt.show()




if __name__ == '__main__':
    # test_tikhonov_few_layer()
    # test_tikhonov_smooth()
    test_fastjacobian()
    # test_lci()
    #test_sci()
    pass
