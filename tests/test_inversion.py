from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd

import matplotlib.pyplot as plt

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
    res_mod = inv.invert(iter_cap=25)        

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
    res_mod = inv.invert(iter_cap=25)        

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
    torch.manual_seed(0)
    times = torch.logspace(-5,-2, 32)
    fwrTD = Forward.example(times=times, tolerance=1e-6, log_scale=True)

    rho = torch.tensor([40, 80, 40])
    thk = torch.tensor([30,40.])
    mod_true = LogResistivity.few_layers(rho, thk)


    dbdt_true = fwrTD(rho, thk).exp()
    d_relerr = compute_relstd(fwrTD.times, dbdt_true, current=10.)
    d_err = torch.log1p(d_relerr)
    d_obs = dbdt_true + dbdt_true * torch.randn_like(dbdt_true) * d_relerr

    d_obs = torch.abs(d_obs).log()


    res_start = LogResistivity.smooth(torch.ones(31)*60, torch.tensor(np.cumsum(np.logspace(0, 1.3, 30))))

    inv = fastJac(d_obs, d_err, res_start.copy(), fwrTD)
    inv.vertical_smoothness=0.1
    res_mod= inv.invert(iter_cap=10)
    doi = inv.doi        

    assert inv.misfits[-1] < inv.misfits[0]

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

def test_analytic_jacobian_matches_log_resistivity_finite_difference():
    times = torch.logspace(-5, -3, 12)
    forward = Forward.example(times=times, tolerance=1e-6)
    rho = torch.tensor([40., 80., 40.])
    thickness = torch.tensor([30., 40.])

    jacobian = forward(rho, thickness, jac=True)
    forward.primary = torch.ones_like(forward.primary)
    assert torch.equal(forward(rho, thickness, jac=True), jacobian)

    epsilon = 1e-4
    for layer in range(len(rho)):
        rho_plus = rho.clone()
        rho_minus = rho.clone()
        rho_plus[layer] *= torch.exp(torch.tensor(epsilon))
        rho_minus[layer] *= torch.exp(torch.tensor(-epsilon))

        finite_difference = (
            forward(rho_plus, thickness) - forward(rho_minus, thickness)
        ) / (2 * epsilon)
        analytic = jacobian[layer] * rho[layer]

        assert torch.allclose(analytic, finite_difference, rtol=1e-6, atol=1e-12)

    batched_rho = torch.column_stack((rho, rho * torch.tensor([1.2, 0.8, 1.1])))
    batched_jacobian = forward(batched_rho, thickness, jac=True)
    for site in range(batched_rho.shape[1]):
        for layer in range(batched_rho.shape[0]):
            rho_plus = batched_rho.clone()
            rho_minus = batched_rho.clone()
            rho_plus[layer, site] *= torch.exp(torch.tensor(epsilon))
            rho_minus[layer, site] *= torch.exp(torch.tensor(-epsilon))

            finite_difference = (
                forward(rho_plus, thickness)[site]
                - forward(rho_minus, thickness)[site]
            ) / (2 * epsilon)
            analytic = batched_jacobian[site, layer] * batched_rho[layer, site]

            assert torch.allclose(analytic, finite_difference, rtol=1e-6, atol=1e-12)


def test_lci_horizontal_constraints_compare_adjacent_sites():
    from math import log
    import scipy as sp

    lci = object.__new__(LCI)
    n_sites = 4
    n_layers = 3
    lci.mvec = torch.log(torch.tensor([
        [10., 20., 30.],
        [11., 20., 30.],
        [13., 18., 30.],
        [13., 18., 35.],
    ])).flatten()
    lci.inv_list = [type('Site', (), {'m': torch.empty(n_layers)})()]
    lci.RL = lci.make_smoothness_matrix()

    residual = lci.RL @ lci.mvec
    expected = lci.mvec.reshape(n_sites, n_layers)[:-1] - lci.mvec.reshape(n_sites, n_layers)[1:]
    assert torch.allclose(torch.as_tensor(residual), expected.flatten())

    penalties = []
    for smoothness in (1.0, 1.5):
        lci.C_rl = sp.sparse.eye(lci.RL.shape[0])/log(1. + smoothness)**2
        penalties.append(lci.horz_misfit())
    assert penalties[0] > penalties[1] > 0


def test_rejected_step_preserves_accepted_linearization():
    class RejectingInversion(Inversion):
        def find_damping(self):
            return torch.ones_like(self.mvec)

        def mforward(self, mvec):
            self.forward_calls += 1
            return mvec + 10.

        def misfit(self):
            return torch.tensor(2.)

        def _model_update(self, mvec):
            self.model_updates += 1

    inversion = object.__new__(RejectingInversion)
    inversion.iteration = 1
    inversion.damping_attempts = 0
    inversion.pslices = [slice(None)]
    inversion.mvec = torch.zeros(2)
    inversion._current_mvec = inversion.mvec.clone()
    inversion._linearization_mvec = inversion.mvec.clone()
    inversion._dpred = torch.full((2,), 10.)
    inversion.misfits = [torch.tensor(1.)]
    inversion.max_step = torch.full((1,), torch.log(torch.tensor(3.)))
    inversion.min_step = torch.full((1,), torch.log(torch.tensor(1.1)))
    inversion.step_down = torch.log(torch.tensor(2.))
    inversion.step_up = torch.log(torch.tensor(1.2))
    inversion.damp_exponent = torch.ones(1)
    inversion.n_local = 0
    inversion.forward_calls = 0
    inversion.model_updates = 0

    inversion.run_iteration()

    assert torch.equal(inversion.mvec, torch.zeros_like(inversion.mvec))
    assert torch.equal(inversion._dpred, torch.full((2,), 10.))
    assert inversion.forward_calls == 1
    assert inversion.model_updates == 0


def test_lci_linearization_reuses_accepted_prediction():
    import scipy as sp

    class CountingLCI(LCI):
        def mforward(self, mvec):
            self.forward_calls += 1
            return torch.full_like(mvec, 99.)

        def jacobian(self, mvec):
            self.jacobian_calls += 1
            self._r_data = torch.zeros_like(mvec).numpy()
            self.cmvec = mvec.numpy()

    inversion = object.__new__(CountingLCI)
    inversion.mvec = torch.ones(2)
    inversion._dpred = torch.full((2,), 10.)
    inversion.CtC = sp.sparse.eye(2, format='csr')
    inversion.n_model_updates = 0
    inversion.forward_calls = 0
    inversion.jacobian_calls = 0

    inversion._linearize_current_model()

    assert inversion.forward_calls == 0
    assert inversion.jacobian_calls == 1
    assert inversion.n_model_updates == 1
    assert torch.equal(inversion._linearization_mvec, inversion.mvec)

def test_damping_remains_active_for_weakly_sensitive_parameters():
    import scipy as sp

    inversion = object.__new__(Inversion)
    inversion.mvec = torch.zeros(2)
    inversion._current_mvec = inversion.mvec.clone()
    inversion._GtG = sp.sparse.diags([1., 1e-4], format='csr')
    inversion.CtC = sp.sparse.diags([1., 1.], format='csr')
    inversion.Imod = sp.sparse.eye(2, format='csr')
    inversion.pslices = [torch.ones(2, dtype=torch.bool)]
    inversion.damp_exponent = torch.ones(1)
    inversion.damping_attempts = 0
    inversion.diag_normalizer = 5e3
    inversion.damping_factor = 3.

    damping = inversion.damp_matrix()

    assert np.all(damping.diagonal() > 0)

def test_lci(n_mods=100):
    # n_mod
    gpu = torch.device('cuda')
    print('number of threads: ', torch.get_num_threads())
    logForward = logify_forward(Forward)

    t1 = torch.logspace(-6, -4, 15)
    t2 = torch.logspace(-5, -3, 31)
    tx = Loop.square(3)
    rx = Receiver(x=10.)
    
    wv1 = Waveform(torch.tensor([-0.5e-3, -0.9e-3, 0., 0.5e-6]), torch.tensor([0, 1, 1, 0]))
    wv2 = Waveform(torch.tensor([-1e-3, -1.5e-3, 0., 0.5e-5]), torch.tensor([0, 1, 1, 0]))

    fwr1 = logForward(tx, rx, t1, waveform=wv1, tolerance=1e-6, do_setup=False)
    fwr2 = logForward(tx, rx, t2, waveform=wv2, tolerance=1e-6, do_setup=False)
    # fwr2.set_device(gpu)

    fwr = fwr1 + fwr2
    fwr.set_device(gpu)


    z = torch.cumsum(torch.logspace(0, np.log10(50.), 30), 0)
    z /= z.max()

    z *= 80.

    rho_true = torch.tensor([75., 5., 60.], device=gpu, dtype=rdtype)
    thk_true = torch.tensor([20, 30], device=gpu, dtype=rdtype)

    n_mod_ones = torch.ones(n_mods, device=gpu, dtype=rdtype)
    rho2d = rho_true[:, None]*n_mod_ones[None,:]
    thk2d = thk_true[:, None]*n_mod_ones[None,:]



    res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z)
    thick1 = torch.linspace(0, 20, n_mods).to(rdtype).to(gpu)

    thk2d[1] += thick1

    dbdts = fwr(rho2d, thk2d)

    # plt.figure()
    # plt.plot(dbdts.T)
    # plt.show()
    # print(dbdts.shape)
    d_errs = compute_relstd(fwr.times, dbdts.exp().T, current=10.).T


    inv_list = []
    m_trues=[]
    from time import time as timer
    bt = timer()
    
    this_fwr = fwr.copy()
    this_fwr.set_device(gpu)

    for i,dthick in enumerate(thick1):
        # this_fwr = fwr.copy()
        # thk = thk_true.clone()
        # thk[0] += dthick
        # crho = rho_true.clone()
        # crho[1] += abs(i-len(thick1)//2)*2.
        # crho[2] -= abs((i-5)*70/len(thick1))
        # m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
        # m_trues.append(m_true)
        
        dbdt = dbdts[i].exp()
        d_err = d_errs[i]
        d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log()
        tmp_inv = fastJac(d_obs, torch.log1p(d_err), res_start.copy(), this_fwr)

        # tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions in {timer()-bt:.2f} seconds")

    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    lci = LCI(inv_list)
    lci.horizontal_smoothness=0.003
    lci.vertical_smoothness=0.003
    lci.reuse_jac = True

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2


    res_mods= lci.invert(iter_cap=15)
    dois = lci.doi
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        # from plog import Log, Section, PieceWiseLine

        fig,ax = plt.subplots(2,1)
        pcm=ax[0].matshow(lci.mvec.reshape(n_mods, -1).exp().log10().T, aspect='auto', cmap='turbo')

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

    import cProfile
    import pstats    
    sci = SCI(inv_list, x = np.array(x_positions), y = np.array(y_positions))
    sci.horizontal_smoothness=0.7
    sci.vertical_smoothness=1

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2

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


def rediscretize_matrices(z_old, z_new):
    """
    Parameters
    ----------
    z_old : ndarray, shape (N, M)
        Boundaries for N old grids.
    z_new : ndarray, shape (K,)
        Boundaries for the new grid.

    Returns
    -------
    ndarray, shape (N, M + 1, K + 1)
        Rediscretization matrices. The last layer of each grid is a
        half-space and is mapped directly to the final new layer.
    """
    z_old = np.asarray(z_old)
    z_new = np.asarray(z_new)

    if z_old.ndim != 2:
        raise ValueError("z_old must have shape (n_axes, n_boundaries)")
    if z_new.ndim != 1:
        raise ValueError("z_new must be one-dimensional")
    if np.any(np.diff(z_old, axis=1) <= 0):
        raise ValueError("Each old axis must be strictly increasing")
    if np.any(np.diff(z_new) <= 0):
        raise ValueError("z_new must be strictly increasing")

    n_axes, n_old_boundaries = z_old.shape
    n_new_boundaries = z_new.size

    old_lower = np.concatenate(
        [
            np.zeros((n_axes, 1), dtype=z_old.dtype),
            z_old,
        ],
        axis=1,
    )

    new_lower = np.concatenate(([0], z_new))
    new_upper = z_new

    # Calculate overlaps for the K finite new layers.
    overlap = (
        np.minimum(
            old_lower[:, :, None],
            new_upper[None, None, :],
        )
        - np.maximum(
            old_lower[:, :, None],
            new_lower[None, None, :-1],
        )
    )

    # The first term should use old upper boundaries.
    old_upper = np.concatenate(
        [
            z_old,
            np.full((n_axes, 1), np.inf),
        ],
        axis=1,
    )

    overlap = (
        np.minimum(old_upper[:, :, None], new_upper[None, None, :])
        - np.maximum(old_lower[:, :, None], new_lower[None, None, :-1])
    )
    overlap = np.maximum(overlap, 0.0)

    # Add the final half-space output layer.
    matrices = np.zeros(
        (n_axes, n_old_boundaries + 1, n_new_boundaries + 1),
        dtype=np.result_type(z_old, z_new, float),
    )
    matrices[:, :, :-1] = overlap
    matrices[:, -1, -1] = 1.0

    column_sums = matrices.sum(axis=1, keepdims=True)

    return np.divide(
        matrices,
        column_sums,
        out=np.zeros_like(matrices),
        where=column_sums != 0,
    )

def test_lci(n_mods=100):
    # n_mod
    gpu = torch.device('cpu')
    print('number of threads: ', torch.get_num_threads())
    logForward = logify_forward(Forward)

    t1 = torch.logspace(-6, -4, 15)
    t2 = torch.logspace(-5, -3, 31)
    tx = Loop.square(3)
    rx = Receiver(x=10.)
    
    wv1 = Waveform(torch.tensor([-0.5e-3, -0.9e-3, 0., 0.5e-6]), torch.tensor([0, 1, 1, 0]))
    wv2 = Waveform(torch.tensor([-1e-3, -1.5e-3, 0., 0.5e-5]), torch.tensor([0, 1, 1, 0]))

    fwr1 = Forward(tx, rx, t1, waveform=wv1, tolerance=1e-6, do_setup=False)
    fwr2 = Forward(tx, rx, t2, waveform=wv2, tolerance=1e-6, do_setup=False)
    # fwr2.set_device(gpu)

    fwr = fwr1 + fwr2
    # fwr.set_device(gpu)
    fwr.log_scale=True


    z = torch.cumsum(torch.logspace(0, np.log10(50.), 30), 0)
    z /= z.max()

    z *= 80.

    rho_true = torch.tensor([75., 5., 60.], device=gpu, dtype=rdtype())
    thk_true = torch.tensor([20, 30], device=gpu, dtype=rdtype())

    n_mod_ones = torch.ones(n_mods, device=gpu, dtype=rdtype())
    rho2d = rho_true[:, None]*n_mod_ones[None,:]
    rho2d[1] += torch.linspace(0, 20, n_mods).to(rdtype()).to(gpu)
    thk2d = thk_true[:, None]*n_mod_ones[None,:]



    res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z)
    thick1 = torch.linspace(0, 20, n_mods).to(rdtype()).to(gpu)
    thk2d[1] += thick1

    M = rediscretize_matrices(torch.cumsum(thk2d, 0).cpu().numpy().T, z.cpu().numpy())
    rho2d_true = np.einsum("nmk,nm->nk", M, rho2d.cpu().numpy().T).T[:-1].T
    print(rho2d_true.shape, rho2d.shape)

    dbdts = fwr(rho2d, thk2d)
    
    d_errs = compute_relstd(fwr.times, dbdts.exp().T, current=10.).T


    inv_list = []
    m_trues=[]
    from time import time as timer
    bt = timer()
    
    this_fwr = fwr.copy()
    # this_fwr.set_device(gpu)

    for i,dthick in enumerate(thick1):
        dbdt = dbdts[i].exp()
        d_err = d_errs[i]
        d_obs = dbdt * (1. + d_err*torch.randn_like(dbdt))
        d_err[d_obs<0.] = torch.inf
        
        tmp_inv = Inversion(d_obs.abs().log(), torch.log1p(d_err), res_start.copy(), this_fwr)

        # tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions in {timer()-bt:.2f} seconds")

    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()

    lci = LCI(inv_list)
    lci.horizontal_smoothness=0.1
    lci.vertical_smoothness=1.
    lci.reuse_jac = False

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2


    res_mods= lci.invert(iter_cap=10)
    dois = lci.doi
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        # from plog import Log, Section, PieceWiseLine

        fig,ax = plt.subplots(2,1)
        pcm=ax[0].matshow(lci.mvec.reshape(n_mods, -1).exp().log10().T, aspect='auto', cmap='turbo', vmin=0, vmax=2)
        pcm=ax[1].matshow(np.log10(rho2d_true).T, aspect='auto', cmap='turbo', vmin=0, vmax=2)
        plt.colorbar(pcm, ax=ax, orientation='vertical', label='log10 resistivity')

        plt.show()






if __name__ == '__main__':
    # test_tikhonov_few_layer()
    # test_tikhonov_smooth()
    # test_fastjacobian()
    test_lci()
    #test_sci()
    pass
