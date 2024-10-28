from anemone.inversion import *

def test_default_model_class():
    exp = EarthModel([torch.ones(1), torch.ones(1)], ['amp', 'rate'])
    assert torch.all(exp.vec == torch.ones(2)), "model vector does not have the right shape or values"

def test_multi_model_setup():
    cmod = SmoothModels.starting_model(30, 50)
    cmod.values[0][5:15,:] = torch.linspace(0.1, 10, 10)[:, None]
    
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        cmod.plot()
        plt.show()
    assert cmod.n_layers == 30, "number of layers doesn't match for multi-model setup."
    assert cmod.n_models == 50, "number of layers doesn't match for multi-model setup."

def test_model_smoothing_constraint():
    cmod = SmoothModel.starting_model(11)
    cmod = FewLayerModel.starting_model(4)
    cmod = SmoothModels.starting_model(11, 3)
    print(cmod.values[0].shape)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots()
        axs.pcolor(cmod.H())
        plt.show()



def test_inversion_noise_free():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import Forward
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3.5, 16, dtype=torch.double)

    fwr = Forward(tx, rx, t)
    mtrue = FewLayerModel(1./torch.tensor([140, 50, 200]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)


    m0 = FewLayerModel.starting_model(3)
    

    inv = Inversion(dbdt, dbdt*dbdt_err, m0, fwr)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(iter_cap=15)
    dbdt_pred = inv.mforward(resmod.vec)
    



    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2)
        mtrue.plot(ax=axs[0], color='C0')
        resmod.plot(ax=axs[0], color='C1')

        m0.plot(ax=axs[0], linestyle=':', color='k')


        axs[1].loglog(t.numpy(), inv.data_observed.numpy())
        axs[1].loglog(t.numpy(), dbdt_pred.numpy())
        axs[1].loglog(t.numpy(), dbdt_start.numpy(), 'k:')        

        plt.show()
    assert inv.misfits[-1] < 1, "misfit is not below one."
        
def test_inversion():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import Forward
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3.5, 16, dtype=torch.double)

    fwr = Forward(tx, rx, t)
    mtrue = FewLayerModel(1./torch.tensor([140, 50, 200]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)


    m0 = FewLayerModel.starting_model(3)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr, data_transform=torch.log10)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(iter_cap=7)
    dbdt_pred = inv.mforward(resmod.vec)



    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,2)
        mtrue.plot(ax=axs[0], color='C0')
        resmod.plot(ax=axs[0], color='C1')

        m0.plot(ax=axs[0], linestyle=':', color='k')


        axs[1].loglog(t.numpy(), 10**inv.data_observed.numpy())
        axs[1].loglog(t.numpy(), 10**dbdt_pred.numpy())
        axs[1].loglog(t.numpy(), 10**dbdt_start.numpy(), 'k:')        

        plt.show()    
    assert inv.misfits[-1] < 1, "misfit is not below one."

def test_inversion_multi_model():
    import matplotlib.pyplot as plt

    from anemone.experiment import Loop, Receiver, compute_relstd, Waveform
    from anemone.forwards import Forward, ForwardLaplace2
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3.5, 16, dtype=torch.double)
    

    gpu = torch.device('cuda')
    fwr = ForwardLaplace2(tx, rx, t.to(gpu), waveform=Waveform.example(), tolerance=1e-5)
    print(fwr.device)
    
    n_layers = 31
    thick = torch.logspace(-1, 1, n_layers-1).to(gpu)


    z = np.arange(n_layers)[:,None]
    x = 30*np.arange(10)[None,:]

    rho = 300*np.ones_like(z+x)
    rho[(0.03*x-0.75*z+10)>0] = 10
    rho[(0.075*x-1.25*z+5)>0] = 40
    # rho[-5:,:] = 300
    
    # rho[(x+0.5*z+25)>=45] = 10.
    # rho[(x+0.5*z>45)*(x+0.5*z+25<45)] = 200.
    
#    rho = 10**np.abs((np.cos(x[None, :])-np.sin(z[:,None]))+2)
    sigma = torch.from_numpy(1./rho).to(gpu)
    thick = torch.logspace(-1, 1, len(z)-1).to(gpu)
    m_true = SmoothModels(sigma, thick, device=gpu)
    # fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
    # if __name__ == '__main__':
    #     m_true.plot(vmin=1, vmax=2.5, cmap='turbo')
    #     plt.show()
    sm,tm = m_true.modelize(m_true.vec)
    assert torch.allclose(sm,sigma), 'something is not right with the model'
    dbdt = fwr(sm,tm)


    n_data = len(dbdt.T.flatten())

    def transform(d, skip_log=False):
        if skip_log:
            return d.reshape((-1,n_data)).squeeze()
        else:
            return torch.log10(torch.abs(d.reshape((-1,n_data)).squeeze()))
    

    dbdt_err = compute_relstd(fwr.times, dbdt)
    dbdt_err5 = compute_relstd(fwr.times, dbdt, relative_noise=0.05)

    m_start = m_true.copy()
    m_start.values[0] = torch.ones_like(m_start.values[0])*0.02

    dbdt_obs = transform(torch.abs(dbdt + dbdt*dbdt_err*torch.randn(dbdt.shape,device=gpu)), skip_log=True)
    dbdt_err5 = transform(dbdt*dbdt_err5, skip_log=True)


    inv = LCI(dbdt_obs, dbdt_err5, m_start, fwr, data_transform=transform)
    inv.data_misfit_acceptance_threshold = 0.9
    print('fwr device is ', fwr.device)
    invm1 = inv.m*1.
    invm1[20:25] *= 2.
    dbdt_flat0 = inv.mforward(inv.m)
    dbdt_flat1 = inv.mforward(invm1)

    # plt.figure()
    # plt.plot(dbdt_flat0.numpy())
    # plt.plot(dbdt_flat1.numpy())
    # plt.show()
    from time import perf_counter
    t0 = perf_counter()
    res_mod = inv.invert(smoothness=1e2, iter_cap=20)
    delta_t = perf_counter()-t0
    
    if __name__ == '__main__':
        print(f'inversion took {delta_t:.2f} seconds.')
        fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
        m_true.plot(ax=axs[0], vmin=1, vmax=2.5, cmap='turbo')
        res_mod.plot(ax=axs[1], vmin=1, vmax=2.5, cmap='turbo')
        plt.show() 
    # mtrue = FewLayerModel(1./torch.tensor([140, 50, 200]), torch.tensor([20, 40]))
    #dpred = fwr(*res_mod.modelize(res_mod.vec))
    # plt.figure()
    # plt.loglog(fwr.times.cpu().numpy(), np.abs(dbdt_obs.cpu().numpy()).T, '|-')
    # plt.loglog(fwr.times.cpu().numpy(), np.abs(dpred.cpu().numpy()).T)
    # plt.show()

    # dbdt = fwr(*mtrue.values)
    # dbdt_err = compute_relstd(t, dbdt, moment=tx.area)


    # m0 = FewLayerModel.starting_model(3)
    

    # inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr)
    # dbdt_start = inv.mforward(m0.vec)
    # resmod = inv.invert(iter_cap=10)
    # dbdt_pred = inv.mforward(resmod.vec)



    # if __name__ == '__main__':
    #     import matplotlib.pyplot as plt
    #     fig,axs = plt.subplots(1,2)
    #     mtrue.plot(ax=axs[0], color='C0')
    #     resmod.plot(ax=axs[0], color='C1')

    #     m0.plot(ax=axs[0], linestyle=':', color='k')


    #     axs[1].loglog(t.numpy(), inv.data_observed.numpy())
    #     axs[1].loglog(t.numpy(), dbdt_pred.numpy())
    #     axs[1].loglog(t.numpy(), dbdt_start.numpy(), 'k:')        
       
    assert inv.misfits[-1] < 1, "misfit is not below one."

if __name__ == '__main__':
    # test_model_smoothing_constraint()
    test_inversion_multi_model()
    # test_inversion()
    # test_inversion_noise_free()
