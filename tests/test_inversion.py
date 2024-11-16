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
    mtrue = FewLayerModel(torch.tensor([140, 50, 200]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)


    m0 = SmoothModel.starting_model(31)
    

    inv = Inversion(dbdt, dbdt*dbdt_err, m0, fwr)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(smoothness=1e3, iter_cap=15)
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
    tx = Loop.square(40)
    rx = Receiver()
    t = torch.logspace(-5, -3, 31, dtype=torch.double)

    fwr = Forward(tx, rx, t, tolerance=1e-6)
    mtrue = FewLayerModel(torch.tensor([70, 50, 100]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)

    m0 = SmoothModel.starting_model(31)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr, data_transform=torch.log10)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(smoothness=10., iter_cap=7)
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

def test_inversionGM():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import ForwardGeomOp
    tx = Loop.square(40)
    rx = Receiver()
    t = torch.logspace(-5, -2, 31, dtype=torch.double)

    fwr = ForwardGeomOp(tx, rx, t, tolerance=1e-6)
    mtrue = FewLayerModel(torch.tensor([70, 20, 100]), torch.tensor([30, 50]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)

    m0 = SmoothModel.starting_model(31, rho=200, depth_bottom=300)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(smoothness=1000., iter_cap=7)
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

def test_inversion2():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import Forward3
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3, 31, dtype=torch.double)

    fwr = Forward3(tx, rx, t)
    mtrue = FewLayerModel(1./torch.tensor([140, 50, 200]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)

    m0 = SmoothModel.starting_model(31)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr, data_transform=torch.log10)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(smoothness=10., iter_cap=7)
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

def test_inversionGM():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import ForwardGeomOp
    tx = Loop.square(40)
    rx = Receiver()
    t = torch.logspace(-5, -2, 31, dtype=torch.double)

    fwr = ForwardGeomOp(tx, rx, t, tolerance=1e-6)
    mtrue = FewLayerModel(torch.tensor([70, 20, 100]), torch.tensor([30, 50]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)

    m0 = SmoothModel.starting_model(31, rho=200, depth_bottom=300)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err, dbdt*dbdt_err, m0, fwr)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(smoothness=1000., iter_cap=7)
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

def test_inversion2():
    from anemone.experiment import Loop, Receiver, compute_relstd
    from anemone.forwards import Forward3
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3.5, 16, dtype=torch.double)

    fwr = Forward3(tx, rx, t)
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
    from anemone.forwards import Forward3#, ForwardLaplace, ForwardGeomOp
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3, 32, dtype=torch.double)
    

    gpu = torch.device('cpu')
    fwr = Forward3(tx, rx, t.to(gpu), waveform=Waveform.example(), tolerance=1e-6)
    print(fwr.device)
    
    n_layers = 31
    thick = torch.logspace(-1, 1, n_layers-1).to(gpu)


    z = np.arange(n_layers)[:,None]
    x = np.arange(4)[None,:]
    
    dzdx1 = 20 / (x.max()-x.min())
    dzdx2 = 40 / (x.max()-x.min())

    rho = 20*np.ones_like(z+x, dtype=float)
    rho[z>dzdx1*x+10] = 100.
    rho[z>dzdx2*x+20] = 40.


    res = torch.from_numpy(rho)
    x = 30*np.arange(3)[None,:]

    rho = 300*np.ones_like(z+x)
    rho[(0.03*x-0.75*z+10)>0] = 10
    rho[(0.075*x-1.25*z+5)>0] = 40
    # rho[-5:,:] = 300
    
    # rho[(x+0.5*z+25)>=45] = 10.
    # rho[(x+0.5*z>45)*(x+0.5*z+25<45)] = 200.
    
#    rho = 10**np.abs((np.cos(x[None, :])-np.sin(z[:,None]))+2)
    sigma = torch.from_numpy(1./rho).to(gpu)
    thick = torch.logspace(-1, 1, len(z)-1).to(gpu)
    m_true = SmoothModels(res, thick, device=gpu)
    # fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
    if __name__ == '__main__':
        m_true.plot(vmin=1, vmax=2.5, cmap='turbo')
        plt.show()
    rho,tm = m_true.modelize(m_true.vec)
    assert torch.allclose(rho,res), 'something is not right with the model'
    dbdt = fwr(rho,tm)

    n_data = len(dbdt.T.flatten())

    def transform(d, skip_log=False):
        if skip_log:
            return d.reshape((-1,n_data)).squeeze()
        else:
            return torch.log10(torch.abs(d.reshape((-1,n_data)).squeeze()))
    

    dbdt_err = compute_relstd(fwr.times, dbdt)
    dbdt_err5 = compute_relstd(fwr.times, dbdt, relative_noise=0.03)

    m_start = m_true.copy()
    m_start.values[0] = torch.ones_like(m_start.values[0])*50

    dbdt_obs = dbdt + dbdt*dbdt_err*torch.randn(dbdt.shape,device=gpu)

    dbdt_obs = transform(torch.abs(dbdt + dbdt*dbdt_err*torch.randn(dbdt.shape,device=gpu)), skip_log=True)
    dbdt_err5 = transform(dbdt*dbdt_err5, skip_log=True)


    inv = LCI(dbdt_obs, dbdt_err5, m_start, fwr, data_transform=transform)
    inv.data_misfit_acceptance_threshold = 0.9
    print('fwr device is ', fwr.device)
    invm1 = inv.m*1.
    invm1[20:25] *= 1.1
    dbdt_flat0 = inv.mforward(inv.m)
    dbdt_flat1 = inv.mforward(invm1)

    from time import perf_counter
    t0 = perf_counter()
    res_mod = inv.invert(smoothness=1e3, iter_cap=20)
    delta_t = perf_counter()-t0
    
    if __name__ == '__main__':
        print(f'inversion took {delta_t:.2f} seconds.')
        fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
        m_true.plot(ax=axs[0], vmin=1, vmax=2.5, cmap='turbo')
        res_mod.plot(ax=axs[1], vmin=1, vmax=2.5, cmap='turbo')
        plt.show() 
       
    assert inv.misfits[-1] < 1, "misfit is not below one."

def test_inversion_multi_model2():
    import matplotlib.pyplot as plt

    from anemone.experiment import Loop, Receiver, compute_relstd, Waveform
    from anemone.forwards import ForwardLaplace3
    tx = Loop.square(40)
    rx = Receiver()
    t = torch.logspace(-5, -3.5, 32, dtype=torch.double)
    

    gpu = torch.device('cpu')
    fwr = ForwardLaplace3(tx, rx, t.to(gpu), waveform=Waveform.example())
    print(fwr.device)
    
    n_layers = 31
    thick = torch.logspace(-1, 1, n_layers-1).to(gpu)

    n_mod = 6

    z = np.arange(n_layers)[:,None]
    x = (300/n_mod)*np.arange(n_mod)[None,:]

    rho = 300*np.ones_like(z+x)
    rho[(0.03*x-0.75*z+10)>0] = 10
    rho[(0.075*x-1.25*z+5)>0] = 40

    rho = torch.from_numpy(rho.astype(float)).to(gpu)
    thick = torch.logspace(-1, 1, len(z)-1).to(gpu)
    m_true = SmoothModels(rho, thick, device=gpu)

    rm,tm = m_true.modelize(m_true.vec)
    assert torch.allclose(rm, rho), 'something is not right with the model'
    dbdt = fwr(rm,tm)


    print('shape of data is :', dbdt.shape)
    # should be >>> shape of data is : torch.Size([10, 16])
    plt.figure()
    plt.semilogy(dbdt.cpu())
    plt.show()



    n_data = len(dbdt.flatten())

    def transform(d, skip_log=False):
        if skip_log:
            return torch.abs(d.reshape((-1,n_data)).squeeze())
        else:
            return torch.log10(torch.abs(d.reshape((-1,n_data)).squeeze()))
    

    dbdt_err = compute_relstd(fwr.times, dbdt)
    dbdt_err5 = compute_relstd(fwr.times, dbdt)

    m_start = m_true.copy()
    m_start.values[0] = torch.ones_like(m_start.values[0])*torch.mean(m_start.values[0])

    dbdt_obs = transform(torch.abs(dbdt + dbdt*dbdt_err*torch.randn(dbdt.shape,device=gpu)), skip_log=True)

    dbdt_err5 = transform(dbdt*dbdt_err5, skip_log=True)


    inv = LCI(dbdt_obs, dbdt_err5, m_start, fwr, data_transform=transform)
    inv.data_misfit_acceptance_threshold = 0.9
    print('fwr device is ', fwr.device)
    invm1 = inv.m*1.
    invm1[20:25] *= 1.1

    from time import perf_counter
    t0 = perf_counter()
    res_mod = inv.invert(smoothness=2e3, iter_cap=20)
    delta_t = perf_counter()-t0
    
    if __name__ == '__main__':
        print(f'inversion took {delta_t:.2f} seconds.')
        fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
        m_true.plot(ax=axs[0], vmin=1, vmax=2.5, cmap='turbo')
        res_mod.plot(ax=axs[1], vmin=1, vmax=2.5, cmap='turbo')
        plt.show() 
       
    assert inv.misfits[-1] < 1, "misfit is not below one."


if __name__ == '__main__':
    # test_model_smoothing_constraint()
    
    test_inversion_multi_model2()

    # test_inversion()
    # test_inversion()
    # test_inversionGM()
    # test_inversion_multi_model2()

