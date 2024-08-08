from em1d.inversion import *

def test_default_model_class():
    exp = EarthModel([torch.ones(1), torch.ones(1)], ['amp', 'rate'])
    assert torch.all(exp.vec == torch.ones(2)), "model vector does not have the right shape or values"


def test_inversion_noise_free():
    from em1d.experiment import Forward, Loop, Receiver, compute_relstd
    tx = Loop.square(25)
    rx = Receiver()
    t = torch.logspace(-5, -3, 16, dtype=torch.double)

    fwr = Forward(tx, rx, t)
    mtrue = FewLayerModel(1./torch.tensor([140, 50, 200]), torch.tensor([20, 40]))

    dbdt = fwr(*mtrue.values)
    dbdt_err = compute_relstd(t, dbdt, moment=tx.area)


    m0 = FewLayerModel.starting_model(3)
    

    inv = Inversion(dbdt+dbdt*torch.randn(len(dbdt))*dbdt_err/2, dbdt*dbdt_err, m0, fwr)
    dbdt_start = inv.mforward(m0.vec)
    resmod = inv.invert(iter_cap=6)
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

    

if __name__ == '__main__':
    test_inversion_noise_free()