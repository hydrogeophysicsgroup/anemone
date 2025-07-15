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


if __name__ == '__main__':
    # test_tikhonov_few_layer()
    # test_tikhonov_smooth()
    test_fastjacobian()