import torch
from anemone.experiment import *
from anemone import rdtype, cdtype

from pytest_config import skip_multi_rx

import pytest


is_main = __name__ == '__main__'

def aarhusInv_stepoff_comparison(ForwardMaker, **kwargs):
    from anemone.experiment import Loop, Receiver
    name = ForwardMaker.__name__
    rtol = kwargs.pop('rtol', 1e-2)

    ainv_dbdt=torch.tensor([2.749861e-04, 1.753167e-04, 1.011688e-04, 6.031561e-05,
        3.650625e-05, 2.059544e-05, 1.194428e-05, 7.079678e-06,
        3.957627e-06, 2.293402e-06, 1.403765e-06, 8.236817e-07,
        4.389777e-07, 2.446409e-07, 1.422868e-07, 7.752075e-08,
        4.302391e-08, 2.479088e-08, 1.362022e-08, 7.621867e-09]).to(rdtype)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    lp = Loop.square(40)
    rx = Receiver()

    rho = 40*torch.ones(3)
    thick = torch.tensor([10, 10])

    fwr = ForwardMaker(lp, rx, times, **kwargs)
    dbdt = fwr(rho, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle(f'Test {name} against AarhusInv 40x40 m^2 loop with stepoff waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=rtol), f"{name} and aarhusInv step-off do not match within {rtol*100} percent"

def aarhusInv_piecewise_comparison(ForwardMaker, **kwargs):
    from anemone.experiment import Loop, Receiver, Waveform
    name = ForwardMaker.__name__
    
    rtol = kwargs.pop('rtol', 2e-2)

    ainv_dbdt = torch.tensor([3.853627e-04, 2.309149e-04, 1.255715e-04, 7.168451e-05,
        4.196206e-05, 2.297534e-05, 1.302996e-05, 7.591422e-06,
        4.179841e-06, 2.394241e-06, 1.452859e-06, 8.455657e-07,
        4.466630e-07, 2.468966e-07, 1.423754e-07, 7.665493e-08,
        4.189503e-08, 2.367454e-08, 1.263738e-08, 6.809730e-09]).to(rdtype)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05, 3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06]).to(rdtype)
    wv = torch.tensor([0, 1, 1, 0]).to(rdtype)
    waveform = Waveform(t, wv)


    lp = Loop.square(40)
    rx = Receiver()

    rho = torch.tensor([40,40])
    thick = torch.tensor([40])

    kwargs['waveform'] = waveform
    fwr = ForwardMaker(lp, rx, times, **kwargs)
    dbdt = fwr(rho, thick)
    dbdt = fwr(rho, thick)
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='pytorch forward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].set_ylabel('dB/dt')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=rtol), f"{name} and aarhusInv piece-wise waveform do not match within {rtol*100} percent"



def check_halfspace_circle_center(ForwardMaker, rtol=3e-2):
    from anemone.experiment import Loop, Receiver
    from anemone.physics import hzdt_halfspace_center_circle_loop, dbdt_center_of_circular_loop_halfspace
    from anemone import constants as c
    name = ForwardMaker.__name__
    radius = 1.
    lp = Loop.circle(radius, n=32)
    # lp = Loop.square((c.pi*radius*radius)**(1/2))
    rx = Receiver()
    times = torch.logspace(-8, 0, 101, dtype=rdtype)

    rho = torch.logspace(0,3, 101, dtype=rdtype)[None, :]*torch.ones(2, dtype=rdtype)[:, None]
    thick = torch.ones(1, dtype=rdtype)*20

    fwr = ForwardMaker(lp, rx, times, tolerance=1e-6)
    # dbdt = fwr(rho, thick)
    dbdt = fwr(rho, thick).real.T
    dbdt_analytic = dbdt_center_of_circular_loop_halfspace(times[:, None].to(torch.double), rho[None, 0, :].to(torch.double), radius).real.to(rdtype)
    
    if __name__ == '__main__':
        
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.numpy(), '.')
        ax[0].loglog(times,      dbdt_analytic.numpy(), 'k--')
        ax[0].set_ylabel('dB/dt')
        ax[0].set_aspect('equal')

        ax[1].loglog(times, 100*torch.abs((dbdt-dbdt_analytic)/dbdt_analytic))
        ax[1].set_ylabel('relative difference [%]')
        # ax[1].set_ylim(None, 1e3)
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('check halfspace circle center:'+f'\n{name} vs analytic response (radius = {radius} m)')
        plt.show()
    
    assert torch.allclose(dbdt_analytic[:,1:], dbdt[:,1:], 0.1), f"{name} (radius = {radius} m) and analytic do not match within {rtol*100} percent"
    assert torch.allclose(dbdt_analytic[times<5e-2,1:], dbdt[times<5e-2,1:], rtol), f"{name} (radius = {radius} m) and analytic before 50ms do not match within {rtol*100} percent"

def test_halfspace_circle_center_forward():
    from anemone.forwards import Forward
    check_halfspace_circle_center(Forward)  

def test_halfspace_circle_center():
    from anemone.experiment import Loop, Receiver
    from anemone.physics import hzdt_halfspace_center_circle_loop, dbdt_center_of_circular_loop_halfspace
    from anemone.forwards import Forward
    from anemone import constants as c
    radius = 1.

    name = 'Central Circular Loop (radius={radius} m)'.format(radius=radius)
    rtol = 2e-2
    
    
    
    times = torch.logspace(-9, 0, 51, dtype=rdtype)

    rho = torch.tensor([[1, 10, 100, 1000], [1, 10, 100, 1000]])
    thick = torch.ones(4, dtype=rdtype)*20

    #fwr = ForwardCentralCircle(radius, times)
    lp = Loop.circle(radius, n=32)
    rx = Receiver()
    fwr = Forward(lp, rx, times, tolerance=1e-6)
    dbdt = fwr(rho, thick).real
    

    dbdt_analytic = dbdt_center_of_circular_loop_halfspace(times[:, None].to(torch.double), rho[None, 0, :].to(torch.double), radius).to(rdtype).real

    


    
    if __name__ == '__main__':
        relerr = torch.nan_to_num(100*torch.abs(dbdt.T-dbdt_analytic)/torch.abs(dbdt_analytic))    

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times, dbdt.T, '.')
        
        ax[0].set_ylabel('dB/dt [V / m$^2$]')
        ax[0].set_aspect('equal')
        ax[0].legend([1, 10, 100, 1000], title='$\\rho$ [$\\Omega$m]', fontsize='small')
        ax[0].loglog(times,      dbdt_analytic, 'k--')

        ax[1].semilogx(times, relerr)
        ax[1].legend([1, 10, 100, 1000], title='$\\rho$ [$\\Omega$m]', fontsize='small')
        ax[1].set_ylabel('relative error [%]')
        ax[1].set_ylim(0, 10)

        ax[1].text(times[0], 8, f'$\\uparrow$ {relerr[0,-1]:0.1f}%\nfirst gate', fontsize='small',  bbox=dict(facecolor='white', edgecolor='black'))
        ax[0].set_xticks([1e-6, 1e-4, 1e-2, 1e0])
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('1: '+name)
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.03,
            right=0.97,
            hspace=0.2,
            wspace=0.0
        )
        plt.show()

    dbdt = fwr(torch.tensor([100., 100.]), torch.tensor([20.])).squeeze().real
    dbdt_analytic = dbdt_center_of_circular_loop_halfspace(times[:, None].to(torch.double), torch.tensor([100.]).to(torch.double), radius).to(rdtype).squeeze().real
    
    if __name__ == '__main__':
        relerr = torch.nan_to_num(100*torch.abs(dbdt.T-dbdt_analytic)/torch.abs(dbdt_analytic))    

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times, dbdt.T, '.')
        
        ax[0].set_ylabel('dB/dt [V / m$^2$]')
        ax[0].set_aspect('equal')
        ax[0].loglog(times,      dbdt_analytic, 'k--')

        ax[1].semilogx(times, relerr)
        ax[1].set_ylabel('relative error [%]')
        ax[1].set_ylim(0, 10)

        # ax[1].text(times[0], 8, f'$\\uparrow$ {relerr[0,-1]:0.1f}%\nfirst gate', fontsize='small',  bbox=dict(facecolor='white', edgecolor='black'))
        ax[0].set_xticks([1e-6, 1e-4, 1e-2, 1e0])
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('2: '+name)
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.03,
            right=0.97,
            hspace=0.2,
            wspace=0.0
        )
        plt.show()

    assert torch.allclose(dbdt_analytic[times<1e-3], dbdt[times<1e-3], rtol=rtol), f"{name} and analytic do not match within {rtol*100} percent"
    rtol = 1e-1
    assert torch.allclose(dbdt_analytic, dbdt, rtol=rtol), f"{name} and analytic do not match within {rtol*100} percent"


def compare_two_Forwards(Forward1, Forward2, rtol=1e-2):
    from anemone.experiment import Loop, Receiver
    name1 = Forward1.__name__
    name2 = Forward2.__name__

    lp = Loop.square(40)
    rx = Receiver()
    times = torch.logspace(-7, 0, 101, dtype=rdtype)
    fwr1 = Forward1(lp, rx, times)


    rho = torch.logspace(0, 3, 5, dtype=rdtype)[None, :]*torch.ones(2, dtype=rdtype)[:, None]
    thick = torch.ones(1, dtype=rdtype)*20

    lp = Loop.square(40)
    rx = Receiver()
    times = torch.logspace(-7, 0, 101, dtype=rdtype)
    fwr2 = Forward2(lp, rx, times)


    dbdt1 = fwr1(rho*1., thick*1.)
    dbdt2 = fwr2(rho*1., thick*1.)
    
    if __name__ == '__main__':
        
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr1.times.numpy(), dbdt1.real.T, '.')
        ax[0].loglog(fwr2.times.numpy(), dbdt2.real.T, 'k--')
        ax[0].set_ylabel('dB/dt')
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, torch.abs(dbdt1.real.T)/torch.abs(dbdt2.real.T))
        ax[1].set_ylabel('relative difference [%]')
        ax[1].set_ylim(0.1, 10)
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle(f'{name1} vs {name2}')
        plt.show()


def check_parallel_models(ForwardMaker):
    name = ForwardMaker.__name__
    from anemone.experiment import Loop, Receiver, Waveform
    lp = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    times = torch.logspace(-5, -3, 31, dtype=rdtype)

    rho = torch.ones(3)*40.
    rho_change = torch.linspace(-30, 260, 94)
    rho = rho[:, None] + rho_change[None, :]
    rho[0, :] = 40.
    rho[2, :] = 40.

    thick = np.ones_like(rho)[:-1, :]*20

    fwr = ForwardMaker(lp, rx, times, waveform=waveform)
    dbdt = fwr(rho, thick)
    
    if __name__ == '__main__':
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(dbdt)
        plt.show()

    assert dbdt.shape == (94, 31), f"{name} running parallel models does not return correct shape"


def check_forward_speeds(ForwardMaker, **kwargs):
    name = ForwardMaker.__name__

    rho = torch.ones(3)*40.
    rho_change = torch.linspace(-30, 260, 100)
    rho = rho[:, None] + rho_change[None, :]
    rho[0, :] = 40.
    rho[2, :] = 40.

    thick = np.ones_like(rho)[:-1, :]*20

    fwr = ForwardMaker.example(**kwargs)
    dbdt = fwr(torch.ones(3)*40, torch.ones(2)*20)


    from time import perf_counter
    t0 = perf_counter()
    _ = fwr(rho, thick)
    t1 = perf_counter()
    dt = t1-t0
    print(f'{name} {len(rho_change)} forward calculation took {dt:0.3f} s')

def test_forward_speeds():
    from anemone.forwards import Forward
    check_forward_speeds(Forward)

def test_forward_parallel_models():
    from anemone.forwards import Forward
    check_parallel_models(Forward)

def test_40x40_aarhusInv_stepoff():
    from anemone.forwards import Forward
    aarhusInv_stepoff_comparison(Forward)


def test_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import Forward
    aarhusInv_piecewise_comparison(Forward) 


def test_Laplace_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import ForwardLaplace
    aarhusInv_piecewise_comparison(ForwardLaplace)

def check_colecoleforwardTD(ForwardMaker):
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=rdtype)
    pyfwr = ForwardMaker.colecole(sq, rx, t0, waveform=waveform, filterfunc=recfilter)

    res  = torch.tensor([4.000e+02, 4.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 8.000e+02, 
                        8.000e+02, 8.000e+02, 8.000e+02, 8.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 
                        5.000e+02, 3.000e+02, 3.000e+02, 3.000e+02], dtype=rdtype)
    m0   = torch.tensor([2.000e+02, 2.000e+02, 2.000e+02, 2.000e+02, 4.000e+02, 4.000e+02, 4.000e+02, 5.000e+02, 
                        5.000e+02, 5.000e+02, 5.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 
                        9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02], dtype=rdtype)*1e-3
    tau  = torch.tensor([1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 2.000e-04, 2.000e-04, 
                        2.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 
                        1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04], dtype=rdtype)
    C  = torch.tensor([6.000e-01, 6.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 6.000e-01, 6.000e-01,
                         6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 9.000e-01, 
                         9.000e-01, 9.000e-01, 9.000e-01, 9.000e-01], dtype=rdtype)
    thick=torch.tensor([1.000e+00, 1.177e+00, 1.386e+00, 1.632e+00, 1.922e+00, 2.263e+00, 2.665e+00, 3.138e+00, 
                        3.695e+00, 4.351e+00, 5.124e+00, 6.034e+00, 7.105e+00, 8.366e+00, 9.851e+00, 1.160e+01, 
                        1.365e+01, 1.608e+01, 1.893e+01], dtype=rdtype)

    ainv_dbdt = torch.tensor([ 4.163592e-05,  2.674770e-05,  1.696241e-05,  1.037700e-05,
        6.115854e-06,  3.470203e-06,  1.870507e-06,  9.252422e-07,
        3.775450e-07,  7.785586e-08, -7.586203e-08, -1.447992e-07,
       -1.685453e-07, -1.687759e-07, -1.544787e-07, -1.313316e-07,
       -1.046215e-07, -7.896327e-08, -5.621327e-08, -3.828716e-08,
       -2.530188e-08, -1.641016e-08, -1.056120e-08, -6.790221e-09,
       -4.377552e-09, -2.834261e-09, -1.834226e-09, -1.188674e-09,
       -7.686370e-10, -4.945201e-10, -3.173105e-10], dtype=rdtype)
    
    dbdt = pyfwr(res,  m0, tau, C, thick)
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(t0, ainv_dbdt, 'k-', label='AarhusInv')
        ax[0].loglog(t0, -ainv_dbdt, 'k--')
        ax[0].loglog(pyfwr.times.numpy(), dbdt.real, 'C0.', label='Cole Cole Forward')
        ax[0].loglog(pyfwr.times.numpy(), -dbdt.real, 'r.')

        ax[0].set_ylabel('dB/dt')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(t0, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt, dbdt, rtol=1e-2), "aarhus inv and pytorch forward do not match within 1 percent"

def check_MPAforwardTD(ForwardMaker):
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    from anemone.physics import mpa2colecole

    # sq = Loop.square(2.9, z=0.93)
    # rx = Receiver(x=-9., z=0.43)
    sq = Loop.square(40)
    rx = Receiver()

    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])

    ainv_dat = torch.tensor([[9.350E-06,  1.745393E-04],
                            [1.145E-05,  1.303665E-04],
                            [1.458E-05,  9.897802E-05],
                            [1.822E-05,  7.424845E-05],
                            [2.255E-05,  5.230275E-05],
                            [2.867E-05,  3.168347E-05],
                            [3.595E-05,  1.788651E-05],
                            [4.460E-05,  9.527986E-06],
                            [5.660E-05,  4.206293E-06],
                            [7.070E-05,  1.706046E-06],
                            [8.630E-05,  5.354002E-07],
                            [1.071E-04, -5.975600E-08],
                            [1.381E-04, -2.894035E-07],
                            [1.748E-04, -3.310659E-07],
                            [2.174E-04, -2.693721E-07],
                            [2.775E-04, -1.754609E-07],
                            [3.515E-04, -1.040936E-07],
                            [4.385E-04, -6.107881E-08],
                            [5.575E-04, -3.306694E-08],
                            [7.035E-04, -1.822600E-08]], dtype=torch.float64)

    

    ainv_dbdt = ainv_dat[:, 1].clone()
    t0 = ainv_dat[:, 0].clone()
    not_zero_xing = torch.ones_like(ainv_dat[:, 0], dtype=torch.bool)
    not_zero_xing[torch.searchsorted(t0, torch.tensor([1.071e-4]))-1] = False
    

    pyfwr = ForwardMaker.mpa(sq, rx, t0, waveform=waveform, filterfunc=recfilter)

    rho=  torch.tensor([5.000e+02,5.000e+02,5.000e+02,8.000e+02,8.000e+02,5.000e+02,5.000e+02,5.000e+02,5.000e+02,2.000e+02,2.000e+02,5.000e+02,5.000e+02,5.000e+02,5.000e+02,5.000e+02,5.000e+02,8.000e+02,8.000e+02,5.000e+02])
    phi = torch.tensor([2.000e+02,2.000e+02,2.000e+02,2.000e+02,2.000e+02,5.000e+02,5.000e+02,5.000e+02,5.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02,8.000e+02])/1e3
    tau = torch.tensor([1.000e-04,1.000e-04,1.000e-04,2.000e-04,2.000e-04,1.000e-04,1.000e-04,9.000e-04,9.000e-04,9.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04,1.000e-04])
    C =   torch.tensor([6.000e-01,6.000e-01,4.000e-01,4.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01,3.000e-01,3.000e-01,3.000e-01,3.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01,6.000e-01])
    thick=torch.tensor([1.000e+00, 1.177e+00, 1.386e+00, 1.632e+00, 1.922e+00, 2.263e+00, 2.665e+00, 3.138e+00, 3.695e+00, 4.351e+00, 5.124e+00, 6.034e+00, 7.105e+00, 8.366e+00, 9.851e+00, 1.160e+01, 1.365e+01, 1.608e+01, 1.893e+01])

    # rho,m_cc,tau_cc,C = mpa2colecole(rho, phi, tau, C)


    dbdt = pyfwr(rho, phi,tau, C, thick)



    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(t0, ainv_dbdt, 'k-', label='AarhusInv')
        ax[0].loglog(t0, -ainv_dbdt, 'k--')
        ax[0].loglog(pyfwr.times.numpy(),  dbdt.real, 'C0.', label='IPforward')
        ax[0].loglog(pyfwr.times.numpy(), -dbdt.real, 'r.')

        ax[0].set_ylabel('dB/dt')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(t0, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')
        for ax_ in ax:
            ax_.set_xlabel('time [s]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt[not_zero_xing], dbdt[not_zero_xing], rtol=3e-2), "aarhus inv and pytorch forward do not match within 3 percent"

def test_colecole_forwardTD():
    from anemone.forwards import Forward
    check_colecoleforwardTD(Forward)

def test_mpaforwardTD():
    from anemone.forwards import Forward
    check_MPAforwardTD(Forward)

def check_colecole_parallel_models(ForwardMaker):
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    name = ForwardMaker.__name__

    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=rdtype)
    pyfwr = ForwardMaker.colecole(sq, rx, t0, waveform=waveform, filterfunc=recfilter)

    rho  = torch.tensor([4.0e+02, 4.0e+02, 5.0e+02], dtype=rdtype)
    m0   = torch.tensor([2.0e+02, 9.0e+02, 5.0e+02], dtype=rdtype)*1e-3
    tau  = torch.tensor([1.0e-04, 2.0e-04, 3.0e-04], dtype=rdtype)
    chg  = torch.tensor([6.0e-01, 5.0e-01, 9.0e-01], dtype=rdtype)
    thick=torch.tensor([20, 30], dtype=rdtype)

    sweep = torch.linspace(0.9, 1.1, 94)
    fixed = torch.ones(94)[None, :]

    
    rho = rho[:, None] * fixed
    rho[1] = rho[1]*sweep

    m0 = m0[:, None] * fixed
    tau = tau[:, None] * fixed
    chg = chg[:, None] * fixed
    thick = thick[:, None] * fixed
    
    dbdt = pyfwr(rho, m0, tau, chg, thick)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)
        ax.loglog(t0,  dbdt.T, '-')
        ax.loglog(t0, -dbdt.T, '--')

        ax.set_ylabel('dB/dt')
        ax.legend()
        ax.set_aspect('equal')
        ax.set_xlabel('time [s]')

        plt.show()
    assert dbdt.shape == (94, 31), f"{name} running parallel models does not return correct shape"


def test_colecole_parallel_models():
    from anemone.forwards import Forward
    check_colecole_parallel_models(Forward)

def test_forward_timing():
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    from anemone.forwards import Forward
    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=rdtype)
    fwr = Forward(sq, rx, t0, waveform=waveform, filterfunc=recfilter, tolerance=1e-6)    

    res  = torch.tensor([40, 200, 180])
    thk  = torch.tensor([20, 40])

    dbdt = fwr(res, thk);
    dbdt = fwr(res, thk)

    from time import perf_counter
    t0 = perf_counter()
    dbdt = fwr(res, thk)
    t1 = perf_counter()
    dt1 = t1-t0
    assert dt1 < 0.2, "forward calculation took too long"

    res = res[:, None] + 10*torch.arange(10)[None, :]
    dbdt = fwr(res, thk)
    t2 = perf_counter()
    dbdt = fwr(res, thk)
    t3 = perf_counter()
    dt2 = (t3-t2)/10
    print('time for 10 forward calculations: ', dt2*10)
    assert dt2 < 0.2, "forward calculation took too long"


def multi_rx_forward_shape_check(ForwardMaker):
    name = ForwardMaker.__name__

    from anemone.experiment import Loop, Receiver, Waveform
    lp = Loop.square(50)
    rx = Receiver(x=torch.linspace(-20, 20, 5)*0.9)
    waveform = Waveform.example()
    times = torch.logspace(-5, -3, 31, dtype=rdtype)

    rho = torch.tensor([40, 40])
    thick = torch.tensor([10])

    fwr = ForwardMaker(lp, rx, times, waveform=waveform)
    dbdt = fwr(rho, thick)
    assert dbdt.shape == (5, 31), f"{name} multireceiver does not return correct shape {dbdt.shape} != (5, 31)"

@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multirx_forward_shape_check():
    from anemone.forwards import Forward
    multi_rx_forward_shape_check(Forward)

@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multireceiver_forward():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import Forward
    import torch


    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    lp = Loop.square(40)
    rx = Receiver(x=torch.linspace(-400, 400, 202), z=10)

    rho = torch.tensor([40, 80])
    thick = torch.tensor([10])

    fwr = Forward(lp, rx, times)
    dbdt = fwr(rho, thick)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.semilogy(rx.x, dbdt.real, 'C0.-', label='mforward')
        ax.semilogy(rx.x, -dbdt.real, 'r.-', label='mforward')
        plt.show()
    assert (rx.x[:, None]+times[None, :]).shape == dbdt.shape, "dbdt shape is not correct"

@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multireceiver_forward_multi_height():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import Forward
    import torch


    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    lp = Loop.square(40)
    rx = Receiver(x=torch.linspace(-400, 400, 202), z=torch.abs(torch.linspace(-400, 400, 202)))

    rho = 40
    sigma = 1/torch.tensor([rho,2*rho])
    thick = torch.tensor([10])

    fwr = Forward(lp, rx, times)
    dbdt = fwr(sigma, thick)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.semilogy(rx.x, dbdt.real, 'C0.-', label='mforward')
        ax.semilogy(rx.x, -dbdt.real, 'r.-', label='mforward')
        plt.show()
    assert (rx.x[:, None]+times[None, :]).shape == dbdt.shape, "dbdt shape is not correct"

@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multireceiver_frequency_domain():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import ForwardFD
    import torch

    lp = Loop.square(50)
    rx = Receiver(x=torch.linspace(-200, 200, 202), y=torch.linspace(-200, 200, 202))
    freqs = torch.tensor([2000], dtype=rdtype)

    rho = torch.ones(2)*40
    thick = torch.tensor([10])

    fwr = ForwardFD(lp, rx, freqs)
    from time import time as timer
    t0 = timer()
    B = fwr(rho, thick)
    print('elapsed time: ', timer()-t0)
    
    t0 = timer()
    B = fwr(rho, thick)#.reshape((202,202))
    print('elapsed time: ', timer()-t0)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.pcolormesh(rx.x,rx.y, torch.log10(torch.abs(B)).reshape((202,202)))
        plt.show()
    assert (rx.x[:, None]+freqs[None, :]).shape == tuple(B.shape), f"dbdt shape is not correct: {B.shape} != {(rx.X[..., None]+freqs[None, :]).shape}"



if __name__ == '__main__':
    
    # test_halfspace_circle_center_forward()
    # test_halfspace_circle_center()
    # test_Laplace_40x40_aarhusInv_piecewise_waveform()
    # test_40x40_aarhusInv_piecewise_waveform()
    # test_forward_parallel_models()
    # test_colecole_forwardTD()
    test_mpaforwardTD()

    #test_multirx_forward_shape_check()
    # test_multireceiver_frequency_domain()
    # test_Laplace_40x40_aarhusInv_piecewise_waveform()
    pass