import torch
from anemone.experiment import *


is_main = __name__ == '__main__'

def aarhusInv_stepoff_comparison(ForwardMaker, **kwargs):
    from anemone.experiment import Loop, Receiver
    name = ForwardMaker.__name__
    rtol = kwargs.pop('rtol', 1e-2)

    ainv_dbdt=torch.tensor([2.749861e-04, 1.753167e-04, 1.011688e-04, 6.031561e-05,
        3.650625e-05, 2.059544e-05, 1.194428e-05, 7.079678e-06,
        3.957627e-06, 2.293402e-06, 1.403765e-06, 8.236817e-07,
        4.389777e-07, 2.446409e-07, 1.422868e-07, 7.752075e-08,
        4.302391e-08, 2.479088e-08, 1.362022e-08, 7.621867e-09], dtype=torch.complex128)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    lp = Loop.square(40)
    rx = Receiver()

    rho = 40*torch.ones(2)
    thick = torch.tensor([10])

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
    
    rtol = kwargs.pop('rtol', 1e-2)

    ainv_dbdt = torch.tensor([3.853627e-04, 2.309149e-04, 1.255715e-04, 7.168451e-05,
        4.196206e-05, 2.297534e-05, 1.302996e-05, 7.591422e-06,
        4.179841e-06, 2.394241e-06, 1.452859e-06, 8.455657e-07,
        4.466630e-07, 2.468966e-07, 1.423754e-07, 7.665493e-08,
        4.189503e-08, 2.367454e-08, 1.263738e-08, 6.809730e-09], dtype=torch.float64)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05, 3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
    wv = torch.tensor([0, 1, 1, 0])
    waveform = Waveform(t, wv)


    lp = Loop.square(40)
    rx = Receiver()

    rho = torch.tensor([40,40])
    thick = torch.tensor([40])

    kwargs['waveform'] = waveform
    fwr = ForwardMaker(lp, rx, times, **kwargs)
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


def check_parallel_models(ForwardMaker):
    name = ForwardMaker.__name__
    from anemone.experiment import Loop, Receiver, Waveform
    lp = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    times = torch.logspace(-5, -3, 31, dtype=torch.double)

    rho = torch.ones(3)*40.
    rho_change = torch.linspace(-30, 260, 94)
    rho = rho[:, None] + rho_change[None, :]
    rho[0, :] = 40.
    rho[2, :] = 40.

    thick = np.ones_like(rho)[:-1, :]*20

    fwr = ForwardMaker(lp, rx, times, waveform=waveform)
    dbdt = fwr(rho, thick)
    print('dbdt shape: ', dbdt.shape)
    
    if __name__ == '__main__':
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.semilogy(dbdt)
        plt.show()

    assert dbdt.shape == (94, 31), f"{name} running parallel models does not return correct shape"




def check_forward_speeds(ForwardMaker):
    name = ForwardMaker.__name__
    from anemone.experiment import Loop, Receiver, Waveform
    lp = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    times = torch.logspace(-5, -3, 31, dtype=torch.double)

    rho = torch.ones(3)*40.
    rho_change = torch.linspace(-30, 260, 100)
    rho = rho[:, None] + rho_change[None, :]
    rho[0, :] = 40.
    rho[2, :] = 40.

    thick = np.ones_like(rho)[:-1, :]*20

    fwr = ForwardMaker(lp, rx, times, waveform=waveform)
    dbdt = fwr(torch.ones(3)*40, torch.ones(2)*20)


    from time import perf_counter
    t0 = perf_counter()
    dbdt = fwr(rho, thick)
    t1 = perf_counter()
    dt = t1-t0
    print(f'{name} {len(rho_change)} forward calculation took {dt:0.3f} s')

def test_forward_speeds():
    from anemone.forwards import Forward, ForwardGeomOp, Forward3, ForwardLaplace, ForwardLaplace2, ForwardLaplace3
    for cForward in [ForwardGeomOp, Forward3, ForwardLaplace3]:
        check_forward_speeds(cForward)


def test_hankel_forward_parallel_models():
    from anemone.forwards import Forward3
    check_parallel_models(Forward3)

def test_ForwardGeoOp_parallel_models():
    from anemone.forwards import ForwardGeomOp
    check_parallel_models(ForwardGeomOp)
    

def test_forwardlaplace3_parallel_models():
    from anemone.forwards import ForwardLaplace3
    check_parallel_models(ForwardLaplace3)

def test_40x40_aarhusInv_stepoff():
    from anemone.forwards import Forward
    aarhusInv_stepoff_comparison(Forward)

def test_40x40_aarhusInv_stepoff_geomop():
    from anemone.forwards import ForwardGeomOp
    aarhusInv_stepoff_comparison(ForwardGeomOp, rtol=2e-2)

def test_40x40_aarhusInv_stepoff_geomOpLaplace():
    from anemone.forwards import ForwardLaplace3
    aarhusInv_stepoff_comparison(ForwardLaplace3, rtol=2e-2)

def test_40x40_aarhusInv_stepoff_laplace():
    from anemone.forwards import ForwardLaplace
    aarhusInv_stepoff_comparison(ForwardLaplace)

def test_40x40_aarhusInv_stepoff_hankel_tolerance():
    from anemone.forwards import Forward
    aarhusInv_stepoff_comparison(Forward, tolerance=1e-5)

def test_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import Forward
    aarhusInv_piecewise_comparison(Forward)

def test_40x40_aarhusInv_piecewise_waveform_geomop():
    from anemone.forwards import ForwardGeomOp
    aarhusInv_piecewise_comparison(ForwardGeomOp, rtol=2e-2)

def test_HankelForwards_aarhusInv_piecewise_waveform():
    from anemone.forwards import Forward, ForwardGeomOp, Forward3
    for cForward in [Forward, ForwardGeomOp, Forward3]:
        aarhusInv_piecewise_comparison(cForward)

def test_Laplace_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import ForwardLaplace
    aarhusInv_piecewise_comparison(ForwardLaplace)


def test_Laplace2_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import ForwardLaplace2
    aarhusInv_piecewise_comparison(ForwardLaplace2)

def test_Laplace3_40x40_aarhusInv_piecewise_waveform():
    from anemone.forwards import ForwardLaplace3
    aarhusInv_piecewise_comparison(ForwardLaplace3, rtol=2e-2)




def test_IPforward():
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    from anemone.forwards import IPForward
    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=torch.double)
    pyfwr = IPForward(sq, rx, t0, waveform=waveform, filterfunc=recfilter)    

    res  = torch.tensor([4.000e+02, 4.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 8.000e+02, 
                        8.000e+02, 8.000e+02, 8.000e+02, 8.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 
                        5.000e+02, 3.000e+02, 3.000e+02, 3.000e+02], dtype=torch.double)
    m0   = torch.tensor([2.000e+02, 2.000e+02, 2.000e+02, 2.000e+02, 4.000e+02, 4.000e+02, 4.000e+02, 5.000e+02, 
                        5.000e+02, 5.000e+02, 5.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 
                        9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02], dtype=torch.double)*1e-3
    tau  = torch.tensor([1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 2.000e-04, 2.000e-04, 
                        2.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 
                        1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04], dtype=torch.double)
    chg  = torch.tensor([6.000e-01, 6.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 6.000e-01, 6.000e-01,
                         6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 9.000e-01, 
                         9.000e-01, 9.000e-01, 9.000e-01, 9.000e-01], dtype=torch.double)
    thick=torch.tensor([1.000e+00, 1.177e+00, 1.386e+00, 1.632e+00, 1.922e+00, 2.263e+00, 2.665e+00, 3.138e+00, 
                        3.695e+00, 4.351e+00, 5.124e+00, 6.034e+00, 7.105e+00, 8.366e+00, 9.851e+00, 1.160e+01, 
                        1.365e+01, 1.608e+01, 1.893e+01], dtype=torch.double)

    ainv_dbdt = torch.tensor([ 4.163592e-05,  2.674770e-05,  1.696241e-05,  1.037700e-05,
        6.115854e-06,  3.470203e-06,  1.870507e-06,  9.252422e-07,
        3.775450e-07,  7.785586e-08, -7.586203e-08, -1.447992e-07,
       -1.685453e-07, -1.687759e-07, -1.544787e-07, -1.313316e-07,
       -1.046215e-07, -7.896327e-08, -5.621327e-08, -3.828716e-08,
       -2.530188e-08, -1.641016e-08, -1.056120e-08, -6.790221e-09,
       -4.377552e-09, -2.834261e-09, -1.834226e-09, -1.188674e-09,
       -7.686370e-10, -4.945201e-10, -3.173105e-10], dtype=torch.double)
    
    dbdt = pyfwr(res, tau, m0, chg, thick)
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(t0, ainv_dbdt, 'k-', label='AarhusInv')
        ax[0].loglog(t0, -ainv_dbdt, 'k--')
        ax[0].loglog(pyfwr.times.numpy(), dbdt.real, 'C0.', label='IPforward')
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
    t0 = torch.logspace(-5, -3, 31, dtype=torch.double)
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

def test_forward_timingGP():
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    from anemone.forwards import ForwardGeomOp
    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=torch.double)
    fwr = ForwardGeomOp(sq, rx, t0, waveform=waveform, filterfunc=recfilter, tolerance=1e-6)    

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

    n = 100
    res = res[:, None] + (10/n)*torch.arange(n)[None, :]
    dbdt = fwr(res, thk)
    t2 = perf_counter()
    dbdt = fwr(res, thk)
    t3 = perf_counter()
    dt2 = (t3-t2)/n
    print(f'time for {n} forward calculations Geom: {dt2*n:0.3f} s', )
    assert dt2 < 0.2, "forward calculation took too long"    



    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1, sharex=True)
        ax.loglog(fwr.times.numpy(), dbdt.T)
        ax.set_ylabel('dB/dt')
        ax.set_aspect('equal')
        plt.show()

def test_forward_timingGPLP():
    from anemone.experiment import Loop, Receiver, Waveform, ButterworthFilter, FilterChain
    from anemone.forwards import ForwardGeomOpLaplace
    sq = Loop.square(40)
    rx = Receiver()
    waveform = Waveform.example()
    recfilter = FilterChain([
        ButterworthFilter(2.4e5, 1),
        ButterworthFilter(6.2e5, 1),
        ButterworthFilter(4.5e5, 2)
    ])
    t0 = torch.logspace(-5, -3, 31, dtype=torch.double)
    fwr = ForwardGeomOpLaplace(sq, rx, t0, waveform=waveform, filterfunc=recfilter, tolerance=1e-6)    

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
    print('time for 10 forward calculations Geom: ', dt2*10)
    assert dt2 < 0.2, "forward calculation took too long"    



    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1, sharex=True)
        ax.loglog(fwr.times.numpy(), dbdt.T)
        ax.set_ylabel('dB/dt')
        ax.set_aspect('equal')
        plt.show()

def multi_rx_forward_shape_check(ForwardMaker):
    name = ForwardMaker.__name__

    from anemone.experiment import Loop, Receiver, Waveform
    lp = Loop.square(50)
    rx = Receiver(x=torch.linspace(-20, 20, 5)*0.9)
    waveform = Waveform.example()
    times = torch.logspace(-5, -3, 31, dtype=torch.double)

    rho = torch.tensor([40, 40])
    thick = torch.tensor([10])

    fwr = ForwardMaker(lp, rx, times, waveform=waveform)
    dbdt = fwr(rho, thick)
    assert dbdt.shape == (5, 31), f"{name} multireceiver does not return correct shape"

def test_multirx_Forward1():
    from anemone.forwards import Forward
    multi_rx_forward_shape_check(Forward)

def test_multirx_Forward2():
    from anemone.forwards import ForwardGeomOp
    multi_rx_forward_shape_check(ForwardGeomOp)


def test_multirx_Forward3():
    from anemone.forwards import Forward3
    multi_rx_forward_shape_check(Forward3)

def test_multirx_Laplace3():
    from anemone.forwards import ForwardLaplace3
    multi_rx_forward_shape_check(ForwardLaplace3)

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

def test_multireceiver_frequency_domain():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import ForwardFD
    import torch

    lp = Loop.square(50)
    rx = Receiver(x=torch.linspace(-200, 200, 202), y=torch.linspace(-200, 200, 202))
    freqs = torch.tensor([2000], dtype=torch.double)

    rho = 40
    sigma = 1/torch.tensor([rho,2*rho])
    thick = torch.tensor([10])

    fwr = ForwardFD(lp, rx, freqs)
    from time import time as timer
    t0 = timer()
    B = fwr(sigma, thick)
    print('elapsed time: ', timer()-t0)
    
    t0 = timer()
    B = fwr(sigma, thick)
    print('elapsed time: ', timer()-t0)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.pcolormesh(rx.x,rx.y, torch.log10(torch.abs(B.reshape((202,202)))))
        plt.show()
    assert (rx.x[:, None]+freqs[None, :]).shape == B.shape, "dbdt shape is not correct"



if __name__ == '__main__':
    # test_Laplace3_40x40_aarhusInv_piecewise_waveform()
    # test_multirx_Laplace3()
    # test_forwardlaplace3_parallel_models()
    test_forward_speeds()

    exit()