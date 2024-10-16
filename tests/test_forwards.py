import torch

def test_40x40_aarhusInv_stepoff():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import Forward
    import torch


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

    rho = 40
    sigma = 1/torch.tensor([rho,rho])
    thick = torch.tensor([10])

    fwr = Forward(lp, rx, times)
    dbdt = fwr(1./sigma, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv 40x40 m^2 loop with stepoff waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=1e-2), "aarhus inv and pytorch forward do not match within 1 percent"


def test_40x40_aarhusInv_stepoffGeomOp():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import ForwardGeomOp
    import torch


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

    rho = 40
    sigma = 1/torch.tensor([rho,rho])
    thick = torch.tensor([10])

    fwr = ForwardGeomOp(lp, rx, times)
    dbdt = fwr(1./sigma, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv 40x40 m^2 loop with stepoff waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=2e-2), "aarhus inv and pytorch forward do not match within 2 percent"

def test_40x40_aarhusInv_stepoffGeomOpLaplace():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import ForwardGeomOpLaplace
    import torch


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

    rho = 40
    sigma = 1/torch.tensor([rho,rho])
    thick = torch.tensor([10])

    fwr = ForwardGeomOpLaplace(lp, rx, times)
    dbdt = fwr(1./sigma, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*torch.abs(dbdt.real-ainv_dbdt)/torch.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv 40x40 m^2 loop with stepoff waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=2e-2), "aarhus inv and pytorch forward do not match within 2 percent"


def test_Laplace_40x40_aarhusInv_stepoff():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import ForwardLaplace
    import torch


    ainv_dbdt=torch.tensor([2.749861e-04, 1.753167e-04, 1.011688e-04, 6.031561e-05,
        3.650625e-05, 2.059544e-05, 1.194428e-05, 7.079678e-06,
        3.957627e-06, 2.293402e-06, 1.403765e-06, 8.236817e-07,
        4.389777e-07, 2.446409e-07, 1.422868e-07, 7.752075e-08,
        4.302391e-08, 2.479088e-08, 1.362022e-08, 7.621867e-09], dtype=torch.complex128)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04], dtype=torch.float64)

    lp = Loop.square(40)
    rx = Receiver()

    rho = torch.tensor([40,40])
    thick = torch.tensor([40])


    fwr = ForwardLaplace(lp, rx, times, tolerance=1e-4)
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

        fig.suptitle('Test against AarhusInv 40x40 m^2 loop with stepoff waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=1e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"

def test_40x40_aarhusInv_stepoff_hankel_tolerance():
    from anemone.experiment import Loop, Receiver
    from anemone.forwards import Forward
    import torch


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

    
    rho = torch.tensor([40,40])
    thick = torch.tensor([40])

    fwr = Forward(lp, rx, times, tolerance=1e-5)
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

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=1e-2), "aarhus inv and pytorch forward do not match within 1 percent"


def test_40x40_aarhusInv_piecewise_waveform():
    from anemone.experiment import Loop, Receiver, Waveform
    from anemone.forwards import Forward
    import torch


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


    fwr = Forward(lp, rx, times, waveform=waveform)
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
    assert torch.allclose(ainv_dbdt, dbdt, rtol=0.5e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"

def test_Laplace_40x40_aarhusInv_piecewise_waveform():
    from anemone.experiment import Loop, Receiver, Waveform
    from anemone.forwards import ForwardLaplace
    import torch


    ainv_dbdt = torch.tensor([3.853627e-04, 2.309149e-04, 1.255715e-04, 7.168451e-05,
        4.196206e-05, 2.297534e-05, 1.302996e-05, 7.591422e-06,
        4.179841e-06, 2.394241e-06, 1.452859e-06, 8.455657e-07,
        4.466630e-07, 2.468966e-07, 1.423754e-07, 7.665493e-08,
        4.189503e-08, 2.367454e-08, 1.263738e-08, 6.809730e-09], dtype=torch.float64)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05, 3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04], dtype=torch.float64)

    t = torch.tensor([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06], dtype=torch.float64)
    wv = torch.tensor([0, 1, 1, 0], dtype=torch.float64)

    waveform = Waveform(t, wv)


    lp = Loop.square(40)
    rx = Receiver()

    rho = torch.tensor([40,40])
    thick = torch.tensor([40])


    fwr = ForwardLaplace(lp, rx, times, waveform=waveform)
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
    assert torch.allclose(ainv_dbdt, dbdt, rtol=0.5e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"

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

if __name__ == '__main__':
    # test_forward_timing()
    test_forward_timingGP()
    test_forward_timingGPLP()
    # test_40x40_aarhusInv_stepoffGeomOp()
#    test_40x40_aarhusInv_stepoffGeomOpLaplace()
    # test_IPforward()
    #test_40x40_aarhusInv_stepoff()
    # test_Laplace_40x40_aarhusInv_piecewise_waveform()