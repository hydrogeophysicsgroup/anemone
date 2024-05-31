from em1d import experiment

import numpy as np

def test_minimum_distance():
    lp         = experiment.Loop.square(50)
    inside_rx  = experiment.Receiver()
    outside_rx = experiment.Receiver(x=250)

    rmin = lp.min_horizontal_distance(inside_rx)
    assert rmin == 25, "minimum distance calculator in loop object is failing"

    rmin = lp.min_horizontal_distance(outside_rx)
    print(rmin)
    assert rmin == 225, "minimum distance calculator in loop object is failing"


def test_maximum_distance():
    lp = experiment.Loop.square(50)
    rx = experiment.Receiver()
    rmax = lp.max_horizontal_distance(rx)
    assert rmax == np.sqrt(2*25**2), "maximum distance calculator in loop object is failing"

def test_40x40_aarhusInv_stepoff():
    from em1d.experiment import Loop, Receiver, Forward
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
    dbdt = fwr(sigma, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*np.abs(dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=0.5e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"

def test_40x40_aarhusInv_stepoff_hankel_tolerance():
    from em1d.experiment import Loop, Receiver, Forward
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

    fwr = Forward(lp, rx, times, tolerance=1e-5)
    dbdt = fwr(sigma, thick);

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*np.abs(dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=1e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"


def test_40x40_aarhusInv_piecewise_waveform():
    from em1d.experiment import Loop, Receiver, Forward, Waveform
    import numpy as np
    import torch


    ainv_dbdt = torch.tensor([3.853627e-04, 2.309149e-04, 1.255715e-04, 7.168451e-05,
        4.196206e-05, 2.297534e-05, 1.302996e-05, 7.591422e-06,
        4.179841e-06, 2.394241e-06, 1.452859e-06, 8.455657e-07,
        4.466630e-07, 2.468966e-07, 1.423754e-07, 7.665493e-08,
        4.189503e-08, 2.367454e-08, 1.263738e-08, 6.809730e-09], dtype=torch.complex128)

    times = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05,
        3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04])

    t = np.array([-1.0526e-03, -9.2963e-04,  0.0000e+00,  2.5000e-06])
    wv = np.array([0, 1, 1, 0])

    waveform = Waveform(t, wv)


    lp = Loop.square(40)
    rx = Receiver()

    rho = 40
    sigma = 1/torch.tensor([rho,rho])
    thick = torch.tensor([10])

    fwr = Forward(lp, rx, times, waveform=waveform)
    dbdt = fwr(sigma, thick)
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,2, sharex=True)
        ax[0].loglog(fwr.times.numpy(), dbdt.real, label='mforward')
        ax[0].loglog(times,      ainv_dbdt, '--', label='AarhusInv')
        ax[0].legend()
        ax[0].set_aspect('equal')

        ax[1].semilogx(times, 100*np.abs(dbdt.real-ainv_dbdt)/np.abs(ainv_dbdt))
        ax[1].set_ylabel('relative difference [%]')

        fig.suptitle('Test against AarhusInv with Piecewise Waveform')
        plt.show()
    assert torch.allclose(ainv_dbdt.real, dbdt.real, rtol=0.5e-2), "aarhus inv and pytorch forward do not match within 1/2 percent"

if __name__ == '__main__':
    # test_minimum_distance()
    test_40x40_aarhusInv_stepoff_hankel_tolerance()
    # test_40x40_aarhusInv_stepoff()