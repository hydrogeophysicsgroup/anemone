from anemone import system
import torch
import numpy as np

def test_minimum_distance_symmetric():
    lp         = system.Loop.square(50)
    inside_rx  = system.Receiver()
    outside_rx = system.Receiver(x=250)

    rmin = lp.min_horizontal_distance(inside_rx)
    assert rmin == 25, "minimum distance calculator in loop object is failing"

    rmin = lp.min_horizontal_distance(outside_rx)
    print(rmin)
    assert rmin == 225, "minimum distance calculator in loop object is failing"

def test_minimum_distance_longline():
    tx = system.Loop([0, 1000], [0,0], [0,0])
    rx = system.Receiver(y = torch.logspace(0, 2, 31))

    rmin = tx.min_horizontal_distance(rx)
    assert rmin == 1., "minimum distance calculator in loop object is failing"


def test_maximum_distance_symmetric():
    lp = system.Loop.square(50)
    rx = system.Receiver()
    rmax = lp.max_horizontal_distance(rx)
    assert rmax == np.sqrt(2*25**2), "maximum distance calculator in loop object is failing"


def test_maximum_distance_nonsymmetric():
    from math import sqrt
    lp = system.Loop.square(50)
    rx = system.Receiver(x=1.)
    rmax = lp.max_horizontal_distance(rx)
    rmax_expected = sqrt(26**2 + 25**2)
    dr = rmax-rmax_expected
    if __name__ == '__main__':
        print('nonsymmetric rmax is: ', rmax)
        print('difference is: ', dr)
    assert abs(rmax-rmax_expected) < 1e-6, "maximum distance calculator in loop object is failing x-disp only"

    rx = system.Receiver(x=1., y=1.)
    rmax = lp.max_horizontal_distance(rx)
    rmax_expected = sqrt(26**2 + 26**2)
    dr = rmax-rmax_expected
    if __name__ == '__main__':
        print('nonsymmetric rmax is: ', rmax)
        print('difference is: ', dr)
    assert abs(rmax-rmax_expected) < 1e-6, "maximum distance calculator in loop object is failing x and y displacement"

    rx = system.Receiver(x=1., y=-1.)
    rmax = lp.max_horizontal_distance(rx)
    rmax_expected = sqrt(26**2 + 26**2)
    dr = rmax-rmax_expected
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        print('nonsymmetric rmax is: ', rmax)
        print('difference is: ', dr)
        plt.figure()
        plt.plot(lp.x, lp.y)
        plt.plot(rx.x, rx.y, 'o')
        plt.show()
    assert abs(rmax-rmax_expected) < 1e-6, "maximum distance calculator in loop object is failing x and y displacement"



def test_minimum_distance_nonsymmetric():
    lp = system.Loop.square(50)
    rx = system.Receiver(x=1.)
    rmin = lp.min_horizontal_distance(rx)
    if __name__ == '__main__':
        print('nonsymmetric rmin is: ', rmin)
    assert rmin == 24., "min distance calculator in loop object is failing"
    


def test_multireceiver():
    rx = system.Receiver(x=[0, 100])
    assert rx.x.shape == (2,), "receiver x cooridinates are not right"
    assert torch.allclose(rx.xyz, torch.tensor([[0, 0, 0], [100, 0, 0]], dtype=torch.double)), "receiver coordinates are not right"


if __name__ == '__main__':
    # test_minimum_distance_nonsymmetric()
    test_maximum_distance_nonsymmetric()
    test_multireceiver()
    test_minimum_distance()
    test_maximum_distance()
    test_minimum_distance_longline()
#    test

def _stepoff_reference(filters, t):
    """Independent reference for the step-off response of cascaded Butterworth
    filters: state-space solution y(t) = C expm(A t) x0, with x0 the steady
    state for a unit input (frequencies normalised to keep A well conditioned)."""
    from scipy import signal
    from scipy.linalg import expm
    w0 = 2*np.pi*filters[0].fcut
    num, den = [1.], [1.]
    for f in filters:
        b, a = signal.butter(f.order, 2*np.pi*f.fcut/w0, analog=True)
        num, den = np.polymul(num, b), np.polymul(den, a)
    A, B, C, D = signal.tf2ss(num, den)
    x0 = -np.linalg.solve(A, B)
    return np.array([(C @ expm(A*ti*w0) @ x0).item() for ti in t])


def test_filterchain_stepoff_repeated_poles():
    """Two identical first-order filters (a double pole) must not raise;
    the step-off response is (1 + w t) exp(-w t)."""
    fcut = 4.8e5
    fc = system.FilterChain([system.ButterworthFilter(fcut, 1), system.ButterworthFilter(fcut, 1)])
    t = torch.logspace(-8, -4, 101, dtype=torch.float64)
    w = 2*np.pi*fcut
    expected = (1 + w*t)*torch.exp(-w*t)
    assert torch.allclose(fc.stepoff_response(t), expected, rtol=0., atol=1e-12)


def test_filterchain_stepoff_distinct_poles():
    """Distinct first-order filters keep giving the partial-fraction result."""
    f1, f2 = 6.79e5, 4.2e5
    fc = system.FilterChain([system.ButterworthFilter(f1, 1), system.ButterworthFilter(f2, 1)])
    t = torch.logspace(-8, -4, 101, dtype=torch.float64)
    w1, w2 = 2*np.pi*f1, 2*np.pi*f2
    expected = w2/(w2-w1)*torch.exp(-w1*t) + w1/(w1-w2)*torch.exp(-w2*t)
    assert torch.allclose(fc.stepoff_response(t), expected, rtol=0., atol=1e-12)


def test_filterchain_stepoff_against_statespace():
    """Chains with repeated cutoffs and filters of order > 1."""
    BF = system.ButterworthFilter
    t = torch.logspace(-8, -4, 61, dtype=torch.float64)
    chains = [
        [BF(6.79e5, 1), BF(4.83e5, 1), BF(4.83e5, 1)],   # typical .gex: TiB filter + coil double pole
        [BF(3e5, 2)], [BF(3e5, 3)], [BF(3e5, 4)],
        [BF(3e5, 2), BF(3e5, 2), BF(1e5, 1)],
    ]
    for filters in chains:
        filt = system.FilterChain(filters) if len(filters) > 1 else filters[0]
        got = filt.stepoff_response(t).numpy()
        assert np.allclose(got, _stepoff_reference(filters, t.numpy()), rtol=0., atol=1e-10), filters


def test_stepoff_before_turnoff_is_one():
    fc = system.FilterChain([system.ButterworthFilter(4.8e5, 1), system.ButterworthFilter(4.8e5, 1)])
    t = torch.tensor([[-1e-6, 0.], [1e-9, 1e-2]], dtype=torch.float64)
    y = fc.stepoff_response(t)
    assert y.shape == t.shape
    assert y[0, 0] == 1. and y[0, 1] == 1.
    assert 0.99 < y[1, 0] < 1. and abs(y[1, 1]) < 1e-12
