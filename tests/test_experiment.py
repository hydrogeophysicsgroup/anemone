from anemone import experiment
import torch
import numpy as np

def test_minimum_distance_symmetric():
    lp         = experiment.Loop.square(50)
    inside_rx  = experiment.Receiver()
    outside_rx = experiment.Receiver(x=250)

    rmin = lp.min_horizontal_distance(inside_rx)
    assert rmin == 25, "minimum distance calculator in loop object is failing"

    rmin = lp.min_horizontal_distance(outside_rx)
    print(rmin)
    assert rmin == 225, "minimum distance calculator in loop object is failing"

def test_minimum_distance_longline():
    tx = experiment.Loop([0, 1000], [0,0], [0,0])
    rx = experiment.Receiver(y = torch.logspace(0, 2, 31))

    rmin = tx.min_horizontal_distance(rx)
    assert rmin == 1., "minimum distance calculator in loop object is failing"


def test_maximum_distance_symmetric():
    lp = experiment.Loop.square(50)
    rx = experiment.Receiver()
    rmax = lp.max_horizontal_distance(rx)
    assert rmax == np.sqrt(2*25**2), "maximum distance calculator in loop object is failing"


def test_maximum_distance_nonsymmetric():
    from math import sqrt
    lp = experiment.Loop.square(50)
    rx = experiment.Receiver(x=1.)
    rmax = lp.max_horizontal_distance(rx)
    rmax_expected = sqrt(26**2 + 25**2)
    dr = rmax-rmax_expected
    if __name__ == '__main__':
        print('nonsymmetric rmax is: ', rmax)
        print('difference is: ', dr)
    assert abs(rmax-rmax_expected) < 1e-6, "maximum distance calculator in loop object is failing x-disp only"

    rx = experiment.Receiver(x=1., y=1.)
    rmax = lp.max_horizontal_distance(rx)
    rmax_expected = sqrt(26**2 + 26**2)
    dr = rmax-rmax_expected
    if __name__ == '__main__':
        print('nonsymmetric rmax is: ', rmax)
        print('difference is: ', dr)
    assert abs(rmax-rmax_expected) < 1e-6, "maximum distance calculator in loop object is failing x and y displacement"

    rx = experiment.Receiver(x=1., y=-1.)
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
    lp = experiment.Loop.square(50)
    rx = experiment.Receiver(x=1.)
    rmin = lp.min_horizontal_distance(rx)
    if __name__ == '__main__':
        print('nonsymmetric rmin is: ', rmin)
    assert rmin == 24., "min distance calculator in loop object is failing"
    


def test_multireceiver():
    rx = experiment.Receiver(x=[0, 100])
    assert rx.x.shape == (2,), "receiver x cooridinates are not right"
    assert torch.allclose(rx.xyz, torch.tensor([[0, 0, 0], [100, 0, 0]], dtype=torch.double)), "receiver coordinates are not right"


if __name__ == '__main__':
    # test_minimum_distance_nonsymmetric()
    test_maximum_distance_nonsymmetric()
    test_multireceiver()
    test_minimum_distance()
    test_maximum_distance()
    test_minimum_distance_longline()