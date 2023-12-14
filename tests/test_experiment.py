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


if __name__ == '__main__':
    test_minimum_distance()