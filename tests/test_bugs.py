import torch
from anemone.experiment import *

def test_gate_sampling():
    """ Some forwards were suffering from numerical instability when not using exact log sampling.
        This is not exactly reproduced here, so the test should be updated to reflect the actual problem.
        This test checks that the cubic spline interpolation is working correctly when using non-log sampling.
    """
    from anemone.forwards import Forward
    from anemone.transforms import CubicSpline
    from math import log10
    
    
    tx = Loop.square(1)
    rx = Receiver()

    rho = torch.tensor([40, 50, 80])
    sigma = 1./rho
    thk = torch.tensor([20, 20])


    lmgates = torch.logspace(log10(1e-6), log10(5e-3), 51)
    fwr = Forward(tx,rx,lmgates)
    dbdt0 = fwr(sigma,thk)
    cb = CubicSpline(fwr.times, dbdt0, transform=torch.log10)

    lmgates = torch.logspace(-6, -3, 32)*3
    fwr = Forward(tx,rx,lmgates)
    dbdt1 = fwr(sigma,thk)

    dbdt01 = cb(lmgates)

    assert torch.allclose(dbdt1, dbdt01)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)
        ax.loglog(fwr.times, -dbdt01)
        ax.loglog(fwr.times, -dbdt1, '.')
        plt.show()

if __name__ == '__main__':
    test_gate_sampling()

