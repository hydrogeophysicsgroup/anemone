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


def test_mulirx_symmetry():
    tx = Loop.square(2)
    from anemone.forwards import Forward
    
    X,Y = torch.linspace(-1, 1, 10)*0.98, np.linspace(-1, 1, 10)*0.98

    

    rho = torch.tensor([40, 50, 80])
    sigma = 1./rho
    thk = torch.tensor([20, 20])

    lm_gates = torch.logspace(-6, -3, 32)

    # resp = torch.zeros(X.shape+Y.shape)
    
    # for ix,x in enumerate(X):
    #     for iy,y in enumerate(Y):
    #         rx = Receiver(x=x, y=y)
    #         fwr = Forward(tx,rx,lm_gates)
    #         # print('rmin rmax: ', fwr.rmin, fwr.rmax)
    #         dbdt = fwr(sigma,thk)
    #         resp[ix,iy] = dbdt[0]

    rx1 = Receiver(x=X, y=Y)
    fwr = Forward(tx,rx1,lm_gates)
    dbdt = fwr(sigma,thk)
    print(dbdt.shape)



    if __name__ == '__main__':

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)
        pcm=ax.pcolor(X,Y, resp-resp.T)
        plt.colorbar(pcm)
        plt.show()

    assert torch.allclose(resp, resp.T), "broken symmetry in multi-receiver forward"


if __name__ == '__main__':
    #test_gate_sampling()
    test_mulirx_symmetry()

