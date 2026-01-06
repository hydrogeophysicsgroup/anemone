import torch
from anemone.experiment import *
from anemone.forwards import *

from pytest_config import *



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
    thk = torch.tensor([20, 20])


    lmgates = torch.logspace(log10(1e-6), log10(5e-3), 51)
    fwr = Forward(tx,rx,lmgates)
    dbdt0 = fwr(rho,thk)
    cb = CubicSpline(fwr.times, dbdt0, transform=torch.log10)

    lmgates = torch.logspace(-6, -3, 32)*3
    fwr = Forward(tx,rx,lmgates)
    dbdt1 = fwr(rho,thk)

    dbdt01 = cb(lmgates)

    assert torch.allclose(dbdt1, dbdt01)

    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)
        ax.loglog(fwr.times, -dbdt01)
        ax.loglog(fwr.times, -dbdt1, '.')
        plt.show()

@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multirx():
    tx = Loop.square(2)
    from anemone.forwards import BaseForward as Forward
    import numpy as np
    
    X,Y = torch.linspace(-1, 1, 7)*0.98, np.linspace(-1, 1, 7)*0.98

    

    rho = torch.tensor([40, 50, 80])
    thk = torch.tensor([20, 20])

    lm_gates = torch.logspace(-6, -3, 32)

    resp = torch.zeros(X.shape+Y.shape, dtype=torch.float64)
    
    for ix,x in enumerate(X):
        for iy,y in enumerate(Y):
            rx = Receiver(x=x, y=y)
            fwr = Forward(tx,rx,lm_gates)
            # print('rmin rmax: ', fwr.rmin, fwr.rmax)
            dbdt = fwr(rho,thk)
            resp[ix,iy] = dbdt[0]

    rx1 = Receiver(x=X, y=Y)
    fwr = Forward(tx,rx1,lm_gates)
    dbdt = fwr(rho,thk)

    resp2 = dbdt[..., 0].reshape(X.shape+Y.shape)

    assert torch.allclose(resp, resp2, rtol=5e-3), 'multi-receiver forward not consistent with single-receiver forwards within 1/2 percent'
    
    if __name__ == '__main__':
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,3)
        pcm=ax[0].pcolor(X,Y, resp)
        plt.colorbar(pcm, ax=ax[0])
        pcm=ax[1].pcolor(X,Y, resp2)
        plt.colorbar(pcm, ax=ax[1])
        pcm=ax[2].pcolor(X,Y, (resp-resp2) / resp)
        plt.colorbar(pcm, ax=ax[2])
        plt.show()
        


@pytest.mark.skipif(skip_multi_rx, reason="multi-rx needs retooling.")
def test_multirx_symmetry():
    tx = Loop.square(2)
    from anemone.forwards import BaseForward
    import numpy as np
    
    X,Y = torch.linspace(-1, 1, 25)*0.9, np.linspace(-1, 1, 25)*0.9

    

    rho = torch.tensor([40, 50, 80])
    thk = torch.tensor([20, 20])

    lm_gates = torch.tensor([1e-5, 2e-5])

    rx1 = Receiver(x=X, y=Y)
    fwr = BaseForward(tx,rx1,lm_gates)
    dbdt = fwr(rho,thk)
    

    resp = dbdt[..., 0].reshape(X.shape+Y.shape).cpu().numpy()
    assert np.allclose(resp, resp[::-1, ::-1]), 'both axes flipped symmetry'
    assert np.allclose(resp, resp[::-1]), 'x axis flipped symmetry'
    assert np.allclose(resp, resp[:, ::-1]), 'y axis flipped symmetry'
    assert np.allclose(resp, resp.T), 'transpose symmetry'

    if __name__ == '__main__':

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(1,1)
        pcm=ax.pcolor(X,Y, resp)
        plt.colorbar(pcm)
        plt.show()

def test_eliye_spring2025_site_31():
    from anemone.inversion import fastJac
    from anemone.inversion import LogResistivity

    fwr_file = __file__.replace('test_bugs.py', 'dmfwr_eliye_spring1.em1')
    logForward = logify_forward(Forward)
    fwr = logForward.load(fwr_file)


    # from site 31
    d_obs =  torch.tensor([-13.2652, -13.9075, -14.4880, -15.0030, -15.4425, -15.8108, -15.2540,
            -15.6625, -16.0183, -16.3622, -16.7070, -17.0591, -17.4193, -17.7953,
            -18.1892, -18.6089, -19.0497, -19.5148, -19.9874, -20.4797, -20.9859,
            -21.4945, -22.0251, -22.5574, -23.1490, -23.7480, -24.3312, -24.9615],
        dtype=torch.float64)
    d_err =  torch.tensor([0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296,
        0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296,
        0.0296, 0.0297, 0.0299, 0.0301, 0.0306, 0.0325, 0.0369, 0.0465, 0.0672,
        0.1042], dtype=torch.float64)

    # t_max = fwr.times.max().item()
    d_max = 150.
    n_layers = 30

    thk = 1.+torch.logspace(-0.5, np.log10(d_max-n_layers+1), n_layers).diff()
    z = thk.cumsum(0)
    res_start=LogResistivity.smooth(torch.ones(n_layers)*100., z)    

    inv = fastJac(d_obs, d_err, res_start, fwr)
    res_mod = inv.invert()
    assert inv.misfits[-1] < 1.1

def test_eliye_spring2025_site_30():
    from anemone.inversion import fastJac
    from anemone.inversion import LogResistivity

    fwr_file = __file__.replace('test_bugs.py', 'dmfwr_eliye_spring1.em1')
    logForward = logify_forward(Forward)
    fwr = logForward.load(fwr_file)

    # from site 30
    d_obs =  torch.tensor([-13.4820, -14.0622, -14.5825, -15.0316, -15.4393, -15.7847, -15.2696,
            -15.6432, -15.9805, -16.3164, -16.6608, -17.0175, -17.3860, -17.7721,
            -18.1764, -18.6033, -19.0538, -19.5237, -20.0098, -20.5082, -21.0126,
            -21.5242, -22.0509, -22.6287, -23.1582, -23.7513, -24.4175, -25.0993], dtype=torch.float64)
    d_err =  torch.tensor([0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296,
            0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0296, 0.0297,
            0.0297, 0.0297, 0.0299, 0.0302, 0.0307, 0.0326, 0.0358, 0.0425, 0.0606,
            0.1086], dtype=torch.float64)

    # t_max = fwr.times.max().item()
    d_max = 150.
    n_layers = 30

    thk = 1.+torch.logspace(-0.5, np.log10(d_max-n_layers+1), n_layers).diff()
    z = thk.cumsum(0)
    res_start=LogResistivity.smooth(torch.ones(n_layers)*100., z)    

    inv = fastJac(d_obs, d_err, res_start, fwr)
    res_mod = inv.invert()
    assert inv.misfits[-1] < 1.1

# from site 

if __name__ == '__main__':
    test_eliye_spring2025_site_31()
    #test_gate_sampling()
    # test_multirx_symmetry()

