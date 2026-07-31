from anemone.inversion import *
from anemone.forwards import Forward, logify_forward
from anemone.system import Loop, Receiver, Waveform, FilterChain, ButterworthFilter
from anemone.experiment import compute_relstd
import pstats   

def test_sci(n_model=25):
    gpu = torch.device('cpu')
    logForward = logify_forward(Forward)

    t1 = torch.logspace(-6, -4, 15)
    t2 = torch.logspace(-5, -3, 31)
    tx = Loop.square(3)
    rx = Receiver(x=10.)
    wv1 = Waveform(torch.tensor([-0.5e-3, -0.9e-3, 0., 0.5e-6]), torch.tensor([0, 1, 1, 0]))
    wv2 = Waveform(torch.tensor([-1e-3, -1.5e-3, 0., 0.5e-5]), torch.tensor([0, 1, 1, 0]))

    fwr1 = logForward(tx, rx, t1, waveform=wv1, tolerance=1e-6)
    fwr2 = logForward(tx, rx, t2, waveform=wv2, tolerance=1e-6)

    fwr = fwr1 + fwr2


    z = torch.cumsum(torch.logspace(0, np.log10(50.), 20), 0)
    z /= z.max()

    z *= 80.

    rho_true = torch.tensor([75., 5., 60.], device=gpu, dtype=rdtype)
    thk_true = torch.tensor([20, 30], device=gpu, dtype=rdtype)


    res_start = LogResistivity.smooth(torch.ones(len(z)+1)*50., z)
    thick1 = torch.linspace(0, 20, n_model).to(rdtype)#[:3]
    inv_list = []
    m_trues=[]

    for i,dthick in enumerate(thick1):
        this_fwr = fwr.copy()
        thk = thk_true.clone()
        thk[0] += dthick
        crho = rho_true.clone()
        crho[1] += abs(i-len(thick1)//2)*2.
        crho[2] -= abs((i-5)*70/len(thick1))
        m_true = LogResistivity.few_layers(torch.clamp(crho, 5), thk)
        m_trues.append(m_true)
        
        dbdt = this_fwr(crho, thk).exp()
        d_err = compute_relstd(fwr.times, dbdt, current=10.)
        d_obs = torch.abs(dbdt * (1. + d_err*torch.randn_like(dbdt))).log()
        tmp_inv = fastJac(d_obs, torch.log1p(d_err), res_start.copy(), this_fwr)

        # tmp_inv.setup_matrices()
        inv_list.append(tmp_inv)    
    print(f"Created {len(inv_list)} inversions")
    n_pts = len(thick1)

    # 1 km radius (in meters)
    radius = 1000.0

    # Random angles [0, 2π)
    angles = np.random.uniform(0, 2*np.pi, n_pts)

    # Random radii with uniform spatial density
    r = radius * np.sqrt(np.random.uniform(0, 1, n_pts))

    # Convert to Cartesian coordinates
    x_positions = r * np.cos(angles)
    y_positions = r * np.sin(angles)
    from time import time as timer
    bt = timer()

    import cProfile
     
    sci = SCI(inv_list, x = np.array(x_positions), y = np.array(y_positions))
    sci.device = gpu
    sci.horizontal_smoothness=0.7
    sci.vertical_smoothness=1

    # lci.smoothness_vert = 1e3
    # lci.smoothness_horiz = 1e2

    profiler = cProfile.Profile()
    profiler.enable()
    m1 = sci.m
    sci.mvec=m1.clone()
    sci._current_mvec = m1.clone()+0.1

    bt = timer()

    _ = sci.data_predicted

    print('\t run1: ', timer()-bt)

    bt = timer()

    _ = sci.data_predicted

    print('\t run2: ', timer()-bt)

    bt = timer()
    
    sci.mvec += 0.2
    sci._model_update(m1+0.2)

    print('\t run3: ', timer()-bt)

    profiler.disable()
    return profiler


for n_model in [5, 10, 25, 40, 50, 75, 100, 200, 500]:
    print(f"Running SCI with {n_model} models...")
    profiler = test_sci(n_model=n_model)
    # ps = pstats.Stats(profiler, stream=f).sort_stats('cumulative')
    # ps.print_stats()
