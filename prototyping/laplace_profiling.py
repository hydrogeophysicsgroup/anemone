from em1d.experiment import *
from em1d.transforms import invLaplace


tx = Loop.square(40)
rx = Receiver()
wv = Waveform.example()
#t = torch.logspace(-5, -3, 16, dtype=torch.double)
t = torch.tensor([9.350e-06, 1.145e-05, 1.458e-05, 1.822e-05, 2.255e-05, 2.867e-05, 3.595e-05, 4.460e-05, 5.660e-05, 7.070e-05, 8.630e-05, 1.071e-04,
        1.381e-04, 1.748e-04, 2.174e-04, 2.775e-04, 3.515e-04, 4.385e-04,
        5.575e-04, 7.035e-04], dtype=torch.float64)

f0 = Forward(tx, rx, t, wv, tolerance=1e-5)
f1 = ForwardLaplace(tx, rx, t, wv, tolerance=1e-5)
# f2 = FastForward(tx, rx, t, tolerance=1e-5)

rho = torch.tensor([40, 200, 70], dtype=torch.double)
thk = torch.tensor([10, 35], dtype=torch.double)
sigma = 1./rho

dbdt0 = f0(sigma, thk)
dbdt1 = f1(sigma, thk)
# dbdt2 = f2(sigma, thk)


from cProfile import Profile
from pstats import SortKey, Stats
print(dir(SortKey))

with Profile() as profile:
    profile.run('f0(sigma, thk)')
    # profile.sort_stats('tottime')
    profile.dump_stats('output.prof')#__file__.replace('.py', '_profile_res.txt'))


stream = open('output1.txt', 'w')
stats = Stats('output.prof', stream=stream)
stats.sort_stats('cumtime')
stats.print_stats()

with Profile() as profile:
    profile.run('f1(sigma, thk)')
    # profile.sort_stats('cumtime')
    profile.dump_stats('output.prof')#__file__.replace('.py', '_profile_res.txt'))


stream = open('output2.txt', 'w')
stats = Stats('output.prof', stream=stream)
stats.sort_stats('cumtime')
stats.print_stats()

    # with open(__file__.replace('.py', '_profile_res.txt'), 'w') as fout:
    #     fout.write(stats.print_stats())

#     res = (f"{f0(sigma, thk)}")
#     (
#         Stats(profile)
#         .strip_dirs()
#         .sort_stats(SortKey.CUMULATIVE)
#         .print_stats()
#     )
#     with open(__file__.replace('.py', '_profile_res.txt'), 'w') as fout:
#         fout.write(res)

# with 