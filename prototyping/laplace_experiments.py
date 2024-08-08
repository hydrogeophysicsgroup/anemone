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


# import time

# for forward,name in zip([f0, f1], ['fourier', 'laplace']):
#     t1 = time.perf_counter(), time.process_time()
#     forward(sigma, thk)
#     t2 = time.perf_counter(), time.process_time()
#     print(name)
#     print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
#     print(f" CPU time: {t2[1] - t1[1]:.2f} seconds\n")

# for forward,name in zip([f0, f1], ['fourier', 'laplace']):
#     t1 = time.perf_counter(), time.process_time()
#     forward(sigma, thk)
#     t2 = time.perf_counter(), time.process_time()
#     print(name)
#     print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
#     print(f" CPU time: {t2[1] - t1[1]:.2f} seconds\n")

from time import time as timer
bt = timer()
n_times = 20
for i in range(n_times):
    dbdt0=f0(sigma, thk)
print(f'run time: {1e3*(timer()-bt)/n_times:1.2f} ms')


from time import time as timer
bt = timer()
n_times = 20
for i in range(n_times):
    dbdt1=f1(sigma, thk)
print(f'run time: {1e3*(timer()-bt)/n_times:1.2f} ms')

import matplotlib.pyplot as plt

plt.figure()
plt.loglog(t.numpy(), dbdt0.numpy())
plt.loglog(t.numpy(), dbdt1.numpy())
plt.show()