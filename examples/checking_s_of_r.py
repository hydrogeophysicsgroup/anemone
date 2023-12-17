import matplotlib.pyplot as plt
import numpy as np

from em1d.experiment import Receiver, Loop

lp = Loop.square(25)

x = np.linspace(-50, 50, 101)
rx = Receiver(x=x, y=x)
rmax = lp.max_horizontal_distance(rx)
rmin = lp.min_horizontal_distance(rx)

r = np.geomspace(0.02, 120, 51)
s = lp.s_of_r(rx.xyz, r)

plt.figure()
plt.plot(s[0,0])
plt.plot(s[-1, -1])
plt.show()