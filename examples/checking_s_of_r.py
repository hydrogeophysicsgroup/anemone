import matplotlib.pyplot as plt
import numpy as np

from anemone.experiment import Receiver, Loop

lp = Loop.square(500)

rx = Receiver()
rmax = lp.max_horizontal_distance(rx)
rmin = lp.min_horizontal_distance(rx)


r = np.geomspace(rmin, rmax, 50)
s = lp.s_of_r(np.zeros(3), r)

plt.figure()
plt.plot(r, s, '.-')
# plt.xscale('log')
plt.show()