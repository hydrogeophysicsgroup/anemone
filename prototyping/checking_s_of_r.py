import matplotlib.pyplot as plt
import torch

from anemone.experiment import Receiver, Loop

lp = Loop.square(500)

rx = Receiver()
rmax = lp.max_horizontal_distance(rx).log10()+0.1
rmin = lp.min_horizontal_distance(rx).log10()-0.1


r = torch.logspace(rmin, rmax, 50)
s = lp.geomz(rx, r).squeeze()

plt.figure()
plt.plot(r, s, '.-')
# plt.xscale('log')

if __name__ == "__main__":
    plt.show()