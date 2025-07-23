import matplotlib.pyplot as plt
import numpy as np

from anemone.forwards import Forward
from anemone.physics import hzdt_halfspace_center_circle_loop
from anemone.experiment import Loop, Receiver
import torch
twopi = torch.pi*2
fourpi = 2*twopi
    
times = torch.logspace(-6, -1, 51)
    
lp = Loop.circle(50)
rx=Receiver()

rho = torch.ones(2)*100
thick = torch.tensor([40])

fwr = Forward(lp, rx, times)

dhdt = fwr(rho, thick)/(fourpi*1e-7)

dhdt_analytic = hzdt_halfspace_center_circle_loop(times, rho[0], 50)


fig,ax = plt.subplots()
ax.plot(times, dhdt_analytic, 'gray', label='analytic')
ax.plot(times, dhdt.real, '--', label='numerical')

ax.legend()
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_aspect('equal')
ax.set_ylim(1e-7, 1e4)

if __name__ == "__main__":
    plt.show()