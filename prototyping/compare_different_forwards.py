#%%
from anemone.forwards import *
import torch
import matplotlib.pyplot as plt
import numpy as np


# %%
t = torch.logspace(-6, -3, 32)
fwr1 = Forward.example(times=t)
fwr2 = ForwardGeomOp.example(times=t)
fwr3 = Forward3.example(times=t)

rho = torch.ones(3)*40
rho[1] = 100.
thick = torch.ones(2)*40

dbdt1 = fwr1(rho, thick)
dbdt2 = fwr2(rho, thick)
dbdt3 = fwr3(rho, thick)

# %%

plt.figure()
plt.loglog(fwr1.times,  dbdt1, '-')
plt.loglog(fwr2.times,  dbdt2, '--')
plt.loglog(fwr3.times,  dbdt3, '.')


# %%
