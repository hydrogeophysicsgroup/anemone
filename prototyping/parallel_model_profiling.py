# %%
from anemone.forwards import Forward
import torch
import matplotlib.pyplot as plt
import numpy as np

# %%
fwr = Forward.example()

rho = torch.ones(10)*40
thk = torch.ones(len(rho)-1)



# %%
dbdt=fwr(rho,thk)

# %%
from time import perf_counter as timer
n_mods = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
t_calc0 = []
t_calc1 = []

fwr = Forward.example()

for n_mod in n_mods:
    rhos_mult = np.linspace(1, 10, n_mod)
    rhos = rho[:, None] * rhos_mult[None, :]
    
    bt0 = timer()
    fwr = Forward.example(tolerance=1e-6)
    bt1 = timer()
    dbdt = fwr(rhos, thk)
    dt0 = timer() - bt0
    dt1 = timer() - bt1
    t_calc0.append(dt0/n_mod)
    t_calc1.append(dt1/n_mod)

#    del dbdt, fwr
    print(f"Calculated {n_mod:04d} models in {dt0:.2f} seconds.")

# %%
plt.figure()
plt.loglog(n_mods, t_calc0, '-o', label='fwr recalc.')
plt.loglog(n_mods, t_calc1, '-o', label='fwr re-used')
plt.legend()
plt.ylim(1e-3, 1)
plt.show()
# %% [markdown]
# Double precision results:
# 
# ![image.png](attachment:image.png)

# %% [markdown]
# 


