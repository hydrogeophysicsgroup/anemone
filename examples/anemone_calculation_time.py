import sys, os
import matplotlib.pyplot as plt
import numpy as np
outname = sys.argv[0][:-3]

n_mods = np.unique(np.geomspace(1, 10000, 30).astype(int))#[::-1]
print('n_steps: ', len(n_mods))

print(n_mods)

def my_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

import inspect
import socket
cpu_name = socket.gethostname()
thismodule = sys.modules[__name__]
calc_part = []
for line in inspect.getsource(thismodule).split('\n'):
    if line.startswith('# plotting'): break
    if not 'print' in line: calc_part.append(line)

filehash = str(my_hash('\n'.join(calc_part)))

from time import perf_counter as timer

a = timer()

def generate_forward(device, dtype):

    if device=='gpu': device='cuda:2'

    import torch
    torch.set_default_device(torch.device('cpu'))
    from anemone.forwards import Forward
    torch.set_default_dtype(torch.float32 if dtype=='32' else torch.float64)
    fwr = Forward.example()

    dev = torch.device(device)
    torch.set_default_device(dev)
    fwr.set_device(dev)
    fwr.primary = fwr.primary.to(dev)
    return torch, fwr    


hardwares = ['gpu-64', 'gpu-32', 'cpu-64', 'cpu-32']

time_per_run = {}

n_times = 10


for hardware in hardwares:
    if os.path.exists(cpu_name+filehash+f'_{hardware}.npy'):
        time_per_run[hardware] = np.load(cpu_name+filehash+f'_{hardware}.npy')
    else:
        time_per_run[hardware] = np.zeros((len(n_mods), n_times))
        torch,fwr = generate_forward(*hardware.split('-'))

        for i_mod, n_mod in enumerate(n_mods):
            # n_times = np.clip(int(np.ceil(10000/n_mod)), 3,10)
            print(f'{hardware} starting {n_mod}\t({n_times:03d} runs)')
            if n_mod>1:
                rho = (torch.logspace(0,3,n_mod)[None,:]*torch.ones(10)[:, None])*torch.randn((10,n_mod))
                thk = torch.ones((9,n_mod))*10.
            else:
                rho = torch.ones(10)*40.
                thk = torch.ones(9)*10.
            
            
            for i_rep in range(n_times):
                bt = timer()
                _ = fwr(rho, thk)
                et = timer()
                time_per_run[hardware][i_mod,i_rep] = et-bt
        time_per_run[hardware] = np.array(time_per_run[hardware])
        np.save(cpu_name+filehash+f'_{hardware}.npy', time_per_run[hardware])

# plotting

fig,axs =plt.subplots(1,1)#, sharex=True, sharey=True)
axs = [axs]
axs[0].loglog(n_mods, time_per_run['cpu-64'].mean(-1)/n_mods, 'r', label='cpu-64')
axs[0].loglog(n_mods, time_per_run['gpu-64'].mean(-1)/n_mods, 'b', label='gpu-64')

axs[0].loglog(n_mods, time_per_run['cpu-32'].mean(-1)/n_mods, 'r--', label='cpu-32')
axs[0].loglog(n_mods, time_per_run['gpu-32'].mean(-1)/n_mods, 'b--', label='gpu-32')
axs[0].set_xlabel('No. of Models')
axs[0].set_ylabel('Time per Model [s]')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(1e-5, 1)

# axs[1].loglog(n_mods, np.max(time_per_run['cpu-64'], -1)/n_mods, 'r', label='cpu-64')
# axs[1].loglog(n_mods, np.max(time_per_run['gpu-64'], -1)/n_mods, 'b', label='gpu-64')

# axs[1].loglog(n_mods, np.max(time_per_run['cpu-32'], -1)/n_mods, 'r--', label='cpu-32')
# axs[1].loglog(n_mods, np.max(time_per_run['gpu-32'], -1)/n_mods, 'b--', label='gpu-32')
# axs[1].set_xlabel('No. of Models')
# axs[1].set_ylabel('Time per Model [s]')
# axs[1].legend()
# axs[1].grid(True)

plt.tight_layout()
plt.show()



if 'save' in sys.argv:
    fig.savefig(outname)

