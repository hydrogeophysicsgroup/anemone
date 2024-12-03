from test_forwards import check_forward_speeds
from anemone.forwards import Forward, ForwardLaplace, Forward3,  DualMoment, DualMomentLaplace, ForwardGeomOpLaplace, ForwardLaplace3

# check_forward_speeds(Forward)
# check_forward_speeds(DualMomentLaplace)
#check_forward_speeds(ForwardLaplace3, tolerance=1e-6)
#check_forward_speeds(ForwardLaplace, tolerance=1e-6)
#check_forward_speeds(Forward3, tolerance=1e-6)
# check_forward_speeds(DualMoment)
#check_forward_speeds(ForwardGeomOpLaplace)

fwr1 = Forward.example()

fwr = DualMoment.colecole(fwr1.source, fwr1.receiver, [fwr1.times/2, fwr1.times], waveforms=[fwr1.waveform, fwr1.waveform])

import torch
res  = torch.tensor([4.000e+02, 4.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 8.000e+02, 
                    8.000e+02, 8.000e+02, 8.000e+02, 8.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 5.000e+02, 
                    5.000e+02, 3.000e+02, 3.000e+02, 3.000e+02], dtype=torch.double)
m0   = torch.tensor([2.000e+02, 2.000e+02, 2.000e+02, 2.000e+02, 4.000e+02, 4.000e+02, 4.000e+02, 5.000e+02, 
                    5.000e+02, 5.000e+02, 5.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02, 
                    9.000e+02, 9.000e+02, 9.000e+02, 9.000e+02], dtype=torch.double)*1e-3
tau  = torch.tensor([1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 2.000e-04, 2.000e-04, 
                    2.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 3.000e-04, 1.000e-04, 1.000e-04, 1.000e-04, 
                    1.000e-04, 1.000e-04, 1.000e-04, 1.000e-04], dtype=torch.double)
chg  = torch.tensor([6.000e-01, 6.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 5.000e-01, 6.000e-01, 6.000e-01,
                        6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 6.000e-01, 9.000e-01, 
                        9.000e-01, 9.000e-01, 9.000e-01, 9.000e-01], dtype=torch.double)
thick=torch.tensor([1.000e+00, 1.177e+00, 1.386e+00, 1.632e+00, 1.922e+00, 2.263e+00, 2.665e+00, 3.138e+00, 
                    3.695e+00, 4.351e+00, 5.124e+00, 6.034e+00, 7.105e+00, 8.366e+00, 9.851e+00, 1.160e+01, 
                    1.365e+01, 1.608e+01, 1.893e+01], dtype=torch.double)


dbdt = fwr(res, m0, tau, chg, thick)
import matplotlib.pyplot as plt
import numpy as np

fig,axs=plt.subplots()
axs.loglog(fwr.times[0], fwr.lm)
axs.loglog(fwr.times[1], fwr.hm*2)
plt.show()

#DualMoment.