from anemone.forwards import Forward
from anemone.system import Loop, Receiver
import torch

import cProfile
import pstats
import numpy as np




profiler = cProfile.Profile()
profiler.enable()

tx = Loop.square(40.)
rx = Receiver(x=np.linspace(-1.,1.,51), y=np.linspace(-1.,1.,51), z=0., grid=True)
rx.weighting = np.ones_like(rx.X.flatten())
rx.weighting /= len(rx.weighting)

times = torch.logspace(-6, -3, 32)


fwr = Forward(tx, rx, times)
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime').print_stats(20)  # Print top 20 by cumulative time
