import numpy as np
import torch
from .. import constants as c
from ..system import *


def compute_relstd(t_gates, data_val, noise_slope_exponent=-0.5, noise_level_at_1ms=1e-8, current=1, relative_noise=0.03):
    b = noise_level_at_1ms/(1e-3**noise_slope_exponent)
    abs_noise = b*(t_gates**noise_slope_exponent)
    y = data_val*current
    std_rel = abs_noise/y.T
    relstd_tot = torch.sqrt(std_rel**2 + relative_noise**2)
    return relstd_tot.to(data_val.device).T

