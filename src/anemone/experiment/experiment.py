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

def plot_error_statistics(times, dbdt, dbdt_ref, ax=None, log_relerr=True, abs_color="C0", rel_color="C3", min_percentile=5., max_percentile=95., ok_relerr_lines=3.):
    if ax is None:
        from matplotlib.pyplot import subplots
        fig,ax = subplots()

    abserr = np.abs(dbdt - dbdt_ref)
    relerr = 100*(dbdt - dbdt_ref) / np.abs(dbdt_ref)

    ax2 = ax.twinx()

    if log_relerr:
        relerr = np.abs(relerr)

    ax.plot(times, relerr.mean(0), '--', color=rel_color)
    ax.plot(times, np.median(relerr, axis=0), '-', color=rel_color)
    lb = np.percentile(relerr, min_percentile, axis=0)
    ub = np.percentile(relerr, max_percentile, axis=0)
    ax.fill_between(times, lb, ub, color=rel_color, alpha=0.2)


    ax2.loglog(times, abserr.mean(0), '--', color=abs_color)
    ax2.loglog(times, np.median(abserr, axis=0), '-', color=abs_color)
    lb = np.percentile(abserr, min_percentile, axis=0)
    ub = np.percentile(abserr, max_percentile, axis=0)
    ax2.fill_between(times, lb, ub, color=abs_color, alpha=0.2)

    ax2.set_ylabel('Absolute Error [V/m$^2$]', color=abs_color)
    ax.set_ylabel('Relative Error [%]', color=rel_color)

    if log_relerr:
        ax.set_yscale('log')

    ax.grid(lw=0.5, which='both')
    if ok_relerr_lines is not None:
        ax.axhline( ok_relerr_lines, color='k', lw=2)
        ax.axhline(-ok_relerr_lines, color='k', lw=2)
    ax.set_xlabel('time [s]')

    return ax,ax2
