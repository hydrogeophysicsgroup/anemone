"""
1D Forward Simulation for a Single Sounding
===========================================

Here we use the module *SimPEG.electromangetics.time_domain_1d* to predict
the stepoff response for a single sounding over a 1D layered Earth.
In this tutorial, we focus on the following:

    - Defining receivers, waveforms, sources and the survey
    - The units of the model and resulting data
    - Defining and running the 1D simulation for a single sounding

Our survey geometry consists of a horizontal loop source with a radius of 6 m
located 20 m above the Earth's surface. The receiver is located at the centre
of the loop and measures the vertical component of the response.


"""

#####################################################
# Import Modules
# --------------
#

import numpy as np
import os
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.utils import plot_1d_layer_model

write_output = False
plt.rcParams.update({"font.size": 16})

# sphinx_gallery_thumbnail_number = 2

#####################################################################
# Create Survey
# -------------
#
# Here we demonstrate a general way to define the receivers, sources, waveforms and survey.
# For this tutorial, the source is a horizontal loop whose current waveform
# is a unit step-off. The receiver measures the vertical component of the magnetic flux
# density at the loop's center.
#

# Source properties
source_location = np.array([0.0, 0.0, 20.0])
source_orientation = "z"  # "x", "y" or "z"
source_current = 1.0  # maximum on-time current
source_radius = 6.0  # source loop radius

# Receiver properties
receiver_location = np.array([0.0, 0.0, 20.0])
receiver_orientation = "z"  # "x", "y" or "z"
times = np.logspace(-5, -2, 31)  # time channels (s)

# Define receiver list. In our case, we have only a single receiver for each source.
# When simulating the response for multiple component and/or field orientations,
# the list consists of multiple receiver objects.
receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxDensity(
        receiver_location, times, orientation=receiver_orientation
    )
)

# Define the source waveform. Here we define a unit step-off. The definition of
# other waveform types is covered in a separate tutorial.
waveform = tdem.sources.StepOffWaveform()

# Define source list. In our case, we have only a single source.
source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=waveform,
        current=source_current,
        radius=source_radius,
    )
]

# Define the survey
survey = tdem.Survey(source_list)


###############################################
# Defining a 1D Layered Earth Model
# ---------------------------------
#
# Here, we define the layer thicknesses and electrical conductivities for our
# 1D simulation. If we have N layers, we define N electrical conductivity
# values and N-1 layer thicknesses. The lowest layer is assumed to extend to
# infinity. If the Earth is a halfspace, the thicknesses can be defined by
# an empty array, and the physical property values by an array of length 1.
#
# In this case, we have a more conductive layer within a background halfspace.
# This can be defined as a 3 layered Earth model.
#

# Physical properties
background_conductivity = 1e-1
layer_conductivity = 1e0

# Layer thicknesses
thicknesses = np.array([40.0, 40.0])
n_layer = len(thicknesses) + 1

# physical property models
model = background_conductivity * np.ones(n_layer)
model[1] = layer_conductivity

# Define a mapping for conductivities
model_mapping = maps.IdentityMap(nP=n_layer)

# Plot conductivity model
thicknesses_for_plotting = np.r_[thicknesses, 40.0]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plot_1d_layer_model(thicknesses_for_plotting, model, ax=ax, show_layers=False)
plt.gca().invert_yaxis()


#######################################################################
# Define the Forward Simulation, Predict Data and Plot
# ----------------------------------------------------
#
# Here we define the simulation and predict the 1D TDEM sounding data.
# The simulation requires the user define the survey, the layer thicknesses
# and a mapping from the model to the conductivities of the layers.
#
# When using the *SimPEG.electromagnetics.time_domain_1d* module,
# predicted data are organized by source, then by receiver, then by time channel.
#

# Define the simulation


# import cProfile
# cProfile.run('simulation.dpred(model)', 'restats')
# import pstats
# from pstats import SortKey
# p = pstats.Stats('restats')
# p.sort_stats(SortKey.CUMULATIVE)
# p.print_stats()

# Predict data for a given model
from time import perf_counter as timer
bt = timer()
simulation = tdem.Simulation1DLayered(
    survey=survey,
    thicknesses=thicknesses,
    sigmaMap=model_mapping,
)
dpred = simulation.dpred(model)
print(f'simpeg single forward time: {timer()-bt} s')



from em1d.physics import mu0
from em1d.experiment import Loop, Receiver
from em1d.transforms import *
from em1d.physics import refl_transverse_electric as rte, freespace_u0 as u0

# first lets compute Hz(R)
# side = np.sqrt(np.pi*source_radius**2)
side = 2*np.pi*source_radius / 4.
print(side)

lp = Loop.square(side)


rx = Receiver()
rmin = lp.min_horizontal_distance(rx)
rmax = lp.max_horizontal_distance(rx)


sigma = np.array([0.1, 1, 0.1])[:,None,None]
thick = np.array([40, 40])[:,None,None]


from scipy.interpolate import interp1d

hkl = defHankel1

r,_ = hkl.range(rmin, rmax)
s = lp.s_of_r(rx, r)




f_t = defFourier

def frq_integrand(omega):
    omega = omega[None,:,None]
    def integrand(wave_length):
        lmb = wave_length[None, None, :]
        tmp_u0 = u0(omega, lmb)
        intg = (np.exp(-tmp_u0*40.)*rte(sigma, thick, omega, lmb))*lmb**2 / tmp_u0
        return np.squeeze(intg/(1j*omega))
    hkl.func = integrand
    hkl.F = integrand(hkl.lamda[hkl.subset])
    #hr = hkl(6)
    
    r, hr = hkl.range(rmin, rmax)
    hr /= r[:, None]

    # hr_int = interp1d(r, hr.T, kind='cubic', fill_value='extrapolate')
    
    # print(r)
    # # 
    # s = np.zeros_like(r)
    # s[2] = 0.25*np.pi
    # s[3] = 1.75*np.pi

    res = hr.T.dot(s)
    return res

    #return hkl(6)*2*np.pi
    #return hr_int(6)*2*np.pi#*6


from time import perf_counter as timer
bt = timer()
# f_t = Fourier(frq_integrand, smp_per_decade=10)
f_t.func = frq_integrand
f_t.F = frq_integrand(f_t.lamda[f_t.subset])*np.sqrt(f_t.lamda[f_t.subset])
t,dbdt = f_t.range(1e-5, 1e-2)
print(f'dipole default time: {timer()-bt} s')




# Plot sounding
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.2, 0.15, 0.75, 0.78])
ax.loglog(times, dpred, "k-o", lw=2)
ax.loglog(t, np.abs(dbdt.real)/795774.71545947673925, 'r.-')
ax.set_xlabel("Times (s)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Magnetic Flux")

plt.show()
