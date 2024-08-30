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

from SimPEG import discretize


import numpy as np
import os
from matplotlib import pyplot as plt

from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem
from SimPEG.utils import plot_1d_layer_model

write_output = False
plt.rcParams.update({"font.size": 12})

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
receiver_location = np.array([0.0, 0., 20.0])
receiver_orientation = "z"  # "x", "y" or "z"
times = np.logspace(-5, -2, 31)  # time channels (s)

nearthick = np.logspace(-1, 1, 5)
deepthick = np.logspace(1, 2, 10)
hx = np.r_[nearthick, deepthick]
mesh1D = discretize.TensorMesh([hx], [0.])


receiver_list = []
receiver_list.append(
    tdem.receivers.PointMagneticFluxTimeDerivative(
        receiver_location, times, orientation=receiver_orientation
    )
)

# Define the source waveform. Here we define a unit step-off. The definition of
# other waveform types is covered in a separate tutorial.
waveform = tdem.sources.StepOffWaveform()

#Define source list. In our case, we have only a single source.

source_list = [
    tdem.sources.CircularLoop(
        receiver_list=receiver_list,
        location=source_location,
        waveform=waveform,
        current=source_current,
        radius=source_radius,
    )
]

from anemone.experiment import *

# lp = Loop.square(6, z=20)
# source_list = [
#     tdem.sources.segmented_line_current_source_term(mesh1D, 
#         lp.xyz,
#         # waveform=waveform,
#         current=source_current,)
# ]

# Define the survey
survey = tdem.Survey(source_list)#, receivers=receiver_list)

# Emerald forward setup...
# tdem.sources.MagDipole(
#                 [tdem.receivers.PointMagneticFluxTimeDerivative(
#                     receiver_location, times[0], self.rx_orientation)],
#                 location=location,
#                 waveform=waveform_lm,
#                 orientation=self.tx_orientation,
#                 i_sounding=idx),

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

simulation = tdem.Simulation1DLayered(
    survey=survey,
    thicknesses=thicknesses,
    sigmaMap=model_mapping,
)


n_forwards=10
from time import perf_counter as timer
bt = timer()
for i in range(n_forwards):
    dpred = simulation.dpred(model)
#dpred = simulation.dpred(model)
print(f'simpeg single forward time: {(timer()-bt)/n_forwards} s')


import torch

from anemone.experiment import Loop, Receiver, Forward
side = np.sqrt(np.pi*6**2)
lp = Loop.square(side, z=20)
rx = Receiver(z=20)

sigma = torch.tensor([0.1, 1, 0.1])
thick = torch.tensor([40, 40])

class Forward:
    def __init__(self, source, receiver, times):
        self.source = source
        self.receiver = receiver
        self.times = times if isinstance(times, torch.Tensor) else torch.tensor(times)
        self.rmin = self.source.min_horizontal_distance(self.receiver).to(times.device)
        self.rmax = self.source.max_horizontal_distance(self.receiver).to(times.device)
        self.srcz = torch.tensor([source.z[0]], dtype=torch.float64)
        self.rz = self.receiver.z
        self.pfield_dz = torch.abs(self.srcz-self.rz)
        self.sfield_dz = torch.abs(self.srcz)+torch.abs(self.rz)

        self.tmin = self.times.min()
        self.tmax = self.times.max()

        self.device = times.device

    def forward(self):
        def frq_integrand(f):
            omega = c.twopi*f[None,:,None]

            def wvlngth_integrand(wave_length):
                """ Hz from horizontal electric dipole. """
                lmb = wave_length[None, None, :]
                tmp_u0 = u0(omega, lmb)
                # intg = torch.exp(-tmp_u0*self.pfield_dz)+
                intg = rte(self.sigma, self.thick, omega, lmb)*torch.exp(-tmp_u0*self.sfield_dz)
                intg *= lmb**2 / tmp_u0
                return torch.squeeze(intg/c.fourpi)
            hkl1 = Hankel(wvlngth_integrand, 1, smp_per_decade=6, range=(self.rmin, self.rmax), device=self.device, tolerance=1e-4)#, tolerance=1e-4)
            return integrate_around_loop(self.source, self.receiver, hkl1)
        
        ift=invFourier(frq_integrand, smp_per_decade=6, range=(self.tmin, self.tmax), device=self.device)#, tolerance=1e-6)
        return ift(self.times)*c.mu0

    def __call__(self, sigma, thickness):
        self.sigma = sigma[...,None,None] + 0j#torch.tensor(sigma,     device=self.device, dtype=torch.complex128)[...,None,None]
        self.thick = thick[...,None,None] + 0j#torch.tensor(thickness, device=self.device, dtype=torch.complex128)[...,None,None]
        return self.forward()
        

fwr = Forward(lp, rx, torch.tensor(times))


n_forwards = 200
t = times

dbdt = fwr(sigma, thick)

sigma = sigma[:, None] + torch.arange(n_forwards)[None,:]
thick = thick[:, None] + torch.arange(n_forwards)[None,:]

from time import perf_counter as timer
bt = timer()
dbdt = fwr(sigma, thick)    
print(f'pytorch default time: {(timer()-bt)/n_forwards} s')




# Plot sounding
fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.2, 0.15, 0.75, 0.78])
ax.loglog(times, np.abs(dpred), "k-o", lw=2, label='simpeg')
ax.loglog(t, np.abs(dbdt[0].numpy().real), 'r.-', label='pytorch')
ax.set_xlabel("Times (s)")
ax.set_ylabel("|H| (A/m)")
ax.set_title("Magnetic Flux")
ax.legend()
plt.show()
