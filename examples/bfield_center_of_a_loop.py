
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from biotsavart import mu0

one_tesla_in_amp_per_m = 795774.7


res = 10.
sigma = 1/res
ts = np.logspace(-6, -1, 51)


rtpi = np.sqrt(np.pi)
theta = np.sqrt(0.25*mu0*sigma / ts)

loop_radius =  5.
#a = loop_radius
a = np.sqrt(1./np.pi)
hz = (3.*np.exp(-theta**2 * a**2)/(rtpi*theta*a) + ( 1.-3/(2*theta**2*a**2))*erf(theta*a))


plt.figure()
plt.loglog(ts, hz.real / one_tesla_in_amp_per_m)
plt.show()




ts = 1e-5
theta = np.sqrt(0.25*mu0*sigma / ts)

hz = (3.*np.exp(-theta**2 * a**2)/(rtpi*theta*a) + ( 1.-3/(2*theta**2*a**2))*erf(theta*a))
print(f'hz in A/m: {hz / one_tesla_in_amp_per_m}')