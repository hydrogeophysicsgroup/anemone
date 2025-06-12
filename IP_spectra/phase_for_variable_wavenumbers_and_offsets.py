import numpy as np
import matplotlib.pyplot as plt


f = np.logspace(3, 6, 51)
r = np.linspace(1, 100, 10)


mu = 4*np.pi*1e-7
w=2*np.pi*f
res = 1
k = np.sqrt(1j*w*mu/res)

plt.figure(figsize=(10, 8))

for i,rtmp in enumerate(r):

    pha = 1j*np.outer(k, rtmp)
    color = plt.cm.viridis(i / len(r))
    plt.plot(np.exp(pha).real, np.exp(pha).imag, color=color)
    print(i,rtmp)

res = 10
k = np.sqrt(1j*w*mu/res)

for i,rtmp in enumerate(r):

    pha = 1j*np.outer(k, rtmp)
    color = plt.cm.viridis(i / len(r))
    plt.plot(np.exp(pha).real, np.exp(pha).imag, color=color)
    print(i,rtmp)

res = 100
k = np.sqrt(1j*w*mu/res)

for i,rtmp in enumerate(r):

    pha = 1j*np.outer(k, rtmp)
    color = plt.cm.viridis(i / len(r))
    plt.plot(np.exp(pha).real, np.exp(pha).imag, color=color)
    print(i,rtmp)


plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imag')
plt.title('Complex Plane - exp(-1i*k*r) - r = 1-100 for all permuatations of  res = [1,10,100]')
# Display the plot
plt.show()