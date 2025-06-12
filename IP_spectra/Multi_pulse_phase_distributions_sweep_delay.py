import numpy as np
import matplotlib.pyplot as plt


# shift_fac_all = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])+1 
shift_fac_all = np.array([0.1,0.3,0.5,0.7,0.9])+1 
# shift_fac_all = np.array([2,5,7]) 

colors = plt.cm.viridis(np.linspace(0, 1, len(shift_fac_all)))

fig2,ax2 = plt.subplots(3,1)

dt = 1e-7
frep = 2000
trep = 1/frep
tau=trep/20
ts1 = 0*tau
t=np.arange(0,trep-dt,dt)

pul1 = np.zeros(len(t))
pul1[(t >= ts1) & (t < ts1 + tau)] = 1
pul1=pul1/np.sum(pul1)
spect1 = np.fft.fft(pul1)/len(t)
freq = np.fft.fftfreq(len(t), d=dt)
freqs = freq[freq >= 0]
spect1 = spect1[freq >= 0]

phase0_1 = np.angle(spect1)
ph_inc_1 = 2*np.pi*freqs*(ts1+tau)
ph_1_shift = (phase0_1+ph_inc_1 + np.pi) % (2 * np.pi) - np.pi

ax2[0].plot(freqs,ph_1_shift,color = 'r',marker = '.')       

for i,shift_fac in enumerate(shift_fac_all):

    ts3 = shift_fac*tau

    pul3 = np.zeros(len(t))

    pul3[(t >= ts3) & (t < ts3 + tau)] = 1

    pul3=pul3/np.abs(np.sum(pul3))

    combo = pul1 + pul3

    # Compute the FFT
    spect3 = np.fft.fft(pul3)/len(t)

    # Keep only the positive frequencies

    spect3 = spect3[freq >= 0]


    spect_combo = np.fft.fft(combo)/len(t)
    spect_combo = spect_combo[freq >= 0]

    phase0_combo = np.angle(spect_combo)
 
    phase0_3 = np.angle(spect3)

    ph_inc_combo1 = 2*np.pi*freqs*(ts1+tau)

    ph_inc_combo3 = 2*np.pi*freqs*(ts3+tau)
    ph_inc_3 = 2*np.pi*freqs*(ts3+tau)

    ph_combo1_shift = (phase0_combo+ph_inc_combo1 + np.pi) % (2 * np.pi) - np.pi

    ph_combo3_shift = (phase0_combo+ph_inc_combo3 + np.pi) % (2 * np.pi) - np.pi
    ph_3_shift = (phase0_3+ph_inc_3 + np.pi) % (2 * np.pi) - np.pi

    ax2[0].plot(freqs,ph_combo1_shift,marker = '.',color = colors[i])
    ax2[0].set_xlabel('Frequency [Hz]')
    ax2[0].set_ylabel('Phase [rad]')

    if i == 0 :
        ax2[1].plot(freqs,ph_3_shift,marker = '.',color = 'r')

    ax2[1].plot(freqs,ph_combo3_shift,marker = '.',color = colors[i])
    ax2[1].set_xlabel('Frequency [Hz]')
    ax2[1].set_ylabel('Phase [rad]')

    ax2[2].plot(t, combo, color = colors[i])
    ax2[2].set_xlabel('Time [s]')
    ax2[2].set_ylabel('Amplitude')

plt.show()
