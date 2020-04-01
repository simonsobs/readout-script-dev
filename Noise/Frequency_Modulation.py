import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#initialize some constants
pi = np.pi
Z_0 = 50

#PSD w/ appropriate normalization to convert from a timestream to dBm
def time_to_dBm(V_in):
	FT = np.fft.fft(V_in)
	FT_norm = 2*np.abs(FT)/len(V_in)
	PS_dBm = 10*np.log10((FT_norm/np.sqrt(2.))**2./Z_0/.001)
	return PS_dBm

#Initialize time and frequnecy arrays
t = np.arange(0,3e-3,1/10e9)
f = np.fft.fftfreq(len(t), d = t[1]-t[0])

#Initialize amplitude of primary carrier
P_in = -70 #dBm
P_W_rms = .001*10**(P_in/10)
V_rms = np.sqrt(P_W_rms*Z_0)
V_pk = np.sqrt(2)*V_rms

#You can uncomment this if you want to plot the unmodulated carrier
'''
V_base = V_pk*np.cos(2*pi*6e9*t)
PS_dBm_base = time_to_dBm(V_base)
plt.figure
plt.subplot(211)
plt.plot(t,V_base)
plt.subplot(212)
plt.plot(f,PS_dBm_base)
'''

#Initialize frequency modulation (i.e. flux ramp tone tracking), 3 cases perfect since wave and two curves w/ lambda = 0.3 and 0.6
V_mod = {}
rand_phase = 0 #Can add some phase to the modulation function here
V_mod[1] = (50e3/20e3)*np.cos(2*pi*20e3*t+rand_phase)
lam = 0.3
Max_Deviation_Hz = 100e3
Amp = 100e3/((lam/(1+lam))-(-lam/(1-lam)))
V_mod[2] = (Amp/20e3)*lam*(np.cos(2*pi*20e3*t+rand_phase))/(1.+lam*np.cos(2*pi*20e3*t+rand_phase))
vlam = 0.6
Max_Deviation_Hz = 100e3
Amp = 100e3/((lam/(1+lam))-(-lam/(1-lam)))
V_mod[3] = (Amp/20e3)*lam*(np.cos(2*pi*20e3*t+rand_phase))/(1.+lam*np.cos(2*pi*20e3*t+rand_phase))

alpha_step = 1
for key in V_mod.keys():
	#Apply frequency modulation to carrier w/ random phase added to the carrier.
	V_car = V_pk*np.cos(2*pi*(4e9+np.random.rand()*1e3)*t+V_mod[key])
	PS_dBm_car = time_to_dBm(V_car)

	#Plot only the positive frequency parts of the PSD
	plt.figure(1)
	plt.plot(f[0:int(len(f)/2)-1],PS_dBm_car[0:int(len(f)/2)-1],alpha = alpha_step)
	alpha_step = alpha_step*0.8
	#Can adjust plot limits + style here
	plt.xlim([4e9-4e5,4e9+4e5]) 
	plt.ylim([-170,-70])
	plt.yticks(np.arange(-170,-70,10))
	plt.grid()




