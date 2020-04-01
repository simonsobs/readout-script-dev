import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from six.moves import cPickle as pickle
plt.ion()
kb = 1.38064852e-23
pi = np.pi

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
    
#Load data calculated and measured from MMBv2p
dat = load_dict('Pseudo_Noise_Nonlinear_Tones_Calc.pkl') 

ind = int(np.round(len(dat['input_tone_f0s'])/2)) 
N_sum = [10,20,30,40,50,60]#[10,100,1000,10000,100000]

#V_wave = np.zeros((len(t), len(N_sum)))
C = np.zeros((len(N_sum),10))
Vpeak = np.zeros((len(N_sum),10))
Cavg = np.zeros(len(N_sum))
Vpkavg = np.zeros(len(N_sum))
for j, N in enumerate(N_sum):
	for navg in range(10):
#N = N_sum[0]
		Npm = int(np.round(N/2))
		nl_tone_closest_ind = np.argmin(np.abs(dat['freqs_nl']-dat['input_tone_f0s'][ind]))
		nl_freq_near_tone = dat['freqs_nl'][nl_tone_closest_ind-Npm:nl_tone_closest_ind+Npm]
		min_freq = np.round(np.min(nl_freq_near_tone),0) - 1e6
		nl_freq_near_tone = nl_freq_near_tone - min_freq
		samp_freq = 2*(np.round(np.max(nl_freq_near_tone),0) + 1e6)
		t = np.arange(0,3,1/samp_freq)
		nl_amp_near_tone = dat['amps_nl'][nl_tone_closest_ind-Npm:nl_tone_closest_ind+Npm]  
			#for i in range(len(nl_amp_near_tone)):
			#	V_wave[:,j] = V_wave[:,j]+nl_amp_near_tone[i]*np.cos(2*pi*(nl_freq_near_tone[i]*t+np.random.random()))
		phi = np.random.rand(len(nl_amp_near_tone))
		f_times_t = np.outer(t,nl_freq_near_tone)
		V_wave = np.sum(np.cos(2*pi*(f_times_t + phi))*nl_amp_near_tone,1)

		if j == len(N_sum) - 1:
			plt.figure()
			plt.plot(t,V_wave)#[:,0])
		print(np.max(V_wave),np.min(V_wave),np.sqrt(np.mean(V_wave**2)),np.max(V_wave)/np.sqrt(np.mean(V_wave**2)))
		C[j,navg] = np.max(V_wave)/np.sqrt(np.mean(V_wave**2))
		Vpeak[j,navg] = np.max(V_wave)
	Cavg[j] = np.mean(C[j,:])
	Vpkavg[j] = np.mean(Vpeak[j,:])
plt.figure(1)
plt.plot(N_sum,Cavg,'o-')
plt.title('Crest Factor Average')
plt.figure(2)
plt.plot(N_sum,Vpkavg/Cavg,'o-')
plt.title('$V_{rms}$ Average')

for i in range(len(N_sum)):
	plt.figure(3)
	plt.plot(C[i,:],label = 'N = '+str(N_sum[i]))
	plt.figure(4)
	plt.plot(Vpeak[i,:],label = 'N = '+str(N_sum[i]))
	plt.figure(5)
	plt.plot(Vpeak/C[i,:],label = 'N = '+str(N_sum[i]))
plt.figure(3)
plt.legend()
plt.title('Crest Factor 10 runs')
plt.figure(4)
plt.legend()
plt.title('$V_{peak}$ 10 runs')
plt.figure(5)
plt.legend()
plt.title('$V_{rms}$ 10 runs')

for j, N in enumerate(N_sum):
	Npm = int(np.round(N/2))
	nl_tone_closest_ind = np.argmin(np.abs(dat['freqs_nl']-dat['input_tone_f0s'][ind]))
	nl_freq_near_tone = dat['freqs_nl'][nl_tone_closest_ind-Npm:nl_tone_closest_ind+Npm]
	nl_amp_near_tone = dat['amps_nl'][nl_tone_closest_ind-Npm:nl_tone_closest_ind+Npm]  
	print(np.sum(nl_amp_near_tone)/
		
