import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from six.moves import cPickle as pickle
plt.ion()
kb = 1.38064852e-23

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
    
#Load data calculated and measured from MMBv2p
dat = load_dict('Pseudo_Noise_Nonlinear_Tones_Calc.pkl') 

#Choose 10 randomly selected tones from MMBv2p to look at fine resolution pseudo noise around them.
tone_inds = np.random.randint(low=0,high=len(dat['input_tone_f0s']),size =1)  
bws = [1000.,5000.,10000.,50000.,100000.]
#noise_bw_step = 5000.
alpha_ind = 0
for n,noise_bw_step in enumerate(bws):
	for ind in tone_inds:
		nl_tone_closest_ind = np.argmin(np.abs(dat['freqs_nl']-dat['input_tone_f0s'][ind]))
		nl_freq_near_tone = dat['freqs_nl'][nl_tone_closest_ind-300000:nl_tone_closest_ind+300000] 
		nl_amp_near_tone = dat['amps_nl'][nl_tone_closest_ind-300000:nl_tone_closest_ind+300000]  
		pseudo_bw_tot = np.max(nl_freq_near_tone) - np.min(nl_freq_near_tone)
		len_freqs = int(round(pseudo_bw_tot/noise_bw_step))
		freqs_binned = np.zeros(len_freqs)
		min_freq = np.min(nl_freq_near_tone)
		freqs_binned = [min_freq + i*noise_bw_step for i in range(len_freqs+1)]
		d = np.digitize(nl_freq_near_tone,freqs_binned)
		amps = np.zeros(len_freqs)
		for fr in range(1,len(freqs_binned)): 
			amps[fr-1] = sum(nl_amp_near_tone[d==fr]) 
		#plt.figure()
		#plt.plot(nl_freq_near_tone,10*np.log10(nl_amp_near_tone/0.001),'o')
		#f = dat['input_tone_f0s'][ind]
		#plt.plot([f,f],[np.min(10*np.log10(nl_amp_near_tone/0.001))-10,np.max(10*np.log10(nl_amp_near_tone/0.001))+10],'--')
		#plt.ylim([np.min(10*np.log10(nl_amp_near_tone/0.001))-5,np.max(10*np.log10(nl_amp_near_tone/0.001))+5])
		#plt.xlabel('Frequency [Hz]') 
		#plt.ylabel('Amplitude of NL Tones [dBm]')
		#plt.title('Pseudo Tone Frequencies and Amplitude near Resonator at '+str(np.round(f/1e9,4))+' GHz')
		#plt.savefig('Pseudo_Tones_all_%i_Hz.png'%int(np.round(f,0)))
		#plt.close()
		
		plt.figure(ind)
		plt.plot(freqs_binned[0:-1],(amps/noise_bw_step)/kb,'o',label = str(int(noise_bw_step/1000))+' kHz binning')#,alpha = (1-alpha_ind*0.1))
		if n == 0:
			f = dat['input_tone_f0s'][ind]
			plt.plot([f,f],[np.min((amps/noise_bw_step)/kb)-100,np.max((amps/noise_bw_step)/kb)+100],'--')
			plt.ylim([np.min((amps/noise_bw_step)/kb)-50,np.max((amps/noise_bw_step)/kb)+50])
			plt.xlabel('Frequency [Hz]',fontsize = 12) 
			plt.ylabel('NL Tones Binned + converted to $T_{Noise}$',fontsize = 12)
			plt.title('1kHz Binned Pseudo Noise near Resonator at '+str(np.round(f/1e9,4))+' GHz',fontsize = 12)
		if n == len(bws)-1:
			plt.legend(fontsize = 12)
			plt.savefig('Pseudo_Noise_1kHz_Bins_%i_Hz_index_%i.png'%(int(np.round(f,0)),ind))
			plt.close()
	alpha_ind = alpha_ind+1
