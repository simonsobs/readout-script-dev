import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
plt.ion()
kb = 1.38064852e-23  

df_mmb = pd.read_pickle('/home/arnoldlabws3/Scratch/MSF/Personal_Laptop_Docs/SO/MMBv2p/MMBv2_20190918-mccarrick-mmbv2pp_resonator_and_noise_resonator-params.pkl.pkl')

def pseudo_tone_loc(f1,f2,f3):
	return (f1+f2-f3)
	
idx_not_nan = np.logical_not(np.isnan(df_mmb['br']))
f0s = np.asarray(df_mmb['f0'][idx_not_nan])
brs = np.asarray(df_mmb['br'][idx_not_nan])
Qs = np.asarray(df_mmb['Q'][idx_not_nan])
Qcs = np.asarray(df_mmb['Qc'][idx_not_nan])

QoverQc = Qs/Qcs
N_reasonable_resonators = len(np.where(QoverQc<1.0)[0])
idx_reasonable = np.where(QoverQc<1.0)[0]

Qs_reas = Qs[idx_reasonable]
Qcs_reas = Qcs[idx_reasonable]
f0s_reas = f0s[idx_reasonable]
brs_reas = brs[idx_reasonable]

Dips = 1. - (Qs_reas/Qcs_reas)
Dips_db = 20*np.log10(Dips)

Amps_dBm = -70.*np.ones(len(f0s_reas)) + Dips_db
Amps = 0.001*10**(Amps_dBm/10.)

f0s_reas = f0s_reas[np.where(Dips_db<-3.0)]
Amps_dBm = Amps_dBm[np.where(Dips_db<-3.0)]

N = len(f0s_reas)
nl_freqs = np.zeros(int((N-1)*N**2/2))
nl_amps = np.zeros(int((N-1)*N**2/2))
inl = 0

for i1 in range(len(f0s_reas)):
	for i2 in range(len(f0s_reas)):
		if i2 < i1:
			continue
		for i3 in range(len(f0s_reas)):
			if i1 == i2 and i2 == i3:
				continue
			if i2 == i3 or i1 == i3:
				continue
			else:
				nl_freqs[inl] = pseudo_tone_loc(f0s_reas[i1],f0s_reas[i2],f0s_reas[i3])
				nl_amps[inl] = (Amps_dBm[i1]+Amps_dBm[i2]+Amps_dBm[i3])+100.-39.
				inl = inl+1
		
noise_bw_step = 100000.
pseudo_bw_tot = np.max(nl_freqs) - np.min(nl_freqs)
min_freq = np.min(nl_freqs)

len_freqs = int(round(pseudo_bw_tot/noise_bw_step)) 
freqs_binned = np.zeros(len_freqs)

nl_freqs_sort_ind = np.argsort(nl_freqs)  
nl_freqs_sort = nl_freqs[nl_freqs_sort_ind]
nl_amps_lin = 0.001*10**(nl_amps/10)         
nl_amps_lin_sort = nl_amps_lin[nl_freqs_sort_ind]



freqs_binned = [min_freq + i*noise_bw_step for i in range(len_freqs+1)]
d = np.digitize(nl_freqs_sort,freqs_binned) 
amps = np.zeros(len_freqs)
for fr in range(1,len(freqs_binned)):
#for i in range(len(amps)):
	#indmin = np.argmin(np.abs(nl_freqs_sort-freqs_binned[i]))
	#indmax = np.argmin(np.abs(nl_freqs_sort-freqs_binned[i+1]))
	#amps[i] = sum(nl_amps_lin_sort[indmin:indmax]) 
	amps[fr-1] = sum(nl_amps_lin_sort[d==fr])

plt.plot(freqs_binned[0:-1],(amps/noise_bw_step)/kb,'o')
for fs in f0s_reas:
	plt.plot([fs,fs],[np.min((amps/noise_bw_step)/kb)-100,np.max((amps/noise_bw_step)/kb)+100],'c-')
plt.ylim([np.min((amps/noise_bw_step)/kb)-50,np.max((amps/noise_bw_step)/kb)+50])



			

