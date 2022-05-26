import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sig
import pandas as pd
import os
import Pseudo_Noise_Time_Domain as PNF
import Noise_Referral as nr
plt.ion()

main_dir = '/home/arnoldlabws3/Scratch/MSF/Personal_Laptop_Docs/SO/'
list_dirs = os.listdir(main_dir)
P_feed = -57
T_acquire = 1e-4
alpha_step = 1.0

out_dict = {}


Noise = {}
lossin, Tavgin = nr.read_loss_in('/home/arnoldlabws3/Scratch/MSF/readout-script-dev/Noise/')
lossout, Tavgout = nr.read_loss_out('/home/arnoldlabws3/Scratch/MSF/readout-script-dev/Noise/')
LNA_TNtyp= [2.1,40.,191.]
T_HEMT = nr.Amp_Chain_Noise_Temp(loss_out = lossout, Tavg_out = Tavgout, LNA_TN=LNA_TNtyp)

Dips_sweep = [-3,-6,-9]
Dips_scatter = [0,0,0]
N_tones = [500,1000]
BW = 150e3
BW_scat = 0
sep_scat = 0

out_dict = {}
Noise = {}

Tn_Avg_fake_combs = [180,500,30,150,7,35]
i = 0
for a in range(len(Dips_sweep)):
	out_dict[Dips_sweep[a]] = {}
	Noise[Dips_sweep[a]] = {}
	for b in range(len(N_tones)):
		print(Dips_sweep[a],N_tones[b])
		Noise[Dips_sweep[a]][N_tones[b]] = {}
		if N_tones[b] == 500:
			f_start = 5e9
			f_stop = 6e9
		if N_tones[b] == 1000:
			f_start = 4e9
			f_stop = 6e9
		
		Qr, Qc, f_res, t, f, V_wave_in, V_wave_out = PNF.Gen_Fake_Comb(P_dBm = P_feed,f_start=f_start,f_stop=f_stop,N_Tones = N_tones[b],separation_scatter= sep_scat,BW=BW,BW_scatter = BW_scat,Depth = Dips_sweep[a],Depth_Scatter = Dips_scatter[a],T = T_acquire)
		
		out_dict[Dips_sweep[a]][N_tones[b]] = PNF.Calc_Non_Linearities(V_wave_in, V_wave_out,t, f,f_res,40,-30,plot_all = False,plot_noise_temp=True,label='N Tones: '+str(N_tones[b])+' and Depth: '+str(Dips_sweep[a]),alpha = alpha_step,fignum = 1)
		alpha_step=alpha_step*0.9
		Noise[Dips_sweep[a]][N_tones[b]]['f0s'] = f_res
		Noise[Dips_sweep[a]][N_tones[b]]['Q'] = Qr
		Noise[Dips_sweep[a]][N_tones[b]]['Qc'] = Qc
		Noise[Dips_sweep[a]][N_tones[b]]['PNF'] = np.zeros(len(f_res))
		Noise[Dips_sweep[a]][N_tones[b]]['DAC'] = np.zeros(len(f_res))
		Noise[Dips_sweep[a]][N_tones[b]]['HEMT'] = np.zeros(len(f_res))
		
		for j in range(len(f_res)):
			Noise[Dips_sweep[a]][N_tones[b]]['PNF'][j] = nr.TN_to_NEI(Tn_Avg_fake_combs[i],P_feed,Qr[j],Qc[j],f_res[j],dfdI=2.06e-2,p=False)
			
			T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_feed+32,Qr[j],Qc[j],f_res[j],dfdI = 2.06e-2)
			
			Noise[Dips_sweep[a]][N_tones[b]]['DAC'][j] = nr.TN_to_NEI(T_DAC,P_feed,Qr[j],Qc[j],f_res[j],dfdI=2.06e-2,p=False)
			
			Noise[Dips_sweep[a]][N_tones[b]]['HEMT'][j] = nr.TN_to_NEI(T_HEMT,P_feed,Qr[j],Qc[j],f_res[j],dfdI=2.06e-2,p=False)
		i = i+1
		print('Median PNF:',np.median(Noise[Dips_sweep[a]][N_tones[b]]['PNF']),'Mean PNF:',np.mean(Noise[Dips_sweep[a]][N_tones[b]]['PNF']),'Median DAC:',np.median(Noise[Dips_sweep[a]][N_tones[b]]['DAC']),'Mean DAC:',np.mean(Noise[Dips_sweep[a]][N_tones[b]]['DAC']),'Median HEMT:',np.median(Noise[Dips_sweep[a]][N_tones[b]]['HEMT']),'Mean HEMT:',np.median(Noise[Dips_sweep[a]][N_tones[b]]['HEMT']))
		
plt.close()

'''
alpha_step = 1.0
for key1 in out_dict.keys():
	for key2 in out_dict[key1].keys():
		plt.semilogy(out_dict[key1][key2]['ADC']['f_WperHZ'],sig.savgol_filter(out_dict[key1][key2]['ADC']['K'],293,2),alpha = alpha_step)
		alpha_step = alpha_step*0.95
'''

Tn_Avg_Assemblies = [7,30,150,7,7]
for i, dirname in enumerate(list_dirs):
	print(dirname)
	pkl_file = main_dir+dirname+'/pickle/'+os.listdir(main_dir+dirname+'/pickle')[0]
	
	f_res, t, f, V_wave_in, V_wave_out = PNF.Gen_V_Wave_From_Assemlby(pkl_file,P_feed,T_acquire)
	out_dict[dirname] = {}
	out_dict[dirname]['fres'] = f_res
	out_dict[dirname] = PNF.Calc_Non_Linearities(V_wave_in, V_wave_out,t, f,f_res,40,-30,plot_all = False,plot_noise_temp=True,label=dirname,alpha = alpha_step,fignum=2)
	alpha_step=alpha_step*0.9
	
	
	Noise[dirname] = {}
	
	df_mmb = pd.read_pickle(pkl_file)
		
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
	
	Dips = 1-(Qs_reas/Qcs_reas)
	Dips_dB = 20*np.log10(Dips)
	
	Noise[dirname]['PNF'] = np.zeros(len(f0s_reas))
	Noise[dirname]['HEMT'] = np.zeros(len(f0s_reas))
	Noise[dirname]['DAC'] = np.zeros(len(f0s_reas))
	
	for j in range(len(f0s_reas)):
		Noise[dirname]['PNF'][j] = nr.TN_to_NEI(Tn_Avg_Assemblies[i],P_feed,Qs_reas[j],Qcs_reas[j],f0s_reas[j],dfdI=2.06e-2,p=False)
		
		T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_feed+32,Qs_reas[j],Qcs_reas[j],f0s_reas[j],dfdI = 2.06e-2)
		
		Noise[dirname]['DAC'][j] = nr.TN_to_NEI(T_DAC,P_feed,Qs_reas[j],Qcs_reas[j],f0s_reas[j],dfdI=2.06e-2,p=False)
		
		Noise[dirname]['HEMT'][j] = nr.TN_to_NEI(T_HEMT,P_feed,Qs_reas[j],Qcs_reas[j],f0s_reas[j],dfdI=2.06e-2,p=False)
		
	
	print('Median Q:',np.median(Qs_reas)/1000,'Depth:',np.median(Dips_dB),'N Tones:',len(f0s_reas))
	print('Median PNF:',np.median(Noise[dirname]['PNF']),'Mean PNF:',np.mean(Noise[dirname]['PNF']),'Median DAC:',np.median(Noise[dirname]['DAC']),'Mean DAC:',np.mean(Noise[dirname]['DAC']),'Median HEMT:',np.median(Noise[dirname]['HEMT']),'Mean HEMT:',np.mean(Noise[dirname]['HEMT']))
