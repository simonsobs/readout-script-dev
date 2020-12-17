import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sig
import pandas as pd
import os
plt.ion()
kb = 1.38064852e-23
Z_0 = 50.
pi = np.pi

def dBm_to_Watts(dBm):
	'''
	Converts power in dBm to power in Watts.
	'''
	return .001*10**(dBm/10.)

def Watts_to_Vrms(W):
	'''
	Converts rms power to volts rms.
	'''
	return np.sqrt(W*Z_0)

def Vrms_to_Vpk(Vrms):
	'''
	Converts volts rms to amplitude/peak voltage.
	'''
	return Vrms*np.sqrt(2)

def nonlinear_model(vin,a1,a2,a3):
	'''
	Applies non-linear model.
	'''
	vo = a1*vin + a2*vin**2 - a3*vin**3
	for n in range(len(vo)):
		if vo[n] > 1.35/2:
			vo[n] = 1.35/2
		if vo[n] < -1.35/2:
			vo[n] = -1.35/2
	return vo

def Gen_V_Wave_From_Assemlby(pickle_file,P_feed_dBm,T):
	#Read in frequencies + Q's from MMB/Assembly
	df_mmb = pd.read_pickle(pickle_file)

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

	#Initialize amplitude into feedline
	P_dBm_feed = P_feed_dBm
	V_feed = Vrms_to_Vpk(Watts_to_Vrms(dBm_to_Watts(P_dBm_feed)))

	#Set time sampling + length
	res_BW_max = np.max(brs_reas)
	f_max = np.max(f0s_reas)

	#Initialize voltage time wave + comb
	t = np.arange(0,T,1.0/(6*f_max))
	f = np.fft.fftfreq(len(t),d=t[1]-t[0])

	V_wave_in = np.zeros(len(t))
	trans = np.ones(len(f))

	#Sum all tones in the time domain + generate comb in the frequency domain
	for i in range(len(f0s_reas)):
		V_wave_in = V_wave_in + V_feed*np.cos(2*pi*(f0s_reas[i]*t+np.random.rand()))
		xplus = (f-f0s_reas[i])/f0s_reas[i]
		xmin = (f+f0s_reas[i])/-f0s_reas[i]
		trans = trans - (Qs_reas[i]/Qcs_reas[i])/(1+2*1j*Qs_reas[i]*xplus) - (Qs_reas[i]/Qcs_reas[i])/(1+2*1j*Qs_reas[i]*xmin)

	#Fourier transform the input wave and pass it through the transfer function
	#Test that a function of known fft returns the fft you expect or if you need to normalize
	comb_in = np.fft.fft(V_wave_in)
	comb_out = comb_in*trans

	#Inverse fourier transform and apply the amplifier nonlinearity
	V_wave_out = np.fft.ifft(comb_out)
	f_res = f0s_reas
	return f_res, t, f, V_wave_in, V_wave_out

def Gen_IP3_Test_Wave(P_dBm,f1_GHz,f2_GHz,fs,T):
	V_feed = Vrms_to_Vpk(Watts_to_Vrms(dBm_to_Watts(P_dBm)))
	t = np.arange(0,T,1.0/fs)
	f = np.fft.fftfreq(len(t),d=t[1]-t[0])

	V_wave_in = V_feed*np.cos(2*pi*(f1_GHz*1e9*t))+V_feed*np.cos(2*pi*(f2_GHz*1e9*t))
	V_wave_out = V_wave_in
	return t, f, V_wave_in, V_wave_out

def Gen_Fake_Comb(P_dBm,f_start,f_stop,N_Tones,separation_scatter,BW,BW_scatter,Depth,Depth_Scatter,T):
	V_feed = Vrms_to_Vpk(Watts_to_Vrms(dBm_to_Watts(P_dBm)))
	t = np.arange(0,T,1.0/(6*f_stop))
	f = np.fft.fftfreq(len(t),d=t[1]-t[0])

	f0s = np.linspace(f_start,f_stop,num=N_Tones)+separation_scatter*np.random.random(N_Tones)*np.random.choice([1,-1],N_Tones)
	BWs = BW*np.ones(N_Tones)+BW_scatter*np.random.random(N_Tones)*np.random.choice([1,-1],N_Tones)
	Qrs = f0s/BWs
	Dips_dB = Depth*np.ones(N_Tones)+Depth_Scatter*np.random.random(N_Tones)*np.random.choice([1,-1],N_Tones)
	Dips_lin = 10**(Dips_dB/20)
	Qcs = Qrs/(1-Dips_lin)

	trans = np.ones(len(f))
	V_wave_in = np.zeros(len(t))
	for i in range(len(f0s)):
		V_wave_in = V_wave_in + V_feed*np.cos(2*pi*(f0s[i]*t+np.random.rand()))
		xplus = (f-f0s[i])/f0s[i]
		xmin = (f+f0s[i])/-f0s[i]
		trans = trans - (Qrs[i]/Qcs[i])/(1+2*1j*Qrs[i]*xplus) - (Qrs[i]/Qcs[i])/(1+2*1j*Qrs[i]*xmin)

	#Fourier transform the input wave and pass it through the transfer function
	#Test that a function of known fft returns the fft you expect or if you need to normalize
	comb_in = np.fft.fft(V_wave_in)
	comb_out = comb_in*trans

	#Inverse fourier transform and apply the amplifier nonlinearity
	V_wave_out = np.fft.ifft(comb_out)
	f_res = f0s

	return Qrs, Qcs, f_res, t, f, V_wave_in, V_wave_out

def Calc_Non_Linearities(V_wave_in,V_wave_out,t,f,fres,Gain,IIP3,plot_all = False,plot_noise_temp =True,label=None,alpha = None,fig_num=2):
	'''
	Slide 12 here: http://rfic.eecs.berkeley.edu/~niknejad/ee142_fa05lects/pdf/lect9.pdf
	np.sqrt(.001*10**(IIP3/10)*2*Z_0) = np.sqrt((4/3)*(a1/a3))
	(3/4)*.001*(10**(IIP3/10))*2*Z_0 = a1/a3
	a3 = a1/((3/4)*.001*(10**(IIP3/10))*2*Z_0)
	'''
	#Initialize nolinearity parameters for nonlinear model
	a1 = np.sqrt(10**(Gain/10))
	a2 = 0
	a3 = -a1/((3/4)*.001*(10**(IIP3/10))*2*Z_0)
	#Apply nonlinearity to wave out of comb
	V_wave_ADC = nonlinear_model(V_wave_out,a1,a2,a3)

	V_in_FFT = np.fft.fft(V_wave_in)
	V_in_FFT_norm = 2*np.abs(V_in_FFT)/len(V_wave_in)
	V_in_PS_dBm = 10*np.log10((V_in_FFT_norm/np.sqrt(2))**2/Z_0/.001)

	V_out_FFT = np.fft.fft(V_wave_out)
	V_out_FFT_norm = 2*np.abs(V_out_FFT)/len(V_wave_out)
	V_out_PS_dBm = 10*np.log10((V_out_FFT_norm/np.sqrt(2))**2/Z_0/.001)

	V_ADC_FFT = np.fft.fft(V_wave_ADC)
	V_ADC_FFT_norm = 2*np.abs(V_ADC_FFT)/len(V_wave_ADC)
	V_ADC_PS_dBm = 10*np.log10((V_ADC_FFT_norm/np.sqrt(2))**2/Z_0/.001)

	out_dic = {}
	out_dic['In'] = {}
	out_dic['Out'] = {}
	out_dic['ADC'] = {}

	if plot_all == True:
		plt.figure()
		plt.plot(f,V_in_PS_dBm,label = 'Into Resonators',alpha=1.0)
		plt.gcf
		plt.plot(f,V_out_PS_dBm,label = 'Out of Resonators',alpha = 0.8)
		plt.gcf
		plt.plot(f,V_ADC_PS_dBm-Gain,label = 'At ADC w/ Nonlinearities ',alpha = 0.6)
		plt.xlabel('Frequency [Hz]')
		plt.ylabel('dBm')
		plt.legend()
		#plt.xlim([4.2e9,6.8e9])
		#plt.ylim([-180,-70])
	if plot_noise_temp == True:
		Gain_nl = a1*(1+(3/4)*(a3/a1)*V_out_FFT_norm**2)
		V_ADC_ref = (V_ADC_FFT_norm/Gain_nl) - V_out_FFT_norm
		NL_HEMT_dBm = 10*np.log10((V_ADC_ref/np.sqrt(2))**2/Z_0/.001)
		if (len(f)%10) == 0:
			f_new = f
		if (len(f)%10) != 0:
			f_new = f[0:-(len(f)%10)]
			NL_HEMT_dBm = NL_HEMT_dBm[0:-(len(f)%10)]
		NL_HEMT_W = 0.001*(10**(NL_HEMT_dBm/10))
		f_WperHz = np.zeros(int(len(f_new)/10))
		NL_HEMT_WperHz = np.zeros(int(len(f_new)/10))
		for i in range(len(f_WperHz)):
			f_WperHz[i] = np.mean(f_new[10*i:10*i+9])
			NL_HEMT_WperHz[i] = np.mean(NL_HEMT_W[10*i:10*i+9])/(f_new[10*i+9]-f_new[10*i])
		NL_HEMT_K = NL_HEMT_WperHz/kb
		plt.figure(fig_num)
		if label == None:
			if alpha == None:
				plt.semilogy(f_WperHz,NL_HEMT_K)
			if alpha != None:
				plt.semilogy(f_WperHz,NL_HEMT_K,alpha = alpha)
		if label != None:
			if alpha == None:
				plt.semilogy(f_WperHz,NL_HEMT_K,label=label)
			if alpha != None:
				plt.semilogy(f_WperHz,NL_HEMT_K,label=label,alpha = alpha)
		'''
		f_WperHz_ourband = f_WperHz[np.argmin(np.abs((2*np.min(fres)-np.max(fres))-f_WperHz)):np.argmin(np.abs((2*np.max(fres)-np.min(fres))-f_WperHz))]

		NL_HEMT_K_ourband = NL_HEMT_K[np.argmin(np.abs((2*np.min(fres)-np.max(fres))-f_WperHz)):np.argmin(np.abs((2*np.max(fres)-np.min(fres))-f_WperHz))]

		NL_HEMT_K_ourband[np.argmin(np.abs(np.min(fres)-f_WperHz_ourband)):np.argmin(np.abs(np.max(fres)-f_WperHz_ourband))] = np.nan

		out_dic['ADC']['f_WperHz_ourband'] = f_WperHz_ourband
		out_dic['ADC']['K_ourband'] = NL_HEMT_K_ourband
		'''
		plt.xlabel('Frequency [Hz]',fontsize = 24)
		plt.ylabel('$T_{Noise}$ [K]',fontsize = 24)
		plt.legend()
		#plt.xlim([1e9,9e9])
		#plt.ylim([1e-1,1e3])
		out_dic['ADC']['f_WperHz'] = f_WperHz
		out_dic['ADC']['WperHz'] = NL_HEMT_WperHz
		out_dic['ADC']['K'] = NL_HEMT_K
		out_dic['ADC']['Gain_nl'] = Gain_nl

	out_dic['In']['f'] = f
	out_dic['In']['t'] = t
	out_dic['In']['dBm'] = V_in_PS_dBm
	out_dic['In']['Volts'] = V_wave_in
	out_dic['Out']['f'] = f
	out_dic['Out']['t'] = t
	out_dic['Out']['dBm'] = V_out_PS_dBm
	out_dic['Out']['Volts'] = V_wave_out
	out_dic['ADC']['f'] = f
	out_dic['ADC']['t'] = t
	out_dic['ADC']['dBm'] = V_ADC_PS_dBm
	out_dic['ADC']['Volts'] = V_wave_ADC
	return out_dic
