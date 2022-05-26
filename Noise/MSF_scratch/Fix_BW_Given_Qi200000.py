import numpy as np
from matplotlib import pyplot as plt
import Noise_Referral as nr
plt.ion()
plt.rcParams.update({'font.size': 22,'legend.fontsize': 16})

'''This script uses functions in the noise referral code to calculate the noise contribution to the resontators due to the SMuRF DAC, the HEMT input noise temperature and the 3rd order nonlinear products pseudo noise as a function of the dip depth of the resonators. The assumptions made are the following:

Assumptions
Designed BW: 100 kHz
Deigned Qc: Assumes Qi = 200,000
f0: linearly spaced between 4-5GHz for 500 tones, 4-6 for 1000 tones, and 4-8 for 2000 tones
Qi/Qr: These are swept based on the resonance depth = 1-Qc/Qr with Qc as defined above
Resonance Depth: Swept between -0.5 and -12.0 dB
dfdI: 4.59e-2 Hz/pA this is the the referral from frequency noise to current noise and comes from averaging the derivative of the slope of the f-phi curve of your resonators. This default value comes from a calculation of expected dfdI given nominal squid parameters for umux100k v1.0
'''

####Number of primary tones to consider###
Ntones_step = [500.,1000.,2000.]

####BW, Q_c, and f0 per assumptions stated in doc string
BW_design = 100e3; #Fix BW designed = 100kHz
Q_i_design = 200000. #Assume Q_i design was 200000.
f_500_tones = np.linspace(4e9,5e9,num=500)
f_1000_tones = np.linspace(4e9,6e9,num=1000)
f_2000_tones = np.linspace(4e9,8e9,num=2000)
Qc_500_tones = 1./((BW_design/f_500_tones) - (1/Q_i_design))
Qc_1000_tones = 1./((BW_design/f_1000_tones) - (1/Q_i_design))
Qc_2000_tones = 1./((BW_design/f_2000_tones) - (1/Q_i_design))

####Sweep Through Dip Depths between 0.5 and 12 dB###
Dip_dB_sweep_500_tones = np.logspace(np.log10(0.5),np.log10(12.),num = 500)
Dip_sweep_500_tones = 10.**(-Dip_dB_sweep_500_tones/20.)
Dip_dB_sweep_1000_tones = np.logspace(np.log10(0.5),np.log10(12.),num = 1000)
Dip_sweep_1000_tones = 10.**(-Dip_dB_sweep_1000_tones/20.)
Dip_dB_sweep_2000_tones = np.logspace(np.log10(0.5),np.log10(12.),num = 2000)
Dip_sweep_2000_tones = 10.**(-Dip_dB_sweep_2000_tones/20.)

#####Calculate Qr given Qc and Dip Depth
Qr_sweep_500_tones = (1-Dip_sweep_500_tones)*Qc_500_tones
Qr_sweep_1000_tones = (1-Dip_sweep_1000_tones)*Qc_1000_tones
Qr_sweep_2000_tones = (1-Dip_sweep_2000_tones)*Qc_2000_tones

####Sweep power into cryostat
P_sweep = np.arange(-38,-32) #Corresponds to -65 to -70 dBm at the feedline
#P_sweep = np.asarray([-38.]); #Corresponds to -70 dBm at the feedline


####Initialize output NEI matrices for each noise source and each number of tones
NEI_PNF_DipSweep_500 = np.zeros((len(P_sweep),len(Dip_sweep_500_tones)))
NEI_PNF_DipSweep_1000 = np.zeros((len(P_sweep),len(Dip_sweep_1000_tones)))
NEI_PNF_DipSweep_2000 = np.zeros((len(P_sweep),len(Dip_sweep_2000_tones)))
NEI_HEMT_DipSweep_500 = np.zeros((len(P_sweep),len(Dip_sweep_500_tones)))
NEI_HEMT_DipSweep_1000 = np.zeros((len(P_sweep),len(Dip_sweep_1000_tones)))
NEI_HEMT_DipSweep_2000 = np.zeros((len(P_sweep),len(Dip_sweep_2000_tones)))
NEI_DAC_DipSweep_500 = np.zeros((len(P_sweep),len(Dip_sweep_500_tones)))
NEI_DAC_DipSweep_1000 = np.zeros((len(P_sweep),len(Dip_sweep_1000_tones)))
NEI_DAC_DipSweep_2000 = np.zeros((len(P_sweep),len(Dip_sweep_2000_tones)))
NEI_TOT_DipSweep_500 = np.zeros((len(P_sweep),len(Dip_sweep_500_tones)))
NEI_TOT_DipSweep_1000 = np.zeros((len(P_sweep),len(Dip_sweep_1000_tones)))
NEI_TOT_DipSweep_2000 = np.zeros((len(P_sweep),len(Dip_sweep_2000_tones)))

####Initialize input/output chain loss and temperatures and dfdI
lossin, Tavgin = nr.read_loss_in('/home/msilvafe/readout-script-dev/Noise/')
lossout, Tavgout = nr.read_loss_out('/home/msilvafe/readout-script-dev/Noise/')
LNA_TNtyp= [2.1,40.,191.]
T_HEMT = nr.Amp_Chain_Noise_Temp(loss_out = lossout, Tavg_out = Tavgout, LNA_TN=LNA_TNtyp)
dfdItyp = 4.59e-2 # Hz/pA
alpha_step = 1.0 - np.linspace(0.1,0.9,num = len(P_sweep)) #This just sets line style for plotting

####Main loop through input power and resonance depth with calculation of each noise source at each loop step.
for nton in range(len(Ntones_step)):
	####Initial figure formatting
	p = plt.figure()
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	
	for P in range(len(P_sweep)):
		Pfeed = nr.p_referred_pertone(loss = lossin, Pin = P_sweep[P])
		if Ntones_step[nton] == 500:
			for dip in range(len(Dip_sweep_500_tones)):
				K_PNF = nr.Pseudo_Noise_Floor_to_K(Pfeed,Qr_sweep_500_tones[dip],Qc_500_tones[dip],f0 = f_500_tones[dip],dfdI = dfdItyp,Ntones = Ntones_step[nton])
				NEI_PNF_DipSweep_500[P,dip] = nr.TN_to_NEI(K_PNF,Pfeed,Qr_sweep_500_tones[dip],Qc_500_tones[dip],f0 = f_500_tones[dip],dfdI = dfdItyp)
				NEI_HEMT_DipSweep_500[P,dip] = nr.TN_to_NEI(T_HEMT,Pfeed,Qr_sweep_500_tones[dip],Qc_500_tones[dip],f0 = f_500_tones[dip],dfdI = dfdItyp)
				T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_sweep[P],Qr_sweep_500_tones[dip],Qc_500_tones[dip],f0 = f_500_tones[dip],dfdI = dfdItyp)
				NEI_DAC_DipSweep_500[P,dip] = nr.TN_to_NEI(T_DAC,Pfeed,Qr_sweep_500_tones[dip],Qc_500_tones[dip],f0 = f_500_tones[dip],dfdI = dfdItyp)
				NEI_TOT_DipSweep_500[P,dip] = nr.quad_sum(nr.quad_sum(NEI_PNF_DipSweep_500[P,dip],NEI_HEMT_DipSweep_500[P,dip]),NEI_DAC_DipSweep_500[P,dip])
			plt.plot(Dip_dB_sweep_500_tones,NEI_PNF_DipSweep_500[P,:],'r--',alpha = alpha_step[P],label = 'PNF: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_500_tones,NEI_HEMT_DipSweep_500[P,:],'g--',alpha = alpha_step[P],label = 'HEMT: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_500_tones,NEI_DAC_DipSweep_500[P,:],'b--',alpha = alpha_step[P],label = 'DAC: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_500_tones,NEI_TOT_DipSweep_500[P,:],'k-',alpha = alpha_step[P],label = 'Total: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
		if Ntones_step[nton] == 1000:			
			for dip in range(len(Dip_sweep_1000_tones)):
				K = nr.Pseudo_Noise_Floor_to_K(Pfeed,Qr_sweep_1000_tones[dip],Qc_1000_tones[dip],f0 = f_1000_tones[dip],dfdI = dfdItyp,Ntones = Ntones_step[nton])
				NEI_PNF_DipSweep_1000[P,dip] = nr.TN_to_NEI(K,Pfeed,Qr_sweep_1000_tones[dip],Qc_1000_tones[dip],f0 = f_1000_tones[dip],dfdI = dfdItyp)
				NEI_HEMT_DipSweep_1000[P,dip] = nr.TN_to_NEI(T_HEMT,Pfeed,Qr_sweep_1000_tones[dip],Qc_1000_tones[dip],f0 = f_1000_tones[dip],dfdI = dfdItyp)
				T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_sweep[P],Qr_sweep_1000_tones[dip],Qc_1000_tones[dip],f0 = f_1000_tones[dip],dfdI = dfdItyp)
				NEI_DAC_DipSweep_1000[P,dip] = nr.TN_to_NEI(T_DAC,Pfeed,Qr_sweep_1000_tones[dip],Qc_1000_tones[dip],f0 = f_1000_tones[dip],dfdI = dfdItyp)
				NEI_TOT_DipSweep_1000[P,dip] = nr.quad_sum(nr.quad_sum(NEI_PNF_DipSweep_1000[P,dip],NEI_HEMT_DipSweep_1000[P,dip]),NEI_DAC_DipSweep_1000[P,dip])				
			plt.plot(Dip_dB_sweep_1000_tones,NEI_PNF_DipSweep_1000[P,:],'r--',alpha = alpha_step[P],label = 'PNF: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))		
			plt.plot(Dip_dB_sweep_1000_tones,NEI_HEMT_DipSweep_1000[P,:],'g--',alpha = alpha_step[P],label = 'HEMT: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_1000_tones,NEI_DAC_DipSweep_1000[P,:],'b--',alpha = alpha_step[P],label = 'DAC: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))						
			plt.plot(Dip_dB_sweep_1000_tones,NEI_TOT_DipSweep_1000[P,:],'k-',alpha = alpha_step[P],label = 'Total: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
		if Ntones_step[nton] == 2000:			
			for dip in range(len(Dip_sweep_2000_tones)):
				K = nr.Pseudo_Noise_Floor_to_K(Pfeed,Qr_sweep_2000_tones[dip],Qc_2000_tones[dip],f0 = f_2000_tones[dip],dfdI = dfdItyp,Ntones = Ntones_step[nton])
				NEI_PNF_DipSweep_2000[P,dip] = nr.TN_to_NEI(K,Pfeed,Qr_sweep_2000_tones[dip],Qc_2000_tones[dip],f0 = f_2000_tones[dip],dfdI = dfdItyp)
				NEI_HEMT_DipSweep_2000[P,dip] = nr.TN_to_NEI(T_HEMT,Pfeed,Qr_sweep_2000_tones[dip],Qc_2000_tones[dip],f0 = f_2000_tones[dip],dfdI = dfdItyp)
				T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_sweep[P],Qr_sweep_2000_tones[dip],Qc_2000_tones[dip],f0 = f_2000_tones[dip],dfdI = dfdItyp)
				NEI_DAC_DipSweep_2000[P,dip] = nr.TN_to_NEI(T_DAC,Pfeed,Qr_sweep_2000_tones[dip],Qc_2000_tones[dip],f0 = f_2000_tones[dip],dfdI = dfdItyp)
				NEI_TOT_DipSweep_2000[P,dip] = nr.quad_sum(nr.quad_sum(NEI_PNF_DipSweep_2000[P,dip],NEI_HEMT_DipSweep_2000[P,dip]),NEI_DAC_DipSweep_2000[P,dip])				
			plt.plot(Dip_dB_sweep_2000_tones,NEI_PNF_DipSweep_2000[P,:],'r--',alpha = alpha_step[P],label = 'PNF: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_2000_tones,NEI_HEMT_DipSweep_2000[P,:],'g--',alpha = alpha_step[P],label = 'HEMT: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_2000_tones,NEI_DAC_DipSweep_2000[P,:],'b--',alpha = alpha_step[P],label = 'DAC: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
			plt.plot(Dip_dB_sweep_2000_tones,NEI_TOT_DipSweep_2000[P,:],'k-',alpha = alpha_step[P],label = 'Total: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
	####Post plotting figure formatting
	plt.title('Pseudo-Noise for $N_{tones}$ = '+str(Ntones_step[nton]))
	plt.xlabel('Dip Depth [dB]')
	plt.ylabel('NEI [pA/rtHz]')
	l = plt.legend(bbox_to_anchor=(1.0,1.0),fontsize = 12, title = 'Noise Source and \n Tone Power [dBm]') 
	ax = plt.axes()
	ax.set_position(pos = [0.1*0.85,0.1*0.85,1.0*0.75,1.0*0.85]) 
	plt.ylim([0,100])
	plt.xlim([0,12])
	plt.xticks(np.arange(0, 13, step=1))
	plt.yticks(np.arange(0, 110, step=10))
	
	####Save figure filepath + name.
	#p.savefig('/home/msilvafe/Pictures/ExploreNoiseSources/SweepDip_NewReferral_FixCoupling/PNF_and_Total'+str(int(Ntones_step[nton]))+'Tones_min70dBm.png',dpi = 96)#,bbox_inches='tight')
	#plt.close()
	
####Initialize output data dictionary
outdat = {}
outdat['Qi Used to Design Qc'] = Q_i_design
outdat['BW Used to Design Qc'] = BW_design
outdat['500 Tones'] = {}
outdat['500 Tones']['Qc'] = Qc_500_tones
outdat['500 Tones']['Resonance Depth'] = Dip_dB_sweep_500_tones
outdat['500 Tones']['f0'] = f_500_tones
outdat['500 Tones']['NEI HEMT'] = NEI_HEMT_DipSweep_500
outdat['500 Tones']['NEI DAC'] = NEI_DAC_DipSweep_500
outdat['500 Tones']['NEI PNF'] = NEI_PNF_DipSweep_500
outdat['500 Tones']['NEI Total'] = NEI_TOT_DipSweep_500
outdat['1000 Tones'] = {}
outdat['1000 Tones']['Qc'] = Qc_1000_tones
outdat['1000 Tones']['Resonance Depth'] = Dip_dB_sweep_1000_tones
outdat['1000 Tones']['f0'] = f_1000_tones
outdat['1000 Tones']['NEI HEMT'] = NEI_HEMT_DipSweep_1000
outdat['1000 Tones']['NEI DAC'] = NEI_DAC_DipSweep_1000
outdat['1000 Tones']['NEI PNF'] = NEI_PNF_DipSweep_1000
outdat['1000 Tones']['NEI Total'] = NEI_TOT_DipSweep_1000
outdat['2000 Tones'] = {}
outdat['2000 Tones']['Qc'] = Qc_2000_tones
outdat['2000 Tones']['Resonance Depth'] = Dip_dB_sweep_2000_tones
outdat['2000 Tones']['f0'] = f_2000_tones
outdat['2000 Tones']['NEI HEMT'] = NEI_HEMT_DipSweep_2000
outdat['2000 Tones']['NEI DAC'] = NEI_DAC_DipSweep_2000
outdat['2000 Tones']['NEI PNF'] = NEI_PNF_DipSweep_2000
outdat['2000 Tones']['NEI Total'] = NEI_TOT_DipSweep_2000

####Output dictionary save filepath + name
#np.save('/home/msilvafe/Pictures/ExploreNoiseSources/SweepDip_NewReferral_FixCoupling/SweepDip_FixCoupling_min70dBm.npy',outdat)
