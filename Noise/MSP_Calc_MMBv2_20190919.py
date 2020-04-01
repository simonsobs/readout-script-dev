import numpy as np
from matplotlib import pyplot as plt
import Noise_Referral as nr
import pandas as pd
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

df_mmb = pd.read_pickle('/home/arnoldlabws3/Scratch/MSF/Personal_Laptop_Docs/SO/MMBv2p/MMBv2_20190918-mccarrick-mmbv2pp_resonator_and_noise_resonator-params.pkl.pkl')
QoverQc = df_mmb['Q']/df_mmb['Qc']
N_reasonable_resonators = len(np.where(QoverQc<1.0)[0])
idx_reasonable = np.where(QoverQc<1.0)[0]
####Number of primary tones to consider###
Ntones_step = [N_reasonable_resonators]

####BW, Q_c, and f0 per assumptions stated in doc string
Qcs = np.asarray(df_mmb['Qc'][idx_reasonable])
Qrs = np.asarray(df_mmb['Q'][idx_reasonable])
Qis = np.asarray(df_mmb['Qi'][idx_reasonable])
f0s = np.asarray(df_mmb['f0'][idx_reasonable])
Sws = np.asarray(df_mmb['Sw'][idx_reasonable])

####Sweep Through Dip Depths between 0.5 and 12 dB###
Dips = 1. - (Qrs/Qcs)
Dips_db = 20*np.log10(Dips)

####Sweep power into cryostat
#P_sweep = np.arange(-38,-32) #Corresponds to -65 to -70 dBm at the feedline
P_sweep = np.asarray([-38.]); #Corresponds to -70 dBm at the feedline

####Initialize output NEI matrices for each noise source and each number of tones
NEI_PNF = np.zeros((len(P_sweep),len(Dips)))
NEI_HEMT = np.zeros((len(P_sweep),len(Dips)))
NEI_DAC = np.zeros((len(P_sweep),len(Dips)))
NEI_TOT = np.zeros((len(P_sweep),len(Dips)))

####Initialize input/output chain loss and temperatures and dfdI
lossin, Tavgin = nr.read_loss_in('/home/arnoldlabws3/Scratch/MSF/readout-script-dev/Noise/')
lossout, Tavgout = nr.read_loss_out('/home/arnoldlabws3/Scratch/MSF/readout-script-dev/Noise/')
LNA_TNtyp= [2.1,40.,191.]
T_HEMT = nr.Amp_Chain_Noise_Temp(loss_out = lossout, Tavg_out = Tavgout, LNA_TN=LNA_TNtyp)
dfdItyp = 2.06e-2 # Hz/pA
alpha_step = 1.0 - np.linspace(0.1,0.9,num = len(P_sweep)) #This just sets line style for plotting
T_PNF = nr.Pseudo_Noise_Floor_to_K(-70.,np.median(Qrs),np.median(Qcs),5e9,dfdItyp,N_reasonable_resonators) 

####Main loop through input power and resonance depth with calculation of each noise source at each loop step.
for nton in range(len(Ntones_step)):
	####Initial figure formatting
	plt.figure()
	#mng = plt.get_current_fig_manager()
	#mng.resize(*mng.window.maxsize())
	
	for P in range(len(P_sweep)):
		Pfeed = nr.p_referred_pertone(loss = lossin, Pin = P_sweep[P])
		for dip in range(len(Dips)):
			NEI_PNF[P,dip] = nr.TN_to_NEI(T_PNF,Pfeed,Qrs[dip],Qcs[dip],f0 = f0s[dip],dfdI = dfdItyp)
			NEI_HEMT[P,dip] = nr.TN_to_NEI(T_HEMT,Pfeed,Qrs[dip],Qcs[dip],f0 = f0s[dip],dfdI = dfdItyp)
			T_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,P_sweep[P],Qrs[dip],Qcs[dip],f0 = f0s[dip],dfdI = dfdItyp)
			NEI_DAC[P,dip] = nr.TN_to_NEI(T_DAC,Pfeed,Qrs[dip],Qcs[dip],f0 = f0s[dip],dfdI = dfdItyp)
			NEI_TOT[P,dip] = nr.quad_sum(nr.quad_sum(NEI_PNF[P,dip],NEI_HEMT[P,dip]),NEI_DAC[P,dip])
		plt.plot(Dips_db,NEI_PNF[P,:],'ro',alpha = alpha_step[P],label = 'PNF: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
		plt.plot(Dips_db,NEI_HEMT[P,:],'go',alpha = alpha_step[P],label = 'HEMT: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
		plt.plot(Dips_db,NEI_DAC[P,:],'bo',alpha = alpha_step[P],label = 'DAC: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
		plt.plot(Dips_db,NEI_TOT[P,:],'ko',alpha = alpha_step[P],label = 'Total: $P_{feed}$ = '+str(np.round(Pfeed,decimals = 0)))
	####Post plotting figure formatting
	plt.title('Pseudo-Noise for $N_{tones}$ = '+str(Ntones_step[nton]+' of '+str(int(1728/2))))
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
	p.savefig('/home/msilvafe/Pictures/ExploreNoiseSources/SweepDip_NewReferral_FixCoupling/PNF_and_Total'+str(np.round(Ntones_step[nton],0))+'Tones_MMBv2_MSP_Calc.png',dpi = 96)#,bbox_inches='tight')
	plt.close()
	
####Initialize output data dictionary
outdat = {}
outdat[str(Ntones_step[0]) + 'Tones'] = {}
outdat[str(Ntones_step[0]) + 'Tones']['Qc'] = Qcs
outdat[str(Ntones_step[0]) + 'Tones']['Resonance Depth'] = Dips_db
outdat[str(Ntones_step[0]) + 'Tones']['f0'] = f0s
outdat[str(Ntones_step[0]) + 'Tones']['NEI HEMT'] = NEI_HEMT
outdat[str(Ntones_step[0]) + 'Tones']['NEI DAC'] = NEI_DAC
outdat[str(Ntones_step[0]) + 'Tones']['NEI PNF'] = NEI_PNF
outdat[str(Ntones_step[0]) + 'Tones']['NEI Total'] = NEI_TOT


####Output dictionary save filepath + name
np.save('/home/msilvafe/Pictures/ExploreNoiseSources/SweepDip_NewReferral_FixCoupling/MMBv2_MSP_Calc.npy',outdat)
