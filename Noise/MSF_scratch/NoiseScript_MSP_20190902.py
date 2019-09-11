import numpy as np
import matplotlib.pyplot as plt
import Noise_Referral as nr

####Sweep Through N-Tones 3 cases, 500, 1000, 2000###
Ntones_sweep = [500.,1000.,2000.]

####Sweep Through Dip Depths between 0.1 and 15 dB###
Dip_dB_sweep = np.logspace(0,np.log10(10.),num = 500)
Dip_sweep = 10.**(-Dip_dB_sweep/20.)

####Fix Q and find Qc given Dip Depths##########
###Q data taken from Heather's MMBV2' slides page 7
Pin_cryostat_fix = -38.
Qc_sweep = np.zeros((4,len(Dip_sweep)))
Q_fix = [17478.,11349.,12931.,13598.]
f0_fix = [5e9,6e9,7e9,8e9]
Qc_sweep[0,:] = Q_fix[0]/(1-Dip_sweep)
Qc_sweep[1,:] = Q_fix[1]/(1-Dip_sweep)
Qc_sweep[2,:] = Q_fix[2]/(1-Dip_sweep)
Qc_sweep[3,:] = Q_fix[3]/(1-Dip_sweep)

####Initialize output NEI matrices for each noise source###
NEI_HEMT = np.zeros((4,len(Dip_sweep)))
NEI_PNF = np.zeros((4,len(Dip_sweep)))
NEI_DAC = np.zeros((4,len(Dip_sweep)))
NEI_tot = np.zeros((4,len(Dip_sweep)))

####Initialize input powers and temperatures and dfdI#####
lossin, Tavgin = nr.read_loss_in('/home/msilvafe/readout-script-dev/Noise/')
lossout, Tavgout = nr.read_loss_out('/home/msilvafe/readout-script-dev/Noise/')
Pfeed = nr.p_referred_pertone(loss = lossin, Pin = Pin_cryostat_fix, p=True)
LNA_TNtyp= [2.1,40.,191.]
T_HEMT = nr.Total_Chain_Noise_Temp(loss_out = lossout, Tavg_out = Tavgout, LNA_TN=LNA_TNtyp)
dfdItyp = 4.59e-2 # Hz/pA

####Loop over all cases and save appropriate plots + data###
####NEED TO CHECK dfdI######
for ntones in Ntones_sweep:
	for i in range(4):
		for j in range(len(Dip_sweep)):
			NEI_HEMT[i,j] = nr.TN_to_NEI_SWH(T_HEMT,Pfeed,Q_fix[i],f0 = f0_fix[i],dfdI = dfdItyp)
			K_PNF = nr.Pseudo_Noise_Floor_to_K(Pfeed,lossin,Q_fix[i],Qc_sweep[i,j],f0 = f0_fix[i],dfdI = dfdItyp,Ntones = ntones)
			NEI_PNF[i,j] = nr.TN_to_NEI_SWH(K_PNF,Pfeed,Q_fix[i],f0 = f0_fix[i],dfdI = dfdItyp)
			K_DAC = nr.refer_phase_noise_to_K(101.,lossin,Tavgin,Pin_cryostat_fix,Q_fix[i],Qc_sweep[i,j],f0 = f0_fix[i],dfdI = dfdItyp)
			NEI_DAC[i,j] = nr.TN_to_NEI_SWH(K_DAC,Pfeed,Q_fix[i],f0 = f0_fix[i],dfdI = dfdItyp)
			NEI_tot[i,j] = nr.quad_sum(nr.quad_sum(NEI_HEMT[i,j],NEI_PNF[i,j]),NEI_DAC[i,j])
		if ntones == Ntones_sweep[0]:
			if i == 0:
				plt.figure()
				plt.semilogy(Dip_dB_sweep,NEI_HEMT[i,:],'r--',label = 'HEMT - 500 Tones, 5GHz')
				plt.semilogy(Dip_dB_sweep,NEI_PNF[i,:],'g--',label = 'Pseudo-Noise - 500 Tones, 5GHz')
				plt.semilogy(Dip_dB_sweep,NEI_DAC[i,:],'b--',label = 'DAC - 500 Tones, 5GHz')
				plt.semilogy(Dip_dB_sweep,NEI_tot[i,:],'k',label = 'Total - 500 Tones, 5GHz')
		if ntones == Ntones_sweep[-1]:
			if i == 3:
				plt.semilogy(Dip_dB_sweep,NEI_HEMT[i,:],'r--',alpha = 0.5,label = 'HEMT - 2000 Tones, 8GHz')
				plt.semilogy(Dip_dB_sweep,NEI_PNF[i,:],'g--',alpha = 0.5,label = 'Pseudo-Noise - 2000 Tones, 8GHz')
				plt.semilogy(Dip_dB_sweep,NEI_DAC[i,:],'b--',alpha = 0.5,label = 'DAC - 2000 Tones, 8GHz')
				plt.semilogy(Dip_dB_sweep,NEI_tot[i,:],'k',alpha = 0.5,label = 'Total - 2000 Tones, 8GHz')
		#plt.title('Noise sources for $N_{tones}$ = '+str(ntones)+', and $f_{max}$ = '+str(f0_fix[i]/1e9)+' GHz')
		plt.xlabel('Dip Depth [dB]')
		plt.ylabel('NEI [pA/rtHz]')
		plt.legend()
		
'''ntones = Ntones_sweep[2]
i =0
dthetadf = np.zeros(len(Dip_sweep))
dImS21df = np.zeros(len(Dip_sweep))
for j in range(len(Dip_sweep)):
	NEI_HEMT[i,j] = nr.TN_to_NEI_MSF(T_HEMT,Pfeed,Q_fix[i],Qc_sweep[i,j],f0 = f0_fix[i],dfdI = dfdItyp)
	NEI_PNF[i,j] = nr.Pseudo_Noise_Floor(Pfeed,lossin,Q_fix[i],Qc_sweep[i,j],f0 = f0_fix[i],dfdI = dfdItyp,Ntones = ntones)
	NEI_DAC[i,j] = nr.refer_phase_noise(101.,lossin,Tavgin,Pin_cryostat_fix,Q_fix[i],Qc_sweep[i,j],f0 = f0_fix[i],dfdI = dfdItyp)
	NEI_tot[i,j] = nr.quad_sum(nr.quad_sum(NEI_HEMT[i,j],NEI_PNF[i,j]),NEI_DAC[i,j])
plt.figure()
plt.semilogy(Dip_dB_sweep,NEI_HEMT[i,:],'--',label = 'dImS21df')
plt.semilogy(Dip_dB_sweep,dImS21df,'--',label = 'S21min')
plt.semilogy(Dip_dB_sweep,NEI_PNF[i,:],'--',label = 'Pseudo-Noise')
plt.semilogy(Dip_dB_sweep,NEI_DAC[i,:],'--',label = 'DAC')
plt.semilogy(Dip_dB_sweep,NEI_tot[i,:],label = 'Total')
plt.title('Noise sources for $N_{tones}$ = '+str(ntones)+', and $f_{max}$ = '+str(f0_fix[i]/1e9)+' GHz')
plt.xlabel('Dip Depth [dB]')
plt.ylabel('NEI [pA/rtHz]')'''
