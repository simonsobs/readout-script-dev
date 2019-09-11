import numpy as np
import matplotlib.pyplot as plt
import Noise_Referral as nr
plt.ion()

T_test = 1. #Use 1K to see Transfer Function

#####Mates 2.163 no Q dependence##########
alpha_test = 1./2.
LJ = 6.69e-11
Lsqin = 2.28e-10

def TN_to_NEI_Mates(T,f0,LJosephson,Min,alpha,p=False):
	k_b = 1.38e-23;
	NEI = (1./np.sqrt(alpha))*np.sqrt((4.*k_b*T*LJosephson)/(np.pi*f0))*(1/Min)
	if p == True:
		print('Noise temp', T, 'K @ LNA-1 input referred to NEI:',np.round(NEI/1e-12,2),'pA/rt(Hz)')
	return NEI/1e-12

def refer_dBc_simple(dBc,Qr, Qc, f0,dfdI, p=False):
	Sphi = 10**(dBc/10)
	Sf = Sphi*(f0*(Qc-Qr)/(2*Qr**2))**2 #This expression is from derivatives of Zmuidizinas Review Paper
	NEI = np.sqrt(Sf/(dfdI**2.))
	if p == True:
		print('Phase Noise of: ',dBc, 'dBc/Hz referred to NEI:',NEI)
	return NEI
	
def TN_to_NEI_MSF(T,Pfeed,Qr,Qc,f0,dfdI,p=False):
	k_b = 1.38e-23;
	Z0 = 50.
	Pnoise_Im = 10*np.log10((4*k_b*T*Z0/np.sqrt(2.))/.001)
	dBc_HEMT = Pnoise_Im - Pfeed
	NEI_HEMT = refer_dBc_simple(dBc_HEMT,Qr,Qc,f0,dfdI)
	return NEI_HEMT
	
def TN_to_NEI_S21min_memo_2Qc_over_fr(T,Pin,Qr,Qc,fr,dfdI,p=False):
	k_b = 1.38e-23;
	Z0 = 50.
	Qi = 1/((1/Qr)-(1/Qc))
	Vnoise = np.sqrt(4*k_b*T*Z0)/2
	S_21min = Qc/(Qc+Qi)
	Pin_W = 10.**(Pin/10.)*.001
	dVdI = S_21min*np.sqrt(Pin_W*Z0)*(2*Qc/fr)*dfdI
	NEI = Vnoise/dVdI
	return NEI
	
def TN_to_NEI_S21min_imagS21_Deriv(T,Pin,Qr,Qc,f0,dfdI,p=False):
	k_b = 1.38e-23;
	Z0 = 50.
	S21min = (1. - (Qr/Qc))
	Vnoise_Im = np.sqrt(4*k_b*T*Z0)/2. #The divide by 2 comes from the fact that only imaginary and real parts are equal but only imag is contributing
	dImS21df = (2.*Qr**2)/(Qc*f0)
	dVdI = S21min*np.sqrt(0.001*(10**(Pin/10.))*Z0)*dImS21df*dfdI
	NEI = Vnoise_Im/dVdI
	return NEI

f0_test = 5e9;
Qr_test = 17478.
Pfeed_test = -70.
dfdI_test = 4.59e-2
Dip_dB_sweep = np.logspace(np.log10(0.5),np.log10(50.),num = 500)
Dip_sweep = 10.**(-Dip_dB_sweep/20.)
Qc_sweep = Qr_test/(1-Dip_sweep)

TF_Mates = np.zeros(len(Dip_sweep))
TF_MSF = np.zeros(len(Dip_sweep))
TF_memo = np.zeros(len(Dip_sweep))
TF_ImS21deriv = np.zeros(len(Dip_sweep))
TF_SWH = np.zeros(len(Dip_sweep))

for i in range(len(Dip_sweep)):
	TF_Mates[i] = TN_to_NEI_Mates(T_test,f0_test,LJ,Lsqin,alpha_test)
	TF_MSF[i] = TN_to_NEI_MSF(T_test,Pfeed_test,Qr_test,Qc_sweep[i],f0_test,dfdI_test)
	TF_memo[i] = TN_to_NEI_S21min_memo_2Qc_over_fr(T_test,Pfeed_test,Qr_test,Qc_sweep[i],f0_test,dfdI_test)
	TF_ImS21deriv[i] = TN_to_NEI_S21min_imagS21_Deriv(T_test,Pfeed_test,Qr_test,Qc_sweep[i],f0_test,dfdI_test)
	TF_SWH[i] = nr.TN_to_NEI_SWH(T_test,Pfeed_test,Qr_test,f0_test,dfdI_test)
	
plt.semilogy(Dip_dB_sweep,TF_Mates,label='Mates 2.163 No Q Dependence')
plt.semilogy(Dip_dB_sweep,TF_MSF,label='Refering to dBc and Using d$\phi$/df')
plt.semilogy(Dip_dB_sweep,TF_memo,label='Following Steps Outlined in Memo, 2Q_c/f_r')
plt.semilogy(Dip_dB_sweep,TF_ImS21deriv,label='dImS21 Derivative')
plt.semilogy(Dip_dB_sweep,TF_SWH,label='dS21df SWH')
plt.xlabel('Dip Depth [dB]')
plt.ylabel('Transfer Function [pA/K]')
plt.title('Comparing Transfer Functions')
plt.legend()

####################################
Qrs = np.linspace(1000,200000,num=10000)
i=0;
TF_SWH = np.zeros(len(Qrs))
for Qr in Qrs:
	TF_SWH[i] = nr.TN_to_NEI_SWH(T_test,Pfeed_test,Qr,f0_test,dfdI_test)
	i = i+1
plt.figure()
plt.semilogx(Qrs,TF_SWH)

