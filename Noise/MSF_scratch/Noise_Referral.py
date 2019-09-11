##########Import Packages##########
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import csv
from numpy import genfromtxt
from math import pi
plt.ion()
plt.rcParams.update({'font.size': 22,'legend.fontsize': 16})
dirname = '/home/msilvafe/Documents/SO/umux/Noise/'

##########Initialize Constants##########
'''k_b = 1.38e-23 #k Boltzmann [J/K]
phi_0 = 2.067833831e-15 #Flux Quantum in [Wb]
Ntones_MF = 1792.
Ntones_MF_db = 10*np.log10(Ntones_MF)
Ntones_HF = 1792.
Ntones_HF_db = 10*np.log10(Ntones_HF)
Ntones_LF = 1195.
Ntones_LF_db = 10*np.log10(Ntones_LF)
Ntonestyp = 2000.
Ntones_db = 10*np.log10(Ntones)
N_MF = 12
N_HF = 6
N_LF = 3
Ntones_tot = N_MF*Ntones_MF+N_HF*Ntones_HF+N_LF*Ntones_LF
Ntones_tot_db = 10*np.log10(Ntones_tot)
Lambda = 0.333
dfdOhi0 = 2.17e5 # Hz/Phi_0
fmin = 4e9
fmax = 8e9
Qtyp = 5e4
dfdItyp = 4.59e-2 # Hz/pA
LJtyp = 6.69e-11'''

#Inductance Matrix
Lsqsq = 2.23e-11
Lsqfr = 2.71e-11
Lsqin=2.28e-10
Lsqres=1.35e-12
Lfrfr=6.49e-10
Lfrin=3.96e-10
Lfrres=2.5e-12
Linin=5.48e-9
Linres=1.86e-11
Lresres=2.28e-10
Psmurfmax=0.
LNA_TNtyp= [2.1,40.,191.]
BWtyp=100e3
Z0typ = 50

##########Define Functions##########

def readcoax(fname,f_eval):
	"""Function for Reading In Coax Attenuation Data and Fit for slope and intercept as a function of temperature at a fixed frequency (f_eval)

	Parameters:
	fname (str): file path
	f_eval (float): frequency to calculate output slope and intercept

	Returns:
	[slope, intercept] (float list):  at given frequency (f_eval) 
	"""

	d = genfromtxt(fname, delimiter=',')
	slope_T1=(d[2,1]-d[1,1])/(d[2,0]-d[1,0])
	slope_T2=(d[2,2]-d[1,2])/(d[2,0]-d[1,0])
	int_T1=d[1,1]-slope_T1*d[1,0]
	int_T2=d[1,2]-slope_T2*d[1,0]
	dT1_feval=slope_T1*f_eval+int_T1
	dT2_feval=slope_T2*f_eval+int_T2
	slope = (dT1_feval-dT2_feval)/(d[0,1]-d[0,2])
	intercept = dT1_feval - slope*d[0,1]
	return [slope, intercept]
	
def read_loss_in(dirname):
	c = pd.read_csv(dirname+'InputChain.csv', delimiter=',')
	d = np.transpose(c.values)
	compin = d[0,:]
	lossin = d[1,:]
	Tavgin = (d[2,:]+d[3,:])/2
	coax_YNin = d[4,:]
	coax_typein = d[5,:]
	coax_lin = d[6,:]
	for i in range(len(compin)):
		if coax_YNin[i] == 'Y':
			t = readcoax(dirname+coax_typein[i]+'.csv',8)
			lossin[i] = (t[0]*Tavgin[i]+t[1])*coax_lin[i]
	return lossin, Tavgin
	
def read_loss_out(dirname):
	c = pd.read_csv(dirname+'OutputChain.csv',delimiter=',')
	d = np.transpose(c.values)
	compout = d[0,:]
	lossout = d[1,:]
	Tavgout = (d[2,:]+d[3,:])/2
	coax_YNout = d[4,:]
	coax_typeout = d[5,:]
	coax_lout = d[6,:]
	Gtot = 0.
	for i in range(len(compout)):
		if coax_YNout[i] == 'Y':
			t = readcoax(dirname+coax_typeout[i]+'.csv',8)
			lossout[i] = (t[0]*Tavgout[i]+t[1])*coax_lout[i]
	return lossout, Tavgout

def ref_noise_temp(loss,temp,T0,p=False):
	"""Function to refer noise temp through the chain from 300K to mux temp stage.

	Parameters:
	loss (float): list of elements with loss in your input chain
	temp (float): list of physical temperatures of the elements in loss must be same length as loss
	T0 (float): temperature at input to referral chain (for a 50 Ohm terminator thermalized to the room this would be 300K)+

	Returns:
	T (float): Noise temperature referred to output of chain named loss
	"""

	T=T0
	loss_lin = 10.**(-loss/10.)
	for i in range(len(loss)):
		Tp = T*loss_lin[i] + temp[i]*(1-loss_lin[i])
		T=Tp
	if p==True:
		print(T0, 'K Noise Temp Referred to Feedline: ',T,' K')    
	return T
    
def p_referred_pertone(loss, Pin, p=False):
	"""Function to refer power from the input to the cryostat to power at the feedline.

	Parameters:
	loss (float): list of elements with loss in your input chain
	Pin (float): Power into cryostat in dB
	p (bool): Will print referred power if true. Default is False.

	Returns:
	Pout (float): Power at the feedline in dB
	"""

	Pout = Pin - sum(loss)
	if p==True:
		print('Power at Feedline: ',Pout,'dB')
	return Pout
    
def refer_phase_noise(dBc,loss,temp,Pin, Qr, Qc, f0,dfdI, p=False):
	"""Function to Calculate Noise (NEI) from SMuRF DAC Chain

	Parameters:
	dBc (float): Input phase noise from DAC chain (no default but typical for SMuRF is -101 dBc/Hz)
	loss (float): list of elements with loss in your input chain
	temp (float: list of physical temperatures of the elements in loss must be same length as loss
	Pin (float): Power into cryostat in dB
	Q (float): Resonator quality factor, default is Qtyp = 50,000.
	f0 (float): Resonance frequency, default is fmax = 8 GHz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA, default is dfdItyp = 4.59e-2 Hz/pA.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from the DAC chain in pA/rt(Hz).
	"""

	k_b = 1.38e-23;
	T_out = (10.**((Pin-dBc)/10.)*.001)/k_b
	T_feed = ref_noise_temp(loss,temp,T_out)
	P_feed = p_referred_pertone(loss,Pin)
	dBc_feed = 10*np.log10(k_b*T_feed/.001) - P_feed
	Sphi = 10.**(dBc_feed/10.)
	Sf = Sphi*(f0*(Qc-Qr)/(2.*Qr**2.))**2. #This expression is from derivatives of Zmuidizinas Review Paper
	NEI = np.sqrt(Sf/(dfdI**2.))
	if p == True:
		print('Phase Noise of: ',dBc, 'dBc/Hz referred to NEI:',NEI)
	return NEI


def refer_phase_noise_to_K(dBc,loss,temp,Pin, Qr, Qc, f0,dfdI, p=False):
	"""Function to Calculate Noise (NEI) from SMuRF DAC Chain

	Parameters:
	dBc (float): Input phase noise from DAC chain (no default but typical for SMuRF is -101 dBc/Hz)
	loss (float): list of elements with loss in your input chain
	temp (float: list of physical temperatures of the elements in loss must be same length as loss
	Pin (float): Power into cryostat in dB
	Q (float): Resonator quality factor, default is Qtyp = 50,000.
	f0 (float): Resonance frequency, default is fmax = 8 GHz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA, default is dfdItyp = 4.59e-2 Hz/pA.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from the DAC chain in pA/rt(Hz).
	"""

	k_b = 1.38e-23;
	T_out = (10.**((Pin-dBc)/10.)*.001)/k_b
	T_feed = ref_noise_temp(loss,temp,T_out)
	P_noise_feed = T_feed*k_b
	dip = 1.-(Qr/Qc)
	P_noise_HEMT_input = dip*P_noise_feed
	K = P_noise_HEMT_input/k_b
	return K
	 
def refer_dBc_simple(dBc,Qr, Qc, f0,dfdI, p=False):
	"""Same function as refer_phase_noise but doesn't refer through input attenuation chain first assumes dBc is constant to the feedline, typically the case for us.

	Parameters:
	dBc (float): Input phase noise from DAC chain, default is -101 dBc/Hz.
	Q (float): Resonator quality factor, default is Qtyp = 50,000.
	f (float): Resonance frequency, default is fmax = 8 GHz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA, default is dfdItyp = 4.59e-2 Hz/pA.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from the DAC chain in pA/rt(Hz).
	"""

	Sphi = 10**(dBc/10)
	Sf = Sphi*(f0*(Qc-Qr)/(2*Qr**2))**2 #This expression is from derivatives of Zmuidizinas Review Paper
	NEI = np.sqrt(Sf/(dfdI**2.))
	if p == True:
		print('Phase Noise of: ',dBc, 'dBc/Hz referred to NEI:',NEI)
	return NEI
    
'''def TN_to_NEI_Mates(T,LJ=LJ,f0=fmin,Min=Lsqin,alpha=1./2.,p=False):
	"""This is the referral used in Mates thesis equation 2.163

	Parameters:
	T (float): Noise temperature at output of feedline (i.e. input to HEMT).
	LJ (float): Josephson inductance of RF-SQUID. Default is LJtyp = 6.69e-11 henries.
	f0 (float): Resonance frequency, default is fmin = 4 GHz.
	Min (float): Mutual inductance between SQUID and TES input. Default is Lsqin = 2.28e-10 henries.
	alpha (float): Degradation factor from flux ramping and not biasing on max slope. Default is 1/2.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from noise temp at the feedline (i.e. HEMT input) in pA/rt(Hz).
	"""

	k_b = 1.38e-23;
	NEI = (1./np.sqrt(alpha))*np.sqrt((4.*k_b*T*LJ)/(pi*f0))*(1/Min)
	if p == True:
		print('Noise temp', T, 'K @ LNA-1 input referred to NEI:',np.round(NEI/1e-12,2),'pA/rt(Hz)')
	return NEI/1e-12'''
    
def TN_to_NEI_MSF(T,Pfeed,Qr,Qc,f0,dfdI,p=False):
	"""This is the referral that I calculate in a memo I wrote called revisiting Mates 2.163 where I recalculate the noise referral to include the scaling with Q in input tone power.

	Parameters:
	T (float): Noise temperature at output of feedline (i.e. input to HEMT).
	Pfeed (float): Power into the feedline on the mux chip in dB.
	Qc (float): Coupling quality factor of resonator.
	Qi (float): Internal quality factor of resonator.
	f0 (float): Resonance frequency, default is fmax = 8 GHz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA, default is dfdItyp = 4.59e-2 Hz/pA.
	Z0 (float): Feedline characteristic impedance. Default is Z0typ = 50 Ohms.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from noise temp at the feedline (i.e. HEMT input) in pA/rt(Hz).
	"""
	
	k_b = 1.38e-23;
	Z0 = 50.
	'''S21min = (1. - (Qr/Qc))
	Vnoise_Im = np.sqrt(4*k_b*T*Z0)/2. #The divide by 2 comes from the fact that only imaginary and real parts are equal but only imag is contributing
	dImS21df = (2.*Qr**2)/(Qc*f0)
	dVdI = S21min*np.sqrt(0.001*(10**(Pfeed/10.))*Z0)*dImS21df*dfdI
	NEI = Vnoise_Im/dVdI'''
	Pnoise_Im = 10*np.log10((4*k_b*T*Z0/np.sqrt(2.))/.001)
	dBc_HEMT = Pnoise_Im - Pfeed
	NEI_HEMT = refer_dBc_simple(dBc_HEMT,Qr,Qc,f0,dfdI)
	return NEI_HEMT

def TN_to_NEI_SWH(T,Pfeed,Qr,f0,dfdI,p=False):
	"""This is the referral that I calculate in a memo I wrote called revisiting Mates 2.163 where I recalculate the noise referral to include the scaling with Q in input tone power.

	Parameters:
	T (float): Noise temperature at output of feedline (i.e. input to HEMT).
	Pfeed (float): Power into the feedline on the mux chip in dB.
	Qc (float): Coupling quality factor of resonator.
	Qi (float): Internal quality factor of resonator.
	f0 (float): Resonance frequency, default is fmax = 8 GHz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA, default is dfdItyp = 4.59e-2 Hz/pA.
	Z0 (float): Feedline characteristic impedance. Default is Z0typ = 50 Ohms.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from noise temp at the feedline (i.e. HEMT input) in pA/rt(Hz).
	"""
	
	k_b = 1.38e-23;
	Pfeed = 10.**(Pfeed/10.)*.001
	Sf = (1./2.)*(f0**2/Qr**2)*(k_b*T/Pfeed)
	NEI_HEMT = np.sqrt(Sf/dfdI**2)
	return NEI_HEMT

def Pseudo_Noise_Floor_to_K(Pfeed,loss,Qr,Qc,f0,dfdI,Ntones):
	"""This is the referral used in Mates thesis equation 2.163

	Parameters:
	Pfeed (float): Power at the feedline in a single tone in dBm 
	loss (float): list of elements with loss in your input chain
	Dip (float): Typical depth of resonator dip in dB
	Ntones (float): Number of tones into the cryostat. Default is Ntonestyp = 2000.

	Returns:
	PNF_NEI (float): Noise equivalent current at the TES from the Pseudo Noise Floor in pA/rt(Hz).
	"""

	k_b = 1.38e-23;
	BWNL = 10*np.log10(12e9)
	NNLtones = 10*np.log10(((Ntones**2)*(Ntones-1))/2)
	Dip = 20*np.log10(1.-(Qr/Qc))
	P_LNA1 = Pfeed+Dip
	P_perspur_out = 3.*P_LNA1+100. #This comes from IP3 Measurements
	P_NL_out = P_perspur_out+NNLtones
	PNP_out = P_NL_out-BWNL
	PNP_in = PNP_out - 39 
	#This comes from the Gain of the 2 stage amplifiers shouldn't hard code this in the long run
	PNF_K = (10.**(PNP_in/10.)*.001)/k_b
	return PNF_K

def quad_sum(a,b):
	"""Adds two noise sources (a and b) in quadrature.

	Parameters:
	a (float): Noise source 1 in pA/rt(Hz)
	b (float): Noise source 2 in pA/rt(Hz)

	Returns:
	Out (float): Quadrature sum of input noise sources in pA/rt(Hz).
	"""

	out=np.sqrt(a**2+b**2)
	return out

def Total_Chain_Noise_Temp(loss_out,Tavg_out,LNA_TN):
	"""Takes in a chain of output amplifiers and attenuators and calculates the total noise temperature referred to the output of the mux chip feedline.

	Parameters:
	loss_out (float): List of elements with loss (and negative loss is gain) in your output chain
	LNA_TN (float): A list of the noise temperature of all of the amplifiers used in the output chain. Default is LNA_TNtyp = [2.1,40.,191.] which corresponds to LNF 4K Low Gain Amp, ASU 40K Amp, and MCL ZX60-83LN-S+ at 300K.

	Returns:
	TNout (float): Total noise temperature from the output amplifier chain referred to the output of the mux chip feedline in Kelvin.
	"""

	TNtot = 0.
	Gaintot_lin = 10**(np.sum((-loss_out))/10.)
	j=0
	loss_lin = 10.**(-loss_out/10.)
	for i in range(len(loss_out)):
		if loss_out[i]>0:
			TNtot = TNtot*loss_lin[i] + Tavg_out[i]*(1.-loss_lin[i])
		if loss_out[i]<0:
			TNtot = (TNtot+LNA_TN[j])*loss_lin[i]
			j=j+1;
	TNout = TNtot/Gaintot_lin
	return TNout

def Calc_All_Noise(loss, temp, loss_out,Pin,LNA_TN=LNA_TNtyp, p=False):
	"""
	Parameters:
	loss (float): list of elements with loss in your input chain
	temp (float: list of physical temperatures of the elements in loss must be same length as loss
	loss_out (float): List of elements with loss (and negative loss is gain) in your output chain
	Pin (float): Power into cryostat in dB
	LNA_TN (float): A list of the noise temperature of all of the amplifiers used in the output chain. Default is LNA_TNtyp = [2.1,40.,191.] which corresponds to LNF 4K Low Gain Amp, ASU 40K Amp, and MCL ZX60-83LN-S+ at 300K.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI_tot (float): Quadrature sum of all calculated noise sources referred to TES in pA/rt(Hz).
	"""   

	#Should fix this function to make sure that all inputs are exposed and not hard coded.
	N_PNF = Pseudo_Noise_Floor(Pin,loss,10)
	N_DAC = refer_phase_noise(101,loss,temp,Pin)
	N_300K = 0.#TN_to_NEI(ref_noise_temp(lossin,Tavgin,300.))
	N_HEMT_TN = TN_to_NEI(Total_Chain_Noise(lossout,LNA_TN=LNA_TN))
	N_TES_bias = 0. #This number is just by design
	NEI_tot = quad_sum(quad_sum(quad_sum(N_PNF,N_DAC),quad_sum(N_300K,N_HEMT_TN)),N_TES_bias)
	if p==True:
		print(N_PNF,N_DAC,N_300K,N_HEMT_TN)
	return NEI_tot
