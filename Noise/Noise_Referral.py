####Import Packages
import numpy as np
import pandas as pd
from numpy import genfromtxt
'''
This is a package that can be imported to calculate noise terms in the mu-mux readout system.

Example use:
import Noise_Referral as nr

loss_in, Temp_in = nr.read_loss_in(dirname)
power_into_cryostat = -38 #in dBm
power_at_feedline = nr.p_referred_pertone(loss_in, power_into_cryostat)
print(power_at_feedline)

returns: -70.20778279626711
'''

####Directory that dewar configuration csv files are saved in
dirname = '/home/msilvafe/Documents/SO/umux/Noise/'


####Define Functions

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
	"""Function for taking in csv describing input coax + attenuator.

	Parameters:
	dirname (str): directory name where InputChain.csv file is stored

	Returns:
	lossin (float list): list of the loss of each attenuator (and coax)
	Tavgin (float list): list of the temperature of each attenuator (and coax)
	"""
	
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
	"""Function for taking in csv describing output coax + amplifiers.

	Parameters:
	dirname (str): directory name where OutputChain.csv file is stored

	Returns:
	lossout (float list): list of the loss of each coax (and amp)
	Tavgin (float list): list of the temperature of each coax (and amp)
	"""
	
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
	"""Function to refer power from the input of the cryostat to power at the feedline.

	Parameters:
	loss (float): list of elements with loss in your input chain
	Pin (float): Power into cryostat in dBm
	p (bool): Will print referred power if true. Default is False.

	Returns:
	Pout (float): Power at the feedline in dBm
	"""

	Pout = Pin - sum(loss)
	if p==True:
		print('Power at Feedline: ',Pout,'dB')
	return Pout

def refer_phase_noise_to_K(dBc,loss,temp,Pin, Qr, Qc, f0,dfdI, p=False):
	"""Function to Calculate Noise (NEI) from SMuRF DAC Chain

	Parameters:
	dBc (float): Input phase noise (positive number) from DAC chain (no default but typical for SMuRF is 101 dBc/Hz)
	loss (float): list of elements with loss in your input chain
	temp (float: list of physical temperatures of the elements in loss must be same length as loss
	Pin (float): Power into cryostat in dBm
	Qr (float): Resonator quality factor.
	Qc (floar): Coupling quality factor.
	f0 (float): Resonance frequency in Hz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA.
	p (bool): Will print referred phase noise to Kelvin if True. Default is False.

	Returns:
	NEI (float): Noise temperature at the input to the output amplifier chain from the DAC chain in pA/rt(Hz).
	"""

	k_b = 1.38e-23;
	T_out = (10.**((Pin-dBc)/10.)*.001)/k_b
	T_feed = ref_noise_temp(loss,temp,T_out)
	P_noise_feed = T_feed*k_b
	dip = 1.-(Qr/Qc)
	P_noise_HEMT_input = dip*P_noise_feed
	K = P_noise_HEMT_input/k_b
	return K

def TN_to_NEI(T,Pfeed,Qr,Qc,f0,dfdI,p=False):
	"""This is the referral that I calculate in a memo using the derivative of S21 with respect to frequency evaluated at the minimum of the resonator.

	Parameters:
	T (float): Noise temperature at output of feedline (i.e. input to HEMT).
	Pfeed (float): Power into the feedline on the mux chip in dBm.
	Qr (float): Resonator quality factor.
	f0 (float): Resonance frequency in Hz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI (float): Noise equivalent current at the TES from noise temp at input to the amplifier chain in pA/rt(Hz).
	"""
	
	k_b = 1.38e-23;
	Pfeed = 10.**(Pfeed/10.)*.001
	Sf = ((Qc**2*f0**2)/(8*Qr**4))*(k_b*T/Pfeed)
	NEI_HEMT = np.sqrt(Sf/dfdI**2)
	return NEI_HEMT

def Pseudo_Noise_Floor_to_K(Pfeed,Qr,Qc,f0,dfdI,Ntones):
	"""This is a calculation of the noise due to 3rd order nonlinear products generated by the output amplifier chain referred to noise temperature in Kelvin at the input to the output amplifier chain.

	Parameters:
	Pfeed (float): Power at the feedline to the mux chip (already referred from room temperature using p_referred_pertone) in a single tone in dBm 
	Qr (float): Resonator quality factor.
	Qc (floar): Coupling quality factor.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA.
	Ntones (float): Number of tones into the cryostat.

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
	PNP_in = PNP_out - 39 #This comes from the Gain of the 2 stage amplifiers shouldn't hard code this in the long run
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

def Amp_Chain_Noise_Temp(loss_out,Tavg_out,LNA_TN):
	"""Takes in a chain of output amplifiers and attenuators and calculates the total noise temperature referred to the output of the mux chip feedline (or equivalently input to the output amplifier chain).

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

def Calc_All_Noise(dBc,Qr,Qc,f0,dfdI,P_feed,loss_in,temp_in,temp_out,loss_out,Pin_300K,LNA_TN,Ntones, p=False):
	"""
	Parameters:
	dBc (float): Input phase noise (positive number) from DAC chain (no default but typical for SMuRF is 101 dBc/Hz)
	Qr (float): Resonator quality factor.
	Qc (floar): Coupling quality factor.
	f0 (float): Resonance frequency in Hz.
	dfdI (float): Resonator average gain from flux ramp in units of Hz/pA.
	Pfeed (float): Power at the feedline to the mux chip (already referred from room temperature using p_referred_pertone) in a single tone in dBm 
	loss_in (float list): list of the loss of each coax and attenuating element on the input chain
	temp_in (float list) : list of the physical temperature of each coax and attenuating element in input chain
	loss_out (float list): list of the loss of each coax (and amp) on the output chain
	Pin_300K (float): Power into cryostat in dBm
	LNA_TN (float): A list of the noise temperature of all of the amplifiers used in the output chain. Default is LNA_TNtyp = [2.1,40.,191.] which corresponds to LNF 4K Low Gain Amp, ASU 40K Amp, and MCL ZX60-83LN-S+ at 300K.
	Ntones (float): Number of tones into the cryostat.
	p (bool): Will print referred phase noise to NEI if True. Default is False.

	Returns:
	NEI_tot (float): Quadrature sum of all calculated noise sources referred to TES in pA/rt(Hz).
	"""   
	T_DAC = refer_phase_noise_to_K(dBc,loss_in,temp_in,Pin_300K,Qr,Qc,f0,dfdI)
	NEI_DAC = TN_to_NEI_SWH(T_DAC,P_feed,Qr,f0,dfdI)
	T_HEMT = Amp_Chain_Noise_Temp(loss_out,temp_out,LNA_TN)
	NEI_HEMT = TN_to_NEI_SWH(T_HEMT,P_feed,Qr,f0,dfdI)
	T_PNF = Pseudo_Noise_Floor_to_K(P_feed,Qr,Qc,f0,dfdI,Ntones)
	NEI_PNF = TN_to_NEI_SWH(T_PNF,P_feed,Qr,f0,dfdI)
	
	NEI_tot = quad_sum(quad_sum(quad_sum(NEI_PNF,NEI_DAC),NEI_HEMT))
	if p==True:
		print(N_PNF,N_DAC,N_300K,N_HEMT_TN)
	return NEI_PNF,NEI_DAC,NEI_HEMT,NEI_tot
