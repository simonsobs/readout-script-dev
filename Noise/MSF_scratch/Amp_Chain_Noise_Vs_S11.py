import Noise_Referral as nr
import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
plt.ion()

dirname = '/home/arnoldlabws3/Scratch/MSF/readout-script-dev/Noise/'
TN_list = []
for s11 in np.arange(-25,-2,1):
	print('S11 =',s11,'dB')
	temp1, temp2 = nr.read_loss_out(dirname = dirname)
	s11_list = s11*np.ones(len(temp1)-1)
	lossout, Tavgout = nr.read_loss_out_with_s11(dirname = dirname, s11 = s11_list)
	TN_temp = nr.Amp_Chain_Noise_Temp(loss_out = lossout,Tavg_out = Tavgout,LNA_TN = [2.1,40.,191.])
	TN_list.append(TN_temp)
	
plt.semilogy(np.arange(-25,-2,1),TN_list)
plt.xlabel('S11 at each interface [dB]',fontsize = 16)
plt.ylabel('$T_{Noise}$ @ HEMT input',fontsize = 16)
plt.grid()
