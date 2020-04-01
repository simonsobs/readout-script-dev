import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/arnoldlabws3/Scratch/MSF/readout-script-dev/princeton/hmccarrick')
import peak_finder as pk
import resonator_model as res
import os
import skrf as rf
from lmfit import Model
plt.ion()

#Specify the directory where your S2P files are
main_dir = '/home/arnoldlabws3/Scratch/MSF/SMB_V2p0_20191204'
#Loops through all of the s2p files in the above directory and gets their filenames in a list
temp = os.listdir(main_dir)
fnames = []
for fl in temp:
	if fl.split('.')[-1] == 's2p':
		fnames.append(fl)

#Creates the folder where the output fitted plots will save
out_dir = main_dir + '/Resonator_Fit_Plots'
if os.path.exists(out_dir) == False:
	os.mkdir(out_dir)

#Creates a dictionary to load all of the s2p data into in network format from python package scikit-rf, and then stitch them all together into one s2p. Can omit these steps if you didn't take your data in multiple frequency sweeps.
C = {}

for f in fnames:
	C[int(f.split('.')[0].split('_')[-1])] = rf.Network(main_dir+'/'+f)

i = 0
for key in C.keys():
	if i == 0:
		net = C[key]
		i = i+1
		continue
	else:
		net = rf.stitch(net,C[key])

#Extracts just the information from the s2p data that you need for fitting and puts them into np arrays
mag = net.s21.s_mag[:,0,0] 
freq = net.frequency.f
real = net.s21.s_re[:,0,0]
imag = net.s21.s_im[:,0,0]

#Finds the peaks for chunking up individual peaks out of the full frequency sweep
freqs, pks = pk.get_peaks(freq,mag,f_delta = 100e3, res_num = 62)

#Loops through found peaks, fits only +/- 400kHz around the peak with the fitting code then plots and saves it. 
results = {}
for fr in freqs:
	f_start = fr-200e3
	f_stop = fr+200e3
	argstart = np.abs(freq-f_start).argmin()
	argstop = np.abs(freq-f_stop).argmin()
	results[fr] = res.full_fit(freq[argstart:argstop],real[argstart:argstop],imag[argstart:argstop])
	plt.figure()
	plt.plot(freq[argstart:argstop],mag[argstart:argstop],'bo',label = 'Data')
	plt.plot(freq[argstart:argstop],np.abs(results[fr].init_fit),'k--',label = 'Initial Guess')
	plt.plot(freq[argstart:argstop],np.abs(results[fr].best_fit),'r-',label = 'Best Fit')
	plt.title('Resonator at '+str(np.round(fr/1e6,0))+' MHz')
	plt.xlabel('Frequency [Hz]',fontsize = 16)
	plt.ylabel('$S_{21}$ Magnitude',fontsize = 16)
	plt.legend(fontsize = 12)
	plt.savefig(out_dir + '/Res_Fit_'+str(round(fr/1e6,2))+'_MHz.png')
	plt.close()

Qs = []
Q_e_reals = []
brs = []
Qis = []

#Organizes fitted results into np arrays and makes histograms.
for key in results.keys():
	Qs.append(results[key].values['Q'])
	Q_e_reals.append(results[key].values['Q_e_real'])
	Qis.append(res.get_qi(results[key].values['Q'],results[key].values['Q_e_real']))
	brs.append(res.get_br(results[key].values['Q'],results[key].values['f_0']))

Qs = np.asarray(Qs)
Qis = np.asarray(Qis)
brs = np.asarray(brs)
plt.subplot(221)
plt.hist(Qs,bins = 20)
plt.xlabel('Q')
plt.subplot(222)
plt.hist(Qis[np.where(Qis<500000)],bins = 20)
plt.xlabel('$Q_i$')
plt.subplot(223)
plt.hist(brs[np.where(brs<250000)],bins = 20)
plt.xlabel('Bandwidth [Hz]')
