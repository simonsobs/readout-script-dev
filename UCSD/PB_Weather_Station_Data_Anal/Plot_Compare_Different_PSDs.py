import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from tqdm import tqdm
from scipy.interpolate  import interp1d
import os
import pickle as pkl
plt.ion()

hour_per_chunk = 24
out_fname = f'/home/msilvafe/Documents/weather_data_{hour_per_chunk}hrchunks_2012_to_2017.pkl'
outplotdir = '/home/msilvafe/Pictures/Weather_Data_Anal_Plots/Compare_TODs_24Hrs'

day_split = pkl.load(open(out_fname,'rb'))
flags = []
for i in range(day_split['span_days']):
    flags.append(day_split[i]['bad_flag'])

good_tods = np.where(np.bitwise_not(np.asarray(flags)))[0]

for i in good_tods:
    #Plot different TODs to compare
    plt.figure(1,figsize = (20,10))
    plt.subplot(1,2,1)
    plt.plot(day_split[i]['rel_t'],day_split[i]['abs_temp'],lw = 3,alpha = 0.5)
    plt.plot(day_split[i]['rel_t_trunc'],day_split[i]['abs_temp_trunc'],lw = 2, alpha = 0.7)
    plt.plot(day_split[i]['time_interp'],day_split[i]['temp_interp'],lw = 1, alpha = 1)
    plt.savefig
    #Plot different PSDs to compare
    plt.subplot(1,2,2)
    plt.loglog(day_split[i]['f'],day_split[i]['Pxx'],lw = 3,alpha = 0.5,label = 'All Cuts + Resample')
    f_trunc,Pxx_trunc = signal.welch(day_split[i]['abs_temp_trunc'],
            nperseg = len(day_split[i]['abs_temp_trunc']),
            fs = np.mean(1/np.diff(day_split[i]['rel_t_trunc'])),
            detrend = 'constant')
    Pxx_trunc = np.sqrt(Pxx_trunc)
    plt.loglog(f_trunc,Pxx_trunc,lw = 2, alpha = 0.7,label = 'Cuts No Resample')
    f,Pxx = signal.welch(day_split[i]['abs_temp'],
            nperseg = len(day_split[i]['abs_temp']),
            fs = np.mean(1/np.diff(day_split[i]['rel_t'])),
            detrend = 'constant')
    Pxx = np.sqrt(Pxx)
    plt.loglog(f,Pxx,lw = 1, alpha = 1,label = 'No Cuts No Resample')
    plt.legend()
    plt.savefig(f'{outplotdir}/tod_{i}_compare_effect.png')
    plt.close()
