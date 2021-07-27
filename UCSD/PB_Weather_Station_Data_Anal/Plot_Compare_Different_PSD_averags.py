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
f_resample = day_split['f_resample']

dfs_max = np.diff(f_resample)[0]
f_max_max = np.max(f_resample)+dfs_max
f_min_min = np.min(f_resample)

Pxx_avg = np.zeros(len(f_resample))
Pxx_avg_nr = np.zeros(len(f_resample))
Pxx_avg_nu = np.zeros(len(f_resample[2:]))
n_avg = np.zeros(len(f_resample))
n_avg_nr = np.zeros(len(f_resample))
n_avg_nu = 0
for i in tqdm(range(day_split['span_days'])):
    if not(day_split[i]['bad_flag']):
        f,Pxx = signal.welch(day_split[i]['abs_temp'],
                nperseg = len(day_split[i]['abs_temp']),
                fs = np.mean(1/np.diff(day_split[i]['rel_t'])),
                detrend = 'constant')
        Pxx = np.sqrt(Pxx)
        for idx,f_i in enumerate(f_resample):
            Pxx_avg[idx] += np.sum(day_split[i]['Pxx'][(np.where((day_split[i]['f'] >= f_i) & (day_split[i]['f'] < (f_i + dfs_max))))])
            n_avg[idx] += len(day_split[i]['Pxx'][(np.where((day_split[i]['f'] >= f_i) &  (day_split[i]['f'] < (f_i + dfs_max))))])
            Pxx_avg_nr[idx] += np.sum(Pxx[(np.where((f >= f_i) & (f < (f_i + dfs_max))))])
            n_avg_nr[idx] += len(Pxx[(np.where((f >= f_i) &  (f < (f_i + dfs_max))))])
    try:
        Pxx = signal.lombscargle(day_split[i]['rel_t'],
                        day_split[i]['abs_temp'],
                        f_resample[2:])
    except ZeroDivisionError:
        continue
    else:
        Pxx = signal.lombscargle(day_split[i]['rel_t'],
                    day_split[i]['abs_temp'],f_resample[2:])
        Pxx_avg_nu += Pxx
        n_avg_nu += 1
Pxx_avg = Pxx_avg[np.where(n_avg != 0)]/n_avg[np.where(n_avg != 0)]
Pxx_avg_nr = Pxx_avg_nr[np.where(n_avg_nr != 0)]/n_avg_nr[np.where(n_avg_nr != 0)]
Pxx_avg_nu = Pxx_avg_nu/n_avg_nu
plt.figure(3)
plt.loglog(f_resample[np.where(n_avg != 0)],
            Pxx_avg[np.where(n_avg != 0)],
            'r--',linewidth = 3,alpha = 0.6)
plt.loglog(f_resample[np.where(n_avg_nr != 0)],
            Pxx_avg_nr[np.where(n_avg_nr != 0)],
            'g--',linewidth = 3,alpha = 0.6)
plt.loglog(f_resample[2:],
            Pxx_avg_nu,
            'b--',linewidth = 3,alpha = 0.6)
