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


#Load in data from textfile, can download zipped version from bolowiki:
#http://bolo.berkeley.edu/~chinoney/data/trash_can/2020-10-22/pb1_weather_temperature_outside_btw_20120315-20171114.txt.bz2
fname = '/home/msilvafe/Downloads/pb1_weather_temperature_outside_btw_20120315-20171114.txt'
hour_per_chunk = 24
make_plot = False
make_cut_chan_plot = True

out_fname = f'/home/msilvafe/Documents/weather_data_{hour_per_chunk}hrchunks_2012_to_2017.pkl'
if os.path.exists(out_fname):
    #Load back in the dictionary if you've already created it before.
    day_split = pkl.load(open(out_fname,'rb'))
    span_days = day_split['span_days']
    hour_per_chunk = day_split['hour_per_chunk']
    day_s = hour_per_chunk*3600
    typ_len = day_split['typ_len']
    abs_temp = day_split['abs_temp']
else:
    #Load data
    temp_12_to_17 = np.genfromtxt(open(fname))
    #Convert chunk time from hours to seconds
    day_s = hour_per_chunk*3600

    #Print start/end date of dataset and store number of chunks in total time span
    t_start = datetime.datetime.fromtimestamp(temp_12_to_17[0,0]).strftime("%a %b %d %H:%M:%S %Y")
    t_end = datetime.datetime.fromtimestamp(temp_12_to_17[-1,0]).strftime("%a %b %d %H:%M:%S %Y")
    span_days = int((temp_12_to_17[-1,0] - temp_12_to_17[0,0])/(day_s))

    print(f'Data spans from {t_start} to {t_end}, total of {span_days} x {hour_per_chunk} hr chunks')

    #Array of time in seconds relative to first data point
    rel_time = temp_12_to_17[:,0]-temp_12_to_17[0,0]

    #Array of absolute rel_time
    abs_time = temp_12_to_17[:,0]

    #Array of temps in Celsius
    abs_temp = temp_12_to_17[:,1]

    #Chunk data
    day_split = {}
    l_tods = []
    for i in tqdm(range(span_days)):
        day_split[i] = {}
        start_idx = np.argmin(np.abs(rel_time-i*day_s))
        end_idx = np.argmin(np.abs(rel_time-(i+1)*day_s))
        #time range closest to a chunk time
        t_ran = rel_time[np.argmin(np.abs(rel_time-(i+1)*day_s))]-rel_time[np.argmin(np.abs(rel_time-i*day_s))]

        #Store time points for each chunk
        day_split[i]['rel_t'] = rel_time[start_idx:end_idx]-rel_time[start_idx]
        #Store the length of each TOD
        l_tods.append(len(day_split[i]['rel_t']))
        #Store the temperature points for each chunk
        day_split[i]['abs_temp'] = abs_temp[start_idx:end_idx]

    #Identify typical i.e. highest probability length of a chunk.
    l_tods = np.asarray(l_tods)
    n, bins, _ = plt.hist(l_tods,bins = np.max(l_tods))
    plt.close()
    typ_len = bins[np.argmax(n)]
    print(f'Typical number of points in a TOD is {typ_len}')

    #Store these values to a dictionary and save to disk
    day_split['span_days'] = span_days
    day_split['hour_per_chunk'] = hour_per_chunk
    day_split['typ_len'] = typ_len
    day_split['abs_temp'] = abs_temp
    pkl.dump(day_split,open(out_fname,'wb'))

max_ts = []
min_ts = []
n_tods = 0
t_interps = np.arange(0,day_s,60)
len_trunc = []
for i in tqdm(range(span_days)):
    #Cut TODs that are longer or much shorter than the typical length of a chunk
    #These TODs are missing data.
    if (len(day_split[i]['rel_t']) < typ_len - 5) or (len(day_split[i]['rel_t']) > typ_len + 1):
        day_split[i]['bad_flag'] = True
        continue

    #Cut tods that have big spikes/derivatives in them
    if np.max(np.abs(np.diff(day_split[i]['abs_temp'])/np.diff(day_split[i]['rel_t']))) > 0.01:
        day_split[i]['bad_flag'] = True
        continue

    #Cut tods that have more than have of their data points pinned to one value
    #i.e. have a zero derivative for more than half of the TOD.
    low_diff_idx = np.where(np.abs(np.diff(day_split[i]['abs_temp'])/np.diff(day_split[i]['rel_t'])) >1e-12)
    if len(low_diff_idx[0]) <= typ_len//2:
        day_split[i]['bad_flag'] = True
        continue

    #Remove all of the repeating points from the remaining TODs.
    day_split[i]['rel_t_trunc'] = day_split[i]['rel_t'][low_diff_idx[0]]
    day_split[i]['abs_temp_trunc'] = day_split[i]['abs_temp'][low_diff_idx[0]]

    #Resample all of the remaining TODs onto a uniform time grid ranging from
    #min_ts to max_ts with time between samples of 60 seconds
    max_ts.append(np.max(day_split[i]['rel_t_trunc']))
    min_ts.append(np.min(day_split[i]['rel_t_trunc']))
    len_trunc.append(len(day_split[i]['rel_t_trunc']))
    cs = interp1d(day_split[i]['rel_t_trunc'],day_split[i]['abs_temp_trunc'])

    #If resampled time points lie outside of data then truncate which points
    #in the resampling array we use for the interpolation (so that we don't
    #interpolate outside the range where there's data)
    min_interp = 0
    max_interp = -1
    if t_interps[0] < np.min(day_split[i]['rel_t_trunc']):
        min_interp = np.argmin(np.abs(t_interps-np.min(day_split[i]['rel_t_trunc'])))
        if t_interps[min_interp] < np.min(day_split[i]['rel_t_trunc']):
            min_interp += 1
    if t_interps[-1] > np.max(day_split[i]['rel_t_trunc']):
        max_interp = np.argmin(np.abs(t_interps-np.max(day_split[i]['rel_t_trunc'])))
        if t_interps[max_interp] > np.max(day_split[i]['rel_t_trunc']):
            max_interp -= 1
    day_split[i]['temp_interp'] = cs(t_interps[min_interp:max_interp])
    day_split[i]['time_interp'] = t_interps[min_interp:max_interp]
    day_split[i]['bad_flag'] = False
    n_tods += 1

print(f'{n_tods} passed quality cuts of {span_days} total.')

#TOD manipulation is done, now the rest is plotting and PSDs.
detrend = 'constant'
fs = 1/np.diff(t_interps)[0]
dfs = []
f_min = []
f_max = []
for i in tqdm(range(span_days)):
    if day_split[i]['bad_flag']:
        if make_cut_chan_plot:
            plt.figure(1)
            #plt.subplot(3,1,3)
            plt.plot(day_split[i]['rel_t'],day_split[i]['abs_temp'])
        continue
    if make_plot:
        plt.figure(2)
        #plt.subplot(3,1,3)
        plt.plot(day_split[i]['time_interp'],day_split[i]['temp_interp'])
    #Use maximum nperseg
    nperseg = len(day_split[i]['temp_interp'])
    f,Pxx = signal.welch(day_split[i]['temp_interp'], nperseg=nperseg,fs=fs, detrend=detrend)
    Pxx = np.sqrt(Pxx)
    day_split[i]['f'] = f
    day_split[i]['Pxx'] = Pxx
    dfs.append(np.diff(f)[0])
    f_min.append(np.min(f))
    f_max.append(np.max(f))
    if make_plot:
        plt.figure(3)
        #plt.subplot(3,1,3)
        plt.loglog(f,Pxx,'b-',alpha = 0.7)

#Since TODs were all slightly different lengths the PSDs will also be
#different lengths so in order to average them we need to rebin the frequencies
#sum up the power in each PSD that falls within those bins then divide by the
#number of things in each bin.

#This is our resampled frequency bins for our PSD average
dfs_max = np.max(np.asarray(dfs))
f_max_max = np.max(np.asarray(f_max))
f_min_min = np.min(np.asarray(f_min))
f_resample = np.arange(f_min_min,f_max_max,dfs_max)

#Pxx_avg will be our averaged PSD and n_avg is the number of counts in each bin
Pxx_avg = np.zeros(len(f_resample))
n_avg = np.zeros(len(f_resample))

for i in tqdm(range(span_days)):
    if day_split[i]['bad_flag']:
        continue
    for idx,f_i in enumerate(f_resample):
        Pxx_avg[idx] += np.sum(day_split[i]['Pxx'][(np.where((day_split[i]['f'] >= f_i) & (day_split[i]['f'] < (f_i + dfs_max))))])
        n_avg[idx] += len(day_split[i]['Pxx'][(np.where((day_split[i]['f'] >= f_i) &  (day_split[i]['f'] < (f_i + dfs_max))))])
Pxx_avg = Pxx_avg/n_avg
if make_plot:
    plt.figure(3)
    #plt.subplot(3,1,3)
    plt.loglog(f_resample,Pxx_avg,'r--',linewidth = 3,alpha = 0.6)

#Now dump all of the other TODs and PSDs that were intermediate to the pickle file
day_split['f_resample'] = f_resample
day_split['Pxx_avg'] = Pxx_avg
pkl.dump(day_split,open(out_fname,'wb'))
