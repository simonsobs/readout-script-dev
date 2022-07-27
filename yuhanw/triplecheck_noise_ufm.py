'''
Code written in Jan 2022 by Yuhan Wang
take noise measurements at SC, in transition and Normal stage
and confirm with some baseline result
UFM needs to be tuned and where to be biased needs to be measured

Starting from SC to normal and to in transition and stays there
'''

import matplotlib
matplotlib.use('Agg')

import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.det_config  import DetConfig
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
import scipy.signal as signal

import warnings
warnings.filterwarnings("ignore")

slot_num = 2
nperseg = 2**16
# hard coded (for now) variables
stream_time = 120
bias_array = np.array([4.1, 3.9, 5.5, 5.5, 4.1, 4. , 5.4, 5.7, 4.,  3.7, 5.3, 5.3, 0.,  0.,  0.])

fs = S.get_sample_frequency()

bias_array_sc = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
S.set_tes_bias_bipolar_array(bias_array_sc)
S.get_tes_bias_bipolar_array()
print(S.get_tes_bias_bipolar_array())
print('waiting 60s')
time.sleep(60)

dat_path_sc = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 20, cool_wait= 3, high_current_mode=False, overbias_voltage= 15)
print(S.get_tes_bias_bipolar_array())
## hard coding wait time here
print('waiting 300s')
time.sleep(300)
dat_path_normal = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()

S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 20, cool_wait= 3, high_current_mode=False, overbias_voltage= 15)
# bias_array = np.array([6.6,6.4,10.5,10.9,6.7,6.6,10.8,11.1,6.7,6.6,11,11.1,0,0,0])
S.set_tes_bias_bipolar_array(bias_array)
print(S.get_tes_bias_bipolar_array())

print('waiting 300s')
time.sleep(300)
dat_path_intran = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()


# dat_path_sc = '/data/smurf_data/20220118/crate1slot4/1642541596/outputs/1642552105.dat'
# dat_path_normal = '/data/smurf_data/20220118/crate1slot4/1642541596/outputs/1642552478.dat'
# dat_path_intran = '/data/smurf_data/20220118/crate1slot4/1642541596/outputs/1642552845.dat'



timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_sc, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')
# hard coded variables
bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
sample_nums = np.arange(len(phase[0]))
t_array = sample_nums / fs
# reorganize the data by band then channel
stream_by_band_by_channel_sc = {}
for band, channel in zip(bands, channels):
    if band not in stream_by_band_by_channel_sc.keys():
        stream_by_band_by_channel_sc[band] = {}
    ch_idx = mask[band, channel]
    stream_by_band_by_channel_sc[band][channel] = phase[ch_idx]


timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_normal, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')
# hard coded variables
bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
sample_nums = np.arange(len(phase[0]))
t_array = sample_nums / fs
# reorganize the data by band then channel
stream_by_band_by_channel_normal = {}
for band, channel in zip(bands, channels):
    if band not in stream_by_band_by_channel_normal.keys():
        stream_by_band_by_channel_normal[band] = {}
    ch_idx = mask[band, channel]
    stream_by_band_by_channel_normal[band][channel] = phase[ch_idx]


timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_intran, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path}')
# hard coded variables
bands, channels = np.where(mask != -1)
phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
sample_nums = np.arange(len(phase[0]))
t_array = sample_nums / fs
# reorganize the data by band then channel
stream_by_band_by_channel_intran = {}
for band, channel in zip(bands, channels):
    if band not in stream_by_band_by_channel_intran.keys():
        stream_by_band_by_channel_intran[band] = {}
    ch_idx = mask[band, channel]
    stream_by_band_by_channel_intran[band][channel] = phase[ch_idx]





# detrend='constant'
# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_sc.keys()):
#     stream_single_band = stream_by_band_by_channel_sc[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for channel in sorted(stream_single_band.keys()):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
        
#         ax_this_band.loglog(f, Pxx, color='C0', alpha=0.05)
#     band_yield = len(stream_single_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('Amp [pA/rtHz]')
#     ax_this_band.grid()
#     ax_this_band.set_title(f'band {band} yield {band_yield}')
#     ax_this_band.set_ylim([1,1e6])

#     stream_single_band = stream_by_band_by_channel_normal[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for channel in sorted(stream_single_band.keys()):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
        
#         ax_this_band.loglog(f, Pxx, color='C1', alpha=0.05)
#     band_yield = len(stream_single_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('Amp [pA/rtHz]')
#     ax_this_band.grid()
#     ax_this_band.set_title(f'band {band} yield {band_yield}')
#     ax_this_band.set_ylim([1,1e6])

#     stream_single_band = stream_by_band_by_channel_intran[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for channel in sorted(stream_single_band.keys()):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
        
#         ax_this_band.loglog(f, Pxx, color='C2', alpha=0.05)
#     band_yield = len(stream_single_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('Amp [pA/rtHz]')
#     ax_this_band.grid()
#     ax_this_band.set_title(f'band {band} yield {band_yield}')
#     ax_this_band.set_ylim([1,1e6])

#     ax_this_band.axhline(140,linestyle='--', alpha=0.6,color = 'C1',label = '140 pA/rtHz')
#     ax_this_band.axhline(60,linestyle='--', alpha=0.6,color = 'C1',label = '60 pA/rtHz')
#     ax_this_band.plot(0,0,color = 'C0', label = 'SC')
#     ax_this_band.plot(0,0,color = 'C1', label = 'normal')
#     ax_this_band.plot(0,0,color = 'C2', label = 'In transition')

# save_name = f'{start_time}_band_psd_stack.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))


start_time = dat_path_sc[-14:-4]


detrend='constant'
fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel_sc.keys()):
    all_f_sc = []
    all_Pxx_sc = []
    all_f_normal = []
    all_Pxx_normal = []
    all_f_intran = []
    all_Pxx_intran = []
    stream_single_band = stream_by_band_by_channel_sc[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        if Pxx[0] < 5e6:
            all_f_sc.append(f)
            all_Pxx_sc.append(Pxx)

    stream_single_band = stream_by_band_by_channel_normal[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        if Pxx[0] < 5e6:
            all_f_normal.append(f)
            all_Pxx_normal.append(Pxx)       
            
 

    stream_single_band = stream_by_band_by_channel_intran[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        if Pxx[0] < 5e6:
            all_f_intran.append(f)
            all_Pxx_intran.append(Pxx)  
            
    ax_this_band.loglog(np.median(all_f_sc, axis = 0), np.median(all_Pxx_sc, axis = 0), color='C0')
    ax_this_band.loglog(np.median(all_f_normal, axis = 0), np.median(all_Pxx_normal, axis = 0), color='C1')
    ax_this_band.loglog(np.median(all_f_intran, axis = 0), np.median(all_Pxx_intran, axis = 0), color='C2')

    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('Frequency [Hz]')
    ax_this_band.set_ylabel('Amp [pA/rtHz]')
    ax_this_band.grid()
    ax_this_band.set_title(f'band {band} yield {band_yield}')
    ax_this_band.set_ylim([10,1e5])

    ax_this_band.axhline(140,linestyle='--', alpha=0.6,color = 'C4',label = '140 pA/rtHz')
    ax_this_band.axhline(60,linestyle='--', alpha=0.6,color = 'C4',label = '60 pA/rtHz')
    ax_this_band.plot(0,0,color = 'C0', label = 'SC')
    ax_this_band.plot(0,0,color = 'C1', label = 'normal')
    ax_this_band.plot(0,0,color = 'C2', label = 'In transition')
    ax_this_band.axvline(1.4,linestyle='--', alpha=0.6,label = '1.4 Hz',color = 'C4')
    ax_this_band.axvline(60,linestyle='--', alpha=0.6,label = '60 Hz',color = 'C4')
    print('finished: {}'.format(band))

    ax_this_band.legend()

save_name = f'{start_time}_band_psd_stack_median.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))









