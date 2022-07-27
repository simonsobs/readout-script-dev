'''
Code written in Jan 2022 by Yuhan Wang

Specially made for TSAT Mv9

this code is dumb as you can see at the end, maybe optimize for loop later....

Median noise is the median from 5Hz to 50Hz

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
from scipy.stats import norm
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")

slot_num = 4
nperseg = 2**16
# hard coded (for now) variables
stream_time = 120
bias_array = np.array([6.6,6.4,10.5,10.9,6.7,6.6,10.8,11.1,6.7,6.6,11,11.1,0,0,0])

fmin=5
fmax=50
##HWP spinning frequency
target_freq = 2
## the range to look for highest peak
span = 0.1
## from how high to start look for peak
threshold_peak = 200




# fs = S.get_sample_frequency()

# bias_array_sc = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# S.set_tes_bias_bipolar_array(bias_array_sc)
# S.get_tes_bias_bipolar_array()
# print(S.get_tes_bias_bipolar_array())
# print('waiting 60s')
# time.sleep(60)

# dat_path_sc = S.stream_data_on()
# # collect stream data
# time.sleep(stream_time)
# # end the time stream
# S.stream_data_off()

# S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 18, cool_wait= 3, high_current_mode=False, overbias_voltage= 15)
# bias_array_normal = np.array([18,18,18,18,18,18,18,18,18,18,18,18,0,0,0])
# S.set_tes_bias_bipolar_array(bias_array_normal)
# print(S.get_tes_bias_bipolar_array())
# ## hard coding wait time here
# print('waiting 300s')
# time.sleep(300)
# dat_path_normal = S.stream_data_on()
# # collect stream data
# time.sleep(stream_time)
# # end the time stream
# S.stream_data_off()

# S.overbias_tes_all(bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11], overbias_wait=1, tes_bias= 18, cool_wait= 3, high_current_mode=False, overbias_voltage= 15)
# bias_array = np.array([6.6,6.4,10.5,10.9,6.7,6.6,10.8,11.1,6.7,6.6,11,11.1,0,0,0])
# S.set_tes_bias_bipolar_array(bias_array)
print(S.get_tes_bias_bipolar_array())

# print('waiting 300s')
# time.sleep(300)
dat_path_intran = S.stream_data_on()
# collect stream data
time.sleep(stream_time)
# end the time stream
S.stream_data_off()


# dat_path_sc = '/data/smurf_data/20220121/crate1slot4/1642729924/outputs/1642785652.dat'
# dat_path_normal = '/data/smurf_data/20220121/crate1slot4/1642729924/outputs/1642786084.dat'
# dat_path_intran = '/data/smurf_data/20220121/crate1slot4/1642729924/outputs/1642786516.dat'



# timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_sc, return_tes_bias=True)
# print(f'loaded the .dat file at: {dat_path}')
# # hard coded variables
# bands, channels = np.where(mask != -1)
# phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
# sample_nums = np.arange(len(phase[0]))
# t_array = sample_nums / fs
# # reorganize the data by band then channel
# stream_by_band_by_channel_sc = {}
# for band, channel in zip(bands, channels):
#     if band not in stream_by_band_by_channel_sc.keys():
#         stream_by_band_by_channel_sc[band] = {}
#     ch_idx = mask[band, channel]
#     stream_by_band_by_channel_sc[band][channel] = phase[ch_idx]


# timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_normal, return_tes_bias=True)
# print(f'loaded the .dat file at: {dat_path}')
# # hard coded variables
# bands, channels = np.where(mask != -1)
# phase *= S.pA_per_phi0 / (2.0 * np.pi)  # uA
# sample_nums = np.arange(len(phase[0]))
# t_array = sample_nums / fs
# # reorganize the data by band then channel
# stream_by_band_by_channel_normal = {}
# for band, channel in zip(bands, channels):
#     if band not in stream_by_band_by_channel_normal.keys():
#         stream_by_band_by_channel_normal[band] = {}
#     ch_idx = mask[band, channel]
#     stream_by_band_by_channel_normal[band][channel] = phase[ch_idx]


timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path_intran, return_tes_bias=True)
print(f'loaded the .dat file at: {dat_path_intran}')
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


start_time = dat_path_intran[-14:-4]


detrend='constant'
fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel_sc.keys()):
    all_f_sc = []
    all_Pxx_sc = []
    all_f_normal = []
    all_Pxx_normal = []
    all_f_intran = []
    all_Pxx_intran = []
    wl_sc = []
    wl_normal = []
    wl_intran = []
    # stream_single_band = stream_by_band_by_channel_sc[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_sc.append(f)
    #         all_Pxx_sc.append(Pxx)
    #         wl = np.median(Pxx[fmask])
    #         wl_sc.append(wl)


    # stream_single_band = stream_by_band_by_channel_normal[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_normal.append(f)
    #         all_Pxx_normal.append(Pxx) 
    #         wl = np.median(Pxx[fmask])
    #         wl_normal.append(wl)      
            
 

    stream_single_band = stream_by_band_by_channel_intran[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)
        if Pxx[0] < 5e6:
            all_f_intran.append(f)
            all_Pxx_intran.append(Pxx) 
            wl = np.median(Pxx[fmask])
            wl_intran.append(wl)   
            
    # ax_this_band.loglog(np.median(all_f_sc, axis = 0), np.median(all_Pxx_sc, axis = 0), color='C0')
    # ax_this_band.loglog(np.median(all_f_normal, axis = 0), np.median(all_Pxx_normal, axis = 0), color='C1')
    ax_this_band.loglog(np.median(all_f_intran, axis = 0), np.median(all_Pxx_intran, axis = 0), color='C2')
    # ax_this_band.axhline(np.median(wl_sc), alpha=1,label = 'SC Median WL: {:.2f} pA/rtHz'.format(np.median(wl_sc)),color = 'C0')
    # ax_this_band.axhline(np.median(wl_normal), alpha=1,label = 'Normal Median WL: {:.2f} pA/rtHz'.format(np.median(wl_normal)),color = 'C1')
    ax_this_band.axhline(np.median(wl_intran), alpha=1,label = 'In transition Median WL: {:.2f} pA/rtHz'.format(np.median(wl_intran)),color = 'C2')

    band_yield = len(stream_single_band)
    ax_this_band.set_xlabel('Frequency [Hz]')
    ax_this_band.set_ylabel('Amp [pA/rtHz]')
    ax_this_band.grid(which='both')
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

save_name = f'{start_time}_band_psd_median.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))


detrend='constant'
fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel_sc.keys()):
    all_f_sc = []
    all_Pxx_sc = []
    all_f_normal = []
    all_Pxx_normal = []
    all_f_intran = []
    all_Pxx_intran = []
    wl_sc = []
    wl_normal = []
    wl_intran = []
    all_peak_height_SC = []
    all_peak_height_normal = []
    all_peak_height_intran = []

    all_peak_frequency_SC = []
    all_peak_frequency_normal = []
    all_peak_frequency_intran = []


    # stream_single_band = stream_by_band_by_channel_sc[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_sc.append(f)
    #         all_Pxx_sc.append(Pxx)
    #         wl = np.median(Pxx[fmask])
    #         wl_sc.append(wl)
    #         freq_mask = ((np.abs(f) < target_freq + span) & (np.abs(f) > target_freq - span))
    #         range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
    #         f_range = f[freq_mask]
    #         Pxx_range = Pxx[freq_mask]
    #         peak_freq = np.float(f_range[Pxx_range.argmax()])
    #         peak_height = np.float(Pxx_range[Pxx_range.argmax()])
    #         target_peak_mask = (np.abs(f) == peak_freq)
    #         target_peak_height = Pxx[target_peak_mask]
    #         all_peak_height_SC.append(target_peak_height[0])
    #         all_peak_frequency_SC.append(peak_freq)




    # stream_single_band = stream_by_band_by_channel_normal[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_normal.append(f)
    #         all_Pxx_normal.append(Pxx) 
    #         wl = np.median(Pxx[fmask])
    #         wl_normal.append(wl) 
    #         freq_mask = ((np.abs(f) < target_freq + span) & (np.abs(f) > target_freq - span))
    #         range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
    #         f_range = f[freq_mask]
    #         Pxx_range = Pxx[freq_mask]
    #         peak_freq = np.float(f_range[Pxx_range.argmax()])
    #         peak_height = np.float(Pxx_range[Pxx_range.argmax()])
    #         target_peak_mask = (np.abs(f) == peak_freq)
    #         target_peak_height = Pxx[target_peak_mask]
    #         all_peak_height_normal.append(target_peak_height[0])
    #         all_peak_frequency_normal.append(peak_freq)     
            
 

    stream_single_band = stream_by_band_by_channel_intran[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)
        if Pxx[0] < 5e6:
            all_f_intran.append(f)
            all_Pxx_intran.append(Pxx) 
            wl = np.median(Pxx[fmask])
            wl_intran.append(wl)

            freq_mask = ((np.abs(f) < target_freq + span) & (np.abs(f) > target_freq - span))
            range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
            f_range = f[freq_mask]
            Pxx_range = Pxx[freq_mask]
            peak_freq = np.float(f_range[Pxx_range.argmax()])
            peak_height = np.float(Pxx_range[Pxx_range.argmax()])
            target_peak_mask = (np.abs(f) == peak_freq)
            target_peak_height = Pxx[target_peak_mask]
            all_peak_height_intran.append(target_peak_height[0])
            all_peak_frequency_intran.append(peak_freq)


    # freq_mean_SC,freq_std_SC=norm.fit(all_peak_frequency_SC)
    # freq_mean_normal,freq_std_normal=norm.fit(all_peak_frequency_normal)
    freq_mean_intran,freq_std_intran=norm.fit(all_peak_frequency_intran)



    # ax_this_band.hist(all_peak_height_SC, range=(0,2000),bins=100, color='C0')
    # ax_this_band.hist(all_peak_height_normal, range=(0,2000),bins=100, color='C1')
    ax_this_band.hist(all_peak_height_intran, range=(0,2000),bins=100, color='C2')

    # ax_this_band.axvline(np.median(all_peak_height_SC), alpha=1,label = 'SC Median pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_SC),freq_mean_SC,freq_std_SC),color = 'C0')
    # ax_this_band.axvline(np.median(all_peak_height_normal), alpha=1,label = 'Normal pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_normal),freq_mean_normal,freq_std_normal),color = 'C1')
    ax_this_band.axvline(np.median(all_peak_height_intran), alpha=1,label = 'In transition pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_intran),freq_mean_intran,freq_std_intran),color = 'C2')


    # ax_this_band.axvline(np.median(wl_sc), linestyle='--', alpha=0.6,label = 'SC Median WL: {:.2f} pA/rtHz'.format(np.median(wl_sc)),color = 'C0')
    # ax_this_band.axvline(np.median(wl_normal),  linestyle='--', alpha=0.6,label = 'Normal Median WL: {:.2f} pA/rtHz'.format(np.median(wl_normal)),color = 'C1')
    ax_this_band.axvline(np.median(wl_intran), linestyle='--', alpha=0.6,label = 'In transition Median WL: {:.2f} pA/rtHz'.format(np.median(wl_intran)),color = 'C2')
    ax_this_band.set_title(f'band {band} yield {band_yield}')

    ax_this_band.legend()
    print('finished: {}'.format(band))


save_name = f'{start_time}_pickup_hist.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))




detrend='constant'
fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
for band in sorted(stream_by_band_by_channel_sc.keys()):
    all_f_sc = []
    all_Pxx_sc = []
    all_f_normal = []
    all_Pxx_normal = []
    all_f_intran = []
    all_Pxx_intran = []
    wl_sc = []
    wl_normal = []
    wl_intran = []
    all_peak_height_SC = []
    all_peak_height_normal = []
    all_peak_height_intran = []

    all_peak_frequency_SC = []
    all_peak_frequency_normal = []
    all_peak_frequency_intran = []


    # stream_single_band = stream_by_band_by_channel_sc[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_sc.append(f)
    #         all_Pxx_sc.append(Pxx)
    #         wl = np.median(Pxx[fmask])
    #         wl_sc.append(wl)
    #         freq_mask = ((np.abs(f) < target_freq + span) & (np.abs(f) > target_freq - span))
    #         range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
    #         f_range = f[freq_mask]
    #         Pxx_range = Pxx[freq_mask]
    #         peak_freq = np.float(f_range[Pxx_range.argmax()])
    #         peak_height = np.float(Pxx_range[Pxx_range.argmax()])
    #         target_peak_mask = (np.abs(f) == peak_freq)
    #         target_peak_height = Pxx[target_peak_mask]
    #         all_peak_height_SC.append(target_peak_height[0])
    #         all_peak_frequency_SC.append(peak_freq)




    # stream_single_band = stream_by_band_by_channel_normal[band]
    # ax_this_band = axs[band // 2, band % 2]
    # for channel in sorted(stream_single_band.keys()):
    #     stream_single_channel = stream_single_band[channel]
    #     f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
    #             fs=fs, detrend=detrend)
    #     Pxx = np.sqrt(Pxx)
    #     fmask = (fmin < f) & (f < fmax)
    #     if Pxx[0] < 5e6:
    #         all_f_normal.append(f)
    #         all_Pxx_normal.append(Pxx) 
    #         wl = np.median(Pxx[fmask])
    #         wl_normal.append(wl) 
    #         freq_mask = ((np.abs(f) < target_freq + span) & (np.abs(f) > target_freq - span))
    #         range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
    #         f_range = f[freq_mask]
    #         Pxx_range = Pxx[freq_mask]
    #         peak_freq = np.float(f_range[Pxx_range.argmax()])
    #         peak_height = np.float(Pxx_range[Pxx_range.argmax()])
    #         target_peak_mask = (np.abs(f) == peak_freq)
    #         target_peak_height = Pxx[target_peak_mask]
    #         all_peak_height_normal.append(target_peak_height[0])
    #         all_peak_frequency_normal.append(peak_freq)     
            
 

    stream_single_band = stream_by_band_by_channel_intran[band]
    ax_this_band = axs[band // 2, band % 2]
    for channel in sorted(stream_single_band.keys()):
        stream_single_channel = stream_single_band[channel]
        f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
                fs=fs, detrend=detrend)
        Pxx = np.sqrt(Pxx)
        fmask = (fmin < f) & (f < fmax)
        if Pxx[0] < 5e6:
            all_f_intran.append(f)
            all_Pxx_intran.append(Pxx) 
            wl = np.median(Pxx[fmask])
            wl_intran.append(wl)

            freq_mask = ((np.abs(f) < 2.4 + span) & (np.abs(f) > 2.4 - span))
            range_peaks, _ = find_peaks(Pxx[freq_mask], height=threshold_peak)
            f_range = f[freq_mask]
            Pxx_range = Pxx[freq_mask]
            peak_freq = np.float(f_range[Pxx_range.argmax()])
            peak_height = np.float(Pxx_range[Pxx_range.argmax()])
            target_peak_mask = (np.abs(f) == peak_freq)
            target_peak_height = Pxx[target_peak_mask]
            all_peak_height_intran.append(target_peak_height[0])
            all_peak_frequency_intran.append(peak_freq)


    # freq_mean_SC,freq_std_SC=norm.fit(all_peak_frequency_SC)
    # freq_mean_normal,freq_std_normal=norm.fit(all_peak_frequency_normal)
    freq_mean_intran,freq_std_intran=norm.fit(all_peak_frequency_intran)



    # ax_this_band.hist(all_peak_height_SC, range=(0,2000),bins=100, color='C0')
    # ax_this_band.hist(all_peak_height_normal, range=(0,2000),bins=100, color='C1')
    ax_this_band.hist(all_peak_height_intran, range=(0,2000),bins=100, color='C2')

    # ax_this_band.axvline(np.median(all_peak_height_SC), alpha=1,label = 'SC Median pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_SC),freq_mean_SC,freq_std_SC),color = 'C0')
    # ax_this_band.axvline(np.median(all_peak_height_normal), alpha=1,label = 'Normal pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_normal),freq_mean_normal,freq_std_normal),color = 'C1')
    ax_this_band.axvline(np.median(all_peak_height_intran), alpha=1,label = 'In transition pickup: {:.2f} pA/rtHz at: {:.2f}Hz std: {:.2f}Hz'.format(np.median(all_peak_height_intran),freq_mean_intran,freq_std_intran),color = 'C2')


    # ax_this_band.axvline(np.median(wl_sc), linestyle='--', alpha=0.6,label = 'SC Median WL: {:.2f} pA/rtHz'.format(np.median(wl_sc)),color = 'C0')
    # ax_this_band.axvline(np.median(wl_normal),  linestyle='--', alpha=0.6,label = 'Normal Median WL: {:.2f} pA/rtHz'.format(np.median(wl_normal)),color = 'C1')
    ax_this_band.axvline(np.median(wl_intran), linestyle='--', alpha=0.6,label = 'In transition Median WL: {:.2f} pA/rtHz'.format(np.median(wl_intran)),color = 'C2')
    ax_this_band.set_title(f'band {band} yield {band_yield}')

    ax_this_band.legend()
    print('finished: {}'.format(band))


save_name = f'{start_time}_pickup_hist_2.4.png'
print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name))







# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_sc.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_sc[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=500)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, SC')
#     ax_this_band.set_xlim([0.1,100])
#     ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_SC_logscale.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))



# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_normal.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_normal[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=500)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, normal')
#     ax_this_band.set_xlim([0.1,100])
#     ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_Normal_logscale.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))




# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_intran.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_intran[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=500)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, intran')
#     ax_this_band.set_xlim([0.1,100])
#     ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_intran_logscale.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))


# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_sc.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_sc[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=300)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, SC')
#     ax_this_band.set_xlim([1,10])
#     # ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_SC_50_100.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))



# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_normal.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_normal[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=300)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, normal')
#     ax_this_band.set_xlim([1,10])
#     # ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_Normal_50_100.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))




# fig, axs = plt.subplots(4, 2, figsize=(24, 48), gridspec_kw={'width_ratios': [2, 2]})
# for band in sorted(stream_by_band_by_channel_intran.keys()):
#     all_f = []
#     all_Pxx = []
#     band_index = []
#     stream_single_band = stream_by_band_by_channel_intran[band]
#     ax_this_band = axs[band // 2, band % 2]
#     for index,channel in enumerate(sorted(stream_single_band.keys())):
#         stream_single_channel = stream_single_band[channel]
#         f, Pxx = signal.welch(stream_single_channel, nperseg=nperseg,
#                 fs=fs, detrend=detrend)
#         Pxx = np.sqrt(Pxx)
#         fmask = (fmin < f) & (f < fmax)
#         if Pxx[0] < 5e6:
#             all_f.append(f)
#             all_Pxx.append(list(Pxx))
#             wl = np.median(Pxx[fmask])
#             band_index.append(index)

#     X,Y=np.meshgrid(f,band_index)  
#     cmap=plt.cm.inferno  
#     sc = ax_this_band.pcolormesh(X,Y,all_Pxx,cmap=cmap,vmin=50,vmax=300)
#     bar = fig.colorbar(sc,ax=ax_this_band)
#     ax_this_band.set_xlabel('Frequency [Hz]')
#     ax_this_band.set_ylabel('detector dummy index (arrange with frequency)')
#     # ax_this_band.grid(which='both')
#     ax_this_band.set_title(f'band {band}, intran')
#     ax_this_band.set_xlim([1,10])
#     # ax_this_band.set_xscale('log')
#     print('finished: {}'.format(band))


# save_name = f'{start_time}_band_waterfall_intran_50_100.png'
# print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
# plt.savefig(os.path.join(S.plot_dir, save_name))






# print('SC noise:{}'.format(dat_path_sc))
# print('Normal noise:{}'.format(dat_path_normal))
print('Intran noise:{}'.format(dat_path_intran))




