'''
Code written in Sep 2022 by Yuhan Wang
study the RF thermal effect
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


fav_tune_files = '/data/smurf_data/tune/1664728549_tune.npy'
bands = [7,6,5,4,3,2,1,0]
slot_num = 2

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()

out_fn = '/data/smurf_data/UFM_testing/Mv27_ph005/RF_thermal_2.csv'

print('plotting directory is:')
print(S.plot_dir)

for target_band in bands:
    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(20)
    S.set_mode_dc()

    S.load_tune(fav_tune_files)

    for band in bands:
        print('setting up band {}'.format(band))

        S.set_att_dc(band,cfg.dev.bands[band]['dc_att'])
        print('band {} dc_att {}'.format(band,S.get_att_dc(band)))

        S.set_att_uc(band,cfg.dev.bands[band]['uc_att'])
        print('band {} uc_att {}'.format(band,S.get_att_uc(band)))

        S.amplitude_scale[band] = cfg.dev.bands[band]['tone_power']
        print('band {} tone power {}'.format(band,S.amplitude_scale[band] ))

        print('setting synthesis scale')
        # hard coding it for the current fw
        S.set_synthesis_scale(band,1)

        print('running relock')
        S.relock(band,tone_power=cfg.dev.bands[band]['tone_power'])
        
        S.run_serial_gradient_descent(band);
        S.run_serial_eta_scan(band);
        
        print('running tracking setup')
        S.set_feedback_enable(band,1) 
        S.tracking_setup(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                         fraction_full_scale=cfg.dev.bands[band]['frac_pp'],
                         make_plot=False, save_plot=False, show_plot=False,
                         channel=S.which_on(band), nsamp=2**18,
                         lms_freq_hz=None, # cfg.dev.bands[band]["lms_freq_hz"],
                         meas_lms_freq=True,#cfg.dev.bands[band]["meas_lms_freq"],
                         feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                         feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                         lms_gain=cfg.dev.bands[band]['lms_gain']
        )
        print('checking tracking')
        S.check_lock(band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
                     fraction_full_scale=cfg.dev.bands[band]['frac_pp'],
                     lms_freq_hz=None,#cfg.dev.bands[band]['lms_freq_hz'],
                     feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
                     feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
                     lms_gain=cfg.dev.bands[band]['lms_gain']
        )

    S.overbias_tes_all(bias_groups=[0,1,2,3,4,5,6,7,8,9,10,11],
                       overbias_wait=1, tes_bias=20, cool_wait=3,
                       high_current_mode=False, overbias_voltage=15
    )
    bias_array = np.array(
        [5.4, 5.4, 9.5, 9.1, 5.2, 5.4, 8.6, 9.0, 3.3, 3.6, 8.6, 8.5])
    bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
    S.set_tes_bias_bipolar_array(bias_array)
    #immediately drop to low current
    S.set_tes_bias_low_current(bias_groups)

    for _ in range(5):
        time.sleep(60)
 
    for new_band in bands:
        print(bias_array)
        step_size = 0.1 
        dat_path = S.stream_data_on()

        for k in [0,1,2,3]: 
            S.set_tes_bias_bipolar_array(bias_array) 
            time.sleep(1) 
            S.set_tes_bias_bipolar_array(bias_array - step_size) 
            time.sleep(1)
            
        S.set_att_uc(new_band, 0)
        time.sleep(5) 

        for k in [0,1,2,3]: 
            S.set_tes_bias_bipolar_array(bias_array) 
            time.sleep(1) 
            S.set_tes_bias_bipolar_array(bias_array - step_size) 
            time.sleep(1) 
            
        S.stream_data_off() 

        S.set_tes_bias_bipolar_array(bias_array) 

        fieldnames = ['time','bias_voltage', 'band', 'off_band', 'off_band_tone',
                      'off_band_att_uc','data_path'] 
        row = {}
        row['time'] = dat_path[-14:-4] 
        row['bias_voltage'] = '_'.join(str(x) for x in bias_array)
        row['band'] = target_band
        row['off_band'] = new_band
        row['off_band_tone'] = S.amplitude_scale[new_band]
        row['off_band_att_uc'] = S.get_att_uc(new_band)
        row['data_path'] = dat_path
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)

        time.sleep(60)

        print(bias_array)
        step_size = 0.1 
        dat_path = S.stream_data_on()

        for k in [0,1,2,3]: 
            S.set_tes_bias_bipolar_array(bias_array) 
            time.sleep(1) 
            S.set_tes_bias_bipolar_array(bias_array - step_size) 
            time.sleep(1)
            
        S.set_att_uc(new_band, 30)
        time.sleep(5) 

        for k in [0,1,2,3]: 
            S.set_tes_bias_bipolar_array(bias_array) 
            time.sleep(1) 
            S.set_tes_bias_bipolar_array(bias_array - step_size) 
            time.sleep(1) 
            
        S.stream_data_off() 

        S.set_tes_bias_bipolar_array(bias_array) 

        fieldnames = ['time','bias_voltage', 'band', 'off_band', 'off_band_tone',
                      'off_band_att_uc','data_path'] 
        row = {}
        row['time'] = dat_path[-14:-4]
        row['bias_voltage'] = '_'.join(str(x) for x in bias_array)
        row['band'] = target_band
        row['off_band'] = new_band
        row['off_band_tone'] = S.amplitude_scale[new_band]
        row['off_band_att_uc'] = S.get_att_uc(new_band)
        row['data_path'] = dat_path
        with open(out_fn, 'a', newline = '') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)


        try:

            timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)
            fs = S.get_sample_frequency()

            bands_look, channels = np.where(mask != -1)
            phase *= S.pA_per_phi0 / (2.0 * np.pi)  # pA
            sample_nums = np.arange(len(phase[0]))
            t_array = sample_nums / fs

            stream_by_band_by_channel = {}
            for band, channel in zip(bands_look, channels):
                if band not in stream_by_band_by_channel.keys():
                    stream_by_band_by_channel[band] = {}
                ch_idx = mask[band, channel]
                stream_by_band_by_channel[band][channel] = phase[ch_idx]

            fmin=5
            fmax=50
            detrend='constant'
            start_time = dat_path[-14:-4] 

            fig = plt.figure(figsize=(11,6))
            ax = fig.add_subplot(1, 1, 1)
            band_plot = 3
            channel = 168
            stream_single_band = stream_by_band_by_channel[band_plot]
            stream_single_channel = stream_single_band[channel]

            stream_single_channel_norm = stream_single_channel - np.mean(stream_single_channel)
            ax.plot(t_array, stream_single_channel_norm, color='C0', alpha=1)

            save_name = f'{start_time}_band{band_plot}_channel{channel}.png'
            print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
            plt.savefig(os.path.join(S.plot_dir, save_name))

        except:
            continue

        time.sleep(60)
