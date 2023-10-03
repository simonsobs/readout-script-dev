'''
takes SC noise and takes IV
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
from sodetlib.smurf_funcs import det_ops
import sodetlib as sdl
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--slot',type=int)
parser.add_argument('--temp',type=str)
parser.add_argument('--bgs', action='append', default=None) #type=int, nargs='+', 
parser.add_argument('--output_file',type=str)
parser.add_argument('--UHF_wait',type=int,default=False) # Do if the very first IV at a given 

args = parser.parse_args()
if args.bgs is None:
    bias_groups = [0,1,2,3,8,9,10,11,4,5,6,7]#range(12)
else:
    if ' ' in args.bgs[0]:
        input_bgs = (args.bgs[0]).split(" ")
    else:
        input_bgs = args.bgs
    bias_groups = [int(bg) for bg in input_bgs]
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()

S.load_tune(cfg.dev.exp['tunefile'])

# S.set_filter_disable(0)
# S.set_rtm_arb_waveform_enable(0)
# S.set_downsample_factor(20)

# for band in [0,1,2,3,4,5,6,7]:
#     try:
# #        S.relock(band, tone_power=cfg.dev.bands[band]['tone_power'])
#         for _ in range(3):
#             S.run_serial_gradient_descent(band)
#             S.run_serial_eta_scan(band)
#         # S.set_feedback_enable(band,1)
#         # S.tracking_setup(
#         #     band,reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
#         #     fraction_full_scale=cfg.dev.bands[band]['frac_pp'],
#         #     make_plot=False, save_plot=False, show_plot=False,
#         #     channel=S.which_on(band)[::10], nsamp=2**18,
#         #     lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
#         #     meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
#         #     feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
#         #     feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
#         #     feedback_gain=cfg.dev.bands[band]["feedback_gain"],
#         #     lms_gain=cfg.dev.bands[band]['lms_gain']
#         # )
#     except:
#         continue



fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type']


for ind, bias_gp in enumerate(bias_groups):
    row = {}
    row['bath_temp'] = bath_temp
    row['bias_line'] = bias_gp
    row['band'] = 'all'
    row['bias_voltage'] = 'HCM_IV'
    row['type'] = 'IV'
    print(f'Taking IV on bias line {bias_gp}')

    if int(args.UHF_wait): # FOR UHF ONLY!
        cool_wait = 300
    else:
        cool_wait = 30
      
    row['data_path'] = det_ops.take_iv(
        S,
        cfg,
        bias_groups=[bias_gp],
        wait_time=0.01,
        bias_high=35 / S.high_low_current_ratio,
        bias_low=0,
        bias_step=0.025 / S.high_low_current_ratio,
        overbias_voltage=19,
        cool_wait=cool_wait,
        high_current_mode=True,
        make_channel_plots=False,
        save_plots=True,
        do_analysis=True,
    )

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    #time.sleep(30) # Daniel only had this for in-between ALL the biaslines happening. 
