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
parser.add_argument('--bgs', type=int, nargs='+', default=None)
parser.add_argument('--bands', type=int, nargs='+', default=None)
parser.add_argument('--overbias-voltage', type=float, default=19)
parser.add_argument('--bias-high', type=float, default=15)
parser.add_argument('--bias-low', type=float, default=0)
parser.add_argument('--high-current-mode', default=False, action='store_true')
parser.add_argument('--skip-tracking', default=False, action='store_true')
parser.add_argument('--output-file',type=str)

args = parser.parse_args()
if args.bgs is None:
    bias_groups = range(12)
else:
    bias_groups = args.bgs
if args.bands is None:
    bands = range(8)
else:
    bands = args.bands

cfg = DetConfig()
cfg.load_config_files(slot=args.slot)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])
if args.high_current_mode:
    hlr = S.high_low_current_ratio
    label="HCM_IV"
else:
    hlr = 1.0
    label="LCM_IV"

if not args.skip_tracking:
    for band in bands:
        for _ in range(3):
            S.run_serial_gradient_descent(band);
            S.run_serial_eta_scan(band);
        S.set_feedback_enable(band,1) 
        S.tracking_setup(
            band, reset_rate_khz=cfg.dev.bands[band]['flux_ramp_rate_khz'],
            fraction_full_scale=cfg.dev.bands[band]['frac_pp'], make_plot=False,
            save_plot=False, show_plot=False, channel=S.which_on(band),
            nsamp=2**18, lms_freq_hz=None, meas_lms_freq=True,
            feedback_start_frac=cfg.dev.bands[band]['feedback_start_frac'],
            feedback_end_frac=cfg.dev.bands[band]['feedback_end_frac'],
            lms_gain=cfg.dev.bands[band]['lms_gain'],
        )

    S.set_filter_disable(0)
    S.set_rtm_arb_waveform_enable(0)
    S.set_downsample_factor(20)

bias_v = 0
bias_array = np.zeros(S._n_bias_groups)
for bg in bias_groups:
    bias_array[bg] = bias_v
S.set_tes_bias_bipolar_array(bias_array)
time.sleep(10)

#take 30s timestream for noise
sid = sdl.take_g3_data(S, 30)
am = sdl.load_session(cfg.stream_id, sid, base_dir=cfg.sys['g3_dir'])
ctime = int(am.timestamps[0])
noisedict = sdl.noise.get_noise_params(
    am, wl_f_range=(10,30), fit=False, nperseg=1024)
noisedict['sid'] = sid
savename = os.path.join(S.output_dir, f'{ctime}_take_noise.npy')
sdl.validate_and_save(
    savename, noisedict, S=S, cfg=cfg, make_path=False
)

fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type']
if not os.path.isfile(args.output_file):
    with open(args.output_file, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

row = {}
row['data_path'] = savename
row['bias_voltage'] = bias_v
row['type'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = args.temp

with open(args.output_file, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

for bias_gp in bias_groups:
    row = {}
    row['bath_temp'] = bath_temp
    row['bias_line'] = bias_gp
    row['band'] = 'all'
    row['bias_voltage'] = label
    row['type'] = 'IV'
    print(f'Taking IV on bias line {bias_gp}')
      
    row['data_path'] = det_ops.take_iv(
        S,
        cfg,
        bias_groups = [bias_gp],
        wait_time=0.01,
        bias_high=args.bias_high/hlr,
        bias_low=args.bias_low/hlr,
        bias_step = 0.025/hlr,
        overbias_voltage=args.overbias_voltage,
        cool_wait=30,
        high_current_mode=args.high_current_mode,
        make_channel_plots=False,
        save_plots=True,
        do_analysis=True,
    )

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    time.sleep(30)
