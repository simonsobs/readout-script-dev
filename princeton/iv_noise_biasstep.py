'''
takes SC and ob noise, IV, and in-transition bias steps and noise.
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
import sodetlib.operations as ops
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
parser.add_argument('--output_file',type=str)

args = parser.parse_args()
if args.bgs is None:
    bias_groups = range(12)
else:
    bias_groups = args.bgs
slot_num = args.slot
bath_temp = args.temp
out_fn = args.output_file

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])

for band in [0,1,2,3,4,5,6,7]:
    S.run_serial_gradient_descent(band)
    S.run_serial_eta_scan(band)
    S.set_feedback_enable(band, 1)
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
for bias_index, bias_g in enumerate(bias_groups):
    S.set_tes_bias_low_current(bias_g)

bias_v = 0
S.set_tes_bias_bipolar_array([bias_v] * S._n_bias_groups)
time.sleep(10)

fieldnames = ['bath_temp','bias_voltage', 'bias_line', 'band', 'data_path','type']
if not os.path.isfile(out_fn):
    with open(out_fn, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

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

row = {}
row['data_path'] = savename
row['bias_voltage'] = 'sc'
row['type'] = 'noise'
row['bias_line'] = 'all'
row['band'] = 'all'
row['bath_temp'] = bath_temp
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

row = {}
row['bath_temp'] = bath_temp
row['bias_line'] = 'all'
row['band'] = 'all'
row['bias_voltage'] = 'IV 18 to 0'
row['type'] = 'IV'
print(f'Taking IV serially on all bias lines, all bands')

iva = ops.take_iv(
    S, cfg, bias_groups=bias_groups, wait_time=0.01, bias_high=18,
    overbias_wait=2, bias_low=0, bias_step=0.025, overbias_voltage=19,
    cool_wait=20, high_current_mode=False, run_serially=True,
    serial_wait_time=30, run_analysis=True, show_plots=False,
)
dat_file = iva.filepath
row['data_path'] = dat_file
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

S.overbias_tes_all(bias_groups=bias_groups, overbias_voltage=19, tes_bias=18,
                   overbias_wait=2, cool_wait=3, high_current_mode=False)

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

row['data_path'] = savename
row['bias_voltage'] = 'ob'
row['type'] = 'noise'
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

bsa = ops.take_bias_steps(S, cfg, bgs=bias_groups)
row['data_path'] = bsa.filepath
row['type'] = 'bias_step'
with open(out_fn, 'a', newline = '') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow(row)

for rfrac in [0.7, 0.5, 0.3]:
    biases = ops.bias_to_rfrac(S, cfg, rfrac=rfrac, bias_groups=bias_groups, math_only=True)
    S.set_tes_bias_bipolar_array(biases)
    time.sleep(30)

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

    row = {}
    row['data_path'] = savename
    row['bias_voltage'] = f'{rfrac}Rn'
    row['type'] = 'noise'
    row['bias_line'] = 'all'
    row['band'] = 'all'
    row['bath_temp'] = bath_temp
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    bsa = ops.take_bias_steps(
        S, cfg, bgs=bias_groups, analysis_kwargs={'transition':True, 'fit_tmin':7.5e-4})
    row['type'] = 'bias_step'
    row['data_path'] = bsa.filepath
    
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

bias_v = 0
S.set_tes_bias_bipolar_array([bias_v] * S._n_bias_groups)
