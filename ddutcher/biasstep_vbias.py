'''
UFM testing script in Pton
loop around different bias voltage and collect biasstep measurement 
'''

import argparse
import numpy as np
import os
import time
import sodetlib as sdl
from sodetlib import noise
from sodetlib.det_config  import DetConfig
from sodetlib.operations.bias_steps import take_bias_steps
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--slot', type=int)
parser.add_argument('--bgs', type=int, nargs='+', default=None)
parser.add_argument('--bias-high', type=float)
parser.add_argument('--bias-low',type=float, default=0)
parser.add_argument('--step-size',type=float, default=0.5)
parser.add_argument('--current-mode', type=str, default='low')
parser.add_argument('--temp', type=str)
parser.add_argument('--output-file', type=str)

args = parser.parse_args()

cfg = DetConfig()
cfg.load_config_files(slot=args.slot)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])

if args.bgs is None:
    bias_groups = range(12)
else:
    bias_groups = args.bgs
if args.current_mode.lower() in ['high','hi']:
    hcm = True
    hlr = S.high_low_current_ratio
else:
    hcm = False
    hlr = 1.

fieldnames = ['bath_temp', 'bias_v', 'band', 'data_path','step_size']
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

S.overbias_tes_all(
    bias_groups=bias_groups,
    overbias_wait=2,
    tes_bias=args.bias_high/hlr,
    cool_wait=3,
    high_current_mode=hcm,
    overbias_voltage=19,
)
time.sleep(120)

step_array = np.arange(
    args.bias_high, args.bias_low - args.step_size, -args.step_size) / hlr

for bias_voltage_step in step_array:
    bias_array = np.zeros(S._n_bias_groups)
    bias_voltage = np.round(bias_voltage_step, 3)
    for bg in bias_groups:
        bias_array[bg] = bias_voltage
    S.set_tes_bias_bipolar_array(bias_array) 
    time.sleep(30)

    bsa = take_bias_steps(S, cfg, bgs=bias_groups, analysis_kwargs={'fit_tmin':7.5e-4, 'transition':True})

    row = {}
    row['bath_temp'] = args.temp
    row['bias_v'] = bias_voltage_step
    row['band'] = 'all'
    row['data_path'] = bsa.filepath
    row['step_size'] = args.step_size

    with open(args.output_file, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    #take 20s timestream for noise
    for bg in bias_groups:
        bias_array[bg] = bias_voltage_step
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(20)

    sid = sdl.take_g3_data(S, 20)
    am = sdl.load_session(cfg.stream_id, sid, base_dir=cfg.sys['g3_dir'])
    ctime = int(am.timestamps[0])
    noisedict = noise.get_noise_params(
        am, wl_f_range=(10,30), fit=False, nperseg=1024)
    noisedict['sid'] = sid
    savename = os.path.join(S.output_dir, f'{ctime}_take_noise.npy')
    sdl.validate_and_save(
        savename, noisedict, S=S, cfg=cfg, make_path=False
    )

    row['data_path'] = savename
    row['step_size'] = 0

    with open(args.output_file, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)
