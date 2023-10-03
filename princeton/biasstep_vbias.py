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
parser.add_argument('--overbias-voltage', type=float, default=19)
parser.add_argument('--bias-high', type=float, default=15)
parser.add_argument('--bias-low',type=float, default=0)
parser.add_argument('--step', type=float, default=0.5)
parser.add_argument('--high-current-mode', default=False, action='store_true')
parser.add_argument('--temp', type=str)
parser.add_argument('--output_file', type=str)

args = parser.parse_args()
if args.bg is None:
    bias_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
else:
    bias_groups = args.bg

cfg = DetConfig()
cfg.load_config_files(slot=args.slot)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])
if args.high_current_mode:
    hlr = S.high_low_current_ratio
else:
    hlr = 1.0
    

fieldnames = ['bath_temp', 'bias_v', 'band', 'data_path','step_size']
if not os.path.exists(args.output_file):
    with open(args.output_file, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

S.overbias_tes_all(
    bias_groups=bias_groups,
    high_current_mode=args.high_current_mode,
    overbias_voltage=args.overbias_voltage,
    overbias_wait=2,
    tes_bias= args.bias_high/hlr,
    cool_wait=3,
)

step_array = np.arange(
    args.bias_high/hlr, (args.bias_low - args.step/hlr), -args.step/hlr
)
for bias_voltage_step in step_array:
    bias_array = np.zeros(S._n_bias_groups)
    bias_voltage = np.round(bias_voltage_step, 3)
    for bg in bias_groups:
        bias_array[bg] = bias_voltage
    S.set_tes_bias_bipolar_array(bias_array) 
    time.sleep(30)

    bsa = take_bias_steps(S, cfg, analysis_kwargs={'transition':True, 'fit_tmin':7.5e-4})

    row = {}
    row['bath_temp'] = args.temp
    row['bias_v'] = bias_voltage_step
    row['band'] = 'all'
    row['data_path'] = bsa.filepath
    row['step_size'] = .05

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    #take 20s timestream for noise
    for bg in bias_groups:
        bias_array[bg] = bias_voltage_step
    S.set_tes_bias_bipolar_array(bias_array)
    time.sleep(10)

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

    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)
