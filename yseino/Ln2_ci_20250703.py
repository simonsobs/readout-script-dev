import os, sys
import numpy as np
import argparse
from sodetlib.det_config import DetConfig
import sodetlib.operations as op
from sodetlib import noise
import logging
import time

sys.path.append('/readout-script-dev/yseino')
import sodetlib.operations.complex_impedance as ci

logger = logging.getLogger()

cfg = DetConfig()

cfg.load_config_files(slot=5)
S = cfg.get_smurf_control(dump_configs=True)
S.load_tune(cfg.dev.exp['tunefile'])


#bias for BL8,9,10,11
v_bias = [2.925,2.6,1.525,1.525] #MC 100mK, use same bias as BL10 for BL11

freqs = np.logspace(0, np.log10(2e3), 80)
bgs = [8,9,10,11]


#Take over bias and super conducting data
ci.take_complex_impedance_ob_sc(S, cfg, bgs, freqs=freqs, run_analysis=True)

for label in ['Rfrac_low','Rfrac_50%','Rfrac_high']:

    if label == 'Rfrac_low':
        v_bias[0] = v_bias[0] - 1
        v_bias[1] = v_bias[1] - 1
        v_bias[2] = v_bias[2] - 0.5 
        v_bias[3] = v_bias[3] - 0.5 
    elif label== 'Rfrac_high':
        v_bias[0] = v_bias[0] + 1
        v_bias[1] = v_bias[1] + 1
        v_bias[2] = v_bias[2] + 0.5 
        v_bias[3] = v_bias[3] + 0.5 


    v_bias_all = [0,0,0,0,0,0,0,0] + v_bias + [0,0,0]
    op.bias_dets.bias_to_volt_arr(S, cfg, biases=v_bias_all, bias_groups=bgs)

    print("***comfirmation of biasing ***")
    print(S.get_tes_bias_bipolar_array())#For confirmation
    print('***')

    time.sleep(60)
    ci.take_complex_impedance(S, cfg, bgs, freqs=freqs, run_analysis=True)
