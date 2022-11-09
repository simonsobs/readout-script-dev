#ocs_test.py
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
args = parser.parse_args()
slot_num = args.slot

cfg = DetConfig()
cfg.load_config_files(slot=slot_num)
S = cfg.get_smurf_control()
S.load_tune(cfg.dev.exp['tunefile'])

print('Success!')
