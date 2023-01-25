import matplotlib
matplotlib.use('Agg')
from sodetlib.det_config import DetConfig
import sodetlib as sdl
import os

import sodetlib as sdl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange, tqdm
from scipy import signal
from pysmurf.client.util.pub import set_action

cfg = DetConfig()
cfg.load_config_files(slot=os.environ['SLOT'])
S = cfg.get_smurf_control(make_logfile=False)

S.set_tes_bias_bipolar_array(np.zeros(12))
