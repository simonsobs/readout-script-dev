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

bands = np.arange(8)

print("Running tracking setup")
for b in bands:
    S.tracking_setup(b, reset_rate_khz=4, fraction_full_scale=0.4)

print("Streaming data for 1 minute...")
sid = sdl.take_g3_data(S, 60)
print("Session id: ", sid)
