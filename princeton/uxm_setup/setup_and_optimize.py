# setup_and_optimize.py

"""
Power amplifiers, optimize tone power and uc_att for each band,
setup with those optimized parameters, and take noise.
"""

import os, sys
import numpy as np
import argparse
from sodetlib.det_config import DetConfig
from sodetlib.operations import uxm_setup as op
from sodetlib import noise
import logging

sys.path.append("readout-script-dev/rsonka/uxm_setup_optimize")
sys.path.append('/readout-script-dev/ddutcher')
from uxm_setup import uxm_setup_main_ops
from uxm_optimize_quick import uxm_optimize
import uxm_setup_optimize_confluence as confl

logger = logging.getLogger()

cfg = DetConfig()

parser = argparse.ArgumentParser(
    description="Parser for setup_and_optimize.py script."
)

parser.add_argument(
    'assem_type',
    type=str,
    choices=['ufm','umm'],
    default='ufm',
    help='Assembly type, ufm or umm. Determines the relevant noise  thresholds.',
    )

parser.add_argument(
    "--bands",
    type=int,
    default=None,
    nargs="+",
    help="The SMuRF bands to target. Will default to the bands "
    + "listed in the pysmurf configuration file."
)

# optional arguments
parser.add_argument(
    "--skip-optimize-fracpp",
    default=False,
    action="store_true",
    help="Include this flag to skip the frac_pp optimization skip",
)
parser.add_argument(
    "--acq-time",
    type=float,
    default=30.0,
    help="float, optional, default is 30.0. The amount of time to sleep in seconds while "
    + "streaming SMuRF data for analysis.",
)
parser.add_argument(
    "--loglevel",
    type=str.upper,
    default=None,
    choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
    help="Set the log level for printed messages. The default is pulled from "
    +"$LOGLEVEL, defaulting to INFO if not set.",
)

parser.add_argument(
    "--bias-TESs",
    type=float,
    default=0,
    help="Bias TESs to this voltage during the setup and optimization. Default 0 (no bias)." 
    + "You might set it high (ex 19V) to remove TES phonon noise from noise circuit. "
    + "If you do, run as umm instead of ufm, as will be closer to umm parameters.",
)

parser.add_argument(
    "--confluence-fp",
    type=str,
    default='default',
    help="filepath to the confluence formatting summary to add to; makes new if not given. ",
)

# parse the args for this script
args = cfg.parse_args(parser)
if args.loglevel is None:
    args.loglevel = os.environ.get("LOGLEVEL", "INFO")
numeric_level = getattr(logging, args.loglevel)
logging.basicConfig(
    format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
)

S = cfg.get_smurf_control(dump_configs=True, make_logfile=(numeric_level != 10))

if args.assem_type == 'ufm':
    high_noise_thresh = 250
    med_noise_thresh = 150
    low_noise_thresh = 135
elif args.assem_type == 'umm':
    high_noise_thresh = 250
    med_noise_thresh = 65
    low_noise_thresh = 45
else:
    raise ValueError("Assembly must be either 'ufm' or 'umm'.")

# power amplifiers
success = op.setup_amps(S, cfg)
if not success:
    raise OSError("Amps could not be powered.")

# Setup the confluence format file, or find an old one
confluence_fp = confl.start_confluence_log_file(S,cfg,args.bands)
args.confluence_fp = confluence_fp

# run the defs in this file
uxm_optimize(
    S=S,
    cfg=cfg,
    bands=args.bands,
    opt_fracpp=(not args.skip_optimize_fracpp),
    low_noise_thresh=low_noise_thresh,
    med_noise_thresh=med_noise_thresh,
    high_noise_thresh=high_noise_thresh,
    bias_TESs=args.bias_TESs,
    confluence_fp=confluence_fp
)

# Setup, take noise, make plots
uxm_setup_main_ops(S, cfg, args)
# ^ The above should turn the biases back off if you put a >0 value in.
# If this script doesn't complete you may need to do so manually. 




