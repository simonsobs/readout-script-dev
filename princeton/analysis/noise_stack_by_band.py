# noise_stack_by_band.py
import argparse
import matplotlib
import numpy as np
matplotlib.use("Agg")
from sodetlib import noise, det_config
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    cfg = det_config.DetConfig()
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--acq-time",
    type=float,
    default=30.0,
    help="float, optional, default is 30.0. The amount of time to sleep in seconds while "
    + "streaming SMuRF data for analysis.",
)

    args = cfg.parse_args(parser)
    S = cfg.get_smurf_control(dump_configs=False, make_logfile=True)
    S.load_tune(cfg.dev.exp['tunefile'])

    nsamps = S.get_sample_frequency() * args.acq_time
    nperseg = 2 ** round(np.log2(nsamps / 5))
    noise.take_noise(
        S, cfg, acq_time=args.acq_time, show_plot=False, save_plot=True, nperseg=nperseg,
    )
