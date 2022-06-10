# uxm_setup.py

import os
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sodetlib.operations import uxm_setup as op
from sodetlib import noise
import logging

logger = logging.getLogger(__name__)

def uxm_setup(S, cfg, bands=None, estimate_phase_delay=False):
    """
    Use values in cfg to setup UXM for use.
    """
    logger.info(f"plotting directory is:\n{S.plot_dir}")
    if bands is None:
        bands = S.config.get("init").get("bands")

    S.all_off()
    S.set_rtm_arb_waveform_enable(0)
    S.set_filter_disable(0)
    S.set_downsample_factor(cfg.dev.exp.get("downsample_factor", 20))
    S.set_mode_dc()

    for band in bands:
        logger.info(f"setting up band {band}")

        S.set_att_dc(band, cfg.dev.bands[band]["dc_att"])
        logger.info("band {} dc_att {}".format(band, S.get_att_dc(band)))

        S.set_att_uc(band, cfg.dev.bands[band]["uc_att"])
        logger.info("band {} uc_att {}".format(band, S.get_att_uc(band)))

        S.amplitude_scale[band] = cfg.dev.bands[band]["tone_power"]
        logger.info(
            "band {} tone power {}".format(band, S.amplitude_scale[band])
        )

        if estimate_phase_delay:
            logger.info("estimating phase delay")
            try:
                S.estimate_phase_delay(band)
            except Exception:
                logger.warning('Estimate phase delay failed due to PV timeout.')
        logger.info("setting synthesis scale")
        S.set_synthesis_scale(band, cfg.dev.exp.get("synthesis_scale", 1))
        logger.info("running find freq")
        if band in [0,4]:
            start_freq = -230
        else:
            start_freq = -250
        S.find_freq(band, start_freq=start_freq, tone_power=cfg.dev.bands[band]["tone_power"], make_plot=True)
        logger.info("running setup notches")
        S.setup_notches(
            band, tone_power=cfg.dev.bands[band]["tone_power"], new_master_assignment=True
        )
        logger.info("running serial gradient descent and eta scan")
        S.run_serial_gradient_descent(band)
        S.run_serial_eta_scan(band)
        logger.info("running tracking setup")
        S.set_feedback_enable(band, 1)
        S.tracking_setup(
            band,
            reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
            fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
            make_plot=True,
            save_plot=True,
            show_plot=False,
            channel=S.which_on(band)[::10],
            nsamp=2 ** 18,
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            meas_lms_freq=cfg.dev.bands[band]["meas_lms_freq"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            feedback_gain=cfg.dev.bands[band]["feedback_gain"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )
        S.check_lock(
            band,
            reset_rate_khz=cfg.dev.bands[band]["flux_ramp_rate_khz"],
            fraction_full_scale=cfg.dev.bands[band]["frac_pp"],
            lms_freq_hz=cfg.dev.bands[band]["lms_freq_hz"],
            feedback_start_frac=cfg.dev.bands[band]["feedback_start_frac"],
            feedback_end_frac=cfg.dev.bands[band]["feedback_end_frac"],
            lms_gain=cfg.dev.bands[band]["lms_gain"],
        )

    S.save_tune()
    cfg.dev.update_experiment({'tunefile': S.tune_file}, update_file=True)


if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig
    
    cfg = DetConfig()

    parser = argparse.ArgumentParser(
        description="Parser for uxm_setup.py script."
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=None,
        nargs="+",
        help="The SMuRF bands to target. Will default to the bands "
        + "listed in the pysmurf configuration file."
    )

    parser.add_argument(
        "--estimate-phase-delay",
        type=bool,
        action='store_true',
        default=False,
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

    args = cfg.parse_args(parser)
    if args.loglevel is None:
        args.loglevel = os.environ.get("LOGLEVEL","INFO")
    numeric_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
    )
    
    S = cfg.get_smurf_control(dump_configs=True, make_logfile=(numeric_level != 10))

    # # power amplifiers
    success = op.setup_amps(S, cfg)
    if not success:
        raise OSError("Amps could not be powered.")
    # run the defs in this file
    uxm_setup(S=S, cfg=cfg, bands=args.bands, estimate_phase_delay=args.estimate_phase_delay)
    # take noise and plot histograms
    nsamps = S.get_sample_frequency() * args.acq_time
    nperseg = 2 ** round(np.log2(nsamps/5))
    noise.take_noise(
        S, cfg, acq_time=args.acq_time, show_plot=False, save_plot=True,
        nperseg=nperseg,
    )
