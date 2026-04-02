import numpy as np
import time
import sodetlib as sdl
import sodetlib.operations.complex_impedance as ci
from sodetlib.operations import bias_dets, take_iv, bias_steps

if __name__ == "__main__":
    import argparse
    from sodetlib.det_config import DetConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--bgs', type=int, nargs='+', default=None)
    parser.add_argument('--biashigh', type=float, default=18)

    cfg = DetConfig()
    args = cfg.parse_args(parser)

    S = cfg.get_smurf_control()

    S.load_tune(cfg.dev.exp['tunefile'])

    if args.bgs is None:
        bgs = range(12)
    else:
        bgs = args.bgs
    bias_high = args.biashigh

    freqs = np.logspace(0, np.log10(2e3), 40)
    ci.take_complex_impedance_ob_sc(S, cfg, bgs, freqs=freqs, run_analysis=True)

    iva = take_iv(
        S, cfg, bias_groups=bgs, wait_time=0.01, bias_high=bias_high,
        overbias_wait=2, bias_low=0, bias_step=0.025, overbias_voltage=19,
        cool_wait=20, high_current_mode=True, run_serially=False,
        run_analysis=True, show_plots=False,
    )

    for i,rfrac in enumerate([0.6, 0.5, 0.4]):
        ob = True if i==0 else False
        bias_dets.bias_to_rfrac(S, cfg, rfrac, overbias=ob)
        time.sleep(60)
        noise = sdl.noise.take_noise(S,cfg, show_plot=False)
        bsa = bias_steps.take_bias_steps(S, cfg)   
        ds = ci.take_complex_impedance(S, cfg, bgs, freqs=freqs, run_analysis=True)
