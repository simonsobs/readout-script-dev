"""
Script for running a series of bias steps with different voltages.

Gets the voltages of the detectors, increments by twice the input bias
increment, and runs a bias step.  It then runs another bias step with the
bias incremented only once by the bias increment.  It then returns to the
original bias and runs a final bias step.

python3 pysmurf_det_setup_script.py --rfrac 0.5 --bias-increment 0.1
"""

import argparse

from sodetlib.det_config import DetConfig
from sodetlib.operations import bias_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rfrac", type=float, default=None)
    parser.add_argument("--bias-increment", type=float, default=0.1)
    args = parser.parse_args()

    rfrac = args.rfrac
    bias_increment = args.bias_increment

    # get config and controller
    cfg = DetConfig()
    _ = cfg.parse_args()
    S = cfg.get_smurf_control(dump_configs=True)

    # get bias voltage for rfrac
    biases = bias_steps.bias_to_rfrac(S, cfg, rfrac=rfrac, math_only=True)

    # increment biases by twice the bias increment
    incremented_biases = biases + 2 * bias_increment
    bias_steps.bias_to_volt_arr(S, cfg, incremented_biases)

    # take the first bias step
    bias_steps.take_bias_steps(S, cfg)

    # increment biases by the bias increment
    incremented_biases = biases + bias_increment
    bias_steps.bias_to_volt_arr(S, cfg, incremented_biases)

    # take second bias step
    bias_steps.take_bias_steps(S, cfg)

    # return to the first bias
    bias_steps.bias_to_volt_arr(S, cfg, biases)
    bias_steps.take_bias_steps(S, cfg)
