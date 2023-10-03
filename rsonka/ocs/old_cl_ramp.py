from ocs.ocs_client import OCSClient
import time
import numpy as np
import yaml
import sys
import signal
import argparse


temperature_list = [9,10,11,12,13,14]

#smurf_script_path = '/readout-script-dev/ddutcher/uxm_bath_iv_noise.py'
#smurf_script_path = '/readout-script-dev/ddutcher/iv_noise_biasstep.py'
#smurf_script_path ='/readout-script-dev/rsonka/lf_bath_iv_noise_low_overbias.py'
smurf_script_path = '/readout-script-dev/rsonka/lf_bath_iv_noise_low_overbias.py'
# SPB_smurf_script_path = '/sodetlib/scratch/ddutcher/spb_bath_iv_noise.py'
#smurf_script_path = '/readout-script-dev/ddutcher/hcm_bath_iv_noise.py'
# swap hcm for uxm to run in high_current_mode
#smurf_script_path = '/readout-script-dev/rsonka/uxm_bath_iv_noise_pos_to_neg.py'

print("Initialize pysmurf client")
#pysmurf2 = OCSClient('pysmurf-controller-s2', args=['--site-http=http://localhost:8001/call'])
#pysmurf3 = OCSClient('pysmurf-controller-s3', args=['--site-http=http://localhost:8001/call'])
#pysmurf4 = OCSClient('pysmurf-controller-s4', args=['--site-http=http://localhost:8001/call'])
#pysmurf5 = OCSClient('pysmurf-controller-s5', args=['--site-http=http://localhost:8001/call'])
#pysmurf6 = OCSClient('pysmurf-controller-s6', args=['--site-http=http://localhost:8001/call'])
pysmurf7 = OCSClient('pysmurf-controller-s7', args=['--site-http=http://localhost:8001/call'])

cl_args = [
    "--site-hub=ws://127.0.0.1:8001/ws",
    "--site-http=http://127.0.0.1:8001/call",
    "--site-host=ocs-docker"
]
ls336 = OCSClient('LSA2AX0', args=cl_args)

def temp_pid(ls336, temperature):
    pid = {"P":30, "I":10, "D":5}
    print(f"Changing PID parameters to {pid['P']}, {pid['I']}, {pid['D']}")
    ls336.set_pid.start(**pid)
    ls336.set_pid.wait()
    print('Setting heater range to low')
    ls336.set_heater_range(range = 'low')
    print("Setting servo setpoint to {}K".format(temperature))
    params = {'temperature': temperature, 'ramp': 2, 'transport': True, 'transport_offset': 1}
    ls336.servo_to_temperature(**params)

    end_script = False
    while end_script is False:
    #### notes from YW Sep 15 2022: I changed the threshold to 0.2K,
    ### I thnk 0.1K takes too long and 0.2K should be fine to serve our purpose
        params = {'threshold': 0.1, 'window': 900}
        ok, msg, session = ls336.check_temperature_stability(**params)

        if session['success'] == True:
            end_script = True
            print("Temperatures adjustment complete.")
            print(msg)
        else:
            print("Temperatures still not stable, waiting 5 min.")
            time.sleep(300)

try:
    for ind, temperature in enumerate(temperature_list):
        if ind != 0:
            t0 = time.time()
            temp_pid(ls336, temperature)
            now = time.time()
            while now-t0 < 30 * 60:
                time.sleep(30)
                now = time.time()

        print("Running {}...".format(smurf_script_path))

        # args_2 = ['--slot', 2,
        #        '--temp',temperature,
        #        '--output_file','/data/smurf_data/UFM_testing/Mv32_ph008/coldload_ramp_20221227.csv'
        #        ]

        # args_3 = ['--slot', 3,
        #         '--temp',temperature,
        #         '--output_file','/data/smurf_data/UFM_testing/Mv14_pg010/coldload_ramp_20220314.csv'
        #         ]

        # args_4 = ['--slot', 4,
        #         '--temp',temperature,
        #         '--output_file','/data/smurf_data/UFM_testing/Mv9_pg010/coldload_ramp_20220314.csv'
        #         ]
        # args_5 = ['--slot', 5,
        #          '--temp',temperature,
        #          '--output_file','/data/smurf_data/UFM_testing/Uv36_ph014/coldload_ramp_20230518.csv'
        #          ]
        #args_6 = ['--slot', 6,
        #       '--temp',temperature,
        #       '--output_file','/data/smurf_data/UFM_testing/Lp2r1_pi001/coldload_ramp_20230622.csv'
        #       ]
        args_7 = ['--slot', 7,
                  '--temp',temperature,
                  '--output_file','/data/smurf_data/UFM_testing/Lp3_pi002/coldload_ramp_split_overbias_20230802.csv'
                  ]

#        pysmurf2.run.start(script=smurf_script_path , args=args_2)
#        pysmurf3.run.start(script=smurf_script_path , args=args_3)
#        pysmurf4.run.start(script=smurf_script_path , args=args_4)
#        pysmurf5.run.start(script=smurf_script_path , args=args_5)
#        pysmurf6.run.start(script=smurf_script_path , args=args_6)
        pysmurf7.run.start(script=smurf_script_path , args=args_7)

        for slot_mc in [pysmurf7]:
            ret = slot_mc.run.wait()
            if not ret.session['success']:
                raise OSError('OCS script failed to run. Check ocs-pysmurf logs.')

        print('nice!')
except OSError:
    print(f'Something broke at coldload temp {temperature} mK.')
    raise

print("Finished coldload ramp. Setting to 9 K")
temp_pid(ls336, 9)
