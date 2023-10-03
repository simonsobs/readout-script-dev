from ocs.ocs_client import OCSClient
import time
import numpy as np
import sys
import signal


# ================ User Parameters ================ 

print("Initialize pysmurf client")
temperature_list = [50,60,70,80,90,100,120,140,150,160,170,180,190,200]
#temperature_list = [50,60,70,80,90,100,120,140,160,180,190,200,210,220]
#temperature_list = [60,80]
pid_thermometer_channel = 15
smurf_script_path = '/readout-script-dev/ddutcher/hcm_bath_iv_noise.py'
#iv_script_path    = '/readout-script-dev/rsonka/iv_takers/hcm_bath_iv_by_BL.py'

slots = [4,5,6]
# filenames v whose slot isn't activated above^ will not be used.
file_names = {2: '/data/smurf_data/UFM_testing/Mv32_ph008/bathramp_noise_20221224.csv',
              3: '/data/smurf_data/UFM_testing/Mv14_pg010/bathramp_noise_20220321_test.csv',
              4: '/data/smurf_data/UFM_testing/Uv41_pi004/bathramp_noise_20230925.csv',
              5: '/data/smurf_data/UFM_testing/Uv42_pi004/bathramp_noise_20230925.csv',
              6: '/data/smurf_data/UFM_testing/Uv44_pi004/bathramp_noise_20230925.csv',
              7: '/data/smurf_data/UFM_testing/Lp3_pi002/bathramp_noise_so_20230808.csv'}
              
# ================ Variable and Function Definitions ================               
              
pysmurfs = [OCSClient(f'pysmurf-controller-s{slot}', 
                      args=['--site-http=http://localhost:8001/call']) for slot in slots]
#pysmurf2 = OCSClient('pysmurf-controller-s2', args=['--site-http=http://localhost:8001/call'])
#pysmurf3 = OCSClient('pysmurf-controller-s3', args=['--site-http=http://localhost:8001/call'])
# pysmurf4 = OCSClient('pysmurf-controller-s4', args=['--site-http=http://localhost:8001/call'])
# pysmurf5 = OCSClient('pysmurf-controller-s5', args=['--site-http=http://localhost:8001/call'])
# pysmurf6 = OCSClient('pysmurf-controller-s6', args=['--site-http=http://localhost:8001/call'])
#pysmurf7 = OCSClient('pysmurf-controller-s7', args=['--site-http=http://localhost:8001/call'])

ls_args = [
    "--site-hub=ws://127.0.0.1:8001/ws",
    "--site-http=http://127.0.0.1:8001/call",
    "--site-host=ocs-docker"
]
client = OCSClient('370154', args=ls_args)

def temp_pid(client, temperature, channel=14):
    # if channel == 14:
    #     pid = {"P":2.5, "I":100, "D":0}
    #     print(f"Changing PID parameters to {pid['P']}, {pid['I']}, {pid['D']}")
    #     client.set_pid.start(**pid)
    #    client.set_pid.wait()
    print("Setting servo setpoint to {} mK".format(temperature))
    pid_success = False
    while pid_success is False:
        client.servo_to_temperature.start(temperature=temperature/1000, channel=channel) #changing to Kelvin
        ok, msg, session = client.servo_to_temperature.wait()
        pid_success = session['success']

    print(f"Adjusting temperature")
    time.sleep(10)

    end_script = False
    while end_script is False:
        params = {'measurements': 20, 'threshold': 0.5e-3, 'attempts': 1, 'pause': 10}
        client.check_temperature_stability.start(**params)
        ok, msg, session = client.check_temperature_stability.wait()

        if session['success']:
            end_script = True
            print("Temperatures adjustment complete.")
            print(msg)
        else:
            print("Temperatures still not stable, waiting.")
            time.sleep(60)

    client.set_autoscan.start(autoscan=True)
    client.set_autoscan.wait()
    print('Starting a data acquisition process...')
    client.acq.start()

# ================ Main Script ================ 

try:
    for temperature in temperature_list:
        if temperature != temperature_list[0]:
            t0 = time.time()
            temp_pid(client, temperature, channel=pid_thermometer_channel)
            now = time.time()
            while now-t0 < 60 * 15:
                time.sleep(30)
                now = time.time()
        ###
        # New section 20220420: Records all Lakeshore  Channels
        ls_temps = []
        for ch in [13,14,15,16]:
            temp_success = False
            while temp_success is False:
                ret = client.get_channel_attribute(attribute='kelvin_reading',channel=ch)
                temp_success = ret.session['success']
            ls_temps.append("%0d" % (round(ret.session['data']['kelvin_reading']*1e3)))
        ls_temps = " ".join(ls_temps)
        ###
        print("Running {}...".format(smurf_script_path))
        
        slot_args = [ ['--slot',slot,
                       '--temp', ls_temps,
                       '--output_file', file_names[slot]] for slot in slots]
         
            
#         # args_2 = ['--slot', 2,
#         #        '--temp', ls_temps,
#         #        '--output_file','/data/smurf_data/UFM_testing/Mv32_ph008/bathramp_noise_20221224.csv'
#         # ]

#         # args_3 = ['--slot', 3,
#         #           '--temp',ls_temps,
#         #           '--output_file','/data/smurf_data/UFM_testing/Mv14_pg010/bathramp_noise_20220321_test.csv'
#         # ]
#         args_4 = ['--slot', 4,
#                   '--temp', ls_temps,
#                   '--output_file','/data/smurf_data/UFM_testing/Uv41_pi004/bathramp_noise_20230921.csv'
#         ]

#         args_5 = ['--slot', 5,
#                   '--temp',ls_temps,
#                   '--output_file','/data/smurf_data/UFM_testing/Uv42_pi004/bathramp_noise_20230921.csv'
#         ]

#         args_6 = ['--slot', 6,
#                  '--temp', ls_temps,
#                  '--output_file','/data/smurf_data/UFM_testing/Uv44_pi004/bathramp_noise_20230921.csv'
#         ]

#         #args_7 = ['--slot', 7,
#         #          '--temp', ls_temps,
#         #          '--output_file','/data/smurf_data/UFM_testing/Lp3_pi002/bathramp_noise_so_20230808.csv'
#         # ]
        
        
        for i in range(len(slots)):
            pysmurfs[i].run.start(script=smurf_script_path, args=slot_args[i])
        
# #        pysmurf2.run.start(script=smurf_script_path , args=args_2)
# #        pysmurf3.run.start(script=smurf_script_path , args=args_3)
#         pysmurf4.run.start(script=smurf_script_path , args=args_4)
#         pysmurf5.run.start(script=smurf_script_path , args=args_5)
#         pysmurf6.run.start(script=smurf_script_path , args=args_6)
#         #pysmurf7.run.start(script=smurf_script_path , args=args_7)

        for slot_mc in pysmurfs:#[pysmurf4,pysmurf5,pysmurf6]:
            ret = slot_mc.run.wait()
            if not ret.session['success']:
                raise OSError(
                    'OCS script failed to run. Check ocs-pysmurf logs.'
                )
        print('nice!')
except OSError:
    print(f'Something broke at MC temp {temperature} mK.')
    raise

print('Finished bath ramp. Setting back to 100 mK')
temp_pid(client, 100, channel=pid_thermometer_channel)
