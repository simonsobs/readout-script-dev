from ocs.ocs_client import OCSClient
import time
import numpy as np
import sys
import signal


# ================ User Parameters ================ 

print("Initialize pysmurf client")
temperature_list = [11,12,13,14]
#temperature_list = [9.5,10,11,12,13,14]
noise_script_path = '/readout-script-dev/rsonka/iv_takers/hcm_bath_noise_by_BL.py'
iv_script_path    = '/readout-script-dev/rsonka/iv_takers/hcm_bath_iv_by_BL.py'

slots = [4,5,6]
# filenames v whose slot isn't activated above^ will not be used.
file_names = {2: '/data/smurf_data/UFM_testing/Mv32_ph008/coldload_ramp_20221224.csv',
              3: '/data/smurf_data/UFM_testing/Mv14_pg010/coldload_ramp_20220321_test.csv',
              4: '/data/smurf_data/UFM_testing/Uv41_pi004/coldload_ramp_20230927.csv',
              5: '/data/smurf_data/UFM_testing/Uv42_pi004/coldload_ramp_20230927.csv',
              6: '/data/smurf_data/UFM_testing/Uv44_pi004/coldload_ramp_20230927.csv',
              7: '/data/smurf_data/UFM_testing/Lp3_pi002/coldload_ramp_so_20230808.csv'}
bias_lines = [0,1,2,3,4,5,6,7,8,9,10,11] # which to run on
bias_lines_solo = [4,7,9,11] # Chosen based on yield results
do_solo_UFMs = True # If true, after simultaneous data taking, will take data on one UFM at a time as well. 


# ================ Variable and Function Definitions ================ 

file_names_solo = {slot: file_names[slot][:-4]+"_solo.csv" for slot in slots}              
pysmurfs = [OCSClient(f'pysmurf-controller-s{slot}', 
                      args=['--site-http=http://localhost:8001/call']) for slot in slots]
cl_args = [
    "--site-hub=ws://127.0.0.1:8001/ws",
    "--site-http=http://127.0.0.1:8001/call",
    "--site-host=ocs-docker"
]
ls336 = OCSClient('LSA2AX0', args=cl_args)

ls_args = [
    "--site-hub=ws://127.0.0.1:8001/ws",
    "--site-http=http://127.0.0.1:8001/call",
    "--site-host=ocs-docker"
]
client = OCSClient('370154', args=ls_args)

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

            
def get_temps(client,cl_temp):
    # New section 20220420: Records all Lakeshore  Channels
    ls_temps = []
    for ch in [13,14,15,16]:
        temp_success = False
        while temp_success is False:
            ret = client.get_channel_attribute(attribute='kelvin_reading',channel=ch)
            temp_success = ret.session['success']
        ls_temps.append("%0d" % (round(ret.session['data']['kelvin_reading']*1e3)))
    ls_temps = str(cl_temp)+" "+ " ".join(ls_temps)
    return ls_temps

# def get_cl_temps(client):
#     # New section 20220420: Records all Lakeshore  Channels
#     ls_temps = []
#     for ch in [2,3,4]:
#         temp_success = False
#         #while temp_success is False:
#         ret = client.get_channel_attribute(attribute='kelvin_reading',channel=ch)
#         print(ret.session.keys())
#         for key,val in ret.session.items():
#             print(key)
#             print(val)  # this produces nothing, no keys. 
#         #temp_success = ret.session['success'] # that key doesn't exist in cl?
#         ls_temps.append("%0d" % (round(ret.session['data']['kelvin_reading']*1e3)))
#     ls_temps = " ".join(ls_temps)
#     return ls_temps

def check_success(pysmurfs):
    for slot_mc in pysmurfs:#[pysmurf4,pysmurf5,pysmurf6]:
        ret = slot_mc.run.wait()
        if not ret.session['success']:
            raise OSError(
                'OCS script failed to run. Check ocs-pysmurf logs.')

def take_noise(noise_script_path,bias_lines, file_names, ls_temps, slots, pysmurfs):
    print(f"taking simultaneous noise at {ls_temps} via {noise_script_path}")
    slot_args = [ ['--slot',slot,
                   '--temp', ls_temps,
                   '--bgs', " ".join([str(bl) for bl in bias_lines]),
                   '--output_file', file_names[slot],
                   '--UHF_wait',0] for slot in slots] # It wasn't working passing bools
    for i in range(len(slots)):
        pysmurfs[i].run.start(script=noise_script_path, args=slot_args[i])
    check_success(pysmurfs)
    return slot_args
        
def take_ivs(iv_script_path, bias_lines, file_names, slots, pysmurfs, slot_args, temperature,
             UHF_wait=False):
    for j in range(len(bias_lines)):
        bl=bias_lines[j]            
        ls_temps = get_temps(client,temperature) 
        print(f"taking slots {slots} bias line {bl} @ {ls_temps} via {iv_script_path}")
        for i in range(len(slot_args)):
            slot_args[i][3] = ls_temps
            slot_args[i][5] = str(bl)
            slot_args[i][9] = 0
            if j==0 and UHF_wait:
                slot_args[i][9] = 1
        for i in range(len(slots)):
            pysmurfs[i].run.start(script=iv_script_path, args=slot_args[i])
        check_success(pysmurfs)

        
# ================ Main Script ================ 

try:    
    for temperature in temperature_list:
        if temperature != temperature_list[0]: 
            t0 = time.time()
            temp_pid(ls336, temperature)
            now = time.time()
            while now-t0 < 60 * 15:
                time.sleep(30)
                now = time.time()
        
        # -------- PID for this temperature set IS COMPLETE.
        ls_temps = get_temps(client,temperature) 
        print(f"Begun temp {temperature} at {ls_temps}")
        
        
                
        ### ------- Simultaneous noise
        slot_args = take_noise(noise_script_path,bias_lines, 
                               file_names, ls_temps, slots, pysmurfs)
        
        ### ------- Truly only one UFM's single bias line at a time.
        # Do them first because they heat things less than simultaneous ones.
        if do_solo_UFMs:
            for i in range(len(slots)):
                slots_solo = [slots[i]]
                pysmurfs_solo = [pysmurfs[i]]
                slot_args_solo = [ ['--slot',slots[i],
                               '--temp', "will be overwritten",
                               '--bgs', " ".join([str(bl) for bl in bias_lines_solo]),
                               '--output_file', file_names_solo[slots[i]],
                               '--UHF_wait',False]]
                if i==0:
                    slot_args_solo[0][9] = True
                take_ivs(iv_script_path, bias_lines_solo, 
                         file_names_solo, slots_solo, pysmurfs_solo, 
                         slot_args_solo,temperature)
                
                
        ### ------- Simultaneous ivs by bias line
        take_ivs(iv_script_path, bias_lines, 
                 file_names, slots, pysmurfs, slot_args, temperature,
                 UHF_wait=True)
        
        
        
                
except OSError:
    print(f'Something broke at MC temp {temperature} mK.')
    raise

print("Finished coldload ramp. Setting to 9 K")
temp_pid(ls336, 9)

