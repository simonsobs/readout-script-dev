from ocs.ocs_client import OCSClient
import time
import numpy as np
import sys
import signal


# ================ User Parameters ================ 

#**TODO**: Change the PIDing back to not PID the first temperature later!!
print("Initialize pysmurf client")
temperature_list = [60,70,80,90,100,120,140,150,160,170,180,190,200] # used to go to 210,220]; didn't include 50 out of fear it wouldn't go that low with the cold load at 14K.
#temperature_list = [120,120,120,120,120,160,160,160,160,160,180,180,180,180,180]
#temperature_list = [80,80,80,80,80,100,100,100,100,100,180,180,180,180,180]
#temperature_list = [50,60,70,80,90,100,120,140,150,160,170,180,190,200] # used to go to 210,220]
# ^If MC X-066,     [58,67,75,84,93,102,119,137,154,172,180,189,198,207] =actual MC temp post calibration of above
#temperature_list = [60,80]
pid_thermometer_channel = 15
noise_script_path = '/readout-script-dev/rsonka/iv_takers/hcm_bath_noise_by_BL.py'
iv_script_path    = '/readout-script-dev/rsonka/iv_takers/hcm_bath_iv_by_BL.py'

slots = [4,5,6]
# filenames v whose slot isn't activated above^ will not be used.
file_names = {2: '/data/smurf_data/UFM_testing/Mv32_ph008/bathramp_noise_2022122.csv',
              3: '/data/smurf_data/UFM_testing/Mv14_pg010/bathramp_noise_20220321_test.csv',
              4: '/data/smurf_data/UFM_testing/Uv41_pi004/bathramp_variance_20230927.csv',
              5: '/data/smurf_data/UFM_testing/Uv42_pi004/bathramp_variance_20230927.csv',
              6: '/data/smurf_data/UFM_testing/Uv44_pi004/bathramp_variance_20230927.csv',
              7: '/data/smurf_data/UFM_testing/Lp3_pi002/bathramp_noise_so_20230808.csv'}
bias_lines = [0,1,2,3,4,5,6,7,8,9,10,11] #[0,1,2,3,4,5,6,7,8,9,10,11] # which to run on
bias_lines_solo = [4,7,9,11] # Chosen based on yield results
do_solo_UFMs = False # If true, before simultaneous data taking, will take data on one UFM at a time as well. 


# ================ Variable and Function Definitions ================ 

file_names_solo = {slot: file_names[slot][:-4]+"_solo.csv" for slot in slots}              
pysmurfs = [OCSClient(f'pysmurf-controller-s{slot}', 
                      args=['--site-http=http://localhost:8001/call']) for slot in slots]
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

def get_temps(client):
    # New section 20220420: Records all Lakeshore  Channels
    ls_temps = []
    for ch in [13,14,15,16]:
        temp_success = False
        while temp_success is False:
            ret = client.get_channel_attribute(attribute='kelvin_reading',channel=ch)
            temp_success = ret.session['success']
        ls_temps.append("%0d" % (round(ret.session['data']['kelvin_reading']*1e3)))
    ls_temps = " ".join(ls_temps)
    return ls_temps

def check_success(pysmurfs):
    for slot_mc in pysmurfs:#[pysmurf4,pysmurf5,pysmurf6]:
        ret = slot_mc.run.wait()
        if not ret.session['success']:
            raise OSError(
                'OCS script failed to run. Check ocs-pysmurf logs.')

def take_noise(noise_script_path,bias_lines, file_names, ls_temps, slots, pysmurfs):
    print(f"taking simultaneous noise at {ls_temps} via {noise_script_path}")
    slot_args = [ ['--slot',slot,
                   '--temp', "will be overwritten",
                   '--bgs', " ".join([str(bl) for bl in bias_lines]),
                   '--output_file', file_names[slot],
                   '--UHF_wait',0] for slot in slots] # It wasn't working passing bools
    for i in range(len(slots)):
        pysmurfs[i].run.start(script=noise_script_path, args=slot_args[i])
    check_success(pysmurfs)
    return slot_args
        
def take_ivs(iv_script_path, bias_lines, file_names, slots, pysmurfs, slot_args, UHF_wait=False):
    for j in range(len(bias_lines)):
        bl=bias_lines[j]            
        ls_temps = get_temps(client)
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
        if temperature != -42: #temperature_list[0]: # Make sure to change back later!
            t0 = time.time()
            temp_pid(client, temperature, channel=pid_thermometer_channel)
            now = time.time()
            while now-t0 < 60 * 15:
                time.sleep(30)
                now = time.time()
        
        # -------- PID for this temperature set IS COMPLETE.
        ls_temps = get_temps(client) 
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
                ls_temps = get_temps(client) 
                slot_args_solo = [ ['--slot',slots[i],
                               '--temp', ls_temps,
                               '--bgs', " ".join([str(bl) for bl in bias_lines_solo]),
                               '--output_file', file_names_solo[slots[i]],
                               '--UHF_wait',False]]
                if i==0:
                    slot_args_solo[0][9] = True
                take_ivs(iv_script_path, bias_lines_solo, 
                         file_names_solo, slots_solo, pysmurfs_solo, 
                         slot_args_solo)
                
                
        ### ------- Simultaneous ivs by bias line
        take_ivs(iv_script_path, bias_lines, 
                 file_names, slots, pysmurfs, slot_args,
                 UHF_wait=True)
        
        
        
                
except OSError:
    print(f'Something broke at MC temp {temperature} mK.')
    raise

print('Finished bath ramp. Setting back to 100 mK')
temp_pid(client, 100, channel=pid_thermometer_channel)
