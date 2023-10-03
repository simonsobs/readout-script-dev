"""
Script combining Heather's VNA class and the VNA param sweep
into one python script. Edit and run.
The actual VNA sweep is defined at the very end of this file.
"""
import os
import socket
import time
import numpy as np
import argparse

# Fine VNA sweep start frequencies. Each sweep 0.2 GHz wide.
freqs = np.arange(3.8e9, 6.2e9, 0.2e9)

# The save directory must already exist on the VNA computer!
save_fmt = os.path.join(
    '"D:/State', # date_dr_device
    '20221017_pton_Mv13_Mv21_Mv26/cold_device/RF1_Mv26_S', # subdirectory structure
    '%.2f-f_%.2f_to_%.2f-bw_%.1f-atten_%.1f-volts_%.3f.%s"',
)
# % Fields in last line of save_fmt will be:
# timestamp,f_start_ghz, f_stop_ghz, if_bandwidth, power_level_abs,
# voltage, file extension

# Should not need to touch stuff below here
save_pth = save_fmt
tmp = save_pth.strip('"').replace("D:/State", "/data/vna")
if not os.path.isdir(os.path.dirname(tmp)):
    raise FileNotFoundError(
        "Cross-mount of output directory not found. "
        + "Does it exist on the VNA computer? "
        + os.path.dirname(save_pth)
    )

class VNA(object):
    def __init__(self,addr = ('192.168.88.7', 5025), save_pth=None):
        self.addr = addr
        
    def send_cmd(self, message):
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect(self.addr)
        s.send(message.encode())
        s.close()
        
    def query(self,message):
        time.sleep(1)
        s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        s.connect(self.addr)
        s.send(message.encode())
        result = s.recv(1024)
        s.close()
        return result
    
    def reset(self):
        message = ':SYST:PRES \n'
        self.send_cmd(self, message)
        time.sleep(2)
        
    def set_start_freq(self,start_freq):
        message = ':SENS1:FREQ:STAR %f \n' %start_freq
        self.send_cmd(message)
    
    def get_start_freq(self):
        message = ':SENS1:FREQ:STAR? \n'
        result = self.query(message)
        print(result)
        return result

    def set_end_freq(self,stop_freq):
        message = ':SENS1:FREQ:STOP %f \n' %stop_freq
        self.send_cmd(message)
    
    def get_stop_freq(self):
        message = ':SENS1:FREQ:STOP? \n'
        result = self.query(message)
        print(result)
        return result

    def set_num_points(self,num_of_points=10000):
        message = ':SENS:SWE:POIN %f \n' %num_of_points
        self.send_cmd(message)
    
    def get_num_points(self):
        message = ':SENS:SWE:POIN? \n'
        result = self.query(message)
        print(result)
        return result

    def set_if_bandwidth(self,if_bandwidth=1000):
        message = ':SENS1:BAND %f \n' %if_bandwidth
        self.send_cmd(message)


    def get_if_bandwidth(self):
        message = ':SENS1:BAND? \n'
        result = self.query(message)
        print(result)
        return result

    def set_power_level(self,power_level=-20):
        message = ':SOUR1:POW %f \n' %power_level
        self.send_cmd(message)
        
    def get_power_level(self):
        message = ':SOUR1:POW? \n'
        result = self.query(message)
        print(result)
        return result

    def set_s21_parameter(self):
        message = ':CALC1:PAR1:DEF S21 \n' 
        print(message)
        self.send_cmd(message)
    
    def get_s21_parameter(self):
        message = ':CALC1:PAR1:DEF? \n'
        result = self.query(message)
        print(result)
        return result

    def get_sweep_time(self):
        message = ':SENS1:SWE:TIME? \n'
        result = self.query(message)
        print(result)
        return result

    def get_s_data(self):
        message = ':CALC1:DATA:FDAT? \n' 
        result = self.query(message)
        print(result)
        return result

    def set_data_format_REIM(self):
        message = ':CALC1:FORM POL \n'
        self.send_cmd(message)
    
    def set_data_format_LOGMAG(self):
        message = ':CALC1:FORM MLOG \n'
        self.send_cmd(message)
    
    def get_data_format(self):
        message = ':CALC1:FORM? \n'
        result = self.query(message)
        print(result)
        return result
    
    def get_freq_data(self):
        message = ':SENS1:FREQ:DATA? \n'
        result = self.query(message)
        print(result)
        return result
    
    def get_data_type(self):
        message = ':FORM:DATA? \n'
        result = self.query(message)
        print(result)
        return result
    
    def set_data_type(self, dtyp='ASC'):
        #REAL OR ASC
        message = ':FORM:DATA %s \n' %dtyp
        self.send_cmd(message)
        
    def write_s2p_file(self,file_path):
        message = ':MMEM:STOR:SNP:FORM RI \n'
        self.send_cmd(message)
        message = ':MMEM:STOR:SNP:TYPE:S2P 1,2 \n'
        self.send_cmd(message)
        message = ':MMEM:STOR:SNP %s \n' % file_path
        self.send_cmd(message)
        print(message)
    
    def write_csv_file(self,file_path):
        message = ':MMEM:STOR:FDAT %s \n' % file_path
        print(message)
        self.send_cmd(message)
        
    def write_screen_image(self, file_path):
        message = ':MMEM:STOR:IMAG %s \n' % file_path
        print(message)
        self.send_cmd(message)
    
    def get_vna_info(self):
        message = ':MMEM:STOR:SNP:TYPE:S2P? \n'
        result = self.query(message)
        print(result)
        return result
    
    def set_active_trace(self,trace_num=1, channel_num=1):
        message = ':CALC1:PAR1:SEL'
        self.send_cmd(message)
        
    def get_error(self):
        message = ':SYST:ERR? \n'
        result = self.query(message)
        print(result)
        return result
    
    def normal_start_up(self):
        message = '*CLS \n'
        self.send_cmd(message)
    
        message = '*OPC? \n'
        self.send_cmd(message)
        
        #set our data type for recieving 
        self.set_data_type(dtyp='ASC')
        self.get_data_type()
        
        #set the number of points
        self.set_num_points(10000)
        self.get_num_points()
        
        #set the power level to -20
        self.set_power_level(power_level=-20)
        self.get_power_level()
    
        #set the vna to s21
        self.set_s21_parameter()
        
        #actiavte the trace 
        self.set_active_trace()
        
        #look at it in logmag
        self.set_data_format_LOGMAG()
        
    def get_freq_sweep(self,f_start, f_stop, voltage=0,
                       if_bandwidth=1000, power_level=-20):
        self.set_start_freq(f_start)
        self.get_start_freq()
        
        self.set_end_freq(f_stop)
        self.get_stop_freq()
        
        self.set_if_bandwidth(if_bandwidth)
        self.get_if_bandwidth()
        
        #get ready to take data and set the format
        self.set_data_format_REIM()
        self.get_data_format()
        
        #get sweep time
        sweep_time = self.get_sweep_time()
        wait_time = int(float(sweep_time)) + 5
        
        print('waiting %s s for freq sweep to be done' % wait_time)
        
        time.sleep(wait_time)
        
        print('finished freq sweep from %s to %s Hz at BW %s Hz and %s dB power'
              % (f_start, f_stop, if_bandwidth, power_level))
        
        f_start_ghz = f_start*1e-9
        f_stop_ghz = f_stop*1e-9
        power_level_abs = abs(power_level)
        
        #save csv file
        timestamp = time.time()
        file_path = save_pth % (
            timestamp, f_start_ghz, f_stop_ghz, if_bandwidth, power_level_abs, voltage, 'csv')
    
        message = ':MMEM:STOR:FDAT %s \n' % file_path
        self.send_cmd(message)
        print(message)
    
        time.sleep(2)
        
        #save s2p file
        file_path = save_pth %(
            timestamp,f_start_ghz, f_stop_ghz, if_bandwidth, power_level_abs, voltage, 's2p')
    
        message = ':MMEM:STOR:SNP:FORM RI \n'
        self.send_cmd(message)
    
        message = ':MMEM:STOR:SNP:TYPE:S2P 1,2 \n'
        self.send_cmd(message)
        
        message = ':MMEM:STOR:SNP %s \n' %file_path
        print(message)
        self.send_cmd(message)
        
        self.get_error()
        
        print('done saving data for sweep')
        
        time.sleep(2)

def run(save_pth, freqs=np.arange(3.8e9, 6.2e9, 0.2e9), if_bandwidth=100):
    '''
    save_path: str
        Formattable string for the vna results to be saved to.
        See VNA class above.
    freqs: array-like
        Frequencies (in Hz) at which to start vna sweeps.
        Each sweep is 0.2 GHz wide.
    '''
    vna = VNA(save_pth=save_pth)
    vna.normal_start_up()

    for freq in freqs:
        if freq < 5.0e9:
            atten = -15
        elif freq >= 5.0e9 and freq < 6.0e9:
            atten = -10
        elif freq >= 6.0e9 and freq < 7.0e9:
            atten = -5
        elif freq >= 7.0e9:
            atten = 0
        print(atten)
        f_start = freq
        f_stop = freq + 0.2e9
        
        vna.get_freq_sweep(f_start, f_stop, if_bandwidth=if_bandwidth, power_level=atten)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', default=False, action='store_true',
                        help="Set this kwarg to do faster VNA, for mapping only",
    )

    args = parser.parse_args()
    if args.fast:
        if_bandwidth = 1000
    else:
        if_bandwidth = 100
    run(save_pth=save_pth, freqs=freqs, if_bandwidth=if_bandwidth)
