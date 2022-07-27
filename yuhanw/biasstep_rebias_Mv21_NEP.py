'''
Code written in Jan 2022 by Yuhan Wang

measure biasstep and rebias TES based on the biasstep measurement 

for half UFM 
'''

import matplotlib
matplotlib.use('Agg')


import pysmurf.client
import argparse
import numpy as np
import os
import time
import glob
from sodetlib.det_config  import DetConfig
import numpy as np
from scipy.interpolate import interp1d
import argparse
import time
import csv
import scipy.signal as signal
from scipy.stats import norm
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")

target_Rtes = 4


def responsivity(data_time, phase_og, mask, tes_bias,band_list,chan_list,bias_groups,v_bias,out_dict,v_name,r_name,resp_name,nep_name):
    from scipy.signal import find_peaks
    import math

    import pysmurf.client
    from sodetlib.det_config  import DetConfig
    related_phase = []
    responsivity_array = []
    bias_power_array = []
    nep_array = []
    I0_array = []
    R0_array = []
    dP_P_array = []
    biasstep_count = 0
    bands, channels = np.where(mask!=-1)
    defined_step = 0.02
    ch_idx_self = 0 
    phase = phase_og * S._pA_per_phi0/(2.*np.pi*1e6) #uA
    period = 4
    fs = 200
    rsh = S._R_sh
    median_wl_chan_array = []
    sampleNums = np.arange(len(phase[ch_idx_self]))
    t_array = sampleNums / fs    
    sample_rate = fs
    phase = np.abs(phase)
    fig, axs = plt.subplots(2, 2, figsize=(11, 11), gridspec_kw={'width_ratios': [2, 2]})
    for i, (b, c) in enumerate(zip(bands, channels)):
        ##magic number here to match the dummy good det list
        for index_k in range(len(band_list)):
            if b == band_list[index_k]  and c == chan_list[index_k]:
                if (int(b),int(c)) not in out_dict.keys():
                    out_dict[(int(b),int(c))]={}
                resp_chan = []
                R0_chan = []
                biasp_chan = []
                I0_chan = []
                dP_P_chan = []
                median_wl_chan = []
        ## identifying possible steps
                target_phase_all = np.array(phase[i])  
                related_phase.append(target_phase_all-np.mean(target_phase_all))
                axs[1,1].plot(t_array,target_phase_all-np.mean(target_phase_all),color='C0',alpha = 0.1)
                if(target_phase_all[0]<target_phase_all[-1]):
                    target_phase_all = -target_phase_all
                average_phase = np.ones(len(target_phase_all))*np.mean(target_phase_all)
                dary = np.array([*map(float, target_phase_all)])
                dary -= np.average(dary)
                step = np.hstack((np.ones(int(1*len(dary))), -1*np.ones(int(1*len(dary)))))
                neg_dary = np.max(dary)-dary
                dary_step = np.convolve(dary, step, mode='valid')
                step_indx = np.argmax(dary_step)
                dip_index,_ = find_peaks(-dary_step)
                peak_index,_ = find_peaks(dary_step)

                mid_points = np.append(dip_index,peak_index)
        ##now looking at each chunk of data
#                 print(mid_points)
                for target_dip in mid_points:
                    target_phase = []
                    target_time = []
                    noise_phase = []
                    noise_time = []
                    start_time = t_array[target_dip]
                    ## defining what data to look at
                    for j, t_array_select in enumerate(t_array):
                        if t_array_select < start_time + 0.25*period and t_array_select > start_time - 0.25*period:
                            target_time.append(t_array_select)
                            target_phase.append(phase[i][j])

                        if t_array_select < start_time - 0.1*period and t_array_select > start_time - 0.9*period:
                            noise_time.append(t_array_select)
                            noise_phase.append(phase[i][j])

                    detrend = 'constant'
                    noise_phase = np.array(noise_phase) *1e6 ## back in pA
                    f, Pxx = signal.welch(noise_phase,
                        fs=fs, detrend=detrend,nperseg=2**16) 
                    Pxx = np.sqrt(Pxx)
                    fmask = (fmin < f) & (f < fmax)
                    wl = np.median(Pxx[fmask])
                    median_wl_chan.append(wl)



                    target_phase = np.array(target_phase)        
                    if(target_phase[0]<target_phase[-1]):
                        target_phase = -target_phase
#                         print(len(target_phase))
                    average_phase = np.ones(len(target_time))*np.mean(target_phase)
                    ## finding the step again
                    dary_perstep = np.array([*map(float, target_phase)])
                    dary_perstep -= np.average(dary_perstep)
                    step = np.hstack((np.ones(int(1*len(dary_perstep))), -1*np.ones(int(1*len(dary_perstep)))))
                    dary_step = np.convolve(dary_perstep, step, mode='valid')
                    step_indx = np.argmax(dary_step)
                    dip_index,_ = find_peaks(-dary_step, height=0)

                    target_time_step= np.arange(0,len(target_phase)).astype(float)/sample_rate
                    sample_rate = fs
                    # linear fitting the upper and lower part of steps
                    target_phase_step_up = target_phase_all[target_dip-int(0.4*period*sample_rate):target_dip-int(0.1*period*sample_rate)]
                    target_time_step_up = t_array[target_dip-int(0.4*period*sample_rate):target_dip-int(0.1*period*sample_rate)]
                    sample_rate = fs
                    target_phase_step_down = target_phase_all[target_dip+int(0.1*period*sample_rate):target_dip+int(0.4*period*sample_rate)]
                    target_time_step_down = t_array[target_dip+int(0.1*period*sample_rate):target_dip+int(0.4*period*sample_rate)]
                    step_size_meas = np.abs(np.median(target_phase_step_up) - np.median(target_phase_step_down))
                    deltaItes = -step_size_meas/1e6 
                    deltaIbias = defined_step / S._bias_line_resistance  #A
                    Ibias = np.abs( v_bias / S._bias_line_resistance) #A
                    bias_power = (Ibias**2 *rsh * ((deltaItes/deltaIbias)**2 - (deltaItes/deltaIbias))/(1 - 2*(deltaItes/deltaIbias))**2) #W
                    temp = Ibias**2 - 4*bias_power/rsh
                    R0 = rsh * (Ibias + np.sqrt(temp))/(Ibias - np.sqrt(temp)) #ohm
                    I0 =  0.5 * (Ibias - np.sqrt(temp)) #A 
                    V0 = rsh*(Ibias - I0) #V
                    pbias_smurf = V0 * I0
#                     dP_dI = (V0 - rsh*bias_power / V0) *1e6 #w/A --> pW/uA
                    dP_dI = (Ibias-2*I0)* rsh  *1e6 #w/A --> pW/uA
# #                         print(dP_dI)
                    dP_P = (dP_dI * deltaItes) / (bias_power * 1e6)
                    Si = -1/(I0*(R0-rsh))
                    Ibias_cal = I0*(1+R0/rsh)
                    vbias_cal = Ibias_cal * S._bias_line_resistance
                    # if not math.isnan(np.median(dP_dI)):
                    resp_chan.append(dP_dI)
                    # if not math.isnan(np.median(R0)):
                    R0_chan.append(R0)
                    biasp_chan.append(np.nanmedian(bias_power))
                    I0_chan.append(np.nanmedian(I0))
                    dP_P_chan.append(np.nanmedian(dP_P))

                # if not math.isnan(np.median(Si)):
                

                responsivity_array.append(np.median(Si)/1e6)
                biasstep_count = biasstep_count + 1
                R0_array.append(np.median(R0_chan))
                bias_power_array.append(np.nanmedian(biasp_chan)*1e12) 
                I0_array.append(np.nanmedian(I0_chan)*1e6) 
                dP_P_array.append(np.nanmedian(dP_P_chan))


                median_wl = np.nanmedian(median_wl_chan)
                median_nep = -1 * median_wl / (np.nanmedian(Si)/1e6)

                nep_array.append(median_nep)

                out_dict[(int(b),int(c))][v_name] = v_bias
                out_dict[(int(b),int(c))][r_name] = np.nanmedian(R0_chan) * 1000
                out_dict[(int(b),int(c))][resp_name] = np.nanmedian(Si)/1e6
                out_dict[(int(b),int(c))][nep_name] = median_nep


    R0_array = np.array(R0_array) * 1000
    common_mode = np.median(related_phase, axis = 0)
    try:
        
        low=0
        high=10
        step=0.1
        axs[0,0].hist(R0_array,bins=np.arange(low,high,step),histtype= u'step',linewidth=2,label="rough Rtes")
        axs[0,0].axvline(np.nanmedian(R0_array),linestyle='--', color='gray',label="median rough R0_array {}".format(np.round(np.nanmedian(R0_array),2)))
        axs[0,0].legend()
        axs[0,0].grid(which='both')
        axs[0,0].set_xlabel('Rtes [mOhm]')
        axs[0,0].set_ylabel('count')
        axs[0,0].set_title('rough BL {}, yield {}, median Rtes {} [mOhm]'.format(bias_groups,biasstep_count,np.round(np.nanmedian(R0_array),2)))

        low=-20
        high=0
        step=0.1
        axs[1,0].hist(responsivity_array,bins=np.arange(low,high,step),histtype= u'step',linewidth=2,label="rough responsivity")
        axs[1,0].axvline(np.nanmedian(responsivity_array),linestyle='--', color='gray',label="median rough responsivity {} [uV-1]".format(np.round(np.nanmedian(responsivity_array),2)))
        axs[1,0].legend()
        axs[1,0].grid(which='both')
        axs[1,0].set_xlabel('Si [uV-1]')
        axs[1,0].set_ylabel('count')
        axs[1,0].set_title('rough BL {}, yield {}, median responsivity {} [uV-1]'.format(bias_groups,biasstep_count,np.round(np.nanmedian(responsivity_array),2)))


        axs[0,1].hist(bias_power_array,bins=100,histtype= u'step',linewidth=2,label="biaspower (pW)")
        axs[0,1].axvline(np.nanmedian(bias_power_array),linestyle='--', color='gray',label="median rough bias_power_array {}".format(np.round(np.nanmedian(bias_power_array),2)))
        axs[0,1].grid(which='both')
        axs[0,1].set_xlabel('Ptes [pW]')
        axs[0,1].set_ylabel('count')
        axs[0,1].set_title('rough BL {}, yield {}, median bias power {} [pW]'.format(bias_groups,biasstep_count,np.round(np.nanmedian(bias_power_array),2)))

        
        # axs[1,1].axvline(np.median(I0_array),linestyle='--', color='gray',label="median rough bias current {}".format(np.round(np.median(I0_array),2)))
        # axs[1,1].plot(t_array,common_mode,color='C0')
        axs[1,1].grid(which='both')
        axs[1,1].set_xlabel('time [s]')
        axs[1,1].set_ylabel('phase [uA]')
        axs[1,1].set_title('rough BL {}, yield {}, median phase [uA]'.format(bias_groups,biasstep_count))

    
        save_name = f'{data_time}_bl{bias_groups}_biasstep_param.png'
        print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
        plt.savefig(os.path.join(S.plot_dir, save_name))
    except:
        print('empty bin')
        
    return responsivity_array,R0_array,bias_power_array,nep_array






## hard coding biasline mapping for now

target_band_chan = [
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[],
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[6, 10, 58, 74, 114, 122, 138, 146, 186, 202, 234, 242, 266, 274, 282, 298, 306, 338, 346, 354, 362, 378, 390, 402, 410, 426, 434, 466, 482, 490, 506, 8, 13, 16, 29, 31, 36, 40, 45, 47, 56, 63, 64, 72, 77, 80, 85, 95, 96, 109, 112, 120, 127, 132, 133, 141, 144, 149, 157, 160, 164, 165, 168, 173, 175, 191, 192, 196, 200, 208, 224, 232, 239, 240, 245, 248, 256, 260, 261, 264, 277, 280, 287, 288, 293, 309, 312, 325, 333, 341, 351, 352, 357, 360, 365, 367, 368, 373, 383, 384, 388, 389, 392, 413, 415, 420, 421, 424, 431, 437, 440, 447, 448, 452, 453, 461, 469, 472, 479, 485, 488, 493, 504],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[3, 13, 51, 61, 83, 91, 107, 125, 141, 147, 155, 171, 179, 205, 211, 221, 235, 245, 253, 259, 269, 275, 283, 317, 333, 347, 445, 461, 467, 491, 2, 6, 12, 26, 34, 44, 50, 66, 74, 76, 82, 90, 98, 108, 114, 134, 138, 140, 146, 154, 156, 162, 172, 188, 194, 202, 204, 210, 234, 236, 242, 250, 252, 258, 268, 282, 284, 306, 322, 354, 370, 378, 380, 386, 396, 402, 412, 418, 428, 442, 444, 460, 490, 492, 498, 500, 508, 0, 7, 15, 16, 23, 31, 32, 47, 59, 63, 71, 87, 91, 95, 111, 119, 135, 143, 151, 159, 160, 175, 183, 191, 199, 208, 219, 224, 231, 251, 255, 256, 263, 272, 283, 311, 315, 320, 327, 335, 336, 343, 351, 352, 367, 375, 379, 384, 399, 400, 411, 416, 423, 443, 448, 455, 463, 471, 479, 487, 495],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[2, 4, 11, 12, 29, 36, 40, 44, 52, 60, 67, 68, 75, 76, 84, 88, 92, 104, 109, 120, 124, 132, 156, 163, 180, 195, 196, 200, 204, 212, 213, 232, 237, 244, 248, 252, 258, 260, 264, 267, 277, 280, 284, 285, 292, 296, 308, 316, 323, 324, 328, 341, 348, 356, 360, 372, 386, 388, 392, 395, 396, 405, 412, 420, 424, 429, 436, 451, 456, 468, 476, 484, 492, 499, 504, 7, 11, 15, 35, 55, 59, 75, 87, 99, 119, 135, 139, 147, 151, 163, 187, 195, 199, 203, 235, 251, 259, 267, 271, 291, 295, 299, 311, 315, 323, 327, 335, 343, 355, 363, 371, 379, 387, 395, 399, 403, 407, 427, 439, 443, 455, 459, 471, 475, 503, 507],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[15, 39, 48, 79, 103, 176, 199, 255, 271, 272, 319, 327, 335, 336, 383, 400, 423, 464, 8, 12, 24, 28, 36, 37, 44, 45, 51, 56, 60, 66, 88, 100, 104, 107, 115, 116, 117, 124, 125, 136, 140, 148, 164, 165, 168, 171, 173, 184, 194, 197, 204, 205, 216, 221, 228, 232, 236, 243, 244, 245, 248, 252, 253, 258, 259, 260, 269, 275, 276, 296, 300, 307, 308, 317, 324, 328, 332, 333, 348, 356, 357, 363, 368, 373, 376, 386, 387, 388, 392, 396, 404, 408, 412, 421, 428, 429, 436, 444, 445, 450, 452, 456, 467, 468, 472, 476, 483, 485, 499, 504, 508, 509],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[7, 31, 50, 59, 82, 90, 96, 111, 114, 122, 123, 151, 154, 160, 175, 178, 183, 187, 192, 202, 218, 242, 251, 262, 263, 266, 282, 287, 288, 306, 311, 314, 320, 338, 343, 346, 352, 367, 370, 379, 407, 410, 416, 431, 434, 443, 448, 466, 471, 474, 490, 495, 4, 16, 20, 21, 29, 31, 36, 37, 45, 48, 56, 68, 72, 77, 85, 88, 93, 95, 101, 104, 112, 127, 128, 144, 168, 173, 181, 196, 200, 205, 208, 221, 223, 228, 229, 232, 240, 256, 276, 287, 292, 293, 296, 301, 309, 319, 324, 328, 333, 336, 349, 351, 352, 357, 360, 365, 368, 384, 397, 400, 408, 415, 416, 421, 424, 437, 447, 452, 453, 456, 477, 479, 480, 484, 496, 504],
[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[3, 7, 11, 27, 35, 39, 43, 51, 59, 67, 75, 83, 91, 103, 115, 123, 125, 163, 187, 203, 211, 215, 227, 231, 235, 251, 259, 267, 283, 291, 295, 307, 311, 315, 343, 355, 359, 363, 379, 381, 395, 407, 411, 419, 423, 435, 445, 451, 455, 459, 467, 471, 475, 483, 491, 509],
]

biasstep_dict = {}

og_V_tes_array = S.get_tes_bias_bipolar_array()
print('current Vbias array')
print(og_V_tes_array)
step_size = 0.02
dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(og_V_tes_array) 
    time.sleep(2) 
    og_V_tes_array_step = np.array(og_V_tes_array)[6:12] + step_size
    og_V_tes_array_step = np.append(og_V_tes_array_step,[0,0,0])
    og_V_tes_array_step = np.append([0,0,0,0,0,0],og_V_tes_array_step)
    S.set_tes_bias_bipolar_array(og_V_tes_array_step) 
    time.sleep(2) 
S.stream_data_off() 

S.set_tes_bias_bipolar_array(og_V_tes_array) 
data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

og_R_tes_array = []
new_bias_array= []


for bl_num in [6,7,8,9,10,11]:
    og_V_tes_array = S.get_tes_bias_bipolar_array()
    # print('og_V_tes_array')
    # print(og_V_tes_array)
    responsivity_array,R0_array,bias_power_array,nep_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,og_V_tes_array[bl_num],biasstep_dict,'v1','r1','si1','nep1')
    print(len(responsivity_array))
    og_R_tes_array.append(np.nanmedian(R0_array))
    # new_bias_array.append(og_V_tes_array[bl_num]+0.2)


    if np.nanmedian(R0_array)<= 4:
        new_bias_array.append(og_V_tes_array[bl_num]+1)
    elif np.nanmedian(R0_array) > 4:
        new_bias_array.append(og_V_tes_array[bl_num]-1)
    # else if:
    #     new_bias_array.append(og_V_tes_array[bl_num]+0.1)
print('current Rtes median:')
print(og_R_tes_array)
new_bias_array = np.array(new_bias_array)
print(len(new_bias_array))
new_bias_array = np.append(new_bias_array,[0,0,0])
new_bias_array = np.append([0,0,0,0,0,0],new_bias_array)
S.set_tes_bias_bipolar_array(new_bias_array)
print('trial step vbias:')

print(S.get_tes_bias_bipolar_array())

time.sleep(10) 

dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(new_bias_array) 
    time.sleep(2) 
    new_bias_array_step = np.array(new_bias_array)[6:12] + step_size
    new_bias_array_step = np.append(new_bias_array_step,[0,0,0])
    new_bias_array_step = np.append([0,0,0,0,0,0],new_bias_array_step)
    S.set_tes_bias_bipolar_array(new_bias_array_step)  
    time.sleep(2) 
S.stream_data_off() 

data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

new_R_tes_array = []
for bl_num in [6,7,8,9,10,11]:
    responsivity_array,R0_array,bias_power_array,nep_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,new_bias_array[bl_num],biasstep_dict,'v2','r2','si2','nep2')
    print(len(responsivity_array))
    new_R_tes_array.append(np.nanmedian(R0_array))
print('current Rtes median')
print(new_R_tes_array)



v_estimate_array = []
for bl_num in [6,7,8,9,10,11]:
    v_estimate_array_bl = []
    weight_bl = []
    for index_k in range(len(target_band_chan[2*bl_num])):
        pick_b = target_band_chan[2*bl_num][index_k]
        pick_c = target_band_chan[2*bl_num+1][index_k]


    # v_estimate = (og_V_tes_array[bl_num] * (target_Rtes - new_R_tes_array[bl_num]) + new_bias_array[bl_num] * (og_R_tes_array[bl_num] - target_Rtes)) /(og_R_tes_array[bl_num] - new_R_tes_array[bl_num])
        try:
            R2 = biasstep_dict[(int(pick_b),int(pick_c))]['r2']
            R1 = biasstep_dict[(int(pick_b),int(pick_c))]['r1']
            NEP1 = biasstep_dict[(int(pick_b),int(pick_c))]['nep1']
            V2 = biasstep_dict[(int(pick_b),int(pick_c))]['v2']
            V1 = biasstep_dict[(int(pick_b),int(pick_c))]['v1']
            NEP2 = biasstep_dict[(int(pick_b),int(pick_c))]['nep1']

        # v_estimate = ((R2-R1)/(target_Rtes-R1))*(V2-V1) + V1
            v_estimate = (V1*(target_Rtes-R2) + V2*(R1-target_Rtes))/(R1-R2)
            
            weight = 1/(NEP1)**2
            weight_bl.append(weight)
            v_estimate_array_bl.append(v_estimate)
        except:
            continue

    v_rough = np.nanmedian(v_estimate_array_bl)
    possible_outcome = []
#     print(v_estimate_array_bl)
#     print(weight_bl)
    for v_choice in np.arange(v_rough-1,v_rough+1,0.1):
        array_nep = 0
        for index , per_weight in enumerate(weight_bl):
            if v_estimate_array_bl[index] > 0:
                weight_NEP = per_weight*np.abs(v_choice-(v_estimate_array_bl[index]))
#             print(weight_NEP)
            if not math.isnan(weight_NEP):
                array_nep = array_nep + weight_NEP
        possible_outcome.append(array_nep)

#     print(possible_outcome)
    min_outcome = np.min(possible_outcome)
    best_V = np.arange(v_rough-1,v_rough+1,0.1)[possible_outcome.index(min_outcome)]
    print(best_V)

    fig = plt.figure(figsize=(11,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(which='minor', color='#CCCCCC', linestyle='--')
    ax.grid(which='major', color='#CCCCCC', linestyle='-')
    ax.plot(np.arange(v_rough-1,v_rough+1,0.1),possible_outcome,color='C0')
    ax.set_xlabel('bias voltage [V]')
    ax.set_ylabel('BL bias voltage NEP metric',color='C0')
    save_name = f'{data_time}_bl{bl_num}_biasvoltage_metric.png'
    print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
    plt.savefig(os.path.join(S.plot_dir, save_name))






    # print("number of estimated v within biasline")
    # print(len(v_estimate_array_bl))
    v_estimate_array.append(best_V)




# #     v_estimate_array.append(v_estimate)

print('new estimation vbias:')
print(v_estimate_array)

print('applying new voltage bias')
v_estimate_array = np.append(v_estimate_array,[0,0,0])
v_estimate_array = np.append([0,0,0,0,0,0],v_estimate_array)
S.set_tes_bias_bipolar_array(v_estimate_array)
time.sleep(10) 

dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(v_estimate_array) 
    time.sleep(2) 
    new_bias_array_step = np.array(v_estimate_array)[6:12] + step_size
    new_bias_array_step = np.append(new_bias_array_step,[0,0,0])
    new_bias_array_step = np.append([0,0,0,0,0,0],new_bias_array_step)
    S.set_tes_bias_bipolar_array(new_bias_array_step)  
    time.sleep(2) 
S.stream_data_off() 



data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

new_R_tes_array = []
new_resp_array = []
for bl_num in [6,7,8,9,10,11]:
    responsivity_array,R0_array,bias_power_array,nep_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,v_estimate_array[bl_num],biasstep_dict,'v3','r3','si3','nep3')
    print(len(responsivity_array))
    new_R_tes_array.append(np.nanmedian(R0_array))
    new_resp_array.append(np.nanmedian(responsivity_array))
print('current Rtes median')
print(new_R_tes_array)
print('current resp median')
print(new_resp_array)


# # print(biasstep_dict[(3,18)])

timestamp=S.get_timestamp()
save_name = '{}_biasstep_rebias.npy'.format(timestamp)
print(f'Saving data to {os.path.join(S.output_dir, save_name)}')
biasstep_rebia_data = os.path.join(S.output_dir, save_name)
path = os.path.join(S.output_dir, biasstep_rebia_data) 
np.save(path, biasstep_dict)


