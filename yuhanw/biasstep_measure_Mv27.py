'''
Code written in Jan 2022 by Yuhan Wang

measure biasstep and rebias TES based on the biasstep measurement 
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

target_Rtes = 4.3


def responsivity(data_time, phase_og, mask, tes_bias,band_list,chan_list,bias_groups,v_bias):
    from scipy.signal import find_peaks
    import math

    import pysmurf.client
    from sodetlib.det_config  import DetConfig
    related_phase = []
    responsivity_array = []
    bias_power_array = []
    I0_array = []
    R0_array = []
    dP_P_array = []
    biasstep_count = 0
    bands, channels = np.where(mask!=-1)
    defined_step = 0.1
    ch_idx_self = 0 
    phase = phase_og * S._pA_per_phi0/(2.*np.pi*1e6) #uA
    period = 4
    fs = 200
    rsh = S._R_sh
    sampleNums = np.arange(len(phase[ch_idx_self]))
    t_array = sampleNums / fs    
    sample_rate = fs
    phase = np.abs(phase)
    for i, (b, c) in enumerate(zip(bands, channels)):
        ##magic number here to match the dummy good det list
        for index_k in range(len(band_list)):
            if b == band_list[index_k]  and c == chan_list[index_k]:
                resp_chan = []
                R0_chan = []
                biasp_chan = []
                I0_chan = []
                dP_P_chan = []
        ## identifying possible steps
                target_phase_all = np.array(phase[i])  
                related_phase.append(target_phase_all-np.mean(target_phase_all))
                if(target_phase_all[0]<target_phase_all[-1]):
                    target_phase_all = -target_phase_all
                average_phase = np.ones(len(target_phase_all))*np.mean(target_phase_all)
                dary = np.array([*map(float, target_phase_all)])
                dary -= np.average(dary)
                step = np.hstack((np.ones(int(1*len(dary))), -1*np.ones(int(1*len(dary)))))
                neg_dary = np.max(dary)-dary
                dary_step = np.convolve(dary, step, mode='valid')
                step_indx = np.argmax(dary_step)
                dip_index,_ = find_peaks(-dary_step, height=0)
                peak_index,_ = find_peaks(dary_step, height=0)

                mid_points = np.append(dip_index,peak_index)
        ##now looking at each chunk of data
#                 print(mid_points)
                for target_dip in mid_points:
                    target_phase = []
                    target_time = []
                    start_time = t_array[target_dip]
                    ## defining what data to look at
                    for j, t_array_select in enumerate(t_array):
                        if t_array_select < start_time + 0.25*period and t_array_select > start_time - 0.25*period:
                            target_time.append(t_array_select)
                            target_phase.append(phase[i][j])
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
                    if not math.isnan(np.median(dP_dI)):
                        resp_chan.append(dP_dI)
                    if not math.isnan(np.median(R0)):
                        R0_chan.append(R0)
                    biasp_chan.append(np.nanmedian(bias_power))
                    I0_chan.append(np.nanmedian(I0))
                    dP_P_chan.append(np.nanmedian(dP_P))
                if not math.isnan(np.median(Si)):
                    responsivity_array.append(np.median(Si)/1e6)
                    biasstep_count = biasstep_count + 1
                    R0_array.append(np.median(R0_chan))
                bias_power_array.append(np.nanmedian(biasp_chan)*1e12) 
                I0_array.append(np.nanmedian(I0_chan)*1e6) 
                dP_P_array.append(np.nanmedian(dP_P_chan))
    R0_array = np.array(R0_array) * 1000
    common_mode = np.median(related_phase, axis = 0)
    try:
        fig, axs = plt.subplots(2, 2, figsize=(11, 11), gridspec_kw={'width_ratios': [2, 2]})
        low=0
        high=10
        step=0.1
        axs[0,0].hist(R0_array,bins=np.arange(low,high,step),histtype= u'step',linewidth=2,label="rough Rtes")
        axs[0,0].axvline(np.median(R0_array),linestyle='--', color='gray',label="median rough R0_array {}".format(np.round(np.median(R0_array),2)))
        axs[0,0].legend()
        axs[0,0].grid(which='both')
        axs[0,0].set_xlabel('Rtes [mOhm]')
        axs[0,0].set_ylabel('count')
        axs[0,0].set_title('rough BL {}, yield {}, median Rtes {} [mOhm]'.format(bias_groups,biasstep_count,np.round(np.median(R0_array),2)))

        low=-20
        high=0
        step=0.1
        axs[1,0].hist(responsivity_array,bins=np.arange(low,high,step),histtype= u'step',linewidth=2,label="rough responsivity")
        axs[1,0].axvline(np.median(responsivity_array),linestyle='--', color='gray',label="median rough responsivity {} [uV-1]".format(np.round(np.median(responsivity_array),2)))
        axs[1,0].legend()
        axs[1,0].grid(which='both')
        axs[1,0].set_xlabel('Si [uV-1]')
        axs[1,0].set_ylabel('count')
        axs[1,0].set_title('rough BL {}, yield {}, median responsivity {} [uV-1]'.format(bias_groups,biasstep_count,np.round(np.median(responsivity_array),2)))


        axs[0,1].hist(bias_power_array,bins=100,histtype= u'step',linewidth=2,label="biaspower (pW)")
        axs[0,1].axvline(np.median(bias_power_array),linestyle='--', color='gray',label="median rough bias_power_array {}".format(np.round(np.median(bias_power_array),2)))
        axs[0,1].grid(which='both')
        axs[0,1].set_xlabel('Ptes [pW]')
        axs[0,1].set_ylabel('count')
        axs[0,1].set_title('rough BL {}, yield {}, median bias power {} [pW]'.format(bias_groups,biasstep_count,np.round(np.median(bias_power_array),2)))

        axs[1,1].plot(t_array,common_mode,color='C0')
        # axs[1,1].axvline(np.median(I0_array),linestyle='--', color='gray',label="median rough bias current {}".format(np.round(np.median(I0_array),2)))
        axs[1,1].grid(which='both')
        axs[1,1].set_xlabel('time [s]')
        axs[1,1].set_ylabel('phase [uA]')
        axs[1,1].set_title('rough BL {}, yield {}, median phase [uA]'.format(bias_groups,biasstep_count))

    
        save_name = f'{data_time}_bl{bias_groups}_biasstep_param.png'
        print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
        plt.savefig(os.path.join(S.plot_dir, save_name))
    except:
        print('empty bin')
        
    return responsivity_array,R0_array,bias_power_array






## hard coding biasline mapping for now

target_band_chan = [
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
,
[12, 15, 18, 26, 34, 35, 44, 55, 58, 60, 71, 74, 75, 83, 87, 91, 99, 106, 107, 108, 114, 116, 131, 135, 140, 151, 155, 156, 163, 167, 170, 171, 172, 178, 179, 183, 187, 188, 195, 204, 215, 219, 226, 227, 231, 234, 235, 236, 243, 247, 251, 253, 258, 263, 266, 267, 274, 275, 279, 283, 291, 298, 299, 307, 308, 314, 315, 322, 323, 327, 331, 332, 338, 347, 348, 359, 371, 372, 375, 378, 380, 387, 395, 402, 403, 407, 423, 426, 435, 439, 442, 443, 450, 459, 466, 467, 475, 476, 483, 487, 491, 492, 498, 499, 500, 507, 509]
,
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
,
[6, 8, 15, 16, 23, 26, 38, 39, 47, 48, 50, 58, 63, 70, 90, 134, 138, 143, 144, 151, 167, 178, 186, 198, 207, 208, 223, 231, 234, 242, 255, 256, 266, 282, 298, 303, 306, 320, 327, 335, 336, 346, 351, 362, 368, 370, 375, 383, 384, 394, 399, 400, 407, 410, 423, 426, 431, 432, 434, 442, 447, 455, 458, 474, 496, 506, 4, 8, 16, 21, 29, 31, 36, 45, 52, 63, 68, 77, 84, 85, 100, 104, 109, 127, 132, 136, 144, 149, 159, 160, 164, 168, 173, 196, 200, 205, 212, 223, 229, 237, 248, 256, 269, 276, 280, 285, 288, 293, 296, 304, 309, 312, 317, 319, 320, 336, 341, 344, 349, 351, 352, 357, 360, 368, 373, 376, 381, 384, 388, 392, 400, 404, 408, 413, 415, 416, 421, 429, 432, 437, 440, 445, 448, 452, 456, 461, 464, 469, 472, 477, 480, 484, 485, 488, 501, 511]
,
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
,
[32, 111, 183, 187, 239, 263, 343, 352, 443, 471, 3, 8, 12, 13, 24, 28, 29, 35, 36, 52, 56, 69, 75, 76, 77, 83, 85, 88, 92, 93, 100, 115, 116, 120, 124, 125, 130, 131, 133, 136, 141, 147, 148, 157, 164, 168, 173, 180, 181, 188, 197, 200, 204, 205, 212, 213, 221, 227, 232, 236, 237, 244, 245, 248, 249, 252, 258, 260, 261, 268, 269, 275, 276, 277, 280, 284, 285, 291, 293, 296, 312, 316, 317, 323, 324, 325, 328, 331, 339, 340, 341, 348, 356, 360, 364, 365, 368, 373, 388, 392, 395, 396, 403, 404, 408, 412, 413, 421, 424, 428, 435, 436, 437, 440, 445, 452, 453, 467, 476, 477, 483, 484, 496, 499, 504, 505, 508]
,
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
,
[3, 12, 24, 27, 34, 36, 40, 44, 45, 51, 52, 56, 60, 66, 68, 76, 77, 88, 104, 116, 124, 125, 132, 152, 155, 156, 162, 164, 179, 180, 194, 196, 204, 205, 211, 212, 216, 220, 228, 232, 244, 252, 253, 258, 259, 260, 268, 276, 280, 296, 300, 312, 316, 328, 339, 340, 347, 348, 364, 376, 386, 388, 396, 397, 404, 411, 420, 424, 428, 436, 440, 444, 468, 484, 492, 504, 508, 7, 15, 19, 23, 27, 35, 39, 47, 51, 59, 71, 79, 87, 91, 103, 107, 111, 115, 119, 131, 135, 139, 143, 147, 155, 167, 171, 175, 179, 183, 187, 199, 207, 215, 219, 227, 231, 235, 247, 251, 259, 267, 279, 291, 299, 311, 323, 331, 339, 347, 371, 379, 395, 407, 419, 435, 443, 451, 459, 467, 471, 475, 483, 499]
,
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
,
[19, 29, 43, 53, 61, 67, 75, 109, 117, 157, 163, 203, 213, 227, 245, 267, 275, 277, 285, 291, 299, 309, 317, 323, 331, 349, 355, 363, 373, 395, 405, 413, 427, 429, 451, 469, 477, 493, 499, 501, 2, 10, 12, 34, 38, 42, 44, 50, 58, 66, 76, 90, 108, 124, 130, 154, 172, 186, 194, 198, 202, 204, 242, 252, 258, 262, 266, 274, 282, 284, 290, 294, 298, 306, 316, 338, 346, 348, 354, 362, 364, 370, 378, 390, 394, 396, 402, 412, 418, 426, 434, 442, 444, 460, 466, 474, 476, 482, 490, 492, 0, 7, 15, 31, 32, 55, 59, 79, 87, 96, 103, 107, 119, 123, 127, 128, 135, 151, 159, 160, 167, 175, 183, 191, 192, 219, 224, 235, 247, 251, 271, 272, 279, 283, 295, 303, 315, 320, 335, 343, 347, 351, 359, 363, 367, 383, 384, 399, 411, 415, 416, 431, 439, 443, 455, 463, 471, 480, 487, 491, 495, 507, 511]
,
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
,
[6, 18, 26, 58, 74, 82, 90, 106, 114, 154, 178, 186, 202, 210, 218, 226, 266, 274, 298, 362, 370, 378, 394, 402, 434, 442, 466, 482, 490, 498, 506, 21, 29, 32, 37, 40, 45, 48, 52, 56, 63, 64, 68, 77, 80, 88, 95, 96, 100, 101, 104, 117, 132, 136, 141, 144, 149, 157, 165, 168, 173, 176, 191, 192, 196, 197, 200, 205, 212, 216, 223, 224, 228, 232, 237, 245, 255, 256, 260, 261, 264, 269, 272, 276, 280, 288, 292, 312, 319, 325, 333, 336, 340, 341, 351, 352, 357, 365, 368, 376, 383, 384, 388, 389, 404, 405, 408, 415, 424, 429, 432, 440, 452, 456, 464, 469, 480, 484, 485, 488, 496, 501, 511]
,
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
,
[10, 26, 42, 50, 58, 74, 90, 98, 114, 146, 154, 170, 178, 186, 202, 226, 234, 242, 250, 266, 274, 306, 338, 346, 362, 378, 394, 426, 442, 474, 482, 0, 4, 5, 8, 16, 20, 24, 29, 31, 36, 37, 40, 45, 68, 69, 72, 80, 84, 85, 95, 96, 104, 109, 112, 117, 128, 132, 141, 144, 148, 152, 159, 165, 173, 184, 197, 205, 208, 228, 237, 245, 255, 261, 277, 288, 292, 296, 304, 309, 319, 320, 324, 328, 333, 336, 340, 341, 344, 352, 356, 357, 360, 365, 368, 383, 384, 389, 397, 405, 408, 415, 416, 420, 421, 424, 432, 437, 447, 448, 452, 453, 456, 464, 472, 479, 480, 485, 488, 501, 504, 505]
,
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
,
[19, 27, 93, 99, 117, 125, 147, 155, 171, 189, 203, 221, 227, 235, 245, 253, 269, 317, 333, 339, 355, 363, 397, 403, 427, 437, 459, 467, 477, 501, 2, 6, 12, 26, 28, 50, 58, 74, 82, 90, 98, 108, 114, 122, 124, 138, 140, 156, 162, 178, 188, 194, 204, 210, 218, 220, 234, 242, 250, 252, 258, 262, 266, 268, 274, 282, 290, 298, 300, 322, 326, 330, 332, 346, 348, 354, 364, 386, 390, 396, 402, 410, 412, 428, 434, 442, 450, 458, 460, 466, 482, 492, 498, 508, 0, 15, 16, 31, 32, 47, 55, 63, 64, 79, 91, 103, 119, 123, 127, 143, 151, 155, 159, 160, 167, 171, 175, 183, 187, 191, 192, 207, 219, 224, 231, 235, 247, 255, 256, 263, 279, 295, 315, 327, 335, 343, 352, 359, 363, 367, 379, 383, 384, 407, 411, 415, 416, 427, 431, 439, 455, 471, 475, 491, 495, 503, 507]
,
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
,
[2, 4, 11, 12, 24, 28, 36, 40, 45, 60, 67, 72, 84, 88, 92, 100, 104, 109, 115, 116, 120, 124, 130, 131, 136, 140, 149, 152, 164, 173, 180, 188, 195, 196, 204, 212, 220, 229, 232, 243, 244, 248, 252, 258, 259, 260, 267, 277, 280, 284, 292, 296, 301, 308, 316, 324, 328, 332, 340, 344, 348, 356, 360, 365, 371, 380, 387, 388, 392, 396, 405, 412, 424, 436, 440, 451, 452, 456, 460, 484, 485, 492, 500, 504, 3, 7, 11, 23, 39, 43, 59, 75, 87, 91, 99, 107, 115, 119, 123, 131, 143, 147, 163, 179, 183, 187, 195, 199, 203, 215, 219, 235, 243, 259, 263, 267, 271, 275, 279, 291, 299, 311, 315, 323, 327, 331, 343, 371, 375, 379, 387, 391, 395, 399, 403, 407, 411, 427, 435, 451, 471, 475, 483, 487, 503, 507]
,
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
,
[15, 39, 48, 63, 103, 143, 144, 167, 191, 207, 208, 255, 304, 327, 335, 336, 383, 399, 400, 423, 447, 455, 2, 3, 4, 8, 11, 13, 19, 20, 21, 24, 28, 29, 35, 36, 40, 56, 60, 61, 68, 69, 75, 76, 83, 84, 85, 93, 99, 101, 104, 109, 112, 116, 120, 124, 125, 133, 136, 141, 147, 148, 157, 163, 168, 172, 176, 181, 188, 196, 203, 212, 213, 216, 220, 221, 228, 236, 243, 245, 248, 252, 253, 259, 261, 267, 268, 276, 277, 280, 284, 292, 293, 299, 300, 301, 307, 309, 312, 317, 323, 325, 333, 339, 340, 344, 356, 360, 364, 365, 368, 372, 373, 380, 387, 392, 395, 396, 397, 405, 413, 419, 420, 421, 428, 429, 435, 440, 445, 451, 453, 459, 461, 467, 469, 485, 488, 493, 496, 500]
,
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
,
[6, 7, 10, 31, 50, 55, 58, 59, 74, 82, 90, 96, 106, 111, 114, 122, 123, 135, 146, 151, 159, 160, 175, 178, 187, 192, 202, 210, 215, 218, 224, 234, 239, 242, 250, 251, 266, 282, 287, 288, 306, 311, 314, 320, 338, 343, 370, 379, 394, 407, 410, 416, 431, 439, 442, 443, 458, 471, 474, 495, 506, 0, 13, 16, 20, 21, 24, 31, 37, 40, 45, 48, 53, 72, 77, 80, 88, 93, 95, 96, 100, 101, 104, 109, 128, 132, 136, 141, 149, 152, 157, 159, 164, 165, 168, 173, 176, 181, 191, 196, 197, 216, 224, 228, 237, 248, 255, 256, 260, 276, 285, 288, 292, 301, 319, 325, 328, 333, 336, 341, 349, 352, 357, 360, 368, 376, 383, 384, 388, 392, 397, 400, 405, 408, 415, 416, 420, 421, 432, 447, 452, 453, 456, 461, 464, 469, 485, 488, 493, 495, 504]
,
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
,
[3, 7, 11, 19, 23, 27, 28, 34, 38, 39, 43, 51, 55, 58, 67, 70, 74, 87, 90, 92, 98, 99, 103, 106, 114, 119, 122, 123, 125, 134, 139, 146, 151, 155, 162, 163, 170, 172, 178, 179, 183, 186, 187, 188, 194, 202, 204, 211, 215, 218, 219, 220, 227, 231, 234, 235, 242, 251, 252, 258, 259, 262, 263, 266, 267, 274, 275, 284, 291, 295, 298, 299, 300, 306, 307, 311, 314, 315, 316, 322, 326, 331, 332, 339, 346, 347, 354, 364, 370, 371, 375, 378, 379, 381, 386, 391, 394, 395, 402, 407, 411, 412, 423, 427, 428, 451, 458, 460, 474, 475, 476, 482, 487, 491, 498, 506, 508]
,
    
    
]

bias_array = S.get_tes_bias_bipolar_array()
print(bias_array)
step_size = 0.1 
dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(bias_array) 
    time.sleep(2) 
    S.set_tes_bias_bipolar_array(bias_array - step_size) 
    time.sleep(2) 
S.stream_data_off() 

S.set_tes_bias_bipolar_array(bias_array) 

data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

og_R_tes_array = []
new_bias_array= []
og_R_si_array = []
for bl_num in [0,1,2,3,4,5,6,7,8,9,10,11]:
    responsivity_array,R0_array,bias_power_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,bias_array[bl_num])
    og_R_tes_array.append(np.nanmedian(R0_array))
    og_R_si_array.append(np.nanmedian(responsivity_array))
    
print('BL Median Rtes at:{} mOhm'.format(og_R_tes_array))
print('BL Median Responsivity at:{} [uV-1]'.format(og_R_si_array))







