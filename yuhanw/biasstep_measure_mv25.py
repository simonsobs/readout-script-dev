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
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[2, 6, 10, 11, 27, 28, 31, 34, 35, 38, 39, 42, 43, 50, 51, 55, 60, 71, 74, 87, 90, 91, 92, 98, 106, 111, 119, 122, 123, 134, 143, 146, 151, 155, 156, 159, 162, 166, 167, 170, 175, 179, 183, 186, 187, 188, 194, 198, 207, 210, 211, 218, 219, 220, 223, 231, 235, 239, 252, 258, 262, 263, 266, 267, 274, 282, 283, 287, 295, 298, 299, 306, 307, 322, 326, 330, 331, 335, 338, 339, 343, 347, 354, 358, 359, 362, 364, 371, 375, 378, 379, 390, 391, 395, 399, 402, 412, 418, 422, 423, 427, 428, 431, 435, 443, 458, 463, 471, 476, 479, 482, 486, 487, 490, 492, 499, 506, 507],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 6, 16, 31, 32, 38, 42, 47, 64, 87, 95, 112, 119, 134, 151, 154, 160, 166, 170, 192, 198, 202, 215, 223, 224, 231, 239, 240, 247, 256, 264, 282, 287, 303, 311, 326, 328, 330, 346, 352, 358, 362, 367, 378, 384, 390, 392, 407, 415, 424, 426, 431, 432, 439, 442, 456, 474, 479, 490, 496, 8, 12, 13, 16, 20, 29, 35, 36, 44, 45, 48, 61, 64, 68, 84, 88, 89, 99, 104, 109, 116, 117, 120, 125, 131, 136, 140, 144, 153, 157, 163, 168, 173, 176, 184, 189, 192, 196, 208, 212, 216, 224, 227, 237, 240, 244, 245, 256, 260, 276, 280, 288, 292, 296, 308, 309, 312, 323, 328, 332, 333, 336, 345, 352, 356, 360, 368, 376, 381, 384, 388, 392, 396, 397, 400, 404, 408, 416, 420, 432, 436, 437, 448, 451, 456, 460, 461, 468, 469, 472, 473, 477, 488, 504, 509],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[15, 39, 48, 63, 79, 80, 191, 199, 271, 295, 336, 383, 399, 423, 447, 464, 2, 3, 4, 10, 11, 12, 13, 18, 19, 20, 28, 43, 44, 52, 53, 60, 61, 77, 82, 83, 92, 98, 100, 108, 115, 116, 120, 125, 130, 131, 140, 163, 164, 171, 172, 181, 184, 189, 196, 203, 204, 205, 210, 212, 221, 226, 228, 235, 236, 242, 245, 252, 260, 268, 275, 276, 284, 285, 290, 292, 306, 315, 316, 317, 324, 332, 340, 348, 349, 355, 363, 364, 370, 372, 373, 376, 379, 380, 386, 388, 396, 402, 403, 412, 413, 418, 419, 427, 434, 435, 436, 437, 443, 444, 452, 461, 466, 467, 469, 475, 476, 482, 483, 484, 491, 493, 499, 507, 509],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[12, 18, 20, 34, 36, 44, 45, 50, 52, 76, 82, 84, 98, 100, 108, 116, 120, 140, 146, 148, 164, 172, 179, 180, 184, 194, 216, 226, 228, 236, 244, 248, 252, 258, 259, 260, 276, 284, 290, 301, 306, 308, 312, 315, 316, 322, 323, 324, 332, 340, 347, 348, 371, 380, 386, 387, 388, 408, 412, 418, 420, 428, 435, 444, 452, 460, 472, 475, 476, 492, 504, 7, 11, 15, 27, 39, 43, 59, 71, 75, 79, 83, 91, 111, 115, 119, 127, 135, 167, 171, 175, 179, 183, 187, 191, 199, 203, 207, 211, 219, 231, 239, 247, 267, 279, 283, 287, 303, 307, 311, 343, 351, 359, 363, 379, 395, 399, 403, 407, 411, 415, 423, 427, 443, 455, 463, 467, 471, 495, 507],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[29, 53, 61, 75, 83, 99, 107, 125, 141, 147, 155, 157, 163, 171, 189, 203, 213, 221, 245, 253, 269, 275, 283, 291, 299, 309, 333, 339, 341, 349, 355, 363, 373, 395, 397, 413, 437, 451, 459, 467, 483, 491, 493, 18, 28, 34, 38, 42, 50, 58, 60, 70, 74, 90, 98, 106, 114, 122, 124, 130, 134, 146, 154, 156, 178, 186, 198, 202, 210, 218, 220, 230, 242, 262, 266, 282, 294, 298, 316, 322, 326, 330, 348, 354, 362, 380, 386, 394, 402, 412, 418, 422, 426, 450, 490, 498, 506, 0, 8, 15, 24, 31, 32, 40, 47, 63, 64, 80, 87, 95, 103, 104, 112, 119, 127, 136, 143, 159, 167, 168, 175, 200, 215, 224, 231, 232, 247, 255, 256, 272, 279, 281, 287, 288, 296, 304, 311, 335, 343, 345, 351, 352, 368, 383, 409, 416, 431, 432, 439, 447, 448, 463, 471, 473, 479, 480, 487],
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[26, 38, 58, 106, 154, 166, 218, 230, 234, 250, 314, 326, 346, 358, 378, 410, 422, 426, 442, 454, 490, 3, 4, 8, 12, 16, 20, 32, 36, 52, 56, 57, 64, 67, 68, 80, 84, 88, 93, 99, 109, 112, 120, 121, 125, 128, 131, 132, 140, 141, 148, 153, 157, 160, 163, 164, 173, 180, 185, 189, 205, 208, 224, 240, 244, 245, 259, 264, 269, 272, 281, 285, 291, 296, 304, 312, 317, 320, 323, 324, 328, 332, 340, 344, 345, 352, 360, 365, 368, 372, 376, 381, 384, 392, 396, 397, 400, 404, 409, 416, 432, 436, 440, 448, 451, 452, 461, 468, 472, 473, 483, 484, 488, 493, 501, 504],
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[6, 26, 38, 42, 58, 90, 102, 106, 154, 170, 186, 198, 250, 294, 326, 346, 362, 378, 390, 442, 474, 486, 0, 8, 12, 13, 20, 32, 35, 40, 48, 52, 56, 57, 61, 68, 72, 77, 84, 88, 96, 99, 109, 112, 117, 120, 128, 131, 132, 136, 140, 148, 153, 157, 163, 168, 176, 180, 184, 185, 189, 196, 208, 224, 240, 245, 256, 259, 260, 269, 272, 280, 285, 288, 304, 312, 320, 323, 328, 333, 340, 344, 345, 355, 356, 365, 372, 373, 376, 383, 404, 416, 424, 436, 445, 448, 451, 452, 456, 468, 472, 473, 477, 480, 484, 500, 504, 509],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[3, 27, 45, 51, 77, 109, 131, 141, 155, 173, 323, 333, 339, 347, 371, 381, 387, 403, 411, 445, 461, 467, 491, 509, 18, 34, 42, 58, 60, 66, 82, 90, 92, 114, 122, 130, 134, 138, 154, 166, 170, 178, 186, 188, 198, 202, 220, 230, 234, 242, 252, 258, 266, 274, 282, 294, 298, 306, 322, 326, 338, 346, 354, 358, 362, 380, 390, 402, 412, 422, 426, 434, 444, 450, 454, 466, 474, 476, 490, 498, 508, 16, 31, 55, 63, 64, 72, 87, 104, 111, 119, 128, 144, 159, 160, 168, 175, 191, 192, 200, 215, 223, 224, 231, 232, 247, 264, 279, 295, 303, 311, 345, 352, 359, 367, 392, 399, 407, 416, 424, 432, 439, 447, 473, 479, 495, 496],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[20, 29, 34, 35, 44, 52, 60, 66, 68, 75, 82, 84, 88, 92, 93, 100, 116, 120, 130, 132, 140, 146, 156, 157, 163, 180, 181, 184, 188, 213, 216, 228, 236, 237, 244, 248, 258, 260, 268, 276, 284, 285, 291, 299, 308, 309, 312, 322, 331, 332, 338, 340, 341, 348, 349, 355, 376, 395, 404, 408, 418, 419, 428, 450, 459, 466, 469, 476, 483, 492, 500, 504, 11, 43, 55, 75, 83, 87, 91, 95, 135, 139, 151, 167, 175, 179, 183, 187, 207, 211, 223, 239, 251, 263, 267, 283, 287, 295, 307, 315, 331, 343, 359, 375, 379, 411, 415, 423, 435, 459, 463, 467, 471, 475, 491, 495, 499],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[31, 32, 55, 95, 119, 123, 159, 160, 183, 239, 263, 343, 352, 367, 379, 416, 448, 471, 480, 507, 2, 3, 12, 18, 28, 34, 35, 43, 44, 50, 52, 56, 59, 60, 68, 77, 82, 84, 92, 93, 99, 100, 109, 115, 117, 120, 130, 131, 140, 146, 148, 156, 157, 164, 171, 173, 179, 180, 181, 187, 194, 195, 210, 220, 226, 227, 242, 245, 260, 269, 274, 275, 290, 292, 301, 306, 307, 309, 316, 317, 323, 324, 331, 332, 333, 339, 340, 347, 349, 354, 356, 364, 370, 371, 376, 380, 387, 388, 396, 397, 411, 418, 419, 428, 434, 443, 444, 452, 459, 460, 461, 467, 468, 475, 483, 484, 492, 501, 504, 508],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[0, 15, 42, 63, 90, 102, 103, 106, 128, 134, 143, 154, 167, 170, 186, 191, 198, 199, 207, 218, 219, 230, 234, 235, 255, 262, 282, 283, 294, 295, 314, 319, 326, 327, 359, 363, 375, 390, 399, 422, 426, 442, 454, 455, 458, 474, 486, 503, 3, 4, 8, 12, 16, 24, 32, 35, 36, 40, 44, 52, 64, 72, 76, 84, 96, 104, 109, 116, 132, 136, 141, 144, 152, 153, 163, 164, 168, 173, 181, 184, 189, 195, 196, 200, 204, 205, 221, 224, 237, 240, 245, 248, 253, 259, 264, 269, 288, 292, 301, 308, 312, 317, 320, 323, 324, 328, 333, 336, 340, 345, 360, 372, 373, 381, 387, 388, 396, 400, 404, 408, 409, 416, 419, 436, 440, 445, 448, 461, 468, 472, 477, 480, 488, 493, 500, 501, 504, 509],
[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[2, 6, 10, 11, 18, 27, 31, 38, 39, 42, 47, 51, 55, 58, 66, 74, 79, 83, 87, 91, 92, 95, 98, 102, 103, 108, 123, 124, 130, 135, 138, 146, 151, 162, 166, 170, 172, 175, 179, 183, 188, 198, 203, 207, 219, 223, 230, 236, 242, 243, 250, 251, 252, 263, 266, 267, 271, 282, 284, 287, 290, 295, 299, 300, 314, 315, 316, 322, 326, 327, 330, 338, 339, 343, 348, 359, 362, 363, 364, 367, 370, 371, 390, 394, 395, 399, 402, 410, 411, 412, 415, 423, 426, 431, 435, 439, 443, 455, 458, 459, 463, 466, 467, 471, 482, 487, 490, 491, 492, 495, 503, 506],
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







