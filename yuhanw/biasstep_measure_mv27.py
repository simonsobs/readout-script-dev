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
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[12, 15, 23, 27, 28, 35, 39, 43, 58, 59, 60, 66, 71, 74, 75, 82, 83, 87, 91, 99, 123, 138, 139, 140, 147, 151, 179, 187, 188, 194, 195, 203, 204, 220, 231, 236, 243, 244, 247, 250, 258, 259, 263, 266, 267, 271, 275, 279, 290, 315, 323, 327, 332, 338, 339, 343, 355, 362, 363, 371, 372, 375, 379, 380, 387, 411, 412, 418, 419, 426, 434, 435, 439, 444, 450, 455, 459, 466, 467, 471, 483, 490, 491, 492, 499, 500, 507],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[0, 6, 8, 10, 23, 26, 38, 39, 48, 58, 63, 70, 74, 79, 80, 90, 111, 114, 122, 127, 134, 136, 138, 143, 154, 166, 186, 192, 198, 199, 207, 208, 218, 223, 234, 240, 247, 255, 272, 279, 295, 298, 303, 304, 306, 319, 320, 330, 343, 359, 362, 368, 370, 384, 399, 400, 407, 410, 423, 426, 431, 432, 447, 458, 474, 498, 4, 16, 21, 29, 32, 36, 40, 45, 52, 63, 68, 72, 80, 84, 85, 95, 96, 100, 104, 109, 132, 136, 144, 157, 160, 168, 173, 176, 180, 191, 196, 200, 205, 212, 213, 224, 228, 229, 232, 237, 240, 253, 255, 256, 269, 272, 276, 277, 280, 292, 301, 304, 309, 312, 317, 320, 324, 328, 336, 344, 349, 351, 365, 368, 373, 376, 381, 384, 397, 404, 405, 408, 413, 415, 420, 421, 424, 437, 440, 445, 447, 448, 456, 464, 469, 472, 477, 479, 488, 501, 504, 511],
[0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[31, 55, 215, 239, 288, 439, 8, 19, 20, 36, 37, 43, 53, 61, 66, 67, 72, 76, 83, 88, 92, 93, 115, 116, 130, 139, 141, 148, 156, 157, 163, 165, 180, 189, 194, 196, 197, 203, 204, 205, 211, 212, 213, 236, 243, 245, 248, 253, 258, 261, 264, 268, 269, 275, 276, 280, 284, 296, 300, 307, 308, 309, 340, 348, 349, 356, 357, 360, 365, 372, 380, 381, 388, 389, 392, 396, 397, 403, 405, 408, 413, 419, 424, 444, 445, 451, 452, 453, 460, 467, 468, 472, 476, 477, 484, 492, 493, 501, 504, 508],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[3, 12, 20, 27, 28, 34, 36, 44, 45, 52, 60, 76, 84, 88, 91, 92, 98, 116, 124, 131, 152, 155, 162, 168, 180, 194, 204, 212, 216, 219, 232, 236, 244, 248, 252, 258, 260, 268, 276, 292, 300, 307, 308, 312, 316, 322, 324, 333, 339, 356, 360, 364, 372, 376, 381, 386, 388, 396, 397, 404, 408, 420, 428, 435, 440, 444, 461, 467, 472, 476, 508, 509, 7, 15, 19, 27, 35, 39, 47, 55, 59, 67, 71, 75, 79, 83, 91, 103, 107, 111, 115, 119, 123, 135, 139, 143, 147, 151, 155, 163, 167, 175, 179, 199, 207, 215, 219, 231, 235, 239, 243, 247, 267, 275, 299, 311, 315, 331, 343, 355, 379, 407, 419, 427, 439, 443, 451, 459, 467, 475, 499, 507],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[11, 13, 35, 53, 61, 75, 85, 93, 99, 107, 117, 147, 149, 163, 171, 181, 189, 195, 203, 221, 245, 267, 275, 285, 291, 299, 309, 323, 331, 355, 363, 365, 395, 419, 429, 459, 469, 483, 493, 501, 2, 10, 34, 58, 66, 76, 90, 106, 108, 122, 138, 162, 172, 178, 186, 194, 204, 210, 218, 242, 252, 258, 274, 282, 294, 298, 300, 306, 316, 326, 330, 338, 348, 354, 362, 370, 378, 380, 386, 390, 394, 396, 402, 410, 412, 422, 434, 444, 454, 458, 460, 466, 474, 476, 490, 506, 0, 27, 31, 55, 59, 63, 79, 80, 91, 95, 96, 103, 119, 123, 128, 135, 143, 144, 151, 155, 159, 167, 175, 183, 192, 199, 219, 223, 224, 239, 247, 272, 283, 288, 315, 320, 327, 335, 343, 359, 367, 383, 384, 391, 415, 416, 431, 439, 443, 448, 463, 471, 480, 487, 503, 507],
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[6, 10, 26, 50, 74, 90, 106, 138, 146, 154, 210, 250, 306, 330, 346, 370, 378, 402, 434, 442, 458, 474, 498, 8, 16, 20, 21, 29, 32, 36, 37, 40, 45, 53, 56, 63, 64, 68, 69, 77, 80, 84, 95, 101, 104, 112, 127, 136, 149, 168, 176, 180, 181, 191, 196, 197, 208, 212, 216, 223, 224, 232, 237, 269, 272, 276, 280, 285, 288, 292, 296, 301, 304, 312, 324, 333, 341, 344, 349, 357, 360, 368, 376, 384, 388, 400, 404, 405, 408, 413, 416, 420, 421, 440, 447, 461, 464, 480, 485, 496, 501, 511],
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[6, 26, 58, 74, 82, 114, 134, 138, 186, 210, 234, 250, 266, 282, 306, 330, 346, 362, 378, 410, 426, 434, 458, 474, 0, 13, 20, 21, 29, 32, 36, 45, 53, 64, 85, 88, 96, 112, 120, 127, 128, 132, 133, 141, 144, 149, 159, 164, 165, 168, 173, 176, 191, 192, 196, 197, 200, 208, 223, 228, 232, 240, 245, 260, 264, 269, 272, 285, 292, 293, 296, 304, 319, 333, 340, 341, 344, 356, 357, 360, 365, 373, 383, 384, 388, 400, 404, 413, 415, 416, 420, 421, 424, 453, 461, 464, 468, 469, 472, 480, 485, 493],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[61, 83, 93, 99, 107, 141, 147, 181, 211, 227, 245, 253, 275, 283, 317, 333, 363, 373, 381, 397, 411, 427, 437, 445, 459, 467, 491, 2, 6, 10, 42, 44, 50, 70, 74, 76, 92, 98, 114, 124, 130, 134, 140, 154, 156, 186, 202, 210, 218, 226, 236, 250, 252, 258, 274, 282, 284, 290, 298, 306, 322, 326, 332, 338, 346, 348, 364, 370, 378, 394, 402, 410, 418, 426, 434, 442, 454, 460, 482, 492, 0, 16, 23, 31, 39, 47, 55, 64, 71, 87, 123, 135, 143, 155, 159, 167, 183, 192, 215, 223, 224, 231, 239, 247, 251, 255, 256, 263, 271, 272, 279, 288, 303, 315, 335, 343, 347, 351, 352, 359, 375, 383, 384, 399, 400, 415, 416, 423, 431, 439, 443, 447, 448, 463, 475, 491, 503, 507],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[2, 4, 11, 21, 35, 40, 52, 66, 67, 85, 88, 92, 100, 116, 124, 130, 131, 132, 139, 140, 152, 156, 164, 168, 173, 188, 195, 200, 212, 216, 220, 228, 232, 237, 243, 244, 248, 252, 260, 268, 277, 280, 284, 300, 301, 316, 323, 324, 328, 332, 360, 365, 371, 372, 376, 386, 387, 388, 396, 405, 412, 420, 424, 429, 436, 452, 456, 460, 468, 488, 493, 499, 500, 11, 27, 35, 39, 55, 71, 79, 83, 87, 99, 115, 119, 123, 131, 135, 139, 151, 171, 187, 203, 207, 211, 215, 219, 235, 243, 247, 251, 263, 271, 275, 291, 299, 311, 315, 323, 327, 331, 363, 395, 399, 407, 419, 439, 443, 451, 459, 483, 499, 507],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[39, 48, 79, 143, 167, 191, 208, 231, 304, 336, 399, 400, 423, 432, 455, 463, 2, 8, 13, 20, 28, 29, 37, 43, 51, 53, 56, 61, 75, 76, 77, 84, 92, 93, 99, 100, 108, 117, 120, 125, 130, 136, 139, 140, 141, 148, 149, 152, 171, 189, 200, 205, 211, 221, 228, 236, 237, 244, 248, 260, 268, 269, 276, 277, 284, 285, 291, 292, 293, 307, 309, 312, 316, 323, 325, 328, 331, 332, 339, 340, 341, 344, 357, 360, 364, 365, 368, 380, 381, 386, 388, 392, 403, 405, 408, 412, 413, 419, 420, 421, 424, 427, 440, 444, 451, 452, 459, 460, 469, 477, 488, 496, 504, 508, 509],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[6, 7, 10, 26, 31, 32, 42, 55, 95, 96, 111, 123, 134, 138, 160, 178, 187, 192, 202, 210, 215, 234, 239, 242, 250, 251, 262, 263, 282, 287, 298, 306, 311, 330, 338, 343, 346, 352, 367, 375, 378, 391, 394, 407, 410, 415, 416, 442, 443, 466, 471, 480, 498, 507, 4, 21, 29, 36, 45, 56, 64, 68, 72, 77, 93, 95, 100, 101, 109, 112, 117, 127, 128, 141, 144, 148, 149, 165, 168, 176, 181, 192, 200, 205, 208, 213, 221, 223, 224, 228, 229, 232, 260, 272, 276, 277, 280, 285, 287, 292, 296, 301, 304, 309, 324, 328, 333, 336, 344, 351, 352, 357, 360, 368, 383, 384, 388, 404, 413, 415, 416, 420, 421, 429, 447, 453, 464, 469, 472, 477, 480, 488, 496, 504],
[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[7, 11, 18, 28, 34, 42, 44, 51, 58, 60, 67, 70, 71, 74, 75, 83, 90, 98, 103, 106, 107, 114, 119, 122, 123, 139, 151, 155, 162, 163, 172, 178, 183, 187, 188, 198, 199, 202, 203, 215, 218, 219, 220, 226, 231, 234, 235, 236, 242, 251, 252, 258, 262, 263, 266, 271, 274, 275, 279, 283, 291, 294, 298, 306, 307, 311, 314, 315, 316, 323, 331, 332, 343, 346, 347, 348, 355, 364, 371, 375, 379, 387, 390, 394, 403, 407, 411, 412, 419, 426, 428, 434, 435, 442, 451, 454, 471, 474, 475, 487, 498, 503, 506, 508, 509],
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







