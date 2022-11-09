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
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[7, 15, 18, 26, 43, 47, 51, 55, 59, 66, 70, 71, 74, 75, 82, 87, 91, 98, 106, 108, 111, 114, 122, 124, 138, 139, 143, 146, 147, 156, 162, 167, 175, 187, 188, 198, 199, 202, 203, 204, 211, 215, 219, 220, 234, 235, 236, 239, 243, 251, 263, 266, 267, 279, 283, 284, 287, 294, 298, 303, 307, 311, 314, 331, 335, 339, 348, 351, 359, 363, 371, 379, 403, 407, 411, 412, 415, 418, 427, 428, 434, 435, 443, 450, 454, 476, 482, 487, 492, 495, 498, 499, 503, 508],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[26, 31, 42, 55, 58, 64, 70, 72, 74, 90, 103, 111, 112, 122, 128, 136, 138, 151, 160, 166, 170, 176, 183, 215, 218, 230, 231, 239, 240, 250, 256, 262, 264, 279, 282, 287, 294, 303, 314, 320, 326, 328, 330, 351, 352, 358, 367, 383, 410, 415, 416, 422, 426, 431, 448, 454, 456, 458, 490, 496, 506, 511, 0, 3, 8, 12, 13, 20, 24, 32, 35, 40, 45, 48, 52, 53, 64, 67, 68, 72, 76, 77, 80, 88, 93, 96, 120, 125, 131, 136, 141, 144, 153, 157, 168, 176, 181, 189, 192, 196, 200, 205, 208, 212, 216, 221, 224, 228, 232, 240, 244, 245, 248, 255, 256, 260, 268, 272, 276, 280, 285, 288, 292, 301, 308, 312, 317, 323, 328, 340, 356, 360, 365, 368, 372, 373, 381, 384, 388, 396, 397, 404, 408, 416, 424, 429, 432, 436, 437, 440, 451, 452, 469, 472, 477, 480, 504, 509],
[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[159, 223, 224, 247, 272, 295, 479, 503, 2, 3, 4, 12, 20, 27, 28, 29, 35, 36, 43, 44, 45, 51, 53, 56, 59, 67, 76, 82, 83, 84, 91, 92, 99, 100, 108, 109, 114, 117, 120, 132, 139, 141, 148, 155, 156, 162, 163, 164, 171, 172, 173, 179, 181, 187, 204, 210, 211, 212, 216, 219, 236, 237, 244, 248, 252, 253, 259, 268, 269, 274, 275, 285, 290, 306, 308, 316, 317, 322, 323, 324, 331, 333, 341, 344, 347, 349, 354, 356, 363, 371, 372, 381, 387, 388, 395, 396, 397, 402, 403, 404, 405, 418, 419, 427, 428, 434, 435, 436, 444, 445, 450, 452, 459, 466, 468, 469, 477, 482, 483, 492, 501],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[3, 12, 18, 28, 34, 44, 45, 52, 56, 60, 68, 84, 88, 91, 98, 100, 108, 115, 116, 120, 130, 131, 132, 146, 148, 155, 156, 162, 172, 173, 184, 194, 212, 219, 220, 228, 236, 248, 260, 276, 280, 290, 307, 308, 316, 322, 324, 332, 333, 338, 344, 348, 354, 372, 376, 386, 387, 402, 412, 420, 428, 435, 436, 452, 460, 461, 467, 468, 472, 475, 482, 484, 492, 500, 504, 508, 509, 11, 15, 23, 27, 31, 39, 71, 79, 83, 87, 115, 123, 139, 159, 167, 171, 175, 187, 203, 207, 211, 215, 223, 231, 235, 239, 251, 263, 271, 275, 279, 283, 303, 307, 311, 315, 327, 331, 347, 351, 359, 363, 367, 371, 395, 399, 403, 407, 411, 415, 423, 427, 431, 435, 439, 455, 463, 467, 471, 475, 483, 507],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[27, 35, 43, 61, 83, 85, 93, 107, 109, 117, 125, 139, 141, 157, 163, 181, 189, 203, 213, 227, 235, 237, 245, 267, 269, 275, 285, 299, 309, 317, 323, 331, 349, 355, 395, 403, 405, 419, 427, 451, 469, 483, 501, 2, 34, 38, 58, 70, 74, 82, 92, 98, 114, 130, 134, 138, 156, 162, 178, 226, 234, 236, 250, 252, 262, 266, 274, 282, 284, 306, 316, 322, 326, 330, 338, 346, 348, 358, 370, 378, 402, 410, 418, 422, 426, 444, 454, 466, 476, 482, 486, 490, 492, 498, 506, 8, 15, 40, 47, 55, 80, 95, 96, 103, 111, 112, 128, 136, 151, 159, 160, 183, 215, 223, 224, 239, 271, 281, 287, 288, 295, 296, 303, 304, 319, 320, 327, 328, 335, 336, 343, 352, 359, 367, 368, 375, 384, 392, 399, 409, 424, 432, 439, 447, 448, 456, 463, 464, 487, 495, 496],
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[26, 38, 42, 58, 122, 154, 166, 170, 198, 218, 230, 234, 294, 314, 326, 346, 358, 362, 390, 410, 426, 442, 458, 486, 0, 3, 12, 16, 29, 32, 35, 45, 48, 56, 57, 61, 64, 68, 84, 88, 96, 100, 104, 109, 112, 116, 117, 125, 131, 132, 152, 153, 160, 168, 180, 181, 184, 192, 200, 208, 212, 217, 237, 245, 248, 256, 260, 264, 268, 269, 272, 280, 288, 291, 304, 308, 317, 323, 324, 328, 344, 345, 349, 352, 356, 365, 368, 372, 373, 376, 381, 384, 397, 400, 404, 409, 419, 432, 440, 445, 451, 456, 461, 468, 472, 477, 484, 496, 500, 504, 509],
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[26, 38, 90, 134, 166, 170, 198, 218, 294, 314, 358, 362, 378, 390, 410, 426, 442, 458, 474, 486, 3, 8, 16, 40, 56, 64, 79, 80, 89, 93, 117, 125, 127, 128, 131, 143, 168, 191, 192, 205, 207, 217, 221, 224, 255, 256, 259, 269, 280, 287, 291, 317, 319, 323, 335, 351, 352, 355, 367, 368, 376, 397, 400, 409, 415, 424, 437, 439, 440, 445, 448, 451, 464, 473, 479, 480, 488, 496, 501],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[3, 13, 19, 83, 91, 107, 141, 155, 189, 205, 211, 227, 235, 245, 253, 275, 283, 299, 307, 317, 333, 381, 397, 403, 427, 445, 467, 477, 491, 501, 10, 28, 34, 38, 50, 60, 74, 92, 98, 102, 106, 108, 122, 124, 130, 156, 162, 198, 202, 226, 236, 250, 262, 266, 274, 290, 294, 298, 306, 316, 338, 348, 362, 378, 390, 394, 402, 422, 434, 442, 444, 450, 458, 466, 476, 482, 490, 492, 498, 0, 39, 40, 47, 55, 63, 71, 72, 79, 80, 95, 103, 104, 127, 143, 144, 159, 160, 176, 200, 208, 223, 231, 240, 255, 256, 264, 279, 287, 295, 296, 304, 319, 335, 351, 367, 375, 384, 392, 407, 416, 423, 431, 439, 447, 479, 495, 496, 503],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6],
[2, 4, 18, 20, 21, 28, 36, 44, 52, 56, 67, 68, 75, 76, 84, 85, 88, 98, 100, 109, 120, 124, 131, 132, 149, 152, 156, 157, 162, 164, 173, 180, 195, 212, 216, 220, 226, 243, 248, 267, 268, 274, 276, 280, 291, 308, 322, 341, 348, 354, 356, 364, 365, 380, 386, 395, 396, 412, 440, 451, 460, 488, 492, 493, 499, 508, 27, 39, 43, 47, 55, 59, 75, 79, 83, 91, 95, 103, 107, 111, 115, 139, 143, 147, 151, 159, 179, 199, 207, 211, 215, 219, 227, 231, 235, 239, 243, 247, 251, 267, 271, 275, 283, 287, 295, 299, 303, 307, 315, 335, 339, 347, 391, 407, 411, 423, 427, 431, 443, 483, 487, 495, 499, 503, 16],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[8, 111, 160, 192, 200, 215, 239, 240, 264, 279, 287, 320, 407, 448, 495, 4, 12, 13, 27, 29, 34, 36, 50, 60, 66, 68, 75, 76, 77, 82, 83, 92, 93, 99, 115, 116, 131, 139, 141, 147, 149, 155, 156, 171, 187, 204, 211, 221, 228, 232, 244, 258, 275, 280, 307, 316, 317, 322, 354, 363, 364, 376, 381, 387, 405, 418, 420, 428, 436, 445, 452, 460, 461, 468, 469, 482, 491, 492, 499, 501, 504, 509],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[6, 15, 16, 26, 38, 39, 42, 48, 70, 71, 74, 80, 95, 102, 119, 122, 138, 143, 144, 167, 176, 186, 191, 198, 202, 223, 230, 231, 234, 247, 250, 255, 262, 266, 272, 282, 319, 326, 327, 335, 336, 362, 378, 383, 391, 394, 399, 410, 422, 423, 426, 447, 455, 458, 464, 479, 503, 0, 3, 4, 12, 20, 32, 40, 45, 52, 53, 56, 61, 64, 68, 77, 84, 85, 88, 96, 112, 125, 132, 136, 144, 148, 157, 168, 176, 180, 181, 184, 200, 205, 213, 216, 221, 224, 228, 240, 244, 245, 253, 256, 259, 269, 272, 280, 285, 288, 304, 309, 312, 317, 320, 323, 328, 340, 341, 344, 349, 356, 360, 368, 372, 383, 384, 387, 388, 392, 404, 408, 413, 416, 420, 424, 436, 437, 440, 445, 448, 461, 464, 468, 480, 484, 493, 496, 501, 504, 511],
[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[18, 26, 27, 51, 59, 61, 67, 74, 82, 83, 92, 98, 107, 109, 115, 122, 123, 130, 134, 135, 139, 147, 156, 166, 167, 170, 171, 172, 188, 189, 194, 198, 211, 219, 234, 235, 243, 275, 294, 295, 300, 322, 326, 327, 331, 338, 348, 349, 359, 378, 379, 381, 386, 390, 395, 402, 418, 419, 426, 444, 450, 455, 458, 459, 475, 476, 482, 483, 493, 498, 506, 509],

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







