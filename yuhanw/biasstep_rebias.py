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


def responsivity(data_time, phase_og, mask, tes_bias,band_list,chan_list,bias_groups,v_bias,out_dict,v_name,r_name,resp_name):
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

                out_dict[(int(b),int(c))][v_name] = v_bias
                out_dict[(int(b),int(c))][r_name] = np.median(R0_chan) * 1000
                out_dict[(int(b),int(c))][resp_name] = np.median(Si)/1e6


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
        
    return responsivity_array,R0_array,bias_power_array






## hard coding biasline mapping for now

target_band_chan = [
[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
[0, 7, 15, 18, 23, 26, 27, 39, 42, 47, 50, 63, 64, 71, 74, 75, 79, 82, 87, 91, 96, 102, 103, 106, 119, 127, 135, 138, 143, 146, 151, 154, 155, 159, 160, 167, 170, 171, 175, 178, 191, 192, 203, 207, 218, 231, 235, 247, 255, 256, 262, 263, 271, 282, 287, 288, 290, 294, 295, 299, 314, 315, 319, 326, 327, 331, 343, 346, 347, 354, 358, 367, 370, 375, 383, 384, 390, 399, 418, 422, 426, 427, 431, 434, 443, 448, 450, 454, 459, 466, 474, 475, 480, 482, 490, 491, 495],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[10, 16, 23, 24, 26, 40, 48, 58, 82, 90, 95, 96, 106, 119, 122, 138, 144, 146, 151, 154, 168, 175, 183, 186, 210, 215, 224, 234, 239, 266, 280, 287, 303, 311, 336, 343, 351, 360, 362, 402, 407, 408, 410, 415, 416, 424, 434, 439, 442, 456, 464, 466, 474, 479, 480, 487, 490, 495, 4, 8, 12, 16, 20, 21, 29, 36, 37, 40, 45, 64, 68, 76, 77, 80, 85, 96, 101, 104, 108, 116, 132, 140, 141, 144, 160, 164, 165, 168, 173, 192, 196, 197, 200, 204, 224, 228, 229, 232, 236, 237, 244, 264, 269, 272, 276, 277, 285, 288, 292, 293, 296, 300, 301, 320, 324, 328, 332, 333, 341, 352, 356, 360, 364, 365, 368, 384, 388, 392, 396, 397, 400, 405, 420, 421, 424, 428, 429, 436, 448, 453, 460, 461, 480, 484, 485, 488, 496, 500, 504],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[0, 64, 103, 128, 136, 192, 207, 264, 271, 295, 335, 368, 399, 455, 2, 3, 5, 19, 21, 29, 42, 45, 51, 53, 60, 61, 66, 69, 77, 82, 93, 98, 99, 114, 115, 117, 121, 130, 131, 132, 140, 149, 162, 165, 179, 180, 181, 188, 196, 205, 210, 221, 226, 227, 237, 245, 249, 253, 260, 267, 268, 269, 276, 284, 285, 290, 292, 293, 301, 306, 307, 316, 323, 330, 333, 338, 339, 341, 357, 364, 365, 372, 380, 381, 387, 394, 397, 403, 404, 405, 421, 426, 428, 434, 436, 444, 451, 452, 453, 458, 466, 468, 469, 484, 485, 499, 500, 504],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
[12, 18, 36, 60, 66, 68, 84, 98, 124, 132, 140, 164, 178, 184, 196, 212, 244, 248, 252, 258, 260, 268, 284, 306, 308, 332, 348, 370, 372, 376, 386, 388, 396, 404, 412, 436, 440, 444, 452, 468, 472, 476, 482, 492, 498, 508, 6, 11, 23, 31, 51, 59, 63, 66, 83, 103, 107, 114, 115, 119, 122, 130, 147, 151, 159, 175, 179, 187, 191, 211, 215, 219, 231, 235, 239, 242, 243, 247, 250, 267, 283, 287, 303, 307, 311, 315, 322, 343, 347, 351, 371, 375, 386, 403, 407, 411, 415, 435, 439, 467, 471, 479, 487, 491, 495, 498, 499, 506, 57, 185, 313, 441],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
[11, 19, 29, 35, 53, 67, 75, 85, 93, 99, 107, 117, 139, 141, 147, 149, 157, 163, 181, 195, 203, 213, 221, 227, 237, 245, 267, 275, 285, 291, 299, 309, 317, 330, 331, 341, 349, 355, 363, 365, 395, 403, 405, 413, 419, 427, 429, 437, 445, 451, 459, 469, 477, 493, 501, 2, 18, 58, 60, 90, 102, 106, 124, 146, 154, 166, 178, 186, 188, 210, 218, 226, 230, 252, 282, 306, 316, 348, 358, 402, 410, 466, 474, 476, 490, 7, 15, 16, 25, 31, 43, 71, 96, 107, 119, 123, 135, 143, 160, 183, 199, 217, 223, 224, 239, 247, 251, 263, 287, 295, 311, 319, 345, 363, 375, 383, 415, 423, 439, 443, 447, 455, 479, 491],
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5],
[6, 58, 70, 90, 166, 186, 198, 218, 230, 250, 262, 346, 358, 378, 410, 442, 454, 474, 2, 3, 8, 19, 24, 25, 28, 29, 35, 40, 51, 57, 60, 61, 66, 68, 80, 83, 84, 89, 92, 93, 99, 109, 115, 120, 121, 124, 130, 140, 147, 148, 152, 157, 173, 176, 179, 180, 185, 188, 195, 200, 208, 211, 212, 221, 237, 248, 253, 260, 261, 268, 276, 284, 285, 291, 296, 301, 304, 312, 317, 323, 324, 328, 332, 340, 344, 345, 348, 355, 364, 365, 368, 371, 372, 377, 380, 388, 389, 396, 408, 412, 419, 429, 432, 436, 440, 444, 445, 460, 464, 468, 472, 473, 476, 483, 488, 492, 496, 500, 504, 505, 508, 486],
[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[26, 38, 70, 134, 154, 198, 218, 314, 346, 358, 378, 410, 422, 442, 454, 474, 486, 24, 25, 28, 40, 44, 45, 48, 51, 52, 56, 57, 60, 61, 67, 72, 76, 80, 84, 88, 89, 112, 120, 125, 132, 148, 152, 153, 156, 157, 163, 172, 173, 179, 184, 188, 189, 195, 200, 208, 212, 217, 220, 236, 240, 249, 253, 258, 260, 261, 264, 268, 272, 275, 285, 301, 340, 344, 345, 348, 349, 355, 356, 360, 364, 372, 376, 380, 387, 389, 392, 396, 400, 403, 412, 419, 429, 432, 436, 441, 464, 467, 472, 473, 480, 483, 484, 488, 493, 496, 500, 504, 505, 508],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[3, 7, 11, 13, 67, 131, 187, 205, 219, 251, 267, 269, 307, 315, 331, 387, 395, 435, 475, 499, 507, 10, 66, 82, 90, 92, 122, 130, 134, 146, 154, 156, 178, 226, 242, 290, 306, 314, 326, 330, 354, 362, 370, 422, 426, 434, 508, 31, 40, 47, 55, 57, 72, 88, 89, 95, 112, 119, 121, 136, 143, 144, 151, 152, 153, 159, 160, 176, 185, 200, 215, 217, 224, 231, 271, 272, 304, 311, 312, 319, 344, 352, 360, 367, 368, 424, 464, 480, 495],
[0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[330, 10, 18, 19, 28, 29, 34, 44, 50, 52, 60, 74, 85, 98, 100, 107, 114, 124, 132, 147, 157, 171, 178, 188, 210, 213, 226, 227, 228, 235, 236, 237, 242, 248, 252, 258, 260, 266, 268, 274, 276, 284, 299, 309, 312, 322, 324, 332, 338, 339, 340, 344, 348, 349, 355, 356, 365, 372, 376, 386, 388, 396, 402, 403, 404, 405, 412, 413, 420, 428, 436, 440, 450, 452, 468, 469, 472, 476, 484, 491, 500, 508, 0, 15, 23, 27, 31, 39, 43, 47, 51, 55, 59, 63, 75, 79, 91, 107, 115, 135, 143, 155, 171, 175, 203, 227, 235, 239, 243, 263, 267, 279, 299, 307, 315, 319, 335, 343, 347, 351, 375, 379, 383, 391, 395, 403, 407, 415, 423, 439, 443, 447, 459, 471, 479, 495, 499, 503, 507],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
[24, 63, 104, 119, 232, 280, 287, 304, 311, 336, 343, 351, 375, 400, 408, 439, 464, 471, 4, 12, 20, 29, 35, 39, 45, 50, 51, 59, 61, 66, 68, 71, 74, 76, 77, 83, 99, 107, 108, 109, 114, 115, 116, 117, 130, 138, 139, 141, 148, 163, 170, 171, 173, 178, 179, 181, 187, 188, 189, 194, 195, 196, 202, 203, 204, 210, 211, 212, 221, 226, 228, 236, 243, 244, 245, 251, 252, 253, 258, 259, 260, 263, 268, 269, 283, 284, 285, 290, 292, 298, 300, 306, 315, 316, 323, 324, 327, 331, 338, 340, 347, 349, 354, 355, 356, 365, 370, 371, 372, 379, 381, 387, 388, 391, 396, 411, 418, 419, 420, 427, 428, 436, 444, 445, 450, 455, 458, 460, 461, 475, 476, 492, 493, 499, 501, 508],
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
[6, 39, 70, 72, 102, 122, 134, 160, 166, 176, 186, 199, 200, 202, 218, 231, 234, 250, 256, 262, 264, 295, 320, 335, 346, 362, 378, 383, 399, 410, 423, 426, 431, 432, 448, 463, 496, 511, 3, 35, 36, 45, 53, 57, 68, 72, 76, 80, 88, 93, 99, 120, 153, 163, 172, 173, 176, 181, 184, 185, 189, 192, 196, 208, 224, 228, 244, 255, 264, 268, 272, 284, 292, 304, 308, 340, 345, 365, 384, 387, 388, 392, 396, 400, 404, 413, 420, 437, 451, 456, 473, 484, 492, 493, 496],
[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
[0, 6, 7, 10, 18, 26, 32, 38, 39, 42, 43, 47, 55, 59, 71, 79, 95, 98, 111, 114, 119, 123, 135, 139, 146, 151, 154, 166, 167, 170, 175, 194, 198, 199, 207, 219, 230, 234, 239, 256, 262, 267, 274, 279, 290, 295, 306, 311, 320, 330, 346, 347, 351, 354, 358, 359, 363, 370, 379, 386, 390, 411, 418, 422, 423, 434, 448, 450, 454, 458, 479, 491, 499, 503, 507],
]

biasstep_dict = {}

og_V_tes_array = S.get_tes_bias_bipolar_array()
print('current Vbias array')
print(og_V_tes_array)
step_size = 0.1
dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(og_V_tes_array) 
    time.sleep(2) 
    og_V_tes_array_step = np.array(og_V_tes_array)[0:6] + step_size
    og_V_tes_array_step = np.append(og_V_tes_array_step,[0,0,0,0,0,0,0,0,0])
    S.set_tes_bias_bipolar_array(og_V_tes_array_step) 
    time.sleep(2) 
S.stream_data_off() 

S.set_tes_bias_bipolar_array(og_V_tes_array) 
data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

og_R_tes_array = []
new_bias_array= []
for bl_num in [0,1,2,3,4,5]:
    og_V_tes_array = S.get_tes_bias_bipolar_array()
    # print('og_V_tes_array')
    # print(og_V_tes_array)
    responsivity_array,R0_array,bias_power_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,og_V_tes_array[bl_num],biasstep_dict,'v1','r1','si1')
    print(len(responsivity_array))
    og_R_tes_array.append(np.nanmedian(R0_array))

    # new_bias_array.append(og_V_tes_array[bl_num]+0.2)


    if np.nanmedian(R0_array)<= 4:
        new_bias_array.append(og_V_tes_array[bl_num]+0.3)
    elif np.nanmedian(R0_array) > 4:
        new_bias_array.append(og_V_tes_array[bl_num]-0.3)
    # else if:
    #     new_bias_array.append(og_V_tes_array[bl_num]+0.1)
print('current Rtes median:')
print(og_R_tes_array)
new_bias_array = np.array(new_bias_array)
print(len(new_bias_array))
new_bias_array = np.append(new_bias_array,[0,0,0,0,0,0,0,0,0])
S.set_tes_bias_bipolar_array(new_bias_array)
print('trial step vbias:')

print(S.get_tes_bias_bipolar_array())

time.sleep(10) 

dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(new_bias_array) 
    time.sleep(2) 
    new_bias_array_step = np.array(new_bias_array)[0:6] + step_size
    new_bias_array_step = np.append(new_bias_array_step,[0,0,0,0,0,0,0,0,0])
    S.set_tes_bias_bipolar_array(new_bias_array_step)  
    time.sleep(2) 
S.stream_data_off() 

data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

new_R_tes_array = []
for bl_num in [0,1,2,3,4,5]:
    responsivity_array,R0_array,bias_power_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,new_bias_array[bl_num],biasstep_dict,'v2','r2','si2')
    print(len(responsivity_array))
    new_R_tes_array.append(np.nanmedian(R0_array))
print('current Rtes median')
print(new_R_tes_array)



v_estimate_array = []
for bl_num in [0,1,2,3,4,5]:
    v_estimate_array_bl = []
    for index_k in range(len(target_band_chan[2*bl_num])):
        pick_b = target_band_chan[2*bl_num][index_k]
        pick_c = target_band_chan[2*bl_num+1][index_k]


    # v_estimate = (og_V_tes_array[bl_num] * (target_Rtes - new_R_tes_array[bl_num]) + new_bias_array[bl_num] * (og_R_tes_array[bl_num] - target_Rtes)) /(og_R_tes_array[bl_num] - new_R_tes_array[bl_num])
        try:
            R2 = biasstep_dict[(int(pick_b),int(pick_c))]['r2']
            R1 = biasstep_dict[(int(pick_b),int(pick_c))]['r1']
            V2 = biasstep_dict[(int(pick_b),int(pick_c))]['v2']
            V1 = biasstep_dict[(int(pick_b),int(pick_c))]['v1']

        # v_estimate = ((R2-R1)/(target_Rtes-R1))*(V2-V1) + V1
            v_estimate = (V1*(target_Rtes-R2) + V2*(R1-target_Rtes))/(R1-R2)
            v_estimate_array_bl.append(v_estimate)
        except:
            continue
    print("number of estimated v within biasline")
    print(len(v_estimate_array_bl))
    v_estimate_array.append(np.nanmedian(v_estimate_array_bl))




#     v_estimate_array.append(v_estimate)

print('new estimation vbias:')
print(v_estimate_array)

print('applying new voltage bias')
v_estimate_array = np.append(v_estimate_array,[0,0,0,0,0,0,0,0,0])
S.set_tes_bias_bipolar_array(v_estimate_array)
time.sleep(10) 

dat_path = S.stream_data_on()
for k in [0,1]: 
    S.set_tes_bias_bipolar_array(v_estimate_array) 
    time.sleep(2) 
    new_bias_array_step = np.array(v_estimate_array)[0:6] + step_size
    new_bias_array_step = np.append(new_bias_array_step,[0,0,0,0,0,0,0,0,0])
    S.set_tes_bias_bipolar_array(new_bias_array_step)  
    time.sleep(2) 
S.stream_data_off() 



data_time = dat_path[-14:-4]

timestamp, phase, mask, tes_bias = S.read_stream_data(dat_path, return_tes_bias=True)

new_R_tes_array = []
new_resp_array = []
for bl_num in [0,1,2,3,4,5]:
    responsivity_array,R0_array,bias_power_array = responsivity(data_time, phase, mask, tes_bias,target_band_chan[2*bl_num],target_band_chan[2*bl_num+1],bl_num,v_estimate_array[bl_num],biasstep_dict,'v3','r3','si3')
    print(len(responsivity_array))
    new_R_tes_array.append(np.nanmedian(R0_array))
    new_resp_array.append(np.nanmedian(responsivity_array))
print('current Rtes median')
print(new_R_tes_array)
print('current resp median')
print(new_resp_array)


print(biasstep_dict[(3,18)])

timestamp=S.get_timestamp()
save_name = '{}_biasstep_rebias.npy'.format(timestamp)
print(f'Saving data to {os.path.join(S.output_dir, save_name)}')
biasstep_rebia_data = os.path.join(S.output_dir, save_name)
path = os.path.join(S.output_dir, biasstep_rebia_data) 
np.save(path, biasstep_dict)


