# -*- coding: utf-8 -*-
'''
Created on Thu Feb 21 10:10:49 2019

@author: Heather McCarrick

peak finder

given an S21 sweep across multiple resonators, this
finds the peaks, which offers a starting point 
for finding f0s; you should then cut on Q etc. using the fitting
function after. 
'''

import numpy as np

def find_nearest(array, value):
    #stolen from stack exchange
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
'''
the function
'''
def get_peaks(freqsarr, s21arr, f_start=None, f_stop=None, f_delta = 1e5, res_num = 64):
    #takes a freq and s21 mag array
    #don't forget to change res_num to number of resonators you want (plus a few probably, then cut later)
    
    if f_start==None:
        #first just get the data for the frequency range we want
        s21loop = s21arr
        freqsloop = freqsarr
    
    else:
        arg_f_start = find_nearest(s21arr, f_start)
        arg_f_stop = find_nearest(freqsarr, f_stop)
    
        s21loop = s21arr[arg_f_start:arg_f_stop]
        freqsloop = freqsarr[arg_f_start:arg_f_stop]
        
    resonance_freq = []
    resonance_s21 = []

    mask = (freqsloop > min(freqsloop)) & (freqsloop < max(freqsloop))

    for i in np.arange(res_num):
		# find new minimum and correpponding frequencies
        argmin_s21 = s21loop[mask].argmin()

        s21_min = s21loop[mask][argmin_s21]
        f_min = freqsloop[mask][argmin_s21]

		# add to our list of resonance frequencies
        resonance_freq.append(f_min)
        resonance_s21.append(s21_min)

		# find some MHz around this minimum
        f_high_index = find_nearest(freqsloop, f_min + f_delta)
        f_low_index = find_nearest(freqsloop, f_min - f_delta)

		# update mask
        mask[f_low_index:f_high_index] = 0

    resonance_freq = np.array(resonance_freq)
    resonance_s21 = np.array(resonance_s21)
	
    return resonance_freq, resonance_s21

