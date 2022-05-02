# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 08:39:24 2019

@author: Heather
"""
import numpy as np
import scipy

def noise_model_pysmurf(f, wl, fk, alpha):
    #f, frequency
    #wl, white noise level
    #fk, fknee
    #alpha, index
    
    #make sure this matches your config file
    b, a = scipy.signal.butter(4, 63, analog=True, btype='low')

    w,h = scipy.signal.freqs(b,a,worN=f)
    tf = np.absolute(h)**2
        
    return (wl)*(1 + (fk/f)**alpha) *tf

def fit_noise_model(wn_average, f, ts_pA, fs):
    #f, frequency
    #ts_pA, timestream in pA
    
    bounds_low = [0.,0.,0.]
    bounds_high = [np.inf,np.inf,np.inf]
    bounds = (bounds_low,bounds_high)
    
    #get psd
    pxx, fr = mlab.psd(ts_pA, NFFT=2**12, Fs=fs)
    pxx = np.sqrt(pxx)
    
    #get estimate for white noise level
    fr_1 = find_nearest(fr,0.5)
    fr_10 = find_nearest(fr,20)
    wn_average = np.average(pxx[fr_1:fr_10])
    
    p0 = [wn_average, 0.02, 0.02]
    popt, pcov = optimize.curve_fit(noise_model_pysmurf, fr[1:], pxx[1:], p0=p0 , bounds=bounds)
    return popt