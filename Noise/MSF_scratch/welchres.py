# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 13:16:08 2017
Combines multiple welch functions to get a power spectrum of varying frequency resolutions.
@author: bjd
"""
import numpy as np
import scipy.signal

def welchres(xs,fs,frange=None,fres=None):
    if frange == None:
        frange = (1,10,500,fs/2)
    if fres == None:
        fres=(0.1,1,10,100)
        
    dim = len(xs[0,:])

    for u in range(dim):
        x = xs[:,u]
        fmin = 0
        overlapfactor = 0.1
        numpts = len(x)
        df = fs/float(numpts)

        for m in range(len(frange)):
            print(frange[m],fres[m])
            fmax = frange[m]
            numptsgroup = np.floor(numpts * df / fres[m])
            numptsoverlap = np.floor(numptsgroup * overlapfactor)
            print(numptsoverlap,numptsgroup,int(numptsoverlap)/int(numptsgroup))
            w,Pxx = scipy.signal.welch(x,fs,window='hann',nperseg=int(numptsgroup),noverlap=int(numptsoverlap))
            lists = np.arange(1+int(np.floor(fmin/fres[m])),np.floor(fmax/fres[m])+1, dtype='int')
            #print lists
            if m==0:
                wall = w[lists]
                Pxxall = Pxx[lists]
            else:
                wall = np.concatenate((wall,w[lists]))
                Pxxall = np.concatenate((Pxxall,Pxx[lists]))
            fmin = fmax
            
        if u==0:
            Pxxalls=np.empty((len(Pxxall),dim))
            walls=np.empty((len(wall),dim))
        Pxxalls[:,u] = Pxxall;
        walls[:,u] = wall;
    return walls,Pxxalls
