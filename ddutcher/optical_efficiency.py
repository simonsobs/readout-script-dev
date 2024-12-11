"""
optical_efficiency.py
Functions to compute optical efficiency, assuming a beam-filling load.
Builds off of data products defined in psatvstemp_analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from glob import glob
from scipy import constants as cnst
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit

filter_dir = "/home/ddutcher/data/filters/"


def filter_func(filters):
    """
    Return combined transmission of the specified LPE filters.
    
    Parameters
    ----------
    `filters` : str or list of str
        If 'lf,'mf' or 'uhf', the standard set of Princeton LPE filters is used.
        If given as a list, the transmission of each filter will be multiplied
        together to yield the total transmission of the filter stack.
    
    Returns
    -------
    filter_func : function
        Function that, when supplied with a list of frequencies in Hz,
        returns the transmission spectrum at those frequencies.
    """
    if isinstance(filters, str):
        if filters.lower() not in ['lf','mf','uhf']:
            raise ValueError(
                "`filters` must either be one of ['lf','mf','uhf'] or " +
                "a list of valid Cardiff LPE filter IDs"
            )
        elif filters.lower() == 'mf':
            filters = ['K2951','K2908']
        elif filters.lower() == 'uhf':
            filters = ['K2938','K2805']
        elif filters.lower() == 'lf':
            filters = ['FARC3226','FP3276']
    elif not isinstance(filters, list):
        raise TypeError("`filters` must either be str or list of str")

    filter_data = []
    for filt_id in filters:
        try:
            data = np.loadtxt(os.path.join(filter_dir, f'{filt_id}.txt'))
        except ValueError:
            data = np.loadtxt(os.path.join(filter_dir, f'{filt_id}.txt'), delimiter=',')
        # Convert frequency column data to be in Hz
        if filt_id in ['FARC3226','FP3276']:
            data[:,0] *= 1e9
        else:
            data[:,0] *= 1e2 * cnst.c
        filter_data += [data]

    filter_func = interp1d((filter_data[0])[:,0], (filter_data[0])[:,1])
    if len(filter_data) == 1:
        return filter_func

    for filt in range(1, len(filter_data)):
        m = filter_data[filt][:,0] < np.max(filter_data[0][:,0])
        filter_func= interp1d(
            (filter_data[filt])[:,0][m],
            (filter_data[filt])[:,1][m] * filter_func((filter_data[filt])[:,0][m])
        )

    return filter_func


def bandpass_funcs(array_freq):
    """
    Parameters
    ----------
    array_freq : {'lf', 'mf', or 'uhf'}
        
    """
    if array_freq.lower() == "lf":
        array_freq = "NIST_LF"
    freq1 = np.loadtxt(os.path.join(filter_dir, f"{array_freq.upper()}_1_peak.txt"))
    freq2 = np.loadtxt(os.path.join(filter_dir, f"{array_freq.upper()}_2_peak.txt"))
    
    freq1_func = interp1d(freq1[:,0]*1e9, freq1[:,1] / np.max(freq1[:,1]))
    freq2_func = interp1d(freq2[:,0]*1e9, freq2[:,1] / np.max(freq2[:,1]))
    
    return freq1_func, freq2_func


def compute_dark_correction(cl_data, used_temps=None, array_freq='mf'):
    if isinstance(cl_data,str):
        cl_data = np.load(cl_data,allow_pickle=True).item()

    if used_temps is None:
        try:
            used_temps = cl_data['metadata']['used_temps']
        except KeyError:
            raise ValueError(
                "Must specify `used_temps` for older version CL datafile.")

    if array_freq.lower() not in ['lf','mf','uhf']:
        raise ValueError("`array_freq` must be one of {'lf','mf','uhf'}")
    if array_freq.lower() == "lf":
        freq1, freq2 = "30", "40"
        bl_freq_map = {bl: freq1 for bl in [10,11]}#[0, 3, 5, 10, 11]}
        bl_freq_map.update({bl: freq2 for bl in [8,9]})#[1, 2, 8, 9]})
    else:
        if array_freq.lower() == "uhf":
            freq1, freq2 = "220", "280"
        else:
            freq1, freq2 = "90", "150"
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})

    avg_dark_deltaPsat = {freq1:[[] for temp in used_temps],
                          freq2:[[] for temp in used_temps]
                         }

    for bg in cl_data['data'].keys():
        if bg in cl_data['metadata']['optical_bl']:
            continue
        freq = bl_freq_map[bg]
        for sb in cl_data['data'][bg].keys():
            for ch, d in cl_data['data'][bg][sb].items():
                if len(d['temp']) < 2:
                    continue
                if d['temp'][0] != used_temps[0]:
                    continue
                delta_psat = d['psat'] - d['psat'][0]
                for ind, t in enumerate(d['temp']):
                    idx = used_temps.index(t)
                    avg_dark_deltaPsat[freq][idx] += [delta_psat[ind]]

    for freq, arr in avg_dark_deltaPsat.items():
        for i, dp_arr in enumerate(arr):
            avg_dark_deltaPsat[freq][i] = np.nanmean(dp_arr)

    return avg_dark_deltaPsat


def compute_opteff(cl_data, used_temps=None, array_freq='mf', filters=None,
                   do_dark_correction=True, dark_correction=None,
                  ):

    if isinstance(cl_data, str):
        cl_data = np.load(cl_data,allow_pickle=True).item()

    if used_temps is None:
        try:
            used_temps = cl_data['metadata']['used_temps']
        except KeyError:
            raise ValueError(
                "Must specify `used_temps` for older version CL datafile.")
    if array_freq.lower() == 'mf':
        freq1, freq2 = '90','150'
        lower_f,upper_f = 62e9, 199e9
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
    elif array_freq.lower() == 'uhf':
        freq1, freq2 = '220','280'
        lower_f, upper_f = 180e9, 329e9
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
    elif array_freq.lower() == 'lf':
        freq1, freq2 = '30','40'
        lower_f, upper_f = 20e9, 60e9
        bl_freq_map = {bl: freq1 for bl in [10,11]}#[0, 3, 5, 10, 11]}
        bl_freq_map.update({bl: freq2 for bl in [8,9]})#[1, 2, 8, 9]})
    else:
        raise ValueError("`array_freq` must be one of {'lf','mf','uhf'}")

    if do_dark_correction:
        if dark_correction is None:
            dark_correction = compute_dark_correction(
                cl_data, used_temps=used_temps, array_freq=array_freq)

    if filters is None:
        filters = array_freq

    filt_func = filter_func(filters)
    freq1_func, freq2_func = bandpass_funcs(array_freq)

    def freq1_fit(temps, C, eta):
        ans=np.zeros(len(temps))
        for i, T in enumerate(temps):
            ans[i] =(C-eta*1e12*quad(
                lambda v: freq1_func(v)*filt_func(v)*cnst.h*v /
                (np.e**(cnst.h*v/(cnst.k*T))-1), lower_f, upper_f)[0])
        return ans        
        
    def freq2_fit(temps, C, eta):
        ans=np.zeros(len(temps))
        for i, T in enumerate(temps):
            ans[i] =(C-eta*1e12*quad(
                lambda v: freq2_func(v)*filt_func(v)*cnst.h*v /
                (np.e**(cnst.h*v/(cnst.k*T))-1), lower_f, upper_f)[0])
        return ans        

    eta_dict = {
        'metadata': cl_data['metadata'],
        'data' : {}
    }

    for bg in cl_data['data'].keys():
        if bg in cl_data['metadata']['optical_bl']:
            coupling = 'optical'
        else:
            coupling = 'dark'
        freq = bl_freq_map[bg]
        if freq==freq1:
            fit_func = freq1_fit
        else:
            fit_func = freq2_fit

        for sb in cl_data['data'][bg].keys():
            for ch, d in cl_data['data'][bg][sb].items():
                psat = np.array(d['psat'])
                temp = np.array(d['temp'])

                if do_dark_correction:
                    for i, t in enumerate(temp):
                        idx = used_temps.index(t)
                        psat[i] -= dark_correction[freq][idx]

                if len(temp)<2:
                    continue

                popt, pcov = curve_fit(fit_func, temp, psat)

                # key creation
                if bg not in eta_dict['data'].keys():
                    eta_dict['data'][bg] = dict()
                if sb not in eta_dict['data'][bg].keys():
                    eta_dict['data'][bg][sb] = dict()
                eta_dict['data'][bg][sb][ch] = popt[1]

    return eta_dict


def plot_opteff(eta_dict, ufm='', array_freq='mf', return_plots=False):
    """
    eta_dict = {
        'optical':{freq1:[],freq2:[]},
        'dark':{freq1:[], freq2:[]},
    }
    """
    if array_freq.lower() == 'mf':
        freq1, freq2 = '90','150'
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
    elif array_freq.lower() == 'uhf':
        freq1, freq2 = '220','280'
        bl_freq_map = {bl: freq1 for bl in [0, 1, 4, 5, 8, 9]}
        bl_freq_map.update({bl: freq2 for bl in [2, 3, 6, 7, 10, 11]})
    elif array_freq.lower() == 'lf':
        freq1, freq2 = '30','40'
        bl_freq_map = {bl: freq1 for bl in [10,11]}#[0, 3, 5, 10, 11]}
        bl_freq_map.update({bl: freq2 for bl in [8,9]})#[1, 2, 8, 9]})
    else:
        raise ValueError("`array_freq` must be one of {'lf','mf','uhf'}")

    eta = {
        'optical':{freq1:[],freq2:[]},
        'dark':{freq1:[], freq2:[]},
    }
    # Process eta_dict to be simpler for plotting.
    for bg in eta_dict['data'].keys():
        if bg in eta_dict['metadata']['optical_bl']:
            coupling = 'optical'
        else:
            coupling = 'dark'
        freq = bl_freq_map[bg]

        for sb in eta_dict['data'][bg].keys():
            eta[coupling][freq] += list(eta_dict['data'][bg][sb].values())

    fig_opt, ax = plt.subplots(figsize=(9,4), ncols=2)

    for i, freq in enumerate([freq1,freq2]):
        to_plot = np.array(eta['optical'][freq])
        to_plot = to_plot[to_plot>0]
        ax[i].hist(to_plot, range=(0,1.2), bins=24)
        med = np.nanmedian(to_plot)
        std = np.nanstd(to_plot)
        ax[i].axvline(med, linestyle='--', color='k', label=f"{med:.2f}$\pm${std:.2f}")
        ax[i].set_title(f'{freq} GHz')

        ax[i].legend(fontsize='small')
        ax[i].set_xlabel("Optical Efficiency")
        ax[i].set_ylabel("Count")

    fig_opt.suptitle(f'{ufm} Optical Efficiency')

    fig_dark, ax = plt.subplots(figsize=(9,4), ncols=2)
    for i, freq in enumerate([freq1, freq2]):
        ax[i].hist(eta['dark'][freq], range=(-0.9,0.9), bins=24)
        med = np.nanmedian(eta['dark'][freq])
        std = np.nanstd(eta['dark'][freq])
        ax[i].axvline(med, linestyle='--', color='k', label=f"{med:.2f}$\pm${std:.2f}")
        ax[i].set_title(f'{freq} GHz')

        ax[i].legend(fontsize='small')
        ax[i].set_xlabel("Dark Efficiency")
        ax[i].set_ylabel("Count")

    fig_dark.suptitle(f'{ufm} Dark Efficiency')

    if return_plots:
        return fig_opt, fig_dark