# psatvstemp_analysis.py
# Functions for analyzing and plotting data taken
# as part of a cold load ramp or bath ramp.

import os
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def collect_psatvstemp_data(
    metadata_fp,
    ls_ch=15,
    cut_increasing_psat=True,
    optical_bl=[],
    temp_list=None,
    temps_to_cut=None,
    thermometer_id=None,
    temp_offset=0,
    temp_scaling=1,
    min_rn=0,
    max_rn=np.inf,
    bl_plot=True,
    freq_plot=False,
    array_freq="mf",
    assem_type='ufm',
    plot_title="",
    figsize=None,
    return_data=False,
):
    """
    Collect Psat vs. temperature data from a metadata csv.
    Works for both bath temp sweeps and cold load sweeps.
    
    Parameters
    ----------
    metadata_fp : str
        Path to a metadata csv file with columns ordered like
        temp, bias_v, bias_line, smurf_band, filepath, note.
    ls_ch : [13, 14, 15, 16]
        Which Lakeshore channel recording from the file to use.
    cut_increasing_psat : bool, default True.
        Cut detector data points that have a higher Psat than
        the preceding point.
    optical_bl : array-like
        Data from these bias lines will be plotted separately.
    temp_list : array-like
    temps_to_cut : array_like, optional
        A list of temperatures to exclude from the collected data.
        Evaluated before any corrections have been applied.
    thermometer_id : ['X-066','X-129'], default None
        Thermometer ID is used to apply a recalibration to the
        temperature readings to align them with a Lakeshore standard.
    temp_offset : float, default 0.
        Apply a fixed offset (in Kelvin) to all temperature values.
        Not used if `thermometer_id` is set.
    temp_scaling : float, default 1.
        Scale temperature values. Not used if `thermometer_id` is set.
    min_rn: float, default 0.
        In ohms.
    max_rn: float, default inf.
        In ohms.
    bl_plot : bool, default True.
        Plot the data by bias line.
    freq_plot : bool, default False.
        Plot the data by observing frequency.
    assem_type : {'ufm', 'spb'}
    array_freq : {'mf', 'uhf'}
    plot_title : str
    figsize : (float, float), optional
    return_data : bool, default False.
        Return the collected data as a dictionary.

    Returns
    -------
    data_dict : dict
        Dictionary organized by bias line, then, smurf band, then channel,
        and containing ordered arrays of temperature and psat values.
        Only returned if *return_data* is True.
    """
    ls_chans =  [13, 14, 15, 16]
    assert ls_ch in ls_chans
    metadata = np.genfromtxt(metadata_fp, delimiter=',', dtype=None,
                             names=True, encoding=None)
    temps_to_cut = np.atleast_1d(temps_to_cut)
    data_dict = {}
    if temp_list is not None:
        temps = []
        for line in metadata:
            if isinstance(line[0], np.int64):
                now = line[0]
            else:
                ls_temps = line[0].split(' ')
                now = ls_temps[2]
            if now not in temps:
                temps.append(now)
        if len(temps) != len(temp_list):
            raise ValueError(
                f"Length of supplied temperature list ({len(temp_list)}) does not"
                +f" match the number of recorded temperatures ({len(temps)})."
            )
    for line in metadata:
        try:
            temp, bias, bl, sb, fp, note = line
        except ValueError:
            temp, bl, sb, fp, note = line
        if isinstance(temp, np.int64):
            pass
        elif isinstance(temp, str):
            temp = temp.split(" ")[ls_chans.index(ls_ch)]
        else:
            raise ValueError()
        if temp_list is not None:
            # Replace temperature with its corresponding
            # entry in temps_list
            temp = temp_list[temps.index(temp)]
        # Put temperatures into Kelvin.
        temp = float(temp)
        if temp > 40:
            temp *= 1e-3
        temp = round(temp, 3)
        if temp in temps_to_cut or temp*1e3 in temps_to_cut:
            continue
        if thermometer_id is not None:
            if thermometer_id.lower() == "x-066":
                temp_scaling = 0.870
                temp_offset = 0.0117
            elif thermometer_id.lower() == "x-129":
                # this includes the 7mK offset of DR MC thermometer
                temp_scaling = 0.875
                temp_offset = 0.021
            else:
                raise ValueError(
                    f"Unsupported thermometer_id {thermometer_id}."
                )
        temp_corr = temp * temp_scaling + temp_offset
        # Ignore files that aren't single-bias-line IV curves
        if bl == "all":
            continue
        bl = int(bl)
        if 'iv_raw_data' in fp:
            iv_analyzed_fp = fp.replace("iv_raw_data", "iv")
        elif 'iv_info' in fp:
            iv_analyzed_fp = fp.replace("iv_info", "iv_analyze")
        else:
            iv_analyzed_fp = fp
        # First look for data on long term storage /data2
        iv_analyzed_fp = iv_analyzed_fp.replace("/data/smurf_data", "/data2/smurf_data")
        # If not there, look for copy on smurf-srv15
        if not os.path.exists(iv_analyzed_fp):
            _,_,_,_,date,slot,sess,_,fp = iv_analyzed_fp.split('/')
            new_fp = os.path.join(
                "/data/smurf/", fp[0:5], slot, "*run_iv/outputs", fp
            )
            try:
                iv_analyzed_fp = glob(new_fp)[0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find {iv_analyzed_fp} or "
                    f"any file matching {new_fp} on daq."
                )
        iv_analyzed = np.load(iv_analyzed_fp, allow_pickle=True).item()
        if 'data' in iv_analyzed.keys():
            iv_analyzed = iv_analyzed['data']

        if bl not in data_dict.keys():
            data_dict[bl] = {}
        for sb in iv_analyzed.keys():
            if sb == "high_current_mode":
                continue
            for ch, d in iv_analyzed[sb].items():

                # same cuts as for iv plots
                ind = np.where(d['p_tes'] > 15)[0]
                if len(ind)==0:
                    continue
                if np.abs(
                    np.std(d['R'][-100:]) / np.mean(d['R'][-100:])
                ) > 5e-3:
                    continue
                if d['R'][-1] < 2e-3:
                    continue
                try:
                    psat_idx = np.where(d['R'] < 0.9 * d['R_n'])[0][-1]
                except:
                    continue
                psat = d['p_tes'][psat_idx]
                if psat < 0:
                    continue

                if cut_increasing_psat:
                    try:
                        prev_psat = data_dict[bl][sb][ch]['psat'][-1]
                        if psat > prev_psat:
                            continue
                    except:
                        pass

                if not min_rn < d['R_n'] < max_rn:
                    continue

                # key creation
                if sb not in data_dict[bl].keys():
                    data_dict[bl][sb] = dict()
                if ch not in data_dict[bl][sb].keys():
                    data_dict[bl][sb][ch] = {
                        "temp" : [], "psat" : [], "R_n" : [], 
                    }
                data_dict[bl][sb][ch]['temp'].append(temp_corr)
                data_dict[bl][sb][ch]['psat'].append(psat)
                data_dict[bl][sb][ch]["R_n"].append(d["R_n"])

    if bl_plot:
        plot_by_bl(data_dict, plot_title, figsize)

    if freq_plot:
        assert assem_type == 'ufm'
        plot_by_freq(data_dict, array_freq, optical_bl, figsize, plot_title)

    if return_data:
        return data_dict


def plot_by_bl(data_dict, plot_title='', figsize=None):
    tot = 0
    ncols = np.min((len(data_dict.keys()), 4))
    nrows = int(np.ceil(len(data_dict.keys()) / 4))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx, bl in enumerate(sorted(list(data_dict.keys()))):
        inds = np.unravel_index(idx, (nrows,ncols))
        ax = axes[inds]
        tes_yield = np.sum(
            [len(data_dict[bl][sb].keys()) for sb in data_dict[bl].keys()]
            )
        tot += tes_yield
        for sb in data_dict[bl].keys():
            for ch, d in data_dict[bl][sb].items():
                if min(d['temp']) > 1:
                    run='Coldload'
                else:
                    run='Bath'
                ax.plot(d['temp'], d['psat'], marker=".", alpha=0.4, linewidth=1)
        ax.set_title(f"BL {bl}, yield {tes_yield}", fontsize=10)
        ax.grid(linestyle="--")
        if inds[1] == 0:
            ax.set_ylabel("Psat [pW]")
        if inds[0] == nrows-1:
            ax.set_xlabel(f"{run} Temperature [K]")
    plt.suptitle(plot_title, fontsize=12)
    plt.tight_layout()
    print(f"Total TES yield: {tot}")    


def plot_by_freq(data_dict, array_freq='mf', optical_bl=[],
                 figsize=None, plot_title=''):
    if array_freq.lower() == "uhf":
        freq1, freq2 = "220","280"
    else:
        freq1, freq2 = "90","150"
    bl_freq_map = {bl:freq1 for bl in [0,1,4,5,8,9]}
    bl_freq_map.update({bl:freq2 for bl in [2,3,6,7,10,11]})
    freq_colors = {
        freq1 : 'C0',
        freq2 :'C2',
        'Dark_' + freq1 : 'C3',
        'Dark_' + freq2 : 'C1',
    }
    labeled_dict = dict()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for idx, bl in enumerate(sorted(list(data_dict.keys()))):
        freq = bl_freq_map[bl]
        if freq == freq1:
            ax = axes[0]
        else:
            ax = axes[1]
        if bl not in optical_bl:
            freq = 'Dark_' + freq
        c = freq_colors[freq]
        for sb in data_dict[bl].keys():
            for ind, (ch, d) in enumerate(data_dict[bl][sb].items()):
                label = None
                if ind == 0:
                    labeled = labeled_dict.get(freq, False)
                    if not labeled:
                        label = freq + 'GHz'
                        labeled_dict[freq] = True
                if min(d['temp']) > 1:
                    run='Coldload'
                else:
                    run='Bath'
                ax.plot(
                    d['temp'], d['psat'], alpha=0.2, color=c, label=label, marker='.',
                )
    axes[0].set_title(freq1 + 'GHz')
    axes[1].set_title(freq2 + 'GHz')
    for ax in axes:
        ax.set_xlabel(f"{run} Temperature [K]")
        ax.set_ylabel('Psat (pW)')
        ax.legend(loc='best')
    plt.suptitle(plot_title, fontsize=12)
    plt.tight_layout()


def fit_bathramp_data(
    data_dict,
    assem_type="ufm",
    array_freq="mf",
    optical_bl=None,
    restrict_n=False,
    fit_g=False,
    return_data=False,
    plot=True,
    plot_title="",
):
    """
    Fit thermal parameters from bath ramp data.

    Parameters
    ----------
    data_dict : dict
        Dictionary returned by collect_psatvstemp_data
    assem_type : {'ufm', 'spb'}
    array_freq : {'mf', 'uhf'}
    optical_bl : array-like
        Psats from these bias lines will be plotted separately.
    restrict_n : bool
        Restrict fit values of n within 2--4
    fit_g : bool
        If True, fits G, n, Tc, Psat.
        Otherwise fits k, n, Tc, Psat, solves for G.
    return_data : bool
        If True, returns dictionary of fit parameters
    plot : bool
        Plot histograms of the fit parameters.
    plot_title : str
        Title for output plots

    Returns
    -------
    results_dict : dict
        Fit thermal parameters, keyed by bias line
        and absolute readout channel.
        
    """
    if not fit_g:
        p0 = [370, .190, 3]
        if restrict_n:
            def PsatofT(Tb, k, Tc, n):
                if (n<2 or n>4):
                    return 9e9
                power = k * (Tc**n - Tb**n)
                return power
        else:
            def PsatofT(Tb, k, Tc, n):
                power = k * (Tc**n - Tb**n)
                return power
    else:
        p0 = [100, .190, 3]
        if restrict_n:
            def PsatofT(Tb, G, Tc, n):
                if (n<2 or n>4):
                    return 9e9 
                return (G/n) * (Tc - Tb**n/Tc**(n-1))
        else:
            def PsatofT(Tb, G, Tc, n):
                return (G/n) * (Tc - Tb**n/Tc**(n-1))

    param_dict={}

    for bl in data_dict.keys():
        for sb in data_dict[bl].keys():
            for ch, d in data_dict[bl][sb].items():
                if len(d['psat']) < 4:
                    continue
                try:
                    popt, pcov = scipy.optimize.curve_fit(
                        PsatofT,
                        d['temp'],
                        np.asarray(d['psat']),
                        p0=p0,
                    )
                except RuntimeError:
                    continue
                kg, tc, n = popt
                sigma_kg, sigma_tc, sigma_n = np.sqrt(np.diag(pcov))
                
#                 # April 2022: testing using a cut on Tc fit uncertainty.
#                 if sigma_tc > 0.001:
#                     continue

                if bl not in param_dict.keys():
                    param_dict[bl] = dict()
                if sb not in param_dict[bl].keys():
                    param_dict[bl][sb] = dict()

                param_dict[bl][sb][ch] = {
                    'R_n' : data_dict[bl][sb][ch]['R_n'][-1]
                }

                if not fit_g:
                    param_dict[bl][sb][ch]["g"] = n * kg * (tc ** (n - 1))
                    param_dict[bl][sb][ch]["k"] = kg
                    param_dict[bl][sb][ch]["sigma_k"] = sigma_kg
                else:
                    param_dict[bl][sb][ch]["g"] = kg
                    param_dict[bl][sb][ch]['sigma_g'] = sigma_kg
                    param_dict[bl][sb][ch]["k"] = kg / (n * tc ** (n - 1))
                param_dict[bl][sb][ch]["tc"] = tc
                param_dict[bl][sb][ch]["sigma_tc"] = sigma_tc
                param_dict[bl][sb][ch]["n"] = n
                param_dict[bl][sb][ch]["sigma_n"] = sigma_n

                if 0.100 in d['temp']:
                    psat = d['psat'][(d['temp'].index(.100))]
                else:
                    psat = param_dict[bl][sb][ch]['k'] * (tc**n - 0.100**n)
                param_dict[bl][sb][ch]['psat100mk'] = psat

    if plot:
        plot_params(
            param_dict, assem_type, array_freq, optical_bl, restrict_n, plot_title
        )

    results_dict = {}
    results_dict['data'] = param_dict
    results_dict['metadata'] = {
        'units':{
            'psat100mk' : 'pW',
            'tc' : 'K',
            'g' : 'pW/K',
            'n' : '',
            'k' : '',
            'R_n' : 'ohms',
        }
    }
    if return_data:
        return results_dict


def plot_params(
    param_dict, assem_type='ufm', array_freq='mf', optical_bl=[],
    restrict_n=True, plot_title='',
):
    param_plot_kw = {
        'psat100mk' : {'range' : (0,20),
                       'label' : '%sGHz: %.1f $\pm$ %.1f pW',
                       'xlabel' : 'Psat at 100 mK [pW]',
                      },
        'tc' : {'range' : (0.120, 0.220),
                'label' : '%sGHz: %.3f $\pm$ %.3f K',
                'xlabel' : 'Tc [K]',
               },
        'g' : {'range' : (20, 600),
               'label' : '%sGHz: %0d $\pm$ %0d pW/K',
               'xlabel' : 'G [pW/K]',
              },
        'n' : {'range' : (1, 5),
               'label' : '%sGHz: %.1f $\pm$ %.1f',
              }
    }
    if restrict_n:
        param_plot_kw['n']['xlabel'] = 'n (restricted to 2--4)'
    else:
         param_plot_kw['n']['xlabel'] = 'n (free)'
        
    if array_freq.lower() == "uhf":
        freq1, freq2 = "220","280"
        freq1_psat = [16.9, 28.1]
        freq2_psat = [18.3, 30.5]
    else:
        freq1, freq2 = "90","150"
        freq1_psat = [2.0, 3.3]
        freq2_psat = [5.4, 9.0]

    if assem_type.lower() == 'ufm':
        bl_freq_map = {bl:freq1 for bl in [0,1,4,5,8,9]}
        bl_freq_map.update({bl:freq2 for bl in [2,3,6,7,10,11]})
        freq_colors = [
            (freq1,'C0'),(freq2,'C2'),
            ('Optical_'+freq1,'C0'),('Optical_'+freq2,'C2')
        ]
    else:
        bl_freq_map = {bl:freq1+'/'+freq2 for bl in np.arange(12)}
        freq_colors = [(freq1+'/'+freq2,'C0')]
        optical_bl = []

    # Build a dictionary that's useful for histograms,
    # and that works for MF/UHF UFMs and SPBs.
    plotting_dict = dict()
    for bl in param_dict.keys():
        for sb in param_dict[bl].keys():
            freq = bl_freq_map[bl]
            if bl in optical_bl:
                freq = 'Optical_'+freq
            if freq not in plotting_dict.keys():
                plotting_dict[freq] = dict()
            for key in ['g','k','tc','n','psat100mk', 'R_n']:
                if key not in plotting_dict[freq].keys():
                    plotting_dict[freq][key] = []
                now_param = [
                    param_dict[bl][sb][ch][key]
                    for ch in param_dict[bl][sb].keys()
            ]
                plotting_dict[freq][key] += now_param

    fig, ax = plt.subplots(nrows=4, figsize=(9,9))
    for freq, c in freq_colors:
        if 'Optical' in freq:
            dark = False
            histtype = 'step'
            ec = c
            lalpha=0.4
        else:
            dark = True
            histtype = 'bar'
            ec = None
            lalpha=1

        for i, param in enumerate(['psat100mk','tc','g','n']):
            d = param_plot_kw[param]
            if not dark and param != 'psat100mk':
                continue
            h = ax[i].hist(
                plotting_dict[freq][param],
                fc=c, alpha=0.4, bins=30, range=d['range'],
                histtype=histtype, ec=ec, linewidth=1.5,
            )
            med = np.nanmedian(plotting_dict[freq][param])
            std = np.nanstd(plotting_dict[freq][param])
            if not np.isnan(med):
                ax[i].axvline(
                    med, linestyle='--', color=c, alpha=lalpha,
                    label=d['label'] % (freq, med, std),
                )
            ax[i].set_xlabel(d['xlabel'])
            ax[i].set_ylabel('# of TESs')
            ax[i].set_title(' ')
            ax[i].legend(fontsize='small', loc='best')

    ax[0].axvspan(freq1_psat[0], freq1_psat[1],
                  hatch='\\', ec='C0', alpha=0.2, fill=False)
    ax[0].axvspan(freq2_psat[0], freq2_psat[1],
                  hatch='\\', ec='C2', alpha=0.2, fill=False)

    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()

    # Rn plot
    plt.figure()
    all_rn = []
    for freq, d in plotting_dict.items():
        all_rn += d['R_n']
    med = np.nanmedian(all_rn)
    plt.hist(all_rn, ec='k', histtype='step', bins=20,
             range=(5e-3,10e-3)
            )
    plt.axvline(
        med, linestyle='--', color='k',
        label='%.1f $\pm$ %.1f mohms' % (med*1e3, np.nanstd(all_rn)*1e3),
    )
    plt.xlabel('R_n (ohms)')
    plt.ylabel('Count')
    plt.title(plot_title)
    plt.legend()


def analyze_bathramp(
    metadata_fp,
    ls_ch=15,
    array_freq="mf",
    assem_type='ufm',
    optical_bl=None,
    restrict_n=False,
    fit_g=False,
    temp_list=None,
    temps_to_cut=None,
    thermometer_id=None,
    temp_offset=0,
    temp_scaling=1,
    min_rn=0,
    max_rn=np.inf,
    cut_increasing_psat=True,
    plot=True,
    plot_title="",
    figsize=None,
    return_data=False,
):
    data_dict = collect_psatvstemp_data(
        metadata_fp,
        ls_ch=ls_ch,
        array_freq=array_freq,
        assem_type=assem_type,
        temp_list=temp_list,
        thermometer_id=thermometer_id,
        temp_offset=temp_offset,
        temp_scaling=temp_scaling,
        cut_increasing_psat=cut_increasing_psat,
        min_rn=min_rn,
        max_rn=max_rn,
        temps_to_cut=temps_to_cut,
        optical_bl=optical_bl,
        bl_plot=plot,
        plot_title=plot_title,
        figsize=figsize,
        return_data=True,
    )
    
    results_dict = fit_bathramp_data(
        data_dict,
        array_freq=array_freq,
        assem_type=assem_type,
        optical_bl=optical_bl,
        restrict_n=restrict_n,
        fit_g=fit_g,
        return_data=return_data,
        plot=plot,
        plot_title=plot_title)

    results_dict['metadata']['dataset'] = metadata_fp
    results_dict['metadata']['restrict_n'] = restrict_n
    results_dict['metadata']['allowed_rn'] = [min_rn, max_rn]
    results_dict['metadata']['cut_increasing_psat'] = cut_increasing_psat
    results_dict['metadata']['thermometer_id'] = thermometer_id
    results_dict['metadata']['temp_list'] = temp_list
    results_dict['metadata']['optical_bl'] = optical_bl
    results_dict['metadata']['temp_offset'] = temp_list
    results_dict['metadata']['temp_scaling'] = temp_list
    results_dict['metadata']['temps_to_cut'] = temps_to_cut
    
    if return_data:
        return data_dict, results_dict


def plot_bathramp_iv(
    metadata_fp,
    temp=100,
    ls_ch=15,
    plot_title="",
    return_data=False,
):
    """
    Plots the IVs for each bias line taken at specified bath temperature.
    Assumes IVs were taken one bias line at a time.
    """
    ls_chans =  [13, 14, 15, 16]
    metadata = np.genfromtxt(metadata_fp, delimiter=',', dtype=None,
                             names=True, encoding=None)
    data_dict = {}
    tot = 0
    for line in metadata:
        try:
            this_temp, bias, bl, sb, fp, note = line
        except ValueError:
            this_temp, bl, sb, fp, note = line
        if isinstance(this_temp, np.int64):
            pass
        elif isinstance(this_temp, str):
            this_temp = this_temp.split(" ")[ls_chans.index(ls_ch)]
        else:
            raise ValueError()
        if not (float(this_temp) == temp or float(this_temp)*1e-3 == temp):
            continue
        # Ignore files that aren't single-bias-line IV curves
        if bl == "all":
            continue
        bl = int(bl)
        if 'iv_raw_data' in fp:
            iv_analyzed_fp = fp.replace("iv_raw_data", "iv")
        elif 'iv_info' in fp:
            iv_analyzed_fp = fp.replace("iv_info", "iv_analyze")
        else:
            iv_analyzed_fp = fp
        # First look for data on long term storage /data2
        iv_analyzed_fp = iv_analyzed_fp.replace("/data/smurf_data", "/data2/smurf_data")
        # If not there, look for copy on smurf-srv15
        if not os.path.exists(iv_analyzed_fp):
            _,_,_,_,date,slot,sess,_,fp = iv_analyzed_fp.split('/')
            new_fp = os.path.join(
                "/data/smurf/", fp[0:5], slot, "*run_iv/outputs", fp
            )
            try:
                iv_analyzed_fp = glob(new_fp)[0]
            except IndexError:
                raise FileNotFoundError(
                    f"Could not find {iv_analyzed_fp} or "
                    f"any file matching {new_fp} on daq."
                )
        iv_analyzed = np.load(iv_analyzed_fp, allow_pickle=True).item()
        if 'data' in iv_analyzed.keys():
            iv_analyzed = iv_analyzed['data']

        plt.figure()
        now_tot = 0

        if bl not in data_dict.keys():
            data_dict[bl] = {}
        for sb in iv_analyzed.keys():
            if sb=="high_current_mode":
                continue
            for ch, d in iv_analyzed[sb].items():
                lbl = "sb%02dch%03d" % (sb, ch)

                # same cuts as for iv plots
                ind = np.where(d['p_tes'] > 15)[0]
                if len(ind)==0:
                    continue
                if np.abs(
                    np.std(d['R'][-100:]) / np.mean(d['R'][-100:])
                ) > 5e-3:
                    continue
                if d['R'][-1] < 2e-3:
                    continue
                try:
                    psat_idx = np.where(d['R'] < 0.9 * d['R_n'])[0][-1]
                except:
                    continue
                psat = d['p_tes'][psat_idx]
                if psat < 0:
                    continue

                # cuts passed

                data_dict[bl][lbl] = d
                plt.plot(d['p_tes'],d['R'])
                now_tot += 1

        plt.title("BL %s: %s good ch" % (bl,now_tot))
        plt.xlim(0,30)
        plt.ylim(0,0.015)

        tot += now_tot
    print(tot)

    if return_data:
        return data_dict
