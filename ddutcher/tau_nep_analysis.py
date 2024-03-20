import numpy as np
import os
import matplotlib.pyplot as plt


def get_tau_nep_data(metadata_fp, bgmap_fp=None, optical_bl=[8,9,10,11]):
    nrows, ncols = 3, 4
    metadata = np.genfromtxt(metadata_fp, delimiter=",",unpack=False,
                             dtype=None, names=True, encoding=None)
    biasstep_lines = metadata[0::2]
    noise_lines = metadata[1::2]

    if bgmap_fp is None:
        # Read in bias group map from last bias step
        bsa_fp = metadata[-4][-2]
        if not os.path.isfile(bsa_fp):
            bsa_fp = bsa_fp.replace('/data/','/data2/')
        bsa = np.load(bsa_fp, allow_pickle=True).item()
        bgmap_fp = bsa['meta']['bgmap_file']
    bg_map = np.load(bgmap_fp, allow_pickle=True).item()

    # Channels might not be the same in bgmap and in noise/biassteps:
    # make a channel mask.
    bsa_fp = biasstep_lines[0][-2]
    if not os.path.isfile(bsa_fp):
        bsa_fp = bsa_fp.replace('/data/','/data2/')
    bias = np.load(bsa_fp, allow_pickle=True).item()
    bgmap_bandchan = list(zip(bg_map['bands'], bg_map['channels']))
    bias_bandchan = list(zip(bias['bands'], bias['channels']))
    if len(bgmap_bandchan) > len(bias_bandchan):
        bg_mask = [(b,c) in bias_bandchan for (b,c) in bgmap_bandchan]
        bs_mask = np.ones(len(bias_bandchan), dtype=bool)
    elif len(bgmap_bandchan) < len(bias_bandchan):
        bs_mask = [(b,c) in bgmap_bandchan for (b,c) in bias_bandchan]
        bg_mask = np.ones(len(bgmap_bandchan), dtype=bool)
    else:
        bg_mask= np.ones(len(bgmap_bandchan), dtype=bool)
        bs_mask = bg_mask

    num_chans = np.min((sum(bg_mask), sum(bs_mask)))
    num_biases = len(biasstep_lines['bias_v'])

    all_data = {}
    all_data['vbias'] = biasstep_lines['bias_v']
    all_data['bands'] = bg_map['bands'][bg_mask]
    all_data['channels'] = bg_map['channels'][bg_mask]
    all_data['bgmap'] = bg_map['bgmap'][bg_mask]

    for k in ['tau_eff', 'Si', 'R0', 'wl', 'Pj', 'tau_std']:
        all_data[k] = np.zeros((num_biases, num_chans))

    for ind, line in enumerate(biasstep_lines):
        bsa_fp = line[-2]
        if not os.path.isfile(bsa_fp):
            bsa_fp = bsa_fp.replace('/data/','/data2/')
        biasstep = np.load(bsa_fp, allow_pickle=True).item()

        for k in ['tau_eff', 'Si', 'R0', 'Pj']:
            all_data[k][ind,:] = biasstep[k][bs_mask]
        tau_std = np.zeros(len(biasstep['step_fit_pcovs']))
        for det in range(len(biasstep['step_fit_pcovs'])):
            tau_std[det] = np.sqrt(np.diag(biasstep['step_fit_pcovs'][det]))[1]
        all_data['tau_std'][ind,:] = tau_std[bs_mask]

    for ind, line in enumerate(noise_lines):
        noise_fp = line[-2]
        if not os.path.isfile(noise_fp):
            noise_fp = noise_fp.replace('/data/','/data2/')
        noise = np.load(noise_fp, allow_pickle=True).item()
        if 'noisedict' in noise.keys():
            noise = noise['noisedict']
        all_data['wl'][ind, :] = noise['noise_pars'][:,0][bs_mask]

    """I forget why I reformatted the data like this,
    maybe it was just to match an even earlier analysis."""

    results_dict = dict()
    metadata = {'dataset':metadata_fp,
            'bgmap_fp':bgmap_fp,
            'tunefile':biasstep['meta']['tunefile'],
            'optical_bl' : optical_bl,
           }

    for idx in range(len(all_data['channels'])):
        bl = all_data['bgmap'][idx]
        if bl == -1:
            continue
        if bl not in results_dict.keys():
            results_dict[bl] = dict()
        sb = all_data['bands'][idx]
        ch = all_data['channels'][idx]
        if sb not in results_dict[bl].keys():
            results_dict[bl][sb] = dict()
        if ch not in results_dict[bl][sb].keys():
            results_dict[bl][sb][ch] = {'v_bias' : all_data['vbias']}
        d = results_dict[bl][sb][ch]
        for k in ['tau_eff', 'Si', 'R0', 'wl', 'Pj', 'tau_std']:
            d[k] = all_data[k][:,idx]
        d['R0'] = np.abs(d['R0'])
        d['rfrac'] = d['R0'] / np.nanmean(d['R0'][:2])
        d['nep'] = d['wl'] * 1e6 / np.abs(d['Si'])        
        d['r_tes'] = d.pop('R0') * 1e3
        d['s_i'] = d.pop('Si')
        d['tau'] = d.pop('tau_eff') * 1e3
        d['tau_std'] = d.pop('tau_std') * 1e3
        d['nei'] = d.pop('wl')
        d['p_tes'] = d.pop('Pj') * 1e12

    return_dict = {'metadata':metadata, 'data':results_dict}
    return return_dict


def compute_transition_values(results_dict):
    """
    Compute the per-bias line values at 50% Rn for tau and NEP.
    """
    transition_values = {
        'bands':[],
        'channels':[],
        'bgmap':[],
        'tau':[],
        'nep':[],
        'rfrac':[],
    }
    if 'data' in results_dict.keys():
        data = results_dict['data']
    for ind, bg in enumerate(sorted(list(data.keys()))):
        for sb in data[bg].keys():
            for ch, d in data[bg][sb].items():
                m = np.zeros(len(d['rfrac']), dtype=bool)
                try:
                    stop = np.where(d['rfrac'] < 0.2)[0]
                    # in case we never reached Rfrac = 0.2:
                    if len(stop) == 0:
                        stop = [len(m)]
                    stop=stop[0]
                    start_inds = np.where(d['rfrac'] > 0.8)[0]
                    # want last start_ind < stop ind
                    start = start_inds[start_inds < stop][-1]
                except IndexError:
                    continue
                m[start:stop] = True
                m[d['tau']<0.25] = False
                m[(d['tau_std']/d['tau']) > 0.1] = False
                m[d['nep'] > 1e3] = False
                try:
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                transition_values['bands']+=[sb]
                transition_values['channels']+=[ch]
                transition_values['bgmap']+=[bg]
                transition_values['tau']+=[d['tau'][m][ind]]
                transition_values['nep']+=[d['nep'][m][ind]]
                transition_values['rfrac']+=[d['rfrac'][m][ind]]

    for k, val in transition_values.items():
        transition_values[k] = np.asarray(val)

    return {'metadata': results_dict['metadata'],
            'data': transition_values}


def plot_transition_hist(
    trans_dict, key, target_bg=range(12),
    plot_by_bl=False, array_freq = 'uhf',
    nrows=3, ncols=4, xrange=None, bins=None, plot_title='',
    return_plot=False,
):
    transition_values = trans_dict['data']
    optical_bl = trans_dict['metadata'].get('optical_bl', [])

    if key.lower() == "tau":
        if xrange is None:
            xrange = (0, 5)
        if bins is None:
            bins = 20
        label = "{med:.2f} ms"
        xlabel = "tau_eff (ms)"
        title = f"{plot_title} Tau at 50% Rn"
    elif key.lower() == 'nep':
        if xrange is None:
            xrange = (0, 100)
        if bins is None:
            bins = 20
        label = "{med:.0f} aW/rtHz"
        xlabel = "NEP (aW/rtHz)"
        title = f"{plot_title} NEP at 50% Rn"
    else:
        raise ValueError("`key` must be one of ['tau','nep']")

    if plot_by_bl:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
        for ind, bg in enumerate(target_bg):
            inds = np.unravel_index(bg, (nrows, ncols))
            ax = axes[inds]
            to_plot = transition_values[key][transition_values['bgmap'] == bg]
            ax.hist(to_plot, range=xrange, bins=bins)
            med = np.nanmedian(to_plot)
            ax.axvline(med, linestyle='--', color='k', label=label.format(med=med))
            ax.legend(fontsize='small', loc='upper right')
            ax.set_title(f'BL {bg}: {len(to_plot)} TESs')
        fig.supxlabel(xlabel)
        fig.suptitle(title, fontsize=16)
        fig.supylabel("Count")
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,5))
        if array_freq.lower() == "lf":
            raise NotImplementedError
        freq1_bg = [0, 1, 4, 5, 8, 9]
        freq2_bg = [2, 3, 6, 7, 10, 11]
        if array_freq.lower() == "uhf":
            freq1, freq2 = "220", "280"
        else:
            freq1, freq2 = "90", "150"
        freq1_opt, freq1_dark = [], []
        freq2_opt, freq2_dark = [], []
        for bg in target_bg:
            now = list(transition_values[key][transition_values['bgmap'] == bg])
            if bg in freq1_bg:
                if bg in optical_bl:
                    freq1_opt += now
                else:
                    freq1_dark += now
            else:
                if bg in optical_bl:
                    freq2_opt += now
                else:
                    freq2_dark += now
        axes[0].hist(freq1_dark, range=xrange, bins=bins, label=f"{freq1} Dark",
                     alpha=0.7)
        axes[0].hist(freq1_opt, range=xrange, bins=bins, label=f"{freq1} Optical",
                     alpha=0.7)
        axes[0].axvline(np.nanmedian(freq1_dark), linestyle='--', color='C0',
                   label=label.format(med=np.nanmedian(freq1_dark)))
        axes[0].axvline(np.nanmedian(freq1_opt), linestyle='--', color='C1',
                   label=label.format(med=np.nanmedian(freq1_opt)))

        axes[1].hist(freq2_dark, range=xrange, bins=bins, label=f"{freq2} Dark",
                    alpha=0.7)
        axes[1].hist(freq2_opt, range=xrange, bins=bins, label=f"{freq2} Optical",
                     alpha=0.7)
        axes[1].axvline(np.nanmedian(freq2_dark), linestyle='--', color='C0',
                   label=label.format(med=np.nanmedian(freq2_dark)))
        axes[1].axvline(np.nanmedian(freq2_opt), linestyle='--', color='C1',
                   label=label.format(med=np.nanmedian(freq2_opt)))
        for ax in axes:
            ax.legend(loc='upper right')

        fig.supxlabel(xlabel)
        fig.suptitle(title, fontsize=16)
        fig.supylabel("Count")
        fig.tight_layout()

    if return_plot:
        return fig, axes


def plot_tau_nep_data(results_dict, x, y, nrows=3, ncols=4, figsize=(9,10),
                      plot_title='', return_plot=False, **plot_kwargs):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    options = ['tau','s_i','r_tes','rfrac', 'nep','v_bias', 'nei', 'p_tes']
    if x not in options or y not in options:
        raise ValueError(f"x and y must each be one of {options}")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        count = 0
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.ones(len(d[x]), dtype=bool)
                if x=='rfrac' or y=='rfrac':
                    try:
                        stop = np.where(d['rfrac'] < 0.2)[0]
                        # in case we never reached Rfrac = 0.2:
                        if len(stop) == 0:
                                stop = [len(m)]
                        stop=stop[0]
                        start_inds = np.where(d['rfrac'] > 0.8)[0]
                        # want last start_ind < stop ind
                        start = start_inds[start_inds < stop][-1] + 1
                    except IndexError:
                        continue
                    m = ~m
                    m[start:stop] = True
                if x=='tau' or y=='tau':
                    m[d['tau']<0.25] = False
                    m[(d['tau_std']/d['tau']) > 0.1] = False
                if 'ne' in y:
                    m[d['nep'] > 1e3] = False
                    ax.semilogy(d[x][m], d[y][m], alpha=0.2, marker='.', markersize=2)
                else:
                    ax.plot(d[x][m], d[y][m], alpha=0.2, marker='.', markersize=2)
                count += 1
        ax.set_title(f'BL {bg}: {count} TESs')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.set(**plot_kwargs)
    fig.supylabel(y, fontsize=20)
    fig.supxlabel(x, fontsize=20)
    plt.suptitle(f'{plot_title} {y} vs {x}', fontsize=16)
    plt.tight_layout()

    if return_plot:
        return fig, axes
