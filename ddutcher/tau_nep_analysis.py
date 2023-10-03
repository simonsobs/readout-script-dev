import numpy as np
import os
import matplotlib.pyplot as plt


def get_tau_nep_data(metadata_fp, bgmap_fp=None):
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
            'tunefile':biasstep['meta']['tunefile']
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
    }
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.zeros(len(d['rfrac']), dtype=bool)
                try:
                    stop = np.where(d['rfrac'] < 0.2)[0][0]
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

    for k, val in transition_values.items():
        transition_values[k] = np.asarray(val)

    return transition_values


def plot_transition_hist(
    transition_values, key, target_bg=range(12),
    nrows=3, ncols=4, xrange=None, bins=None, plot_title='',
    return_plot=False,
):
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

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(target_bg):
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        to_plot = transition_values[key][transition_values['bgmap'] == bg]
        ax.hist(to_plot, range=xrange, bins=bins)
        med = np.nanmedian(to_plot)
        ax.axvline(med, linestyle='--', color='k', label=label.format(med=med))
        ax.legend(fontsize='small', loc='upper right')
        ax.set_title(f'BL {bg}')
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
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.ones(len(d[x]), dtype=bool)
                if x=='rfrac' or y=='rfrac':
                    try:
                        stop = np.where(d['rfrac'] < 0.2)[0][0]
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
                    ax.semilogy(d[x][m], d[y][m], alpha=0.2)
                else:
                    ax.plot(d[x][m], d[y][m], alpha=0.2)
        ax.set_title(f'BL {bg}')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.set(**plot_kwargs)
    fig.supylabel(y, fontsize=20)
    fig.supxlabel(x, fontsize=20)
    plt.suptitle(f'{plot_title} {y} vs {x}', fontsize=16)
    plt.tight_layout()

    if return_plot:
        return fig, axes


def plot_tau_hist(results_dict, nrows=3, ncols=4, xrange=(0,10), plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        tau_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.zeros(len(d['rfrac']), dtype=bool)
                try:
                    stop = np.where(d['rfrac'] < 0.2)[0][0]
                    start_inds = np.where(d['rfrac'] > 0.8)[0]
                    # want last start_ind < stop ind
                    start = start_inds[start_inds < stop][-1]
                except IndexError:
                    continue
                m[start:stop] = True
                m[d['tau']<0.25] = False
                m[(d['tau_std']/d['tau']) > 0.1] = False
                try: 
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                tau_list.append(d['tau'][m][ind])
        ax.hist(tau_list, range=xrange, bins=20)
        med = np.nanmedian(tau_list)
        ax.axvline(med, linestyle='--', color='k', label=f'{med:.3} ms')
        ax.set_title(f'BL {bg}')
        if inds[1] == 0:
            ax.set_ylabel('Count')
        if inds[0] == nrows-1:
            ax.set_xlabel('tau_eff (ms)')
        ax.legend(fontsize='small', loc='upper right')
    plt.suptitle(f'{plot_title} Tau at 50% Rn', fontsize=16)
    plt.tight_layout()


def plot_nep_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        nep_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.zeros(len(d['rfrac']), dtype=bool)
                try:
                    stop = np.where(d['rfrac'] < 0.2)[0][0]
                    start_inds = np.where(d['rfrac'] > 0.8)[0]
                    # want last start_ind < stop ind
                    start = start_inds[start_inds < stop][-1]
                except IndexError:
                    continue
                m[start:stop] = True
                m[d['nep'] > 1e3] = False
                try:
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                nep_list.append(d['nep'][m][ind])
        ax.hist(nep_list, bins=20, range=(0,100))
        med = np.nanmedian(nep_list)
        try:
            ax.axvline(med, linestyle='--', color='k', label=f'{int(med)} aW/rtHz')
        except ValueError:
            pass
        ax.set_title(f'BL {bg}')
        if inds[0] == nrows-1:
            ax.set_xlabel('NEP [aW/rtHz])')
        if inds[1] == 0:
            ax.set_ylabel('Count')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')
    plt.suptitle(f'{plot_title} NEP at 50%Rn', fontsize=16)
    plt.tight_layout()


def plot_ptes_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        ax = axes[inds]
        ptes_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.zeros(len(d['rfrac']), dtype=bool)
                try:
                    stop = np.where(d['rfrac'] < 0.2)[0][0]
                    start_inds = np.where(d['rfrac'] > 0.8)[0]
                    # want last start_ind < stop ind
                    start = start_inds[start_inds < stop][-1]
                except IndexError:
                    continue
                m[start:stop] = True
                try:
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                ptes_list.append(d['p_tes'][m][ind])
        ax.hist(ptes_list, range=(0,50),bins=25)
        med = np.nanmedian(ptes_list)
        ax.axvline(med, linestyle='--', color='k', label=f'{med:0.1f} pW')
        ax.set_title(f'BL {bg}')
        if inds[1] == 0:
            ax.set_ylabel('Count')
        if inds[0] == nrows-1:
            ax.set_xlabel('p_tes (pW)')
        ax.legend(fontsize='small', loc='upper right')
    plt.suptitle(f'{plot_title} P_tes at 50% Rn', fontsize=16)
    plt.tight_layout()
