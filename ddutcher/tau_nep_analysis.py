import numpy as np
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

    all_data = {}
    all_data['vbias'] = biasstep_lines['bias_v']
    all_data['bands'] = bg_map['bands']
    all_data['channels'] = bg_map['channels']
    all_data['bgmap'] = bg_map['bgmap']

    num_chans = len(bg_map['channels'])
    num_biases = len(biasstep_lines['bias_v'])

    # Channels might not be the same in bgmap and in noise/biassteps:
    # make a channel mask.
    bsa_fp = biasstep_lines[0][-2]
    if not os.path.isfile(bsa_fp):
        bsa_fp = bsa_fp.replace('/data/','/data2/')
    bias = np.load(bsa_fp, allow_pickle=True).item()
    bgmap_bandchan = list(zip(bg_map['bands'], bg_map['channels']))
    bias_bandchan = list(zip(bias['bands'], bias['channels']))
    mask = [(b,c) in bgmap_bandchan for (b,c) in bias_bandchan]

    for k in ['tau_eff', 'Si', 'R0', 'wl']:
        all_data[k] = np.zeros((num_biases, num_chans))

    for ind, line in enumerate(biasstep_lines):
        bsa_fp = line[-2]
        if not os.path.isfile(bsa_fp):
            bsa_fp = bsa_fp.replace('/data/','/data2/')
        biasstep = np.load(bsa_fp, allow_pickle=True).item()
        if len(biasstep['channels'][mask]) != len(bg_map['channels']):
            print(len(biasstep['channels'][mask]), len(bg_map['channels']))
            raise ValueError("Channel list mismatch")
        elif not (biasstep['channels'][mask] == bg_map['channels']).all():
            print(len(biasstep['channels'][mask]), len(bg_map['channels']))
            raise ValueError("Channel list mismatch")
        for k in ['tau_eff', 'Si', 'R0']:
            all_data[k][ind,:] = biasstep[k][mask]

    for ind, line in enumerate(noise_lines):
        noise_fp = line[-2]
        if not os.path.isfile(noise_fp):
            noise_fp = noise_fp.replace('/data/','/data2/')
        noise = np.load(noise_fp, allow_pickle=True).item()
        if 'noisedict' in noise.keys():
            noise = noise['noisedict']
        if not (noise['channels'][mask] == bg_map['channels']).all():
            print(len(noise['channels']), len(bg_map['channels']))
            raise ValueError("Channel list mismatch")
        all_data['wl'][ind, :] = noise['noise_pars'][:,0][mask]

    """I forget why I reformatted the data like this,
    maybe it was just to match an even earlier analysis."""

    results_dict = dict()

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
        for k in ['tau_eff', 'Si', 'R0', 'wl']:
            d[k] = all_data[k][:,idx]
        d['rfrac'] = d['R0'] / np.nanmean(d['R0'][:3])
        d['nep'] = d['wl'] * 1e6 / np.abs(d['Si'])        
        d['r_tes'] = d.pop('R0') * 1e3
        d['s_i'] = d.pop('Si')
        d['tau'] = d.pop('tau_eff') * 1e3
        d['nei'] = d.pop('wl')

    return results_dict

def plot_tau_nep_data(results_dict, x, y, nrows=3, ncols=4,
                 figsize=(9, 10), plot_title=''):
    options = ['tau','s_i','r_tes','rfrac', 'nep','v_bias', 'nei']
    if x not in options or y not in options:
        raise ValueError(f"x and y must each be one of {options}")
    ylims = {
        'r_tes': (0,10),
        'rfrac':(0,1),
        'tau':(0,5),
        'nei':(1e1,1e4),
        'nep':(3,300),
        'v_bias':(0,18)
    }
    xlims = {
        'r_tes': (0,10),
        'rfrac':(0,1),
        'tau':(0,5),
        'nei':(1e1,1e4),
        'nep':(3,300),
        'v_bias':(0,18),
    }
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(ind, (nrows, ncols))
        ax = axes[inds]
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                if x=='rfrac' or y=='rfrac':
                    m = np.where(
                        (d['r_tes'] > 1e-1) &
                        (d['rfrac'] < 0.8) &
                        (np.diff(d['rfrac'], append=0) < 0)
                    )[0]
                else:
                    m=np.ones(len(d[x]), dtype=bool)
                if 'ne' in y:
                    ax.semilogy(d[x][m], d[y][m], alpha=0.2)
                else:
                    ax.plot(d[x][m], d[y][m], alpha=0.2)
        ax.set_title(f'BL {bg}')
        ax.set_ylim(ylims[y])
        ax.set_xlim(xlims[x])
        if inds[1] == 0:
            ax.set_ylabel(y)
        if inds[0] == nrows-1:
            ax.set_xlabel(x)
        ax.grid(which='both', linestyle='--', alpha=0.5)
    plt.suptitle(f'{plot_title} {y} vs {x}', fontsize=16)
    plt.tight_layout()


def plot_tau_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(ind, (nrows, ncols))
        ax = axes[inds]
        tau_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.where(
                    (d['r_tes'] > 1e-1) &
                    (d['rfrac'] < 0.8) &
                    (np.diff(d['rfrac'],append=0) < 0)
                )[0]
                try: 
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                tau_list.append(d['tau'][m][ind])
        ax.hist(tau_list, range=(0,10),bins=20)
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
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9, 10))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(ind, (nrows, ncols))
        ax = axes[inds]
        nep_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m = np.where(
                    (d['r_tes'] > 1e-1) &
                    (d['rfrac'] < 0.8) &
                    (np.diff(d['rfrac'],append=0) < 0)
                )[0]
                try:
                    ind = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
                except ValueError:
                    continue
                if np.abs(d['rfrac'][m][ind] - 0.5) > 0.1:
                    continue
                nep_list.append(d['nep'][m][ind])
        ax.hist(nep_list, bins=20, range=(0,100))
        med = np.nanmedian(nep_list)
        ax.axvline(med, linestyle='--', color='k', label=f'{int(med)} aW/rtHz')
        ax.set_title(f'BL {bg}')
        if inds[0] == nrows-1:
            ax.set_xlabel('NEP [aW/rtHz])')
        if inds[1] == 0:
            ax.set_ylabel('Count')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')
    plt.suptitle(f'{plot_title} NEP at 50%Rn', fontsize=16)
    plt.tight_layout()
