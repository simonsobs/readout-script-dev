import numpy as np
import matplotlib.pyplot as plt


def plot_noise_bg(noise_fp, bgmap=None, title='', array_freq='mf'):
    if isinstance(noise_fp, dict):
        noisedict = noise_fp
    else:
        noisedict=np.load(noise_fp, allow_pickle=True).item()
    bands=noisedict['bands']
    wls = noisedict['noise_pars'][:,0]
    if isinstance(bgmap, dict):
        bgmap = bgmap
    elif bgmap is not None:
        bgmap = np.load(bgmap, allow_pickle=True).item()
    else:
        bgmap = np.load(noisedict['meta']['bgmap_file'], allow_pickle=True).item()

    bgmap = make_bgmaps_agree(bgmap, noisedict)
    
    #Plot ASDs
    fig_asd, axes_asd = plt.subplots(6, 2, figsize=(16, 12),
                             gridspec_kw={'hspace': 0})
    fig_asd.patch.set_facecolor('white')

    min_x, max_x = (1, 0)
    for bg in range(12):
        if bg in [0,1,4,5,8,9]:
            col = 0
        else:
            col = 1
        rows = [0,1,0,1,2,3,2,3,4,5,4,5]
        ax = axes_asd[rows[bg], col]
        m = bgmap['bgmap'] == bg
        med_wl = np.nanmedian(wls[m])
        f_arr = np.tile(noisedict['f'], (sum(m),1))
        x = ax.loglog(f_arr.T, noisedict['axx'][m].T, color='C0', alpha=0.1)
        ax.axhline(med_wl, color='red',  alpha=0.6,
                   label=f'Med. WL: {med_wl:.1f} pA/rtHz')
#         ax.axvline(1.4, color='k', label="1.4 Hz")
        ax.set(ylabel=f'BG {bg}\nASD (pA/rtHz)')
        ax.grid(linestyle='--', which='both')
        ax.legend(loc='lower left')
        min_x = min(ax.get_xlim()[0], min_x)
        max_x = max(ax.get_xlim()[1], max_x)

    if array_freq.lower() == 'mf':
        freq1, freq2 = '90','150'
    elif array_freq.lower() == 'uhf':
        freq1, freq2 = '220','280'
    else:
        raise ValueError(array_freq)
    axes_asd[0][0].set(title=f"{freq1} GHz")
    axes_asd[0][1].set(title=f"{freq2} GHz")
    axes_asd[-1][0].set(xlabel="Frequency (Hz)")
    axes_asd[-1][1].set(xlabel="Frequency (Hz)")
    for _ax in axes_asd:
        for ax in _ax:
            ax.set(xlim=[min_x, max_x], ylim=[1, 1e5])
    plt.suptitle(title)


def make_bgmaps_agree(bgmap, noisedict):
    if len(bgmap['channels']) > len(noisedict['channels']):
        new_bgmap = {'bands':[], 'channels':[], 'bgmap':[]}
        bandchans = [(b,ch) for b, ch in zip(noisedict['bands'], noisedict['channels'])]
        for i in np.arange(len(bgmap['channels'])):
            if (bgmap['bands'][i], bgmap['channels'][i]) in bandchans:
                for k in new_bgmap.keys():
                    new_bgmap[k].append(bgmap[k][i])
        for k,v in new_bgmap.items():
            new_bgmap[k] = np.asarray(v)
        bgmap = new_bgmap
    elif len(bgmap['channels']) < len(noisedict['channels']):
        new_bgmap = {'bands':[], 'channels':[], 'bgmap':[]}
        bandchans = [(b,ch) for b, ch in zip(bgmap['bands'], bgmap['channels'])]
        for i in np.arange(len(noisedict['channels'])):
            new_bgmap['channels'].append(noisedict['channels'][i])
            new_bgmap['bands'].append(noisedict['bands'][i])
            if (noisedict['bands'][i], noisedict['channels'][i]) in bandchans:
                new_bgmap['bgmap'].append(bgmap['bgmap'][
                    np.logical_and(
                        bgmap['channels'] == noisedict['channels'][i],
                        bgmap['bands'] == noisedict['bands'][i]
                    )][0])
            else:
                new_bgmap['bgmap'].append(-1)
        for k,v in new_bgmap.items():
            new_bgmap[k] = np.asarray(v)
        bgmap = new_bgmap
    assert((bgmap['channels'] == noisedict['channels']).all())
    
    return bgmap