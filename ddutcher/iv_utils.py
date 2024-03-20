import os, sys
import numpy as np
import matplotlib.pyplot as plt


def plot_yield_iv(
    metadata,
    target_bg=range(12),
    x_axis='p_tes',
    y_axis='R',
    plot_title="",
    return_data=False,
    **plot_kwargs,
):
    """
    Plots the IVs for each bias line.
    """
    if isinstance(metadata, str):
        if metadata.endswith('.csv'):
            data_dict = np.genfromtxt(metadata, delimiter=",", dtype=None, names=True, encoding=None)
            data = data_dict['data_path']
            try:
                data = [data.item()]
            except:
                pass
        elif metadata.endswith('.npy'):
            data = [metadata]
        else:
            raise NotImplementedError
    elif isinstance(metadata, dict):
        data_dict = metadata
    else:
        raise TypeError

    all_bl_iv = dict()

    if len(data) == 1:
        # if serial IV, read in the one file.
        now = np.load(data[0], allow_pickle=True).item()
        for i, bl in enumerate(target_bg):
            if bl not in all_bl_iv.keys():
                all_bl_iv[bl] = dict()
            idx = now['bgmap'] == bl
            for ind in range(np.sum(idx)):
                # converting data to resemble old format
                sb = now['bands'][idx][ind]
                chan = now['channels'][idx][ind]
                d = dict()
                for k in [
                        'R', 'R_n','R_L', 'p_tes','v_tes', 'i_tes',
                        'p_sat','si',
                        ]:
                    d[k] = now[k][idx][ind]
                d['p_tes'] *= 1e12
                d['p_sat'] *= 1e12
                d['v_bias'] = now['v_bias']
                # IV cuts
                if (d['R'][-1] < 5e-3):
                    continue
                if len(np.where(d['R'] > 15e-3)[0]) > 0:
                    continue
                if sb not in all_bl_iv[bl].keys():
                    all_bl_iv[bl][sb] = dict()
                all_bl_iv[bl][sb][chan] = d

    else:
        for bl in target_bg:
            # if per-bg IVs, read in correct file
            try:
                ind = np.where(data_dict['bias_line'] == bl)[0][0]
                fp = data[ind]
            except KeyError:
                fp = data_dict[bl]
            fp = fp.replace("/data/smurf_data/", "/data2/smurf_data/")
            if bl not in all_bl_iv.keys():
                all_bl_iv[bl] = dict()
            now = np.load(fp, allow_pickle=True).item()
            if 'data' in now.keys():
                now = now['data']
            for sb in now.keys():
                if sb not in range(8):
                    continue
                if len(now[sb].keys()) != 0:
                    all_bl_iv[bl][sb] = dict()
                for chan, d in now[sb].items():
                    # IV cuts
                    if (d['R'][-1] < 2e-3):
                        continue
                    elif np.abs(np.std(d["R"][-100:]) / np.mean(d["R"][-100:])) > 5e-3:
                        continue
                    all_bl_iv[bl][sb][chan] = d

    fig, axs = plt.subplots(3, 4,figsize=(9,9), sharex=True, sharey=True)
    tot = 0
    for bl in sorted(all_bl_iv.keys()):
        now_tot = 0
        ax = axs[np.unravel_index(bl, (3,4))]
        for sb in all_bl_iv[bl].keys():
            for ch, d in all_bl_iv[bl][sb].items():
                ax.plot(d[x_axis],d[y_axis])
                now_tot += 1
        ax.set_title("BL %s: %s TESs" % (bl, now_tot), fontsize=12)
        ax.set(**plot_kwargs)
        ax.grid(linestyle='--')

        tot += now_tot
    print(tot)

    fig.supylabel(y_axis, fontsize=20)
    fig.supxlabel(x_axis, fontsize=20)
    plt.suptitle(plot_title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if return_data:
        return all_bl_iv


def plot_iv_bgmap(metadata, tunefile, target_bg=range(12),
                  target_bands=range(8),
                  suptitle=None, return_bgmap=False):

    target_bg = np.atleast_1d(target_bg)
    target_bands = np.atleast_1d(target_bands)

    if isinstance(metadata, str):
        data_dict = np.genfromtxt(metadata, delimiter=",", dtype=None, names=True, encoding=None)
        data = data_dict['data_path']
        try:
            data = [data.item()]
        except:
            pass
    elif isinstance(metadata, dict):
        data_dict = metadata
    else:
        raise TypeError
    bgmap = {'bands':[], 'channels':[], 'bgmap':[],
             'meta' : {'tunefile' : tunefile}}
    all_bl_iv = dict()

    bl_colors = ['C0','C1','C2','C3','C4','C5']
    bl_colors += bl_colors[::-1]
    bl_labeled = [False] * 12
    if len(data) == 1:
        # if serial IV, read in the one file.
        now = np.load(data[0], allow_pickle=True).item()
        for i, bl in enumerate(target_bg):
            if bl not in all_bl_iv.keys():
                all_bl_iv[bl] = dict()
            idx = now['bgmap'] == bl
            for ind in range(np.sum(idx)):
                # converting data to resemble old format
                sb = now['bands'][idx][ind]
                chan = now['channels'][idx][ind]
                d = dict()
                for k in [
                        'R', 'R_n','R_L', 'p_tes','v_tes', 'i_tes',
                        'p_sat','si',
                        ]:
                    d[k] = now[k][idx][ind]
                d['p_tes'] *= 1e12
                d['p_sat'] *= 1e12
                d['v_bias'] = now['v_bias']
                # IV cuts
                if (d['R'][-1] < 5e-3):
                    continue
                elif len(np.where(d['R'] > 15e-3)[0]) > 0:
                    continue
                if sb not in all_bl_iv[bl].keys():
                    all_bl_iv[bl][sb] = dict()
                all_bl_iv[bl][sb][chan] = d
                bgmap['bands'] += [sb]
                bgmap['channels'] += [chan]
                bgmap['bgmap'] += [bl]
    else:
        for bl in target_bg:
            # if per-bg IVs, read in correct file
            try:
                ind = np.where(data_dict['bias_line'] == bl)[0][0]
                fp = data[ind]
            except KeyError:
                fp = data_dict[bl]
            if not os.path.isfile(fp):
                fp = fp.replace("/data/smurf_data/", "/data2/smurf_data/")
            if bl not in all_bl_iv.keys():
                all_bl_iv[bl] = dict()
    #         try:
            now = np.load(fp, allow_pickle=True).item()
    #         except:
    #            print(bl, ind, data, data[ind])
            if 'data' in now.keys():
                now = now['data']
            for sb in now.keys():
                if sb not in range(8):
                    continue
                if len(now[sb].keys()) != 0:
                    all_bl_iv[bl][sb] = dict()
                for chan, d in now[sb].items():
                    # IV cuts
                    if (d['R'][-1] < 5e-3):
                        continue
                    elif len(np.where(d['R'] > 15e-3)[0]) > 0:
                        continue
    #                 ind2 = np.where(d['p_tes'] < 0.05)[0][-1]
    #                 if len(np.where(d['R'][ind2:] > 15e-3)[0]) > 0:
    #                     continue
                    all_bl_iv[bl][sb][chan] = d
                    bgmap['bands'] += [sb]
                    bgmap['channels'] += [chan]
                    bgmap['bgmap'] += [bl]
    bgmap['bands'] = np.array(bgmap['bands'])
    bgmap['channels'] = np.array(bgmap['channels'])
    bgmap['bgmap'] = np.array(bgmap['bgmap'])

    tune = np.load(tunefile, allow_pickle=True).item()
    nrows = len(np.unique(target_bands//4))
    fig, axes = plt.subplots(nrows=nrows, figsize=(9,4*nrows))
    for smurf_band in sorted(target_bands):
        if nrows > 1:
            ax = axes[smurf_band // 4]
        else:
            ax = axes
        freqs, subbands, chans, _ = np.loadtxt(
            tune[smurf_band]['channel_assignment'], delimiter=',', unpack=True)
        ax.vlines(freqs, linestyle ='--', color='gray', linewidth=1, ymin=0, ymax=1)
        for bl in target_bg:
            if smurf_band not in all_bl_iv[bl].keys():
                continue
            if bl_labeled[bl]:
                label = None
            else:
                label = bl
                bl_labeled[bl]= True
            good_freqs = []
            for ch in all_bl_iv[bl][smurf_band].keys():
                    good_freqs += [freqs[chans == ch][0]]
            ax.vlines(good_freqs, linestyle ='-', color=bl_colors[bl], linewidth=1,
                     ymin=0, ymax=1, label=label)
    for ax in np.atleast_1d(axes):
#         handles, labels = ax.get_legend_handles_labels()
#         order = [labels.index(str(bl)) for bl in (np.arange(6) + half * 6)]

        ax.legend(
#            [handles[idx] for idx in order],["BL %s" % labels[idx] for idx in order],
            ncol=6, loc='upper center', framealpha=1,
        )
        ax.set_xlabel("Frequency (MHz)")
#         ax.set_ylabel(f"AMC {half}")
    np.atleast_1d(axes)[0].set_title("Channels with IV curves", fontsize=12)
    plt.suptitle(suptitle)
    plt.tight_layout()
    
    if return_bgmap:
        return bgmap
