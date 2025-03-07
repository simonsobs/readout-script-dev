import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/rsonka/repos/readout-script-dev/rsonka')
import python_utilities as pu


def get_tau_nep_data(metadata_fp, bgmap_fp=None,nrows=3,ncols=4,return_dax=False):
    metadata = np.genfromtxt(metadata_fp, delimiter=",",unpack=False,
                             dtype=None, names=True, encoding=None) 
    # names=True takes first row, puts it in dtypes
    biasstep_lines = metadata[0::2]
    noise_lines = metadata[1::2]
    
    # loading the bgmap
    if bgmap_fp is None:
        # Read in bias group map from last bias step
        bsa_fp = metadata[-4][-2]
        if not os.path.isfile(bsa_fp):
            bsa_fp = bsa_fp.replace('/data/','/data2/')
        bsa = np.load(bsa_fp, allow_pickle=True).item()
        bgmap_fp = bsa['meta']['bgmap_file']
    bg_map = np.load(bgmap_fp, allow_pickle=True).item()
    bgmap_bandchan = list(zip(bg_map['bands'], bg_map['channels']))
    
    # Previewing to efficiently creat dict_axis
    bsa_objects = np.full((len(biasstep_lines),),"?",dtype=object)
    num_points = np.full((len(biasstep_lines),),-42,dtype=int)
    masks = np.full((len(biasstep_lines),),"?",dtype=object)
    for ind, line in enumerate(biasstep_lines):
        #print(f"reading biasstep line {ind}")
        bsa_fp = line[-2]
        if not os.path.isfile(bsa_fp):
            bsa_fp = bsa_fp.replace('/data/','/data2/')
        bsa = np.load(bsa_fp, allow_pickle=True).item()
        assert bsa['meta']['bgmap_file'] == bgmap_fp, \
            f"bg_map {bsa['meta']['bgmap_file']} different from supplied {bg_map}"
        bsa_objects[ind] = bsa
        bias_bandchan = list(zip(bsa['bands'], bsa['channels']))
        # If you uxm_relock between data points (perhaps due to epics),
        # the mask below is NOT the same for all biasstep_lines. 
        masks[ind] = [(b,c) in bgmap_bandchan for (b,c) in bias_bandchan]
        num_points[ind] = sum(masks[ind])
        
    # Setting up a dict_axis    
    tot_points = sum(num_points)
    b_d_ax = {'idx':np.arange(tot_points),
              'bl':np.full((tot_points,),-42,dtype=int),
              'bands':np.full((tot_points,),-42,dtype=int),
              'channels':np.full((tot_points,),-42,dtype=int),
              'sb_ch':np.full((tot_points,),-42,dtype=int)}
    for key in ['v_bias', 'wl','tau_eff', 'Si', 'R0', 'Pj','tau_std']:
        b_d_ax[key] = np.full((tot_points,),np.nan,dtype=float)
    b_d_ax['bsa'] = np.full((tot_points,),'?',dtype=object)
    
    # Now fully setup. load in the data:
    i = 0
    for ind, bsa in enumerate(bsa_objects):
        n_ind = num_points[ind]
        # biasing
        
        b_d_ax['bsa'][i:i+n_ind] = bsa
        b_d_ax['v_bias'][i:i+n_ind] = biasstep_lines['bias_v'][ind]
        for key in ['bands','channels','tau_eff', 'Si', 'R0', 'Pj']:
            b_d_ax[key][i:i+n_ind] = bsa[key][masks[ind]]
        for j in range(n_ind):
            b_d_ax['tau_std'][i+j] = np.sqrt(np.diag(bsa['step_fit_pcovs'][j]))[1]
        b_d_ax['sb_ch'][i:i+n_ind] = 1000*b_d_ax['bands'][i:i+n_ind]+b_d_ax['channels'][i:i+n_ind]
        # noise:
        noise_fp = noise_lines[ind][-2]
        if not os.path.isfile(noise_fp):
            noise_fp = noise_fp.replace('/data/','/data2/')
        noise = np.load(noise_fp, allow_pickle=True).item()
        if 'noisedict' in noise.keys():
            noise = noise['noisedict']
        b_d_ax['wl'][i:i+n_ind] = noise['noise_pars'][:,0][masks[ind]]
        i += n_ind

    """Daniel: I forget why I reformatted the data like this,
    maybe it was just to match an even earlier analysis.""" 
    # Reformatting, calculating rfrac/nep, and adding bl to axis
    results_dict = dict()
    for idx in range(len(bgmap_bandchan)):
        sb,ch = bgmap_bandchan[idx]
        bl = bg_map['bgmap'][idx]
        if bl == -1:
            continue
        if bl not in results_dict.keys():
            results_dict[bl] = dict()
        if sb not in results_dict[bl].keys():
            results_dict[bl][sb] = dict()
        if ch not in results_dict[bl][sb].keys():
            results_dict[bl][sb][ch] = dict()
        d = results_dict[bl][sb][ch]
        
        idxs = pu.dax.find_idx_matches(b_d_ax,[('bands','=',sb),
                                               ('channels','=',ch)])  
        b_d_ax['bl'][idxs] = bl
        for key in ['v_bias','tau_eff', 'Si',  'wl', 'Pj', 'tau_std']:
            d[key] = b_d_ax[key][idxs]
        d['R0'] = abs(b_d_ax['R0'][idxs])
        d['rfrac'] = d['R0'] / np.nanmean(d['R0'][:2]) # Problematic if bad point in first two
        d['nep'] = d['wl'] * 1e6 / np.abs(d['Si'])        
        d['r_tes'] = d.pop('R0') * 1e3
        d['s_i'] = d.pop('Si')
        d['tau'] = d.pop('tau_eff') * 1e3
        d['tau_std'] = d.pop('tau_std') * 1e3
        d['nei'] = d.pop('wl')
        d['p_tes'] = d.pop('Pj') * 1e12
    
    if return_dax:
        return (results_dict,b_d_ax)
    else:
        return results_dict # backwards compatibility
    
def in_transition_mask(d,old=True,tau_cut=False,nep_cut=False,confirm_r50=True):
    # Old version:
    if old:
        m = np.where(
            (d['r_tes'] > 1e-1) &
            (d['rfrac'] < 0.8) &
            (np.diff(d['rfrac'],append=0) < 0)
        )[0]
    else:
        m = np.zeros(len(d['rfrac']), dtype=bool)
        try:
            stop = np.where(d['rfrac'] < 0.2)[0][0]
            start_inds = np.where(d['rfrac'] > 0.8)[0]
            # want last start_ind < stop ind
            start = start_inds[start_inds < stop][-1]
        except IndexError:
            return np.array([]),np.nan # tells for loop to continue
        m[start:stop] = True 
        if tau_cut:
            m[d['tau']<0.25] = False
            m[(d['tau_std']/d['tau']) > 0.1] = False
        if nep_cut:
            m[d['nep'] > 1e3] = False
    if confirm_r50:
        try:
            r50_idx = np.nanargmin(np.abs(d['rfrac'][m] - 0.5))
        except ValueError:
            return np.array([]),np.nan # tells for loop to continue
        if np.abs(d['rfrac'][m][r50_idx] - 0.5) > 0.1:
            return np.array([]),np.nan # tells for loop to continue
        return m,r50_idx
    return m,np.nan

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
                m,r50_idx = in_transition_mask(d,tau_cut=True, nep_cut=True)
                if len(m) == 0:
                    continue
                transition_values['bands']+=[sb]
                transition_values['channels']+=[ch]
                transition_values['bgmap']+=[bg]
                transition_values['tau']+=[d['tau'][m][r50_idx]]
                transition_values['nep']+=[d['nep'][m][r50_idx]]

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
    
def plot_tau_nep_data(results_dict, x, y, nrows=3, ncols=4, figsize='default',
                  plot_title='', return_plot=False, old=True, **plot_kwargs):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    if figsize=='default':
        figsize = (9/4*ncols, 10/3*nrows)
    options = ['tau','s_i','r_tes','rfrac', 'nep','v_bias', 'nei', 'p_tes']
    if x not in options or y not in options:
        raise ValueError(f"x and y must each be one of {options}")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        if nrows==1:
            inds= inds[1]
        ax = axes[inds]
        n_c = 0
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                if x=='rfrac' or y=='rfrac':
                    m,_ = in_transition_mask(d, confirm_r50=False, old=old,
                            tau_cut=(('tau'==x) or ('tau'==y)),
                            nep_cut=('ne' in y))
                if len(m) == 0:
                    continue
                if 'ne' in y:
                    ax.semilogy(d[x][m], d[y][m], alpha=0.2)
                else:
                    ax.plot(d[x][m], d[y][m], alpha=0.2)
                if len(d[x][m]) >0:
                    n_c +=1
        ax.set_title(f'BL {bg}') #ax.set_title(f'BL {bg} ({n_c} chans)')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.set(**plot_kwargs)
    fig.supylabel(y, fontsize=20)
    fig.supxlabel(x, fontsize=20)
    plt.suptitle(f'{plot_title} {y} vs {x}', fontsize=16)
    plt.tight_layout()
    if return_plot:
        return fig, axes
    
def plot_tau_hist(results_dict, nrows=3, ncols=4, xrange=(0,10), xlim=(0,10),plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9/4*ncols, 10/3*nrows))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        if nrows==1:
            ax = axes[inds[1]]
        else:
            ax = axes[inds]
        tau_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m,r50_idx = in_transition_mask(d)
                if len(m) == 0:
                    continue
                tau_list.append(d['tau'][m][r50_idx])
        ax.hist(tau_list, range=xrange, bins=20)
        med = np.nanmedian(tau_list)
        ax.axvline(med, linestyle='--', color='k', label=f'{med:.3} ms')
        ax.set_title(f'BL {bg}')
        ax.legend(fontsize='small', loc='upper right')
        ax.set_xlim(xlim)
    fig.supylabel('Count', fontsize=20)
    fig.supxlabel('tau_eff [ms]', fontsize=20)
    plt.suptitle(f'{plot_title} Tau at 50% Rn', fontsize=16)
    plt.tight_layout()
    
def plot_nep_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9/4*ncols, 10/3*nrows))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        if nrows==1:
            ax = axes[inds[1]]
        else:
            ax = axes[inds]
        nep_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m,r50_idx = in_transition_mask(d)
                if len(m) == 0:
                    continue
                nep_list.append(d['nep'][m][r50_idx])
        ax.hist(nep_list, bins=20, range=(0,100))
        med = np.nanmedian(nep_list)
        try:
            ax.axvline(med, linestyle='--', color='k', label=f'{int(med)} aW/rtHz')
        except ValueError:
            pass
        ax.set_title(f'BL {bg}')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')
    fig.supylabel('Count', fontsize=20)
    fig.supxlabel('NEP [aW/rtHz]', fontsize=20)
    plt.suptitle(f'{plot_title} NEP at 50%Rn', fontsize=16)
    plt.tight_layout()
    
def plot_ptes_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    if 'data' in results_dict.keys():
        results_dict = results_dict['data']
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9/4*ncols, 10/3*nrows))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        if nrows==1:
            ax = axes[inds[1]]
        else:
            ax = axes[inds]
        ptes_list = []
        for sb in results_dict[bg].keys():
            for ch, d in results_dict[bg][sb].items():
                m,r50_idx = in_transition_mask(d)
                if len(m) == 0:
                    continue
                ptes_list.append(d['p_tes'][m][r50_idx])
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