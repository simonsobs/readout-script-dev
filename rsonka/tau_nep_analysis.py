import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/rsonka/repos/readout-script-dev/rsonka')
import python_utilities as pu




def get_tau_nep_data_diff_channels(metadata_fp, bgmap_fp=None,nrows=3,ncols=4,return_dax=False):
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
    for key in ['v_bias', 'wl','tau_eff', 'Si', 'R0', 'Pj']:
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
        for key in ['v_bias','tau_eff', 'Si',  'wl', 'Pj']:
            d[key] = b_d_ax[key][idxs]
        d['R0'] = abs(b_d_ax['R0'][idxs])
        d['rfrac'] = d['R0'] / np.nanmean(d['R0'][:2]) # Problematic if bad point in first two
        d['nep'] = d['wl'] * 1e6 / np.abs(d['Si'])        
        d['r_tes'] = d.pop('R0') * 1e3
        d['s_i'] = d.pop('Si')
        d['tau'] = d.pop('tau_eff') * 1e3
        d['nei'] = d.pop('wl')
        d['p_tes'] = d.pop('Pj') * 1e12
    
    if return_dax:
        return (results_dict,b_d_ax)
    else:
        return results_dict # backwards compatibility
    
def in_transition_mask(d):
    m = np.where(
        (d['r_tes'] > 1e-1) &
        (d['rfrac'] < 0.8) &
        (np.diff(d['rfrac'], append=0) < 0)
    )[0]
    return m
    
def plot_tau_nep_data(results_dict, x, y, nrows=3, ncols=4,
                 figsize='default', plot_title='', **plot_kwargs):
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
                    m = in_transition_mask(d) # didn't originally have this as function, 
                else: #but saw you made it more complicated. And in several plotting functions.
                    m=np.ones(len(d[x]), dtype=bool)
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
    

def plot_tau_hist(results_dict, nrows=3, ncols=4, xrange=(0,10), xlim=(0,10),plot_title=''):
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
                m = in_transition_mask(d)
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
        ax.legend(fontsize='small', loc='upper right')
        ax.set_xlim(xlim)
    fig.supylabel('Count', fontsize=20)
    fig.supxlabel('tau_eff [ms]', fontsize=20)
    plt.suptitle(f'{plot_title} Tau at 50% Rn', fontsize=16)
    plt.tight_layout()
    
def plot_nep_hist(results_dict, nrows=3, ncols=4, plot_title=''):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9/4*ncols, 10/3*nrows))
    for ind, bg in enumerate(sorted(list(results_dict.keys()))):
        inds = np.unravel_index(bg, (nrows, ncols))
        print(inds)
        if nrows==1:
            ax = axes[inds[1]]
        else:
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
        try:
            ax.axvline(med, linestyle='--', color='k', label=f'{int(med)} aW/rtHz')
        except ValueError:
            pass
        ax.set_title(f'BL {bg}')
        ax.grid(which='both', linestyle='--', alpha=0.5)
        ax.legend(fontsize='small', loc='upper right')
    fig.supylabel('Count', fontsize=20)
    fig.supxlabel('tau_eff [ms]', fontsize=20)
    plt.suptitle(f'{plot_title} NEP at 50%Rn', fontsize=16)
    plt.tight_layout()
    
def plot_ptes_hist(results_dict, nrows=3, ncols=4, plot_title=''):
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
                m = in_transition_mask(d)
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