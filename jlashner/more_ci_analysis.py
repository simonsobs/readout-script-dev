import os
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import sys
from tqdm.auto import trange, tqdm
from scipy.optimize import curve_fit
import sodetlib as sdl


sys.path.insert(0, '/home/jlashner/repos/sotodlib')
import sotodlib.io.load_smurf as ls
ls.logger.setLevel('WARNING')

from sotodlib.core import (
    AxisManager, IndexAxis, LabelAxis
)

from sodetlib.operations import bias_steps


sys.path.append('/home/jlashner/repos/readout-script-dev/jlashner')
import new_complex_impedance as ci

reload(ci)
reload(ls)
ls.logger.setLevel('WARNING')

# Bias steps
# ['bands', 'channels', 'bgmap', 'polarity', 'edge_idxs', 'edge_signs',
# 'step_resp', 'popts', 'pcovs', 'taus', 'dets', 'biaslines', 'steps', 'samps']


NBGS = 12

arc = ls.G3tSmurf(
    '/mnt/so1/data/ucsd-k2so/timestreams/',
    db_path='/home/jlashner/indexes/k2so.db',
    meta_path='/mnt/so1/data/ucsd-k2so/smurf',
)


sw_ass = {
     'channels': ['dets'],
     'bands': ['dets'],
     'bgmap': ['dets'],
     'polarity': ['dets'],
     'Ibias': ['biaslines', 'freqs'],
     'Ites': ['dets', 'freqs'],
     'res_freqs': ['dets'],
     'Ibias_dc': ['biaslines'],
     'Zeq': ['dets', 'freqs'],
     'Vth': ['dets', 'freqs'],
     'Ztes': ['dets', 'freqs'],
     'beta_I': ['dets'],
     'L_I': ['dets'],
     'tau_I': ['dets'],
     'Rfit': ['dets'],
     'tau_eff': ['dets'],
     'bias_steps': ['dets', 'biaslines', 'steps', 'samps'],
     'ztes_rsquared': ['dets'],
     'flags': ['dets', 'samps']
}

bs_ass = {
    'bands': ['dets'],
    'channels': ['dets'],
    'bgmap': ['dets'],
    'polarity': ['dets'],
    'edge_idxs': ['biaslines', 'steps'],
    'edge_signs': ['biaslines', 'steps'],
}

def get_meta_path():
    pass

def load_sw_bsa():
    ddir = '/home/jlashner/scratch/mv11/CI_03_03'

    bsa_file = os.path.join(ddir, 'bias_step_analysis.npy')
    sw_file = os.path.join(ddir, 'trans.npy')
    ob_file = os.path.join(ddir, 'ob.npy')
    sc_file = os.path.join(ddir, 'sc.npy')

    sw = ci.CISweep.load(sw_file)
    sw.ob = ci.CISweep.load(ob_file)
    sw.ob_path = ob_file
    sw.sc = ci.CISweep.load(sc_file)
    sw.sc_path = sc_file

    bsa = bias_steps.BiasStepAnalysis.load(bsa_file)
    return sw, bsa

def dict_to_am(d, skip_bad_types=False):
    allowed_types = (str, int, float, np.ndarray)
    am = AxisManager()
    for k, v in d.items():
        if isinstance(v, allowed_types):
            am.wrap(k, v)
        elif not skip_bad_types:
            raise ValueError(
                f"Key {k} is of type {type(v)} which cannot be wrapped by an "
                 "axismanager")
    return am

def sw_to_ds(sw):
    ndets = len(sw.channels)
    nfreqs = len(sw.freqs)
    ds = AxisManager(
        LabelAxis('dets', vals=[f"r{x:0>4}" for x in range(ndets)]),
        IndexAxis('freqs', count=nfreqs),
        IndexAxis('biaslines', count=NBGS),
    )



# Now need to load in sweep and bias step dsets
# Data obj
# ['freqs', 'bgs', 'channels', 'bands', 'bgmap', 'polarity', 'Ibias', 'Ites',
# 'res_freqs', 'Ibias_dc', 'sc', 'ob', 'Zeq', 'Vth', 'Ztes', 'beta_I', 'L_I',
# 'tau_I', 'Rfit', 'tau_eff', 'bias_s teps', 'ztes_rsquared', 'flags', 'dets',
# 'freqs', 'biaslines', 'steps', 'samps']

def create_sw_ds(sw, ds=None):
    ndets = len(sw.channels)
    nfreqs = len(sw.freqs)
    if ds is None:
        # Axes: dets, freqs, biaslines
        ds = AxisManager(
            LabelAxis('dets', vals=[f"r{x:0>4}" for x in range(ndets)]),
            IndexAxis('freqs', count=nfreqs),
            IndexAxis('biaslines', count=NBGS),
        )

    for k, v in sw_ass.items():
        if hasattr(sw, k):
            ass = [(i, f) for i, f in enumerate(v)]
            val = getattr(sw, k)
            if val is None:
                continue
            ds.wrap(k, val, ass)

    return ds


def create_bs_dset(
        bsa: bias_steps.BiasStepAnalysis,
        interval=(-5e-3, 10e-3),
        show_pb=True, weight_exp=0,
        rcs=None, arc=None
    ):

    am = bsa._load_am(arc=arc)
    bsa._find_bias_edges()

    ndets = len(bsa.channels)
    nsteps = np.max([len(x) for x in bsa.edge_idxs])

    fsamp = 1./np.median(np.diff(am.timestamps))
    interval = np.array(interval)
    samp_int = (interval * fsamp).astype(int)
    nsamps = samp_int[1] - samp_int[0]

    ds = AxisManager(
        LabelAxis('dets', vals=[f"r{x:0>4}" for x in range(ndets)]),
        IndexAxis('biaslines', count=NBGS),
        IndexAxis('bias_step', count=nsteps),
        IndexAxis('samps', count=nsamps),
    )

    ass = {
        'bands': ['dets'], 'channels': ['dets'],
        'bgmap': ['dets'], 'polarity': ['dets'],
    }
    for k, v in ass.items():
        a = [(i, f) for i, f in enumerate(v)]
        ds.wrap(k, getattr(bsa, k), a)

    # Get edge idxs and stuff
    ds.wrap_new('edge_idxs', shape=('biaslines', 'bias_step'),
                cls=np.full, fill_value=-1, dtype=int)
    ds.wrap_new('edge_signs', shape=('biaslines', 'bias_step'),
                cls=np.full, fill_value=0, dtype=int)
    for bg in range(NBGS):
        ns = len(bsa.edge_idxs[bg])
        ds.edge_idxs[bg, :ns] = bsa.edge_idxs[bg]
        ds.edge_signs[bg, :ns] = bsa.edge_signs[bg]

    # Compute step response and stuff
    ds.wrap_new('step_resp', shape=('dets', 'bias_step', 2, 'samps'),
                cls=np.full, fill_value=np.nan, dtype=np.float64)
    ds.wrap_new('taus', shape=('dets', 'bias_step'),
                cls=np.full, fill_value=np.nan, dtype=np.float64)
    ds.wrap_new('popts', shape=('dets', 'bias_step', 3),
                cls=np.full, fill_value=np.nan, dtype=np.float64)
    ds.wrap_new('pcovs', shape=('dets', 'bias_step', 3, 3),
                cls=np.full, fill_value=np.nan, dtype=np.float64)
    A_per_rad = bsa.meta['pA_per_phi0'] / (2*np.pi) * 1e-12
    if rcs is None:
        rcs = np.arange(ndets)
    rcs = np.atleast_1d(rcs)
    for rc in tqdm(rcs, disable=not show_pb, desc='calc step response'):
        bg = ds.bgmap[rc]
        if bg == -1:
            continue
        for i in range(nsteps):
            ei = ds.edge_idxs[bg, i]
            es = ds.edge_signs[bg, i]
            if ei == -1:
                continue
            sl = slice(
                ei + samp_int[0], ei + samp_int[1]
            )
            ts = am.timestamps[sl]
            sig = am.signal[rc, sl] * ds.polarity[rc] * es

            # Correct timestamps and signal so (0, 0) is when the step
            # crosses it's midpoint
            s0, s1 = np.mean(sig[:-samp_int[0]]), np.mean(sig[-10:])
            sig = sig - (s0 + s1) / 2


            i0 = np.where(sig > 0)[0][-1]
            i1 = i0 + 1
            if i1 == len(sig):
                continue

            t0, t1 = ts[[i0, i1]]
            s0, s1 = sig[[i0, i1]]
            smid = 0.

            tmid = t0 + (smid - s0) / (s1 - s0) * (t1 - t0)
            ts = ts - tmid

            ds.step_resp[rc, i, 0] = ts
            ds.step_resp[rc, i, 1] = sig

            # Fit tau effective and stuff
            m = ts > 0
            p0 = (0.1, 0.001, np.ptp(sig) / 2)
            if np.sum(m) < 3:
                continue
            try:
                popt, pcov = curve_fit(
                    bias_steps.exp_fit, ts[m], sig[m],
                    sigma=ts[m]**weight_exp, p0=p0
                )
                ds.popts[rc, i] = popt
                ds.pcovs[rc, i] = pcov
                ds.taus[rc, i]  = popt[1]
            except RuntimeError:
                continue

    return ds

def wrap_bs(ds, bs):
    """
    Wraps bias steps aman in a ci aman.
    """
    idxmap = sdl.map_band_chans(
        ds.bands, ds.channels,
        bs.bands, bs.channels,
    )

    bsnew = AxisManager(
        bs.biaslines, bs.bias_step, bs.samps,
        ds.dets
    )

    if 'bs' in ds._fields:
        ds.move("bs", None)
    ds.wrap('bs', bsnew)

    for k, v in bs._assignments.items():
        if 'dets' in v:
            ds.bs.wrap(
                k, bs._fields[k][idxmap], 
                [(i, n) for i, n in enumerate(v) if n is not None]
            )
        else:
            ds.bs.wrap(k, bs._fields[k], [(i, n) for i, n in enumerate(v)])

    return ds

def plot_bias_steps(ds, rc):
    pass
