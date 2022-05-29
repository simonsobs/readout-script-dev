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

### Random stuff from old CI code that I don't want to delete

def sw_to_dset(sw):
    ndets = len(sw.channels)
    nfreqs = len(sw.freqs)
    ds = AxisManager(
        LabelAxis('dets', vals=[f"{x:0>4}" for x in range(ndets)]),
        IndexAxis('biaslines', count=NBGS),
        IndexAxis('steps', count=nfreqs)
    )

    ds.wrap('meta', dict_to_am(sw.meta))
    ds.wrap('run_kwargs', dict_to_am(sw.run_kwargs))
    ds.wrap('freqs', sw.freqs, [(0, 'steps')])
    ds.wrap('bgs', sw.bgs)
    ds.wrap_new('start_times', ('biaslines', 'steps'), 
                cls=np.full, fill_value=np.nan)
    ds.wrap_new('stop_times', ('biaslines', 'steps'), 
                cls=np.full, fill_value=np.nan)
    ds.wrap_new('sid', ('biaslines',), cls=np.zeros, dtype=int)

    for i, bg in enumerate(sw.bgs):
        ds.start_times[bg] = sw.start_times[i]
        ds.stop_times[bg] = sw.stop_times[i]
        ds.sid[bg] = sw.sid[i]

    ds.wrap('bands', sw.bands, [(0, 'dets')])
    ds.wrap('channels', sw.channels, [(0, 'dets')])
    ds.wrap('bgmap', sw.bgmap, [(0, 'dets')])
    ds.wrap('polarity', sw.polarity, [(0, 'dets')])
    ds.wrap('state', sw.state)
    ds.wrap('ob_path', sw.ob_path)
    ds.wrap('sc_path', sw.sc_path)
    return ds

class CISweep:
    """
    Complex impedance overview class.

    Attributes
    -----------
    Ibias : np.ndarry
        Phasors created from the commanded bias data for each bias-group.  The
        amplitude is the amplitude of the sine-wave on the bias line (in pA),
        and the phase is the phase relative to the start time of the
        freq-segment after restricting the axis-manager to nperiods.
    Ites : np.ndarray
        Phasor of the TES current, with the amp being the amplitude of the
        sine wave of the TES current (pA), and phase being the phase of the
        sine wave relative commanded bias-signal. (relative to the Ibias phase)
    Ztes : np.ndarray
        Array of z-tes
    """
    def __init__(self, *args, **kwargs):
        self.bg_loaded = None
        self.am = None

        if len(args) > 0:
            self.initialize(*args, **kwargs)

    def initialize(self, S, cfg, run_kwargs, sid, start_times, stop_times,
                   bands, channels, state):
        self._S = S
        self._cfg = cfg
        self.meta = sdl.get_metadata(S, cfg)
        self.run_kwargs = run_kwargs
        self.freqs = run_kwargs['freqs']
        self.bgs = run_kwargs['bgs']
        self.tickle_voltage = run_kwargs['tickle_voltage']
        self.start_times = start_times
        self.stop_times = stop_times
        self.sid = sid
        self.bias_array = S.get_tes_bias_bipolar_array()
        self.bands = bands
        self.channels = channels
        self.nchans = len(channels)
        self.nfreqs = len(self.freqs)
        self.nbgs = len(self.bias_array)

        # Load bgmap data
        self.bgmap, self.polarity = sdl.load_bgmap(self.bands, self.channels,
                                                   self.meta['bgmap_file'])

        self.ob_path = cfg.dev.exp.get('complex_impedance_ob_path')
        self.sc_path = cfg.dev.exp.get('complex_impedance_sc_path')
        self.state = state

    def load_am(self, bg, arc=None):
        if self.bg_loaded == bg:
            return self.am

        bgi = np.where(self.bgs == bg)[0][0]
        if arc is not None:
            start = self.start_times[bgi, 0]
            stop = self.stop_times[bgi, -1]
            self.am = arc.load_data(start, stop, show_pb=False)
        else:
            sid = self.sid[bgi]
            self.am = sdl.load_session(self.meta['stream_id'], sid)

        self.bg_loaded = bg
        return self.am

    @classmethod 
    def load(cls, path): 
        """ 
        Loads a CISweep object from file
        """
        self = cls()
        for k, v in np.load(path, allow_pickle=True).item().items():
            setattr(self, k, v)

        self.filepath = path

        # Re-initializes some important things that may not be saved in the
        # output file

        # bg-indexes
        self.bgidxs = np.full(12, -1, dtype=int)
        for i, bg in enumerate(self.bgs):
            self.bgidxs[bg] = i

        return self 


def save(self, path=None):
    saved_fields = [
        'meta', 'run_kwargs', 'freqs', 'bgs', 'start_times', 'stop_times',
        'sid', 'bias_array', 'bands', 'channels', 'ob_path',
        'sc_path', 'state', 'bgmap', 'polarity',

    ]
    data = {k: getattr(self, k) for k in saved_fields}

    results = [
        'Ibias', 'Ites', 'res_freqs', 'Ztes', 'Ibias_dc',
        'Rn', 'Rtes', 'Zeq', 'Vth', 'beta_I', 'L_I', 'tau_I',
        'Rfit', 'tau_eff'
    ]
    for field in results: 
        # These can be empty if analysis hasn't happened
        data[field] = getattr(self, field, None)

    if path is not None:
        np.save(path, data, allow_pickle=True)
        self.filepath = path
        return path
    else:
        filepath = sdl.validate_and_save(
            'ci_sweep.npy', data, S=self._S, cfg=self._cfg, register=True)
        self.filepath = filepath
        return filepath


def _sine(ts, amp, freq, phase):
    return amp * np.sin(2*np.pi*freq*ts + phase)


def get_amps_and_phases(am, restrict_to_nperiods=None):
    bg = 0
    f0, f1 = .9 * am.cmd_freq, 1.1*am.cmd_freq
    m = (f0 < am.fs) & (am.fs < f1)
    idx = np.argmax(am.bias_axx[bg, m])
    f = am.fs[m][idx]
    f = am.fs[m][idx]
    nchans, nsamps = am.signal.shape
    am.sig_amps = np.ptp(am.filt_sig, axis=1) / 2

    if restrict_to_nperiods is not None:
        dur = restrict_to_nperiods / f
        tmid = (am.timestamps[-1] - am.timestamps[0]) / 2 + am.timestamps[0]
        t0, t1 = tmid - dur / 2, tmid + dur/2
        restrict_to_times(am, t0, t1, in_place=True)

    phis = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    bias_corr = np.zeros(len(phis))
    sig_corr = np.zeros((nchans, len(phis)))

    for i, phi in enumerate(phis):
        template = _sine(am.timestamps, 1, f, phi)
        bias_corr[i] = np.sum(template * am.biases[bg])
        sig_corr[:, i] = np.sum(template[None, :] * am.filt_sig, axis=1)

    if 'phis' not in am._fields:
        am.wrap('phis', phis, [(0, IndexAxis('phase_idx'))])
        am.wrap('bias_corr', bias_corr, [(0, 'phase_idx')])
        am.wrap('sig_corr', sig_corr, [(0, 'dets'), (1, 'phase_idx')])
    else:
        am.phis = phis
        am.bias_corr = bias_corr
        am.sig_corr = sig_corr

    am.sig_freq = f
    am.bias_phase = phis[np.argmax(bias_corr)]
    am.sig_phases = phis[np.argmax(sig_corr, axis=1)]


