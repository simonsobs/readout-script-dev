import numpy as np
from tqdm.auto import tqdm, trange
import os
import time
import numpy as np
import sodetlib as sdl
from scipy.signal import welch, hilbert
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from sotodlib.tod_ops.filters import gaussian_filter, fourier_filter 
from sotodlib.core import IndexAxis, AxisManager, OffsetAxis, LabelAxis
from sodetlib.operations.bias_steps import BiasStepAnalysis

from astropy.nddata import NDData, NDDataRef, nduncertainty

from pysmurf.client.base.smurf_control import SmurfControl


class RestrictionException(Exception):
    """Exception for when cannot restrict AxisManger properly"""

def restrict_to_times(am, t0, t1, in_place=False):
    m = (t0 < am.timestamps) & (am.timestamps < t1)
    if not m.any():
        raise RestrictionException
         
    i0, i1 = np.where(m)[0][[0, -1]] + am.samps.offset
    return am.restrict('samps', (i0, i1), in_place=in_place)

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

NBGS = 12
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

def load_tod(ds, bg, arc=None):
    if arc is not None:
        start = ds.start_times[bg, 0]
        stop = ds.stop_times[bg, -1]
        seg = arc.load_data(start, stop, show_pb=False)
    else:
        sid = ds.sid[bg]
        seg = sdl.load_session(ds.meta.stream_id, sid)
    return seg


def add_bias_steps(ds, bsa):
    if isinstance(bsa, str):
        bsa = BiasStepAnalysis.load(str)


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


def downsample(am, axis_name, ds_factor, in_place=True):
    """
    Bootleg AxisManager downsample function. Is this ok? Who knows... I hope
    it works. I think the only caveat is that offsets are not preserved.
    """
    self = am
    if in_place:
        dest = self
    else:
        dest = self.copy(axes_only=True)
        dest._assignments.update(self._assignments)

    sl = slice(None, None, ds_factor)
    count = int(np.ceil(am._axes[axis_name].count / ds_factor))
    new_ax = OffsetAxis(axis_name, count=count)
    for k, v in self._fields.items():
        if isinstance(v, AxisManager):
            dest._fields[k] = v.copy()
            if axis_name in v._axes:
                downsample(dest._fields[k], axis_name, ds_factor)
        elif np.isscalar(v) or v is None:
            dest._fields[k] = v
        else:
            sslice = [sl if n == axis_name else slice(None)
                      for n in dest._assignments[k]]
            sslice = dest._broadcast_selector(sslice)
            dest._fields[k] = v[sslice]
    dest._axes[axis_name] = new_ax
    return dest


def analyze_seg(ds, tod, bg, i, arc=None, samps_per_period=50):
    """
    Analyzes segment. Stores results in axismanager
    """
    t0, t1 = ds.start_times[bg, i], ds.stop_times[bg, i]
    am = restrict_to_times(tod, t0, t1, in_place=False)

    sample_rate = 1./np.median(np.diff(am.timestamps))

    # This is pretty much required or else analysis takes too long on
    # low-freq segments due to the high sample rate
    ds_factor = int(sample_rate / ds.freqs[i] / samps_per_period)
    if ds_factor > 1:
        downsample(am, 'samps', ds_factor)
        sample_rate = 1./np.median(np.diff(am.timestamps))

    nchans = len(am.signal)

    # Convert everything to pA
    am.signal = am.signal * ds.meta['pA_per_phi0'] / (2*np.pi)

    # Index mapping from am readout channel to sweep chan index.
    chan_idxs = sdl.map_band_chans(am.ch_info.band, am.ch_info.channel,
                                   ds.bands, ds.channels)
    am.signal *= ds.polarity[chan_idxs, None]

    pA_per_bit = 2 * ds.meta['rtm_bit_to_volt']        \
        / ds.meta['bias_line_resistance']              \
        * ds.meta['high_low_current_ratio'] * 1e12

    am.biases = am.biases * pA_per_bit

    # Remove offset from signal
    am.signal -= np.mean(am.signal, axis=1)[:, None]
    am.biases -= np.mean(am.biases, axis=1)[:, None]

    # Fix up timestamps based on frame-counter
    t0, t1 = am.timestamps[0], am.timestamps[-1]
    fc = am.primary["FrameCounter"]
    fc = fc - fc[0]
    ts = t0 + fc/fc[-1] * (t1 - t0)
    am.timestamps = ts
    am.wrap('sample_rate', sample_rate)
    am.wrap('cmd_freq', ds.freqs[i])
    
    # Get psds cause we'll want that
    nsamp = len(am.signal[0])
    fs, bias_pxx = welch(am.biases[bg], fs=sample_rate, nperseg=nsamp)

    # Gaussian filter around peak freq of bias asd.
    nchans, nsamps = am.signal.shape
    f0, f1 = .9 * am.cmd_freq, 1.1*am.cmd_freq
    m = (f0 < fs) & (fs < f1)

    idx = np.argmax(bias_pxx[m])
    f = fs[m][idx]

    filt = gaussian_filter(f, f_sigma=f / 30)
    filt = gaussian_filter(f, f_sigma=f / 5)
    filt_sig = fourier_filter(am, filt)
    am.wrap('filt_sig', filt_sig, [(0, 'dets'), (1, 'samps')])

    # Lock in amplification!
    # To get ref + ref offset by 90-deg, take Hilbert transform, and then 
    # real part gives you ref and imag part is offset by 90 deg.
    sig = filt_sig
    ref = hilbert(am.biases[bg] / np.max(am.biases[bg]))
    X = sig * ref.real
    Y = sig * ref.imag
    # We're averaging over enough periods where we don't really need to
    # restrict to an int number of periods...
    xmean = np.mean(X, axis=1)
    ymean = np.mean(Y, axis=1)
    phase = -np.arctan2(ymean, xmean)
    amp = 2*np.sqrt(xmean**2 + ymean**2)
    Ites = amp * np.exp(1.0j * phase)
    am.wrap('lockin_x', X)
    am.wrap('lockin_y', Y)
    am.wrap('Ites', Ites)

    return am

def analyze_tods(ds, bgs=None, tod=None, arc=None, show_pb=True):
    if bgs is None:
        bgs = ds.bgs
    bgs = np.atleast_1d(bgs)

    # Delete temp fields if they exist
    for f in ['_Ites', '_Ibias', '_Ibias_dc', '_res_freqs']:
        if f in ds._fields:
            ds.move(f, None)

    ds.wrap_new('_Ites', ('dets', 'steps'), cls=np.full,
                fill_value=np.nan, dtype=np.complex128)
    ds.wrap_new('_Ibias', ('biaslines',), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_Ibias_dc', ('biaslines',), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_res_freqs', ('dets',), cls=np.full, fill_value=np.nan)

    pA_per_bit = 2 * ds.meta['rtm_bit_to_volt']        \
        / ds.meta['bias_line_resistance']              \
        * ds.meta['high_low_current_ratio'] * 1e12

    ntot = len(bgs) * ds.steps.count
    pb = tqdm(total=ntot, disable=(not show_pb))
    for bg in bgs:
        if tod is None:
            pb.set_description(f"Loading tod for bg {bg}")
            _tod = load_tod(ds, bg, arc=arc)
        else:
            _tod = tod
        chmap = sdl.map_band_chans(
            _tod.ch_info.band, _tod.ch_info.channel,
            ds.bands, ds.channels
        )

        ds._Ibias_dc[bg] = np.mean(_tod.biases[bg] * pA_per_bit)
        ds._Ibias[bg] = 0.5 * np.ptp(_tod.biases[bg] * pA_per_bit)
        ds._res_freqs[chmap] = _tod.ch_info.frequency

        pb.set_description(f"Analyzing segments for bg {bg}")
        for i in range(ds.steps.count):
            try:
                seg = analyze_seg(ds, _tod, bg, i)
            except RestrictionException:
                pb.update()
                continue
            ds._Ites[chmap, i] = seg.Ites
            pb.update()
        del _tod

    for f in ['_Ites', '_Ibias', '_Ibias_dc', '_res_freqs']:
        ds.move(f, f[1:])

    return ds

def get_ztes(ds):
    ob, sc, tr = ds.ob, ds.sc, ds
    sc_idxs = sdl.map_band_chans(sc.bands, sc.channels, tr.bands, tr.channels)
    ob_idxs = sdl.map_band_chans(ob.bands, ob.channels, tr.bands, tr.channels)

    nchans, nfreqs = len(tr.channels), len(tr.freqs)

    fields = ['_Rn', '_Rtes', '_Vth', '_Zeq', '_Ztes']
    for f in fields:
        if f in ds._fields:
            ds.move(f, None)
 
    ds.wrap_new('_Rn', ('dets',))
    ds.wrap_new('_Rtes', ('dets',))
    ds.wrap_new('_Vth', ('dets',))
    ds.wrap_new('_Zeq', ('dets',))
    ds.wrap_new('_Ztes', ('dets',))
   
    # Calculates Rn
    Ib_ob = ob.Ibias[ob.bgmap][:, None]
    ds._Rn = ob.meta.R_sh * (np.abs(Ib_ob / ob.Ites) - 1)[ob_idxs, 0]
    ds._Rn[ob_idxs == -1] = np.nan

    # Calculate Rtes for in-transition dets
    Ib = tr.Ibias_dc[tr.bgmap]
    dIrat = np.real(tr.Ites[:, 0]) / np.abs(tr.Ibias[tr.bgmap])
    I0 = Ib * dIrat / (2 * dIrat - 1)
    Pj = I0 * tr.meta.R_sh * (Ib - I0)
    tr._Rtes = np.abs(Pj / I0**2)

    ds._Vth = 1./((1./ob.Ites - 1./sc.Ites) / ds._Rn[:, None])
    ds._Zeq = ds._Vth / sc.Ites

    ds._Ztes = ds._Vth / tr.Ites - ds._Zeq

    for f in fields:
        ds.move(f, f[1:])

    return True


def Ztes_fit(f, R, beta, L, tau):
    """
    Ztes equation from Irwin/Shaw eq 42 
    """
    return R * (1 + beta) \
        + R * L / (1 - L) * (2 + beta) / (1 + 2j * np.pi * f * tau)


def guess_fit_params(sw, idx):

    R = sw.Rtes[idx]

    min_idx = np.argmin(np.imag(sw.Ztes[idx]))
    tau_guess = -1./(2*np.pi*sw.freqs[min_idx])

    beta_guess = np.abs(sw.Ztes[idx, -1]) / R - 1

    Z, f = sw.Ztes[idx, min_idx], sw.freqs[min_idx]
    L_guess = 1000 # Just assume something very high
    return (beta_guess, L_guess, tau_guess)


def fit_single_det_params(sw, idx, x0=None, weights=None, fmax=None):
    R = sw.Rtes[idx]
    if x0 is None:
        x0 = guess_fit_params(sw, idx)

    if weights is None:
        weights = np.ones_like(sw.freqs)

    if fmax is not None:
        weights[sw.freqs > fmax] = 0

    def chi2(x):
        beta, L, tau = x
        zfit = Ztes_fit(sw.freqs, R, *x)
        c2 = np.nansum(weights * np.abs(sw.Ztes[idx] - zfit)**2)
        return c2

    def chi2(x):
        zfit = Ztes_fit(sw.freqs, *x)
        c2 = np.nansum(weights * np.abs(sw.Ztes[idx] - zfit)**2)
        return c2

    x0 = (R, *x0)
    # This is the default ftol, but we need it saved for uncertainty calc.
    ftol = 2.220446049250313e-09 
    res = minimize(chi2, x0, tol=ftol)
    if list(res.x) == list(x0):
        res.success = False

    # Uncertainties
    # (https://stackoverflow.com/questions/43593592/errors-to-fit-parameters-of-scipy-optimize)
    dx = np.sqrt(max(1, abs(res.fun)) * ftol * res.hess_inv.diagonal())
    return res, dx


def calc_tau_eff(ds):
    # This is the equation: (there must be a better way)
    #   tau_eff = tau_I * (1 - L_I) * (1 + beta_I + RL / Rfit) \
    #             / (1 + beta_I + RL / Rfit + L_I * (1 - RL / Rfit))
    RL = NDDataRef(ds.meta.R_sh)
    one = NDDataRef(1)
    Rfit, beta_I, L_I, tau_I =  [
        NDDataRef(ds.fit_x[:, i], 
              nduncertainty.StdDevUncertainty(ds.fit_dx[:, i]))
        for i in range(4)
    ]

    f1 = one.subtract(L_I)
    f2 = one.add(beta_I).add(RL.divide(Rfit))
    f3 = one.add(beta_I).add(RL.divide(Rfit)).add(
        L_I.multiply(one.subtract(RL.divide(Rfit)))
    )
    tau_eff = tau_I.multiply(f1).multiply(f2).divide(f3)
    for f in ['tau_eff', 'dtau_eff']:
        if f in ds._fields:
            ds.move(f, None)

    ds.wrap('tau_eff', tau_eff.data, [(0, 'dets')])
    ds.wrap('dtau_eff', tau_eff.uncertainty.array, [(0, 'dets')])
    return tau_eff

def fit_det_params(ds, x0=None, weights=None, pb=True, fmax=None):
    fields = ['_beta_I', '_L_I', '_tau_I', '_Rfit', '_tau_eff','_fit_x',
              '_fit_dx', '_fit_labels']
    for f in fields:
        if f in ds._fields:
            ds.move(f, None)

    for f in ['_beta_I', '_L_I', '_tau_I', '_Rfit', '_tau_eff']:
        ds.wrap_new(f, ('dets', ), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_fit_x', ('dets', 4), cls=np.full, fill_value=np.nan)
    ds.wrap_new('_fit_dx', ('dets', 4), cls=np.full, fill_value=np.nan)
    ds.wrap('_fit_labels', np.array(['R', 'beta_I', 'L_I', 'tau_I']))

    for i in trange(len(ds.channels), disable=(not pb)):
        if ds.bgmap[i] == -1: continue

        res, dx = fit_single_det_params(ds, i, fmax=fmax)
        ds._fit_x[i, :] = res.x
        ds._fit_dx[i, :] = dx

    for f in fields:
        ds.move(f, f[1:])

    calc_tau_eff(ds)
    return True

###########################################################################
# Plotting functions
###########################################################################
def plot_bg_transfers(sw, bgs):
    bgs = np.atleast_1d(bgs)

    fig, axes=  plt.subplots(1, 2, figsize=(12, 5))
    nchans = 0
    for i in range(len(sw.channels)):
        if sw.bgmap[i] not in bgs:
            continue
        nchans += 1
        mag = np.abs(sw.Ites[i]) / np.abs(sw.Ibias[sw.bgmap[i]])
        phase = np.unwrap(np.angle(sw.Ites[i]))
        if phase[0] < -1:
            phase += 2*np.pi
        axes[0].plot(sw.freqs, mag, color='grey', alpha=0.1)
        axes[1].plot(sw.freqs, phase, color='grey', alpha=0.1)
    axes[0].set(xscale='log', yscale='log')
    axes[0].set_ylabel("Transfer Function")
    axes[1].set_ylabel("Phase (rad)")
    for ax in axes.ravel():
        ax.set_xlabel('Freq (Hz)')

    axes[0].text(0.05, 0.05, f"Showing {nchans} chans", transform=axes[0].transAxes)
    return fig, axes

def plot_state_transfers(sw, idx):
    fig, axes=  plt.subplots(1, 2, figsize=(12, 5))

    colors=['C0', 'C1', 'C2']
    labels = ['Trans', 'OB', 'SC']
    for i, s in enumerate([sw, sw.ob, sw.sc]):
        mag = np.abs(s.Ites[idx]) / np.abs(s.Ibias[s.bgmap[idx]])
        phase = np.unwrap(np.angle(s.Ites[idx]))
        if phase[0] < -1:
            phase += 2*np.pi

        isort = np.argsort(sw.freqs)
        axes[0].plot(sw.freqs[isort], mag[isort], color=colors[i],
                     label=labels[i])
        axes[1].plot(sw.freqs[isort], phase[isort], color=colors[i], 
                     label=labels[i])

    axes[0].set(xscale='log', yscale='log')
    axes[0].set_ylabel("Transfer Function")
    axes[1].set_ylabel("Phase (rad)")
    for ax in axes.ravel():
        ax.legend()
        ax.set_xlabel('Freq (Hz)')

def plot_ztes(sw, idx, fit=True):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    fs = np.linspace(0, np.max(sw.freqs), 2000)
    ztes = sw.Ztes[idx] * 1000
    if hasattr(sw, 'fit_x'):
        zfit = Ztes_fit(fs, *sw.fit_x[idx]) * 1000
    else:
        zfit = Ztes_fit(fs, sw.Rfit[idx], sw.beta_I[idx], sw.L_I[idx],
                        sw.tau_I[idx]) * 1000

    ax = axes[0]
    im = ax.scatter(np.real(ztes), np.imag(ztes), c=np.log(sw.freqs), s=20)
    ax.plot(np.real(zfit), np.imag(zfit))
    ax.set_xlabel("Re[Z$_\mathrm{TES}$]", fontsize=14)
    ax.set_ylabel("Im[Z$_\mathrm{TES}$]", fontsize=14)
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    ax.scatter(sw.freqs, np.real(ztes), s=20)
    ax.scatter(sw.freqs, np.imag(ztes), s=20)
    ax.plot(fs, np.real(zfit), label='Real')
    ax.plot(fs, np.imag(zfit), label='Imag')
    ax.set(xscale='log')
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("Z$_\mathrm{TES}$ (mOhms)")
    ax.legend()

    b, c, bg = sw.bands[idx], sw.channels[idx], sw.bgmap[idx]
    fig.suptitle(f"Band {b}, Channel {c}, BG {bg}", fontsize=20)

    return fig, axes




###########################################################################
# Data Taking Functions
###########################################################################

@sdl.set_action()
def take_complex_impedance(
        S, cfg, bgs, freqs=None, state='transition', nperiods=500,
        max_meas_time=20., tickle_voltage=0.005):
    """
    Takes a complex impedance sweep. This will play sine waves on specified
    bias-groups over the current DC bias voltage. This returns a CISweep object.
        Ib = sw.Ibias[bg_idxs[sw.bgmap[i]]]
        transfer = np.abs(sw.Ites[i] / Ib)
        phase = np.unwrap(np.angle(sw.Ites[i] / Ib))

    Args
    ----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    freqs : array, optional
        List of frequencies to sweep over.
    state : str
        Current detector state. Must be 'ob', 'sc', or 'transition'
    nperiods : float
        Number of periods to measure for at each frequency. If the meas_time 
        ends up larger than ``max_meas_time``, ``max_meas_time`` will be used
        instead. This makes it so we don't spend unreasonably long amounts of
        time at higher freqs.
    max_meas_time : float
        Maximum amount of time to wait at any given frequency
    tickle_voltage : float
        Tickle amplitude in low-current-mode volts.
    """
    if state not in ['ob', 'sc', 'transition']:
        raise ValueError("State must be 'ob', 'sc', or 'transition'")

    bgs = np.atleast_1d(bgs)

    if freqs is None:
        freqs = np.logspace(0, np.log10(4e3), 20)
    freqs = np.atleast_1d(freqs)

    run_kwargs = {
        'bgs': bgs, 'freqs': freqs, 'nperiods': nperiods,
        'max_meas_time': max_meas_time, 'tickle_voltage': tickle_voltage,
    }
    nfreqs = len(freqs)
    nbgs = len(bgs)
    start_times = np.zeros((nbgs, nfreqs), dtype=float)
    stop_times = np.zeros((nbgs, nfreqs), dtype=float)
    sids = np.zeros(nbgs, dtype=int)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()

    bgmap_dict = np.load(cfg.dev.exp['bgmap_file'], allow_pickle=True).item()
    bgmap_bands, bgmap_chans, bgmap = [
        bgmap_dict[k] for k in ['bands', 'channels', 'bgmap']
    ]

    bands = []
    channels = []
    scale_array = np.array([S.get_amplitude_scale_array(b) for b in range(8)])

    pb = tqdm(total=nfreqs*nbgs, disable=False)
    try:
        S.set_downsample_factor(1)
        S.set_filter_disable(1)
        sdl.set_current_mode(S, bgs, 1)
        tickle_voltage /= S.high_low_current_ratio

        init_biases = S.get_tes_bias_bipolar_array()

        for i, bg in enumerate(bgs):
            # We want to run with channels that are in the specified bg
            # and are enabled.
            m = (scale_array[bgmap_bands, bgmap_chans] > 0) & (bgmap == bg)
            channel_mask = bgmap_bands[m] * S.get_number_channels() + bgmap_chans[m]
            bands.extend(bgmap_bands[m])
            channels.extend(bgmap_chans[m])

            sids[i] = sdl.stream_g3_on(S, channel_mask=channel_mask)
            for j, freq in enumerate(freqs):
                meas_time = min(1./freq * nperiods, max_meas_time)
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                start_times[i, j] = time.time()
                time.sleep(meas_time)
                stop_times[i, j] = time.time()
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, init_biases[bg])
                pb.update()
            sdl.stream_g3_off(S)


        bands = np.array(bands)
        channels = np.array(channels)
        sweep = CISweep(S, cfg, run_kwargs, sids, start_times, stop_times,
                        bands, channels, state)
        sweep.save()
    finally:
        sdl.set_current_mode(S, bgs, 0)
        S.set_downsample_factor(initial_ds_factor)
        S.set_filter_disable(initial_filter_disable)
        sdl.stream_g3_off(S)

    return sweep

def take_complex_impedance_ob_sc(S, cfg, bgs, overbias_voltage=19.9,
                                 tes_bias=15.0, overbias_wait=5.0,
                                 cool_wait=30., **ci_kwargs):
    """
    Takes overbiased and superconducting complex impedance sweeps. These are
    required to analyze any in-transition sweeps.

    Args
    -----
    S : SmurfControl
        Pysmurf Instance
    cfg : DetConfig
        Det config instance
    bgs : array, int
        List of bias groups to run on
    overbias_voltage : float
        Voltage to use to overbias detectors
    tes_bias : float
        Voltage to set detectors to after overbiasing
    overbias_wait : float
        Time to wait at the overbias_voltage
    cool_wait : float
        Time to wait at the tes_bias after overbiasing
    **ci_kwargs : 
        Any additional kwargs will be passed directly to the
        ``take_complex_impedance`` function.
    """

    # Takes SC sweep
    for bg in bgs:
        S.set_tes_bias_bipolar(bg, 0)
    sc = take_complex_impedance(S, cfg, bgs, state='sc', **ci_kwargs)

    S.overbias_tes_all(bias_groups=bgs, overbias_voltage=overbias_voltage,
                       tes_bias=tes_bias, overbias_wait=overbias_wait,
                       cool_wait=cool_wait)
    ob = take_complex_impedance(S, cfg, bgs, state='ob', **ci_kwargs)

    cfg.dev.update_experiment({
        'complex_impedance_sc_path': sc.filepath,
        'complex_impedance_ob_path': ob.filepath
    }, update_file=True)

    return sc, ob
