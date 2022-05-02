import numpy as np
from tqdm.auto import tqdm, trange
import os
import time
import numpy as np
import sodetlib as sdl
from scipy.signal import welch
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from sotodlib.tod_ops.filters import gaussian_filter, fourier_filter
from sotodlib.core import IndexAxis, AxisManager, OffsetAxis, LabelAxis


from pysmurf.client.base.smurf_control import SmurfControl

NBIASLINES = 12


def get_meta_am():
    pass


def _sine(ts, amp, freq, phase):
    return amp * np.sin(2*np.pi*freq*ts + phase)


def get_amps_and_phases(am, restrict_to_nperiods=None):
    nchans, nsamps = am.signal.shape
    f0, f1 = .9 * am.cmd_freq, 1.1*am.cmd_freq
    m = (f0 < am.fs) & (am.fs < f1)
    bg = np.where(am.bias_amps != 0)[0][0]
    
    idx = np.argmax(am.bias_axx[bg, m])
    f = am.fs[m][idx]

    filt = gaussian_filter(f, f_sigma=f / 30)
    am.filt_sig[:, :] = fourier_filter(am, filt)
    am.sig_amps = np.ptp(am.filt_sig, axis=1) / 2

    if restrict_to_nperiods is not None:
        dur = restrict_to_nperiods / f
        tmid = (am.timestamps[-1] - am.timestamps[0]) / 2 + am.timestamps[0]
        t0, t1 = tmid - dur / 2, tmid + dur/2
        sdl.restrict_to_times(am, t0, t1, in_place=True)

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


def get_seg_am(sw, bg, i, arc=None, nperiods=20, samps_per_period=50):
    """
    Returns an axis-manager for a given bg/bias-freq combination. This
    will do some pre-processing on the axis-manager and add several additional
    fields. This converts the ``signal`` and ``biases`` fields both to units
    of pA, and populates the following fields in the axis-manager:
     - sample_rate (float): Sample rate of data
     - cmd_freq (float): Commanded frequency of data segment
     - fs (f_idx): Array of frequencies corresponding to the amplitude spectral density
     calc
     - sig_axx (dets, f_idx): Amplitude spectral density for all channels
     - bias_axx (bias_lines, f_idx): Amplitude spectral density for all bias
     data

    It also initializes the following fields which will be populated by the
    ``get_amps_and_phases`` function

     - sig_amps (dets) : Amplitude 
    """
    sw.load_am(bg, arc=arc)

    bgi = np.where(sw.bgs == bg)[0][0]
    t0, t1 = sw.start_times[bgi, i], sw.stop_times[bgi, i]
    am = sdl.restrict_to_times(sw.am, t0, t1)
    sample_rate = 1./np.median(np.diff(am.timestamps))

    # This is pretty much required or else analysis takes too long on
    # low-freq segments due to the high sample rate
    ds_factor = int(sample_rate / sw.freqs[i] / samps_per_period)
    if ds_factor > 1:
        downsample(am, 'samps', ds_factor)
        sample_rate = 1./np.median(np.diff(am.timestamps))

    nchans = len(am.signal)

    # Convert everything to pA
    am.signal = am.signal * sw.meta['pA_per_phi0'] / (2*np.pi)

    # Index mapping from am readout channel to sweep chan index.
    chan_idxs = sdl.map_band_chans(am.ch_info.band, am.ch_info.channel,
                                   sw.bands, sw.channels)
    am.signal *= sw.polarity[chan_idxs, None]

    pA_per_bit = 2 * sw.meta['rtm_bit_to_volt']        \
        / sw.meta['bias_line_resistance']              \
        * sw.meta['high_low_current_ratio'] * 1e12

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
    am.wrap('cmd_freq', sw.freqs[i])

    # Get psds cause we'll want that
    nsamp = len(am.signal[0])
    fs, sig_pxx = welch(am.signal, fs=sample_rate, nperseg=nsamp)
    _, bias_pxx = welch(am.biases, fs=sample_rate, nperseg=nsamp)
    am.wrap('fs', fs, [(0, IndexAxis('f_idx'))])
    am.wrap('sig_axx', np.sqrt(sig_pxx), [
        (0, 'dets'), (1, 'f_idx')
    ])
    am.wrap('bias_axx', np.sqrt(bias_pxx), [
        (0, 'bias_lines'), (1, 'f_idx')
    ])

    am.wrap('sig_amps', np.zeros(nchans), [(0, 'dets')])
    am.wrap('bias_amps', np.ptp(am.biases, axis=1)/2, [(0, 'bias_lines')])
    am.wrap('sig_phases', np.zeros(nchans), [(0, 'dets')])
    am.wrap('bias_phase', 0)
    am.wrap_new('filt_sig', shape=('dets', 'samps'))

    return am


def compute_phasors(sw, bg, arc=None, num_periods=20, pbar=True):
    """
    Computes Ibias and Ites phasors for all channels on a particular bias-group
    """
    sw.load_am(bg, arc=arc)
    nchans, nfreqs = len(sw.am.signal), len(sw.freqs)
    Ibias = np.zeros(nfreqs, dtype=np.complex128)
    Ites = np.zeros((nchans, nfreqs), dtype=np.complex128)
    if isinstance(pbar, tqdm):
        pass
    else:
        pbar = tqdm(total=nfreqs, disable=(pbar!=True))
    for i in range(nfreqs):
        am = get_seg_am(sw, bg, i, arc=arc)
        get_amps_and_phases(am, restrict_to_nperiods=num_periods)
        Ites[:, i] = am.sig_amps * np.exp(1.0j * (am.sig_phases-am.bias_phase))
        Ibias[i] = am.bias_amps[bg] * np.exp(1.0j * am.bias_phase)
        pbar.update()
    return Ibias, Ites


def run_analysis(sw, bgs=None, arc=None, num_periods=20):
    nchans = len(sw.channels)
    nfreqs = len(sw.freqs)
    nbgs = len(sw.bgs)

    if bgs is None:
        bgs = sw.bgs
    bgs = np.atleast_1d(bgs)

    if not hasattr(sw, 'Ibias'):
        sw.Ibias = np.zeros((nbgs, nfreqs), dtype=np.complex128)
    if not hasattr(sw, 'Ites'):
        sw.Ites = np.zeros((nchans, nfreqs), dtype=np.complex128)
    if not hasattr(sw, 'res_freqs'):
        sw.res_freqs = np.full(nchans, np.nan)

    pbar = tqdm(total=len(bgs) * nfreqs)

    # Load DC bias values from first am if this doesn't exist
    if not hasattr(sw, 'Ibias_dc'):
        sw.Ibias_dc = np.full(nbgs, np.nan)
        pA_per_bit = 2 * sw.meta['rtm_bit_to_volt']        \
            / sw.meta['bias_line_resistance']              \
            * sw.meta['high_low_current_ratio'] * 1e12
        sw.load_am(sw.bgs[0], arc=arc)
        for bgi, bg in enumerate(sw.bgs):
            sw.Ibias_dc[bgi] = pA_per_bit * np.mean(sw.am.biases[bg])

    for bg in bgs:
        if bg not in sw.bgs:
            continue
        bgi = np.where(sw.bgs == bg)[0][0]
        sw.load_am(bg, arc=arc)
        chan_idxs = sdl.map_band_chans(
            sw.am.ch_info.band, sw.am.ch_info.channel,
            sw.bands, sw.channels
        )
        sw.res_freqs[chan_idxs] = sw.am.ch_info.frequency

        Ib, It = compute_phasors(sw, bg, arc=arc, num_periods=num_periods,
                                 pbar=pbar)
        sw.Ibias[bgi] = Ib
        sw.Ites[chan_idxs, :] = It



def _get_channel_biases(sw):
    return sw.Ibias[sw.bgidxs[sw.bgmap]]


def get_ztes(sw):
    ob, sc = sw.ob, sw.sc
    sc_idxs = sdl.map_band_chans(sc.bands, sc.channels, sw.bands, sw.channels)
    ob_idxs = sdl.map_band_chans(ob.bands, ob.channels, sw.bands, sw.channels)

    nchans, nfreqs = len(sw.channels), len(sw.freqs)
    
    # Calculates Rn
    Ib_ob = _get_channel_biases(ob)
    sw.Rn = ob.meta['R_sh'] * (np.abs(Ib_ob / ob.Ites) - 1)[ob_idxs, 0]
    sw.Rn[ob_idxs == -1] = np.nan

    # Calculate Rtes for in-transition dets
    Ib = sw.Ibias_dc[sw.bgidxs[sw.bgmap]]
    dIrat = np.real(sw.Ites[:, 0]) / np.abs(sw.Ibias[sw.bgidxs[sw.bgmap], 0])
    I0 = Ib * dIrat / (2 * dIrat - 1)
    Pj = I0 * sw.meta['R_sh'] * (Ib - I0)
    sw.Rtes = np.abs(Pj / I0**2)

    Vth = np.full((nchans, nfreqs), np.nan, dtype=np.complex128)
    Zeq = np.full((nchans, nfreqs), np.nan, dtype=np.complex128)

    Vth = 1./((1./ob.Ites - 1./sc.Ites) / sw.Rn[:, None])
    Zeq = Vth / sc.Ites
    sw.Zeq = Zeq
    sw.Vth = Vth

    sw.Ztes = Vth / sw.Ites - Zeq
    return sw.Ztes


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
    res = minimize(chi2, x0)
    if list(res.x) == list(x0):
        res.success = False

    return res


def fit_sweep_det_params(sw, x0=None, weights=None, pb=True, fmax=None):
    nchans = len(sw.channels)
    sw.beta_I = np.full(nchans, np.nan)
    sw.L_I = np.full(nchans, np.nan)
    sw.tau_I = np.full(nchans, np.nan)
    sw.Rfit = np.full(nchans, np.nan)

    for i in range(nchans):
        if sw.bgmap[i] == -1:
            continue

        res = fit_single_det_params(sw, i, fmax=fmax)
        sw.Rfit[i], sw.beta_I[i], sw.L_I[i], sw.tau_I[i] = res.x

    # Calculates remaining params
    RL = sw.meta['R_sh']
    sw.tau_eff = sw.tau_I * (1 - sw.L_I) \
        * (1 + sw.beta_I + sw.meta['R_sh'] / sw.Rfit) \
        / (1 + sw.beta_I + RL / sw.Rfit + sw.L_I * (1 - RL / sw.Rfit))


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
        mag = np.abs(sw.Ites[i]) / np.abs(sw.Ibias[sw.bgidxs[sw.bgmap[i]]])
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
    tr_sw = get_transfers(sw)
    tr_ob = get_transfers(sw.ob)
    tr_sc = get_transfers(sw.sc)

    fig, axes=  plt.subplots(1, 2, figsize=(12, 5))

    colors=['C0', 'C1', 'C2']
    labels = ['Trans', 'OB', 'SC']
    for i, tr in enumerate([tr_sw, tr_ob, tr_sc] ):
        mag = np.abs(tr[idx])
        phase = np.unwrap(np.angle(tr[idx]))
        if phase[0] < -1:
            phase += 2*np.pi
        axes[0].plot(sw.freqs, mag, color=colors[i], label=labels[i])
        axes[1].plot(sw.freqs, phase, color=colors[i], label = labels[i])

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
    zfit = Ztes_fit(fs, sw.Rfit[idx], sw.beta_I[idx], sw.L_I[idx],
                    sw.tau_I[idx]) * 1000

    ax = axes[0]
    im = ax.scatter(np.real(ztes), np.imag(ztes), c=np.log(sw.freqs))
    ax.plot(np.real(zfit), np.imag(zfit))
    ax.set_xlabel("Re[Z$_\mathrm{TES}$]", fontsize=14)
    ax.set_ylabel("Im[Z$_\mathrm{TES}$]", fontsize=14)
    fig.colorbar(im, ax=ax)

    ax = axes[1]
    ax.scatter(sw.freqs, np.real(ztes))
    ax.scatter(sw.freqs, np.imag(ztes))
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
    bgs = np.atleast_1d(bgs)

    if freqs is None:
        freqs = np.logspace(0, np.log10(4e3), 20)
    freqs = np.atleast_1d(freqs)

    if state not in ['ob', 'sc', 'transition']:
        raise ValueError("State must be 'ob', 'sc', or 'transition'")

    run_args = locals()

    # Set up initial axis manager
    dset = AxisManager()
    run_args = AxisManager()
    for k, v in run_args.items():
        if k in ['S', 'cfg']:
            continue
        run_args.wrap(k, v)
    dset.wrap('run_args', run_args)

    nfreqs = len(freqs)
    nbgs = len(bgs)

    initial_ds_factor = S.get_downsample_factor()
    initial_filter_disable = S.get_filter_disable()

    bgmap_dict = np.load(cfg.dev.exp['bgmap_file'], allow_pickle=True).item()
    bgmap_bands, bgmap_chans, bgmap_full = [
        bgmap_dict[k] for k in ['bands', 'channels', 'bgmap']
    ]

    # Only run on channels with assigned bias-groups which are already enabled.
    bands = []
    channels = []
    bgmap = []
    scale_array = np.array([S.get_amplitude_scale_array(b) for b in range(8)])
    for bg in bgs:
        m = (scale_array[bgmap_bands, bgmap_chans] > 0) & (bgmap_full == bg)
        bands.extend(bgmap_bands[m])
        channels.extend(bgmap_chans[m])
        bgmap.extend(bgmap_full[m])

    dset.wrap('biaslines', IndexAxis('biaslines', count=NBIASLINES))
    dset.wrap('segs', IndexAxis('segs', count=len(freqs)))
    dset.wrap('meta', get_meta_am())

    dset.wrap_new('start_times', ('biaslines', 'segs'), dtype=float)
    dset.wrap_new('stop_times', ('biaslines', 'segs'), dtype=float)
    dset.wrap_new('sids', ('biaslines'), dtype=int)

    dset.wrap(
        'dets', LabelAxis('dets', vals=[str(x) for x in range(len(bands))])
    )
    dset.wrap('bands', np.array(bands), [(0, 'dets')])
    dset.wrap('channels', np.array(bgmap), [(0, 'dets')])

    chans_per_band = S.get_number_channels()

    pb = tqdm(total=nfreqs*nbgs, disable=False)
    try:
        S.set_downsample_factor(1)
        S.set_filter_disable(1)
        sdl.set_current_mode(S, bgs, 1)
        tickle_voltage /= S.high_low_current_ratio
        init_biases = S.get_tes_bias_bipolar_array()

        for i, bg in enumerate(bgs):
            # We want to run with channels that are in the specified bg and are
            # enabled.
            m = dset.bgmap == bg
            channel_mask = dset.channels[m] + dset.bands[m] * chans_per_band
            dset.sids[bg] = sdl.stream_g3_on(S, channel_mask=channel_mask)
            for j, freq in enumerate(freqs):
                meas_time = min(1./freq * nperiods, max_meas_time)
                S.log(f"Tickle with bg={bg}, freq={freq}")
                S.play_sine_tes(bg, tickle_voltage, freq)
                dset.start_times[bg, j] = time.time()
                time.sleep(meas_time)
                dset.stop_times[bg, j] = time.time()
                S.set_rtm_arb_waveform_enable(0)
                S.set_tes_bias_bipolar(bg, init_biases[bg])
                pb.update()
            sdl.stream_g3_off(S)


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
