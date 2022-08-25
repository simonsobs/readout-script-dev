import matplotlib
matplotlib.use('Agg')
from sodetlib.det_config import DetConfig
from sodetlib.operations import squid_curves as sc
import os

import sodetlib as sdl
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.auto import trange, tqdm
from scipy import signal
from pysmurf.client.util.pub import set_action

def plot_fixed_tone_profile(res, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(res['ft_freqs'], res['psd_i'], '.', alpha=0.8, label='digital I')
    ax.plot(res['ft_freqs'], res['psd_q'], '.', alpha=0.8, label='digital Q')
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(f"Noise at {res['offset_freq']} Hz [dBc/Hz]")
    title = ', '.join([
        f"tone_power: {res['tone_power']}",
        f"att_dc: {res['att_dc']}",
        f"att_uc: {res['att_uc']}",
    ])
    ax.set_title(title)
    ax.legend()
    return fig, ax

def setup_fixed_tone_array(S, cfg, tones_per_band=250, tone_power=13,
                           bands=None, show_pb=True):
    if bands is None:
        bands = np.arange(8)
    bands = np.atleast_1d(bands)
    center_freqs_mhz = 4250 + 500 * bands

    ft_freqs = np.hstack([
        np.linspace(cfreq - 249.5, cfreq + 250, tones_per_band, endpoint=False)
        for cfreq in center_freqs_mhz
    ])
    nfts = len(ft_freqs)
    ft_bands = np.zeros(nfts, dtype=int)
    ft_chans = np.zeros(nfts, dtype=int)

    for i, f in enumerate(tqdm(ft_freqs, disable=(not show_pb))):
        ft_bands[i], ft_chans[i] = S.set_fixed_tone(f, tone_power)

    return ft_freqs, ft_bands, ft_chans


@set_action()
def fixed_tone_profile_sweep(S, cfg, ft_freqs, ft_bands, ft_chans, tone_powers,
                             atts, **kwargs):
    res_array = []
    for tp in tone_powers:
        for att in atts:
            res = fixed_tone_profile(S, cfg, ft_freqs, ft_bands, ft_chans,
                                     tone_power=tp, att_uc=att, att_dc=att,
                                     **kwargs)
            res_array.append(res)
    return res_array

@set_action()
def fixed_tone_profile(S, cfg, ft_freqs=None, ft_bands=None, ft_chans=None,
                       offset_freq=30e3, meas_tones_per_band=10, nsamps=2**20,
                       nperseg=2**18,
                       tone_power=13, att_uc=6, att_dc=6, bands=None,
                       show_pb=True):
    """
    Enables fixed-tones accross specified bands, and measures the I/Q psd
    value at a specified frequency for a selection of them.

    Args
    -----
    S : SmurfControl
        Pysmurf control instance
    cfg

    """

    if ft_freqs is None:
        ft_bands = []
        ft_chans = []
        ft_freqs = []
        for band in range(8):
            asa = S.get_amplitude_scale_array(band)
            m = asa != 0
            ft_bands.extend([band for _ in range(np.sum(m))])
            ft_chans.extend(np.where(m)[0])

            freqs = S.get_center_frequency_array(band) \
                    + S.get_tone_frequency_offset_mhz(band) \
                    + S.get_band_center_mhz(band)
            ft_freqs.extend(freqs[m])
        ft_bands = np.array(ft_bands)
        ft_freqs = np.array(ft_freqs)
        ft_chans = np.array(ft_chans)
    nfts = len(ft_freqs)

    tones_per_band = int(len(ft_freqs)/len(np.unique(ft_bands)))
    ft_step = int(tones_per_band / meas_tones_per_band)
    take_meas = np.zeros(nfts, dtype=int)
    take_meas[::ft_step] = 1

    for b in range(8):
        asa = S.get_amplitude_scale_array(b)
        asa[asa != 0] = tone_power
        S.set_amplitude_scale_array(b, asa)
        S.set_att_uc(b, att_uc)
        S.set_att_dc(b, att_dc)

    psd_i = np.full(nfts, np.nan, dtype=float)
    psd_q = np.full(nfts, np.nan, dtype=float)

    fsamp = S.get_channel_frequency_mhz() * 1e6
    
    S.log(f"Taking measurements on {np.sum(take_meas)} channels")
    for i in trange(nfts, disable=(not show_pb)):
        if not take_meas[i]:
            continue

        sig_i, sig_q, sync = S.take_debug_data(
            band=ft_bands[i], channel=ft_chans[i], rf_iq=True, nsamp=nsamps
        )
        ffi, pxxi = signal.welch(sig_i, fs=fsamp, nperseg=nperseg)
        ffq, pxxq = signal.welch(sig_q, fs=fsamp, nperseg=nperseg)

        # scale to dBc/Hz by the voltage magnitude
        magfac = np.mean(sig_q)**2 + np.mean(sig_i)**2
        pxxi_dbc = 10. * np.log10(pxxi/magfac)
        pxxq_dbc = 10. * np.log10(pxxq/magfac)

        # get the index for offset
        freq_idx = np.where(ffi >= offset_freq)[0][0]
        psd_i[i] = pxxi_dbc[freq_idx]
        psd_q[i] = pxxq_dbc[freq_idx]

    res = {
        'ft_freqs': ft_freqs,
        'ft_bands': ft_bands,
        'ft_chans': ft_chans,
        'take_meas': take_meas,
        'psd_i': psd_i,
        'psd_q': psd_q,
        'offset_freq': offset_freq,
        'tone_power': tone_power,
        'att_uc': att_uc,
        'att_dc': att_dc,
    }

    fname = sdl.make_filename(S, 'fixed_tone_profile.npy')
    np.save(fname, res)
    S.pub.register_file(fname, 'fixed_tone_profile', format='npy')


    fig, ax = plot_fixed_tone_profile(res)
    fname = sdl.make_filename(S, 'fixed_tone_profile.png', plot=True)
    fig.savefig(fname)
    S.pub.register_file(fname, 'fixed_tone_profile', format='png', plot=True)


    return res

def shut_down_slot(slot):
    cfg = DetConfig()
    cfg.load_config_files(slot=slot)
    S = cfg.get_smurf_control(make_logfile=False)

    S.all_off()
    S.C.write_ps_en(0)
    S.flux_ramp_off()


def check_tes_bias(S, bg, hcm):
    for _bg in range(12):
        S.set_tes_bias_bipolar(_bg, 0)

    S.set_tes_bias_bipolar(bg, 1)
    if hcm:
        S.set_tes_bias_high_current(bg)
    else:
        S.set_tes_bias_low_current(bg)

def check_amp_bias(S, i):
    amps = ['50k1', '50k2', 'hemt1', 'hemt2']
    for a in amps:
        S.set_amp_gate_voltage(a, 0)
        S.set_amp_drain_voltage(a, 0)

    a = amps[i//2]
    if i % 2 == 0:
        print(f"Setting {a} gate to 1")
        S.set_amp_gate_voltage(a, 1)
    else:
        if 'hemt' in a:
            v = 1
        else:
            v = 4
        print(f"Setting {a} drain to {v}")
        S.set_amp_drain_voltage(a, v)


def set_ft_at_band_centers(slot):
    cfg = DetConfig()
    cfg.load_config_files(slot=slot)
    S = cfg.get_smurf_control(make_logfile=False)

    print("Band off")
    for b in range(8):
        S.band_off(b)
        S.set_att_uc(b ,0)
        S.set_att_dc(b ,0)

    print("Enabling fixed tones")
    for band in range(8):
        S.set_fixed_tone(S.get_band_center_mhz(band), 12)


def check_optical_on_off(S, band):
    """Checks band response with and without optical amps"""
    S.set_att_uc(band, 30)
    S.set_att_dc(band, 30)
    S.C.write_optical(0b11)
    time.sleep(0.2)
    f0, f1=-2e8, 2e8
    freqs, resp = S.full_band_resp(band)
    m = (f0 < freqs) & (freqs < f1)
    plt.plot(freqs[m], np.abs(resp)[m])
    S.C.write_optical(0b00)
    time.sleep(0.2)
    freqs, resp = S.full_band_resp(band)
    m = (f0 < freqs) & (freqs < f1)
    plt.plot(freqs[m], np.abs(resp)[m])


def take_squid_curve_cross(S, cfg, wait_time=0.1, Npts=4, Nsteps=500,
                     bands=None, channels=None, lms_gain=None, out_path=None,
                     run_analysis=True, analysis_kwargs=None, show_pb=False,
                     run_serial_ops=True, frac_full_scale_max=0.3, fr_smurfs=None):
    """
    Takes data in open loop (only slow integral tracking) and steps through flux
    values to trace out a SQUID curve. This can be compared against the tracked
    SQUID curve which might not perfectly replicate this if these curves are
    poorly approximated by a sine wave (or ~3 harmonics of a fourier expansion).

    Args
    ----
    S : ``pysmurf.client.base.smurf_control.SmurfControl``
        ``pysmurf`` control object
    cfg : ``sodetlib.det_config.DeviceConfig``
        device config object.
    wait_time: float
        how long you wait between flux step point in seconds
    Npts : int
        number of points you take at each flux bias step to average
    Nsteps : int
        Number of flux points you will take total.
    bands : int, list
        list of bands to take dc SQUID curves on
    channels : dict
        default is None and will run on all channels that are on
        otherwise pass a dictionary with a key for each band
        with values equal to the list of channels to run in each band.
    lms_gain : int
        gain used in tracking loop filter and set in ``tracking_setup``
        defaults to ``None`` and pulls from ``det_config``
    out_path : str, filepath
        directory to output npy file to. defaults to ``None`` and uses pysmurf
        plot directory (``S.plot_dir``)
    run_serial_ops : bool
        If true, will run serial grad descent and eta scan
    frac_full_scale_max : float
        Max value of fraction full scale to use.
    Returns
    -------
    data : dict
        This contains the flux bias array, channel array, and frequency
        shift at each bias value for each channel in each band. As well as
        the dictionary of fitted values returned by fit_squid_curves if
        run_analysis argument is set to True.
    """
    cur_mode = S.get_cryo_card_ac_dc_mode()
    if cur_mode == 'AC':
        S.set_mode_dc()
    ctime = S.get_timestamp()
    if out_path is None:
        out_path = os.path.join(S.output_dir, f'{ctime}_fr_sweep_data.npy')

    # This calculates the amount of flux ramp amplitude you need for 1 phi0
    # and then sets the range of flux bias to be enough to achieve the Nphi0s
    # specified in the fucnction call.
    if bands is None:
        bands = np.arange(8)
    if channels is None:
        channels = {}
        for band in bands:
            channels[band] = S.which_on(band)

    bias_peak = frac_full_scale_max

    # This is the step size calculated from range and number of steps
    bias_step = np.abs(2*bias_peak)/float(Nsteps)
    if bands is None:
        bands = np.arange(8)
    if channels is None:
        channels = {}
        for band in bands:
            channels[band] = S.which_on(band)

    channels_out = []
    bands_out = []
    for band in bands:
        channels_out.extend(channels[band])
        bands_out.extend(list(np.ones(len(channels[band]))*band))
    biases = np.arange(-bias_peak, bias_peak, bias_step)

    # final output data dictionary
    data = {}
    data['meta'] = sdl.get_metadata(S, cfg)
    data['bands'] = np.asarray(bands_out)
    data['channels'] = np.asarray(channels_out)
    data['fluxramp_ffs'] = biases
    data['res_freq_vs_fr'] = []

    unique_bands = np.unique(np.asarray(bands_out, dtype=int))
    prev_lms_enable1 = {}
    prev_lms_enable2 = {}
    prev_lms_enable3 = {}
    prev_lms_gain = {}

    # take SQUID data
    try:
        for band in unique_bands:
            band_cfg = cfg.dev.bands[band]
            if lms_gain is None:
                lms_gain = band_cfg['lms_gain']
            S.log(f'{len(channels[band])} channels on in band {band},'
                  ' configuring band for simple, integral tracking')
            S.log(
                f'-> Setting lmsEnable[1-3] and lmsGain to 0 for band {band}.')
            prev_lms_enable1[band] = S.get_lms_enable1(band)
            prev_lms_enable2[band] = S.get_lms_enable2(band)
            prev_lms_enable3[band] = S.get_lms_enable3(band)
            prev_lms_gain[band] = S.get_lms_gain(band)
            S.set_lms_enable1(band, 0)
            S.set_lms_enable2(band, 0)
            S.set_lms_enable3(band, 0)
            S.set_lms_gain(band, lms_gain)

        fs = {}
        S.log(
            '\rSetting flux ramp bias to 0 V\033[K before tune'.format(-bias_peak))
        set_flux_ramp_one_or_all(S, 0.0, fr_smurfs=fr_smurfs)

        for band in unique_bands:
            fs[band] = []
            if run_serial_ops:
                S.run_serial_gradient_descent(band)
                S.run_serial_eta_scan(band)
            S.toggle_feedback(band)

        small_steps_to_starting_bias = np.arange(
            -bias_peak, 0, bias_step)[::-1]

        # step from zero (where we tuned) down to starting bias
        S.log('Slowly shift flux ramp voltage to place where we begin.')

        for b in small_steps_to_starting_bias:
            set_flux_ramp_one_or_all(S, b, fr_smurfs=fr_smurfs)
            time.sleep(wait_time)

        # make sure we start at bias_low
        S.log(f'\rSetting flux ramp bias low at {-bias_peak} V')
        set_flux_ramp_one_or_all(S, -bias_peak, fr_smurfs=fr_smurfs, do_config=False)
        #S.set_fixed_flux_ramp_bias(-bias_peak, do_config=False)
        time.sleep(wait_time)

        S.log('Starting to take flux ramp.')

        for b in tqdm(biases, disable=(not show_pb)):
            set_flux_ramp_one_or_all(S, b, fr_smurfs=fr_smurfs,
                                     do_config=False)
            #S.set_fixed_flux_ramp_bias(b, do_config=False)
            time.sleep(wait_time)
            for band in unique_bands:
                fsamp = np.zeros(shape=(Npts, len(channels[band])))
                for i in range(Npts):
                    fsamp[i, :] = S.get_loop_filter_output_array(band)[
                        channels[band]]
                fsampmean = np.mean(fsamp, axis=0)
                fs[band].append(fsampmean)

        S.log('Done taking flux ramp data.')
        fres = []
        for i, band in enumerate(unique_bands):
            fres_loop = S.channel_to_freq(band).tolist()
            fres.extend(fres_loop)
            # stack
            lfovsfr = np.dstack(fs[band])[0]
            fvsfr = np.array(
                [arr+fres for (arr, fres) in zip(lfovsfr, fres_loop)])
            if i == 0:
                data['res_freq_vs_fr'] = fvsfr
            else:
                data['res_freq_vs_fr'] = np.concatenate(
                    (data['res_freq_vs_fr'], fvsfr), axis=0)

        data['res_freq'] = np.asarray(fres)
        data['filepath'] = out_path

    # always zero and restore state of system
    finally:
        # done - zero and unset
        #S.set_fixed_flux_ramp_bias(0, do_config=False)
        set_flux_ramp_one_or_all(S, 0, fr_smurfs=fr_smurfs, do_config=False)
        #S.set_fixed_flux_ramp_bias(-bias_peak, do_config=False)
        time.sleep(wait_time)

        S.unset_fixed_flux_ramp_bias()
        if fr_smurfs is not None:
            for _S in fr_smurfs.values():
                _S.unset_fixed_flux_ramp_bias()

        for band in unique_bands:
            S.set_lms_enable1(band, prev_lms_enable1[band])
            S.set_lms_enable2(band, prev_lms_enable2[band])
            S.set_lms_enable3(band, prev_lms_enable3[band])
            S.set_lms_gain(band, lms_gain)

        if cur_mode == 'AC':
            S.set_mode_ac()

    if run_analysis:
        if analysis_kwargs is not None:
            fit_dict = sc.fit_squid_curves(data, **analysis_kwargs)
        else:
            fit_dict = sc.fit_squid_curves(data)
        data['df'] = fit_dict['df']
        data['dfdI'] = fit_dict['dfdI']
        data['higher_harmonic_power'] = fit_dict['higher_harmonic_power']
        data['ffs_per_phi0'] = fit_dict['model_params'][:, 0]
        data['popts'] = fit_dict['model_params']
#       plt_txt is an array of strings that are to be used with plot_squid_fit
        data['plt_txt'] = fit_dict['plt_txt']

    # save dataset for each iteration, just to make sure it gets
    # written to disk
    np.save(out_path, data)
    S.pub.register_file(out_path, 'dc_squid_curve', format='npy')

    return data

def set_flux_ramp_one_or_all(S, b, fr_smurfs=None, **kwargs):
        if fr_smurfs is None:
            S.set_fixed_flux_ramp_bias(b, **kwargs)
        else:
            for _S in fr_smurfs.values():
                _S.set_fixed_flux_ramp_bias(b, **kwargs)


if __name__ == '__main__':
    cfg = DetConfig()
    cfg.load_config_files(slot=os.environ['SLOT'])
    S = cfg.get_smurf_control(make_logfile=False)

    def set_fts():
        print("Bands off")
        for b in range(8):
            S.band_off(b)
        ft_freqs, ft_bands, ft_chans = setup_fixed_tone_array(
            S, cfg, tone_power=12,
        )

    def run_fixed_tone_profile():
        print("Running fixed tone profiling")

        fixed_tone_profile(S, cfg, ft_freqs=None, ft_bands=None, ft_chans=None,
                           meas_tones_per_band=20,
                           tone_power=13, att_uc=6, att_dc=6)

    def run_find_freq():
        for band in range(8):
            print(f"Running find freq on band {band}")
            S.find_freq(band, tone_power=12, make_plot=True, save_plot=True)

    # run_find_freq()
    set_fts()
