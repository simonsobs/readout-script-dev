"""
Code related to analyzing fine sweep VNA data.
Adopted primarily from code written by hmccarrick

Example usage: python3 /home/simons/code/vna_analysis.py /data/vna/20211028_P-G-005_512_SPB-13_Mv9_Mv11/cold_device/RF1_Mv11_North -o /home/simons/data/P-G-005/UFM-Mv11 -l Mv11_North
"""

import os
import numpy as np
import lmfit
import scipy.interpolate, scipy.signal
import matplotlib.pyplot as plt
from matplotlib import ticker
from glob import glob
import pandas as pd
import argparse


def analyze_vna(
    data_dir, output_dir=None, bw=0.2,
    plot_peaks=True, plot_params=True, label='',
    return_data=False,
):
    """
    Analyze fine-sweep VNA data.
    
    Parameters
    ----------
    data_dir : str
        Absolute filepath to the directory containing fine-sweep VNA
        data for one RF chain.
    output_dir : str, default: None
        Absolute filepath to the directory for returned plots and data.
        If the directory does not exist, it will be created.
    bw : float, default: 0.2
        The bandwidth of each sweep file in GHz.

    Returns
    -------
    res_params : pandas.DataFrame
        Returned if `return_data` is True, always saved to disk as csv.
        Columns are index, resonator_index, f0, Qi, Qc, Q, br, depth.
    """
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if not os.path.isdir(data_dir):
        raise OSError(f"No such directory: '{data_dir}'.")
    tmp_files = sorted(glob(os.path.join(data_dir, "*.CSV")))
    if len(tmp_files) == 0:
        raise FileNotFoundError(f"No csv files found in {data_dir}")
    vna_files = []
    for ind, path in enumerate(tmp_files):
        file = path.split("/")[-1]
        _tmp = file.replace("-", "_").split("_")
        # The following lines are to identify the start/stop freqs from the file name.
        # If the file is named weirdly, it might crash here with a ValueError.
        try:
            start_freq_ind = _tmp.index('F') + 1
            stop_freq_ind = _tmp.index('TO') + 1
        except ValueError:
            print(f"I couldn't parse path {path}, skipping this file")
            continue
        if np.isclose(float(_tmp[stop_freq_ind]) - float(_tmp[start_freq_ind]), bw):
            vna_files.append(path)

    freq, resp = np.array([]), np.array([], dtype=complex)
    for pth in vna_files:
        now_freq, now_resp_real, now_resp_im = np.loadtxt(
            pth, skiprows=3, delimiter=',', unpack=True)
        freq = np.concatenate((freq, now_freq))
        resp = np.concatenate((resp, now_resp_real + 1j * now_resp_im))

    detrended_s21 = correct_trend(freq, resp)
    peak_inds, props = scipy.signal.find_peaks(
            -1 * detrended_s21,
            height=2,
            prominence=0.2,
        )
        

    if plot_peaks:
        plot_vna_peaks(
            freq, resp, peak_inds, suptitle=label, output_dir=output_dir,
        )
        plot_vna_muxband_peaks(
            freq, resp, peak_inds, suptitle=label, output_dir=output_dir,
        )

    res_params = fit_vna_resonances(freq, resp, peak_inds)

    if plot_params:
        plot_vna_params(res_params, suptitle=label,  output_dir=output_dir)

    if output_dir is not None:
        s21_sweep = {'freq':freq, 's21_db':20 * np.log10(np.abs(resp))}
        data_fname = "_".join([label, "vna_sweep.npy"]).strip("_")
        np.save(
            os.path.join(output_dir, data_fname), s21_sweep, allow_pickle=True,
        )
        params_fname = "_".join([label, "vna_params.csv"]).strip("_")
        res_params.to_csv(
            os.path.join(output_dir, params_fname),
        )

    if return_data:
        return res_params


def plot_vna_muxband_peaks(freq, resp, peak_inds, suptitle='', output_dir=None):
    # mux band definitions
    half_lims = np.array([
        (4.018,4.147),
        (4.151,4.280),
        (4.284,4.414),
        (4.418,4.581),
        (4.583,4.715),
        (4.717,4.848),
        (4.850,4.981),
    ])
    mux_band_lims = np.concatenate((half_lims, half_lims+1))
    muxband_colors = ['cyan','red','blue','magenta','green','yellow','brown'] * 2

    # smurf band definitons
    sb_cutoffs = [4.0,4.5,5.0,5.5,6.0]

    #Plot S21 in db
    peak_freqs = freq[peak_inds]
    fig, ax = plt.subplots(figsize=(9,4))
    for band in np.arange(6):
        if band == 4:
            fstart, fstop = 0, 4e9
            label = "Below band: {num_peaks}"
        elif band == 5:
            fstart, fstop = 6e9, np.inf
            label = "Above band: {num_peaks}"
        else:
            fstart, fstop = 4e9 + (0.5e9) * band, 4e9 + (0.5e9) * (band + 1)
            label = "SMuRF Band %s/%s: {num_peaks}" % (band, band + 4)
        inds = np.where((freq >= fstart) & (freq < fstop))
        num_peaks = len(np.where(
            (peak_freqs >= fstart) & (peak_freqs < fstop)
        )[0])

        ax.plot(
            freq[inds] * 1e-9 ,
            20 * np.log10(np.abs(resp[inds])),
            c='k',
            linewidth=1,
            label=label.format(num_peaks=num_peaks)
        )
    for i, band in enumerate(mux_band_lims):
        ax.axvspan(band[0],band[1], fc=muxband_colors[i], alpha=0.3)
        ax.text(band[0]+0.01, -29, "Band %02d" % i, color=muxband_colors[i],
                fontsize=6, weight='heavy')
    for sb, sb_f in enumerate(sb_cutoffs):
        ax.axvline(sb_f, color='k', linestyle=':')
        if sb>3:
            break
        ax.text(sb_f+0.15, 13.5, "SMuRF Band %0d/%0d" % (sb,sb+4),
                color='k', fontsize=6, weight='heavy')

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 Magnitude (dB)")
    ax.set_xticks(np.arange(4, 6.3, 0.1), minor=True)
    ax.set_ylim(-30,15)
    ax.set_xlim(3.9,6.1)
    ax.set_title("%0d resonances" % len(peak_inds))
    ax.legend(loc='upper right', ncol=3, fontsize='small')
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    if output_dir is not None:
        fname = "_".join([suptitle, "vna_muxband_peaks.png"]).strip("_")
        plt.savefig(os.path.join(output_dir, fname))


def plot_vna_peaks(freq, resp, peak_inds, suptitle='', output_dir=None):
    #Plot S21 in db
    peak_freqs = freq[peak_inds]
    fig, ax = plt.subplots(figsize=(9,4))
    for band in np.arange(6):
        if band == 4:
            fstart, fstop = 0, 4e9
            label = "Below band: {num_peaks}"
            c = 'k'
        elif band == 5:
            fstart, fstop = 6e9, np.inf
            label = "Above band: {num_peaks}"
            c = 'k'
        else:
            fstart, fstop = 4e9 + (0.5e9) * band, 4e9 + (0.5e9) * (band + 1)
            label = "Band %s/%s: {num_peaks}" % (band, band + 4)
            c = None
        inds = np.where((freq >= fstart) & (freq < fstop))
        num_peaks = len(np.where(
            (peak_freqs >= fstart) & (peak_freqs < fstop)
        )[0])

        ax.plot(
            freq[inds] * 1e-9 ,
            20 * np.log10(np.abs(resp[inds])),
            c=c,
            linewidth=1,
            label=label.format(num_peaks=num_peaks)
        )

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("S21 Magnitude (dB)")
    ax.set_xticks(np.arange(4, 6.3, 0.1), minor=True)
    ax.set_ylim(-30,30)
    ax.set_title("%0d resonances" % len(peak_inds))
    ax.legend(loc='upper right', ncol=3, fontsize='small')
    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    if output_dir is not None:
        fname = "_".join([suptitle, "vna_peaks.png"]).strip("_")
        plt.savefig(os.path.join(output_dir, fname))


def plot_vna_params(res_params, suptitle='', output_dir=None):
    fig = plt.figure(figsize=(9,6))
    ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1,0))
    ax3 = plt.subplot2grid((2, 2), (1,1))


    df = np.diff(res_params['f0']) * 1e-6
    df = df[df < 5]
    ax1.hist(df, range=(0,5), bins=25)
    ax1.set_xlabel('Resonator Spacing (MHz)')
    ax1.set_ylabel('Count')
    med = np.median(df)
    ax1.axvline(med, color='b',label="median = %.1f MHz" % med)
    ax1.legend(fontsize='medium')

    ax2.hist(res_params['depth'], range=(0,25), bins=25, label=None)
    ax2.set_xlabel('Dip Depth (dB)')
    ax2.set_ylabel('Count')
    med = np.median(res_params['depth'])
    ax2.axvline(med,color='b',label="median = %.1f" % med)
    ax2.legend(fontsize='medium')

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 
    ax3.xaxis.set_major_formatter(formatter) 
    ax3.hist(res_params['Qi'], range=(0,5e5), bins=25, label=None)
    ax3.set_xlabel('Qi')
    ax3.set_ylabel('Count')
    med = np.median(res_params['Qi'])
    ax3.axvline(med,color='b',label="median = %0d" % med)
    ax3.legend(fontsize='medium')

    plt.suptitle(suptitle)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_dir is not None:
        fname = "_".join([suptitle, "vna_params.png"]).strip("_")
        plt.savefig(os.path.join(output_dir, fname))


def s21_find_baseline(freq, s21, avg_over=800):
    """
    Average the data every avg_over points to find the baseline
    of s21.
    """
    num_points_all = s21.shape[0]
    num_2 = num_points_all % avg_over
    num_1 = num_points_all - num_2

    s21_reshaped = s21[:num_1].reshape(num_1 // avg_over, avg_over)
    freq_reshaped = freq[:num_1].reshape(num_1 // avg_over, avg_over)

    x = np.squeeze(np.median(freq_reshaped, axis=1))
    y = np.squeeze(np.amax(s21_reshaped, axis=1))

    if num_2 != 0:
        x2 = np.median(freq[num_1:num_points_all])
        y2 = np.amax(s21[num_1:num_points_all])
        x = np.append(x, x2)
        y = np.append(y, y2)

    tck = scipy.interpolate.splrep(x, y, s=0)
    ynew = scipy.interpolate.splev(freq, tck, der=0)

    return ynew


def correct_trend(freq, s21, avg_over=800):
    """
    Input the raw s21, output the s21 in db without the trend.
    """
    s21_db = 20 * np.log10(np.abs(s21))
    baseline = s21_find_baseline(freq, s21, avg_over)
    bl_db = 20 * np.log10(baseline)
    s21_corrected = s21_db - bl_db
    return s21_corrected


def get_qi(Q, Q_e_real, Q_e_imag):
    return (Q**-1 - np.real((Q_e_real+1j*Q_e_imag)**-1))**-1


def get_br(Q, f_0):
    return f_0 * (2 * Q) ** -1


def get_dip_depth(s21):
    """
    Input an array of s21, find the
    depth between maximal and minimal point.
    """
    s21_mag = np.abs(s21)
    return 20 * np.log10(max(s21_mag)) - 20 * np.log10(min(s21_mag))


def full_fit(freq, s21_complex):

    # takes numpy arrays of freq, complex s21

    argmin_s21 = np.abs(s21_complex).argmin()
    fmin, fmax = freq.min(), freq.max()
    f_0_guess = freq[argmin_s21]

    Q_min = 0.1 * (f_0_guess / (fmax - fmin))
    delta_f = np.diff(freq)
    min_delta_f = delta_f[delta_f > 0].min()
    Q_max = f_0_guess / min_delta_f
    Q_guess = np.sqrt(Q_min * Q_max)

    s21_min = np.abs(s21_complex[argmin_s21])
    s21_max = np.abs(s21_complex).max()
    Q_e_real_guess = Q_guess / (1 - s21_min / s21_max)

    A_slope, A_offset = np.polyfit(
        freq - fmin, np.abs(s21_complex), 1
    )
    A_mag = A_offset
    A_mag_slope = A_slope / A_mag

    phi_slope, phi_offset = np.polyfit(
        freq - fmin, np.unwrap(np.angle(s21_complex)), 1
    )
    delay = -phi_slope / (2 * np.pi)
    Q_i = 1 / (1 / Q_guess - 1 / Q_e_real_guess)

    # make our model
    def linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag):
        Q_e = Q_e_real + 1j * Q_e_imag
        return 1 - (Q * Q_e ** (-1) / (1 + 2j * Q * (f - f_0) / f_0))


    def cable_delay(f, delay, phi, f_min):
        return np.exp(1j * (-2 * np.pi * (f - f_min) * delay + phi))


    def general_cable(f, delay, phi, f_min, A_mag, A_slope):
        phase_term = cable_delay(f, delay, phi, f_min)
        magnitude_term = ((f - f_min) * A_slope + 1) * A_mag
        return magnitude_term * phase_term

    def resonator_cable(
        f, f_0, Q, Q_e_real, Q_e_imag, delay, phi, f_min, A_mag, A_slope
    ):
        # combine above functions into our full fitting functions
        resonator_term = linear_resonator(f, f_0, Q, Q_e_real, Q_e_imag)
        cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
        return cable_term * resonator_term

    totalmodel = lmfit.Model(resonator_cable)
    params = totalmodel.make_params(
        f_0=f_0_guess,
        Q=Q_guess,
        Q_e_real=Q_e_real_guess,
        Q_e_imag=0,
        delay=delay,
        phi=phi_offset,
        f_min=fmin,
        A_mag=A_mag,
        A_slope=A_mag_slope,
    )

    # set some bounds
    params["f_0"].set(min=freq.min(), max=freq.max())
    params["Q_e_real"].set(min=1, max=1e7)
    params["Q_e_imag"].set(min=-1e7, max=1e7)
    params["Q"].set(min=Q_min, max=Q_max)
    totalmodel.set_param_hint("Qi", min=0, expr="1/Q-1/abs(Q_e_real+1j*Q_e_imag)")

    params["phi"].set(min=phi_offset - np.pi, max=phi_offset + np.pi)

    # fit it
    result = totalmodel.fit(s21_complex, params, f=freq)
    return result


def fit_vna_resonances(
    freq, s21_complex, peak_inds,
    low_indice=[], high_indice=[], f_delta=2e5,
):
    """
    This function takes in s21 data and position of the peak,
    fits the peaks into models and outputs a dataframe of the parameters.

    You can choose to input f_delta to specify the width of the peak.
    Alternatively, you can choose to input two index arrays to specify the
    left and right bounds of the peak.
    """

    dres = {
        "resonator_index": [],
        "f0": [],
        "Qi": [],
        "Qc": [],
        "Q": [],
        "br": [],
        "depth": [],
    }
    dfres = pd.DataFrame(dres)

    for k, ind in enumerate(peak_inds):
        fs = freq[ind]
        if len(low_indice) == 0:
            mask = (freq > (fs - f_delta)) & (freq < (fs + f_delta))
        else:
            mask = np.arange(low_indice[k], high_indice[k])
        f_res = freq[mask]
        s21_res = s21_complex[mask]
        try:
            result = full_fit(f_res, s21_res)

            s21_fit = np.abs(result.best_fit)

            f0 = result.best_values["f_0"]
            Q = result.best_values["Q"]
            Qc = result.best_values["Q_e_real"]
            Qi = get_qi(result.best_values["Q"], result.best_values["Q_e_real"], result.best_values["Q_e_imag"])
            br = get_br(result.best_values["Q"], result.best_values["f_0"]) / 1.0e6
            res_index = k
            depth = get_dip_depth(result.best_fit)
            dfres = dfres.append(
                {
                    "resonator_index": int(res_index),
                    "f0": f0,
                    "Qi": Qi,
                    "Qc": Qc,
                    "Q": Q,
                    "br": br,
                    "depth": depth,
                },
                ignore_index=True,
            )
            k = k + 1
        except Exception as error:
            print(error)
            pass
    return dfres


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing fine sweep VNA data"
                        + " corresponding to one RF chain.",
                       )
    parser.add_argument("--out", "-o", type=str, required=True,
                        help="Directory to save the outputs to.",
                       )
    parser.add_argument("--label", "-l", type=str,
                        help="Label for plots and filenames.",
                       )
    args = parser.parse_args()

    plt.switch_backend('Agg')
    
    analyze_vna(
        args.data_dir, output_dir=args.out, bw=0.2,
        plot_peaks=True, plot_params=True, label=args.label,
        return_data=False,
    )
