# tes_yield.py
'''
Check TES yield by taking bias tickle (from sodetlib) and IV curves.
Display quality in biasability, 50% RN target V bias, Psat and Rn.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import time
import glob
import csv
import pysmurf.client
from sodetlib.det_config  import DetConfig
from sodetlib.operations.iv import take_iv
from sodetlib.operations.bias_steps import take_bgmap
from pysmurf.client.util.pub import set_action
import logging

logger = logging.getLogger(__name__)

def tickle_and_iv(
        S, target_bg, bias_high, bias_low, bias_step,
        bath_temp, start_time, current_mode, make_bgmap,
):
    target_bg = np.array(target_bg)
    save_name = '{}_tes_yield.csv'.format(start_time)
    tes_yield_data = os.path.join(S.output_dir, save_name)
    logger.info(f'Saving data to {tes_yield_data}')
    out_fn = os.path.join(S.output_dir, tes_yield_data) 

    if make_bgmap:
        bsa = take_bgmap(S, cfg, bgs=target_bg, show_plots=False)

    fieldnames = ['bath_temp', 'bias_line', 'band', 'data_path','notes']
    with open(out_fn, 'w', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    if current_mode.lower() in ['high, hi']:
        high_current_mode = True
        bias_high /= S.high_low_current_ratio
        bias_low /= S.high_low_current_ratio
        bias_step /= S.high_low_current_ratio
    else:
        high_current_mode = False

    row = {}
    row['bath_temp'] = str(bath_temp)
    row['bias_line'] = 'all'
    row['band'] = 'all'

    logger.info(f'Taking IV on bias lines serially, all smurf bands.')

    iva = take_iv(
        S, cfg,
        bias_groups=target_bg, wait_time=0.01, bias_high=bias_high,
        overbias_wait=2, bias_low=bias_low, bias_step=bias_step,
        overbias_voltage=12, cool_wait=30, high_current_mode=high_current_mode,
        show_plots=False,run_serially=True, serial_wait_time=30,
    )
    dat_file = iva.filepath
    row['data_path'] = dat_file
    with open(out_fn, 'a', newline = '') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

    return out_fn


@set_action()
def tes_yield(S, target_bg, out_fn, start_time):
    data_dict = np.genfromtxt(out_fn, delimiter=",", dtype=None, names=True, encoding=None)
    
    data = np.atleast_1d(data_dict['data_path'])
    
    good_chans = 0
    all_data_IV = dict()
    now = np.load(data[0], allow_pickle=True).item()
    for i, bl in enumerate(target_bg):
        if bl not in all_data_IV.keys():
            all_data_IV[bl] = dict()

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
            if (d['R'][-1] < 5e-3):
                continue
            elif len(np.where(d['R'] > 15e-3)[0]) > 0:
                continue
            if sb not in all_data_IV[bl].keys():
                all_data_IV[bl][sb] = dict()
            all_data_IV[bl][sb][chan] = d

    S.pub.register_file(out_fn, "tes_yield", format='.csv')

    operating_r = dict()
    # operating_r will contain list of achieved rfracs for each vbias
    target_vbias_dict = dict()
    # target_vbias_dict will hold one vbias value for each bias line.
    target_rfrac = 0.5
    all_rn = []
    for bl in target_bg:
        target_v_bias = []
        operating_r[bl] = dict()
        for sb in all_data_IV[bl].keys():
            if len(all_data_IV[bl][sb].keys()) == 0:
                continue
            first_chan = next(iter(all_data_IV[bl][sb]))
            # v_bias the same for all ch on one bl, different for different bl
            v_biases = all_data_IV[bl][sb][first_chan]['v_bias']
            for ind, v in enumerate(v_biases):
                v = np.round(v,3)
                if v not in operating_r[bl].keys():
                    operating_r[bl][v] = []
                for chan, d in all_data_IV[bl][sb].items():
                    operating_r[bl][v].append(d['R'][ind]/d['R_n'])
            # This finds where R/Rn crosses target_rfrac; the corresponding
            # vbias value is then added to a list. The median of that list
            # is then used as the optimal vbias for a bias line.
            for ch, d in all_data_IV[bl][sb].items():
                rfrac = d['R']/d['R_n']
                cross_idx = np.where(
                    np.logical_and(
                        rfrac - target_rfrac >= 0,
                        np.roll(rfrac - target_rfrac, 1) < 0
                    )
                )[0]
                if len(cross_idx) == 0:
                    continue
                all_rn.append(d['R_n'])
                v_bias = round(d['v_bias'][cross_idx][-1], 3)
                target_v_bias.append(v_bias)

        med_target_v_bias = np.nanmedian(np.array(target_v_bias))
        # if the list length is even, median is average of two values.
        # I don't want that.
        if med_target_v_bias not in operating_r[bl].keys():
            med_target_v_bias = v_biases[
                np.argmin(np.abs(v_biases-med_target_v_bias))
            ]

        target_vbias_dict[bl] = round(med_target_v_bias, 3)

    target_vbias_fp = os.path.join(S.output_dir, f"{start_time}_target_vbias.npy")
    np.save(target_vbias_fp, target_vbias_dict, allow_pickle=True)
    S.pub.register_file(target_vbias_fp, "tes_yield", format='npy')

    fig, axs = plt.subplots(6, 4,figsize=(25,30), gridspec_kw={'width_ratios': [2, 1,2,1]})
    tes_total = 0
    for ind, bl in enumerate(target_bg):
        ax_rv = axs[bl//2, bl%2*2]
        if np.isnan(target_vbias_dict[bl]):
            continue
        if len(operating_r[bl].keys()) == 0:
            continue
        count_num = 0
        for sb in all_data_IV[bl].keys():
            for ch,d in all_data_IV[bl][sb].items():
                ax_rv.plot(d['v_bias'], d['R'], alpha=0.6)
                count_num += 1
        tes_total += count_num

        ax_rv.set_xlabel('V_bias [V]')
        ax_rv.set_ylabel('R [Ohm]')
        ax_rv.grid()
        ax_rv.axhspan(0.2*np.nanmedian(all_rn), 0.9*np.nanmedian(all_rn),
                      facecolor='gray', alpha=0.2)
        ax_rv.axvline(target_vbias_dict[bl], linestyle='--', color='gray')
        ax_rv.set_title('bl {}, yield {}'.format(bl,count_num))
        ax_rv.set_ylim([-0.001,0.012])

        ax_vb = axs[bl//2,bl%2*2+1]
        thisbl_vbias = target_vbias_dict[bl]
        try:
            to_plot = operating_r[bl][thisbl_vbias]
        except KeyError as e:
            print(thisbl_vbias, operating_r[bl].keys())
            raise e
        h = ax_vb.hist(to_plot, range=(0,1), bins=40)
        ax_vb.axvline(
            np.median(to_plot),
            linestyle='--',
            color='gray',
        )
        ax_vb.set_xlabel("percentage Rn")
        ax_vb.set_ylabel("{} TESs total".format(count_num))
        ax_vb.set_title("optimal Vbias {}V for median {}Rn".format(
            target_vbias_dict[bl],
            round(np.median(operating_r[bl][target_vbias_dict[bl]]), 3))
        )

    plt.suptitle(f"TES total yield: {tes_total}")
    save_name = os.path.join(S.plot_dir, f'{start_time}_IV_yield.png')
    logger.info(f'Saving plot to {save_name}')
    plt.savefig(save_name)

    S.pub.register_file(save_name, "tes_yield", plot=True)

    fig, axs = plt.subplots(6, 4, figsize=(25,30))
    tes_total = 0
    target_rfrac = 0.9
    for bl in target_bg:
        count_num = 0
        Rn = []
        psat = []
        for sb in all_data_IV[bl].keys():
            for ch, d in all_data_IV[bl][sb].items():
                rfrac = d['R']/d['R_n']
                cross_idx = np.where(
                    np.logical_and(
                        rfrac - target_rfrac >= 0,
                        np.roll(rfrac - target_rfrac, 1) < 0
                    )
                )[0]
                if len(cross_idx) == 0:
                    continue
                now_psat = d['p_tes'][cross_idx][-1]
                Rn.append(d['R_n'])
                psat.append(now_psat)
                count_num += 1
        tes_total += count_num

        ax_psat = axs[bl//2,bl%2*2]
        ax_psat.set_xlabel('P_sat (pW)')
        ax_psat.set_ylabel('count')
        ax_psat.grid()
        ax_psat.hist(psat, range=(0,15), bins=50,histtype= u'step',linewidth=2,color = 'r')
        ax_psat.axvline(np.median(psat), linestyle='--', color='gray')
        ax_psat.set_title('bl {}, yield {} median Psat {:.2f} pW'.format(
            bl,count_num,np.median(psat))
        )
        
        ax_rn = axs[bl//2,bl%2*2+1]
        
        h = ax_rn.hist(Rn, range=(0.005,0.01), bins=50, histtype= u'step',linewidth=2,color = 'k')
        ax_rn.axvline(np.median(Rn),linestyle='--', color='gray')
        ax_rn.set_xlabel("Rn (Ohm)")
        ax_rn.set_ylabel('count')
        ax_rn.set_title('bl {}, median Rn {:.4f} Ohm'.format(bl,np.median(Rn)))

    plt.suptitle(f"TES total yield: {tes_total}")
    save_name = os.path.join(S.plot_dir, f'{start_time}_IV_psat.png')
    logger.info(f'Saving plot to {save_name}')
    logger.info(f"TES total yield: {tes_total}")
    plt.savefig(save_name)

    S.pub.register_file(save_name, "tes_yield", plot=True)

    return target_vbias_dict


def run(S, cfg, bias_high=20, bias_low=0, bias_step=0.025, bath_temp=100,
        current_mode='low', make_bgmap=False):
    start_time = S.get_timestamp()
    target_bg = range(12)

    out_fn = tickle_and_iv(
        S, target_bg, bias_high, bias_low, bias_step, bath_temp, start_time,
        current_mode, make_bgmap)
    target_vbias = tes_yield(S, target_bg, out_fn, start_time)
    logger.info(f'Saving data to {out_fn}')
    return target_vbias


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--temp', type=str,
                        help="For record-keeping, not controlling,"
    )
    parser.add_argument('--bias-high', type=float, default=19)
    parser.add_argument('--bias-low', type=float, default=0)
    parser.add_argument('--bias-step', type=float, default=0.025)
    parser.add_argument('--current-mode', type=str, default='low')
    parser.add_argument('--make-bgmap', default=False, action='store_true')
    parser.add_argument(
        "--loglevel",
        type=str.upper,
        default=None,
        choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
        help="Set the log level for printed messages. The default is pulled from "
        +"$LOGLEVEL, defaulting to INFO if not set.",
    )

    cfg = DetConfig()
    args = cfg.parse_args(parser)
    if args.loglevel is None:
        args.loglevel = os.environ.get("LOGLEVEL","INFO")
    numeric_level = getattr(logging, args.loglevel)
    logging.basicConfig(
        format="%(levelname)s: %(funcName)s: %(message)s", level=numeric_level
    )

    S = cfg.get_smurf_control(make_logfile=(numeric_level != 10))

    S.load_tune(cfg.dev.exp['tunefile'])

    run(S, cfg, bias_high=args.bias_high, bias_low=args.bias_low,
        bias_step=args.bias_step, bath_temp=args.temp, current_mode=args.current_mode,
        make_bgmap=args.make_bgmap
    )
