# uxm_setup_optimize_confluence.py
# Use:
# (not meant for calling on command line)
# Sets up a confluence-formatted log of setup/optimize, OR polishes one with noise

import time
import os, sys
import numpy as np
import scipy.signal as signal
from sodetlib.operations import uxm_setup as op
import sodetlib.util as su
from uc_tuner import UCTuner
import logging
import matplotlib
matplotlib.use('Agg')

sys.path.append("/readout-script-dev/ddutcher")
sys.path.append("readout-script-dev/rsonka/uxm_setup_optimize")
from optimize_fracpp import optimize_fracpp

logger = logging.getLogger(__name__)



def start_confluence_log_file(S,cfg,bands):
    """
    Start a file summary of the uxm_optimize data in 
    Confluence markdown format for easy copy+paste to Confluence
    and the UFM dashboard.

    Parameters
    ----------
    S :
        Pysmurf control instance
    cfg :
        DetConfig instance
    bands : array-like
        List of SMuRF bands being optimized

    Returns
    --------
    filepath to write further summary data to.
    """
    # RS: making this file store & confluence-markdown-format your data 
    if bands is None:
        bands = S.config.get("init").get("bands")
    confluence_fp = os.path.join(S.output_dir,f"{S.name}_optimization_summary.txt")
    with open(confluence_fp,'a') as cfile:
        dev_name, crate_and_slot, start_date = get_config_vals(S,cfg)
        cfile.write(f"h4. *{dev_name} {crate_and_slot}*\n")
        #cfile.write(f"optimization with `{__file__}")
        cfile.write("* Ran {{" + f"{' '.join(sys.argv)}" +"}}\n")
        band_str = ','.join([str(band) for band in bands])
        cfile.write(f"* Plots of bands {band_str} taken {start_date} in " +\
                    "{{" +f"{S.plot_dir}" +"}}\n")
        cfile.write("* resultant tunefile: **TODO**\n\n")
        cfile.write("|| ||Indiv.||-||-||-||-||togeth||-||\n")
        table_top="||SMuRF band||uc att (.5dBs)||tone power (3dB steps)||"+\
            "dc att (.5dBs)||Num. Channels||Med. White Noise (pA/rtHz)||"+\
            "Num. Channels||Med. White Noise (pA/rtHz)||\n"
        cfile.write(table_top)
    logger.info(f"made new confluence summary at:\n{confluence_fp}")
    return confluence_fp


def get_config_vals(S,cfg):
    """"
    Fetches the device name, crate&slot, and start date of the
    data taking from S and cfg.
    
    Parameters
    ----------
    S :
        Pysmurf control instance
    cfg :
        DetConfig instance

    Returns
    --------
    (device_name, crate_and_slot, start_date) as strings."""
    dev_name = cfg.dev_file.split('/')[-1].replace('_',' ')[:-5]
    crate_and_slot = 'TODO crate and slot'
    start_date = 'TODO date'
    try: # RS: someone could theoretically use different file structure
        plot_dir_list = S.plot_dir.split('/')
        crate_and_slot = plot_dir_list[-3].replace("crate", "Crate ").replace("slot"," Slot ")
        start_date = time.ctime(int(plot_dir_list[-2]))
    except:
        pass
    return (dev_name, crate_and_slot, start_date)

def append_opt_band(S, low_noise_thresh, med_noise_thresh, confluence_fp,opt_band, 
    uctuner):
    """ 
    Appends the solo-band noise summary data to a confluence-markdown style 
    summary txt file.
    Parameters
    ----------
    S :
        Pysmurf control instance
    low_noise_threshold : float
        Italicize noise results below this goal.
    med_noise_thresh : float
        Bold noise results above this baseline threshold (failing).
    confluence_fp : str
        filepath to append to
    opt_band : int
        Which band this data is from
    uc_tuner : UCTuner instance
        Where we get the summary data from. 
    """
    with open(confluence_fp,'a') as cfile: # Making the markdown string to append to file
        top="||SMuRF band||uc att (.5dB steps)||tone power (3dB steps)||dc att (.5dB steps)||Num. Channels||Med. White Noise (pA/rtHz)||Num. Channels||Med. White Noise (pA/rtHz)||"
        fmt = ""
        if uctuner.best_wl <= low_noise_thresh:
            fmt='_'
        elif uctuner.best_wl >= med_noise_thresh:
            fmt='*'
        confluence_row = f"|{opt_band}|{uctuner.best_att}|{uctuner.best_tone}|"+\
                         f"{S.get_att_dc(opt_band)}|{uctuner.best_length}|" +\
                         f"{fmt}{uctuner.best_wl}{fmt}| | |\n"
        cfile.write(confluence_row)
        if opt_band == 7:
            cfile.write("||Total|| || || || || || TODO || ||\n ")

def start_or_continue(S,cfg,bands,confluence_fp='default'):
    """
    Start a new file summary of the uxm_optimize data in 
    Confluence markdown format for easy copy+paste to Confluence
    and the UFM dashboard, OR add the start data for a continuation
    to an old one. Note the continuation isn't fully functional yet.

    Parameters
    ----------
    S :
        Pysmurf control instance
    cfg :
        DetConfig instance
    bands : array-like
        List of SMuRF bands being optimized
    confluence_fp :
        filepath of old confluence, or 

    Returns
    --------
    filepath to write further summary data to.
    """
    if confluence_fp=='default':
        return start_confluence_log_file(S,cfg,bands)
    continue_config(S,cfg,bands,confluence_fp)
    
        
# NOTE: RS: I didn't finish the below; you will have to manually append
# and bands taken after a crash. 

def continue_config(S,cfg,bands,confluence_fp):
    """
    Incomplete function that would add new noise
    data (probably taken after a crash) to an older
    file. Would be important for later adding the 
    all bands data automatically.""""
    with open(confluence_fp,mode='r') as old_file:
         old_str = old_file.read()
    old_lines = old_file.split("\n")
    check_continuation(S,cfg,bands,old_lines)
    
    
def check_continuation(S,cfg,bands,old_lines): 
    dev_name, crate_and_slot, start_date = get_config_vals(S,cfg)
    new_header = f"h4. **{dev_name} {crate_and_slot}**\n"
    if not old_lines[0] == new_header:
        logger.info(f"CONFL WARNING: given {new_header} != continued {old_lines[0]}")
    old_bands = []
    for line in old_lines:
        if len(line)>2 and line[0] == "|" and not line[1] == "|":
            old_bands.append(int(line.split("|")[1]))
    for band in old_bands:
        if band in bands:
            logger.info(f"CONFL WARNING: old file had requested band {band}!")
    
    





   

