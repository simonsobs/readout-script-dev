#!/usr/bin/env python3
'''
python 3

author: rsonka
desc: collection of classes and methods related to mux map loading,
bath ramps, and cold load ramp analysis

Code necessary for constructing the class are methods.
Other code as separate functions so reloading isn't necessary.
Due to legacy usage, many functions have associated methods that just call them.

FILE Layout: 
Global Functions/Values
    class Test_Device:
Test_Device 

'''

# NOTE! remember you have to use importlib's reload() to get a new version of 
# these classes in JupyterLab/ipython.

import os, sys
from glob import glob
import numpy as np
import scipy.optimize
from scipy.optimize import curve_fit
import scipy.signal
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import SPB_opt.SPB_opt_power as spb_opt # Also, think about name. 
from datetime import datetime
import copy
import math
import warnings
from pathlib import Path
import random as rand

sys.path.append('/home/rsonka/repos/readout-script-dev/rsonka')
import python_utilities as pu
from python_utilities.numpy_plotting import make_filesafe as make_filesafe

# raw loading
import sodetlib.det_config
from sodetlib.legacy.analysis import det_analysis #If this is having problems, check which kernel you're using and sys.path. 
from scipy.interpolate import interp1d

mpl.use('nbAgg') # can remove this line when Daniel updates the system sodetlib install so that import optimize_params no longer switches the plotting backend.
default_figsize =  (5,5) # firefox (2.5,2.5) # chrome (5,5) # nowadays firefox 5,5 too


#2345678901234567890123456789012345678901234567890123456789012345678901234567890
# =======80 characters =========================================================


# ==============================================================================
# ------------------------- Global functions/values ----------------------------
# ==============================================================================


# ============ Support functions 

# pulled in from my pthon utils:
'''
make_filesafe    (<-direct import)

is_float(num_string)
try_to_numerical(num_string)
round_to_sf(x, sf)
round_to_sf_str(x, sf,sci_at=4)'''

is_float = pu.is_float
try_to_numerical = pu.try_to_numerical
round_to_sf = pu.round_to_sf
round_to_sf_str = pu.round_to_sf_str

def ml(mlstr):
    '''Alias of str_to_match_list for useful typing reduction'''
    return str_to_match_list(mlstr)

def str_to_num_or_bool(strv):
    if strv == 'False':
        return False
    if strv == 'True':
        return True
    return try_to_numerical(strv)

def str_to_match_list(mlstr):
    ''''NOTE: IF UPDATE, must update the separate class version!
    I got sick of typing so much for match list.
    Converts a string match_keymatch_typematch_val|...
    to [(match_key,match_type,match_val),...],
    may struggle with list match_vals if list includes comma,
    will fail if you ever put a match_type in the axis keys.
    In those cases use the long form match_list instead.'''
    if mlstr == "":
        return []
    if not type(mlstr) == str:
        return mlstr
    mstrs = mlstr.split('&')
    match_list = []
    for mstr in mstrs:
        match_type = '='
        for mt in ['>=','<=','>','<','!=']:
            if mt in mstr:
                match_type = mt
                break
        match_key, val_str = mstr.split(match_type)
        if val_str[0] == '[' and val_str[-1] == ']':
            val_list = val_str[1:-1].split(",")
            matchy = (match_key,match_type,
                      [str_to_num_or_bool(val) for val in val_list])
        else:
            matchy = (match_key,match_type,
                       str_to_num_or_bool(val_str))
        match_list.append(matchy)
    return match_list

# ==============================================================================
# --------------------------- Test_Device Support ------------------------------
# ==============================================================================
kai2d= {'smurf_freq' : 'freq',
        'vna_freq' : '',
        'design_freq' : '',
        'index' : 'mux_index',  #1-65, in order of increasing frequency, on a mux chip
        'mux_band' : '**SPEC**', # Daniel lists order of mux chip appearance (see assembly) under mux_chip, but this is not the band 
        'pad' : 'mux_pad',
        'mux_posn' : 'mux_chip', # since LF has it's own.
        'biasline' : 'bias_group',
        'pol' : '**SPEC**', #'pol_ang', # but Daniel has numbers (-1 for D I think?) -> A, B, D. More than two numbers too. Apparently that's part of Kaiwen's copper_map_corrected.csv
        'TES_freq' : '**SPEC**', # from dan. TES_id. #(T/B/D) -> # or NC
        'det_row' : '',
        'det_col' : '',
        'rhomb' : '',
        'opt' : '**SPEC**', #   -> 0 for does not see light, 1 for does see light (and is optical fab, Ithink)
        'det_x' : 'x_mm',
        'det_y' : 'y_mm'}

def dan_to_kai_mux_quick(dan_fp,out_fp): # **TODO**: VERY DIRTY!
    alph = 'abcdefghijklmnopqrstuvwxyz'
    kaiwen_arr = [ac.orig_map_atts]
    daniel_mux = np.load(dan_fp,allow_pickle=True).item()
    i = 0
    for sb in daniel_mux.keys():
        for ch, d in daniel_mux[sb].items():
            if len(d.keys()) > 1:
                to_add = np.full((18,),'?',dtype=object)
                to_add[0] = str(sb)
                if pu.is_float(ch):
                    to_add[1] = str(ch)
                else: # Daniel separates -1's with -1.a, -1.b.... I like the idea but need ints, so:
                    to_add[1] = str(-8*1000 -1 - alph.find(ch[3:])) # -8*512 to make sb_ch work
                i = 2
                for key in ac.orig_map_atts[2:]:
                    val = kai2d[key]
                    if val == '':
                        to_add[i] = '?'
                    elif not val == "**SPEC**":
                        to_add[i] = str(d[val])
                    else:
                        t_id = str(d['tes_id'])
                        if key == 'mux_band':
                            to_add[i] = '?' #  **TODO**: can get from mux_chip
                        elif key == 'pol':
                            if t_id[-1] == 'T':
                                to_add[i] = 'A'
                            else:
                                to_add[i] = t_id[-1]
                            #print(to_add[i])
                        elif key == 'TES_freq':
                            if d['bonded'] == 'N': # **TODO**: NOT SURE THIS CORRECT
                                to_add[i] = 'NC'
                            else:
                                to_add[i] = t_id[:-1]
                        elif key == 'opt':
                            if t_id[-1] == 'D':
                                to_add[i] = '0.0'
                            else:
                                to_add[i] = '1.0'
                    #print(f"{key} {to_add[i]}")
                    i+=1
                kaiwen_arr.append(to_add)
    with open(out_fp,'w') as f:
        for row in kaiwen_arr:
            f.write(",".join(row)+"\n")



# ==============================================================================
# --------------------------- Test_Device Class --------------------------------
# ==============================================================================

#### Should be able to write to/load from a config file of file names
#### Saving plots also a good idea. 

# This has to be global for the function that updates mux_map
orig_map_atts = """smurf_band, smurf_chan, smurf_freq, vna_freq, design_freq, index, 
         mux_band, pad, mux_posn, biasline, pol, TES_freq, det_row, det_col, 
         rhomb, opt, det_x, det_y""".replace("\n","").replace(" ","").split(",")

map_atts = """smurf_band, smurf_chan, smurf_freq, vna_freq, design_freq, index, 
         mux_band, pad, mux_posn, biasline, pol, TES_freq, det_row, det_col, 
         rhomb, opt, det_x, det_y, det_r, masked, OMT""".replace("\n","").replace(" ","").split(",")

class Test_Device:
    '''A class for representing a UMM, UFM, SPB, mux box, 512 box, or other such
    device. Really anything we're testing through SMuRF. 
    It stores the datafiles associated with that device thus far and the data 
    structures calculated from them for analysis purposes. 
    Eventually it should be able to read/write (at least) the datafiles to 
    human-readable config file. 
    
    # =============================== VARIABLES ================================
    # --------- REQUIRED Vars: ---------
    dName    # Used in plots and filenames.
    out_path  # to the directory that contains the files it writes out. 
    # Config file = outPath/dName_config.(?) <-TODO
    # Config file should note the path to its full saved data if such exists. 
    # Possibly add "number of mux" expected?
    mux_map_fp       # FILEPATH In Kaiwen's format
    
    # --------- Other OPTIONAL Vars: ---------
    device_type      # MF_UFM <default>, UHF_UFM, LF_UFM;  add SPB someday; 
    masked_biaslines # array. Used to update mux maps of Kaiwen's format to have 
                     # opt account for masking. Class var so can be referenced 
                     # on detectors  that weren't mapped, had bl found later.
    bl_freq_dict     # {biasline:TES_freq}. Overrides the mux map. 
    mux_chips        # 28/#mux-length list of 1 if mux chip present, 0 if thru
                     # REMEMBER: mux chip position numbers are serpentine:
                     # run 0 at top left to 2 at top right, then down one to 3, 
                     # then left to 4, 5, then top left of the first 4-mux chip
                     # row is position 6, then right to 7, etc.
                     # Mv15 assembly doc has some of the pos. #s. 
    bonded_TESs      # List of TESs by biasline. Used for %yield calcs.  
                     # defaults to being constructed by mux_chips if not provided.
    
    # --------- Mapping ---------
    num_bl           # set by device_type
    num_mux_slots    # set by device_type; THEORETICAL #mux, not actual 
    tes_freqs        # Expected TES freqs in device.
    base_mux_map     # the loaded array
    orig_map_atts    # Listing Kaiwen's headers (set in above global var):
        # smurf_band, smurf_chan, smurf_freq, vna_freq, design_freq, index, 
        # mux_band, pad, mux_posn, biasline, pol, TES_freq, det_row, det_col,
        # rhomb, opt, det_x, det_y
    map_atts         # my headers: Kaiwen's plus: det_r, masked, OMT
    opt_dict         # {mux_map opt number: (opt_name,linestyle)}
    mux_map          # CRITICAL: Used for lots of things. format:
        # {smurf_band : {channel: 
        #  {map_atts[i] : mux_map[sb&channel][i] for i in range(2,len(map_atts))}
        # }}
    tes_dict         # tes-exclusive (hopefully). {sb:ch:{<tes data>}} 
                     # contains data calculated for tes's and from mux_map. 
                     # guaranteed: {sb:{ch:{'TES_freq','pol','opt','opt_name',
                     #                      'linestyle','masked','OMT'}}}
                     # masked and OMT = 'Y','N',or '?'
                     # other data may join it from tests. ex:
        # If you do normal correction, Temp_Ramp adds:
           # R_n and R_n_NBL (Normal Branch Length, v_bias [V]).
        # Bath_Ramp adds the following keys:
           # k, k_err, Tc, Tc_err, n, n_err, cov_G_Tc_n (default) OR cov_K_Tc_n, 
           # G, G_err, p_sat100mK, p_sat100mK_err
    tes # axis version of mux_map and tes_dict. dict() of key:numpy array of length=#smurf_channels
        # Keys:
     
    
    # --------- Possibly Added Vars: ---------
    # Added by anything that gets channel data with associated biasline data:
    bls_seen         # {sb:{ch:[list of biaslines of associated channel datas]}}   
                     # ^ Exists because Kaiwen's mux_map doesn't tell me some bls
                     # and you can get data from unpowered bias lines sometimes
    bls_standalone   # bls_seen, but only the ones that passed standalone cuts.
    # ---- Fine_VNA object
    fine_VNA         # = Fine_VNA(vna_directory)
    # ---- Bath temperature ramp
    bath_ramp        # Bath_Ramp object
    
    # ======================= METHODS (Helpers indented) =======================
    __init__(s, dName, outPath, mux_map_file, opt_dict=None, 
             masked_biaslines=[])
        bonded_TESs_from_mux(s, mux_list)
    check_mux_map_for_channel_and_init(s, sb, ch)
        blank_mux_channel(s, sb, ch)
    check_tes_dict_for_channel_and_init(s, sb, ch)
        blank_tes_channel(s, sb, ch)
    calculate_bls(s)
        assign_bl(s, sb, ch, bl)
            add_bl_data(s, d, bl)
            muxify_tes(s, sb, ch)
    '''
    
    # Obviously not complete. 
    def __init__(s, dName, out_path, mux_map_fp, 
                 opt_dict='default', device_type="MF_UFM", masked_biaslines=[], 
                 bl_freq_dict={}, mux_chips='default', bonded_TESs=[]):
        # mux_map_file must be a csv in Kaiwen's standard format. I made it in excel for SPB-B14; 
        # frankly, I think that's the right way to go for SPB, where P_sat knowledge
        # is important and you often have to swap things around/have crazier detectors.
        s.dName = dName
        s.out_path = out_path
        s.device_type=device_type
        s.masked_biaslines = masked_biaslines
        s.bl_freq_dict = bl_freq_dict
        muxy = {"MF_UFM":(12,28,[90,150]),"UHF_UFM":(12,28,[225,285]), "LF_UFM":(4,4,[30,40])}
        assert s.device_type in muxy.keys(), f"Unknown device_type: {device_type}. Known: {muxy.keys()}"
        s.num_bl, s.num_mux_slots, s.tes_freqs = muxy[device_type]
        if mux_chips=='default':
            s.mux_chips = [1]*muxy[s.device_type][1]
        else:
            s.mux_chips = mux_chips
        
        if bonded_TESs: # overriding normal construction. 
            s.bonded_TESs = bonded_TESs
        else:
            # This part done here so user can alter the mux_chip_to_BL list if desired
            if s.device_type in ["MF_UFM","UHF_UFM"]:
                # default all-mux bonded tes:
                # bonded_TESs = [128,172,144,144,172,128,128,160,150,150,160,128] 
                simple_bl_list = [13,2,2,3,3,13,4,4,5,5,0,0,1,1,10,10,11,11,6,6,7,7,13,8,8,9,9,13]
                s.mux_chip_to_BL_list = [{simple_bl_list[i]:64} for i in range(28)] 
                # now fix the 13s
                s.mux_chip_to_BL_list[0]  = {1:44,2:16} # 
                s.mux_chip_to_BL_list[5]  = {4:44,3:16}
                s.mux_chip_to_BL_list[22] = {7:32,8:22}
                s.mux_chip_to_BL_list[27] = {10:32,9:22}
                # Now account for the dowel pin slots
                # I have traced the detector wafer and gotten the number of TESs blocked by dowel pins fixed.
                #  (see 20221026_SO_UFM_Num_and_Detector_Num_summary.pptx, and 
                #   Erin Healy slack, November 2nd, 2022, and)!!
                # the outer slot (top left of rhombus A)
                s.mux_chip_to_BL_list[ 1][2] = s.mux_chip_to_BL_list[ 1][2]-2 
                s.mux_chip_to_BL_list[13][1] = s.mux_chip_to_BL_list[13][1]-2
                # the central dot (bottom right of rhombus A).
                s.mux_chip_to_BL_list[10][0] = s.mux_chip_to_BL_list[10][0]-2
                s.mux_chip_to_BL_list[ 5][3] = s.mux_chip_to_BL_list[ 5][3]-2 
            if s.device_type == "LF_UFM":
                s.mux_chip_to_BL_list = [{0:34},{1:54},{2:57},{3:40}]
            s.bonded_TESs = s.bonded_TESs_from_mux(s.mux_chips)
            
        s.bls_seen={}
        s.bls_standalone = {}
        s.mux_map_fp = mux_map_fp
        if mux_map_fp:
            s.base_mux_map = np.genfromtxt(mux_map_fp,delimiter=",",skip_header=1, dtype=None)
        else:
            s.base_mux_map = {}
        s.map_atts = map_atts # has to be global for the function that updates mux_map        
        # This was a poor decision to put it all in opt. 
        # Should have separate masked/unmasked and dark-fab/opt-fab values.
        # unfortunately changing at this point would require a ton of work. 
        # ....I mean, I do keep fields that have them separate now, it's just that 
        # several Temp_Ramp functions only look at opt.
        if opt_dict == 'default':
            # Defining the opt map here!
            # I think Kaiwen just uses -1, 0 and 1. Does not account for masks here.
            # opt_dict = {opt_num: (name,linestyle)}
            s.opt_dict = {-42: ('?','?'),
                          -1.0: ('no TES', 'None'),
                          -0.75: ('Fabrication error TES (effectively disconnected)', 'None'),
                          0.0: ('dark masked', 'dotted'),
                          0.125: ('fab?, masked','dashdot'),
                          0.25: ('opt masked', 'dashdot'),
                          0.75: ('dark horn-pixel', 'dashed'),
                          0.875: ('fab? horn', 'solid'),
                          1.0: ('opt horn', 'solid')} 
            if device_type == "LF_UFM":
                s.opt_dict[0.75] = ('dark lenslet', 'dashed')
                s.opt_dict[0.875] = ('fab? lenslet', 'solid')
                s.opt_dict[1.0] = ('opt lenslet', 'solid')
        else:
            s.opt_dict = opt_dict
        
        # At least in Kaiwen's Mv6, opt=0.0 if dark fab OR opt fab masked. opt=1.0 if opt and unmasked.
        for i in range(len(s.base_mux_map)):  
            if int(s.base_mux_map[i][9]) in masked_biaslines: # det[9] = the bias line. Is masked.
                #print("recognized a masked bias line")
                if (str(s.base_mux_map[i][10])[2:-1] == 'A' or str(s.base_mux_map[i][10])[2:-1] == 'B'): #float(s.base_mux_map[i][15]) == 1.0:        # det[15] = opt. masked optical
                    s.base_mux_map[i][15] = 0.25               # masked dark (0.0) stays the same. 
                #print(s.mux_map[i][10])
                if (str(s.base_mux_map[i][10])[2:-1] == 'A' or str(s.base_mux_map[i][10])[2:-1] == 'B') and float(s.base_mux_map[i][15])==0.0: # pol=D=Kaiwen's dark fab note 
                    s.base_mux_map[i][15] = 0.25  # That's actually an optical masked detector. 
            else:                                         # unmasked
                if float(s.base_mux_map[i][15]) == 0.0 and 'D' in str(s.base_mux_map[i][10]): #str(s.base_mux_map[i][10])[2:-1] == 'D':
                    s.base_mux_map[i][15] = 0.75               # dark fab on area getting light. 
                
        mux_map = {}
        tes_dict = {}
        for chan in s.base_mux_map:
            sb, ch = try_to_numerical(chan[0]), try_to_numerical(chan[1])
            if not sb in mux_map.keys():
                mux_map[sb] = {}
            mux_map[sb][ch] = {}
            for i in range(2,len(orig_map_atts)):
                item = chan[i]
                if type(item) == np.bytes_:
                    item = str(item)[2:-1]
                item = try_to_numerical(item) # will leave as string if can't be cast to float or int
                mux_map[sb][ch][map_atts[i]] = item
            #mux_map[sb][ch]['TES_freq'] = str(mux_map[sb][ch]['TES_freq'])[2:-1]
            # My additional parameters
            mux_map[sb][ch]['det_r'] = (mux_map[sb][ch]['det_x']**2 + mux_map[sb][ch]['det_y']**2)**(1/2)
            mux_map[sb][ch]['masked'], mux_map[sb][ch]['OMT'] = '?','?'
            if mux_map[sb][ch]['opt'] in s.opt_dict.keys():
                opt_name = s.opt_dict[mux_map[sb][ch]['opt']][0]
                if 'masked' in opt_name:
                    mux_map[sb][ch]['masked'] = "Y"  
                elif 'horn' in opt_name or 'lenslet' in opt_name:
                    mux_map[sb][ch]['masked'] = 'N'
                if 'opt ' in opt_name:
                    mux_map[sb][ch]['OMT'] = 'Y'
                elif 'dark ' in opt_name:
                    mux_map[sb][ch]['OMT'] = 'N'
            if mux_map[sb][ch]['opt'] >= 0: # TES expected
                if not sb in tes_dict.keys():
                    tes_dict[sb] = {}
                tes_dict[sb][ch] = {}
                # Yes, the below is s.muxify_tes() (and not DRY), but using that means
                # putting s. in front of every mux_map and tes_dict in init,
                # which actually makes init less readable.
                for tes_key in ['TES_freq','pol','opt','masked','OMT','biasline','TES_freq']:
                    tes_dict[sb][ch][tes_key] = mux_map[sb][ch][tes_key]
                if mux_map[sb][ch]['opt'] in s.opt_dict.keys():
                    tes_dict[sb][ch]['opt_name'] = s.opt_dict[mux_map[sb][ch]['opt']][0]
                    tes_dict[sb][ch]['linestyle'] = s.opt_dict[mux_map[sb][ch]['opt']][1]
                else:
                    #print(f"{ch} unknown opt: {mux_map[sb][ch]['opt']}")
                    tes_dict[sb][ch]['opt_name'] = f"? opt: {mux_map[sb][ch]['opt']}"
                    tes_dict[sb][ch]['linestyle'] = (0, (3, 5, 1, 5, 1, 5)) # dashdotdotted  
        # override frequencies
        if bl_freq_dict:
            for sb in mux_map.keys():
                for ch, d in mux_map[sb].items():
                    if d['biasline' ] in bl_freq_dict.keys():
                        d['TES_freq'] = bl_freq_dict[d['biasline']]
                        if sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                            tes_dict[sb][ch]['TES_freq']  = bl_freq_dict[d['biasline']]
        s.mux_map = mux_map
        s.tes_dict = tes_dict
   
    def bonded_TESs_from_mux(s,mux_list):
        bonded_TESs = np.full((s.num_bl,),0,dtype=int)
        for i in range(s.num_mux_slots):
            for bl,num_TESs in s.mux_chip_to_BL_list[i].items():
                bonded_TESs[bl] += num_TESs*mux_list[i]
        return bonded_TESs

    def blank_mux_channel(s, sb, ch):
        s.mux_map[sb][ch] = {}
        for i in range(2,len(map_atts)):
            s.mux_map[sb][ch][map_atts[i]] = -42 # Unknown
            
    def check_mux_map_for_channel_and_init(s, sb, ch):
        mux_map = s.mux_map
        if  sb not in mux_map.keys():
            mux_map[sb] = {}
        if ch not in mux_map[sb].keys():
            s.blank_mux_channel(sb, ch)
            
    def blank_tes_channel(s, sb, ch):
        s.tes_dict[sb][ch] = {}
        s.tes_dict[sb][ch]['TES_freq'] = -42 # Unknown
        s.tes_dict[sb][ch]['linestyle'] = (0, (3, 5, 1, 5, 1, 5)) # dashdotdotted      
        for tes_key in ['pol','opt','opt_name','masked','OMT','biasline']:
            s.tes_dict[sb][ch][tes_key] = "?"
                   
                
    def check_tes_dict_for_channel_and_init(s, sb, ch):
        tes_dict = s.tes_dict
        if sb not in tes_dict.keys():
            tes_dict[sb] = {}
        if ch not in tes_dict[sb].keys():
            s.blank_tes_channel(sb, ch)
        # inherit from mux_map
        if sb in s.mux_map.keys() and ch in s.mux_map[sb].keys():
            for key,val in s.mux_map[sb][ch].items():
                if key in s.tes_dict[sb][ch]:
                    s.tes_dict[sb][ch][key] = val
            
    
    def add_bl_data(s, d, bl):
        # ONLY CALL IF OVERWRITING!!!
        # d the dict to apply it to
        d['biasline'] = bl
        if bl in s.bl_freq_dict.keys():
            d['TES_freq'] = s.bl_freq_dict[bl]
        if bl in range(s.num_bl): # real bl!
            if s.masked_biaslines:
                if bl in s.masked_biaslines:
                    d['masked'] = 'Y'
                    d['opt'] = 0.125 # masked, fab unknown.
                else:
                    d['masked'] = 'N'
                    d['opt'] = 0.875 # unmasked, fab unknown
                    
    def muxify_tes(s, sb, ch):
        for tes_key in ['pol','opt','masked','OMT','biasline','TES_freq']:
            s.tes_dict[sb][ch][tes_key] = s.mux_map[sb][ch][tes_key]
        if s.mux_map[sb][ch]['opt'] in s.opt_dict.keys():
            s.tes_dict[sb][ch]['opt_name'] = s.opt_dict[s.mux_map[sb][ch]['opt']][0]
            s.tes_dict[sb][ch]['linestyle'] = s.opt_dict[s.mux_map[sb][ch]['opt']][1]
        else:
            s.tes_dict[sb][ch]['opt_name'] = -42 # Unknown
            s.tes_dict[sb][ch]['linestyle'] = (0, (3, 5, 1, 5, 1, 5)) # dashdotdotted  
        
    def assign_bl(s, sb, ch, bl):
        # just in case
        s.check_mux_map_for_channel_and_init(sb, ch)
        if try_to_numerical(s.mux_map[sb][ch]['biasline']) != try_to_numerical(bl):
            s.blank_mux_channel(sb, ch)
            s.add_bl_data(s.mux_map[sb][ch],bl)
            if sb in s.tes_dict.keys() and ch in s.tes_dict[sb].keys():
                s.muxify_tes(sb, ch) # actually don't want to blank it entirely.
                s.add_bl_data(s.tes_dict[sb][ch], bl)
        else: # not changing mux map bl
            # just gotta check tes_dict
            if sb in s.tes_dict.keys() and ch in s.tes_dict[sb].keys():
                if try_to_numerical(s.tes_dict[sb][ch]['biasline']) != try_to_numerical(bl):
                    s.muxify_tes(sb, ch)
                    s.add_bl_data(s.tes_dict[sb][ch], bl)
                    
    def calculate_bls(s): 
        # Run only after bls_standalone has been populated.
        # decides what bl each channel is on if that not given by map, performs cuts,
        # and sets up ramp_raw_arr and ramp_arr
        for sb in s.bls_standalone.keys():
            for ch,bl_list in s.bls_standalone[sb].items():
#                 # extra weight curves that passed standalone tests
#                 bl_listy = bl_list
#                 if sb in s.test_device.bls_standalone.keys() \
#                    and ch in s.test_device.bls_standalone[sb].keys():
#                     bl_listy = bl_list + 3*s.test_device.bls_standalone[sb][ch]
                bl_counts = np.unique(np.array(bl_list),return_counts=True)
                mode_idx = np.where(bl_counts[1]==max(bl_counts[1]))[0]
                if len(mode_idx)==1: # one clear mode
                    new_bl = try_to_numerical(bl_counts[0][mode_idx[0]])
                    # Let's update the mux_map.
                    # b/c Kaiwen's map didn't contain all of these.  
                    s.check_mux_map_for_channel_and_init(sb, ch)
                    s.assign_bl(sb, ch, new_bl)
                else: # If I don't do this, it bathramp+coldloadramp can lead to weirdness.
                    # where it gets one bl in bath ramp, but then coldloadramp 
                    # balances that bl. 
                    s.check_mux_map_for_channel_and_init(sb, ch)
                    s.blank_mux_channel(sb,ch)
                    if sb in s.tes_dict.keys() and ch in s.tes_dict[sb].keys():
                        s.blank_tes_channel(sb,ch)
                # Is it possible to get equal of a wrong and a right?
                # Theoretically yes, but I don't think I've ever seen it


# ==============================================================================
# --------------------------- Temp_Ramp Support --------------------------------
# ==============================================================================
              
                
# ============ Temp_Ramp metadata adjustment functions 
def temp_conversion_dict(temp_tab_table_string,col_in,col_out):
    '''# tab-separated temp list, titles acceptable; EX:
    MC Ch 15, X-066 (PID channel)	Corrected MC (mK)	Cu Mount Ch 16, X-005 (mK)	Mv5 clamp Ch 14, 6317 (mK)
    60	64	73	84
    80	81	88	95
    100	99	103	110
    120	115	120	125
    140	134	137	141
    160	151	154	156
    180	168	171	173'''
    ttts = temp_tab_table_string
    temp_list = [[try_to_numerical(num) for num in line.split("\t")] for line in ttts.split("\n")]
    return {line[col_in]:line[col_out] for line in temp_list}

def new_metadata_dif_temps(metadata_file_path,new_file_path,temp_conversion_dict):
    """temp_conversion_dict = {<oldtemp>:<newtemp>}, both temps floats"""
    tcd = temp_conversion_dict
    metadata = np.genfromtxt(metadata_file_path,delimiter=',', dtype='str')
    #Path(new_file_path).mkdir(parents=True, exist_ok=True)
    with open(new_file_path,'w') as nf:
        for line in metadata:# bl = bias line, sbs= smurf bands (not subband), fp = file path
            temp, stuff, bl, sbs, fp, meas_type = line 
            try:
                nf.write(f"{tcd[float(temp)]},"+",".join([stuff,bl,sbs,fp,meas_type])+"\n")
            except Exception as inst:
                print(f"{inst}: skipping line = {line}!")
    #print(np.genfromtxt(new_file_path,delimiter=',', dtype='str'))
    return new_file_path    


# =========== Temp_Ramp loading functions useful for messing around with things

def fill_iva_from_preset_i_tes_i_bias_and_r_n(s,iv_py, iv):
    # unload previous r_n
    r_n = iv['R_n']
    r_sh = s.r_sh
    iv['R'] = r_sh * (iv['i_bias']/iv['i_tes'] - 1)
    ''' # Wasn't using these and they somehow caused problems with Uv42.
    #  get pysmurf's target R_op_target:
    #py_target_idx = np.ravel(np.where(iv_py['v_tes']==iv_py['v_tes_target']))[0]
    # above doesn't work b/c sc offset removal can remove the point that it targeted, so:
    v_targ = iv_py['v_tes_target']
    if not is_float(v_targ): # can be np.nan in some + to -V runs
        #sc_idx, nb_idx = iv['trans idxs']
        #v_targ = np.median(iv_py['v_tes'][sc_idx:nb_idx]) # can still be np.nan
        v_targ = 5 # this could be horribly wrong, but I don't htink we ever use this value.
        print(f"{iv['sb']} {iv['ch']}  {iv['temp']} {iv['bl']} {iv['v_tes_target']}; new_targ={v_targ} {iv['v_tes']}")
        iv['v_tes_target'] = v_targ
    
    py_target_idx = np.ravel(np.where( abs(iv_py['v_tes']-v_targ) == min(abs(iv_py['v_tes']-v_targ)) ))[0]
    R_op_target = iv_py['R'][py_target_idx]    
    
    
    # Now just refill the rest of the keys same way pysmurf does (except do correct v_bias_target):
    # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
    # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]]
    #iv['R'] = r_sh * (iv['i_bias']/iv['i_tes'] - 1) # moved up to l
    # the correct equivalent of pysmurf's i_R_op
    i_R_op = 0
    for i in range(len(iv['R'])-1,-1,-1):
        if iv['R'][i] < R_op_target:
            i_R_op = i
            break
    iv['v_bias_target'] = iv['v_bias'][i_R_op]
    
    iv['v_tes_target'] = iv['v_tes'][i_R_op]
    '''
    iv['v_tes'] = iv['i_bias']/(1/r_sh + 1/iv['R']) # maybe use R_sh*(i_bias-i_tes)?
    iv['p_tes'] = iv['v_tes']**2/iv['R']
    # pysmurf's p_trans: 
    iv['p_trans'] = np.median(iv['p_tes'][iv['trans idxs'][0]:iv['trans idxs'][1]])
    # SO p_trans (if possible):
    for per in [40,50,60,70,80,90]:
        if len(np.ravel(np.where(iv['R']<per/100.0*iv['R_n']))) > 0:
            iv[f'p_b{per}'] = iv['p_tes'][np.ravel(np.where(iv['R']<per/100.0*iv['R_n']))[-1]]
        else:
            iv[f'p_b{per}'] = -42.0 # you never reach it. 
    # TODO: STILL NEED TO ADD si and si_target! 
    return iv 



# Important: This is not the primary contextless_is_IV being developed!!! 
# THis is just to make Uv8_low work okay!!
def contextless_is_iv(iva,debug=False, r_sh=0.0004,save_to_iva=False): 
    # TOD: REmove the R_sh argument, require be given v_bias and i_Tes
    #i_bias =  iva['v_tes']*(1/s.r_sh + 1/iva['R'])
#     if not "i_bias" in iva.keys():
#         iva['i_bias'] =  iva['v_tes']*(1/r_sh + 1/iva['R'])
#     i_bias = iva['i_bias']
    if "i_tes" in iva.keys():
        i_tes=iva['i_tes']
    else:
        i_bias =  iva['v_tes']*(1/r_sh + 1/iva['R'])
        i_tes = i_bias/(iva['R']/r_sh + 1) 
        iva['i_bias'] = i_bias
        iva['i_tes'] = i_tes
    v_bias = iva['v_bias']
        
    #v_bias = iva['v_bias']
    #v_bias,i_tes = iva['v_bias'],iva['i_tes'] 
    
    cii_result = contextless_is_iv_proper(v_bias,i_tes,debug=debug)
    # Let's go ahead and save the results. 
    if save_to_iva:
        if not type(cii_result) == str: # Is an IV curve. 
            sc_idx,nb_idx,cii_note = cii_result[0:3]
            iva['contextless_is_iv'] = True
            iva['contextless_idxs'] = [sc_idx,nb_idx]
            iva['contextless_note'] = cii_note
            if not debug:
                iva['contextless_mask'] = cii_result[3]
        else:
            iva['contextless_is_iv'] = False
            iva['contextless_note'] = cii_result
    return cii_result

# 2022-05-25 Version!!!!
def contextless_is_iv_proper(v_bias,i_tes,debug=False):
    """Given numpy arrays v_bias and i_tes, 
    ---> which MUST both be in units of Volts and microAmps respectively <----
    returns a string saying what it thinks the problem is
    if it isn't an iv.
    otherwise, returns [sc_idx,nb_idx,note (ex wrong pol),noise_mask]
    noise_mask a len(i_tes) array of 0 if noise,1 if not;
    omits the noise_mask if run in debug mode for cleaner viewing.
    
    IMPORTANT NOTE: If a curve has less than ~10 data points in its 
    superconducting branch and transition, contextless_is_iv will think
    it's normal. Sometimes when bath temperatures are high enough
    you can actually want those--but in those cases I assume you also
    have contextful cuts to work with."""
    # It has to be v_bias and i_Tes, because those are the measurables. 
    # I_bias depends on your R_tes calculation. 
    # I realized this after trying I_bias instead of v_bias. 
    # ^ there is a problem making this not work though. 
    note = "" # the note to return.
    # before we do anything else, throw out ones that are actually open or unbiased.
    if max(i_tes)-min(i_tes) < 1: # uA:
        return "no i_tes"
    
    
    #steps = np.diff(v_bias)
    #d_tes = np.diff(i_tes)#/steps # this doesn't work, some steps =0
    #dd_tes = np.diff(i_tes)#/steps
    sc_idx = 0 # big issue: we don't care if SC bad. 
    # (ex. Mv13 7,198 t=103; bl=11 I think) So can't count on the 
    # superconducting branch behaving in any normal way
    nb_idx = 0
    
    
    # TODO: find a way to ignore sections that are too noisy. 
    # Maybe fit line between the averages, stdev of the residuals?
    
    # We do need the transition, so can 
    # so, really, what we want is nb_idx.
    # 
    #partitions = min(50, len(i_tes)) #min(50, len(i_tes))
    #siz = len(i_tes) // partitions
    siz = 15
    partitions = len(i_tes) //siz
    partition_idx = [i for i in range(0,len(i_tes),siz)]
    sizTop = 2 # ---HARDCODE--- #max(int(siz/5),2) # only consider sizTop most noisy points per transition (will subtract most noisy out)
    min_resid_size =0.5 # 0.5, (max-min)/10---HARDCODE--- # won't consider anything within that many uA of the line as noise.
    
    # debug:
    if debug:
        plt.figure(figsize=default_figsize)
        # do the below later for clearer view
        #plt.plot(v_bias,i_tes,color="black",linewidth=0.3) #-siz//2
        #plt.plot(v_bias[0:siz*partitions],i_tes[0:siz*partitions],color="black",linewidth=0.3) #-siz//2
    
    lines = []
    noise_val = []
    # Now, we want to ignore areas that are too noisy. 
    # For now, just ignoring the sides. 
    #x1,y1 = v_bias[0],i_tes[0]
    i1 = partition_idx[0]
    x1,y1 = v_bias[i1],np.average(i_tes[0:i1+siz//2])
    #n_v_bias,n_i_tes = np.asarray([]),np.asarray([])
    noise_mask =np.zeros(len(i_tes))
    for i in range(0,len(partition_idx)-1):
        i2 = partition_idx[i+1]
        x2,y2 = v_bias[i2],np.average(i_tes[i2-siz//2:min(i2+siz//2,len(i_tes)-1)])
        slope = (y2-y1)/(x2-x1)
        const = y2-slope*x2
        #sss = min_resid_sign_swap_sum(min_resid_size,v_bias,i_tes,slope,const)
        #lines.append([slope,const])
        line = slope*v_bias[i1:i2]+const
        resid = i_tes[i1:i2] - line
        #stdev = np.std(i_tes[i1:i2] - line) # not really what I want
        #resid_noise = sum(abs(np.diff(np.diff(resid))))-max(abs(np.diff(np.diff(resid)))) # remove sc_idx
        # How does # of times slope changes sign work?
        #sign_swap = sum(abs(np.diff(np.sign(np.diff(i_tes[i1:i2])))))/2
        #ssp = 100*sign_swap/(i2-i1) #  sign_swap% ()
        
        # orl serves to ignore noise that is too small that might otherwise kill real data
        orl = i_tes[i1:i2][np.where(abs(resid)>min_resid_size)] #outside resid limit 
        sss = 0
        sign_swaps = abs(np.diff(np.sign(np.diff(orl))))/2 #i_tes[i1:i2]
        ssa = sign_swaps*abs(np.diff(np.diff(orl))) #i_tes[i1:i2]
        if len(ssa)==0:
            sss=0
        else:
            # to stop catching ones with lots of tiny, barely eye-visible
            # variations that still basically look like a line, take only the top
            # <MAX(1/5th of size, 2)>; see where siz is established
            ssat = -np.partition(-ssa,min(sizTop,len(ssa)-1))[:min(sizTop,len(ssa)-1)]
            sss = (sum(ssat)-max(ssa))#sum(ssat) # remove superconducting peak.
#             if debug and sss>0:
#                  print(f"{i1}: {int(sum(sign_swaps))},{sum(ssa)-max(ssa)}; {int(len(ssat))},{sum(ssat)-max(ssa)}")
        
        noise_val.append(sss)
        sss_lim=0.1 #, resid_lim = 0.1,0 #0.1,1 <-what I had at presentation  #0.1,5 # ---HARDCODE---
        if  sss < sss_lim:# or resid_noise < resid_lim:# and resid_noise < resid_lim:#i>40:    #noisiness < 15 and
            #n_i_tes = np.append(n_i_tes,i_tes[i1:i2])
            #n_v_bias = np.append(n_v_bias,v_bias[i1:i2])
            noise_mask[i1:i2] = 1
        elif debug:
            label = f'{i1}: '
            if sss >= sss_lim:
                label += f'ss{sss:.1f} '
#             else:
#                 label+="______ "
#             if resid_noise >= resid_lim:
#                 label += f'r{resid_noise:.1f}'
#             else:
#                 label += "_____ "
            plt.plot(v_bias[i1:i2],line,label=label)#,linestyle='dashed'
        #print(f"{i1}:{resid}")
        i1,x1,y1 = i2,x2,y2
    # If it's surrounded by a ton of noise, it is probably also noise. 
    # see noise_mask_smoother for development                
    made_change = True
    while made_change:
        made_change=False
        # v makes idxs line up how I want.
        nmd = np.concatenate(([0],np.diff(noise_mask[:partitions*siz]))) # only to the end of the partitions.
        # v to make the first 
        patch_trans = np.concatenate(([0],np.where(abs(nmd)==1)[0])) #  first idx of every patch
        patch_sizes= [patch_trans[i+1]-patch_trans[i] for i in range(len(patch_trans)-1)]+[len(nmd)-patch_trans[-1]]
        if len(patch_sizes) >=2:
            # first, the starting and ending ones:
            if noise_mask[0] == 1 and patch_sizes[1]/2 >=patch_sizes[0]:
                made_change=True
                noise_mask[0:patch_trans[1]] = 0
            # Remember, patch_sizes and patch_trans only go out to end of partitions.
            if noise_mask[partitions*siz-1] == 1 and patch_sizes[-2]/2>=patch_sizes[-1]:
                made_change=True
                noise_mask[patch_trans[-1]:] = 0
            # now resolve the middle
            patch_type = noise_mask[patch_trans[1]]
            for i in range(1,len(patch_trans)-1):
                if patch_type ==1 and (patch_sizes[i-1]+patch_sizes[i+1])/2>=patch_sizes[i]: # formerly min
                    made_change = True
                    noise_mask[patch_trans[i]:patch_trans[i+1]] = 0
                    if debug:
                        plt.plot(v_bias[patch_trans[i]:patch_trans[i+1]],[i_tes[patch_trans[i]]]*patch_sizes[i],label="nms cut")
                patch_type = (patch_type+1) % 2
    n_v_bias = v_bias[np.where(noise_mask==1)[0]]    
    n_i_tes = i_tes[np.where(noise_mask==1)[0]]
    # plot the original and the offset version
    if debug:    
        plt.plot(v_bias,i_tes,color="black",linewidth=0.3) #-siz//2
        #^above: jsut plot the original datea.
        my_off = 0.2*(max(i_tes)-min(i_tes))
        plt.plot(n_v_bias,n_i_tes-my_off,color="black",marker=".", \
                 markersize=0.2,linestyle='None',label="NOT noise-cut")
        plt.legend()
        
    # begin evaluating if it is in fact an iv curve.
    if len(n_i_tes)/len(i_tes) < 0.45: # ---HARDCODE---
        return "noise"
    #v_bias, i_tes = n_v_bias, n_i_tes
    
    
    # ----- DETERMINING nb_idx ----- 
    # The noise filter kills the bigger oscillations. Need a smaller step size 
    # to catch mini curves
    asiz = min(5,siz) # ---HARDCODE---
    averages = np.asarray([np.average(n_i_tes[i:i+asiz]) for i in range(0,len(n_i_tes),asiz)] \
                          + [np.average(n_i_tes[asiz*(len(n_i_tes)//asiz):])])
    
    
    rough_nb_av = len(averages)
    for i in range(len(averages)-2,-1,-1):
        if averages[i] > averages[i+1]:
            rough_nb_av = i
            break
    
    if rough_nb_av == len(averages):
        return "normal"
    
    # The minimum could be in the segment before the one that increased.
    ip=min(1,(len(averages)-1)-(i+1)) # ensures no issues if i=len(averages)-1 or len(averages)-2.
    nb_idx_segment = n_i_tes[i*asiz:(i+1+ip)*asiz]
    if i/len(averages) > 0.03: # not a high bath temp, teeny-tiny transition that fits in 5 points
        nb_idx = np.where(nb_idx_segment == min(nb_idx_segment))[0][0] + i*asiz
    else:  # very low v_bias transition # I don't think this is working great right now, not sure...
        # go ahead and do step by step almost pointwise
        for i in range(len(nb_idx_segment)-2,-1,-1):
            if nb_idx_segment[i]>nb_idx_segment[i+1]:#sum(nb_idx_segment[i-1:i+1])/2 > sum()
                nb_idx = i+1 + i*asiz

    # CHeck normal resistance. 
    # HAS TO BE IBIAS FOR THIS TO WORK! Edit: nope, I just can't do this contextless. 
    
    if debug:
        plt.vlines([n_v_bias[nb_idx]],
                min(i_tes),max(i_tes), linestyles="dashdot")
    
    # Add a check for multiple ups and downs...
 
    # ----- Checking the normal branch ----- 
    # Fit R_n, checking for negative value (don't need context to know that's wrong!).
    nb_fit_idx = nb_idx + (len(n_v_bias)-nb_idx)//2 # kind of cheap R_n adjustment, admittedly...
    # ^ But I'm only trying to get pos/neg right now, don't care about getting EXACT R_n
    
    if len(n_v_bias)-nb_fit_idx<3: # linear error, Singular Matrix error with polyfit.
        slope = -42 # skip to flipping polarity, only possible way this could still be right. 
    else:
        param, cov = np.polyfit(n_v_bias[nb_fit_idx:],n_i_tes[nb_fit_idx:],1,cov="unscaled")
        slope,i_tes_offset = param
        
        nb_fit_percent = (len(n_v_bias)-nb_fit_idx)/len(n_v_bias)
        # slcov_lim = 0.1/(nb_fit_percent) legitimately small normal branches are an issue, but fakes have low as well
        # TODO: THIS HAS TO GO. IT IS x-AXIS SCALE DEPENDENT.
        slcov_lim = 0.15# 0.15 #0.15 # 0.05# ---HARDCODE--- # very little normal branch, higher cov...(typically < 0.001, but close to it)
        if cov[0][0] > slcov_lim: 
            return "noisy normal branch "+ f'slcov_lim{slcov_lim:.1f} slcov{cov[0][0]:.4f} {cov[1][1]:.4f}'
        
        resid = n_i_tes[nb_fit_idx:] - (i_tes_offset + slope*n_v_bias[nb_fit_idx:])
        nb_max_abs_resid = max(abs(resid))
#         if nb_max_abs_resid>1:
#             return "noisy normal branch "+ f'max(abs(resid)) {nb_max_abs_resid}'
    # possibly bad polarity given
    # TODO: make sure no clash with this and other stuff!
    if slope <= 0:
        if len(n_i_tes)-nb_idx < nb_idx-sc_idx and n_i_tes[-1]<= n_i_tes[nb_idx]:
            #pol_iva = {'v_bias':v_bias,'i_tes':i_tes*-1} 
            #pol_iva = {'v_bias':n_v_bias,'i_tes':n_i_tes*-1} # if give new, possible to repeat!!
            #pol = contextless_is_iv(pol_iva,debug=debug,r_sh=r_sh)
            pol = contextless_is_iv_proper(v_bias, i_tes*-1,debug=debug)
            if not type(pol) == str:
                if debug:
                    return pol[0:2] + [note + pol[2]+" POL WRONG"]
                else:
                    return pol[0:2] + [note + pol[2]+" POL WRONG"] + [noise_mask]
        return "negative slope normal branch" 
    
    
    # ----- DETERMINING sc_idx ----- 
    # This needs major improvement to deal with weird superconducting stuff. 
    # actually, the noise removal seems to be making this work surprisingly well,
    # though TODO: should probably add the nearby slope check. 
    # TODO: Check removed areas near this, maybe?
    if len(n_i_tes[:nb_idx])==0:
        return "no transition region"
    sc_idx = np.where(n_i_tes[:nb_idx] == max(n_i_tes[:nb_idx]))[0][0]
    if debug:
        plt.vlines([n_v_bias[sc_idx]],
                min(i_tes),max(i_tes),  linestyles="dotted")
    
    # ----- Checking the transition a bit ----- 
    # now a simplistic directional check to make sure I'm not looking at something flat.
    trans_slope = (n_i_tes[nb_idx]-n_i_tes[sc_idx])/(n_v_bias[nb_idx]-n_v_bias[sc_idx])
    if trans_slope >-0.25:
        return "~flat or positive in-transition slope"
    
    if nb_idx-sc_idx <=5 or len(i_tes)-nb_idx <=5: # ---HARDCODE---
        return "cut-data idx too close" + str([sc_idx,nb_idx])
    if len(n_i_tes)-nb_idx < nb_idx-sc_idx:
        note = note + "nb<transition!"
    
    
        
    # Returning in the original. SHOULD POSSIBLY CHANGE, TODO:
    # That is, need to do analysis with the CUT data.
    
    sc_idx = np.where(i_tes==n_i_tes[sc_idx])[0][0]
    nb_idx = np.where(i_tes==n_i_tes[nb_idx])[0][0]
    
    if nb_idx-sc_idx <=5 or len(i_tes)-nb_idx <=5: # ---HARDCODE---
        return "idx too close" + str([sc_idx,nb_idx])
    #{slope:.2f} slcov_lim{slcov_lim:.1f}
    if debug:
        return [sc_idx,nb_idx,note + f'sl:{slope:.2f}  max(abs(resid)):{nb_max_abs_resid:.4f}'] #slcov{cov[0][0]:.6f} {cov[1][1]:.4f}
    return [sc_idx,nb_idx,note + f'sl:{slope:.2f}  max(abs(resid)):{nb_max_abs_resid:.4f}', noise_mask] #slcov{cov[0][0]:.6f} {cov[1][1]:.4f}

# ==============================================================================
# --------------------------- Temp_Ramp Class ----------------------------------
# ==============================================================================

class Temp_Ramp:
    '''
    This class serves as a data container for EITHER a bath temperature sweep
    OR a cold load temperature ramp. Those are child classes of this one.
    
    # We do add channels to mux_map if they're missing there; not to tes_dict.
    -- adds 'R_n' to tes_dict if not already there (updates if gets better one),
    # using the R_n reported in IV curves that weren't removed by cuts that 
    # has the longest normal branch. (because while pysmurf has less normal 
    # region to work with, it (usually---can go other way, we account for this) 
    # overestimates R_n (1/slope of fit orange dashed line) due to including 
    # bits of the IV curve). Our non-raw p_b90's are also p_b90's of THAT R_n. 
    
    # =============================== VARIABLES ================================
    r_sh                 # 0.0004 # Pysmurf's r_sh, in ohms.
    expected_R_n_min     # 0.006
    expected_R_n_max     # 0.010
    # ---- Init argument vals
    test_device          # Test_Device object of this array/SPB
    ramp_type = 'bath' or 'coldload' 
    temp_unit = 'mK' for bath or 'K' for coldoad 
    dName                # device name. TODO: add to Temp_Ramp plots.
    therm_cal            # [a, b] of ax + b thermometer calibration
    metadata_fp_arr      # [FILEPATHS]; if NOT array, (one fp), reloading from a save! 
    norm_correct         # Take the highest-T_b IV curve's R_n as R_n for all of 
                         # that detector? Default True.
    use_per              # what percent of R_n to calculate p_sat at. Default 90.
    p_sat_cut            # what p_sat= too high, but the normal branch MUST
                         # reach at least this power. 15 for MF, 30 for UHF?
    use_p_sat_SM         # Whether to consider pysmurf's p_sat in cuts.
    input_file_type      # defaults "pysmurf", set "original_sodetlib" for the 
                         # original take_iv.
    sc_offset            # if False and ^=original_sodetlib, loads raw w/ no sc_off
    bin_tod              # if False and ^=original_sodetlib, doesn't bin TOD.
    use_cii              # use contextless_is_iv? No=Faster, Yes=better cuts 
    save_raw_ivas        # save the iv_analyzed dict post-load pre-analysis.
                         # Make a string to override default filename,
                         # otherwise bool. Exists mostly b/c of how long 
                         # sc_off=False takes to load from tod. 
    # ---- init_notes
    reloaded_from        # The fp reloaded from, if reloaded.

    # ---- reference/analysis from loaded data
    key_info             # {key in (other) dictionary in this class : {
                         #    'name','units','lim'}} # 'lim' MAY NOT EXIST.
    idx_fix_fails        # idx_fix debugging dict {sb:{ch:{temp:{
                         #     sc_idx_old,nb_idx_old,nb_idx_new}}}
    temp_list_raw        # List of all temps loaded in, in increasing order
    temp_list            # Increasing List of all temps with >= 1 uncut data point 
    det_R_n              # {sb:{ch:R_n}} made if you do normal correction.
    det_R_n_NBL          # LEGACY: {sb:{ch:<normal branch length in v_bias [V] 
                         #          of IV curve det_R_n taken from>}}
    iv_cat               # categorizor dict of lists of iv names = (sb,ch,temp,bl):
                         # {'is_iv':[],'not_iv':[],
                         #  'wrong_bl':[],'bad_channel':[],'bad_temp':[]}
    
    # ---- loaded ramp vals.
    # NOTE: "p_satSM" = pysmurf p_sat = median p_tes in [sc_idx:nb_idx]
    # If s.norm_correct, also stores <below array name>_nnc with uncorrected data.
    #  ... AND it makes ramp_raw_nnc, a merged ramp_raw_arr_nnc.
    # In the below, replace all "90" uwth use_per
    # Below arrs are [for each metadata, <stuff in comments>]
    iv_analyzed_info_arr # [for each temp: {'temp', 'bl', 'iv_analyzed_fp', 
                         #                  'iv_analyzed','human_time'}]
                         # 'iv_analyzed':{sb:{ch:{'trans idxs', 'v_bias', 
                         #     'i_bias', 'i_tes', 'c_norm', 'R', 'R_nSM', 'R_nSM_raw', 
                         #     'v_bias_target', 'v_tes', 'v_tes_target', 
                         #     'p_tes', 'p_trans', 'p_b90', 'bl', 'is_iv', 
                         #     'is_iv_info','contextless_is_iv', 
                         #     'contextless_note','temp','sb','ch','sbch'}}
                         #     if s.norm_correct: 'all_fit_R_n','all_fit_ITO',
                         #       'is_normal',
                         #       'all_fit_stats':{'r','p','std_err','r_log'}
                         #     if cut: 'cut_info'
                         #     if cii: 'contextless_idxs', 'contextless_mask'
                         #     if instability finder run on it, 'instability_idx'
    ramp_raw_arr         # {sb:{ch:{'temp_raw', 'p_satSM', 'R_nSM_raw', 
                         #          'all_fit_R_n_raw', # -42 if iva not 'is_normal'
                         #          'temp90R_n', 'p_b90R_n'}}}
                         # p_satSM = pysmurf 'p_sat'  | p_b90R_n = SO p_sat
    ramp_arr             # {sb:{ch:{'temp', 'p_b90','R_nSM'}}} 
                         # ^ ramp_arr has bad "IVs" cut out.
    # Only one array, has all from each metadata combined:                     
    ramp                 # the combination of the ramp_arr dictionaries into one.
    
    # ====================== EXTERNAL Support FUNCTIONS =======================
    fill_iva_from_preset_i_tes_i_bias_and_r_n(s,iv_py, iv)
    # Important: This is not the primary contextless_is_IV being developed!!! 
    contextless_is_iv(iva,debug=False, r_sh=0.0004,save_to_iva=False) 
    contextless_is_iv_proper(v_bias,i_tes,debug=False) # 2022-05-25 Version!!!!

    # ======================= METHODS (Helpers indented) ======================
    # ---------- Initialization and loading methods
    __init__(s, test_device, ramp_type, therm_cal, 
             metadata_fp_arr, metadata_temp_idx=0,
             norm_correct=True,use_per=90,p_sat_cut='default',
             use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="pysmurf",
             sc_offset=True, bin_tod=True,use_cii=False, save_raw_ivas=False)
        # __init__ structure: 
        # 1. basic init
        # 2. appends s.load_iv_analyzed(metadata)'s to iv_analyzed_info_arr
        # 3. s.load_temp_sweep_IVs()
        # 4. if norm_correct, make non-corrected data structures and s.do_normal_correction
        # 5. merge ramps for s.ramp. Categorize ivs.
        setup_key_info(s)
        load_iv_analyzed(s,metadata) # loads file data 
            load_raw_iv_analyzed(s,iv_analyzed_fp) 
            original_sodetlib_to_pysmurf(s, sod_iv_analyzed)
        load_temp_sweep_IVs(s) # decides bls, performs cuts, makes ramp_raw_arr and ramp_arr
            count_bls_seen(s)
                fix_neg_v_b(s,iv)
                fix_pysmurf_trans_idx(s, py_ch_iv_analyzed,sb,ch,temp)
                standalone_IV_cut_save(s,d) # called for tes_dict setup 
                    standalone_IV_cut(s,d) 
                         pre_p_sat_cuts(s,d,nb_idx,save_cut_info=False)
            # calls a.test_device.calculate_bls()
            cut_iv_analyzed(s,bathsweep_raw,bathsweep,now_bad_IVs,iai)
                all_curve_lin_fit(s,iva)
                update_temp_list(s,temp, temp_list_name)
                add_ramp_raw_iv(s, bathsweep_raw, temp, sb, ch, d)
                    calc_P_b90(s,r_n,r_tes,p_tes)
                # calls standalone_IV_cut_save() if it's a redo.
                add_ramp_iv(s,bathsweep, temp, sb, ch, d, p_b90=-42)
        do_normal_correction(s)
            calculate_det_R_ns(s)
            pysmurf_iv_data_correction(s, py_ch_iv_analyzed,\
                                 nb_start_idx=0,nb_end_idx=-1,force_R_n=-42.0,
                                 force_c=-42.0)
        merge_ramps(s,bs1,bs2,sort_key="temp", arr_keys=['temp','p_b90','R_nSM'])
            dUnique(s,d1,d2)
        categorize_ivs(s)
            categorize_ivs_helper(s,iv_line)
    redo_cuts(s) # maybe run this when norm_correct makes BIG difference
    set_bl_bad(s,rm_bl) # sets all the is_iv, contextless_is_iv to False for that bl
    
    # ---------- Data Finders
    find_iva(s,sb,ch,temp,bl=-42,run_indexes=[],nnc=False)
    find_iva_matches(s,match_list,dict_detail=False,run_indexes=[],nnc=False)
        write_match(s,match_list,num_levels,to_return,ivaia_dict,sb,ch,dict_detail)
            iva_val(s,match_key,ivaia_dict,sb,ch)
            val_matches(s,match_type,match_val,val)

    # ---------- Plotting Methods
    # --- individual IV curve or det plotters ---
    plot_det_IVs(s,sb,ch,bl=-42,max_temp=3000, x_lim=[],y_lim=[], tes_dict=False, 
                 run_indexes=[], own_fig=True,linewidth=1,do_idx_lines=True)
        plot_IV(s,sb,ch,temp,x_lim=[],y_lim=[], tes_dict=True, bl=-42,run_indexes=[], 
                                  own_fig=True,linewidth=1,
                                  label_include=['default'],do_idx_lines=True)
    plot_det_RPs(s,sb,ch,bl=-42,nnc=False,per_R_n=False,max_temp=3000,
                     lines=['default'],label_include=['temp'],plot_args={}, 
                     x_lim=[],y_lim=[]
        plot_RP(s,sb, ch, temp, tes_dict=True, bl=-42, run_indexes=[],nnc=False,
                                  own_fig=True,lines=['default'],
                                  label_include=[],plot_args={})
            construct_crv_label(s,label_include,label_options)
    plot_iv_analyzed_keys(s,x_key,y_key, sb, ch, temp, 
                                  x_lim=[],y_lim=[], tes_dict=True, bl=-42,
                                  run_indexes=[], own_fig=True,linewidth=1,
                                  label_include=['default'])
        internal_plot_iv_analyzed_keys(s,d,sb,ch,temp, x_key,y_key,
                                  x_lim=[],y_lim=[], tes_dict=True, bl=-42,
                                  run_indexes=[], own_fig=True,linewidth=1)
    # --- MASS IVA PLOTTERS ---
    plot_key_v_key_by_key(s,x_key,y_key,by_key,match_list=[],plot_args={},
                          own_fig=True,do_label=False)
    plot_key_v_key_by_key_by_BL(s,x_key,y_key,by_key,match_list=[],plot_args={},
                                own_fig=True,x_lim=[],y_lim=[], prefix='')
    # ----- RAMP plotters ---
    # --- by sb plotters ---
    plot_ramp_raw(s,ramp_raw,tes_dict=True)
    plot_ramp_raw90R_n(s,ramp_raw,tes_dict=True)
    plot_ramp(s,ramp,tes_dict=True,zero_starts=False)
    plot_ramp_keys_by_sb(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                                 prefix='',tes_dict=True,zero_starts=False)
    plot_ramp_keys_by_sb_2legend(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                                 prefix='',tes_dict=True,zero_starts=False)
    # --- by BL plotters ---
    plot_ramp_by_BL(s,bathsweep,tes_dict=True,y_lim=[0,8])
    plot_ramp_keys_by_BL(s,bathsweep, x_key,y_key,
                                 x_lim=[],y_lim=[], prefix='',tes_dict=True)

    # ====================== EXTERNAL Analysis FUNCTIONS ======================
    examine_ramp(s)
    report_ramp_IV_cat_breakdown(s)
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal,
                 metadata_fp_arr, metadata_temp_idx=0,
                 norm_correct=True,use_per=90,p_sat_cut='default',
                 use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="guess",
                 sc_offset=True, bin_tod=True,use_cii=False, save_raw_ivas=False):
        """Set input_file_type = "original_sodetlib" for the original take_iv."""
        s.test_device = test_device
        # these three come up so often, they get their own variables. 
        s.dName = test_device.dName
        s.mux_map = test_device.mux_map
        s.tes_dict = test_device.tes_dict
        assert ramp_type == 'bath' or ramp_type == 'coldload', \
           "ramp_type must be 'bath' or 'coldload'"
        s.ramp_type = ramp_type # used in key info
        s.therm_cal  = therm_cal
        s.metadata_fp_arr = metadata_fp_arr
        s.metadata_temp_idx = metadata_temp_idx
        
        # optional args
        s.norm_correct = norm_correct
        s.use_per= use_per
        s.p_sat_cut = p_sat_cut
        if p_sat_cut == "default":
            default_p_sat_cuts = {"LF_UFM":10,"MF_UFM":15,"UHF_UFM":55}
            s.p_sat_cut = default_p_sat_cuts[s.test_device.device_type]
        s.use_p_satSM=use_p_satSM
        s.fix_pysmurf_idx=fix_pysmurf_idx
        assert input_file_type in ["guess","pysmurf","original_sodetlib"], \
                f'input file type:{input_file_type} not in ["guess", "pysmurf","original_sodetlib"]'
        s.input_file_type = input_file_type
        s.sc_offset=sc_offset
        s.bin_tod=bin_tod
        s.use_cii=use_cii
        s.save_raw_ivas=save_raw_ivas
        
        # Useful class variables
        s.r_sh = 0.0004 # Pysmurf's r_sh, in ohms. 
        # NIST
        s.expected_R_n_min=0.006
        s.expected_R_n_max=0.010
        # berkeley
        if s.test_device.device_type=="LF_UFM":
            s.expected_R_n_min=0.002
            s.expected_R_n_max=0.015 # raised from 0.090 for Lp3 #0.090
        s.use_frac_R_n = 0 # doesn't use normal branches to find R_n for normal correction.
        
        s.redo=False
        
        # cleanup of various things
        s.plot_RP_easy = s.plot_RP
        s.idx_fix_fails={}
        
        # initialize data structures
        s.temp_list = []
        s.temp_list_raw = []
        s.iv_analyzed_info_arr = []
        s.ramp_raw_arr = []
        s.ramp_arr = []
        # load data
        if type(metadata_fp_arr) == str: # RELOADING
            reload_dict = np.load(metadata_fp_arr, allow_pickle=True).item()
            s.reloaded_from = metadata_fp_arr
            assert (type(reload_dict)==dict \
                   and 'iv_analyzed_info_arr' in reload_dict.keys() \
                   and 'options' in reload_dict.keys()), \
                   f"string metadata_fp_arr, but not a reload file! Make it a list?"
            rd = reload_dict['options']
            # Check the settings make sense
            assert s.test_device.mux_map_fp == rd['mux_map_fp'], \
                   f"ReloadErr: {s.test_device.mux_map_fp} != {rd['mux_map_fp']}"
            for var in ['fix_pysmurf_idx','sc_offset','bin_tod']:
                assert getattr(s,var) == rd[var], f"ReloadErr: {getattr(s,var)} != {rd[var]}"
                #setattr(s,var,rd[var])
            s.metadata_fp_arr = rd['metadata_fp_arr']
            s.iv_analyzed_info_arr = reload_dict['iv_analyzed_info_arr']
            print(f"Reloaded raw iv_analyzed_info_arr from {s.reloaded_from}")
            if s.save_raw_ivas == True:
                print("NOT saving ivas, b/c would overwrite metadata reference!")
                s.save_raw_ivas = False
        else: # should be an array
            for metadata_fp in s.metadata_fp_arr: 
                metadata = np.genfromtxt(metadata_fp,delimiter=',', dtype='str') #skip_header=0
                s.iv_analyzed_info_arr.append(s.load_iv_analyzed(metadata)) 
        # "raw" iv_analyzed_info_arr has been loaded. save it now if specified
        if s.save_raw_ivas:
            Path(s.test_device.out_path).mkdir(parents=True, exist_ok=True)
            save_dict = {'iv_analyzed_info_arr': s.iv_analyzed_info_arr,
                         'options':{'dName': s.dName,
                                    'mux_map_fp':s.test_device.mux_map_fp,
                                    'metadata_fp_arr':s.metadata_fp_arr,
                                    'input_file_type':s.input_file_type,
                                    'fix_pysmurf_idx':s.fix_pysmurf_idx,
                                    'sc_offset':s.sc_offset,
                                    'bin_tod':s.bin_tod}
                        }
            if type(s.save_raw_ivas) == str:
                fname = make_filesafe(s.save_raw_ivas)
            else:
                #fname = f"{int(datetime.timestamp(datetime.now()))}_" # timestamp. Worried that I"ll make too many.
                # I really want to include the metadata, so that I can save different ramps 
                # (at different Other Temperatures, so no combining) of the same device.
                # only get here if DO have a metadata arr, so
                met_fp = s.metadata_fp_arr[0]
                fold= ''
                if "/" in met_fp:
                    fold = '.*/'
                met_name = re.match(fold + "(.*)\.csv",met_fp).group(1)
                fname = f"raw_iva_{make_filesafe(s.dName)}_{met_name}" #_ft-{s.input_file_type[:2]}
                fname = fname +f"_sc_off-{str(s.sc_offset)[0]}.npy" #bin_tod-{str(s.bin_tod)[0]}.npy" Super rare to not have.
            save_path = os.path.join(s.test_device.out_path, fname)
            np.save(save_path, save_dict, allow_pickle=True)
            print(f"Saved raw iv_analyzed_info_arr to {save_path}")
        #print("metadata loaded, no load_temp_sweep_IV")
        
        s.load_temp_sweep_IVs() # checks bls, sets up ramp_raw_arr and ramp_arr
        #print("data loaded and cut")
        
        # Turns out I want these even if not normal-correcting, for dashboard.
        s.det_R_n = {}
        s.det_R_n_NBL = {}
        if norm_correct:
            s.iv_analyzed_info_arr_nnc = copy.deepcopy(s.iv_analyzed_info_arr)
            s.ramp_raw_arr_nnc = copy.deepcopy(s.ramp_raw_arr)
            s.ramp_arr_nnc = copy.deepcopy(s.ramp_arr)
            # did this part just to have an easy plot of differences 
            s.ramp_nnc = s.ramp_arr_nnc[0]
            for i in range(1,len(s.ramp_arr_nnc)):
                s.ramp_nnc = s.merge_ramps(s.ramp_nnc, s.ramp_arr_nnc[i])
            # did THIS part to use fully normal detector resistances
            s.ramp_raw_nnc = s.ramp_raw_arr_nnc[0]
            for i in range(1,len(s.ramp_raw_arr_nnc)):
                s.ramp_raw_nnc = s.merge_ramps(s.ramp_raw_nnc, s.ramp_raw_arr_nnc[i], 
                                               sort_key='temp_raw',
                                               arr_keys=['temp_raw', 'p_satSM', 
                                                         'R_nSM_raw', 'all_fit_R_n_raw'])
            # perform the normal corrections.
            s.do_normal_correction() # updates the non-nnc arrays
            #print('ITO ("normal") correction done')
            
                  
        s.ramp = s.ramp_arr[0] # merging
        for i in range(1,len(s.ramp_arr)):
            s.ramp = s.merge_ramps(s.ramp, s.ramp_arr[i])
        if not norm_correct:
            s.ramp_raw = s.ramp_raw_arr[0]
            for i in range(1,len(s.ramp_raw_arr)):
                s.ramp_raw = s.merge_ramps(s.ramp_raw, s.ramp_raw_arr[i], 
                                               sort_key='temp_raw',
                                               arr_keys=['temp_raw', 'p_satSM', 
                                                         'R_nSM_raw', 'all_fit_R_n_raw'])
            s.calculate_det_R_ns()
        s.setup_key_info() # Has to come last b/c looks at what's loaded. 
        s.categorize_ivs() # can remove if I ever don't need anymore 
        #print("Temp_Ramp fully initialized, child class __init__ next")

    
    
    
    
    # ============== non-data init functions =================
    def setup_key_info(s):
        # name and units 
        s.key_info = { \
            'p_satSM'  : {'name' : 'Pysmurf $P_{bias}$ in transition', 
                          'units' : 'pW', 'lim': [0,s.p_sat_cut]},#12.0/15.0*
            f'p_b{s.use_per}R_n' : {'name' : "$P_{b"+f"{s.use_per}"+"}$,"+r" the $P_{bias}$ at " + f"R={s.use_per}%"+ r"$R_{n}$",
                          'units' : 'pW', 'lim': [0,s.p_sat_cut]}, #12.0/15*
            f'p_b{s.use_per}'    : {'name' : "$P_{b"+f"{s.use_per}"+"}$,"+r" the $P_{bias}$ at " + f"R={s.use_per}%"+ r"$R_{n}$",
                          'units' : 'pW', 'lim': [0,s.p_sat_cut]},#12.0/15*
            'R_nSM_raw': {'name' : 'Pysmurf $R_n$ fit', 
                          'units' : 'Ω', 'lim': [s.expected_R_n_min,s.expected_R_n_max]},
            'R_nSM'    : {'name' : 'Pysmurf $R_n$ fit', 
                          'units' : 'Ω', 'lim': [s.expected_R_n_min,s.expected_R_n_max]}, # Now, iv_analyzed ones
            'v_bias'   : {'name' : 'Bias voltage through bias line full DR', 
                          'units' : 'V'}, #'lim': [0,20]
            'v_tes'    : {'name' : 'TES voltage aka $R_{sh}$ voltage', 
                          'units' : 'μV'}, #'lim': [0,0.5]
            'R'        : {'name' : 'TES resistance', 
                          'units' : 'Ω', 'lim': [0,s.expected_R_n_max]},
            'p_tes'    : {'name' : 'Electrical power on TES', 
                          'units' : 'pW', 'lim': [0,12]},          
            'i_bias'   : {'name' : 'Bias current on bias line', # only in corrected
                          'units' : 'μA', 'lim': [0,1200]},   
            'i_tes'    : {'name' : 'Current through TES', 
                          'units' : 'μA'}, # no limits, varies much with temp.
            'sb'       : {'name' : 'smurf band', 
                          'units' : '', 'lim': [0,8]},
            'ch'       : {'name' : 'smurf channel', 
                          'units' : '', 'lim': [0,512]},
            'sbch'     : {'name' : 'smurf id', 
                          'units' : ''}}    
        if s.ramp_type == 'bath':
            temp_dict = {'name': 'Bath temperature', 'units': 'mK'}     
        elif s.ramp_type == 'coldload':
            temp_dict = {'name': 'Coldload temperature', 'units': 'K'}
        s.key_info['temp_raw']  = temp_dict
        s.key_info[f'temp{s.use_per}R_n'] = temp_dict.copy()
        s.key_info['temp']      = temp_dict.copy()
        s.key_info['temp_raw']['lim'] = [0.95*s.temp_list_raw[0],s.temp_list_raw[-1]+0.05*s.temp_list_raw[0]]
        s.key_info[f'temp{s.use_per}R_n']['lim'] = [0.95*s.temp_list_raw[0],s.temp_list_raw[-1]+0.05*s.temp_list_raw[0]]
        if len(s.temp_list)==0: 
            s.key_info['temp']['lim'] = [0.95*s.temp_list_raw[0],s.temp_list_raw[-1]+0.05*s.temp_list_raw[0]]
        else:
            s.key_info['temp']['lim'] = [0.95*s.temp_list[0],s.temp_list[-1]+0.05*s.temp_list[0]]
        
    # ============== Functions to load data, cut bad IVs, and merge data ==========    
    def load_iv_analyzed(s,metadata):
        # loading iv_analyzed data from file and track bias lines seen.
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        analyzed_iv_info = []
        for line in metadata:
            temp, _, bl, sbs, fp, meas_type = line # bl = bias line, sbs= smurf bands (not subband), fp = file path
            if not meas_type.upper() == 'IV': # cut out the noise measurements.
                continue
            temp = temp.split(" ")[s.metadata_temp_idx]
            #iv_analyzed_fp = fp.replace("/data/smurf_data", "/data/legacy/smurfsrv")
            iv_analyzed_fp = fp.replace("/data/smurf_data", "/data2/smurf_data")
            iv_analyzed_fp = iv_analyzed_fp.replace("iv_raw_data", "iv")
            if s.input_file_type == "guess":                    
                if "iv_info" in iv_analyzed_fp:
                    print('Assuming input_file_type="original_sodetlib"')
                    s.input_file_type = "original_sodetlib"
                else:
                    print('Assuming input_file_type="pysmurf"') 
                    s.input_file_type="pysmurf"
            iv_analyzed_fp = iv_analyzed_fp.replace("iv_info", "iv_analyze") # orig. sodetlib
            if not s.sc_offset and s.input_file_type == "original_sodetlib":
                # We're loading this from raw. 
                iv_analyzed = s.load_raw_iv_analyzed(iv_analyzed_fp)
#                 for sb in iv_analyzed['data'].keys():
#                     for ch, d in iv_analyzed['data'][sb].items():
#                         print(d['i_tes_offset'])
#                         break
#                     break
            else:
                iv_analyzed = np.load(iv_analyzed_fp, allow_pickle=True).item()
            # Now, if this isn't really a pysmurf dictionary, adapt it to that form
            if s.input_file_type == "original_sodetlib":
                iv_analyzed = s.original_sodetlib_to_pysmurf(iv_analyzed)
#                 for sb in iv_analyzed.keys():
#                     if is_float(sb):
#                         for ch, d in iv_analyzed[sb].items():
#                             print(d['i_tes_offset'])
#                             break
#                         break
            # Let's grab times
            ctime = int(re.search(r'.*/(\d+)_iv.*', iv_analyzed_fp).group(1))
            ctime -= 4*60*60 # I guess it's the wrong timezone? 
            human_time = datetime.utcfromtimestamp(ctime).strftime('%Y-%m-%d %H:%M:%S')
            # calibrate temperatures
            temp = float(temp)*s.therm_cal[0] + s.therm_cal[1]
            
            # If doing +V to -V IVs, need to fix pysmurf putting the negative v_bias
            # inputs in as positive, and recalculating anything.
            
            
            analyzed_iv_info.append({'temp':temp, 'bl':int(bl), \
                                     'iv_analyzed_fp':iv_analyzed_fp, \
                                     'iv_analyzed':iv_analyzed,'human_time':human_time})
            # former location of code in count_bls_seen()
            
        return  analyzed_iv_info
    
    

    # ---- three load_iv_analyzed(s,metadata) helper functions:
    def original_sodetlib_to_pysmurf(s, sod_iv_analyzed):
        given = sod_iv_analyzed
        iv_analyzed = {'high_current_mode': \
                       given['metadata']['iv_info']['high_current_mode']}
        for sb in given['data']:
            iv_analyzed[sb] = {}
            for ch, d in given['data'][sb].items():# i_tes not in pysmurf, keepin it 
                iv_analyzed[sb][ch] = copy.deepcopy(d) # we're fixing the things this is wrong for below.
#                 iv_analyzed[sb][ch] = {key:d[key] \
#                                        for key in ["R","R_n","p_tes","i_tes",\
#                                                    "v_bias","v_tes","si","i_tes_offset"]}
                sc_idx, nb_idx = int(d["idxs"][0]),int(d["idxs"][2])
                #v how I discovered that it will send you stuff that has sc_idx > nb_idx
                #print(f"sb{sb}ch{ch} p_tes len{len(d['p_tes'])} idxs{[sc_idx, nb_idx]}")
                if sc_idx > nb_idx:
#                     print(f"sb{sb}ch{ch} p_tes len{len(d['p_tes'])}" +\
#                           f" idxs{[sc_idx, nb_idx]}; swapping idxs")
                    temporary = sc_idx
                    sc_idx = nb_idx
                    nb_idx = sc_idx
                iv_analyzed[sb][ch]['trans idxs'] = [sc_idx, nb_idx]    
                                
                iv_analyzed[sb][ch]['p_trans'] = \
                    np.median(d['p_tes'][sc_idx:nb_idx])
                target_idxs = np.where(d["R"] < 0.007)[0]
                if len(target_idxs) == 0:
                    target_idx = 0 # something is horribly wrong though
                else:  
                    target_idx = target_idxs[-1]
                for targ in ["si","v_bias","v_tes"]:
                    # Si being derivative based, one shorter than R
                    if targ == "si" and target_idx == len(d['si']): 
                        iv_analyzed[sb][ch][targ+"_target"] = d[targ][target_idx-1]
                    else:
                        iv_analyzed[sb][ch][targ+"_target"] = d[targ][target_idx]
        return iv_analyzed

    def load_raw_iv_analyzed(s,iv_analyzed_fp):
        # loads from raw timestreams because that's the only way
        # to truly access data without sc_offset.
        iv_info_fp = iv_analyzed_fp.replace("iv_analyze","iv_info")
        stuff = re.search(r".*/(crate\d+slot(\d+))/.*",iv_info_fp)
        stream_id=stuff.group(1)
        slot_num=int(stuff.group(2))
        cfg = sodetlib.det_config.DetConfig(slot=slot_num)
        #cfg.load_config_files(slot=2)
        # And now we cheat this, since we only need it to supply two values
        # and sadly config wasn't saved:
        cfg.sys = {'g3_dir':"/data2/timestreams", 
                   'slots':{f"SLOT[{slot_num}]":{'stream_id':stream_id}}}
        timestamp,phase,mask,v_bias = det_analysis.load_from_sid(cfg,iv_info_fp)
        # Change this to proper function someday instead of 
        # the copy at the bottom of this file:
        return analyze_iv_info_no_sc_offset(iv_info_fp, timestamp,phase, v_bias, mask,
                    phase_excursion_min=3.0, psat_level=s.use_per/100.0, sc_offset=s.sc_offset,
                                           bin_tod=s.bin_tod)


#     def remove_sc_offset(s,iva):
#         # should really be loading the data without the orig analysis if 
#         # needing offset-less data. See sodetlib to do so. 
#         # note: important to only run this once because of the deletion!
#         # This does not work; needs to properly propagate here if
#         # going to use, I think. Remember can't do this in v_tes!
#         sc_idx, nb_idx = iva['trans idxs']
#         if sc_idx == 0:
#             return iva
#         # does in v_tes because that's what's returned
#         gap = iva['i_tes'][sc_idx] - iva['i_tes'][sc_idx-1]
# #         pre = iva['i_tes'][0] 
#         iva['i_tes'][:sc_idx] += gap
# #         if iva['i_tes'][0] == pre:
# #             print("nogap change")
# #             print("hey" + pre)
#         # take out the point behind sc_idx if there's more than one, so not two identical 
#         # messing up the algorithm
#         data_len = len(iva['i_tes'])
#         rmv = min(sc_idx-1,1) # let's not pull the only sc branch point if there's only one.
#         for key,val in iva.items():
#             if type(val) == type(iva['i_tes']) and len(val) == data_len:
#                 iva[key] = np.concatenate((val[:sc_idx-rmv],val[sc_idx:]))
#         if rmv ==1:
#             iva['trans idxs'][0] -=1 # gotta reset sc_idx
#         return iva
    
    def fix_pysmurf_trans_idx(s, py_ch_iv_analyzed,sb,ch,temp):
        # LEGACY for now
        iv = py_ch_iv_analyzed
        i_bias = iv['v_tes']*(1/s.r_sh + 1/iv['R'])
        resp_bin = i_bias/(iv['R']/s.r_sh + 1) # the current through TES
        sc_idx = iv['trans idxs'][0]
        # the +1 is necessary because the sc_idx could be on either side of the artificial jump
        #correct_normal_index = np.where(resp_bin == min(resp_bin[iv['trans idxs'][0]+1:]))[0][0]
        i_tes_diff_sign = np.sign(np.diff(resp_bin[sc_idx:]))
        # Note that if sc_idx is much smaller than it should be,  could screw this
        # but if I weight the downward slopes too much, crashes on the "IV"s that are noisy lines
        correct_normal_index = sc_idx+dict_balance_idx(i_tes_diff_sign,{-1:1},{1:1})
        # Debugging
        #sc_idx = iv['trans idxs'][0]
        nb_idx_old = iv['trans idxs'][1]
        nb_idx_new = correct_normal_index
        debug = False
        if debug and not nb_idx_old == nb_idx_new and (nb_idx_new<sc_idx or nb_idx_new>nb_idx_old):
            print(f"{temp:.0f} {sb} {ch} sc_idx_old{sc_idx}(i_b{i_bias[sc_idx]:.3},i_t{resp_bin[sc_idx]:.3},R_t{iv['R'][sc_idx]:.3})")
            print(f"{temp:.0f} {sb} {ch} nb_idx_old{nb_idx_old}(i_b{i_bias[nb_idx_old]:.3},i_t{resp_bin[nb_idx_old]:.3},R_t{iv['R'][nb_idx_old]:.3})")
            print(f"{temp:.0f} {sb} {ch} nb_idx_new{nb_idx_new}(i_b{i_bias[nb_idx_new]:.3},i_t{resp_bin[nb_idx_new]:.3},R_t{iv['R'][nb_idx_new]:.3})")
        # Uncomment the below once debugged
        #if correct_normal_index > sc_idx and correct_normal_index<=:
        # I'm cheating here.
        if correct_normal_index > sc_idx+2 and correct_normal_index <= nb_idx_old: #len(resp_bin): # CHEATER
            iv['trans idxs'] = [iv['trans idxs'][0], correct_normal_index]
        else:
            if not sb in s.idx_fix_fails.keys():
                s.idx_fix_fails[sb]={}
            if ch not in s.idx_fix_fails[sb].keys():
                s.idx_fix_fails[sb][ch]={}
            s.idx_fix_fails[sb][ch][temp] = \
                {'sc_idx_old':sc_idx,'nb_idx_old':nb_idx_old,'nb_idx_new':nb_idx_new}
            #print(f"cheated idx_fix: {temp:.0f} {sb} {ch} sc_idx_old{sc_idx} nb_idx_old{nb_idx_old} nb_idx_new{nb_idx_new}")
        return iv
    
    

        
    # ------ standalone cutting functions
    
    def pre_p_sat_cuts(s,d,nb_idx,save_cut_info=False):
        # standalone_IV_cuts helper function
        # now:
        #indy = np.where(d['p_tes']>s.p_sat_cut)[0] 
        lp = len(d['p_tes'])
        nb_far_idx = int(nb_idx+(lp-nb_idx)*3/4)
        p_marg = min(s.p_sat_cut,d['p_tes'][nb_far_idx])
        indy = np.where(d['p_tes']>p_marg)[0]#  | (d['p_tes']>s.p_sat_cut))[0] 
        
        ind = indy[np.where(indy > nb_idx)[0]] # don't get wrecked by low %R_n instability.
        # for cutting ones when 
        if len(ind)==0: # there wasn't really bias line power, probably?
            if save_cut_info:
                d['cut_info'] = 'no p_tes past nb_idx > d["p_tes"][nb_idx+int((lp-nb_idx)*3/4)])'
            return False 
        # The below (unfortunately) cuts plots that have wild upswings of power 
        # in/after the 'transition.' Originally 7e-3 but that hits some real ones. 
        # ^ should no longer do this now that I'm bounding with nb_idx
        # 6e-3 worked for a while...
        if np.min(d['R'][ind]) < s.expected_R_n_min: # #edit: disabling it causes problems.  
            if save_cut_info:
                d['cut_info'] = 'too small R_n'
            return False
        # The file stores p_sat and R in
        # reverse time-order to how it's taken; superconducting first.
        # So the below is checking that the normal resistance is steady.
        # problem is, if normal branch small enough, it literally might not GET to a steady R_n.
        # originally, "np.std(d['R'][-100:]) > 1e-4"; definitely some do not have that many points.
        if np.std(d['R'][nb_far_idx:]) > 1e-4:  
            if save_cut_info: 
                d['cut_info'] = 'std(R in normal branch) >1e-4'
            return False 
        return True
        
        
    # called count_bls for tes_dict setup, and load_temp_sweep_IVs's helper functions for cuts:
    def standalone_IV_cut_save(s,d):
        ans = s.standalone_IV_cut(d)
        d['standalone'] = ans
        return ans
    
    def standalone_IV_cut(s,d):
        # TODO: GET THE FULL CONTEXTLESS IV EVALUATOR IN HERE!
        # Jack lashner said cutting on R_n was effective...and it does seem to be!
        # Well, for Mv13. Less so for Mv5. Not sure I should rely on this. 
        #if d['R_n'] <
        
         
        if not 'i_tes' in d.keys():
            d['i_tes'] = d['v_tes']/d['R']
        # is it just noise on a wrong bl? A really crazy sc_off might confuse this, 
        # but that's rare for this type of issue
        if max(d['i_tes'])-min(d['i_tes']) < 2: # uA
            d['cut_info'] = "trivial_resp"
            return False
        
        # This whole thing below should really use nb_idx to set normal branch start. 
        pass_pre_p_sat_cuts = False
        p_sat  = False
        sc_idx, nb_idx = d['trans idxs']
        if s.use_cii:
            cii_yes = False
            if (not s.redo) and 'contextless_is_iv' in d.keys():
                cii_yes = d['contextless_is_iv']
            else:
                cii_result = contextless_is_iv(d,save_to_iva=True,debug=False)
                #contextless_is_iv only writes iva keys that begin with 'contextless...'
                # So doesn't overwrite any p_satinfo.
                if type(cii_result) == list: # is an iv curve
                    cii_yes = True
            if cii_yes:
                cii_sc_idx , cii_nb_idx = d['contextless_idxs']
                cii_nb_idx = sum(d['contextless_mask'][:cii_nb_idx]) #work with masked data
                # TODO: flip it and phase_offset correct if polarity is wrong!!! 
                cii_d = {key:d[key][np.where(d[key]*d['contextless_mask'] != 0)[0]] \
                         for key in ['R','p_tes']}

                pass_pre_p_sat_cuts = s.pre_p_sat_cuts(cii_d,cii_nb_idx,save_cut_info=True)
                if pass_pre_p_sat_cuts:
                    p_sat = s.calc_P_b90(d['R_n'],cii_d['R'],cii_d['p_tes'])
        if not pass_pre_p_sat_cuts:
            pass_pre_p_sat_cuts = s.pre_p_sat_cuts(d,nb_idx,save_cut_info=True)
            if not pass_pre_p_sat_cuts:
                return False
            p_sat = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
            
        # Original:
#         ind = np.where(d['p_tes']>s.p_sat_cut)[0] # for cutting ones when 
#         if len(ind)==0: # there wasn't really bias line power, probably?
#             d['cut_info'] = 'no p_tes > s.p_sat_cut'
#             return False 
#         # The below cuts plots that have wild upswings of power 
#         # in/after the 'transition.' Originally 7e-3 but that hits some real ones. 
#         # screw it, just do contextless_is_iv.
#         # let's avoid cutting low-%R_n noise instability:
#         if np.min(d['R'][ind]) < 6e-3: # #edit: disabling it causes problems.  I've disabled it because the ones with this instability are often still useful at higher %R_n.
#             d['cut_info'] = '
#             return False
#         if np.std(d['R'][-100:]) > 1e-4: # The file stores p_sat and R in 
#             return False # reverse time-order to how it's taken; superconducting first.
#         # So the above is checking that the normal resistance is steady. 
        # now instead:
        

        # cuts on p_sat
        if not p_sat or np.isnan(p_sat): #np.isnan....pretty sure that shouldn't ever happen, but hey. 
            d['cut_info'] = 'could not calculate p_sat'
            return False
        #p_sat = d['p_tes'][p_sat_idx]
        # Don't change the min p_sat to above 0, you'll junk lots of good fits.  
        # sane P_sats. Daniel had P_sat > 10, I'll take up to 15 pW for MF; like 50 for UHF
        if p_sat > s.p_sat_cut: 
            d['cut_info'] = f'p_b{s.use_per} > p_sat_cut ({s.p_sat_cut})'
            return False
        if p_sat < 0:
            d['cut_info'] = f'p_b{s.use_per}<0'
            return False
        
        if s.use_p_satSM:
            if d['p_trans'] < 0 or d['p_trans'] > s.p_sat_cut: # Sane smurf/py P_sats. upped to 15 pW as well
                d['cut_info'] = 'p_satSM<0 or p_satSM > p_sat_cut'
                return False
            # If there's a BIG difference in py_p_sat and 90% R_n P_sat, something is probably wrong. 
            #if s.key_info['temp'][1] = 
            if max(p_sat,d['p_trans'])/min(p_sat,d['p_trans']) > 2 and \
               max(p_sat,d['p_trans'])-min(p_sat,d['p_trans']) > 2:
                d['cut_info'] = f'p_b{s.use_per} very different p_satSM'
                return False
        # It's crazy that this actually hits things that passed the previous.
        # but rarely, it does (ex 5,50 of mv5_CL10_bath_ramp):
        if s.use_cii and cii_yes: 
            lp = len(cii_d['p_tes'])
            if not (lp-cii_nb_idx <=3 or cii_nb_idx - cii_sc_idx <=3):
                return p_sat
        lp = len(d['p_tes'])
        if lp-nb_idx <=3 or nb_idx -sc_idx <=3:
            d['cut_info'] = 'nb or transition size <=3 points'
            return False
        return p_sat
    
    
    
    # -------- The cut controller and ramp_raw_arr and ramp_arr loader
    def load_temp_sweep_IVs(s): 
        # decides what bl each channel is on if that not given by map, performs cuts,
        # and sets up ramp_raw_arr and ramp_arr
        s.count_bls_seen()
        s.test_device.calculate_bls()
        
        for ramp_run in s.iv_analyzed_info_arr:
            bathsweep_raw = {} # originally just made this for bathsweep. 
            bathsweep = {}
            now_bad_IVs = {} # sb: bad channel list. Cuts remaining data from channel if it throws something bad.
            for iai in ramp_run:
                s.cut_iv_analyzed(bathsweep_raw,bathsweep,now_bad_IVs,iai)
            s.ramp_raw_arr.append(bathsweep_raw)
            s.ramp_arr.append(bathsweep)
    
    def count_bls_seen(s): 
        for metadata in s.iv_analyzed_info_arr:
            for run_d in metadata:
                iv_analyzed = run_d['iv_analyzed']
                bl = run_d['bl']
                temp = run_d['temp']
                for sb in iv_analyzed.keys():
                    if sb == "high_current_mode":
                        continue
                    for ch,d in iv_analyzed[sb].items():
                        # REMOVE THIS next line IF the superconducting offset thing comes out!!!
                        #iv_analyzed[sb][ch] = s.remove_sc_offset(iv_analyzed[sb][ch])
                        iv_analyzed[sb][ch]['R_nSM'] = d['R_n']
                        # Stuff that has to be in iva for plot_key_v_key_by_key
                        iv_analyzed[sb][ch]['bl'] = int(bl)
                        iv_analyzed[sb][ch]['temp'] = run_d['temp']
                        iv_analyzed[sb][ch]['sb'] = sb
                        iv_analyzed[sb][ch]['ch'] = ch
                        iv_analyzed[sb][ch]['sbch'] = (sb,ch)
                        s.fix_neg_v_b(iv_analyzed[sb][ch])
                        if s.fix_pysmurf_idx:
                            iv_analyzed[sb][ch] = s.fix_pysmurf_trans_idx(d, sb,ch,temp)

                        if sb not in s.test_device.bls_seen.keys():
                            s.test_device.bls_seen[sb] = {}
                        if ch not in s.test_device.bls_seen[sb].keys():
                            s.test_device.bls_seen[sb][ch] = []
                        s.test_device.bls_seen[sb][ch].append(int(bl)) # tracks bls seen
                        
                        if s.standalone_IV_cut_save(d): # that would have saved cut reason.
                            if sb not in s.test_device.bls_standalone.keys():
                                s.test_device.bls_standalone[sb] = {}
                            if ch not in s.test_device.bls_standalone[sb].keys():
                                s.test_device.bls_standalone[sb][ch] = []
                            s.test_device.bls_standalone[sb][ch].append(int(bl)) # tracks bls seen
    
    def fix_neg_v_b(s,iv):
        # pysmurf rawest is 'v_bias' and 'v_tes', also has 'p_tes', 'R'
        # sodetlib also has 'i_tes', which is rawer.
        # my sodetlib loader should fix it before it gets here.
        
        diffy = np.diff(iv['v_bias'])
        if len(np.where(diffy<0)[0]) <= 0:
            if 'pvb_idx' not in iv.keys():
                iv['pvb_idx'] = np.where(iv['v_bias'] >=0)[0][0]
            return iv 
        # Does do zero removal? but still seems to have some np.nan? Maybe?
        flip_idx = np.where(diffy < 0)[0][-1] + 1 
        iv['pvb_idx'] = flip_idx
        for key, item in iv.items():
            try:
                if len(item) >10:
                    iv[key] = np.concatenate((iv[key][:flip_idx], iv[key][flip_idx+1:]))
            except:
                pass
        for i in [0,1]:
            if iv['trans idxs'][i] == len(iv['v_bias']):
                iv['trans idxs'][i] = iv['trans idxs'][i] - 1
        

        iv['v_bias'][:flip_idx] *= -1.0

        iv['i_bias'] =  iv['v_tes']*(1/s.r_sh + 1/iv['R'])
        # have to extract resp_bin with UNFLIPPED i_bias first
        # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
        resp_bin = iv['i_bias']/(iv['R']/s.r_sh + 1) # i_tes, effectively.
        iv['i_tes'] = resp_bin # not doing any ITO right now.
        # now flip i_bias
        iv['i_bias'][:flip_idx]*= -1.0
        iv_py = iv #copy.deepcopy(iv) # that is a bit awkward, but let's try it
        iv = fill_iva_from_preset_i_tes_i_bias_and_r_n(s,iv_py, iv)

        
    def cut_iv_analyzed(s,bathsweep_raw,bathsweep,now_bad_IVs,iai):
        # names because I originally made it just for bathsweep.
        # iai an iv_analyzed_info 
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        temp, bl, iv_analyzed = iai['temp'],iai['bl'],iai['iv_analyzed']
        for sb in iv_analyzed.keys():
            if sb=="high_current_mode":
                continue
            # prep now_bad_IVs:
            if sb not in now_bad_IVs.keys():
                now_bad_IVs[sb] = []
            for ch, d in iv_analyzed[sb].items(): # d for dictionary.
                # Let's update the mux_map.
                # b/c Kaiwen's map didn't contain all of these.  
                s.test_device.check_mux_map_for_channel_and_init(sb, ch)
                
                # Because axis manager is amazing, basically:
                d['sb'] = sb
                d['ch'] = ch
                d['temp'] = temp
                d['bl'] = bl # I think this one's added elsewhere too...
                
                
                # don't do anything when there definitely wasn't actually power in the bias line.
                # actually...just get rid of these if we knew their bias line. 
                # This does potentially keep bad ones from Kaiwen's omissions. 
                # hopefully that won't be a problem. 
                # actually...Kaiwen's bl assignment could be wrong if she typod. 
                # it would be better to do this 
                if not int(mux_map[sb][ch]['biasline']) == int(bl):
                    d['cut_info'] = 'wrong_bl'
                    continue # yes, it does sometimes think there's an IV curve when no power was being run.
                
                # Otherwise, Load data into bathsweep_raw and temp_list_raw no matter what.  
                # (first do advanced R_n calculation if norm correcting):
                if s.norm_correct:
                    s.all_curve_lin_fit(d)
                s.update_temp_list(temp, 'temp_list_raw')
                p_b90R_n_orig = s.add_ramp_raw_iv(bathsweep_raw, temp, sb, ch, d) 

                # Now exit if the channel's been killed
                if ch in now_bad_IVs[sb]:
                    d['cut_info'] = 'channel was killed'
                    continue

                # Now, check for a valid IV curve. 
                if not s.redo:
                    p_sat = d['standalone'] # count_bls() calls standalone_IV_cut_save)
                else:
                    p_sat = s.standalone_IV_cut_save(d)
                # Slightly hacky since a restructuring would be better,
                # but this should ensure prev_py_p_sat_ind exists when relevant.
                if p_b90R_n_orig and not p_b90R_n_orig == p_sat:
                        bathsweep_raw[sb][ch][f'p_b{s.use_per}R_n'][-1] = p_sat
                if not p_sat:
                    # should have already saved the cut type.
                    continue
                

                if sb in bathsweep.keys() and ch in bathsweep[sb].keys() and len(bathsweep[sb][ch][f'p_b{s.use_per}'])>0:
                    # cut on increasing 90% R_n p_b and kills further data reading from the channel if that occurs
                    # Pysmurf p_sat is a more variable than 90% R_n p_sat when things are working, so 
                    # it doesn't kill on this stuff.
                    prev_p_sat = bathsweep[sb][ch][f'p_b{s.use_per}'][-1] 
                    if s.ramp_type == "bath" and p_sat > prev_p_sat: 
                        now_bad_IVs[sb].append(ch)
                        d['cut_info'] = 'p_sat > prev_p_sat'
                        continue
                    elif s.ramp_type =="coldload" and p_sat > 1.1*prev_p_sat:
                        now_bad_IVs[sb].append(ch)
                        d['cut_info'] = 'p_sat > 1.1*prev_p_sat'
                        continue
                    if s.use_p_satSM:
                        # cut, but not kill, on SIGNIFICANTLY increasing pysmurf p_sat since last good.
                        # prev_py_p_sat set below when a point is accepted. 
                        try: # from some issues with ptn
                            prev_py_p_sat_ind = np.argwhere(bathsweep_raw[sb][ch][f'p_b{s.use_per}R_n']==prev_p_sat)[0][0]
                        except IndexError: #s.dName[:4] == "Mv34" or s.dName[:4] == "Mv35":
                            print(f"{sb} {ch} {temp}\nramp: {bathsweep[sb][ch]}\nramp_raw: {bathsweep_raw[sb][ch]}")
                        prev_py_p_sat = bathsweep_raw[sb][ch]['p_satSM'][prev_py_p_sat_ind] 
                        if d['p_trans'] > 1.5*prev_py_p_sat:
                            d['cut_info'] = 'p_satSM > 1.5*prev_p_satSM'
                            continue 
                
                # I think this might be too aggressive with the cuts.
                # more sophisticated version of above using slopes:
                # But the slopes are too gentle in coldload ramps. 
                if s.ramp_type == 'TEMPORARILY DISABLED' and sb in bathsweep.keys() and ch in bathsweep[sb].keys() and len(bathsweep[sb][ch][f'p_b{s.use_per}'])>1:
                    # cut on notably slope-increasing 90% p_sat and DON'T kill further data reading from the channel if that occurs
                    # Pysmurf p_sat is a more variable than 90% R_n p_sat when things are working, so 
                    # it doesn't kill on this stuff.
                    prev_p_sat_slope = (bathsweep[sb][ch][f'p_b{s.use_per}'][-1]-bathsweep[sb][ch][f'p_b{s.use_per}'][-2])/ \
                                        (bathsweep[sb][ch]['temp'][-1]-bathsweep[sb][ch]['temp'][-2])
                    this_p_sat_slope = (p_sat-bathsweep[sb][ch][f'p_b{s.use_per}'][-1])/ \
                                        (temp-bathsweep[sb][ch]['temp'][-1])
                    if this_p_sat_slope > prev_p_sat_slope*0.7: # slopes are negative; keep a little fudge factor
                        #now_bad_IVs[sb].append(ch)
                        # It will likely kill itself with this if necessary, b/c denom just getting bigger.
                        d['cut_info']= 'this_p_sat_slope > prev_p_sat_slope*0.7'
                        continue
                    # doing this with pysmurf is annoying, start with just 90R_n
                    # cut, but not kill, on SIGNIFICANTLY increasing pysmurf p_sat slope since last good.
#                         prev_py_p_sat_ind = np.argwhere(bathsweep_raw[sb][ch][f'p_b{s.use_per}R_n']==prev_p_sat)[0][0]
#                         #print(prev_py_p_sat_ind)
#                         prev_py_p_sat = bathsweep_raw[sb][ch]['p_satSM'][prev_py_p_sat_ind] 
#                         if d['p_trans'] > 1.5*prev_py_p_sat:
#                             continue 
#                         prev_py_p_sat = d['p_trans']

                # Passed cuts -- include it!       
                s.add_ramp_iv(bathsweep, temp, sb, ch, d, p_b90=p_sat) 
                s.update_temp_list(temp, 'temp_list') # Also, update temperature list 
                # What about updating tes_dict? In theory mux_map should contain all tes
                # that get iv curves, but Kaiwen's map didn't. 
                # That said, it's possible (though unlikely-not that unlikely actually) for one crazy 'IV' to get through cuts on non-tes.
                # 'TES_freq','pol','opt','opt_name','linestyle' <-required.
                # s.test_device.check_tes_dict_for_channel_and_init(sb, ch)                     
    
    def all_curve_lin_fit(s,iva):
        # Is this curve on a fully normal detector? If so, what are its stats?
        if  'i_bias' not in iva.keys():
            iva['i_bias'] =  iva['v_tes']*(1/s.r_sh + 1/iva['R'])
        if 'i_tes' not in iva.keys():
            iva['i_tes'] = iva['i_bias']/(iva['R']/s.r_sh + 1)
        if s.sc_offset == False:
            i_bias, i_tes = iva['i_bias'], iva['i_tes']
        if s.sc_offset == True:
            sc_idx = iva['trans idxs'][0]
            i_bias = np.concatenate((iva['i_bias'][:sc_idx-1], iva['i_bias'][sc_idx:]))
            diffy = iva['i_tes'][sc_idx]-iva['i_tes'][sc_idx-1]
            i_tes = np.concatenate((iva['i_tes'][:sc_idx-1] + diffy, iva['i_tes'][sc_idx:]))
        # It's too easy to get things that still have a small transition in them passing.
        # have to cut it. 
        if s.use_frac_R_n > 0:
            fit_st = -int(len(i_bias)*s.use_frac_R_n)
        else:
            fit_st = 0
        slope, intercept, r_value, p_value, _ = scipy.stats.linregress(i_bias[fit_st:],i_tes[fit_st:])
        # Still want stderr from the whole thing
        _, _, _, _, std_err = scipy.stats.linregress(i_bias,i_tes)
        iva['all_fit_R_n'] = s.r_sh*( 1/slope -1)
        # add i_tes_offset to get unaltered data. It was subtracted from raw.  
        iva['all_fit_ITO'] =  intercept
        if 'i_tes_offset' in iva.keys(): # might not have been loaded raw...
            iva['all_fit_ITO'] = iva['i_tes_offset'] + intercept
        iva['all_fit_stats'] = {'r':r_value, 'p': p_value, 'std_err':std_err, 'r_log':np.log(1-r_value)}
        
        # NOW, is that a fully normal detector?
        # Note false positives are probably more problematic here than false negatives,
        # because of how used in normal correction/finding ITO.
        # Settled on chosen conditions based on uv8_low, uv8, Mv20_PH003, Mv5_PG009_CL_10K
        if s.expected_R_n_min <= iva['all_fit_R_n'] \
           and iva['all_fit_R_n'] <= s.expected_R_n_max \
           and np.log(std_err) <= -10:
            iva['is_normal'] = True
        else:
            iva['is_normal'] = False
        # debug from Mv35
#         if iva['sb'] == 0 and iva['ch'] == 7:
#             print(str(iva['is_normal'])+" ".join([pu.round_to_sf_str(val,5) for val in [iva['temp'], slope, 
#                                                  intercept, std_err, iva['R_nSM'], 
#                                                  iva['R_n'], iva['all_fit_R_n']]]))
            
        return iva
    
    
    def calc_P_b90(s,r_n,r_tes,p_tes): 
        # last two arrays.
        p_sat_idx = np.where(r_tes < s.use_per/100.0 * r_n)[0] # standard SO P_sat def when use_per=90
        if len(p_sat_idx) == 0:
            return False
        return  p_tes[p_sat_idx][-1] # again, data is time-reversed
                    
    def add_ramp_raw_iv(s, bathsweep_raw, temp, sb, ch, d):
        if sb not in bathsweep_raw.keys():
               bathsweep_raw[sb] = {}
        if ch not in bathsweep_raw[sb].keys():
            bathsweep_raw[sb][ch] = {"temp_raw":[], "p_satSM":[],
                                     "R_nSM_raw":[], "all_fit_R_n_raw":[], # -42 if iva not 'is_normal'
                                     f"temp{s.use_per}R_n":[],f"p_b{s.use_per}R_n":[]}
        bathsweep_raw[sb][ch]['temp_raw'].append(temp)
        bathsweep_raw[sb][ch]['p_satSM'].append(d['p_trans']) #Use pysmurf P_sat, since guaranteed to exist.
        bathsweep_raw[sb][ch]['R_nSM_raw'].append(d['R_nSM'])
        # If we're doing normal correction, gotta 
        if 'all_fit_R_n' not in d.keys() or not d['is_normal']:
            bathsweep_raw[sb][ch]['all_fit_R_n_raw'].append(-42)
        else:
            bathsweep_raw[sb][ch]['all_fit_R_n_raw'].append(d['all_fit_R_n'])
        # Try to also add the 90% R_n p_sat
        p_b90 = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
        if p_b90:
            bathsweep_raw[sb][ch][f'temp{s.use_per}R_n'].append(temp)
            bathsweep_raw[sb][ch][f'p_b{s.use_per}R_n'].append(p_b90)
            return p_b90
        return False
        
        
    def add_ramp_iv(s,bathsweep, temp, sb, ch, d, p_b90=-42):
        if sb not in bathsweep.keys():
            bathsweep[sb] = {}
        if ch not in bathsweep[sb].keys():
            bathsweep[sb][ch] = {"temp":[], f"p_b{s.use_per}":[], 'R_nSM':[]}
        bathsweep[sb][ch]['temp'].append(temp)
        if p_b90 == -42:
            p_b90 = s.calc_P_b90(d['R_n'],d['R'],d['p_tes'])
        bathsweep[sb][ch][f'p_b{s.use_per}'].append(p_b90)
        bathsweep[sb][ch]['R_nSM'].append(d['R_nSM'])
    
    def do_normal_correction(s): 
        # corrects for pysmurf's poor fitting of the normal branch
        # That is: this is what gets ITO setup. 
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        s.calculate_det_R_ns() # figure out what each detector's R_n is. 
        # now we need to change: [for each metadata, <stuff in comments>]
        # iv_analyzed_info_arr: [for each temp: 'iv_analyzed_fp']
        # AND: we reconstruct s.ramp_raw_arr and s.ramp_arr using the IV adders, 
        # referencing s.ramp_raw_arr_nnc and s.ramp_arr_nnc for which
        # temperatures had valid IV curves. 
        s.ramp_raw_arr = []
        s.ramp_arr = []
        for run_idx in range(len(s.iv_analyzed_info_arr_nnc)):
            iv_analyzed_info = s.iv_analyzed_info_arr[run_idx]
            orig_ramp = s.ramp_arr_nnc[run_idx] # needed to check if this real IV
            s.ramp_raw_arr.append({})
            s.ramp_arr.append({})   
            bathsweep_raw = s.ramp_raw_arr[run_idx]
            bathsweep = s.ramp_arr[run_idx]
            #ramp_raw = s.ramp_raw_arr()
            # currently just forcing R_n, nothing fancy with c:
            for line in iv_analyzed_info:
                iv_analyzed = line['iv_analyzed']
                temp = line['temp']
                for sb in iv_analyzed:
                    if sb == 'high_current_mode':
                        continue
                    for ch, iv in iv_analyzed[sb].items():
                        # we can only do meaningful corrections
                        # for channels that have actual iv curves. 
                        # if sb in tes_dict.keys() and ch in tes_dict[sb].keys()\
                        # and 'R_n' in tes_dict[sb][ch].keys():
                        # ...Let's apply the correction to everything that can get it, actually.
                        if sb in s.det_R_n.keys() and ch in s.det_R_n[sb].keys():
                            sc_idx, nb_idx = iv['trans idxs'][0],iv['trans idxs'][1]
                            # v hot fix: **TODO** real fix!! #py_start_idx == 0: 
                            lp = len(iv['p_tes'])
                            norm_idx_too_close = (lp-nb_idx <=3 or nb_idx -sc_idx <=3)
                            
                            if 'contextless_idxs' in iv.keys():
                                #cii_sc_idx, cii_nb_idx = 0,0
                                cii_sc_idx, cii_nb_idx = iv['contextless_idxs']
                                # convert to masked versions:
                                cii_sc_idx = sum(iv['contextless_mask'][:cii_sc_idx])
                                cii_nb_idx = sum(iv['contextless_mask'][:cii_nb_idx])
                                cii_idx_too_close = (lp-cii_nb_idx <=3 \
                                                        or cii_nb_idx -cii_sc_idx <=3)
                                if not(norm_idx_too_close == cii_idx_too_close):
                                    print(f"({sb}, {ch}, {temp:.1f}, bl={line['bl']}) "+\
                                          f"idx_too_close:{norm_idx_too_close} cii_idx_too_close:{cii_idx_too_close}")
                            
                            # the rest of the hot fix
                            do_correct = False
                            if ('is_normal' in iv.keys() and iv['is_normal'] and s.use_frac_R_n > 0):
                                    py_start_idx = int(-lp*s.use_frac_R_n)
                                    do_correct = True
                            if (not norm_idx_too_close):
                                # maybe check if cut reason changes.
                                do_correct=True
                                py_start_idx = -int((lp-nb_idx)/2)
                                # DON'T fix if it's normal, no use_frac_R_n!
                                if 'is_normal' in iv.keys() and iv['is_normal']:
                                    do_correct=False
                            if do_correct:
                                iv_analyzed[sb][ch] = \
                                    s.pysmurf_iv_data_correction(iv,
                                            nb_start_idx=py_start_idx,
                                            force_R_n=s.det_R_n[sb][ch])
                                # Definitely redo this. 
                                # THe below DOESN"T matter, will be redone 
                                # elsewhere if would matter
                                # And leaving it here slows cii a lot
                                #s.standalone_IV_cut_save(iv_analyzed[sb][ch])
# **TODO**: Import my formal slope_fix_correct_one_iv instead!!
#                                 iv_analyzed[sb][ch] = \
#                                     slope_fix_correct_one_iv(iv,r_sh=s.r_sh,
#                                             nb_start_idx=py_start_idx,
#                                             force_R_n=s.det_R_n[sb][ch])
#                             else: 
#                                 print(f"SKIPPING R_n-correct: temp{temp} sb{sb}ch{ch} p_tes len{lp} idxs{iv['trans idxs']}")
                        # NOTE that this does NOT add in corrected ones 
                        # that get better standalone cuts. 
                        if sb in s.ramp_raw_arr_nnc[run_idx] \
                        and ch in s.ramp_raw_arr_nnc[run_idx][sb] \
                        and temp in s.ramp_raw_arr_nnc[run_idx][sb][ch]['temp_raw']: 
                            s.add_ramp_raw_iv(bathsweep_raw, temp, sb, ch, iv_analyzed[sb][ch])
                        if  line['bl'] == mux_map[sb][ch]['biasline'] \
                        and sb in s.ramp_arr_nnc[run_idx].keys() \
                        and ch in s.ramp_arr_nnc[run_idx][sb].keys() \
                        and temp in s.ramp_arr_nnc[run_idx][sb][ch]['temp']: 
                            s.add_ramp_iv(bathsweep, temp, sb, ch, iv_analyzed[sb][ch])
    
    def calculate_det_R_ns(s):
        '''this used to just take the R_n calculated from the highest-bath temperature
        # run recognized as an IV curve in the pre-normal correction cuts.
        # This was problematic with Uv8 285 GHz, which often didn't recognize
        # ANY of the curves as legit IV curves because the ITO (I_TES Offset)
        # issue was so bad. 
        
        Now, for each IV, it checks to see if fitting a line to the entire 
        curve (with the sc_offset fudged out) seems to indicate a fully normal detector
        (see all_curve_lin_fit() for criteria). If so, though, it only uses the
        upper half of the curve for the "all_fit" fitting. For each sb-ch, it takes 
        R_n to be the 
        R_n from the highest bath temperature curve that either a) is fully
        normal (thus it uses all_fit_R_n) or b) is classified as an actual 
        IV curve (using R_nSM_raw) if such a curve exists. If no such 
        curve exists, it gives up on that sb-ch. If the highest bath temp point
        with a valid R_n has both an all_fit_R_n and an R_nSM_raw, and the 
        R_nSM_raw is in [s.expected_R_n_min, s.expected_R_n_max], it uses the R_nSM_raw. 
        
        Hence iterating over ramp_raw_nnc.
        '''
        tes_dict = s.tes_dict
        if s.norm_correct:
            ramp_raw_nnc, ramp_nnc = s.ramp_raw_nnc, s.ramp_nnc
        else:
            ramp_raw_nnc, ramp_nnc = s.ramp_raw, s.ramp
        for sb in ramp_raw_nnc.keys():
            if sb not in s.det_R_n.keys():
                s.det_R_n[sb] = {}
            for ch, d in ramp_raw_nnc[sb].items():
                cut_R_n_SMs = []
                if sb in ramp_nnc.keys() and ch in ramp_nnc[sb].keys():
                    cut_R_n_SMs = ramp_nnc[sb][ch]['R_nSM']
                for i in range(len(d['temp_raw'])-1, -1, -1):
                    # temporarily disabling the all_fit thing to see if that's the problem
                    # it is...and should probably be replaced by superconducting+TANO, because
                    # the false positives are just too problematic. 
                    if s.use_frac_R_n > 0:
                        if not d['all_fit_R_n_raw'][i] == -42:
                            if d['R_nSM_raw'][i] in cut_R_n_SMs \
                               and s.expected_R_n_min <= d['R_nSM_raw'][i] \
                               and d['R_nSM_raw'][i] <= s.expected_R_n_max:
                                s.det_R_n[sb][ch] = d['R_nSM_raw'][i]
                                break
                            else: # is an all_fit_R_n_raw, no believable s.det_R_n[sb][ch] = d['all_fit_R_n_raw'][i]
                                s.det_R_n[sb][ch] = d['all_fit_R_n_raw'][i]
                                break
                    if d['R_nSM_raw'][i] in cut_R_n_SMs:
                        s.det_R_n[sb][ch] = d['R_nSM_raw'][i]
                        break 
                if ch in s.det_R_n[sb].keys() and sb in tes_dict.keys() \
                   and ch in tes_dict[sb].keys(): # note don't force it here, b/c not guaranteed fit.
                    tes_dict[sb][ch]['R_n'] = s.det_R_n[sb][ch]
                    
    
    def pysmurf_iv_data_correction(s, py_ch_iv_analyzed,\
                                   nb_start_idx=0,nb_end_idx=-1,force_R_n=-42.0,
                                  force_c=-42.0):
        # **TODO**: Import my formal slope_fix_correct_one_iv instead!!
        # nb_end_idx exists because I often saw this little downturn in the 
        # iv curve at the very highest v_bias. I'm not sure why, but it's in mv6 too.
        # takes a pysmurf data dictionary for a given (sb,ch), 
        # backcalculates the rawer values, makes corrected & expanded version.
        # Except I am not yet dealing with the responsivity calculations. 
        # And, if r_n provided, forces the norm fit line's slope to be 1/(r_n/r_sh+1)
        # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
        # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]])
        r_sh = s.r_sh 
        iv = {} # what I will return. iv data
        iv_py = py_ch_iv_analyzed
        # first, things I don't need to change, really:
        # Arguably should copy everything...
        for key in ['trans idxs','v_bias','R_nSM','bl']:
            iv[key] = iv_py[key]
        # I don't think I want to copy the standalone here, honestly...maybe redo.
        # as of 11/10/2021, pysmurf's v_bias_target is bugged, 
        # hence not just copying iv's value for it.
        
        #  Now, get fundamentals: v_bias_bin, i_bias_bin [uA], and i_tes [uA] (with an offset) = resp_bin  
        # v_tes = i_bias_bin*1/(1/R_sh + 1/R), so i_bias_bin = v_tes*(1/R_sh + 1/R)
        iv['i_bias'] =  iv_py['v_tes']*(1/s.r_sh + 1/iv_py['R'])
        # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
        resp_bin = iv['i_bias']/(iv_py['R']/r_sh + 1)
        # Now the moment we're all waiting for, fixing resp_bin's offset. 
        # A fit really isn't the best R_n estimate. 
        # The best R_n estimate is the last resistance in a proper R calculation.
        # But I can't get that without making some R_n estimate to get c. 
        #nb_fit_idx = int(s.r_n_start_percent*len(resp_bin))# Ours here. Not pysmurf's.
        #nb_fit_idx = s.r_n_start_idx # defaults to -10. 
        if (not force_c == -42) and (not force_R_n==-42):
            r_n = force_R_n
            norm_fit = [1/(r_n/r_sh+1),force_c]
        else:
            if nb_start_idx: # caller wants a specific nb_start_idx
                nb_fit_idx = nb_start_idx
            else:
                nb_idx = iv['trans idxs'][1] # point with minimum derivative btw sc_idx and end
                py_start_idx = -int((len(iv['i_bias'])-nb_idx)/2)
                nb_fit_idx = py_start_idx # TODO: do the derivative thing!!!
                # Katie does some smoothing.
                d_resp = np.diff(resp_bin)
                # looking for the place where second discrete derivative is flat 
                # and constant the same.
                # TODO!!!!!! this is still taken from Katie mostly. Needs work!!
#                 try:
                new_index = nb_fit_idx
                old_index = nb_fit_idx
                new_test = np.polyfit(np.arange(len(d_resp[new_index:-10])),
                                      d_resp[new_index:-10], deg=1)
                while new_index <= -20: # originally -10
                    old_test = new_test
                    old_index = new_index

                    new_index = int(0.75*new_index)

                    new_test = np.polyfit(np.arange(len(d_resp[new_index:-10])), \
                                      d_resp[new_index:-10], deg=1)
                    delta_slope = (new_test[1]-old_test[1])/old_test[1]
                    if np.abs(delta_slope)<=0.02: # 0.05:
                        break
                nb_fit_idx=old_index
#                 except:
#                     iv.flag[i]=True
#                     nb_idx = int(len(iv.i_bias[b])*0.2)
            i_bias_nb = iv['i_bias'][nb_fit_idx:nb_end_idx]
            #smooth_i_bias_nb = find_smooth_indices()
            # pysmurf fixed polarity for us, so don't need to worry about that. 
            if not force_c == -42.0: # caller gave us a i_tes_offset to force, but not R_n
                i_tes_forced_i_tes_offset = lambda i_bias, slope: slope*i_bias + force_c
                (popt, pcov) = curve_fit(i_tes_forced_i_tes_offset,
                                        iv['i_bias'][nb_fit_idx:nb_end_idx],
                                        resp_bin[nb_fit_idx:nb_end_idx])
                norm_fit = [popt[0],force_c]
                r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n
            elif force_R_n == -42.0: # Caller hasn't given us an r_n to force. 
                norm_fit = np.polyfit(iv['i_bias'][nb_fit_idx:nb_end_idx],\
                                      resp_bin[nb_fit_idx:nb_end_idx],1)
                r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n
            else: # caller gave us an r_n to force. 
                r_n = force_R_n
                norm_fit = []
                forced_slope = 1/(r_n/r_sh+1)
                i_tes_forced_r_n = lambda i_bias, c : forced_slope*i_bias + c            
                norm_fit.append(forced_slope) # forced slope. 
                # We don't want to make a fit because we know the lower points
                # will be more off. 
                # Instead, run norm_fit through a high-V_b value to calc c, 
                # forced line's y-intercept.
                # TODO: Need to do some averaging or something here!!!
                #norm_fit.append(resp_bin[nb_idx_end] - forced_slope*iv['i_bias'][-1])
                # However, last value always has that downturn.
                # So use a point a little bit away.

                # I think we have to make a fit, because running through just 1 point
                # is crazy risky, and I don't see an obvious way to smooth it effectively.
                # I guess literally just median smoothing could work...
                (popt, pcov) = curve_fit(i_tes_forced_r_n,
                                        iv['i_bias'][nb_fit_idx:nb_end_idx],
                                        resp_bin[nb_fit_idx:nb_end_idx])
                norm_fit.append(popt[0]) # c!
            
        # TODO: make residual plotter! 
        iv['i_tes'] = resp_bin - norm_fit[1] # there we go. 
        if 'i_tes_offset' in iv_py.keys():
            iv['i_tes_offset'] = iv_py['i_tes_offset'] + norm_fit[1]
        else:
            iv['i_tes_offset'] = norm_fit[1]
        
        iv['R'] = r_sh * (iv['i_bias']/iv['i_tes'] - 1)
        iv['R_n'] = r_n 
        '''# I've never used these and they caused Uv42 to crash:
        #  get pysmurf's target R_op_target:
        #py_target_idx = np.ravel(np.where(iv_py['v_tes']==iv_py['v_tes_target']))[0]
        # above doesn't work b/c sc offset removal can remove the point that it targeted, so:
        v_targ = iv_py['v_tes_target']
        py_target_idx = np.ravel(np.where( abs(iv_py['v_tes']-v_targ) == min(abs(iv_py['v_tes']-v_targ)) ))[0]
        R_op_target = iv_py['R'][py_target_idx]    
        # Now just refill the rest of the keys same way pysmurf does (except do correct v_bias_target):
        # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
        # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]]
        #iv['R'] = r_sh * (iv['i_bias']/iv['i_tes'] - 1) # moved up
        #iv['R_n'] = r_n # moved out of target section
        # the correct equivalent of pysmurf's i_R_op
        i_R_op = 0
        for i in range(len(iv['R'])-1,-1,-1):
            if iv['R'][i] < R_op_target:
                i_R_op = i
                break
        iv['v_bias_target'] = iv['v_bias'][i_R_op]
        iv['v_tes_target'] = iv['v_tes'][i_R_op]'''
        iv['v_tes'] = iv['i_bias']/(1/r_sh + 1/iv['R']) # maybe use R_sh*(i_bias-i_tes)?
        iv['p_tes'] = iv['v_tes']**2/iv['R']
        # pysmurf's p_trans: 
        iv['p_trans'] = np.median(iv['p_tes'][iv['trans idxs'][0]:iv['trans idxs'][1]]) 
        # SO p_trans (if possible):
        if len(np.ravel(np.where(iv['R']<s.use_per/100.0*iv['R_n']))) > 0:
            iv[f'p_b{s.use_per}'] = iv['p_tes'][np.ravel(np.where(iv['R']<s.use_per/100.0*iv['R_n']))[-1]]
        else:
            iv[f'p_b{s.use_per}'] = -42.0 # you never reach it. 
        # TODO: STILL NEED TO ADD si and si_target! 
        
        iv['from_py'] = []
        for key in iv_py:
            if key not in iv.keys(): #I wanna keep some of these.
                iv[key] = iv_py[key]
                iv['from_py'].append(key)
        return iv   
    
    def update_temp_list(s,temp, temp_list_name):
        temp_list = getattr(s,temp_list_name)
        if temp not in temp_list:
            if len(temp_list) == 0:
                setattr(s,temp_list_name, [temp])
            else:
                i=0
                stop = False
                while i < len(temp_list) and not stop:
                    if temp < temp_list[i]:
                        setattr(s,temp_list_name,temp_list[:i] + [temp] + temp_list[i:])
                        stop = True
                    i+=1
                if not stop: # greatest in the list
                    setattr(s,temp_list_name,temp_list + [temp])
                    #temp_list.append(temp) 

    def merge_ramps(s,bs1,bs2,sort_key="temp", arr_keys=['temp','p_b90','R_nSM']):
        # sort_key is the one to order the arrays with reference to
        # (so, a temperature array).
        # arr_keys is the list of keys:arrays to be ordered
        # For ramp: sort_key = 'temp', arr_keys=['temp',f'p_b{s.use_per}','R_nSM']
        # for ramp_raw: sort_key = 'temp_raw', 
        #    arr_keys=['temp_raw', 'p_satSM', 'R_nSM_raw', 'all_fit_R_n_raw']
        if 'p_b90' in arr_keys:
            arr_keys[arr_keys.index('p_b90')] = f'p_b{s.use_per}'
        mbs = {} # merged bathsweep
        # The below will be useful if I ever add R_n to ramp...
        #arr_keys = ['temp',f'p_b{s.use_per}','R_nSM'] # keys with arrays that go with temp 
                                    # and need to be merged
        sbs = s.dUnique(bs1,bs2) 
        for sb in sbs:
            mbs[sb]={}
            sb_chs = s.dUnique(bs1[sb],bs2[sb])
            for ch in sb_chs:
                mbs[sb][ch]={}
                for key in arr_keys:
                    mbs[sb][ch][key] = []
                if ch in bs1[sb].keys() and ch in bs2[sb].keys(): # Both have this channel. Interleave. 
                    i1, i2 = 0,0
                    temp1, temp2 = bs1[sb][ch][sort_key], bs2[sb][ch][sort_key]
                    d1,d2 = bs1[sb][ch],bs2[sb][ch]
                    #p_sat1, p_sat2 = bs1[sb][ch][f'p_b{s.use_per}'], bs2[sb][ch][f'p_b{s.use_per}']
                    while i1 < len(temp1) and i2 < len(temp2):
                        if temp1[i1] <= temp2[i2]:
                            for key in arr_keys:
                                mbs[sb][ch][key].append(d1[key][i1])
                            i1 += 1
                        else:
                            for key in arr_keys:
                                mbs[sb][ch][key].append(d2[key][i2])
                            i2 += 1
                    if i1 < len(temp1):
                        for key in arr_keys:
                            mbs[sb][ch][key] += d1[key][i1:]
                    if i2 < len(temp2):
                        for key in arr_keys:
                            mbs[sb][ch][key] += d2[key][i2:]

                elif ch in bs1[sb].keys(): # only bs1 has this channel
                    for key in arr_keys: # can't use d1/d2 here, they aren't defined!
                        mbs[sb][ch][key] = bs1[sb][ch][key]
                else:                      # only bs2 has this channel
                    for key in arr_keys:
                        mbs[sb][ch][key] = bs2[sb][ch][key]
        return mbs
    
    def dUnique(s,d1,d2):
        # merge two dictionary's key lists into numpy array.
        return np.unique([key for key in d1.keys()]+[key for key in d2.keys()])
    
    
    def categorize_ivs(s):
        # run if you want to do iv analysis.
        s.iv_cat = {'is_iv':[],'not_iv':[],
                    'wrong_bl':[],'bad_channel':[],'bad_temp':[]}
        for iv_analyzed_info in s.iv_analyzed_info_arr:
            for iv_line in iv_analyzed_info:
                s.categorize_ivs_helper(iv_line)
                
    def categorize_ivs_helper(s,iv_line):
        iva_dict = iv_line['iv_analyzed']
        for sb in iva_dict.keys():
            if sb == 'high_current_mode':
                continue
            for ch,d in iva_dict[sb].items():
                full_name = (sb,ch,iv_line['temp'],iv_line['bl'])
                if not s.test_device.mux_map[sb][ch]['biasline'] == iv_line['bl']:
                    s.iv_cat['wrong_bl'].append(full_name)
                    d['is_iv_info'] = "wrong_bl"
                elif sb not in s.ramp.keys() or ch not in s.ramp[sb].keys():
                    s.iv_cat['bad_channel'].append(full_name)
                    d['is_iv_info'] = "bad_channel"
                elif iv_line['temp'] not in s.ramp[sb][ch]['temp']:
                    s.iv_cat['bad_temp'].append(full_name)
                    d['is_iv_info'] = "bad_temp"    
                else:
                    s.iv_cat['is_iv'].append(full_name)
                    d['is_iv'] = True
                    d['is_iv_info'] = "is_iv"
                    continue
                d['is_iv'] = False
                s.iv_cat['not_iv'].append(full_name)
    
    def set_bl_bad(s,rm_bl):
        # re make iv_cat
        new_iv_cat = {}
        new_iv_cat[f"rm_bl_is_iv"] = []
        new_iv_cat[f"rm_bl_not_iv"] = []
        for key,lst in s.iv_cat.items():
            new_iv_cat[key] = []
            new_iv_cat[f"rm_bl_{key}"] = []
            for name in lst:
                sb,ch,temp,bl = name
                if not (bl == rm_bl):
                    new_iv_cat[key].append(name)
                else:
                    new_iv_cat[f"rm_bl_{key}"].append(name)
        # bl_ivas
        bl_ivas = s.find_iva_matches([('bl','=',rm_bl)])
        for iva in bl_ivas:
            for key in ['is_iv','is_iv_info','contextless_is_iv','contextless_note']:
                if key in iva.keys():
                    iva['orig_' + key] = iva[key]
            iva['is_iv'] = False
            iva['is_iv_info'] = f"rm_bl"
            iva['contextless_is_iv'] = False
            iva['contextless_note'] = f"rm_bl"
        s.iv_cat = new_iv_cat
    
    def redo_cuts(s):
        # maybe run this when norm_correct makes BIG difference
        # redoes from s.iv_analyzed_info_arr contents.
        s.redo = True
        s.ramp_raw_arr = []
        s.ramp_arr = []
        s.ramp = []
        s.load_temp_sweep_IVs()
        s.ramp = s.ramp_arr[0] # merging
        for i in range(1,len(s.ramp_arr)):
            s.ramp = s.merge_ramps(s.ramp, s.ramp_arr[i])    
        s.redo = False
        if s.ramp_type == "bath":
            s.do_fits() # update the fit results and summary plots
                
    # ==================== DATA FINDERS ===============
    def find_iva(s,sb,ch,temp,bl=-42,run_indexes=[],nnc=False):
        if s.norm_correct and nnc:
            ivaia = s.iv_analyzed_info_arr_nnc
        else:
            ivaia = s.iv_analyzed_info_arr
        if not run_indexes:
            run_indexes = range(len(ivaia))
        temp_ch_iv_analyzed_arr = []
        for i in run_indexes:
            analyzed_iv_info = ivaia[i]
            for dicty in analyzed_iv_info:
                if abs(dicty['temp']-temp) < 0.5 \
                and sb in dicty['iv_analyzed'].keys() \
                and ch in dicty['iv_analyzed'][sb].keys():
                    # Make sure the real one, or chosen one, goes first
                    if (dicty['bl'] == s.test_device.mux_map[sb][ch]['biasline'] \
                    and len(temp_ch_iv_analyzed_arr)>0 and bl==-42)\
                    or dicty['bl']==bl:
                        temp_ch_iv_analyzed_arr = [dicty['iv_analyzed'][sb][ch]] \
                                                  + temp_ch_iv_analyzed_arr
                    elif bl==-42:
                        temp_ch_iv_analyzed_arr.append(dicty['iv_analyzed'][sb][ch])
                    dicty['iv_analyzed'][sb][ch]['bl'] = dicty['bl'] 
                    # just adding a key for plotRP easy.
        return temp_ch_iv_analyzed_arr
    
    def find_temp_ch_iv_analyzed(s,sb,ch,temp,bl=-42,run_indexes=[],nnc=False):
        # backwards compatibility name for find_iva
        return s.find_iva(sb,ch,temp,bl=bl,run_indexes=run_indexes,nnc=nnc)
    

    def find_iva_matches(s,match_list,dict_detail=False,run_indexes=[],nnc=False):
        '''Finds all ivas that match criteria in match_list; and returns 
        them as a tiered dictionary based on match_list ordering. Specifically:
        
        EDIT: Match list can now be a string separated by &,no spaces, see 
        general str_to_match_list docstring
        match_list should be a list of (match_key,match_type,match_val(s)), where
            str | match_key  : name of quality to match
            str | match_type : '<', '<=', '>=', '>', or '=' ('!=' does not work)
            float or list | match_val(s) : value to compare to for non-"=" 
                                  OR for "=", list of acceptable values OR 
                                  'all','any',['all'] or ['any'] to accept (and label) everything
        Then, find_iva_matches returns a dict of the structure (for ex. two match triples)
        {(match_key1,match_type1,<specific match_val1.1>) : 
              {(match_key2,match_type2,<specific match_val2.1>) : [ivas matching 1.1 and 2.1]
               (match_key2,match_type2,<specific match_val2.2>) : [ivas matching 1.1 and 2.2]
              }
         (match_key1,match_type1,<specific match_val1.2>) : 
              {(match_key2,match_type2,<specific match_val2.1>) : [ivas matching 1.2 and 2.1]
               (match_key2,match_type2,<specific match_val2.2>) : [ivas matching 1.2 and 2.2]
              }
        }
        EXCEPT if there is only one match_val for a given triple (it's not a list or 'all'),
        it doesn't make that a subdict; if NONE are lists or 'all', it will just return a list!
        '''
        # TODO: Check that my inputs are valid.
        # can't pre-construct the to_return because of 'all']...
        
        to_return=[]
        num_levels=0
        if type(match_list) == str:
            match_list = str_to_match_list(match_list)
        for match_key,match_type,match_val in match_list:
            if match_val in ['all','any'] or type(match_val) == list:
                to_return = {}
                num_levels +=1
        if num_levels > 0:
            to_return = {}
        matches=0
        ivaia = s.iv_analyzed_info_arr
        if s.norm_correct and nnc:
            ivaia = s.iv_analyzed_info_arr_nnc
        if not run_indexes:
            run_indexes = range(len(ivaia))
        # begin iterating
        for i in run_indexes:
            analyzed_iv_info = ivaia[i]
            for ivaia_dict in analyzed_iv_info:
                ivad = ivaia_dict['iv_analyzed']
                for sb in ivad.keys():
                    if sb=="high_current_mode":
                        continue
                    for ch in ivad[sb].keys():
                        matches += s.write_match(match_list,num_levels,to_return,ivaia_dict,sb,ch,dict_detail)
                
        #print(f"Found {matches} matches to {match_list}")
        return to_return
    
    def write_match(s,match_list,num_levels,to_return,ivaia_dict,sb,ch,dict_detail):
        dict_loc = to_return
        lvl = 0
        for i in range(len(match_list)):
            match_key, match_type, match_val = match_list[i]
            match = s.val_matches(match_type,match_val,\
                                  s.iva_val(match_key,ivaia_dict,sb,ch))
            if match == False:
                return 0
            if match_val in ['all','any'] or type(match_val) == list:
                lvl+=1
                if type(match) == list: # This breaks when match=True, why I cannot figure out: "not type(match) == bool":
                    match_val = match[0] # unpack. # I think this should ALWAYS happen?
                if dict_detail: 
                    key = (match_key,match_type,match_val)
                else:
                    key = match_val
                if not key in dict_loc.keys():
                    if lvl == num_levels:
                        dict_loc[key] = []
                    else:
                        dict_loc[key] = {}
                dict_loc = dict_loc[key]
        dict_loc.append(ivaia_dict['iv_analyzed'][sb][ch])
        return 1   
    
    def iva_val(s,match_key,ivaia_dict,sb,ch):
        if match_key == 'sb':
            return sb
        if match_key == 'ch':
            return ch
        if match_key in ivaia_dict.keys():
            return ivaia_dict[match_key]
        # possibly make this more extensible.
        sbchDictys = [ivaia_dict['iv_analyzed'], s.tes_dict, s.test_device.mux_map]
        for dicty in sbchDictys:
            if sb in dicty.keys() and ch in dicty[sb].keys() \
            and match_key in dicty[sb][ch].keys():
                return dicty[sb][ch][match_key]
        return "VAL NOT FOUND"
    
    def val_matches(s,match_type,match_val,val):
        if val == "VAL NOT FOUND":
            return False
        if match_type == '=':
            if type(match_val) == list:
                if len(match_val)== 0 or match_val[0] in ['all','any']:
                    return [val] # why the wrapping? because val could be "False"
                for mval in match_val:
                    if val == mval:
                        return [val]
                return False
            if match_val in ['all','any']:
                return [val]
            return val == match_val
        if match_type == '!=':
            return not (val == match_val)
        # possibly check val's floatiness
        val = float(val)
        match_val = float(match_val)
        if match_type == '<':
            return val < match_val
        if match_type == '<=':
            return val <= match_val
        if match_type == '>=':
            return val >= match_val
        if match_type == '>':
            return val > match_val
    
    # ==================== PLOTTING FUNCTIONS ===============
    def plot_det_IVs(s,sb,ch,bl=-42,max_temp=3000, x_lim=[],y_lim=[], tes_dict=False, 
                 run_indexes=[], own_fig=True,linewidth=1,do_idx_lines=True,
                    plot_args={}):
        plt.figure(figsize=default_figsize)
        xmin,ymin = 100000000,100000000
        xmax,ymax = -1,-1
        for temp in s.temp_list:
            if temp<=max_temp:
                (newmx,newmy,newx,newy) = s.plot_IV(sb, ch, temp, bl=bl,own_fig=False,
                                        run_indexes=run_indexes,tes_dict=tes_dict,
                                       linewidth=linewidth,label_include=['temp'],
                                       x_lim=x_lim,y_lim=y_lim,do_idx_lines=do_idx_lines,
                                                   plot_args=plot_args)
                xmin=min(xmin,newmx)
                ymin=min(ymin,newmy)
                xmax=max(xmax,newx)
                ymax=max(ymax,newy)
         
        plt.title(f"{s.dName} sb{sb} ch{ch} bl{s.mux_map[sb][ch]['biasline']}"+\
                  f"[{s.mux_map[sb][ch]['TES_freq']}GHz,"+\
                  f"{s.mux_map[sb][ch]['masked']} {s.mux_map[sb][ch]['OMT']}]")
        if x_lim:
            plt.xlim(x_lim)
        else:
            plt.xlim([xmin,xmax])
        if y_lim:
            plt.ylim(y_lim)
        else:
            plt.ylim([ymin,ymax])
        
    def plot_IV(s,sb,ch,temp ,bl=-42,x_lim=[],y_lim=[], tes_dict=True, 
                 run_indexes=[], own_fig=True,linewidth=1,do_idx_lines=True,
                label_include=['default'],plot_args={}):
        temp_ch_iv_analyzed_arr = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,
                                                             run_indexes=run_indexes)
        # This is only going to do the first one if no run_index specified, so that 
        # I can do own_fig stuff. 
        # issue of multiple bl options. 
        if not temp_ch_iv_analyzed_arr:
            return []
        d = temp_ch_iv_analyzed_arr[0]
        if 'i_bias' in d.keys() and 'i_tes' in d.keys():
            i_bias, i_tes = d['i_bias'], d['i_tes']
        else:
            i_bias = d['v_tes']*(1/s.r_sh + 1/d['R'])
            d['i_bias'] = i_bias
            # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
            i_tes = i_bias/(d['R']/s.r_sh + 1)
            d['i_tes'] = i_tes
        (plotted,d) = s.internal_plot_iv_analyzed_keys(d,sb,ch,temp,'i_bias','i_tes', 
                              x_lim=x_lim,y_lim=y_lim, tes_dict=tes_dict, 
                              bl=bl,run_indexes=run_indexes,own_fig=own_fig,
                             linewidth=linewidth,label_include=label_include,
                                                      plot_args=plot_args)
        color = plotted[0].get_color()
        if do_idx_lines:
            plt.vlines([i_bias[d['trans idxs'][0]]],
                    min(i_tes),max(i_tes), colors=color, linestyles="dotted")
            plt.vlines([i_bias[d['trans idxs'][1]]],
                    min(i_tes),max(i_tes), colors=color, linestyles="dashdot")
        if own_fig:
            return plotted
        else:
            return (min(d['i_bias']),min(d['i_tes']),max(d['i_bias']),max(d['i_tes']))
    
    def plot_det_RPs(s,sb,ch,bl=-42,nnc=False,per_R_n=False,max_temp=3000,
                     lines=['default'],label_include=['temp'],plot_args={}, 
                     x_lim=[],y_lim=[]):
        plt.figure(figsize=default_figsize)
        xmax,ymax = -1,-1
        xmin,ymin = 10000000000,1000000000
        for temp in s.temp_list:
            if temp<=max_temp:
                (newmx,newmy,newx,newy) = s.plot_RP(sb, ch, temp, bl=bl,nnc=nnc,own_fig=False,
                                        per_R_n=per_R_n, lines=lines,
                                        label_include=label_include,
                                        plot_args=plot_args)
                xmin=min(xmin,newmx)
                ymin=min(ymin,newmy)
                xmax=max(xmax,newx)
                ymax=max(ymax,newy)
        nnc_l=''
        if nnc:
            nnc_l='NoNormCorrect '
        plt.title(f"{s.dName} {nnc_l}sb{sb} ch{ch} bl{s.mux_map[sb][ch]['biasline']}"+\
                  f"[{s.mux_map[sb][ch]['TES_freq']}GHz,"+\
                  f"{s.mux_map[sb][ch]['masked']}, {s.mux_map[sb][ch]['OMT']}]")
        if x_lim:
            plt.xlim(x_lim)
        else:
            plt.xlim([xmin,xmax])
        if y_lim:
            plt.ylim(y_lim)
        else:
            plt.ylim([ymin,ymax])

        
    def plot_RP(s,sb, ch, temp, bl=-42,run_indexes=[],nnc=False, 
                per_R_n=False, own_fig=True, tes_dict=True,
                lines=['default'],label_include=[],plot_args={}):
        # ---- Set up options.
        if lines == ['default']:
            if own_fig:
                lines = ['R_nSM','p_trans',f'p_b{s.use_per}']
                if s.norm_correct:
                    lines.append('R_n')
            else:
                lines = [] #f'p_b{s.use_per}'
        if tes_dict:
            tes_dict=s.tes_dict
        nnc_n = ''
        if s.norm_correct and nnc:
            nnc_n = ' NNC!'
        # v ones that don't vary between these ivas
        label_options = {'sb':f"sb{sb}", 'ch':f"ch{ch}", 
                         'temp':f"temp{int(temp)}{s.key_info['temp']['units']}",
                         'nnc':nnc_n, 'freq':f"{s.mux_map[sb][ch]['TES_freq']}GHz",
                         'opt_name':f"{s.test_device.opt_dict[s.mux_map[sb][ch]['opt']][0]}"} 
        # ---- find the requested data. 
        plot_array = []
        ivas = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,run_indexes=run_indexes,nnc=nnc)
        # ---- make the plot(s)/add to existing plot
        xmax = -1
        ymax = -1
        xmin,ymin = 100000000,100000000
        for d in ivas:  #d = iva # dicty['iv_analyzed'][sb][ch]
            # ---- Define/label the axis
            plt.xlabel(f"P_tes [{s.key_info['p_tes']['units']}]")
            if per_R_n:
                r2plot= d['R']/d['R_n']
                plt.ylabel('R_tes/R_n []')
                ymax = max(1.1,ymax) # needed for lines even if we don't set the plot limits
            else:
                r2plot = d['R']
                plt.ylabel(f"R_tes [{s.key_info['R']['units']}]")
                ymax = max(d['R_n']*1.1,ymax) # needed for lines even if we don't set the plot limits
            ymin= min(min(r2plot),ymin)
            xmax = max(max(d['p_tes']),xmax)
            # ---- do own_fig stuff.
            if own_fig:
                plot_array.append(plt.figure(figsize=default_figsize))
                plt.title(f"{s.dName} sb{sb} ch{ch} "+\
                          f"temp{int(temp)} {s.key_info['temp']['units']} bl{d['bl']}"+\
                         nnc_n)
                plt.xlim(0, max(d['p_tes']))
                plt.ylim(0, ymax)
            # ---- legend entry of the RP line
            label_options['bl'] = f"bl{d['bl']}"
            plot_args['label'] = s.construct_crv_label(label_include,label_options)
            # ---- Plot the RP curve
            plt.plot(d['p_tes'],r2plot,**plot_args)
            # ---- Add annotation lines 
            if 'R_nSM' in lines:
                plt.hlines(d['R_nSM'], 0, max(d['p_tes']), colors=['c'], \
                           linestyles='dashed', \
                           label=f"R_nSM this temp={d['R_nSM']:.4} {s.key_info['R_nSM']['units']}")
            if 'R_n' in lines and tes_dict and 'R_n' in tes_dict[sb][ch].keys():
                plt.hlines(tes_dict[sb][ch]['R_n'], 0, max(d['p_tes']), colors=['b'],\
                           linestyles='dashed',\
                           label=f"R_n TES best ={tes_dict[sb][ch]['R_n']:.4} {s.key_info['R_nSM']['units']}")
            if 'p_trans' in lines:
                plt.vlines(d['p_trans'],0, ymax, colors=['r'],linestyles="dashed",\
                           label=f"Pysmurf (median) P_sat={int(d['p_trans']*10)/10.0} pW")
            if f'p_b{s.use_per}' in lines:
                p_sat = d['p_tes'][np.where(d['R'] < s.use_per/100.0 * d['R_n'])[0][-1]] # should probably use calc_p_b90()...
                plt.vlines(p_sat,0, ymax, colors=['g'],linestyles="dashed", \
                           label=f"{s.use_per}% R_n P_b={int(p_sat*10)/10.0} pW")
            if label_options or lines:
                plt.legend()
        if own_fig:
            return plot_array
        else:
            return (xmin,ymin,xmax,ymax)
        
    def construct_crv_label(s,label_include,label_options):
        if label_include:                 
            label = ''
            for option in label_include:
                if option not in label_options.keys():
                    print(f"'{option}' is not in label options dict")
                else:
                    label += label_options[option]
        return label
    
    def plot_iv_analyzed_keys(s,x_key,y_key, sb, ch, temp, 
                              x_lim=[],y_lim=[], tes_dict=True,bl=-42, run_indexes=[], 
                              own_fig=True,linewidth=1):
        # separate function so that i can check if i_bias and i_tes exist and ad them
        # if they don't. 
        temp_ch_iv_analyzed_arr = s.find_temp_ch_iv_analyzed(sb,ch,temp,bl=bl,
                                                             run_indexes=run_indexes)
        # This is only going to do the first one if no run_index specified, so that 
        # I can do own_fig stuff. 
        if not temp_ch_iv_analyzed_arr:
            return []
        d = temp_ch_iv_analyzed_arr[0]
        return s.internal_plot_iv_analyzed_keys(d, sb,ch,temp,x_key,y_key,
                              x_lim=x_lim,y_lim=y_lim, tes_dict=tes_dict, 
                              bl=bl, run_indexes=run_indexes,own_fig=own_fig,
                                linewidth=linewidth)
        
    def internal_plot_iv_analyzed_keys(s,d,sb,ch,temp, x_key,y_key,
                              x_lim=[],y_lim=[], tes_dict=True, bl=-42,run_indexes=[], 
                              own_fig=True,linewidth=1,label_include=['default'],
                                      plot_args={}):    
        linestyle = 'solid'
        if tes_dict and sb in s.tes_dict.keys() and ch in s.tes_dict[sb].keys():
            linestyle = s.tes_dict[sb][ch]['linestyle']
        label_options = {'bl':f"BL{d['bl']}",'sb':f"sb{sb}", 'ch':f"ch{ch}", 
                         'temp':f"temp{int(temp)}{s.key_info['temp']['units']}",
                         'freq':f"{s.mux_map[sb][ch]['TES_freq']}GHz",
                         'opt_name':f"{s.test_device.opt_dict[s.mux_map[sb][ch]['opt']][0]}"} 
        if own_fig:
            if label_include== ['default']:
                label_include = ['bl','sb','ch','temp','freq','opt_name']
            curve_name = f"{s.dName} {s.construct_crv_label(label_include,label_options)}"
            plt.figure(figsize=default_figsize)
            plotted = plt.plot(d[x_key],d[y_key], linestyle=linestyle,
                               linewidth=linewidth,label=curve_name,**plot_args)
            plt.title(curve_name)
        else:
            if label_include == ['default']:
                label_include = []
            if not label_include:
                plotted =plt.plot(d[x_key],d[y_key], 
                              linestyle=linestyle,linewidth=linewidth,**plot_args)
            else:
                plotted =plt.plot(d[x_key],d[y_key],
                              label=s.construct_crv_label(label_include,label_options),
                              linestyle=linestyle,linewidth=linewidth,**plot_args)
            plt.title(f"{s.key_info[y_key]['name']} vs. {s.key_info[x_key]['name']}\n{s.dName} sb{sb}ch{ch}temp{temp}")
        plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
        plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
        if x_lim:
            plt.xlim(x_lim[0],x_lim[1])
        else:
            if 'lim' in s.key_info[x_key].keys():
                plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
        if y_lim:
            plt.ylim(y_lim[0],y_lim[1])
        else:
            if 'lim' in s.key_info[y_key].keys():
                plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
        if not own_fig:
            plt.legend()#bbox_to_anchor=(1.01,1),loc="upper left"
        return (plotted,d)
    
       
    
                
    def plot_ramp_raw(s,ramp_raw,tes_dict=True):
        return s.plot_ramp_keys_by_sb_2legend(ramp_raw,'temp_raw','p_satSM', \
                    prefix='Raw ',tes_dict=tes_dict)

    def plot_ramp_raw90R_n(s,ramp_raw,tes_dict=True):
        return s.plot_ramp_keys_by_sb_2legend(ramp_raw,f'temp{s.use_per}R_n',f'p_b{s.use_per}R_n', \
                    prefix='Raw ',tes_dict=tes_dict)

    def plot_ramp(s,ramp,tes_dict=True,zero_starts=False):
        if not zero_starts:
            y_lim = [0,15]
        else:
            y_lim = []
        return s.plot_ramp_keys_by_sb_2legend(ramp,'temp',f'p_b{s.use_per}',y_lim=y_lim, \
                    prefix='Cut ',tes_dict=tes_dict,zero_starts=zero_starts)

    def plot_ramp_keys_by_sb(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                             prefix='',tes_dict=True,zero_starts=False):
        plot_array = []
        linestyle,label =  'solid', ''
        tab20 = plt.get_cmap('tab20')
        if tes_dict:
            tes_dict=s.tes_dict
        for sb in bathsweep.keys():
            #hsv = plt.get_cmap('hsv')
            colors = tab20(np.linspace(0, 1.0, len(bathsweep[sb].keys())+1))
            col_ind=0
            plot_array.append((plt.figure(figsize=default_figsize), sb)) #(default_figsize[0]*1.5,default_figsize[1])
            for ch, d in bathsweep[sb].items():
                if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                    linestyle = tes_dict[sb][ch]['linestyle']
                    label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                ys = d[y_key]
                if zero_starts:
                    ys = d[y_key]-d[y_key][0]
                else:
                    ys = d[y_key]
                plt.plot(d[x_key], ys, label=f'ch{ch}' + label,
                         linestyle=linestyle, color=colors[col_ind])
                col_ind += 1
            if x_lim:
                plt.xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                plt.ylim(y_lim[0],y_lim[1])
            elif not zero_starts:
                if 'lim' in s.key_info[y_key].keys():
                    plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            plt.legend() #bbox_to_anchor=(1.01,1),loc="upper left"
            plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
            plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
            plt.title(f"{prefix}{y_key} vs. {x_key} smurf band {sb} {s.dName}")
            #plt.tight_layout(rect=[0,0,0.75,1])
            #plt.subplots_adjust(right=0.75)
        return plot_array
    
    def plot_ramp_keys_by_sb_2legend(s,bathsweep,x_key,y_key,x_lim=[],y_lim=[], 
                             prefix='',tes_dict=True,zero_starts=False):
        plot_array = []
        linestyle,label =  'solid', ''
        tab20 = plt.get_cmap('tab20')
        opt_names = []
        linestyles = []
        if tes_dict:
            tes_dict=s.tes_dict
        for sb in bathsweep.keys():
            #hsv = plt.get_cmap('hsv')
            colors = tab20(np.linspace(0, 1.0, len(bathsweep[sb].keys())+1))
            col_ind=0
            plot_array.append((plt.figure(figsize=(default_figsize[0]*1.2,default_figsize[1])), sb))
            for ch, d in bathsweep[sb].items():
                if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                    linestyle = tes_dict[sb][ch]['linestyle']
                    if tes_dict[sb][ch]["opt_name"] not in opt_names:
                        opt_names.append(tes_dict[sb][ch]["opt_name"])
                        linestyles.append(linestyle)
                    #label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                    label = f' {str(tes_dict[sb][ch]["TES_freq"])}'
                ys = d[y_key]
                if zero_starts:
                    ys = d[y_key]-d[y_key][0]
                else:
                    ys = d[y_key]
                plt.plot(d[x_key], ys, label=f'ch{ch}' + label,
                         linestyle=linestyle, color=colors[col_ind])
                col_ind += 1
            if x_lim:
                plt.xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    plt.xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                plt.ylim(y_lim[0],y_lim[1])
            elif not zero_starts:
                if 'lim' in s.key_info[y_key].keys():
                    plt.ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            ch_legend = plt.legend(bbox_to_anchor=(1.01,1),loc="upper left") # bbox_to_anchor=(1.2,0.1) the positioning doesn't work...
            # Now do the linestyle legend
            if len(linestyles) > 0:
                line_handles = []
                for i in range(len(opt_names)):
                    handle, = plt.plot([],[],linestyle=linestyles[i],color='0')
                    line_handles.append(handle)
                plt.legend(line_handles,opt_names,loc='best')
                plt.gca().add_artist(ch_legend) # put channel legend back in.
            plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
            plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
            plt.title(f"{prefix}{y_key} vs. {x_key} smurf band {sb} {s.dName}")
            #plt.tight_layout(rect=[0,0,0.75,1])
            plt.subplots_adjust(right=0.77)
        return plot_array

    # biasline
    def plot_ramp_by_BL(s,bathsweep,tes_dict=True,y_lim=[0,8]):
        return s.plot_ramp_keys_by_BL(bathsweep,'temp',f'p_b{s.use_per}',
                                     y_lim=y_lim,prefix="Cut ",tes_dict=tes_dict)
    
    def plot_ramp_keys_by_BL(s,bathsweep, x_key,y_key,
                             x_lim=[],y_lim=[], prefix='',tes_dict=True):
        bonded_TESs = s.test_device.bonded_TESs
        mux_map = s.mux_map
        if tes_dict:
            tes_dict=s.tes_dict
        linestyle,label =  'solid', ''
        num_bl = s.test_device.num_bl
        nrows, ncols = math.ceil(num_bl*1.0/4), min(4,num_bl)
        figsize = (ncols*9.0/4,10.0/3*nrows)
#         if s.test_device.device_type == "LF_UFM":
#             nrows, ncols, figsize = 2,2,(3,5)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        total_num_with_IV = 0
        total_num_with_4 = 0
        for bl in range(ncols*nrows):
            r = math.floor(bl/ncols)
            c = bl - r*ncols
            if nrows==1:
                p=ax[bl]
            else:
                p= ax[r][c]
#         for r in range(len(ax)):
#             row = ax[r]
#             for c in range(len(row)):
#                 p = row[c]
            len_list = []
            for sb in bathsweep.keys():
                for ch, d in bathsweep[sb].items():
                    #print(mux_map[sb][ch])
                    if mux_map[sb][ch]['biasline'] == ncols*r + c:
                        if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                            linestyle = tes_dict[sb][ch]['linestyle']
                            label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                        len_list.append(len(d[y_key]))
                        p.plot(d[x_key], d[y_key], 
                               label=f'ch{ch}'+label,alpha=0.5,
                               linestyle=linestyle,linewidth=0.75,
                               marker=".",markersize=2)#,markersize=1)#,markersize=0.75
            #p.set_xlabel('bath temp [mK]')
            #p.set_ylabel('90% R_n P_sat [pW]')
            with_IV = len(len_list)
            total_num_with_IV += with_IV
            num_with_4 = len(np.where(np.array(len_list) >=4)[0])
            total_num_with_4 += num_with_4
            if bonded_TESs[ncols*r + c] == 0:
                t_yield = "N/A"
            else:
                t_yield  = int(100*float(num_with_4)/bonded_TESs[ncols*r + c])
            p.set_title(f"BL {ncols*r + c}: {with_IV}, {num_with_4} ({t_yield}%)")
            p.grid(linestyle='dashed')
            if x_lim:
                p.set_xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    p.set_xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                p.set_ylim(y_lim[0],y_lim[1])
            else:
                if 'lim' in s.key_info[y_key].keys():
                    p.set_ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            #p.legend(fontsize='small') # These plots don't really have space for legend.
        plt.suptitle(f"{s.dName} {prefix}{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}] vs. {s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}] \n with # chs with 1+, 4+ IVs ('4+' % of #bonded). Overall: {total_num_with_IV}, {total_num_with_4} ({100.0*total_num_with_4/sum(bonded_TESs):.0f}%)")
        plt.tight_layout()
        return (fig, ax)
    
    def plot_key_v_key_by_key(s,x_key,y_key,by_key,match_list=[],plot_args={},own_fig=True,do_label=False):
        if own_fig == True:
            graph = plt
            plt.figure(figsize=default_figsize)
            # Do these first to check the keys are in key_info
            plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
            plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
            restrictions=''
            if match_list:
                comp = [mtc[0]+mtc[1]+str(mtc[2]) for mtc in match_list]
                restrictions = "\n" + ",".join(comp)
            plt.title(f"{s.dName} {y_key} vs {x_key} by {by_key}{restrictions}")
            if x_lim:
                p.set_xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    p.set_xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                p.set_ylim(y_lim[0],y_lim[1])
            else:
                if 'lim' in s.key_info[y_key].keys():
                    p.set_ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
        elif own_fig != False: # passed an axis
            graph = own_fig
        else: #own_fig = False; just grab the plt, assume that's correct
            graph = plt

        num_by_key = 0
        iva_d = s.find_iva_matches([(by_key,'=','any')]+match_list)
        for by_val,iva_l in iva_d.items():
            my_x,my_y = [], []
            for iva in iva_l:
                if x_key in iva.keys() and y_key in iva.keys():
                    my_x.append(iva[x_key])
                    my_y.append(iva[y_key])
            if len(my_x) >0:
                graph.plot(my_x,my_y,**plot_args,label=f"{by_val}")
                num_by_key += 1
        
        return num_by_key

    def plot_key_v_key_by_key_by_BL(s,x_key,y_key,by_key,match_list=[],plot_args={},own_fig=True,
                                 x_lim=[],y_lim=[], prefix=''):
        #fig, ax = plt.subplots(nrows=4, figsize=(9,9))
        #num_chs_with_iv = [[]]*12 THIS WILL APPEND TO ALL OF THEM, ALL SAME REFERENCE!!
        num_bl = s.test_device.num_bl
        nrows, ncols = math.ceil(num_bl*1.0/4), min(4,num_bl)
        figsize = (ncols*9.0/4,10.0/3*nrows) # (7,7) at some point?
#         if s.test_device.device_type == "LF_UFM":
#             nrows, ncols, figsize = 2,2,(3,5)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        restrictions=''
        if match_list:
            comp = [mtc[0]+mtc[1]+str(mtc[2]) for mtc in match_list]
            restrictions = "\n" + ",".join(comp)
        plt.suptitle(f"{prefix}{s.dName}: {s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}] vs {s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}], by {by_key}{restrictions}")
        for bl in range(ncols*nrows):
            r = math.floor(bl/ncols)
            c = bl - r*ncols
            if nrows==1:
                p=ax[bl]
            else:
                p= ax[r][c]
#         for r in range(len(ax)):
#             row = ax[r]
#             for c in range(len(row)):
#             p = row[c]
            len_list = []
            num_by_key = plot_key_v_key_by_key(s,x_key,y_key,by_key,match_list=match_list+[('bl','=',ncols*r+c)],
                                  plot_args=plot_args,own_fig=p)
            p.set_title(f"BL {ncols*r + c}; {num_by_key} lines")
            p.grid(linestyle='dashed')
            if x_lim:
                p.set_xlim(x_lim[0],x_lim[1])
            else:
                if 'lim' in s.key_info[x_key].keys():
                    p.set_xlim(s.key_info[x_key]['lim'][0],s.key_info[x_key]['lim'][1])
            if y_lim:
                p.set_ylim(y_lim[0],y_lim[1])
            else:
                if 'lim' in s.key_info[y_key].keys():
                    p.set_ylim(s.key_info[y_key]['lim'][0],s.key_info[y_key]['lim'][1])
            #p.legend(fontsize='small') # These plots don't really have space for legend.
        plt.tight_layout()
        return (fig, ax) 

# ==============================================================================
# -------------------------- Temp_Ramp Analysis --------------------------------
# ==============================================================================

def examine_rampy(s,tes_dict=False):
    return examine_ramp(s,tes_dict)

def examine_ramp(s,tes_dict=False):
    # you know, maybe make this nnc?
    if tes_dict:
            tes_dict = s.test_device.tes_dict
    if s.norm_correct:
        s.plot_ramp_keys_by_BL(s.ramp_raw_arr_nnc[0], f'temp{s.use_per}R_n',
                               f'p_b{s.use_per}R_n',prefix='raw, nnc ',
                               tes_dict=tes_dict)
    else:
        s.plot_ramp_keys_by_BL(s.ramp_raw_arr[0], f'temp{s.use_per}R_n',
                               f'p_b{s.use_per}R_n',prefix='raw, nnc ',tes_dict=tes_dict)
    s.plot_ramp_keys_by_BL(s.ramp, 'temp',f'p_b{s.use_per}',prefix='Cut ',tes_dict=tes_dict)
    if s.ramp_type == 'bath':
        s.make_summary_plots()
        pbr,tcr,gr,nr = ([],[],[],[])
        if s.test_device.device_type == 'LF_UFM':
            pbr,tcr,gr,nr = ([0,10],[150,200],[0,175],[2,5])
        elif s.test_device.device_type == 'MF_UFM':
            pbr,tcr,gr,nr = ([2,15],[150,200],[50,300],[2,5])
        elif s.test_device.device_type == 'UHF_UFM':
            pbr,tcr,gr,nr = ([20,60],[150,200],[300,1200],[2,5])
        s.make_summary_plots(p_sat_Range=pbr, t_c_Range=tcr,
                                 g_Range=gr,n_Range=nr)


def report_ramp_IV_cat_breakdown(s):
    num_datasets = len(s.iv_cat['is_iv'])+len(s.iv_cat['not_iv'])
    p_good = int(100*len(s.iv_cat['is_iv'])/num_datasets)
    p_w_bl = int(100*len(s.iv_cat['wrong_bl'])/num_datasets)
    p_b_ch = int(100*len(s.iv_cat['bad_channel'])/num_datasets)
    p_b_tp = int(100*len(s.iv_cat['bad_temp'])/num_datasets)
    print(f"{s.dName}: {num_datasets} 'IVs'; {p_good}% good, {p_w_bl}% wrong bl, {p_b_ch}% bad channel, {p_b_tp}% bad_temp")
    # Hey, have it do cut_info reasons too---probably put that in iv_cat maker...
    to_return = {"'IVs'":num_datasets}
    for name,d in s.iv_cat.items():
        to_return[name] = len(d)
    return to_return        

# Debugging ----------------

def plot_norm_correct_p_b90_differences(s):
    p_b90s_nnc = []
    p_b90s = []
    #p_b90_difs = []
    p = plt.figure(figsize=default_figsize)
    lims = s.key_info[f'p_b{s.use_per}']['lim']
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims,lims,linestyle='--',color="black",label="y=x line")
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            p_b90s += d[f'p_b{s.use_per}']
            p_b90s_nnc += s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    
    plt.plot(p_b90s_nnc, p_b90s, linestyle="None",marker=".", markersize=0.5,label="data")
    #lims = [0, max(max(p_b90s_nnc),max(p_b90s))*1.1]
    
    plt.title(f"{s.dName} Norm-corrected p_b{s.use_per}s vs. nnc p_b{s.use_per}s")
    plt.ylabel(f"Norm-corrected {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.xlabel(f"no norm-correction {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.legend()
    return (p_b90s_nnc,p_b90s,p)


# ITO correction examination 
# ======= Looking at the effect normal correction has on p_sats!

# EXAMPLE USE OF THE ABOVE

def nnc_check(s,sb=-42,ch=-42,x_lim=[],y_lim=[]):
    # takes a bathramp
    plot_norm_correct_p_b90_differences(s)
    (p_b90s_nnc,p_b90s,ratios,p) = plot_norm_correct_p_b90_ratio_temp_details(s)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    (p_b90s_nnc,p_b90s,ratios,p) = plot_norm_correct_p_b90_ratio_freq_opt_details(s)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    if sb != -42 and ch != 42:
        add_det_line(s,sb,ch,color='black')
    plt.legend()
    

def plot_norm_correct_p_b90_differences(s):
    p_b90s_nnc = []
    p_b90s = []
    #p_b90_difs = []
    p = plt.figure(figsize=default_figsize)
    lims = s.key_info[f'p_b{s.use_per}']['lim']
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims,lims,linestyle='--',color="black",label="y=x line")
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            p_b90s += d[f'p_b{s.use_per}']
            p_b90s_nnc += s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    
    plt.plot(p_b90s_nnc, p_b90s, linestyle="None",marker=".", markersize=0.5,label="data")
    #lims = [0, max(max(p_b90s_nnc),max(p_b90s))*1.1]
    
    plt.title(f"{s.dName} Norm-corrected p_b{s.use_per}s vs. nnc p_b{s.use_per}s")
    plt.ylabel(f"Norm-corrected {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.xlabel(f"no norm-correction {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.legend()
    return (p_b90s_nnc,p_b90s,p)

def plot_norm_correct_p_b90_ratio(s):
    p_b90s_nnc = []
    p_b90s = []
    ratios = []
    #p_b90_difs = []
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            p_b90s += d[f'p_b{s.use_per}']
            p_b90s_nnc += s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
            if not len(d[f'p_b{s.use_per}']) == len(s.ramp_nnc[sb][ch][f'p_b{s.use_per}']):
                print(f"{sb} {ch}")
                return (d, s.ramp_nnc[sb][ch])
            ratios += [d[f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                       for i in range(len(d[f'p_b{s.use_per}']))]
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    p = plt.figure(figsize=default_figsize)
    plt.plot(p_b90s_nnc, ratios, linestyle="None",marker=".", label="data")
    #lims = [0, max(max(p_b90s_nnc),max(p_b90s))*1.1]
    lims = s.key_info[f'p_b{s.use_per}']['lim']
    plt.xlim(lims)
    #plt.plot(lims,lims,linestyle='--',color="black",label="y=x line")
    plt.title("Norm-corrected p_b90/nnc p_b90 vs. nnc p_b90")
    plt.ylabel(f"Norm-corrected p_b90/nnc p_b90")
    plt.xlabel(f"no norm-correction {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.legend()
    return (p_b90s_nnc,p_b90s,ratios,p)

def plot_norm_correct_p_b90_ratio_temp_details(s):
    # sort by ramp temperature. 
    p_b90s_nnc = {}
    p_b90s = {}
    ratios = {}
    dict_arr = [p_b90s_nnc,p_b90s,ratios]
    for dicty in dict_arr:
        for temp in s.temp_list:
            dicty[temp]=[]
    #p_b90_difs = []
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            for i in range(len(d['temp'])):
                p_b90s[d['temp'][i]].append(d[f'p_b{s.use_per}'][i])
                p_b90s_nnc[d['temp'][i]].append(s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i])
                ratios[d['temp'][i]].append(d[f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i])
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    p = plt.figure(figsize=default_figsize)
    for temp in p_b90s_nnc.keys():
        plt.plot(p_b90s_nnc[temp], ratios[temp],
                 linestyle="None", marker=".", markersize=1,
                 label=f"{temp}")
    do_norm_correct_plot_lims_and_titling(s)
    return (p_b90s_nnc,p_b90s,ratios,p)

def plot_norm_correct_p_b90_ratio_freq_opt_details(s,opacity=0.01):
    # sort by TES frequency and opt type. have a different color combination for each, since this scatterplot.
    # (and there's no way I'll be able to see linestyles on this crowded a plot.)
    mux_map = s.mux_map
    opt_dict = s.test_device.opt_dict 
    p_b90s_nnc = {}
    p_b90s = {}
    ratios = {}
    dict_arr = [p_b90s_nnc,p_b90s,ratios]
    #p_b90_difs = []
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            tes_freq = mux_map[sb][ch]['TES_freq']
            opt = mux_map[sb][ch]['opt']
            if (tes_freq,opt) not in dict_arr[0].keys():
                for dicty in dict_arr:
                    dicty[(tes_freq,opt)] = []
            p_b90s[(tes_freq,opt)] += d[f'p_b{s.use_per}']
            p_b90s_nnc[(tes_freq,opt)] += s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
            new_ratios = [d[f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                       for i in range(len(d[f'p_b{s.use_per}']))]
            ratios[(tes_freq,opt)] +=  new_ratios
#             if min(new_ratios) < 0.99:
#                 print(f"{sb} {ch} {opt_dict[mux_map[sb][ch]['opt']][0]} {min(new_ratios):.3}")
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    p = plt.figure(figsize=default_figsize)
    for (tes_freq,opt) in p_b90s_nnc.keys():
        if opt not in opt_dict.keys():
            continue
            opt_type = "mystery"
        else:
            opt_type = opt_dict[opt][0]
        plt.plot(p_b90s_nnc[(tes_freq,opt)], ratios[(tes_freq,opt)],
                 linestyle="None", marker=".", markersize=2.5, alpha=opacity, 
                 label=f"{tes_freq}GHz {opt_type}")
    do_norm_correct_plot_lims_and_titling(s)
    return (p_b90s_nnc,p_b90s,ratios,p)





def plot_norm_correct_p_b90_ratio_p_sat100mK_details(s,bins):
    p_b90s_nnc = {}
    p_b90s = {}
    ratios = {}
    dict_arr = [p_b90s_nnc,p_b90s,ratios]
    for dicty in dict_arr:
        for i in range(1, len(bins)):
            dicty[bins[i]] = []
    #p_b90_difs = []
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            if  sb not in s.results.keys() or ch not in s.results[sb].keys():
                continue
            i = 0
            while s.results[sb][ch]['p_sat100mK'] > bins[i] and i <len(bins):
                i+=1
            p_b90s[bins[i]] += d[f'p_b{s.use_per}']
            p_b90s_nnc[bins[i]] += s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
            ratios[bins[i]] += [d[f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                       for i in range(len(d[f'p_b{s.use_per}']))]
            #p_b90_difs += d[f'p_b{s.use_per}']-s.ramp_nnc[sb][ch][f'p_b{s.use_per}']
    p = plt.figure(figsize=default_figsize)
    for pb in p_b90s_nnc.keys():
        plt.plot(p_b90s_nnc[pb], ratios[pb],
                 linestyle="None", marker=".", markersize=1,\
                 label=f"{pb}")
    do_norm_correct_plot_lims_and_titling(s)
    return (p_b90s_nnc,p_b90s,ratios,p)

def plot_norm_correct_p_b90_ratio_detlines_details(s):
    #p_b90_difs = []
    count = 0
    p = plt.figure(figsize=default_figsize)
    for sb in s.ramp.keys():
        for ch, d in s.ramp[sb].items():
            if count+1 >= 10:
                break
            count += 1
            ratios = [d[f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                       for i in range(len(d[f'p_b{s.use_per}']))]
            plt.plot(d[f'p_b{s.use_per}'],ratios,linewidth=1,marker=".",markersize=1,\
                    label=f"sb{sb} ch{ch}")
    do_norm_correct_plot_lims_and_titling(s)
    return (p)

def do_norm_correct_plot_lims_and_titling(s):
    lims = s.key_info[f'p_b{s.use_per}']['lim']
    #lims = [0, max(max(p_b90s_nnc),max(p_b90s))*1.1]
    plt.xlim(lims)
    #plt.plot(lims,lims,linestyle='--',color="black",label="y=x line")
    plt.title(f"{s.dName} Norm-corrected p_b90/nnc p_b90 vs. nnc p_b90")
    plt.ylabel(f"Norm-corrected p_b90/nnc p_b90")
    plt.xlabel(f"no normal correction {s.key_info[f'p_b{s.use_per}']['name']} [{s.key_info[f'p_b{s.use_per}']['units']}]")
    plt.legend()

def get_ratios(s,sb,ch):
    return [s.ramp[sb][ch][f'p_b{s.use_per}'][i]/s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                    for i in range(len(s.ramp[sb][ch][f'p_b{s.use_per}']))]

def get_deltas(s,sb,ch):
    return [s.ramp[sb][ch][f'p_b{s.use_per}'][i]-s.ramp_nnc[sb][ch][f'p_b{s.use_per}'][i] \
                    for i in range(len(s.ramp[sb][ch][f'p_b{s.use_per}']))]




def add_det_line(s,sb,ch,color="black"):
    plt.plot(s.ramp_nnc[sb][ch][f'p_b{s.use_per}'],get_ratios(s,sb,ch),
             color=color,linewidth=1,linestyle=s.tes_dict[sb][ch]['linestyle'],
             label=f"sb{sb} ch{ch}; {s.tes_dict[sb][ch]['TES_freq']}GHz {s.tes_dict[sb][ch]['opt_name']}")
    plt.legend()
    






# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# =================== Bath_Ramp Class (Temp_Ramp child class) ==================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class Bath_Ramp(Temp_Ramp):
    '''
    This child class deals with fitting and plotting results of 
    bath Temp_Ramps. 
    
    Adds the following to its passed tes_dict's channel dictionaries if can fit
    (even if the channel is excluded from results because of its opt_value):
    k, k_err, Tc, Tc_err, n, n_err, cov_G_Tc_n (default) OR cov_k_Tc_n, 
    G, G_err, p_sat100mK, p_sat100mK_err
    
    ONE BIG FLAW: no accounting for optical power on unmasked detectors TODO
    ^somewhat done with cold load ramp...
    
    # =========================== (Child) VARIABLES ============================
    opt_values_exclude# Which to exclude from 'results' (NOT all_results)
    min_b_temp_span   # minimum temperature range of bath that channel must have 
                      # points spanning in order to fit it as a real channel. 
    use_k             # Whether to fit to k (if not, to G). G IS DEFAULT. 
    g_guess_by_freq   # {TES_freq (int): guessed G for fits in pW/mK } 
    Tc_guess          # in mK
    n_guess           # unitless
    
    # --- Constructed
    all_results       # {sb:{ch:{k, G, Tc, n, p_sat100mK, cov_G_Tc_n OR cov_k_Tc_n
    
    # Organizational, for plotting, does not include excluded opt_values:
    results           # {sb:{ch:{k, G, Tc, n, p_sat100mK, cov_G_Tc_n OR cov_k_Tc_n
    to_plot           # {'k':[],'G':[],'Tc':[],'n':[],'p_sat100mK':[]} <-but filled
    to_plot_TES_freqs # {TES_freq: {'k':[],'Tc':[],'n':[],'G':[],'p_sat100mK':[]}}
    
    # ======================= METHODS (Helpers indented) =======================
    __init__(s, test_device, ramp_type, therm_cal, metadata_fp_arr,metadata_temp_idx=0, 
             norm_correct=True, use_per=90, p_sat_cut=15, use_p_satSM=True, 
             fix_pysmurf_idx=False,input_file_type="pysmurf",
             sc_offset=True, bin_tod=True,use_cii=False, save_raw_ivas=False,
             opt_values_exclude=[],min_b_temp_span=21,use_k=False,
             g_guess_by_freq={30:0.050,40:0.080,90:0.090,150:0.165,225:0.3,285:0.5},
             Tc_guess=160,n_guess=3.5)
        do_fits(s)
            add_thermal_param_to_dict(s,dicty,use_k,k,g,Tc,n,cov,p_sat100mK_err)
            p_satofT(s,tb, k, Tc, n)
            p_satofTandG(s,tb, g, Tc, n)
    # ---------- Statistics Methods
    param_stats(s)
    # ---------- Plotting Methods
    fit_param_correlation_scatter(s, param1,param2, all_results=True)
    make_summary_plots(s, nBins=60,split_freq=True, p_sat_Range=None,
                       t_c_Range=None,g_Range=None, n_Range=None)
    
    # ====================+ EXTERNAL Analysis FUNCTIONS =======================
    examine_bath_ramp(rampy)  # cutting and thermal parameter summary
        thermal_table(s,masked_only=True) # for Confluence/dashboard
    make_thermal_param_export(s,ufm_name,cooldown, thermom_id=None) #for Kaiwen
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal, metadata_fp_arr, metadata_temp_idx=0, 
                 norm_correct=True, use_per=90, p_sat_cut=15, use_p_satSM=True, 
                 fix_pysmurf_idx=False,input_file_type="guess",
                 sc_offset=True, bin_tod=True,use_cii=False, save_raw_ivas=False,
                 opt_values_exclude=[],min_b_temp_span=21,use_k=False,
                 g_guess_by_freq={30:0.050,40:0.080,90:0.090,150:0.165,225:0.3,285:0.5},
                 Tc_guess=160,n_guess=3.5): 
        # SHOULD be opt_values_exclude, not bias lines
        super().__init__(test_device, ramp_type, therm_cal,metadata_fp_arr, 
                         metadata_temp_idx=metadata_temp_idx,
                         norm_correct=norm_correct,use_per=use_per, p_sat_cut=p_sat_cut,
                        use_p_satSM=use_p_satSM,fix_pysmurf_idx=fix_pysmurf_idx,
                        input_file_type=input_file_type,sc_offset=sc_offset,
                        bin_tod=bin_tod,use_cii=use_cii,save_raw_ivas=save_raw_ivas)
        # Now that the Temp_Ramp is set up...let's fit that data. 
                
        # Guesses for the fitting to start from. Important for yield. 
        s.g_guess_by_freq = g_guess_by_freq # int(freq):G in pW/mK
        s.Tc_guess = Tc_guess # mK
        s.n_guess = n_guess
        # passing do_fits more stuff
        s.opt_values_exclude = opt_values_exclude
        s.use_k = use_k
        s.min_b_temp_span = min_b_temp_span
        
        # This is a separate function so I can fiddle with things and re-run
        # As ex. I did with light leak experiment in SPB14, and in SPB3 when I 
        # didn't have time to fix the fitting function for the wacky procedure
        # from back when we didn't have real PID.
        s.do_fits()
        
        s.key_info['k']     = {'name' : 'Coefficient of Thermal Conduction',
                               'units' : 'pW/mK^n'}
        s.key_info['G']     = {'name' : "$G$, Differential Thermal Conductance",
                               'units' : 'pW/mK'}
        s.key_info['G1000'] = {'name' : "$G$, Differential Thermal Conductance",
                               'units' : 'pW/K'}
        s.key_info['Tc']    = {'name' : r"$T_c$, TES critical temperature",
                               'units' : 'mK'}
        s.key_info['n']     = {'name' : \
                           "$n$, power law index (1+thermal conductance exponent)",
                               'units' : ''}
        s.key_info['p_sat100mK'] = {'name' : 'calculated $P_{b'+ f'{s.use_per}'+'}$@$T_b$=100mK, no optical power',
                                     'units' : 'pW'}
        # ^ADD THE OTHERS TO THIS!!!
   

        
        
    def do_fits(s):
        opt_values_exclude,use_k, g_guess_by_freq,Tc_guess, n_guess = \
          (s.opt_values_exclude,s.use_k, s.g_guess_by_freq,s.Tc_guess, s.n_guess)
        mux_map = s.mux_map
        tes_dict = s.tes_dict
        
        # tes_dict = {sb:{ch:{TES_atts}}}; TES_freq,pol,opt,opt_name,linestyle
        #       added by data analysis: R_n?; k,Tc,n,G,psat_100mK; opt_eff; 
        
        s.all_results = {}
        results = {}
        
        #print(['k','Tc','n'])
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items():
                if len(d[f'p_b{s.use_per}']) < 4: # originally < 4
                    continue
                # Merging bathsweeps resulted in ones getting 4 points that have no business being fit
                # So, check for a minimum temperature range too:
                if max(d['temp'])-min(d['temp']) < s.min_b_temp_span: # 21; switched to 10 for SPB3 # in mK
                    continue
                if str(s.mux_map[sb][ch]['TES_freq']).isnumeric() and \
                   int(s.mux_map[sb][ch]['TES_freq']) in s.g_guess_by_freq.keys():
                    g_guess = s.g_guess_by_freq[int(s.mux_map[sb][ch]['TES_freq'])]
                else:
                    g_guess = 0.130
                k_guess = g_guess/(n_guess*Tc_guess**(n_guess-1))
                try:
                    if use_k:
                        parameters = curve_fit(
                            s.p_satofT,
                            np.array(d['temp']),
                            np.asarray(d[f'p_b{s.use_per}']), # really p_bias @ p_tes=p_sat
                            p0 = [k_guess, Tc_guess, n_guess],
                            absolute_sigma=True)
                    else:
                        parameters = curve_fit(
                            s.p_satofTandG,
                            np.array(d['temp']),
                            np.asarray(d[f'p_b{s.use_per}']),  # really p_bias @ p_tes=p_sat
                            p0 = [g_guess, Tc_guess, n_guess],
                            absolute_sigma=True)
                except RuntimeError:
                    continue
                cov = parameters[1]
                # a fit exists!
                # NOW we're convinced this is real, update tes_dict if it doesn't have it:
                s.test_device.check_tes_dict_for_channel_and_init(sb, ch)
                if use_k:
                    k, tc, n = parameters[0]
                    k_err, Tc_err, n_err = np.sqrt(np.diag(parameters[1]))
                else:
                    g, tc, n = parameters[0]
                    g_err, Tc_err, n_err = np.sqrt(np.diag(parameters[1]))
                # The above is...not exactly enough, as covariances likely exist. 
                # Oh yeah. Everything but Tc has larger covariances with another variable than
                # its own variance. Tc has one (with n) that's close in magnitude (though negative). 
                # MUST NOT FORGET COVARIANCE IS LARGE
                if use_k:
                    g = n*k*(tc**(n-1))
                    # Jacobian of G = [[∂G/∂k, ∂G/∂Tc, ∂G/∂n]] evaluated at k,Tc,n=parameters[0]
                    jac_G = np.array([[n*tc**(n-1), 
                                       n*(n-1)*k*tc**(n-2), 
                                       k*tc**(n-1)+n*k*np.log(tc)*tc**(n-1)]])
                    g_err = np.sqrt(\
                        np.matmul(jac_G, np.matmul(parameters[1],
                                                   np.transpose(jac_G)))[0][0])
                else:
                    k= g / (n*tc**(n-1))
                    # Jacobian of k = [[∂k/∂G, ∂k/∂Tc, ∂k/∂n]] evaluated at G,Tc,n=parameters[0]
                    jac_k = np.array([[1/(n*tc**(n-1)),
                                       g*(1-n)/(n*tc**n),
                                       -g*(1/(n**2*tc**(n-1))+np.log(tc)/(n*tc**(n-1)))]])
                    k_err = np.sqrt(\
                        np.matmul(jac_k,np.matmul(parameters[1],
                                                  np.transpose(jac_k)))[0][0])
                # Jacobian of p_sat100mk = [[∂p_sat/∂k, ∂p_sat/∂Tc, ∂p_sat/∂n]] evaluated at T_b=100mK and k,Tc,n=parameters[0]
                tb = 100
                if use_k:
                    jac_p_sat = np.array([[tc**n-tb**n, 
                                           n*k*tc**(n-1),
                                           k*(np.log(tc)*tc**n-np.log(tb)*tb**n)]])
                    p_sat100mK_err = np.sqrt(\
                        np.matmul(jac_p_sat,np.matmul(parameters[1],
                                                      np.transpose(jac_p_sat)))[0][0])
                else:
                    jac_p_sat = np.array([[(tc-tb**n/tc**(n-1))/n, 
                                           (g/n)*(1-(1-n)*tb**n/tc**n),
                                           tc*g*(-1/n**2-(1/n)*np.log(tb/tc)*(tb/tc)**n+(1/n**2)*(tb/tc)**n)]])
                    p_sat100mK_err = np.sqrt(\
                        np.matmul(jac_p_sat,np.matmul(parameters[1],
                                                      np.transpose(jac_p_sat)))[0][0]) 
                
                # update TES dict.
                r_n = s.det_R_n[sb][ch]
                s.add_thermal_param_to_dict(tes_dict[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err,r_n)
                # not sure why I made these separate, since cov already gets em
                tes_dict[sb][ch]['k_err'] = k_err
                tes_dict[sb][ch]['G_err'] = g_err
                tes_dict[sb][ch]['Tc_err'] = Tc_err
                tes_dict[sb][ch]['n_err'] = n_err


                    
                
                                                   
                # update internal results
                if sb not in s.all_results.keys():
                    s.all_results[sb] = {}
                s.all_results[sb][ch] = {}
                s.add_thermal_param_to_dict(s.all_results[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err,r_n)
                
                # Don't include this one in the analysis.
                if mux_map[sb][ch]['opt'] in opt_values_exclude:
                    continue

                # Should really change this to tes_dict, include everything (no cuts above)
                # and make to_plots here. 
                # Should I change to use same array org. as mux_map? makes two-tier fors...
                # but easier-to-remember access, no re's. Add 'name' if added. 
                #k, Tc, n = parameters[0]
                if sb not in results.keys():
                    results[sb]={}
                results[sb][ch] = {}
                s.add_thermal_param_to_dict(results[sb][ch],use_k,k,g,tc,n,cov,p_sat100mK_err,r_n)
        s.results  = results  
        
        # CONTINUING ON TO PREP FOR PLOTTING
        to_plot = {'k':[],'G':[],'Tc':[],'n':[],'p_sat100mK':[]}
        for sb in results.keys():
            for ch, d in results[sb].items():
                for param, val in d.items():
#                     if param not in to_plot.keys():
#                         to_plot[param] = []
                    if param in to_plot.keys():
                        to_plot[param].append(val)
        s.to_plot = to_plot
        # to_plot_TES_freqs = {TES_freq: {'k':[],'Tc':[],'n':[],'G':[],'p_sat100mK':[]}}
        to_plot_TES_freqs = {}
        for sb in results.keys():
            for ch, d in results[sb].items():
#             match = re.search('sb(\d)+_ch(\d+)',sbch)
#             sb, ch = int(match.group(1)), int(match.group(2))
                if tes_dict[sb][ch]['TES_freq'] not in to_plot_TES_freqs.keys():
                    to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']] = {'k':[],'G':[],'Tc':[],'n':[],'p_sat100mK':[]}
                    #print(tes_dict[sb][ch]['TES_freq'])
                for param, val in d.items():
                    if param in to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']].keys():
                        to_plot_TES_freqs[tes_dict[sb][ch]['TES_freq']][param].append(val)
        #print(to_plot_TES_freqs)
        s.to_plot_TES_freqs = to_plot_TES_freqs
            
    def add_thermal_param_to_dict(s,dicty,use_k,k,g,tc,n,cov,p_sat100mK_err,r_n):
        dicty['k'] = k
        dicty['G'] = g
        dicty['Tc'] = tc
        dicty['n'] = n
        if use_k:
            dicty['cov_k_Tc_n'] = cov
        else:
            dicty['cov_G_Tc_n'] = cov
        dicty['p_sat100mK'] = s.p_satofTandG(100,g, tc, n)
        dicty['p_sat100mK_err'] = p_sat100mK_err
        dicty['R_n'] = r_n
        
    # =================== Fitting methods
    def p_satofT(s,tb, k, tc, n):
        return k * (tc**n - tb**n)

    def p_satofTandG(s,tb, g, tc, n):
        return (g/n) * (tc - tb**n/tc**(n-1))
    
    # ==================== statistics methods
    def param_stats(s):
        # all_results option not yet implemented. 
        # Should have this iterate through the dictionary,
        # Really, to find max/min/outliers sb/ch
        # and then easy to have all_results or normal
        # should restructure bath ramp a bit for it though. 
        stats_dict = {}
        statsy = {'k':{},'G':{},'Tc':{},'n':{},'p_sat100mK':{}}
        for tes_freq in s.to_plot_TES_freqs:
            freq_dict = copy.deepcopy(statsy)
            for stat, statlst in s.to_plot_TES_freqs[tes_freq].items():
                lst = np.array(statlst)
                # Units should get in here. 
                freq_dict[stat]['units'] = '[' + s.key_info[stat]['units'] +']'
                freq_dict[stat]['median'] = np.median(lst)
                freq_dict[stat]['average'] = np.average(lst)
                freq_dict[stat]['stdev'] = np.std(lst)
                freq_dict[stat]['min'] = min(lst)
                freq_dict[stat]['max'] = max(lst)            
            stats_dict[tes_freq] = freq_dict
        return stats_dict
    
    # ==================== plotting methods
    def fit_param_correlation_scatter(s, param1,param2, all_results=True):
        p1 = {}
        p2 = {}
        fit_dict = s.results
        if all_results:
            fit_dict = s.all_results # Will include tes's in excluded_opt_values, typically the unmasked.
        for sb in fit_dict.keys():
            for ch, d in fit_dict[sb].items():
                freq = s.tes_dict[sb][ch]['TES_freq']
                opt_name = s.tes_dict[sb][ch]['opt_name']
                if freq not in p1.keys():
                    p1[freq] = {}
                    p2[freq] = {}
                if opt_name not in p1[freq].keys():
                    p1[freq][opt_name] = []
                    p2[freq][opt_name] = []
                p1[freq][opt_name].append(d[param1])
                p2[freq][opt_name].append(d[param2])
        order_list = []    
        plt.figure(figsize=default_figsize) 
        for freq in p1.keys():
            for opt_name in p1[freq].keys():
                plt.plot(p1[freq][opt_name],p2[freq][opt_name],marker=".",\
                         linestyle="None",alpha=0.5,label=f"{freq}GHz {opt_name}")        
        plt.legend()
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title(f"{s.dName} {param2} vs. {param1}")
        return (p1,p2)

    def make_summary_plots(s, nBins=60,split_freq=True,
                           p_sat_Range=None,t_c_Range=None,g_Range=None, n_Range=None):
        to_plot = s.to_plot
        to_plot_TES_freqs = s.to_plot_TES_freqs
        if not split_freq:
            p_sats, t_cs, gs, ns = [to_plot['p_sat100mK']], [to_plot['Tc']],[[1000*g for g in to_plot['G']]], [to_plot['n']]
            tes_freqs_labels = ["all freqs"]
        else: # to_plot_TES_freqs = {TES_freq: {'k':[],'Tc':[],'n':[],'G':[],'p_sat100mK':[]}}
            freq_order_guess = []
            for freq in to_plot_TES_freqs.keys():
                if is_float(freq):
                    freq_order_guess.append(try_to_numerical(freq))
            freq_order_guess = [freq for freq in np.sort(np.array(freq_order_guess))]
            if freq_order_guess[0] <= 0:# this is for consistency in frequency color display
                freq_order_guess = freq_order_guess[1:]+[freq_order_guess[0]]
            tes_freqs_labels = [str(freq) + " GHz" for freq in freq_order_guess]
            p_sats = [to_plot_TES_freqs[key]['p_sat100mK'] for key in freq_order_guess]
            t_cs = [to_plot_TES_freqs[key]['Tc'] for key in freq_order_guess]
            gs = [[1000*g for g in to_plot_TES_freqs[key]['G']] for key in freq_order_guess]
            ns = [to_plot_TES_freqs[key]['n'] for key in freq_order_guess]
        
        if not p_sat_Range:
            p_sat_Range = (min(to_plot['p_sat100mK']), max(to_plot['p_sat100mK']))  
        if not t_c_Range:
            t_c_Range = (min(to_plot['Tc']), max(to_plot['Tc'])) 
        if not g_Range:
            g_Range = (1000*min(to_plot['G']), 1000*max(to_plot['G'])) 
        if not n_Range:
            n_Range = (min(to_plot['n']), max(to_plot['n']))

        std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(nrows=4, figsize=(6,9))
        
        '''alpha argument to .hist is transparency, which is unnecessary here because only plotting one dataset
        on that histogram. Potentially useful if I separate by frequency, however.'''

        #h = ax[0].hist(p_sats,  alpha=0.4, bins=nBins, label=tes_freqs_labels, rwidth=1.0) # range=(0,12)
        for i in range(len(tes_freqs_labels)):
            h = ax[0].hist(p_sats[i],  alpha=0.4, bins=nBins, range=p_sat_Range, label=tes_freqs_labels[i]) # range=(0,12)
            med = np.nanmedian(p_sats[i])
            ax[0].axvline(med, linestyle='--',color=std_colors[i],label=f'median={med:.1f} pW')
        ax[0].set_xlabel('p_sat at 100 mK [pW]')
        ax[0].set_ylabel('# of TESs')
        ax[0].set_title(' ')

        #h = ax[1].hist(t_cs,  alpha=0.4, bins=nBins, label=tes_freqs_labels ) #range=(100,210) #to_plot['Tc'], range=(175,205), alpha=0.4, bins=30
        for i in range(len(tes_freqs_labels)):
            h = ax[1].hist(t_cs[i],  alpha=0.4, bins=nBins, range=t_c_Range, label=tes_freqs_labels[i] )
            med = np.nanmedian(t_cs[i])
            ax[1].axvline(med, linestyle='--',color=std_colors[i],label=f'median={med:.0f} mK')
        # med = np.nanmedian(to_plot['Tc'])
        # ax[1].axvline(med, linestyle='--',label='median=%.3fmK'%med)
        ax[1].set_xlabel('Tc [mK]')
        ax[1].set_ylabel('# of TESs')
        ax[1].set_title(' ')

        #h = ax[2].hist(gs, alpha=0.4, bins=nBins, label=tes_freqs_labels) #to_plot['G'], alpha=0.4, range=(30,270), bins=30
        for i in range(len(tes_freqs_labels)):
            h = ax[2].hist(gs[i], alpha=0.4, bins=nBins, range=g_Range, label=tes_freqs_labels[i])
            med = np.nanmedian(gs[i])
            ax[2].axvline(med, linestyle='--',color=std_colors[i],label=f'median={med:.0f} pW/K')
        ax[2].set_xlabel('G [pW/K]')
        ax[2].set_ylabel('# of TESs')
        ax[2].set_title(' ')

        #h = ax[3].hist(ns,  alpha=0.4, bins=nBins, label=tes_freqs_labels) #range=(0,5)
        for i in range(len(tes_freqs_labels)):
            h = ax[3].hist(ns[i],  alpha=0.4, bins=nBins, range=n_Range, label=tes_freqs_labels[i])
            med = np.nanmedian(ns[i])
            ax[3].axvline(med, linestyle='--',color=std_colors[i],label=f'median={med:.2f}')
        ax[3].set_xlabel('n')
        ax[3].set_ylabel('# of TESs')
        ax[3].set_title(' ')

        for ind in [0,1,2,3]:
            ax[ind].legend(fontsize='small') #, loc=2

        totNumDets = sum([len(dataset) for dataset in t_cs])
        numDets = "(numFit = "+ str(totNumDets) + "; "+ ", ".join([str(len(t_cs[i])) + "x"+ tes_freqs_labels[i][:-4] for i in range(len(tes_freqs_labels))]) + ")"
        
        # Maybe add something to note if unmasked excluded?
        plt.suptitle(f"{s.dName}\n" + numDets, fontsize=16)
        plt.tight_layout()
        return ax

    
# ==============================================================================
# -------------------------- Bath_Ramp Analysis --------------------------------
# ==============================================================================

def examine_bath_ramp(rampy):
    # you know, maybe make this nnc?
    if rampy.norm_correct:
        rampy.plot_ramp_keys_by_BL(rampy.ramp_raw_arr_nnc[0],
                                   'temp_raw','p_satSM',prefix="Raw, nnc ")
    else:
        rampy.plot_ramp_keys_by_BL(rampy.ramp_raw_arr[0],'temp_raw',
                                   'p_satSM', prefix="Raw, nnc ")
    #rampy.plot_ramp_keys_by_BL(rampy.ramp_raw_arr_nnc[0], 'f'temp{s.use_per}R_n'','p_b90R_n',prefix='raw, nnc ',tes_dict=True)
    report_ramp_IV_cat_breakdown(rampy)
    rampy.plot_ramp_keys_by_BL(rampy.ramp, 'temp','p_b90',prefix='Cut ',tes_dict=True)
    if rampy.ramp_type == 'bath':
        rampy.make_summary_plots()
        rampy.make_summary_plots(p_sat_Range=[0,15], t_c_Range=[150,200],g_Range=[0,400],n_Range=[2,5])
    thermal_table(rampy)

def thermal_table(s,masked_only=True):
    '''Creates a confluence markdown thermal summary table. 
    Also prints tab-separated version, for the uxm dashboard,
    and a version with stdevs, for visibility.'''
    rslt = s.results
    if not masked_only:
        rslt=s.all_results
    num_fit = 0
    for sb in rslt.keys():
        num_fit += len(rslt[sb].keys())
    ax = {'TES_freq':np.full((num_fit,),-42,dtype=int),
          'sb':np.full((num_fit,),-42,dtype=int),
          'ch':np.full((num_fit,),-42,dtype=int),
          'p_sat100mK':np.full((num_fit,),np.nan),
          'Tc':np.full((num_fit,),np.nan),
          'R_n':np.full((num_fit,),np.nan),
          'G':np.full((num_fit,),np.nan),
          'n':np.full((num_fit,),np.nan)} # maybe include all the data and 
          # save this?
    i=0
    for sb in rslt.keys():
        for ch, d in rslt[sb].items():
            ax['sb'][i] = sb
            ax['ch'][i] = ch
            ax['TES_freq'][i] = s.tes_dict[sb][ch]['TES_freq']
            ax['R_n'][i] = s.tes_dict[sb][ch]['R_n']*1000
            ax['G'][i] = d['G']*1000
            for key in ['p_sat100mK','Tc','n']:
                ax[key][i] = d[key]
            i+=1
    freq1,freq2 = s.test_device.tes_freqs # to_plot does cleverer version...
    print(f"   ==================== {s.dName} ======================")
    print("---- Can't use colspan in Confluence markdown, have to merge cells manually."\
          +" Ctrl+Shift+C -> Ctrl+Shift+V does not preserve merging.\n")
    head1 = "|| || Rn (mΩ) || Tc (mK) || || || Psat (pW) (Masked, at 100mK, at 90% Rn)|| ||  G (pW/K) || || n || ||"
    head2 = f"|| || All || All ||" +f"{freq1}|| {freq2}||"*4
    print(head1)
    print(head2)
    def val_arr(ax,freq_list,fn ):
        val_starts = [fn(ax['R_n']),fn(ax['Tc'])] 
        therms = [[fn(ax[key][ax['TES_freq'] == freq]) for freq in freq_list] 
                  for key in ['Tc','p_sat100mK','G','n']]
        for therm in therms:
            val_starts = val_starts + therm
        return val_starts
    averages = val_arr(ax,[freq1,freq2],np.average) 
    medians = val_arr(ax,[freq1,freq2],np.median) 
    stdevs = val_arr(ax,[freq1,freq2],np.std)
    #therms = '||'.join(['||'.join([f"{np.average(ax[key][ax['TES_freq'] == freq])}" 
#                                    for freq in [freq1,freq2]])
#                         for key in ['Tc','p_sat100mK','G','n']])
    round_to_n = lambda x, n: x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    rnd_avgs = [f"{int(round_to_n(avg,3))}" if round_to_n(avg,3) >= 100 else f"{round_to_n(avg,2)}" for avg in averages]
    rnd_med = [f"{int(round_to_n(avg,3))}" if round_to_n(avg,3) >= 100 else f"{round_to_n(avg,2)}" for avg in medians]
    rnd_stdevs = [f"{int(round_to_n(avg,3))}" if round_to_n(avg,3) >= 100 else f"{round_to_n(avg,2)}" for avg in stdevs]
    
    
    rstr = f"|| {s.dName[:s.dName.index(' ')]} (Detector-id, batch)  |" #[f"{round_to_n(avg,3)}" for avg in averages]
    print(rstr + "|".join( rnd_med )+ "|") #[f"{avg:.3}" for avg in averages]
    print("")
    print("---- uxm_dashboard, may need to paste to notepad first then the dashboard:")
    print('\t'.join(rnd_avgs))
    print("---- excel/sheets viewable with stdevs, may need notepad first")
    print(head1[3:].replace("||","\t")[1:])
    print(head2[3:].replace("||","\t")[1:])
    print( "\t".join([rnd_avgs[i]+"±"+f"{int(100*stdevs[i]/averages[i])}%" for i in range(len(rnd_avgs))]))
    #print("---- Farthest Outliers -----")
    #print("**TODO**")
       

    
def make_thermal_param_export(s,ufm_name,cooldown, thermom_id=None):
    '''The kind that Kaiwen uses for optical stuff. She mostly just uses R_n and G
       s is a bathramp for now'''
    tpe = {'data':{},
           'metadata':{'units': {'psat100mk': 'pW',
                                    'tc': 'K',
                                    'g': 'pW/K',
                                    'n': '',
                                    'k': '',  # that's wrong, k has units! weird ones too.
                                    'R_n': 'ohms'},
                        'dataset': s.metadata_fp_arr,
                        'restrict_n': False,
                        'allowed_rn': [0.007, 0.009],
                        'cut_increasing_psat': True,
                        'thermometer_id': thermom_id,
                        'temp_list': s.temp_list_raw,
                        'optical_bl': [],#[i if not i in s.test_device.masked_biaslines for i in np.arange(12)]]
                        'temp_offset': None,
                        'temp_scaling': None,
                        'temps_to_cut': None}} # should ass nso, cii, other options.
    for i in np.arange(12):
        if i not in s.test_device.masked_biaslines:
            tpe['metadata']['optical_bl'].append(i)
    #if len(s.metadata_fp_arr) > 1:
    #    tpe['metadata']['dataset']=s.metadata_fp_arr
    c_dict = {'R_n':'R_n',
              'g':'G', 'sigma_g':'G_err',
              'k':'k', 'sigma_k':'k_err',
              'tc':'Tc', 'sigma_tc':'Tc_err',
              'n':'n', 'sigma_n':'n_err',
             'psat100mK':'p_sat100mK','sigma_psat100mK':'p_sat100mK_err'}
    for sb in s.tes_dict.keys():
        for ch,d in s.tes_dict[sb].items():
            if not 'biasline' in d.keys() \
               or d['biasline'] not in [0,1,2,3,4,5,6,7,8,9,10,11]:
                continue
            bl = d['biasline']
            has_data = True
            for key, item in c_dict.items():
                if item not in d.keys():
                    has_data=False
                    print(f'{sb} {ch} bl={bl} no {key}')
                    break
            if not has_data:
                continue
            if bl not in tpe['data'].keys():
                tpe['data'][bl] = {}
            if sb not in tpe['data'][bl].keys():
                tpe['data'][bl][sb] = {}
            tpe['data'][bl][sb][ch] = {key:d[item] for key,item in c_dict.items()}
            # Adjust units
            tpe['data'][bl][sb][ch]['tc'] = tpe['data'][bl][sb][ch]['tc']/1000.0
            tpe['data'][bl][sb][ch]['sigma_tc'] = tpe['data'][bl][sb][ch]['sigma_tc']/1000.0
            tpe['data'][bl][sb][ch]['g'] = tpe['data'][bl][sb][ch]['g']*1000.0
            tpe['data'][bl][sb][ch]['sigma_g'] = tpe['data'][bl][sb][ch]['sigma_g']*1000.0
    if not os.path.exists(f'/data/uxm_results/{ufm_name}/{cooldown}'):
        os.makedirs(f'/data/uxm_results/{ufm_name}/{cooldown}')
    fp = f'/data/uxm_results/{ufm_name}/{cooldown}/{make_filesafe(s.dName)}_{s.metadata_fp_arr[0][-12:-4]}.npy'
    np.save(fp,tpe)
    print("Thermal parameter exports for Kaiwen at:")
    print(fp)
    return tpe          
            

    





# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# =============== Coldload_Ramp Class (Temp_Ramp child class) ==================
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

class Coldload_Ramp(Temp_Ramp):
    '''
    This child class deals with fitting and plotting results of 
    coldload Temp_Ramps. You should already have run a Bath_Ramp 
    on the tes_dict you pass to Coldload_Ramp.
    
    # ONLY WORKS FOR SPB THUS FAR. 
    
    Adds the following to its passed tes_dict's channel dictionaries:
    - TODO
    
    DEPRECATED, no longer does this: Adds the following to super().ramp, per channel: 
    'bath_temp': [] and 'bath_temp_err': []
    # ^ estimated. Not really using those now though.  
   
    =========================== (Child) VARIABLES ============================
    # ---- REQUIRED Child init var:
    bath_ramp   # a Bath_Ramp() that this Coldload_Ramp() references.
    # ---- Optional Child init vars (and defaults):
    do_opt      # = False
    device_z    # = -42   # (not provided)
    opt_power   # = {}    # loads if given, or constructs if not (& do_opt)
    br_cl_temp  # = -42   # The cold load temperature during the bath ramp. 
    spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
    # these below 3 are for me exploring what could cause T_b_est issue
    scale_G     # = 1.0   
    scale_n     # = 1.0
    add_Tc      # = 0
    
    # --- Borrowed vars; gave this class direct references b/c used frequently
    tes_dict    # same as the test_device's.
    mux_map     # same as the test_device's. 
    
    # ---- Constructed Vars:
    opt_power   # {(det_x,det_y):{'load_obj':a spb_opt.SPBOptLd, <TES_freq (int)>:{
                #      'opt_power': [in pW, for each cl_temp], 
                #      'opt_fit': ([<slope>, <constant>],covariance matrix) 
                #      'opt_power_during_bath_ramp': pW}   }}
    # 'results' exists to not have to check the big ramp arrays for if ch was fit
    # and has a tes_dict entry. It also summarizes results.
    results     # {sb:{ch:{  (see below)
        # All: (Note:  p_b90 = p_bias at saturation (90% R_n))
            # 'cl_temp': [], 'p_b90_fit': ([<slope>, <constant>],covariance matrix),
        # If masked/dark: 
            # 't_b_est_thermal':{'t_bath':[],'t_bath_error':[]}
            # 'bath_temp_fit' = (popt, pcov) # linear fit
        # If opt unmasked: 
            # 'no_subtraction_opt_eff', and 
            # 'loc' = (det_x, det_y) # from mux_map b/c otherwise unwieldy.
            # 't_b_est':{'t_bath':[],'t_bath_error':[]} # used in together fit; 
                # }}
    slopes      # Slopes of fitlines of the tb_est_thermal for each masked/dark
    
    # ======================= METHODS (Helpers indented) =======================
    # ---------- Initialization and loading methods
    __init__(s, test_device, ramp_type, therm_cal, 
             metadata_fp_arr, metadata_temp_idx=0,
             bath_ramp, norm_correct=True, p_sat_cut=15, 
             use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="pysmurf",
             cooldown_name='',
             do_opt=False, device_z=-42, opt_power={}, br_cl_temp=-42,
             spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
             scale_G=1.0, scale_n=1.0,add_Tc=0)
        init_opt_power(s,device_z, cooldown_name, freq,loc)

    # ---------- Fitting Methods
    together_fit_opt_unmasked(s,tb_fit,sb,ch,make_plot=False, own_fig=True)
    # --- scipy.optimize.curvefit() fit functions (helpers of multiple funcs)
        tbath_of_dark_TES(s,P_b, g,T_c,n)
        tbath_line(s,tcl, slope, constant)
        fit_line(s, x, slope, constant)
        coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp_arr, G, Tc, n, 
                                                      opt_eff)
            point_coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp, G, Tc, n,
                                                                opt_eff)

    # ---------- Plotting Methods
    plot_est_T_bath(s)
    plot_est_T_bath_by_BL(s,tes_dict=True)
    plot_T_b_slopes_vs_param(s,param_name)
    '''
    
    def __init__(s, test_device, ramp_type, therm_cal, 
                 metadata_fp_arr, bath_ramp, metadata_temp_idx=0,
                 norm_correct=True, use_per=90, p_sat_cut=15, 
                 use_p_satSM=True, fix_pysmurf_idx=False,input_file_type="guess",
                 sc_offset=True,bin_tod=True,use_cii=False,save_raw_ivas=False,
                 cooldown_name='',
                 do_opt=False, device_z=-42, opt_power={}, br_cl_temp=-42,
                 spb_light_leak_sim=0, # adds horn's opt_power * this to masked opts
                 scale_G=1.0, scale_n=1.0,add_Tc=0): 
        # SHOULD be opt_values_exclude, not bias lines
        super().__init__(test_device, ramp_type, therm_cal, metadata_fp_arr, 
                         metadata_temp_idx=metadata_temp_idx,
                         norm_correct=norm_correct,use_per=use_per,p_sat_cut=p_sat_cut,
                        use_p_satSM=use_p_satSM,fix_pysmurf_idx=fix_pysmurf_idx,
                        input_file_type=input_file_type,sc_offset=sc_offset,
                        bin_tod=bin_tod,use_cii=use_cii,save_raw_ivas=save_raw_ivas)
        # Now that the Temp_Ramp is set up...set up results, calculate universal stuff.
        tes_dict = s.tes_dict
        mux_map = s.mux_map
        s.bath_ramp = bath_ramp
        s.br_cl_temp = br_cl_temp 
        #s.spb_light_leak_sim = spb_light_leak_sim
        
        # ========= Setting up s.results: which dets can we work with
        s.results = {}  
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items(): # below: need optical classification to do anything here
                if sb not in tes_dict.keys() or ch not in tes_dict[sb].keys():
                    continue
                if not is_float(tes_dict[sb][ch]['opt']):
                    continue # again, need that optical classification to do anything here.
                if not len(d['temp']) >=4: # Gotta have data to do analysis.
                    continue
                if sb not in s.results.keys():
                    s.results[sb] = {}
                if ch not in s.results[sb].keys():
                    s.results[sb][ch] = {}
                s.results[sb][ch]['cl_temp'] = s.ramp[sb][ch]['temp']
    
        # ========= Calculating/importing optical load =================
        # ONLY WORKS FOR SPB CURRENTLY.
        # You have to follow the setup explained in SPB_opt/Get_optical_power.ipynb first
        # Calculating optical power takes a long time,
        # Many detectors are under the same horn center.
        # so we do it by horn center location and call that dictionary when we want it.
        # Also, we provide an option to import a previously-established opt_power
        if do_opt:
            if opt_power: # did they import one?
                s.opt_power = opt_power
                do_calc=False
            else:# gotta do it ourself.
                s.opt_power = {}
                do_calc = True
            for sb in s.results.keys():
                for ch, d in s.results[sb].items():
                    if tes_dict[sb][ch]['opt'] == 1 \
                      and is_float(tes_dict[sb][ch]['TES_freq'])  \
                      and len(s.ramp[sb][ch]['temp'])>2:
                        loc = (mux_map[sb][ch]['det_x'],
                               mux_map[sb][ch]['det_y'])
                        s.results[sb][ch]['loc'] = loc
                        freq = int(float(tes_dict[sb][ch]['TES_freq']))
                        if do_calc:
                            # Below had excessive line length, so made it a function
                            s.init_opt_power(device_z,
                                             cooldown_name, 
                                             freq,loc)
                        s.opt_power[loc][freq]['opt_power_during_bath_ramp'] = \
                            s.opt_power[loc]['load_obj'].get_power(T=s.br_cl_temp,freq=freq)
        
        
        # ========== Fit p_b90s as lines.  
        # ==== And Adjust opt masked p_b90s if spb_light_leak_sim > 0
        for sb in s.results.keys():
            for ch in s.results[sb].keys():
                if do_opt and spb_light_leak_sim > 0 \
                and tes_dict[sb][ch]['opt'] == 0.25: # optical masked
                    # only one horn, this is spb-only
                    for key in s.opt_power.keys():
                        loc=key
                    opt_powers_to_add = s.opt_power[loc][int(s.tes_dict[sb][ch]['TES_freq'])]['opt_power']
                    # this...actually WON'T  fail if more than one ramp_raw
                    for i in range(len(s.ramp[sb][ch]['temp'])):
                        j = s.temp_list_raw.index(s.ramp[sb][ch]['temp'][i])
                        s.ramp[sb][ch][f'p_b{s.use_per}'][i] += spb_light_leak_sim*opt_powers_to_add[j] 
                    #print(s.ramp[sb][ch][f'p_b{s.use_per}'])
                (popt, pcov) = curve_fit(s.fit_line, s.results[sb][ch]['cl_temp'], 
                                         s.ramp[sb][ch][f'p_b{s.use_per}'])
                s.results[sb][ch][f'p_b{s.use_per}_fit'] = (popt, pcov)
                # That isn't really a line! Look at mv6 to see clearly. 
            # should I do this:
            # 'opt', 'opt_name' # stolen from tes_dict because dang this uses them a lot. 
            
        
        # ========== Opt power w/out dark subtraction (if passed opt_power)
        # Assuming that T_b stays the same... 
        # η = -(d[P_bas_unmasked]/d[T_CL]) / (d[P_opt]/d[T_CL])
        if do_opt:
            for sb in s.results.keys():
                for ch, d in s.results[sb].items():
                    # Tes_dict can contain ones with only one point with Kaiwen's stuff..
                    if not 'loc' in d.keys(): # is optical unmasked, has sensible TES_freq, at least 3 points to fit
                        continue
                    my_slope = d[f'p_b{s.use_per}_fit'][0][0]
                    my_freq = int(float(tes_dict[sb][ch]['TES_freq']))
                    my_opt_slope = s.opt_power[d['loc']][my_freq]['opt_fit'][0][0]
                    s.results[sb][ch]['no_subtraction_opt_eff'] = -my_slope/my_opt_slope

        # optical power ignoring n differences
        # (should import from outside function?)   
        

        # ==================== T_b_est
        # let's check T_b, and look at the
        # slope of the resulting lines.
        s.slopes=[]
        br_r = s.bath_ramp.all_results
        for sb in s.ramp.keys():
            for ch, d in s.ramp[sb].items():
                if sb not in s.tes_dict.keys() or ch not in s.tes_dict[sb].keys():
                    continue # should I be iterating over tes_dict instead?
                if sb not in br_r.keys() or ch not in br_r[sb].keys():
                    continue
                # Tes_dict can contain ones with only one point with Kaiwen's stuff..
                if not s.tes_dict[sb][ch]['opt'] == 1.0 and len(d['temp']) >=4:   
                    # ^ not unmasked optical fab, at least 4 points
                    if sb not in s.results.keys():
                        s.results[sb] = {}
                    if ch not in s.results[sb].keys():
                        s.results[sb][ch] = {}
                    s.results[sb][ch]['cl_temp'] = s.ramp[sb][ch]['temp']
                    s.results[sb][ch]['t_b_est_thermal']={'t_bath':[],'t_bath_err':[]}
                    
                    g, tc, n, cov = (scale_G*br_r[sb][ch]['G'], add_Tc+br_r[sb][ch]['Tc'], \
                               scale_n*br_r[sb][ch]['n'], br_r[sb][ch]['cov_G_Tc_n'])
                    #print(f'{ch},{g},{tc},{n},{cov}')
                    for i in range(len(d['temp'])):
                        pb = d[f'p_b{s.use_per}'][i]
                        bt = s.tbath_of_dark_TES(pb, g,tc,n)
                        s.results[sb][ch]['t_b_est_thermal']['t_bath'].append(bt)
                        # Now, let's get the error. 
                        # I checked my derivative with mathematica, so only errors should be typos...
                        # Double checked resultant numbers with mathematica too. 
                        bttn = bt**n 
                        jac_tb = np.array([[
                            (1/n)*(bttn)**(1/n-1) * (n*pb*tc**(n-1)/g**2),
                            (1/n)*(bttn)**(1/n-1) * (n*tc**(n-1)-n*(n-1)*pb*tc**(n-2)/g), 
                            bttn**(1/n)*(-np.log(tc**n-n*pb*tc**(n-1)/g)/n**2 + \
                                         (np.log(tc)*tc**n - (pb/g)*(tc**(n-1)+n*np.log(tc)*tc**(n-1))) \
                                         /(n*(tc**n-n*pb*tc**(n-1)/g)))]])
                        #print(jac_tb)
                        bt_err = \
                           np.sqrt(np.matmul(jac_tb,
                                             np.matmul(cov,
                                                       np.transpose(jac_tb)))[0][0])
                        s.results[sb][ch]['t_b_est_thermal']['t_bath_err'].append(bt_err)
                        #print([float(f'{val:.3f}') for val in [d['temp'][i],pb,bt,bt_err]])
                    
                    try:
                        (popt, pcov) = curve_fit(s.tbath_line, 
                                                 s.results[sb][ch]['cl_temp'], 
                                                 s.results[sb][ch]['t_b_est_thermal']['t_bath'],
                                                 sigma=s.results[sb][ch]['t_b_est_thermal']['t_bath_err'],
                                                 absolute_sigma=True)
                        s.results[sb][ch]['bath_temp_fit'] = (popt, pcov)
                        s.slopes.append(popt[0])
                    except:
                        print(f"{sb} {ch} t_b_est: {s.results[sb][ch]['t_b_est_thermal']['t_bath']} t_b_est_error:{s.results[sb][ch]['t_b_est_thermal']['t_bath_err']}")
            
                    #for_disp = [float(f'{param:.2f}') for param in popt]
                    #for_disp = [float(f'{temp:.1f}') for temp in s.ramp[sb][ch]['t_b_est_thermal']['t_bath']]
                    #print(f"{for_disp};{ch};{tes_dict[sb][ch]['opt_name']}")
        print(f"!opt-unmasked t_bath_est slopes: average{np.mean(s.slopes)},stdev{np.std(s.slopes)}")
        
        
    # ======== more init functions =========== 
    def init_opt_power(s,device_z, cooldown_name, freq,loc):
        if loc not in s.opt_power.keys():
            s.opt_power[loc] = {}
            beam_name = ''
            # I believe MF/UHF TES are never mixed on a single horn. 
            if 65<=freq and freq<=190: # expecting 90 or 150
                beam_name = 'MF_F'
            elif 190<=freq:
                print("update UHF freqs/beam names! Check center vs. name")
                beam_name = 'UHF_F' # a wild guess
            # this is the time consuming step. Hence doing it by loc.
            s.opt_power[loc]['load_obj'] = spb_opt.SPBOptLd(cooldown_name,
                                        loc[0],loc[1],device_z,
                                        beam_name=beam_name)
        if freq  not in s.opt_power[loc].keys():
            s.opt_power[loc][freq] = {'opt_power':[]}
            for temp in s.temp_list:
                # names of UHF freqs != beam center. May need to update this
                power = s.opt_power[loc]['load_obj'].get_power(T=temp,freq=freq)
                s.opt_power[loc][freq]['opt_power'].append(power)
            (popt, pcov) = curve_fit(s.fit_line, s.temp_list, 
                                     s.opt_power[loc][freq]['opt_power'])
            s.opt_power[loc][freq]['opt_fit'] = (popt,pcov)
                        
    #def bathramp_coldload_together_fits(s, tes_dict, ) 
    
    
    # ======== Fitting functions ======  
    def together_fit_opt_unmasked(s,tb_fit,sb,ch,make_plot=False, own_fig=True):
        # does not update tes_dict (or anything!) with the new values....
        # also need to have coldload power during bath_ramp
        tes_dict = s.tes_dict
        # set up the given t_b est line:
        ma, mb = tb_fit[0],tb_fit[1]
        s.results[sb][ch]['t_b_est']={'t_bath':[],'t_bath_err':[]}
        for temp in s.results[sb][ch]['cl_temp']:
            s.results[sb][ch]['t_b_est']['t_bath'].append(ma*temp +mb )
        # setup s.cur_det and run scipy.optimize.curvefit 
        # setup the function to be optimized
        bath_ramp = s.bath_ramp
        loc = s.results[sb][ch]['loc']
        s.cur_det = {}
        freq = int(tes_dict[sb][ch]['TES_freq'])
        s.cur_det['opt_power'] = s.opt_power[loc][freq]['opt_power']
        s.cur_det['cl_temp'] = s.results[sb][ch]['cl_temp']
        s.cur_det['t_b_cl_ramp'] = s.results[sb][ch]['t_b_est']['t_bath']
        # Should think about what should own this value. Note different for different locs.
        s.cur_det['cl_power_during_bath_ramp'] = s.opt_power[loc][freq]['opt_power_during_bath_ramp']
        # now, setup x and y data:
        x_data = s.results[sb][ch]['cl_temp'] + bath_ramp.ramp[sb][ch]['temp']
        y_data = s.ramp[sb][ch][f'p_b{s.use_per}'] + bath_ramp.ramp[sb][ch][f'p_b{s.use_per}']
        # now, parameter estimates:
        p0 = [bath_ramp.all_results[sb][ch]['G'],
              bath_ramp.all_results[sb][ch]['Tc'],
              bath_ramp.all_results[sb][ch]['n'],
              0.5] # opt_eff
        # here we go:     
        (popt, pcov) = curve_fit(s.coldload_bathramp_opt_unmasked_p_b90_of_param,
                                              x_data,y_data, p0=p0)
        # this should be moved to plotting functions once this saves data. 
        if make_plot:
            if own_fig:
                plt.figure(figsize=default_figsize)
            g, tc, n, opt_eff = popt
            offscreen = s.bath_ramp.temp_list_raw[-1]*1.05 # let you see x-intercept
            together_fit_y = s.coldload_bathramp_opt_unmasked_p_b90_of_param(\
                                    x_data + [offscreen], g, tc, n, opt_eff)
            plt.plot(x_data+[offscreen], together_fit_y,label="together_fit")
            d = bath_ramp.results[sb][ch]
            g,tc,n,opt_eff = d['G'],d['Tc'],d['n'],0
            bathramp_fit_y = s.coldload_bathramp_opt_unmasked_p_b90_of_param(\
                              bath_ramp.ramp[sb][ch]['temp']+[offscreen], g, tc, n, opt_eff)
            plt.plot(bath_ramp.ramp[sb][ch]['temp']+[offscreen],bathramp_fit_y, \
                     label="bathramp_fit", linewidth=1)
            plt.plot(x_data,y_data,marker=".",linestyle="None", label=f"measured p_b{s.use_per}")
            if own_fig:
                plt.title(f"sb{sb} ch{ch} together fit")
                plt.ylim(0,4.0/5*s.p_sat_cut)
                plt.legend()
        return (popt,pcov)
        
    # -------- for scipy.optimize.curvefit() ;
    def tbath_of_dark_TES(s,P_b, g,T_c,n):
        return (T_c**n-P_b*n*T_c**(n-1)/g)**(1/n)

    def tbath_line(s,tcl, slope, constant):
        return slope*tcl + constant
    
    def fit_line(s, x, slope, constant):
        return slope*x + constant
    
    # and here is where I discovered that curve_fit passes x_data as an array.
    def coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp_arr, G, tc, n, opt_eff):
        # first, the function to be fit. 
        # We have to pass some det-specific arguments into this by saving 
        # them as class variables, so curve_fit works
        # these next 3 arrays size all = # of coldload ramp p_b90 points for that det
        # s.cur_det = {'opt_power':[],'cl_temp':[],'t_b_cl_ramp':[]
        #         'cl_power_during_bath_ramp'} 
        # just in case scipy later passes individual values:
        if is_float(temp_arr):
            val = s.point_coldload_bathramp_opt_unmasked_p_b90_of_param(temp_arr, G, tc, n, opt_eff)
            return val
        to_return = []
        for temp in temp_arr:
            val = s.point_coldload_bathramp_opt_unmasked_p_b90_of_param(temp, G, tc, n, opt_eff)
            to_return.append(val)
        return np.array(to_return) 
    
    # -------- fitting methods not directly for scipy.optimize.curvefit()
    def point_coldload_bathramp_opt_unmasked_p_b90_of_param(s,temp, G, tc, n, opt_eff):
        # first, the function to be fit. 
        # We have to pass some det-specific arguments into this by saving 
        # them as class variables, so curve_fit works
        # these next 3 arrays size all = # of coldload ramp p_b90 points for that det
        # s.cur_det = {'opt_power':[],'cl_temp':[],'t_b_cl_ramp':[]
        #         'cl_power_during_bath_ramp'}        
        if temp < 50: # because temp's really in K, the cold load temp
            point_ind = s.cur_det['cl_temp'].index(temp)
            t_b_point = s.cur_det['t_b_cl_ramp'][point_ind]
            p_sat = G/n*(tc-t_b_point**n/tc**(n-1))
            p_opt = s.cur_det['opt_power'][point_ind]
        else: # this temp is a bathramp point
            p_sat = G/n*(tc-temp**n/tc**(n-1))
            p_opt = s.cur_det['cl_power_during_bath_ramp']
        return p_sat - opt_eff*p_opt
    
    # ============ Plotting methods =========
    def plot_est_T_bath(s):
        # possibly could call the Parent class plot ramp by param thing?
        # oh...not quite, because errorbar. 
        tes_dict = s.tes_dict
        plot_array = []
        for sb in s.results.keys():
            plot_array.append((plt.figure(figsize=default_figsize), sb))
            for ch, d in s.results[sb].items():
#                 if 'bath_temp' not in s.ramp[sb][ch].keys():
#                     continue
                if not tes_dict[sb][ch]['opt'] == 1.0:
                    mylabel = f'ch{ch} {tes_dict[sb][ch]["TES_freq"]} {tes_dict[sb][ch]["opt_name"]}'
                    plt.errorbar(d['cl_temp'], d['t_b_est_thermal']['t_bath'], 
                                 yerr=d['t_b_est_thermal']['t_bath_err'],
                                 linestyle=tes_dict[sb][ch]["linestyle"],
                                 label=mylabel)

            #plt.ylim(0, 15)
            plt.ylabel("estimated bath temp [mK]")
            plt.xlabel("cold load temperature [K]")
            plt.title(f"estimated bath temp from bath_ramp and CLramp data, sb{sb}")
            plt.legend()
        return plot_array
    
    def plot_est_T_bath_by_BL(s,tes_dict=True):
        if tes_dict:
            tes_dict = s.tes_dict
        mux_map = s.mux_map
        linestyle,label =  'solid', ''
        #fig, ax = plt.subplots(nrows=4, figsize=(9,9))
        fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(9,7))
        for r in range(len(ax)):
            row = ax[r]
            for c in range(len(row)):
                p = row[c]
                for sb in s.results.keys():
                    for ch, d in s.results[sb].items():
                        #print(mux_map[sb][ch])
                        if mux_map[sb][ch]['biasline'] == 4*r + c:
                            if tes_dict and sb in tes_dict.keys() and ch in tes_dict[sb].keys():
                                linestyle = tes_dict[sb][ch]['linestyle']
                                label = f' {str(tes_dict[sb][ch]["TES_freq"])} {tes_dict[sb][ch]["opt_name"]}'
                            if not tes_dict[sb][ch]['opt'] == 1.0:
                                p.errorbar(d['cl_temp'], d['t_b_est_thermal']['t_bath'], 
                                     yerr=d['t_b_est_thermal']['t_bath_err'],
                                     linestyle=linestyle,
                                     label=f'ch{ch}'+label)
                #p.set_xlabel('bath temp [mK]')
                #p.set_ylabel('90% R_n P_sat [pW]')
                p.set_title(f"Bias line {4*r + c}")
                #p.set_ylim([0, 8])
                p.set_xlim([0.95*s.temp_list[0], s.temp_list[-1] + (0.05*s.temp_list[0])])
                #p.legend(fontsize='small')
        plt.suptitle(f"{s.dName} estimated bath temp [mK] from bath_ramp and CLramp data vs. CL temp [K]")
        plt.tight_layout()
        return (fig, ax)
    
    def plot_T_b_slopes_vs_param(s,param_name):
        # Should really fix this to use test_device.opt_dict.
        tes_dict = s.tes_dict
        series = {'masked 90 opt': ([],[]),
                  'masked 150 opt': ([],[]),
                  'masked dark': ([],[]),
                  'unmasked dark': ([],[])} # param, slopes []
        for sb in s.results.keys():
            for ch, d in s.results[sb].items():
                if is_float(tes_dict[sb][ch]['opt']):
                    s_key = ''
                    if tes_dict[sb][ch]['opt'] == 0.75:
                        s_key = 'unmasked dark'
                    elif tes_dict[sb][ch]['opt'] == 0.25: 
                        s_key = f"masked {str(tes_dict[sb][ch]['TES_freq'])} opt"
                    elif tes_dict[sb][ch]['opt'] == 0: 
                        s_key = 'masked dark'
                    if s_key:
                        series[s_key][0].append(tes_dict[sb][ch][param_name])
                        series[s_key][1].append(s.results[sb][ch]['bath_temp_fit'][0][0])
        p = plt.figure(figsize=default_figsize)     
        for key in series:
            plt.plot(series[key][0], series[key][1], marker=".", linestyle="None", label=key)
        
        #plt.plot(unmasked_param, unmasked_slopes, marker=".", linestyle="None", label="dark fab TESs in unmasked area")
        plt.ylabel("slope of fit line of T_b estimate [mK/K]")
        plt.xlabel(f"{param_name}")
        plt.title(f"{s.dName} T_b_est fit line slopes vs. {param_name}")
        plt.legend()
        return p



    
# ==============================================================================
# -------------------- Ramp_Combination Support Functions ----------------------
# ==============================================================================    
    
# More DRY version of loading! 
def ramp_comb_mega_load(dName,mux_map_fp,other_temps,metadata_arrs,opt_power_fp,
                        ramp_l=[], # If already have the ramps loaded, just use them!
                        masked_biaslines=range(0,8),
                        bl_freq_dict={0: 90, 1: 90, 4: 90, 5: 90, 8: 90, 
                                      9: 90, 2: 150, 3: 150, 6: 150, 
                                      7: 150, 10: 150, 11: 150}, # MV5 bl_freq_dict.
                        mux_chips=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                        bonded_TESs=[],
                        therm_cal=[1,0], input_file_type="pysmurf", norm_correct=True,
                        sc_offset=True,use_cii=False,bin_tod=True,save_raw_ivas=False,
                        redo_cuts=True,opt_values_exclude=[-42,0.75,0.875,1.0], # add coldload args?
                        opt_filetype="reload", metadata_temp_idx=0,
                        use_per=90): # note the FIRST time opt_filetype="kaiwen"
    '''note: coldloads must be IMMEDIATELY after their bathramp in the metadata_arr list
    # bathramp metadata should contain "bathramp" or "bath_ramp" in title, coldload 
    ramp contain "coldload" in title. 
    If you pass a ramp list in ramp_l of (ramp_object, <other_temp(s)>), it will use that instead
    of loading from the metadata_arrs. 
    <other_temp(s)> should be either a scalar representing what temperature the coldload was at 
    for a bath ramp or the bath was at for a coldload temperature, or a list of length equal to 
    the number of temperatures data was taken at in the ramp with the appropriate "other" temperature
    at each of those points.
    '''
    ramp_list = []
    params = ""
    if not norm_correct:
        params = params + " nnc"
    if not sc_offset and not use_cii:
        params = params + " no sc_off"
    if use_cii:
        params = params + " cii"
    dName = dName + params
    if ramp_l:
        ramp_list = ramp_l
    else:
        for i in range(len(other_temps)):
            ot = other_temps[i]
            metadata_arr = metadata_arrs[i]
            if type(metadata_arr) == list:
                t_name = metadata_arr[0]
            else: # string, a reload
                t_name = metadata_arr
            if 'bathramp' in t_name or 'bath_ramp' in t_name:
                rt = 'bath'
            elif 'coldload' in t_name:
                rt = 'coldload'
            else: 
                print(f"{i} can't find bath type in metadata name: {t_name}")
                rt = "UNFOUND"
            if rt == 'bath':
                my_dName = f"{dName} CL{ot}" 
                print(f"====== Doing {my_dName}")
                td = Test_Device(my_dName, 
                                    f"analysis_outputs/{make_filesafe(dName)}/",
                                    mux_map_fp,
                                    masked_biaslines=masked_biaslines,
                                    bl_freq_dict=bl_freq_dict,
                                    mux_chips=mux_chips,
                                    bonded_TESs=bonded_TESs)
                my_ramp = Bath_Ramp(td, 'bath', therm_cal, 
                             metadata_arr,
                             input_file_type=input_file_type, norm_correct=norm_correct,
                             use_per=use_per,
                             sc_offset=sc_offset, use_cii=use_cii, bin_tod=bin_tod, 
                             save_raw_ivas=save_raw_ivas,
                             opt_values_exclude=opt_values_exclude,
                             metadata_temp_idx=metadata_temp_idx) 
            else: # it better be a coldload!
                #my_dName = f"{dName} T_b{ot}"
                td = ramp_list[-1][0].test_device 
                td.dName = td.dName + f" T_b{ot}"
                print(f"====== Doing {td.dName}")
                my_ramp = Coldload_Ramp(td, rt, therm_cal, 
                             metadata_arr,
                             input_file_type=input_file_type, norm_correct=norm_correct,
                             use_per=use_per,
                             sc_offset=sc_offset, use_cii=use_cii, bin_tod=bin_tod, 
                             save_raw_ivas=save_raw_ivas,
                             bath_ramp=ramp_list[-1][0],
                             metadata_temp_idx=metadata_temp_idx) 
            if redo_cuts:
                my_ramp.redo_cuts()
            ramp_list.append((my_ramp,ot))
    # ramp_list constructed. Time to make the combination. 
    final_td = Test_Device(dName, 
                   f"analysis_outputs/{make_filesafe(dName)}/",
                   mux_map_fp,
                   masked_biaslines=masked_biaslines,
                   bl_freq_dict=bl_freq_dict)
    return Ramp_Combination(final_td, ramp_list,
                               opt_power_fp,opt_filetype=opt_filetype,use_per=use_per)    
    
    
#2345678901234567890123456789012345678901234567890123456789012345678901234567890  <--80 chars
#234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890 <- 100 chars
class Ramp_Combination:
    '''
    This class serves as a data container for multiple bath or cold load ramps 
    of the same detectors with the same mux mapping. It investigates p_b90s as 
    a function of two variables, bath and cold load temperature,
    rather than one. 
    
    It also uses a more Axis-Manager compatible data storage structure.
    
    # =============================== VARIABLES ===============================
    # ---- Init argument vals (see also ramp_comb_mega_loader above!!!)
    test_device     # Test_Device object of this array/SPB. 
                    # Should be a clean one, not any of the subarrays' ones.
    ramp_l          # List of (Temp_Ramp, other_constant_temperature) pairs.
                    # ex a Bath_Ramp paired with the constant temperature 
                    # of the cold load during that bath ramp.
    opt_filename    # filename of Kaiwen-style optical power dictionary
                    # OR string filename of previously computed
                    # opt_power_axis (see below). 
    opt_filetype    # "kaiwen" or "reload"
    use_per         # what percent R_n to use for "p_b90"s **TODO** update labelling!
    # ---- Self organization
    axes            # {'axis_name':axis}; 'ramps', 'opt_powers','crv'
    # ---- pulled from init arguments data
    # -- from test_device
    dName           # device name and usually cooldown name. 
    mux_map         # s.test_device.mux_map reference
    bl_freq_dict    # {biasline:TES_freq}. Overrides the mux map. 
    masked_biaslines# array of bias line numbers
    # -- from ramp_l
    r_sh            # the shunt resistance.
    key_info        # {<key_name, as used in axes>:{
                    #      'name': <full name for display>, 
                    #      'ax_name': <name of primary axis it appears in>
                    #      'type': <discrete 'd', continuous 'c', or array continuous 'ac'>
                    #      'units':<units of values; often '' for discrete>
                    #      # if discrete:
                    #      'vals': np.sort()ed ndarray of all unique possible associated values
                    #      'colors': (num_vals,3)-shape ndarray of val-associated RGB colors
                    #      # else, if continuous:
                    #      'extremes': [<lowest value>,<highest value>]
                    #      'lim':[<lowest val to display>,<highest val to display>]
                    #      # note: right now lim=extremes, but did some work on changing,
                    #      can manually set it.
    num_curves      # number of curves in all the ramps
    num_sbchs       # total unique sbch pairs found (or started in mux_map)
    sbchs           # All unique [(sb,ch)] in mux_map after fully loading that.
    tes_sbchs       # All [(sb,ch)] that a bathramp successfully thermal fit
    ramps           # dictionary of key:numpy array of length=num ramps
        # ramps's keys & contents of an index in that key's array:
        ramp, ramp_name, color, idx   # color for plotting purposes 
        bath_temps  # array of bath temperature at each point in ramp curve
        cl_temps    # array of coldload temperature at each point in ramp curve
    # -- from optical_powers
    opt_powers      # dictionary of key:numpy array of length=#exposed TES
        # opt_power's keys & contents of an index in that key's array:
        # a) identifiers
        sb          # smurf_band, if was found in this cooldown; else np.nan.
        ch          # smurf_channel, if was found in this cooldown; else np.nan.
        sb_ch       # sb*1000 + ch. Unique channel specifier.
        # These 5 uniquely specify a physical TES, I believe.
        det_x, det_y, TES_freq, pol, rhomb
        # Kaiwen stuff
        det_row    # I think det_row & det_column specify the hexagon 
        det_col    # of three pixels the TES is in, I think
        # Kaiwen's original key is some unintuitive (to me, anyway) 
        # combination of frequency + det_x and det_y I think. pol A goes first.
        # edit: Kaiwen says it's random. 
        kaiwen_key1, kaiwen_key2
        idx        # indexes. Organizational thing.
        # b) Data!! 
        # Kaiwen's opt powers are (as of 10/22/2022) perfect lines.
        p_opt_s    # slope of the optical power line for this tes
        p_opt_c    # y-intercept of the (cl_temp,p_opt) <x,y> line
        p_opt_err_s# slope of the p_opt_err line for this tes
        p_opt_err_c# y-intercept of the (cl_temp,p_opt_err) <x,y> line 
        
    # ---- Master Loaded Data structures    
    crv             # curves. dictionary of key:array, with array length = number
                    # of total data points. Each data point represents
                    # one "iv curve" taken (note: may not show an 
                    # actual iv curve.)
    # crv's keys should all be either:
    #    a) data/data descriptors unique to that data point's "iv"
    #    b) necessary to calculate iv parameters,
    #    OR
    #    c) something I really want to sort ivs by. 
    # crv's keys & contents of an index in that key's array:
    # a)-type (data of that point's unique "iv") 
        # -- arrays of [the following] at each point in the curve:
        v_bias        # Voltage bias (whole bias line&DR), [low-current mode Volts]
        i_bias        # commanded bias_current [μA]
        v_tes         # TES voltage aka R_sh voltage
        i_tes         # Current through TES [μA] (relies on ITO being correct!)
        R             # TES resistance at each point in curve.
        p_tes         # Electrical power on TES
        # -- others:
        bl_powered    # the bias line being powered when this curve was taken.
        i_tes_offset  # I_TES Offset. Add to i_tes for raw response (if loaded in
                      # a fashion that retains this info). formerly 'c_norm'
        p_trans       # pysmurf style median p_tes between sc_idx and nb_idx
        p_b90         # p_tes at R=90% R_n
        R_nSM         # resistance fitted to upper half of believed 
                      # normal branch, I think.
        all_fit_R_n   # Resistance from fitting whole (plausibly normal) curve as straight line
        all_fit_ITO   # I_TES_Offset from fitting whole (plausibly normal) curve as straight line
        all_fit_stats # {r,p,std_err,r_log} from fitting whole (plausibly normal) curve as straight line
        sc_idx        # index of end of superconducting branch
        nb_idx        # index of beginning of normal branch
    # b)-type (necessary to calculate iv parameters)
        R_n           # best normal resistance estimate for this detector.
                      # here because of its use in obtaining ITO.
        p_opt         # Optical power on detector's location (as if unmasked)
        p_opt_err     # error on the above. 
    # b)/c) hybrids (necessary for calculations, possibly sort by)
        bath_temp     # bath temperature [mK]
        cl_temp       # cold load temperature [K]
    # c)-type (sorting/reference keys) arrays
        # -- background
        sb            # smurf_band. Not combining with smurf channel b/c... 
        ch            # smurf_channel. ...sodetlib/iv.py precedent.
        sb_ch         # sb*1000+ch
        bl            # detector biasline
        det_x, det_y, det_r  # [mm] pixel <>-location relative to wafer (not DR)
        pol, TES_freq      
        OMT           # no OMT ('N'), has OMT ('Y'), or unknown '?'
        masked        # masked ('N'),unmasked [receiving light] ('Y'), or ('-') 
        ramp_name     # NAME of the ramp it's from
        ramp_from     # pointer to the ramp object it's from. 
        iva           # its analysis object in its ramp. Useful for 
                      # many Temp_Ramp functions.
                      # However, I probably won't update it.
        # -- classifications & annotations
        is_iv         # **TODO**
        is_iv_info    # **TODO**
        is_normal     # **TODO**
    # for code use
        idx           # indexes. 
    # **TODO**: Consider doing axis-manager style detector info storage 
    # instead of tes_dict. Yeah... probably should...
    tes               # dictionary of key:numpy array of 
                      # length=num TES's at least one bathramp thermal-fit
        # tes's keys & contents of an index in that key's array:
        sb, ch, sb_ch, bl, det_x, det_y, det_r, pol, TES_freq, OMT, masked, 
        R_n, was_fit, G, G_err, Tc, Tc_err, n, n_err, 
        opt_eff_no_subtraction, opt_eff_no_subtraction_err, 
        cov_G_Tc, cov_G_n, cov_G_eff, cov_Tc_n, cov_Tc_eff, 
        cov_n_eff, p_sat100mK, p_sat100mK_err, opt_eff
        # if above float not calcable for particular TES, = np.nan or -42.0
        was_fit       # was able to fit opt_eff_no_subtraction
        G             # fit val: Thermal Conductance of det island 
        G_err         # sqrt(variance of G) in thermal fit
        Tc            # fit val: TES critical temperature
        Tc_err        # sqrt(variance of Tc) in thermal fit
        n             # fit val: power law index (1+thermal conductance exponent)
        n_err         # sqrt(variance of n) in thermal fit
        opt_eff_no_subtraction # eff in fit of p_b90 = G/n *(tc-tb**n/tc**(n-1)) - eff*p_opt
        opt_eff_no_subtraction_err # sqrt(variance of eff) in thermal fit
        # below: cov_A_B = covariance of A and B in thermal fit
        cov_G_Tc, cov_G_n cov_G_eff, cov_Tc_n, cov_Tc_eff, cov_n_eff, 
        p_sat100mK    # calculated p_b90@Tb=100mK, no optical power
        p_sat100mK_err# using jacobian from thermal fits
        opt_eff       # optical efficiency with dark subtraction

    # ======================= METHODS (Helpers indented) =======================
    # ---------- Initialization and loading methods
    # NOTE: see the non-class ramp_comb_mega_load() function in same file.
    # Described above the actual init here, it's basically an init that does 
    # the ramps first before combining.
    # NOTE v: CL_ramp metadata must be RIGHT after their bathramp in metadata_arrs!
    ramp_comb_mega_load(dName,mux_map_fp,other_temps,metadata_arrs,opt_power_fp,
                        masked_biaslines=range(0,8),
                        bl_freq_dict={0: 90, 1: 90, 4: 90, 5: 90, 8: 90, 
                                      9: 90, 2: 150, 3: 150, 6: 150, 
                                      7: 150, 10: 150, 11: 150}, # MV5 bl_freq_dict.
                        therm_cal=[1,0], input_file_type="pysmurf", norm_correct=True,
                        sc_offset=True,use_cii=False,bin_tod=True,save_raw_ivas=False,
                        opt_values_exclude=[-42,0.75,0.875,1.0], # coldload args?
                        opt_filetype="reload"): 
    __init__(s, test_device, ramp_l, opt_filename,opt_filetype="kaiwen")
        count_curves_and_sbch(s)
        consolidate_mux_maps(s)
        create_opt_power_axis(s, kaiwen_opt_dict)
            find_sbch(s,tes_opt_power_dict)
        load_curves_axis(s)
        count_TESs(s)
        init_TES_axis(s)
        no_dark_subtraction_opt_eff(s,debug=0)
            attempt_no_dark_subtraction_opt_eff(s,i,debug=0,num_fit=0,all_bad_idxs=[])
                p_b90_of_bath_temp_and_p_opt(s,tb_and_p_opt,G,tc,n,eff)
                p_b90_of_bath_temp(s,tb,G,tc,n):
        setup_key_info(s)

    lim_finder(s,vals) # Maybe use in setup_key_info someday
        
    # ---------- Data Finders
    find_idx_matches(s,match_list,dict_return=False,ax_name='crv')
        apply_match(s,ax,idx_list,match)
            match_mask(s,ax,idx_list,match)
                single_match_mask(s,ax,idx_list,match)
        add_dict_level(s,ax,idx_list,level_list,d)

    # ====================== EXTERNAL Analysis FUNCTIONS ======================
    ax_line_as_dict(ax,idx):
    is_unknown(val)
    replace_rc_thermal_with_bath_ramp_results(s,ramp)
    
    # ---------- Plotting Methods
    ton_of_sbchs(s,x_key,y_key, start=0,stop='all',match_list=[],
                 exclude_unknowns=False,
                 x_lim=[],y_lim=[],plot_args={'linewidth':0.5})
    plot_by_ramp(s,x_key,y_key,match_list=[],x_lim=[],y_lim=[],
                 prefix='',plot_args={},own_fig=True)
    plot_key_v_key(s,x_key,y_key,match_list=[],x_lim=[],y_lim=[],
                   prefix='',plot_args={},ax_name='crv',own_fig=True)
    plot_key_v_key_grouped_by_key(s,x_key,y_key,by_key,match_list=[],exclude_unknowns=False,
                                      ax_name='crv',x_lim=[],y_lim=[], prefix='',plot_args={},
                                      own_fig=True,legend='default')
    plot_key_v_key_colored_by_key(s,x_key,y_key,by_key,match_list=[],exclude_unknowns=False,
                                  ax_name='crv',x_lim=[],y_lim=[], prefix='',title_override = '',
                                  plot_args={'marker':'.'},own_fig=True, v_lim=[],
                                  color_scale=plt.cm.inferno, outlier_colors=['blue','green'],
                                  xy_overlap_offset=1)
    '''
    
    def __init__(s, test_device, ramp_l, opt_filename,opt_filetype="kaiwen",use_per=90):
        # SEE ALSO the ramp_comb_mega_load() function above this class in this
        # file, which is basically an external init that does all the 
        # ramps first before combining. 
        s.test_device = test_device
        s.ramp_l = ramp_l
        s.opt_filename = opt_filename
        s.use_per = use_per
        s.p_bp = f"p_b{s.use_per}"

        # make ramp_ax 
        tab20 = plt.get_cmap('tab20')
        bl_colors = tab20(np.linspace(0, 1.0, 14))
        s.ramps = {'ramp':np.array([ramp for ramp,ot in s.ramp_l]),
                     'bath_temps':np.full((len(ramp_l),),[-42],dtype=object), 
                     'cl_temps':np.full((len(ramp_l),),[-42],dtype=object),
                     'ramp_name':np.full((len(ramp_l),),"-42",dtype=object),
                     'ramp_type':np.full((len(ramp_l),), "-42", dtype=object), # bath or cold
                     'color':np.full((len(ramp_l),), "-42", dtype=object), # filled post-key_info
                     'idx':np.arange(len(s.ramp_l))}
        for i in range(len(s.ramp_l)):
            o_temp = ramp_l[i][1]
            ramp = s.ramps['ramp'][i]  
            s.ramps['ramp_type'][i] = ramp.ramp_type
            if ramp.ramp_type == 'bath':
                prim = 'bath_temps'
                sec = 'cl_temps'
                s.r_sh = ramp.r_sh # should all be the same anyway, same device and all. 
            else:
                prim = 'cl_temps'
                sec = 'bath_temps'
            s.ramps[prim][i] = np.array(ramp.temp_list_raw)
            assert is_float(o_temp) or len(o_temp)  == len(ramp.temp_list_raw) , \
                  f"r{i}: {o_temp} other_temps in ramp_l should be float or list of temps of length= len(ramp.temp_list_raw)"
            if is_float(o_temp): 
                s.ramps[sec][i] = np.array([o_temp]*len(ramp.temp_list_raw))
                ot_str = str(o_temp)
            elif len(o_temp)  == len(ramp.temp_list_raw):
                s.ramps[sec][i] = np.array(o_temp)
                ot_str = f"{min(o_temp)}-{max(o_temp)}"                
            s.ramps['ramp_name'][i] = f"r{i}:{ramp.ramp_type} OT{ot_str}"
            ramp.ramp_name = s.ramps['ramp_name'][i]
            ramp.other_temps = s.ramps[sec][i]
            s.ramps['other_temps'] = o_temp # it's just too convenient.
               
        
        # pull some device/cooldown reference info from test_device
        # that I think I might want smaller names to reference to. 
        s.mux_map = s.test_device.mux_map
        s.dName = s.test_device.dName
        s.bl_freq_dict = s.test_device.bl_freq_dict
        s.masked_biaslines = s.test_device.masked_biaslines
        
        
        
        # CONSOLIDATE ALL MUX MAP KNOWLEDGE (aka corrections discovered):
        # NEEDS TO HAPPEN before the main axes loaded.
        s.count_curves_and_sbch()
        s.consolidate_mux_maps()
        
        # ensure analysis directory exists (for optical powers)
        Path(s.test_device.out_path).mkdir(parents=True, exist_ok=True)
        
        # ------ get optical powers set up -----
        if opt_filetype=="reload":
            s.opt_powers = np.load(opt_filename,allow_pickle=True).item()
        elif opt_filetype=="kaiwen":
            s.opt_powers=s.create_opt_power_axis(np.load(opt_filename,
                                                          allow_pickle=True).item())
        else:
            print("UNSUPPORTED optical powers filetype!")
            s.opt_powers == "UNSUPPORTED optical powers filetype!"
            
        # ------ We're ready. COMBINE THE RAMPS! -------------
        s.load_curves_axis()
            
        # ------ Setup for the actual together-fits. -------
        # organization, for find_idxs to pull from.
        s.axes = {'ramps':s.ramps, 'opt_powers':s.opt_powers,
                  'crv':s.crv} # we'll add s.tes
        s.count_TESs() # POssibly move into init_TES_axis and make it a helper function
        s.init_TES_axis() # setup s.tes
        # ------ No-dark-subtraction opt eff, once moved out of stuff -------
        s.no_dark_subtraction_opt_eff()
        # ------ s.setup_key_info() goes here once all the calculations are done-------
        s.setup_key_info()
        
        # Okay, slightly hacky, but useful for some plotting:
        s.ramps['color'] = s.key_info['ramp_name']['colors'] #useful for plotting, it turns out
        
        
    def setup_key_info(s):
        #**TODO**!!!
        # pull some from the ramps.
        #s.key_info = s.ramp_l[0][0].key_info
        # here we go. discrete/continuous,
        # key_info setup
        # discrete or continuous affects whether it makes limits or assigns colors
        # key name:  (ramp_to_ref OR key to copy colors/lim from, discrete (d)/continuous (c)/array continuous (ac), units, full name)
        kis = {'ramp_name'    :('ramps','d','','Ramp Name'),
               'ramp_type'    :('ramps','d','','Type of Ramp'),
               'p_opt_s'      :('opt_powers','c','pW/K','Slope of optical_power(cold-load temp)'),
               'p_opt_err_s'  :('opt_powers','c','pW/K','Slope of optical_power_err(cold-load temp)'),
               'p_opt_c'      :('opt_powers','c','pW','y-intercept of optical_power(cold-load temp)'),
               'p_opt_err_c'  :('opt_powers','c','pW','y-intercept of optical_power_err(cold-load temp)'),
               'sb'           :('crv','d','','SMuRF band'),
               'ch'           :('crv','c','','SMuRF channel'),
               'sb_ch'        :('crv','c','','SMuRF band*1000 + SMuRF channel'),
               'bl'           :('tes','d','', 'Biasline'),
               'det_x'        :('crv','c','mm', 'detector X-pos on UFM'),
               'det_y'        :('crv','c','mm', 'detector Y-pos on UFM'),
               'det_r'        :('crv','c','mm', 'detector distance from UFM center'),
               'pol'          :('crv','d','','Polarity'),
               'TES_freq'     :('crv','d','GHz','TES Coupled Optical Frequency'),
               'OMT'          :('crv','d','','Has OMT (OrthoMode Transducer'),
               'masked'       :('crv','d','','Masked'),
               'masked_freq'  :('tes','d','','string combination of unmasked/freq status'),
               'is_iv'        :('crv','d','','Is curve an IV?'),
               'is_iv_info'   :('crv','d','','iv-ness categorization'),
               'cut_info'     :('crv','d','','info on reason cut if it was'),
               'is_normal'    :('crv','d','','Is curve a normal detector?'),
               'bath_temp'    :('crv','d','mK','Bath Temperature'), # formerly continuous.
               'cl_temp'      :('crv','d','K','Coldload Temperature'),
               'R_n'          :('crv','c','Ω',"$R_n$, detector's normal resistance"),
               'p_opt'        :('crv','c','pW',"$P_{opt}$, Optical power on detector's location (as if unmasked)"),
               'p_opt_err'    :('crv','c','pW',""),
               'bl_powered'   :('crv','d','',"Biasline powered when taking curve"),
               'i_tes_offset' :('crv','c','≈μA',"Offset in the TES current readout"),
               'p_trans'      :('crv','c','pW',"pysmurf style median $P_{tes} between sc_idx and nb_idx"),
               f'p_b{s.use_per}':('crv','c','pW',"$P_{b"+f"{s.use_per}"+"}$,"+r" the $P_{bias}$ at " + f"R={s.use_per}%"+ r"$R_{n}$"),
               'R_nSM'        :('crv','c','Ω','resistance fitted to upper half of believed normal branch'), 
               'all_fit_R_n'  :('crv','c','Ω','Resistance from fitting whole curve as straight line'), #straight-line fit of whole (plausibly normal) curve
               'all_fit_ITO'  :('crv','c','≈μA','I_TES_Offset from fitting whole curve as straight line'),
               'all_fit_stats':('crv','c','','{r,p,std_err,r_log} from fitting whole curve as straight line'),
               'sc_idx'       :('crv','c','','index of end of superconducting branch'),
               'nb_idx'       :('crv','c','','index of beginning of normal branch'),
               "v_bias"       :("crv","ca","V","Voltage bias (whole bias line&DR), [low-current mode Volts]"),
               "i_bias"       :("crv","ca","μA","commanded bias_current"),
               "v_tes"        :("crv","ca","μV","TES voltage aka $R_{sh}$ voltage"),
               "i_tes"        :("crv","ca","μA","Current through TES (relies on ITO being correct!)"),
               "R"            :("crv","ca","Ω","TES resistance at each point in curve."),
               "p_tes"        :("crv","ca","pW","Electrical power on TES"),
               "was_fit"      :("tes","d","","was able to fit opt_eff_no_subtraction"),
               "G"            :("tes","c","pW/mK","$G$, Differential Thermal Conductance "),
               "G_err"        :("tes","c","pW/mK","sqrt(variance of G) in thermal fit"),
               "Tc"           :("tes","c","mK",r"$T_c$, TES critical temperature"),
               "Tc_err"       :("tes","c","mK","sqrt(variance of Tc) in thermal fit"),
               "n"            :("tes","c","","$n$, power law index (1+thermal conductance exponent)"),
               "n_err"        :("tes","c","","sqrt(variance of n) in thermal fit"),
               "opt_eff_no_subtraction":("tes","c","",r"$\eta$"+f"in fit of p_b{s.use_per} = G/n *(tc-tb**n/tc**(n-1)) -" + r"$\eta$*p_opt"),
               "opt_eff_no_subtraction_err":("tes","c","",r"sqrt(variance of $\eta$) in thermal fit"),
               "cov_G_Tc"     :("tes","c","pW","covariance between $G$ and $T_c$ in thermal fits"),
               "cov_G_n"      :("tes","c","pW/mK","covariance between $G$ and $n$ in thermal fits"),
               "cov_G_eff"    :("tes","c","pW/mK",r"covariance between $G$ and $\eta$  in thermal fits"),
               "cov_Tc_n"     :("tes","c","mK","covariance between $T_c$ and $n$ in thermal fits"),
               "cov_Tc_eff"   :("tes","c","mK",r"covariance between $T_c$ and $\eta$ in thermal fits"),
               "cov_n_eff"    :("tes","c","",r"covariance between $n$ and $\eta$  in thermal fits"),
               "p_sat100mK"   :("tes","c","pW",'calculated $P_{b'+ f'{s.use_per}'+'}$@$T_b$=100mK, no optical power'),
               "p_sat100mK_err":("tes","c","pW","using jacobian from thermal fits")
               #"opt_eff"      :("tes","c","","optical efficiency with dark subtraction"),
              }
        s.key_info = {}
        s.t20 = plt.cm.tab20(np.linspace(0, 1.0, 19+1))
        s.t10 = plt.cm.tab10(np.linspace(0,1.0,9+1))
        for key, setup in kis.items():
            ax_name, dca, units, name = setup
            # moved into separate helper function to make it easier to 
            # add keys after setup, as I often find myself wanting to do.
            # probably slower, but not ridiculously so
            s.update_key_info(key,name,ax_name,dca,units)
        print("Reminder: lim_finder(vals) can get decent no-outlier limits estimates.")
    
    def update_key_info(s,key,name,ax_name,dca,units):
        '''Adds or updates key info
        s = Ramp_Combination; ca =continuous of arrays
        key_info = {<key_name, as used in axes>:{
            #      'name': <full name for display>, 
            #      'ax_name': <name of primary axis it appears in>
            #      'type': <discrete 'd', continuous 'c', or continuous arrays 'ca'>
            #      'units':<units of values; often '' for discrete>
            #      # if discrete:
            #      'vals': np.sort()ed ndarray of all unique possible associated values
            #      'colors': (num_vals,3)-shape ndarray of val-associated RGB colors
            #      # else, if continuous:
            #      'extremes': [<lowest value>,<highest value>]
            #      'lim':[<lowest val to display>,<highest val to display>]
            #      # note: right now lim=extremes, but did some work on changing,
            #      can manually set it.}
        }'''
        axis = s.axes[ax_name]
        return pu.dax.update_key_info(s.key_info,key,name,axis,dca,units,
                                  more_info={'ax_name':ax_name})
#         if 'c' in dca:
#             # now, extremes, limits
#             if 'a' in dca:
#                 vals=[]
#                 v_min, v_max = np.inf,-np.inf
#                 arrs = axis[key][s.find_idx_matches([],exclude_unknowns=[key],ax_name=ax_name)]
#                 try:
#                     for arr in arrs:
#                         if min(arr) < v_min:
#                             v_min = min(arr)
#                         if max(arr) > v_max:
#                             v_max = max(arr)
#                     vals=[v_min,v_max]
#                 except BaseException as err:
#                     print(f"{key} key_info err:{err}")
#             else:
#                 vals = axis[key][s.find_idx_matches([],exclude_unknowns=[key],ax_name=ax_name)]
#             try:
#                 kd['extremes'] = [min(vals),max(vals)]
#                 kd['lim'] = kd['extremes'] # for now....maybe use lim finder later
#             except BaseException as err:
#                 print(f"{key} key_info err:{err}")
#         else: # discrete. 
#             vals = axis[key][s.find_idx_matches([],exclude_unknowns=[key],ax_name=ax_name)]
#             # Dynamic allocation is crazy slow, don't do that.
#             # not sure why, but my attempt at avoiding it is taking much longer, so back to that.
#             #unique_vals = np.full((len(vals),),-42,dtype=object)  # Handles everything pretty well
#             unique_vals = []
#             j = 0
#             for val in vals:
#                 if not val in unique_vals: # this handles 
#                     unique_vals.append(val)
#                     j += 1
#             try:
#                 #unique_vals=np.sort(np.array(unique_vals[unique_vals != -42])) 
#                 unique_vals=np.sort(np.array(unique_vals)) 
#             except TypeError as err:
#                 print(f"couldn't sort {key}: {err}")
#             kd['vals'] = unique_vals

#             if len(unique_vals) <= 10:
#                 kd['colors'] = np.array([s.t10[j,0:3] for j in range(len(unique_vals))])
#             elif len(unique_vals) <=20:
#                 kd['colors'] = np.array([s.t20[j,0:3] for j in range(len(unique_vals))])
#             else:
#                 print(f"20+ discrete variable?!? {key}")
#         s.key_info[key] = kd
#         return "added"

    def lim_finder(s,vals):
        ''' Someday maybe I'll use this...'''
        vals = np.sort(np.unique(vals))
        #axis = s.axes[ax_name]
        #vals = np.sort(np.unique(axis[key][s.find_idx_matches([],exclude_unknowns=[key],ax_name='tes')]))
        diffs = np.diff(vals)
        #print(vals)

        b_i, t_i = 0, len(vals)-1
        n_b_i, n_t_i = b_i, t_i
        bot, top = vals[b_i],vals[t_i]
        n_bot, n_top = bot, top
        point_dist = 0.0008 # median diff at least (that*100) % of screen width 0.001
        rel_point_dist = 0.01
        min_points_shown = 10
        while (n_t_i-n_b_i > min_points_shown) \
              and  np.average(diffs[n_b_i:n_t_i]) \
                   < (n_top-n_bot) * point_dist:
            #rel_point_dist*max(diffs[n_b_i:n_t_i]):
            #and np.median(diffs[n_b_i:n_t_i])/(n_top-n_bot) > point_dist:
            b_i, t_i, bot, top = n_b_i, n_t_i, n_bot, n_top
            #print(n_bot,n_top,np.median(diffs[n_b_i:n_t_i]), rel_point_dist*max(diffs[n_b_i:n_t_i]))
            if diffs[b_i] > diffs[t_i-1]:
                n_b_i = b_i + 1
                n_t_i = t_i
            else:
                n_b_i = b_i
                n_t_i = t_i -1
            n_bot,n_top = vals[n_b_i],vals[n_t_i]
        edge = (top-bot)*0.05
        return [bot-edge,top+edge]  

    # ============== Functions to load data ========== 
    def count_curves_and_sbch(s):
        # Dynamic allocation would really slow the curve loading, so count # curves.
        nc = 0 # number of curves
        for (ramp, other_temp) in s.ramp_l:
            for metadata in ramp.iv_analyzed_info_arr:
                for run_d in metadata:
                    for sb,chs in run_d['iv_analyzed'].items():
                        if sb == "high_current_mode":
                            continue
                        nc += len(chs.keys())
                        for ch in chs.keys():
                            s.test_device.check_mux_map_for_channel_and_init(sb, ch)
        s.num_curves = nc
        # all possible sbch have been found and added to mux
        s.num_sbchs = 0 # (this is used to upperbound tes axis size).
        s.sbchs = []
        for sb, chs in s.mux_map.items():
            s.num_sbchs += len(chs.keys())
            for ch in chs:
                s.sbchs.append((sb,ch)) # I ended up wanting this list too much. 
        return nc
    
    def consolidate_mux_maps(s):
        # consolidates all the info updates in ramps' mux maps
        # and uses bl_freq_dict and masked_biaslines to assign 
        # everything as correctly as we know. 
        # count_curves_and_sbch has already ensured every sbch is in our mux map. 
        # ...but probably not with the correct bl. 
        # **TODO**: MAKE A SMARTER way of assigning R_n!!
        for sb in s.mux_map.keys():
            for ch in s.mux_map[sb].keys():
                bls_standalone = []
                r_n_selected = False
                for ramp, other_temp in s.ramp_l:
                    # coldloads use the same test_device as their bathramp! 
                    # Don't double-count!
                    if ramp.ramp_type != "coldload": 
                        r_bs = ramp.test_device.bls_standalone
                        if sb in r_bs.keys() and ch in r_bs[sb].keys():
                            bls_standalone += r_bs[sb][ch]  
                            rtd = ramp.tes_dict
                            if not r_n_selected \
                               and sb in rtd.keys() and ch in rtd[sb].keys() \
                               and 'R_n' in rtd[sb][ch].keys():
                                if not sb in s.test_device.tes_dict.keys():
                                    s.test_device.tes_dict[sb] = {}
                                # THe below DOES trigger,  for mv12 on 415   
                                if not ch in s.test_device.tes_dict[sb].keys():
                                    s.test_device.tes_dict[sb][ch]={}
                                # This stuff might get overwritten?    
                                s.test_device.tes_dict[sb][ch]['R_n'] = rtd[sb][ch]['R_n']
                                r_n_selected=True
                    if len(bls_standalone) > 0:
                        if sb not in s.test_device.bls_standalone.keys():
                            s.test_device.bls_standalone[sb] = {}
                        s.test_device.bls_standalone[sb][ch] = bls_standalone
        s.test_device.calculate_bls()
    
    # ----- optical powers ---------
    def create_opt_power_axis(s, kaiwen_opt_dict):
        # run if powers_type = Kaiwen. Saves the results and prints 
        # where to obtain in the future with reload. 
        mux_map = s.mux_map
        kod = kaiwen_opt_dict
        num_TES = len(kod.keys())
        # figure out how long the arrays need to be, avoid resizing the array memory allocation
        num_points = 0 # num_TES
        for key, d in kod.items():
            num_points += len(d['T'])
        # mapping names prep
        det_keymap = {'det_row':'det_row','det_col':'det_col', 'rhomb':'rhomb','TES_freq':'freq','pol':'pol','det_x':'x','det_y':'y'}
        for key,val in kod.items():
            if 'det_x' in val['det'].keys():
                det_keymap['det_x']='det_x'
                det_keymap['det_y'] ='det_y'
                break
            break
        data_keymap = {'cl_temp':'T','p_opt':'p_opt','p_opt_err':'p_opt_err'}
        data_my_map = ['p_opt_s','p_opt_c','p_opt_err_s','p_opt_err_c']
        # initialize the axis!     
        opt_powers = {'sb':np.full((num_points, ), -42,dtype=int),
                      'ch':np.full((num_points, ), -42,dtype=int),
                      'sb_ch':np.full((num_points, ), -42,dtype=int),
                      'is_linear':np.full((num_points, ), True,dtype=bool)}
        for key in det_keymap.keys():
            if key == 'pol' or key=='rhomb':
                opt_powers[key] = np.full((num_points, ), 'Z')
            elif key=='det_x' or key=='det_y':
                opt_powers[key] = np.full((num_points, ), np.nan) # defaults to float
            else: #np.nan is float, can cast to int this way: produces massive nonsense numbers
                opt_powers[key] = np.full((num_points, ), np.nan,dtype=int)
        for key in data_keymap.keys():
            opt_powers[key] = np.full((num_points, ), np.nan)  # defaults to float  
        for name in data_my_map:
            opt_powers[name] = np.full((num_points, ), np.nan)  # defaults to float  
        # I just wanted the below in the bottom of the iterable list
        opt_powers['kaiwen_key1'] = np.full((num_points, ), np.nan,dtype=int)
        opt_powers['kaiwen_key2'] = np.full((num_points, ), np.nan,dtype=int)
        opt_powers['idx']         = np.arange(num_points)

        op = opt_powers # just alias to speed this up a bit
        # Start filling it up. 
        i = 0 # index we're on
        for (kkey1,kkey2), d in kod.items():
            num_temps = len(d['T'])
            sb, ch = s.find_sbch(d)
            op['sb'][i:i+num_temps] = sb
            op['ch'][i:i+num_temps] = ch
            op['sb_ch'][i:i+num_temps] = sb*1000+ch
            op['kaiwen_key1'][i:i+num_temps] = kkey1
            op['kaiwen_key2'][i:i+num_temps] = kkey2
            for key,kkey in det_keymap.items():
                op[key][i:i+num_temps] = d['det'][kkey]
            # now, convert data to pW, then do slope-intercept
            # because they are exact lines.
            cl_temps = d['T']
            p_opts = d['p_opt']
            p_opt_errs = d['p_opt_err']
            cl_dif = d['T'][-1]-d['T'][0]
            op['p_opt_s'][i:i+num_temps] = 1e12*(d['p_opt'][-1]-d['p_opt'][0])/cl_dif
            op['p_opt_err_s'][i:i+num_temps] = 1e12*(d['p_opt_err'][-1]-d['p_opt_err'][0])/cl_dif
            op['p_opt_c'][i:i+num_temps] = 1e12*d['p_opt'][0] - d['T'][0]*op['p_opt_s'][i]
            op['p_opt_err_c'][i:i+num_temps] = 1e12*d['p_opt_err'][0] - d['T'][0]*op['p_opt_err_s'][i]
            for j in range(len(p_opts)):
                diffy = p_opts[j] - (cl_temps[j]*op['p_opt_s'][i]+op['p_opt_c'][i])
                if not diffy == 0 :
                    #print(f"non-linear! idx{i} sb{sb} ch{ch} diffy {diffy}")
                    op['is_linear'][i:i+num_temps] = False
                    break
            for key, kkey in data_keymap.items():
                op[key][i:i+num_temps] = d[kkey]
                
            #convert to picoWatts
            for j in range(i,i+num_temps):
                op['p_opt'][j] = op['p_opt'][j] *1e12
                op['p_opt_err'][j] = op['p_opt_err'][j]*1e12
            i += num_temps

        filename = os.path.join(s.test_device.out_path,f"{make_filesafe(s.dName)}_opt_powers.npy")
        np.save(filename,opt_powers)
        print(f"saved opt_powers to: {filename}")
        return op
    
    
    def find_sbch(s,tes_opt_power_dict):
        # create_opt_power_axis helper function
        ktopd = tes_opt_power_dict # kaiwen tes_opt_power_dict
        kdet = ktopd['det']
        kp=''
        if 'det_x' in kdet.keys():
            kp = 'det_'
        for sb in s.mux_map.keys():
            for ch, d in s.mux_map[sb].items():
                # machine precision can cause problems. why mux has machine precision issues, not clear to me.
                # the det values are stored to 1e-6 order of magnitude. 
                if abs(kdet[kp+'x'] - d['det_x']) <=1e-12 \
                   and abs(kdet[kp+'y'] - d['det_y']) <= 1e-12 \
                   and kdet['pol'] == d['pol'] and try_to_numerical(kdet['freq']) == d['TES_freq']:
                    return (sb, ch)
        return (-42, -42)  # smurf didn't find this detector.
    
    # ----- now: the ramps. ---------
    def load_curves_axis(s):
        # -------- INITIALIZING CRV ---------
        nc = s.num_curves
        # We have to declare a LOT of numpy arrays. 
        # we're going to call curves crv for now
        crv = {}
        # crv key: (init fill with, dtype, whereFrom)
        crv_init = {'bl_powered':(-42,int,'special'),
                    'i_tes_offset':(np.nan,float,'iva'),
                    'p_trans':(np.nan,float,'iva'),
                    f'p_b{s.use_per}':(np.nan,float,'iva'),
                    'R_nSM':(np.nan,float,'iva'),
                    'all_fit_R_n':(np.nan,float,'iva'),
                    'all_fit_ITO':(np.nan,float,'iva'),
                    'all_fit_stats':(np.nan,object,'iva'),
                    'sc_idx':(-42000,int,'special'),
                    'nb_idx':(-42000,int,'special'),
                    'R_n':(np.nan,float,'iva'),
                    'p_opt':(np.nan,float,'special'), # powers dict
                    'p_opt_err':(np.nan,float,'special'), # powers dict
                    'bath_temp':(np.nan,float,'special'),
                    'cl_temp':(np.nan,float,'special'),
                    'sb':(-42,int,'special'),
                    'ch':(-42,int,'special'),
                    'sb_ch':(-42,int,'special'),
                    'bl':(-42,int,'special'), # SHOULD HAVE A consolidate_mux function. 
                    'det_x':(np.nan,float,'mux'), # ^That happens before this
                    'det_y':(np.nan,float,'mux'), 
                    'det_r':(np.nan,float,'mux'), # move into mux.
                    'pol':('?',str,'mux'),    
                    'TES_freq':(-42,int,'mux'), # get from biasline map
                    'OMT':('?',str,'mux'),
                    'masked':('?',str,'mux'), # masked biaslines
                    'masked_freq':('?',object,'special'),
                    'ramp_from':(np.nan,object,'special'),
                    'ramp_name':('?',object,'special'),
                    'is_iv':(False,bool,'iva'),
                    'is_iv_info':('?',object,'iva'),
                    'cut_info':('?',object,'iva'),
                    'is_normal':(False,bool,'iva'),
                    'v_bias':(np.nan,object,'iva'),
                    'i_bias':(np.nan,object,'iva'),
                    'v_tes':(np.nan,object,'iva'),
                    'i_tes':(np.nan,object,'iva'),
                    'R':(np.nan,object,'iva'),
                    'p_tes':(np.nan,object,'iva'),
                    'i_tes_stdev':(np.nan,object,'iva'),
                    'i_bias_unbin_sc':(np.nan,object,'iva'),
                    'i_tes_unbin_sc':(np.nan,object,'iva'),
                    'pvb_idx':(-42,int,'iva'),
                    'iva':(np.nan,object,'special')}
        for key,init_info in crv_init.items():
            crv[key] = np.full((nc,), init_info[0], dtype=init_info[1])
        # Special, for coding
        crv['idx'] = np.arange(nc)
        
        # -------- FILLING CRV ---------
        i = 0 # curve index
        for r in range(len(s.ramps['ramp'])):
            ramp = s.ramps['ramp'][r]
            for metadata in ramp.iv_analyzed_info_arr:
                for run_d in metadata:
                    if ramp.ramp_type == 'bath':
                        bath_temp = run_d['temp']
                        cl_temp = s.ramps['cl_temps'][r][s.ramps['bath_temps'][r] == bath_temp]
                    else:
                        cl_temp = run_d['temp']
                        bath_temp = s.ramps['bath_temps'][r][s.ramps['cl_temps'][r] == cl_temp]
                    bl_powered = run_d['bl']
                    for sb in run_d['iv_analyzed'].keys():
                        if sb == "high_current_mode":
                            continue
                        for ch,iva in run_d['iv_analyzed'][sb].items():
                            # special: higher up.
#                             print(f"{ramp.ramp_name} {bath_temp} {cl_temp} {bl_powered}")
#                             print(f"{s.ramps['bath_temps'][r] }")
#                             print(f"{s.ramps['ramp'][r].temp_list_raw}")
                            crv['ramp_from' ][i] = ramp
                            crv['ramp_name' ][i] = ramp.ramp_name
                            crv['bath_temp' ][i] = bath_temp
                            crv['cl_temp'   ][i] = cl_temp
                            crv['bl_powered'][i] = bl_powered
                            crv['bl'][i] = s.mux_map[sb][ch]['biasline']
                            crv['sb'][i] = sb
                            crv['ch'][i] = ch
                            crv['sb_ch'][i] = sb*1000+ch
                            spc = (3 - len(str(s.mux_map[sb][ch]['TES_freq'])))*" "
                            if is_unknown(s.mux_map[sb][ch]['masked']) \
                               or is_unknown(s.mux_map[sb][ch]['TES_freq']):
                                crv['masked_freq'][i] = '?'
                            elif s.mux_map[sb][ch]['masked'] == 'Y':
                                crv['masked_freq'][i] = f"masked {spc}{s.mux_map[sb][ch]['TES_freq']}"
                            elif s.mux_map[sb][ch]['masked'] == 'N':
                                crv['masked_freq'][i] = f"unmasked {spc}{s.mux_map[sb][ch]['TES_freq']}"
                            # mux and iva items. 
                            for key, init_info in crv_init.items():
                                if init_info[2] == 'mux':
                                    crv[key][i] = s.mux_map[sb][ch][key]
                                if init_info[2] == 'iva' and key in iva.keys(): 
                                    crv[key][i] = iva[key]
                                    # SPECIAL CHEAT TO SEE WHAT IT LOOKS LIKE
                                    if key == 'p_b90' and s.use_per != 90:
                                        # We recalculate the p_b90 with the cheated percentage
                                        if len(np.ravel(np.where(iva['R']<s.use_per/100.0*iva['R_n']))) > 0:
                                            crv[key][i] = iva['p_tes'][np.ravel(np.where(iva['R']<s.use_per/100.0*iva['R_n']))[-1]]
                                        else:
                                            crv[key][i] = np.nan # you never reach it. 
                            crv['sc_idx'][i], crv['nb_idx'][i] = iva['trans idxs']
                            crv['iva'][i] = iva
                            # optical power
                            opt_i_l = np.where((s.opt_powers['sb'] == sb) 
                                              & (s.opt_powers['ch'] == ch))[0]
                            if len(opt_i_l) == 0:
                                # no opt_i info, likely due to no mux match
                                #print(f"no opt info?!? idx{i} sb{sb} ch{ch}")
                                i+=1
                                continue
                            if cl_temp in s.opt_powers['cl_temp'][opt_i_l]:
                                idx = opt_i_l[s.opt_powers['cl_temp'][opt_i_l]==cl_temp]
                                crv['p_opt'][i] = s.opt_powers['p_opt'][idx]
                                crv['p_opt_err'][i] = s.opt_powers['p_opt_err'][idx]
                            else:
                                # interpolate
                                closest_t_i = np.where(abs(s.opt_powers['cl_temp'][opt_i_l]-cl_temp) == min(abs(s.opt_powers['cl_temp'][opt_i_l]-cl_temp)))[0][0]
                                if s.opt_powers['cl_temp'][opt_i_l[closest_t_i]] < cl_temp:
                                    low_i = closest_t_i
                                else:
                                    low_i = closest_t_i - 1
                                high_i = low_i + 1
                                if low_i < 0 or high_i >= len(opt_i_l):
                                    crv['p_opt'][i] = np.nan
                                    crv['p_opt_err'][i] = np.nan
                                else:
                                    p_opts = s.opt_powers['p_opt'][opt_i_l]
                                    cl_temps = s.opt_powers['cl_temp'][opt_i_l]
                                    slope = (p_opts[high_i]-p_opts[low_i])/(cl_temps[high_i]-cl_temps[low_i])
                                    const = p_opts[high_i] - cl_temps[high_i]*slope
                                    crv['p_opt'][i] = slope*cl_temp + const
                                    p_opt_errs = s.opt_powers['p_opt_err'][opt_i_l]
                                    crv['p_opt_err'][i] = max(p_opt_errs[low_i],p_opt_errs[high_i])
                                
#                             if len(opt_i_l) == 1:
#                                 opt_i = opt_i_l[0]
#                                 crv['p_opt'][i] = s.opt_powers['p_opt_s'][opt_i]*cl_temp \
#                                                   +s.opt_powers['p_opt_c'][opt_i]
#                                 crv['p_opt_err'][i] = s.opt_powers['p_opt_err_s'][opt_i]*cl_temp \
#                                                       +s.opt_powers['p_opt_err_c'][opt_i]
                            #else:
                                #print(f"crv{i} sb{sb} ch{ch} cl_t{cl_temp}: {opt_i_l}")
                                #pass
                            # onto the next curve
                            i += 1
        s.crv = crv
        return crv
    
    # ----- And here it is, the actual together-fits setup. -----
    def count_TESs(s):
        '''List the TESs that were ever fit well enough 
        by a BATH ramp to have thermal data. Sadly Coldload_Ramp
        doesn't do anything with them if no bath_ramp got them,
        I think.
        THIS IS CURRENTLY UNUSED, 
        '''
        # we can avoid dynamic allocation because we know the 
        # upperbound on the tes_sbchs: num_sbch
        tes_sbchs = [(-42,-42) for i in range(s.num_sbchs)]
        # we're going to make our own decisions
        # so can avoid a second loop by tracking the sbchs. 
        cur_idx = 0
        for ramp in s.ramps['ramp']:
            if not ramp.ramp_type == 'bath':
                continue
            for sb in ramp.tes_dict.keys():
                for ch, d in ramp.tes_dict[sb].items():
                    if 'n' in d.keys() and not (sb,ch) in tes_sbchs: # successful thermal fit, new tes
                        tes_sbchs[cur_idx] = (sb,ch)
                        cur_idx+=1
        s.tes_sbchs = tes_sbchs[:cur_idx]
    
    def init_TES_axis(s):
        # -------- INITIALIZING TES ---------
        nt = len(s.sbchs)
        # We have to declare quite a few numpy arrays
        tes = {}
        # crv key: (init fill with, dtype, whereFrom)
        tes_init = {'sb':(-42,int,'special'),
                    'ch':(-42,int,'special'),
                    'sb_ch':(-42,int,'special'),
                    'bl':(-42,int,'special'), # Just because I want to call it bl. :/ 
                    'det_x':(np.nan,float,'mux'), # ^That happens before this
                    'det_y':(np.nan,float,'mux'), 
                    'det_r':(np.nan,float,'mux'), 
                    'pol':('?',str,'mux'),    
                    'TES_freq':(-42,int,'mux'), # get from biasline map
                    'OMT':('?',str,'mux'),
                    'masked':('?',str,'mux'), # now begin the data.
                    'masked_freq':('?',object,'special'),
                    'R_n':(np.nan,float,'special'), # oh boy....going to have to copy some code there
                    'was_fit':(False,bool,'thermal'),
                    'G': (np.nan,float,'thermal'), # ---- thermal begins
                    'G_err': (np.nan,float,'thermal'),
                    'Tc': (np.nan,float,'thermal'),
                    'Tc_err': (np.nan,float,'thermal'),
                    'n': (np.nan,float,'thermal'),
                    'n_err': (np.nan,float,'thermal'),
                    'opt_eff_no_subtraction': (np.nan,float,'thermal'), 
                    'opt_eff_no_subtraction_err': (np.nan,float,'thermal'),
                    'cov_G_Tc': (np.nan,float,'thermal'),
                    'cov_G_n': (np.nan,float,'thermal'),
                    'cov_G_eff':(np.nan,float,'thermal'),
                    'cov_Tc_n': (np.nan,float,'thermal'),
                    'cov_Tc_eff': (np.nan,float,'thermal'),
                    'cov_n_eff': (np.nan,float,'thermal'),
                    'p_sat100mK': (np.nan,float,'thermal'),
                    'p_sat100mK_err': (np.nan,float,'thermal'),
                    'opt_eff': (np.nan,float,'optical'), # -----also, T_b est lines & stuff
                   }
        for key,init_info in tes_init.items():
            tes[key] = np.full((nt,), init_info[0], dtype=init_info[1])
        # Special, for coding
        tes['idx'] = np.arange(nt)

        # We can do the easy filling right now
        for i in range(len(s.sbchs)):
            sb, ch = s.sbchs[i]
            tes['sb'][i] = sb
            tes['ch'][i] = ch
            tes['sb_ch'][i] = sb*1000+ch
            tes['bl'][i] = s.mux_map[sb][ch]['biasline']
            spc = (3 - len(str(s.mux_map[sb][ch]['TES_freq'])))*" "
            if is_unknown(s.mux_map[sb][ch]['masked']) \
               or is_unknown(s.mux_map[sb][ch]['TES_freq']):
                tes['masked_freq'][i] = '?'
            elif s.mux_map[sb][ch]['masked'] == 'Y':
                tes['masked_freq'][i] = f"masked {spc}{s.mux_map[sb][ch]['TES_freq']}"
            elif s.mux_map[sb][ch]['masked'] == 'N':
                tes['masked_freq'][i] = f"unmasked {spc}{s.mux_map[sb][ch]['TES_freq']}"
            for key, init_info in tes_init.items():
                if init_info[2] == 'mux':
                    tes[key][i] = s.test_device.mux_map[sb][ch][key]
        s.tes_init = tes_init
        s.tes = tes
        s.axes['tes'] = tes
        
    def p_b90_of_bath_temp_and_p_opt(s,tb_and_p_opt,G,tc,n,eff):
        tb, p_opt = tb_and_p_opt
        return G/n *(tc-tb**n/tc**(n-1)) - eff*p_opt


    def p_b90_of_bath_temp(s,tb,G,tc,n):
        return G/n *(tc-tb**n/tc**(n-1))

    # THIS DOES NOT USE THE ERRORS (ex. on p_opt) EFFECTIVELY YET. 
    #parameters,cov = 

    def attempt_no_dark_subtraction_opt_eff(s,i,debug=0,num_fit=0,all_bad_idxs=[]):
        try:
            sb, ch = s.tes['sb'][i], s.tes['ch'][i]
            idxs = s.find_idx_matches([('sb','=',sb),('ch','=',ch),('is_iv','=',True)])
            #True: # innate "dark subtraction" 
            if True: #s.tes['masked'][i] == 'N' and s.tes['OMT'][i] == 'Y':
                calc_eff = True
            else:
                calc_eff = False
            # THIS IS A HACK: **TODO** fix the missing p_b90 in is_iv==True bug!!
            bad_idxs = []
            good_idxs = []
            for idx in idxs: 
                if (is_float(s.crv['bath_temp'][idx]) #and is_float(s.crv['p_opt'][idx]) 
                    and is_float(s.crv[f'p_b{s.use_per}'][idx])) \
                    and ((not calc_eff) or is_float(s.crv['p_opt'][idx]) ):
                    good_idxs.append(idx)
                else:
                    bad_idxs.append(idx)
            idxs = np.array(good_idxs)
            all_bad_idxs = all_bad_idxs + bad_idxs
            if len(idxs) < 5: # Gotta have more points than variables!
                return (False,"<5 is_ivs")  #continue # too few points to work with
            tbs, p_opts, p_b90s = s.crv['bath_temp'][idxs],s.crv['p_opt'][idxs], s.crv[f'p_b{s.use_per}'][idxs]  

        #             if i < debug:
        #                 print(f"{i}:Looking at {sb} {ch} bad_idxs:{bad_idxs}")

            # get the g_guess and min_b_temp_span
            for ramp, other_temp in s.ramp_l: # just getting the guesses
                if ramp.ramp_type == 'bath':
                    ex = ramp
                    break
            if max(tbs)-min(tbs) < ex.min_b_temp_span: # 21; switched to 10 for SPB3 # in mK
                return (False, f"points don't span {ex.min_b_temp_span} mK") # continue
            # Merging bathsweeps resulted in ones getting 4 points that have no business being fit
            # So, check for a minimum temperature range too:
            if str(s.mux_map[sb][ch]['TES_freq']).isnumeric() and \
               int(s.mux_map[sb][ch]['TES_freq']) in ex.g_guess_by_freq.keys():
                g_guess = ex.g_guess_by_freq[int(s.mux_map[sb][ch]['TES_freq'])]
            else:
                g_guess = 0.130
            try:

                # IMPORTANT NOTE: absolute_sigma=True could be better, for the errors.
                # It basically assumes all inputs have errorbars of 1.
                # To improve that I should provide an errorbar (somehow...)

                # Now we only want to fit p_opt for unmasked, optical-fab detectors
                if calc_eff:
                    (popt,cov_G_Tc_n_eff) = curve_fit(s.p_b90_of_bath_temp_and_p_opt,
                                           (tbs,p_opts),
                                           p_b90s,
                                           p0 = [g_guess, ex.Tc_guess,ex.n_guess,0.5],
    #                                        bounds = (np.array([-np.inf,-np.inf,-np.inf,0]),
    #                                                  np.array([np.inf,np.inf,np.inf,1])),
                                           absolute_sigma=True)
                    if i< debug:
                        print(f"{i}: sb{sb} ch{ch} {popt} {cov_G_Tc_n_eff}")
                else: # unknown locations have to go here, b/c will have no p_opt info
                    (popt,cov_G_Tc_n_eff) = curve_fit(s.p_b90_of_bath_temp,
                                           tbs,
                                           p_b90s,
                                           p0 = [g_guess, ex.Tc_guess,ex.n_guess],
                                           absolute_sigma=True)
            except RuntimeError:
                return (False,"RuntimeError in curvefit") # continue

            # debug:
        #             if i <debug:
        #                 print(f"{sb} {ch} {popt} {cov_G_Tc_n_eff}") 

            # Success!!! So let's unpack. 
            s.tes['was_fit'][i] = True
            num_fit += 1
            G,Tc,n = popt[:3]
            G_err,Tc_err,n_err = np.sqrt(np.diag(cov_G_Tc_n_eff))[:3] 

            s.tes['G'][i] = G
            s.tes['G_err'][i] = G_err
            s.tes['Tc'][i] = Tc
            s.tes['Tc_err'][i] = Tc_err
            s.tes['n'][i] = n
            s.tes['n_err'][i] = n_err
            s.tes['cov_G_Tc'][i] = cov_G_Tc_n_eff[0,1]
            s.tes['cov_G_n'][i] = cov_G_Tc_n_eff[0,2]
            s.tes['cov_Tc_n'][i] = cov_G_Tc_n_eff[1,2]

            s.tes['p_sat100mK'][i] = s.p_b90_of_bath_temp(100,G,Tc,n)
            #s.tes['p_sat100mK_err'][i] = p_sat100mK_err **TODO**: jacobian!!
            if calc_eff: #s.tes['masked'][i] == 'N' and s.tes['OMT'][i] == 'Y':
                s.tes['opt_eff_no_subtraction'][i] = popt[3]
                s.tes['opt_eff_no_subtraction_err'][i] = np.sqrt(cov_G_Tc_n_eff[3,3])
                s.tes['cov_G_eff'][i] = cov_G_Tc_n_eff[0,3]
                s.tes['cov_Tc_eff'][i] = cov_G_Tc_n_eff[1,3]
                s.tes['cov_n_eff'][i] = cov_G_Tc_n_eff[2,3]
        except BaseException as err:
            print(f"Unexpected {err}, {type(err)}: tes idx{i} sb{sb} ch{ch}")
            raise
        return (num_fit, all_bad_idxs)

    def no_dark_subtraction_opt_eff(s,debug=0):
        num_fit = 0
        all_bad_idxs = []
        # Setup fail_reasons
        for ramp, other_temp in s.ramp_l: # just getting the guesses
            if ramp.ramp_type == 'bath':
                ex = ramp
                break
        fail_reasons = {"<5 is_ivs":[],  
                        f"points don't span {ex.min_b_temp_span} mK":[],
                       "RuntimeError in curvefit":[]}
        for i in range(len(s.tes['idx'])):
            result = s.attempt_no_dark_subtraction_opt_eff(i,num_fit=num_fit,
                                                           all_bad_idxs=all_bad_idxs,debug=debug)
            if result[0]:
                num_fit, all_bad_idxs = result
            else: 
                fail, fail_reason = result
                fail_reasons[fail_reason].append(i)
        print(f"FIT {num_fit} TESs")
        print("Failures: " + "; ".join([f"{key}: {len(fail_reasons[key])}" for key in fail_reasons.keys()]))
        return fail_reasons #all_bad_idxs   
    
               
               
    # ============== Functions to FIND data ========== 
    def find_idx_matches(s,match_list,dict_return=False,ax_name='crv',
                         exclude_unknowns=False):
        '''Returns what indices of the ax match the match_list criteria.
        See Temp_Ramp.find_iva_matches() and str_to_match_list() aka ml()
        set exclude_unkowns to a list of axis keys. It will exclude
        any idxs that have a -42, np.nan, "-", or "?" as that key's value.
        See below this class for the general implementation.'''
        # like find_ivas, but returns idxs. 
        ax = s.axes[ax_name]
        return find_idx_matches(ax,match_list,dict_return=dict_return, exclude_unknowns=exclude_unknowns)
    
    # Plotting function aliases for backwards compatibility
#2345678901234567890123456789012345678901234567890123456789012345678901234567890
    def plot_by_ramp(s,x_key,y_key,match_list=[],x_lim=[],y_lim=[],prefix='',
                     plot_args={},own_fig=True):
        return plot_by_ramp(s,x_key,y_key,match_list=match_list,x_lim=x_lim,
                        y_lim=y_lim, prefix=prefix,plot_args=plot_args,
                        own_fig=own_fig)

    def plot_key_v_key(s,x_key,y_key,match_list=[],x_lim=[],y_lim=[], prefix='',
                       plot_args={},ax_name='crv',own_fig=True):
        return plot_key_v_key(s,x_key,y_key,match_list=match_list,x_lim=x_lim,
                              y_lim=y_lim, prefix= prefix,plot_args=plot_args,
                              ax_name=ax_name,own_fig=own_fig)
    def plot_key_v_key_grouped_by_key(s,x_key,y_key,by_key,match_list=[],
                                      exclude_unknowns=False,ax_name='crv',
                                      x_lim=[],y_lim=[], prefix='',
                                      plot_args={},own_fig=True,legend='default'):
        return plot_key_v_key_grouped_by_key(s,x_key,y_key,by_key, 
                                             match_list=match_list,
                                             exclude_unknowns=exclude_unknowns,
                                             ax_name=ax_name,x_lim=x_lim,
                                             y_lim=y_lim, prefix=prefix,
                                             plot_args=plot_args,own_fig=own_fig,
                                             legend=legend)
    def plot_key_v_key_colored_by_key(s,x_key,y_key,by_key,match_list=[],
                                      exclude_unknowns=False, ax_name='crv',
                                      x_lim=[],y_lim=[], prefix='',
                                      title_override='', plot_args={'marker':'.'},
                                      own_fig=True, v_lim=[],
                                      color_scale=plt.cm.inferno, 
                                      outlier_colors=['blue','green'], 
                                      xy_overlap_offset=1):
        return plot_key_v_key_colored_by_key(s,x_key,y_key,by_key,
                                             match_list=match_list,
                                             exclude_unknowns=exclude_unknowns, 
                                             ax_name= ax_name,x_lim=x_lim,
                                             y_lim=y_lim, prefix= prefix,
                                             title_override=title_override, 
                                             plot_args= plot_args,own_fig=own_fig, 
                                             v_lim= v_lim,color_scale=color_scale,
                                             outlier_colors= outlier_colors, 
                                             xy_overlap_offset= xy_overlap_offset)
     



 

    

# ==============================================================================
# -------------------- Ramp_Combination Analysis Functions ---------------------
# ==============================================================================

# ----------- Manipulation functions

def ax_line_as_dict(ax,idx):
    '''convenience function for some axis referencing'''
    return {key:ax[key][idx] for key in ax.keys()}
    
def is_unknown(val):
    '''is it a val on rc's "unknown" list?
       that is, -42, -42.0, '-', '?', np.nan'''
    for uv in [-42,-42.0,-4.20,'-','?']:
        if val == uv:
            return True
    try: # needs a special comparison, always returns False to == or !=
        if np.isnan(val): 
            return True
    except TypeError: # it's not a np.nan then!
        return False
    return False

def replace_rc_thermal_with_bath_ramp_results(s,ramp):
    '''Useful for running many rc functions on a bath_ramp
    Note tes_dict gets updated from rc, I think, need to use
    all_results.'''
    ramp.opt_values_exclude=[]
    ramp.do_fits()
    rtes = ramp.all_results
    direct_copyables = ['R_n','G', 'Tc',  
                        'n',   'p_sat100mK', 'p_sat100mK_err'] 
    cov_copyables = ['G_err','Tc_err','n_err','cov_G_Tc', 'cov_G_n', 'cov_Tc_n']
    more_scrub = direct_copyables + cov_copyables
    cov_idxs = {'G':0,'Tc':1,'n':2}
    for i in s.tes['idx']:
        dv = ax_line_as_dict(s.tes,i)
        for key in ['opt_eff_no_subtraction', 'opt_eff_no_subtraction_err',
                    'cov_G_eff','cov_Tc_eff','cov_n_eff','opt_eff']:
            s.tes[key][i] = np.nan            
        if dv['sb'] in rtes.keys() and dv['ch'] in rtes[dv['sb']].keys()\
           and 'G' in rtes[dv['sb']][dv['ch']].keys()\
           and not is_unknown(rtes[dv['sb']][dv['ch']]['G']):
            bd = rtes[dv['sb']][dv['ch']]
            s.tes['was_fit'][i] = True
            for key in direct_copyables:
                s.tes[key][i] = bd[key]
            for c1,c2 in [('G','G'),('Tc','Tc'),('n','n'),
                          ('G','Tc'),('G','n'),('Tc','n')]:
                key_name = f'cov_{c1}_{c2}'
                if c1 == c2:
                    key_name = f'{c1}_err'
                s.tes[key_name][i] = np.sqrt(bd['cov_G_Tc_n'][cov_idxs[c1]][cov_idxs[c2]])
        else:
            # scrub it!
            s.tes['was_fit'][i] = False
            for key in more_scrub:
                s.tes[key][i] = np.nan
    # update limits
    for key in direct_copyables + cov_copyables:
        if key in s.key_info.keys():
            d = s.key_info[key]
            s.update_key_info(key,d['name'],d['ax_name'],'c',d['units'])
                



# ============== Functions to PLOT data ========== 
  
    
def ton_of_sbchs(s,x_key,y_key, start='random',stop=20,match_list=[],exclude_unknowns=False,
              x_lim=[],y_lim=[],plot_args={'linewidth':0.5}):
    # s=ramp combination
    # check if the number is restricted by our restrictions
    my_sb_chs = np.unique(s.crv['sb_ch'][s.find_idx_matches(match_list,ax_name='crv',
                                                            exclude_unknowns=exclude_unknowns)])
    if stop=='all':
        stop = len(my_sb_chs)
    if start == 'random':
        rand_sb_chs = np.array([my_sb_chs[rand.randrange(0,len(my_sb_chs))] for i in range(stop)])
        my_sb_chs = rand_sb_chs
        start= 0
    
    ncols = 4
    nrows = math.ceil((stop-start)/ncols)
    figsize = (ncols*2,nrows*2)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    i = 0
    argy = copy.deepcopy(plot_args)
    for sb_ch in my_sb_chs[start:stop]: 
        sb = int(sb_ch/1000)
        ch = sb_ch - 1000*sb
        r = math.floor(i/ncols)
        c = i- ncols*r
        if nrows == 1:
            p = ax[i]
        else:
            p=ax[r][c]
        #p.plot([i for i in range(3)],[i/100 for i in range(3)])
        match_listy = match_list + [('sb','=',sb),('ch','=',ch)]
        argy['linestyle'] = 'dashed'        
        plot_key_v_key_grouped_by_key(s,x_key,y_key,'bath_temp',match_list=match_listy,
                                        exclude_unknowns=exclude_unknowns,
                                      ax_name='crv',x_lim=x_lim,y_lim=y_lim, prefix='',
                                        plot_args=argy,
                                      own_fig=p,legend=False)
        match_listy = match_list + [('sb','=',sb),('ch','=',ch),('is_iv','=',True)]
        argy['linestyle'] = 'solid'  
        plot_key_v_key_grouped_by_key(s,x_key,y_key,'bath_temp',match_list=match_listy,
                                        exclude_unknowns=exclude_unknowns,
                                      ax_name='crv',x_lim=x_lim,y_lim=y_lim, prefix='',
                                        plot_args=argy,
                                      own_fig=p,legend=False)
        if x_lim:
            p.set_xlim(x_lim)
        if y_lim:
            p.set_ylim(y_lim)
        bl= s.tes['bl'][s.find_idx_matches([('sb','=',sb),('ch','=',ch)],ax_name='tes')[0]]
        p.set_title(f"{sb},{ch}; bl{bl}")
        i+=1
        
    plt.suptitle(f"{s.dName} (sb,ch) {y_key} vs. {x_key} ",y=0.999)
    plt.tight_layout()
    


def plot_key_v_key(s,x_key,y_key,match_list=[],x_lim=[],y_lim=[],prefix='',title_override='',
                   label='',plot_args={},ax_name='crv',own_fig=True,exclude_unknowns=False):
    # If I wanted separate plots I'd just use the ramps themselves.
    if own_fig:
        plt.figure(figsize=default_figsize)
    axis = s.axes[ax_name]
    #restrict = 

    restrictions = ', '.join([f"{mk}{mt}{mv}" for mk,mt,mv in match_list])
    if not label:
        label=restrictions
    if exclude_unknowns == True:
        exclude_unknowns = [x_key,y_key]
    my_idxs = s.find_idx_matches(match_list,ax_name=ax_name,exclude_unknowns=exclude_unknowns) 
    p = plt.plot(axis[x_key][my_idxs], axis[y_key][my_idxs],label=label, **plot_args)# not a good way of doing labelling
    plt.title(f"{prefix} {y_key} vs. {x_key}\n({s.dName})\n{restrictions}")
    if not own_fig:
        plt.legend()
        plt.title(f"{prefix} {y_key} vs. {x_key}\n({s.dName})")
    if title_override:
        plt.title(title_override)
    plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]") # TODO: better axis! Use key_info!
    plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
    #plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]") 
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    return p

#234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
def plot_key_v_key_grouped_by_key(s,x_key,y_key,by_key,match_list=[],exclude_unknowns=False,
                                  ax_name='crv',x_lim=[],y_lim=[], prefix='',plot_args={},
                                  own_fig=True,legend='default'):
    axis = s.axes[ax_name]
    restrictions = ''
    if exclude_unknowns or len(match_list) > 0:
        restrictions = '\n'
    if exclude_unknowns:
        restrictions = restrictions + 'no unknowns '
        if exclude_unknowns == True:
            exclude_unknowns = [x_key,y_key,by_key]
    if len(match_list) >0:
        restrictions = restrictions + f"[{', '.join([f'{mk}{mt}{mv}' for mk,mt,mv in match_list])}]"
    idxs = s.find_idx_matches(match_list,ax_name=ax_name,exclude_unknowns=exclude_unknowns)
    xs,ys = axis[x_key][idxs], axis[y_key][idxs]

    group_names = np.sort(np.unique(axis[by_key][idxs]))
    if (not by_key in s.key_info.keys()) or not s.key_info[by_key]['type'] == 'd':
        # probably going to have other problems, but let it be for now:
        print(f"warning: {by_key} not in key_info or is not discrete!")
        if len(group_names) > 10:
            tab20 = plt.get_cmap('tab20')
            colors = tab20(np.linspace(0, 1.0, len(group_names)+1))
        else:
            tab10 = plt.get_cmap('tab10')
            colors = tab10(np.linspace(0, 1.0, len(group_names)+1))
    else:
        colors = s.key_info[by_key]['colors'][np.in1d(s.key_info[by_key]['vals'],group_names).nonzero()]
    p = plt
    if own_fig == True:
        plt.figure(figsize=default_figsize)
    elif not own_fig == False:
        p= own_fig
    
    color_override = False
    if 'color' in plot_args.keys():
        color_override=True
    for i in range(len(group_names)):
        name = group_names[i]

        #name_color = s.key_info[by_key]['colors'][np.where(s.key_info[by_key]['vals'])[0][0],:]
        if not color_override:
            plot_args['color'] = colors[i]
        g_idxs = idxs[axis[by_key][idxs] == name]
        num_points = len(g_idxs)
        if name in [-42,'?',np.nan]:
            name="?"
        # Check if these are arrays or not
        if is_float(axis[x_key][g_idxs[0]]):
            p.plot(axis[x_key][g_idxs], axis[y_key][g_idxs],label=f"{name} ({num_points} pts)", **plot_args)
        else:
            labeller = p.plot([],[],label=f"{name}", **plot_args)# ({num_points} array pairs)
#             plot_args['color'] = labeller[0].get_color()
            for j in range(len(g_idxs)):
                p.plot(axis[x_key][g_idxs[j]], axis[y_key][g_idxs[j]], **plot_args)
    if own_fig == True:
        p.title(f"{prefix} {y_key} vs. {x_key} BY {by_key}\n{s.dName}{restrictions}")
        #plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]") 
        x_label = f"{x_key}"
        if x_key in s.key_info.keys():
            x_label = f"{s.key_info[x_key]['name']}"
            if 'units' in s.key_info[x_key].keys(): # should be if continuous
                x_label = x_label + f" [{s.key_info[x_key]['units']}]"
        p.xlabel(x_label)
        y_label = f"{y_key}"
        if y_key in s.key_info.keys():
            y_label = f"{s.key_info[y_key]['name']}"
            if 'units' in s.key_info[y_key].keys(): # should be if continuous
                y_label = y_label + f" [{s.key_info[y_key]['units']}]"
        p.ylabel(y_label)
        if x_lim:
            p.xlim(x_lim)
        if y_lim:
            p.ylim(y_lim)
    if legend=='default':
        if len(group_names) > 20:
            legend = False
    if legend:
        p.legend()
    return p

def plot_key_v_key_colored_by_key(s,x_key,y_key,by_key,match_list=[],exclude_unknowns=False,
                                  ax_name='crv',x_lim=[],y_lim=[], prefix='',title_override = '',
                                  plot_args={'marker':'.'},own_fig=True, v_lim=[],
                                  color_scale=plt.cm.inferno, outlier_colors=['blue','green'],
                                   xy_overlap_offset=1):
    axis = s.axes[ax_name]
    restrictions = ''
    if exclude_unknowns or len(match_list) > 0:
        restrictions = '\n'
    if exclude_unknowns:
        restrictions = restrictions + 'no unknowns '
        if exclude_unknowns == True:
            exclude_unknowns = [x_key,y_key,by_key]
    if len(match_list) >0:
        restrictions = restrictions + f"[{', '.join([f'{mk}{mt}{mv}' for mk,mt,mv in match_list])}]"

    idxs = s.find_idx_matches(match_list,ax_name=ax_name,exclude_unknowns=exclude_unknowns)

    # analyzing values
    #np.array([float(axis[by_key][idx]) for idx in idxs]) # has to all be floats, but come on... how would it not be?
    vals = axis[by_key][idxs] 
    #print(vals)
    if v_lim:
        v_min, v_max = v_lim
    else:
        v_max, v_min = max(vals), min(vals)
    ratios = (vals-v_min)/(v_max-v_min)
    print([v_min, v_max])

    # coloring
    try: # no reason to make a new one every time we graph
        s.color_converter
    except AttributeError:
        s.color_converter = mpl.colors.ColorConverter()
    cc = s.color_converter
    rgb_outlier_colors = [np.array(cc.to_rgb(outlier_colors[0])),
                      np.array(cc.to_rgb(outlier_colors[1]))]
    # want it to handle itself, so...
    colors = np.full((len(idxs),3), [0.0,0.0,0.0]) # cc just an arbitrary reference here to avoid dynamic allocation
    for i in range(len(ratios)):
        if ratios[i] < 0.0:
            colors[i,:] = rgb_outlier_colors[0]
        elif ratios[i] > 1.0:
            colors[i,:] = rgb_outlier_colors[1]
        else:
            with_alpha = color_scale(ratios[i])
            #print(with_alpha)
            colors[i,:] = np.array([with_alpha[0],with_alpha[1],with_alpha[2]])

    # **TODO**: wedges and ordering ranks instead of  EXTREMELY HACKY SOLUTION TO OVERLAP!
    xs = axis[x_key][idxs]
    ys = axis[y_key][idxs]
    if xy_overlap_offset > 0:
        x_uniques, x_counts = np.unique(xs,return_counts=True)
        for i_x in range(len(x_uniques)):
            if x_counts[i_x] == 1: # no overlaps
                continue
            x_unique = x_uniques[i_x]
            y_uniques,y_counts = np.unique(ys[xs==x_unique],return_counts=True)
            if max(y_counts) == 1: # all unique
                continue
            for i_y in range(len(y_uniques)):
                if y_counts[i_y] == 1:
                    continue
                overlap_idxs = np.where((xs==x_unique) & (ys == y_uniques[i_y]))[0]
                for i_theta in range(y_counts[i_y]):
                    xs[overlap_idxs[i_theta]] += xy_overlap_offset*np.cos(2*np.pi*i_theta/y_counts[i_y]) 
                    ys[overlap_idxs[i_theta]] += xy_overlap_offset*np.sin(2*np.pi*i_theta/y_counts[i_y]) 

    # plotting time!
    if own_fig:
        fig = plt.figure(figsize=default_figsize)
        ax = fig.add_subplot(111)
    else:
        ax = plt.gca()

    # I dislike this way of doing this, but I have to plot a different one ...
    pt = ax.scatter([],[],c=[],vmin=v_min,vmax=v_max,cmap=color_scale, **plot_args)
    bar = plt.colorbar(pt)

    if not 's' in plot_args.keys():
        plot_args['s'] = 6
#     if not 'edgecolors' in plot_args.keys():
#         plot_args['edgecolors'] = 'k'
    ax.scatter(xs,ys,c=colors, **plot_args)

    plt.title(f"{prefix} {y_key} vs. {x_key} colored BY {by_key}\n{s.dName}{restrictions}")
    if title_override:
        plt.title(title_override)
    #plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]") 
    plt.xlabel(f"{s.key_info[x_key]['name']} [{s.key_info[x_key]['units']}]")
    plt.ylabel(f"{s.key_info[y_key]['name']} [{s.key_info[y_key]['units']}]")
    low_outlier, high_outlier = '',''
    if min(ratios) < 0.0:
        low_outlier = f" <{v_min}={outlier_colors[0]}"
    if max(ratios) > 1.0:
        high_outlier = f" >{v_max}={outlier_colors[1]}"
    bar.set_label(f"{s.key_info[by_key]['name']} [{s.key_info[by_key]['units']}]; {low_outlier} {high_outlier}")
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    return ax
        

# ------------ Histograms and Thermal summary plotting --------------------

def key_hist_by_freq_and_mask(s,key,x_lim=[],bin_size=None,bin_offset=0.15,
                              ax_name='tes',match_list=[],exclude_unknowns=False,own_fig=True,
                             full_legend_label=True,exclude_bath_ramp_fails=False,
                              hatch_amount=3,sig_figs=3):
    """s a Ramp Combination, but I think this can also have s as a Bath_Ramp?!?"""
    axis = s.axes[ax_name] # **TODO**: HACK!
    # THis is a hack. 
    if exclude_bath_ramp_fails == True:
        match_list.append(('bath_ramp_fit','=',True))
        #if "bath_ramp_fit" not in s.tes.keys():
        s.tes['bath_ramp_fit'] = np.full((len(s.tes['sb']),),False,dtype=bool)
        br_r = s.ramp_l[0][0].tes_dict #all_results
        for i in range(len(s.tes['sb'])):
            sb, ch = s.tes['sb'][i] , s.tes['ch'][i] 
            if sb in br_r.keys() and ch in br_r[sb].keys()\
                and 'G' in br_r[sb][ch].keys():
                s.tes['bath_ramp_fit'][i] = True
    if x_lim:
        min_x, max_x = x_lim
    else:
        min_x, max_x = -np.inf, np.inf
    if not bin_size:
        min_all, max_all = min(axis[key]),max(axis[key])
        bin_size = (max_all-min_all)/50
        
    restrictions = ''
    if exclude_unknowns or len(match_list) > 0:
        restrictions = '\n'
    if exclude_unknowns:
        restrictions = restrictions + 'no unknowns '
        if exclude_unknowns == True:
            exclude_unknowns = ['TES_freq','masked',key] # TES_freq already exclude unknowns
    
    if len(match_list) >0:
        restrictions = restrictions + f"[{', '.join([f'{mk}{mt}{mv}' for mk,mt,mv in match_list])}]"
    
    if own_fig == True:
        plt.figure(figsize=default_figsize)
        mplt = plt
    elif own_fig != False: # passed a matplotlib axis
        mplt = own_fig
        
    i=0 # offset the bars a bit to see colors easier; also sets the color
    std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    my_hatches = [hatchy*hatch_amount for hatchy in ['\\','/','|','-']]
    tot_plotted = 0
    num_plotted = {}
    for mask_stat in ['Y','N']:
        if mask_stat == 'N':
            if full_legend_label:
                mtext = 'unmasked'
            else:
                mtext='unmskd'
        elif mask_stat == 'Y':
            if full_legend_label:
                mtext = 'masked'
            else:
                mtext = 'mskd'
        for freq in s.key_info['TES_freq']['vals']:
            if str(type(s)) == str(type(Ramp_Combination.__new__(Ramp_Combination))):
                axis = s.axes[ax_name]
                vals = axis[key][\
                       s.find_idx_matches([('TES_freq','=',freq),
                                           ('masked','=',mask_stat)] + match_list,
                                          ax_name=ax_name,
                                         exclude_unknowns=exclude_unknowns)]
                #(key,'<=', max_x),(key,'>', min_x)
            else: 
                vals = []
                for sb in s.results.keys():
                    for ch, params in s.results[sb].items():
                        if s.mux_map[sb][ch]['TES_freq'] == freq \
                        and s.mux_map[sb][ch]['masked'] == mask_stat:
                            vals.append(params[key])
            tot_plotted += len(vals)
            num_plotted[f"{mtext} {freq}s"] = len(vals)
            if len(vals) == 0:
                continue
            
            offy = i*bin_size*bin_offset
            if x_lim: # consider switching to floor for official publications...
                bins = np.arange(min_x+offy,min_x+offy + bin_size*(np.ceil((max_x-min_x)/bin_size)+2),bin_size)
            else:
                mi_x, ma_x = min(vals),max(vals)
                bins = np.arange(mi_x+offy,mi_x+offy + bin_size*(np.ceil((ma_x-mi_x)/bin_size)+2),bin_size)
            med = np.median(vals)
            med_sig = round_to_sf_str(med,sig_figs) # for display
            stdev = np.std(vals)
            stdev_sig = round_to_sf_str(stdev,sig_figs)
            print(f"{mask_stat}\t{freq}\t{med}\t{stdev}") #\t{len(vals)}
            
            
            if full_legend_label:
                label=f"{mtext} {freq} GHz: med={med_sig} ± std={stdev_sig} [{s.key_info[key]['units']}]"
            else:
                label=f"{mtext} {freq}s; {med_sig}" #±{stdev:.2}
            
            p = mplt.hist(vals, bins=bins,alpha=0.4,color=std_colors[i],
                         label=label,hatch=my_hatches[i],edgecolor=std_colors[i],linewidth=0)
            mplt.axvline(med,color=std_colors[i],linestyle='dashed')
#             else:
#                 own_fig.hist(vals, bins=bins,alpha=0.4,color=std_colors[i],
#                              label=f"{freq} GHz, masked={mask_stat}: med={med:.3} [{s.key_info[key]['units']}]")
#                 own_fig.axvline(med,color=std_colors[i],linestyle='dashed')
            i+=1
    if x_lim:
        plt.xlim(x_lim)
        
    if s.key_info[key]['units'] == '':
        x_ax_label = f"{s.key_info[key]['name']}"
    else:
        x_ax_label = f"{s.key_info[key]['name']} [{s.key_info[key]['units']}]"
    y_ax_label = "number of TESs"
    title = f"{s.dName} {key}{restrictions}"
    mplt.legend()
    print(f"tot plotted: {tot_plotted}")
        
    if own_fig != False: # passed an axis # used to have own_fig !=True as part of it, what?
        mplt.legend()
        try: # I assume it's usually going to be a subplot
            mplt.set_xlabel(x_ax_label) 
            if full_legend_label:
                mplt.set_ylabel(y_ax_label) 
            if x_lim:
                mplt.set_xlim(x_lim)
        except AttributeError: # it's a full plot
            plt.xlabel(x_ax_label)
            if full_legend_label:
                plt.ylabel(y_ax_label)
            plt.title(f"{s.dName} {key}{restrictions}")
            if x_lim:
                mplt.xlim(x_lim)
    return num_plotted

def rc_thermal_summary_plots(s,match_list=[],sig_figs=3,exclude_bath_ramp_fails=False,
                             x_Ranges=[[],[],[],[]], y_Ranges=[[],[],[],[]],
                             bin_sizes=[0.3, 2, 0.01,0.05]):
    """set the x_Ranges to 'default' to get default values for your UFM frequency type."""
    if x_Ranges == 'default': 
        if s.test_device.device_type == "LF_UFM":
            # I'm really not sure about these.
            x_Ranges = [[0,5],[135,190],[0,0.200],[1.5,4.5]]
        if s.test_device.device_type == "MF_UFM":
            x_Ranges = [[0,15],[135,190],[0,0.350],[1.5,4.5]]
        if s.test_device.device_type == "UHF_UFM":
            x_Ranges = [[0,50],[135,190],[0,1.20],[1.5,4.5]]

    #thermal_y_Ranges = [[0,200],[0,160],[0,250],[0,175]]#[[0,110],[0,90],[0,250],[0,175]]
    thermal_params=['p_sat100mK','Tc','G','n']

    fig_t, ax_t = plt.subplots(nrows=4, figsize=(8,9)) 
    for i in range(4):
        num_plotted_dict = key_hist_by_freq_and_mask(s,thermal_params[i], 
                                  bin_size=bin_sizes[i], x_lim=x_Ranges[i],
                                  ax_name='tes',exclude_unknowns=True,own_fig=ax_t[i],
                                  full_legend_label=False, exclude_bath_ramp_fails=False,
                                  match_list=[],sig_figs=sig_figs) 
        if y_Ranges[i]:
            ax_t[i].set_ylim(y_Ranges[i])
        ax_t[i].set_ylabel(" ")
    
    
    restrictions = [f"{mk}{mt}{mv}" for mk,mt,mv in match_list]
    if exclude_bath_ramp_fails:
        restrictions = ["exclude_bath_ramp_fails=True"] + restrictions
    
    title = f"Thermal Parameters of {s.dName} (medians listed in legends)\n"
    if not x_Ranges == [[],[],[],[]]:
        title = title + "plot x_Ranges may hide outliers" 
    if len(restrictions) > 0:
        title = title + f"(restrictions:[{', '.join(restrictions)}])"
    num_str = ""
    tot_plotted = 0
    for key,num in num_plotted_dict.items():
        num_str = num_str + f"{num} {key}; "
        tot_plotted+=num
    if not title[-1] == "\n":
        title = title+"\n"
    num_str = f"numFit={tot_plotted}. " + num_str[:-2]
    title = title + num_str
    plt.suptitle(title)

    plt.tight_layout()
    fig_t.text(0.03,0.5, '------------ # of TESs ------------', ha='center', va='center',rotation='vertical', fontsize=12)
    return ax_t
    

def opt_eff_hack_hist(s, x_lim=[],bin_size=0.1,bin_offset=0.15 ):
    key_hist_by_freq_and_mask(s,'opt_eff_no_subtraction',ax_name='tes',
                              x_lim=x_lim,bin_size=bin_size,bin_offset=bin_offset,
                              exclude_unknowns=['opt_eff_no_subtraction'])


def external_bath_ramp_thermal_summary_plots(s,freq_order, nBins=60, 
                                             x_Ranges=[[],[],[],[]], y_Ranges=[[],[],[],[]]):
    # s is a bath ramp!
    subplot_keys = ['p_sat100mK','Tc','G','n']
    vals = [[s.to_plot_TES_freqs[freq][subplot_key] for freq in freq_order] for subplot_key in subplot_keys]
    return external_thermal_summary_plots(s,freq_order,vals,nBins=nBins,x_Ranges=x_Ranges,y_Ranges=y_Ranges)



    
def external_thermal_summary_plots(s,freq_order, vals, nBins=60, 
                                   x_Ranges=[[],[],[],[]], y_Ranges=[[],[],[],[]]):
    # s for dName and key_info

    subplot_keys = ['p_sat100mK','Tc','G','n'] # need an indexable order for ax[] ref, hence no dictionary

    #freq_order = freq_order_guess
    #vals = [[s.tes[subplot_key][freq_idx[key]] for key in freq_order] for subplot_key in subplot_keys]
    tes_freqs_labels = [str(freq) + " GHz" for freq in freq_order]

    # possibly use the key_info lims or the limit finding function here?
    #x_R = [p_sat_Range,t_c_Range,g_Range,n_Range]
    for ax_i in range(4):
        if not x_Ranges[ax_i]:
            x_Ranges[ax_i] = [min([min(freq_vals) for freq_vals in vals[ax_i]]),
                            max([max(freq_vals) for freq_vals in vals[ax_i]])]
    if len(y_Ranges) == 2:
        # one y_Range for all four
        y_Ranges = [y_Ranges,y_Ranges,y_Ranges,y_Ranges]

    # Begin plottting
    std_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(nrows=4, figsize=(5,9)) # formerly 9,9. 5,10 for ASC
    for ax_i in range(4):
        for freq_i in range(len(tes_freqs_labels)):
            h = ax[ax_i].hist(vals[ax_i][freq_i],  alpha=0.4, bins=nBins, range=x_Ranges[ax_i], label=tes_freqs_labels[freq_i]) # range=(0,12)
            med = np.nanmedian(vals[ax_i][freq_i])
            ax[ax_i].axvline(med, linestyle='--',color=std_colors[freq_i],label=f"median={med:.3} {s.key_info[subplot_keys[ax_i]]['units']}")
        ax[ax_i].set_xlabel(f"{s.key_info[subplot_keys[ax_i]]['name']} [{s.key_info[subplot_keys[ax_i]]['units']}]") # ,fontsize=10 # Minimum DISPLAY fontsize >=10
        #ax[0].set_xlabel('p_sat at 100 mK (pW)',fontsize=10)
        #ax[0].set_ylabel('# of TESs')
        ax[ax_i].set_ylabel(' ') # spacing reasons
        #ax[ax_i].set_title(' ') # needed for spacing reasons
        if y_Ranges[ax_i]:
            ax[ax_i].set_ylim(y_Ranges[ax_i])
        ax[ax_i].legend() #fontsize='small'  #, loc=2
    # y_axis label
    fig.text(0.03,0.5, '------------ # of TESs ------------', ha='center', va='center',rotation='vertical', fontsize=12)
    # title
    # restrictions here. 
    t_cs = vals[1]
    totNumDets = sum([len(dataset) for dataset in t_cs])
    numDets = "(numFit = "+ str(totNumDets) + "; "+ ", ".join([str(len(t_cs[i])) + "x"+ tes_freqs_labels[i][:-4] for i in range(len(tes_freqs_labels))]) + ")"

    # Maybe add something to note if unmasked excluded?
    plt.suptitle(f"{s.dName} " + numDets, fontsize=16)
    plt.tight_layout()
    return ax


def make_thermal_summary_plots(s, nBins=60, p_sat_Range=None,t_c_Range=None,
                               g_Range=None, n_Range=None, 
                               x_Ranges=[[],[],[],[]],y_Ranges=[[],[],[],[]],
                               match_list=[], exclude_unknowns=False):
    to_plot = s.tes
    subplot_keys = ['p_sat100mK','Tc','G','n'] # need an indexable order for ax[] ref, hence no dictionary

    freq_order_guess = [i for i in np.sort(np.array([freq for freq in np.unique(s.tes['TES_freq'])]))]
    if freq_order_guess[0] <= 0:# this is for consistency in frequency color display
        freq_order_guess = freq_order_guess[1:]+[freq_order_guess[0]]

    freq_idx = {freq:s.find_idx_matches([('TES_freq','=',freq),('was_fit','=',True)] + match_list, 
                                       ax_name='tes', exclude_unknowns=['TES_freq']+subplot_keys) \
                for freq in freq_order_guess}
    # make exclude unknowns work
    for freq, idx_list in freq_idx.items():
        if len(idx_list) == 0:
            freq_order_guess.pop(freq_order_guess.index(freq)) 

    freq_order = freq_order_guess
    vals = [[s.tes[subplot_key][freq_idx[key]] for key in freq_order] for subplot_key in subplot_keys]
    if x_Ranges == [[],[],[],[]]:
        x_Ranges = [p_sat_Range,t_c_Range,g_Range,n_Range]
    
    return external_thermal_summary_plots(s,freq_order,vals,nBins=60,x_Ranges=x_Ranges,y_Ranges=y_Ranges)
    


# -------------- Formatted outputs -------------

def kaiwen_opt_calc_prereq(rc,add_temps=[]):
    """takes an rc. use add_temps=[] optional argument to add ex. cl_temp
    during a bath ramp (or two). Kaiwen needs to calculate optical powers."""
    to_ret = {}
    cl_temps = np.array([val for val in np.unique(rc.crv['cl_temp'])] + add_temps)
    cl_temps = [val for val in np.sort(np.unique(cl_temps))]
    print(cl_temps)
    for sb_ch in np.unique(rc.crv['sb_ch']):
        ch = sb_ch % 1000
        sb = int((sb_ch-ch)/1000)
        to_ret[(sb,ch)] = {'T':cl_temps}
    Path(rc.test_device.out_path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(rc.test_device.out_path,f"{make_filesafe(rc.dName)}_cl_temps.npy")
    np.save(filename,to_ret)
    print(f"saved to: {filename}")
    return to_ret

def make_daniel_cl_ramp_output(rc, cl, uxm_id, cooldown_id, filepath='default'):
    """UXM ID is ex Uv36. Cooldown ex. P-H-014"""
    if filepath=='default':
        filepath = f"/data/uxm_results/{uxm_id}/{cooldown_id}/" 
        filename = make_filesafe(f"{uxm_id}_coldload_{str(np.unique(cl.other_temps))[1:-1]}mK_{cl.metadata_fp_arr[0][-12:-4]}")+".npy"
        filepath = make_filesafe(filepath)
    to_ret = {'metadata':{'units': {'temp': 'K', 'psat': 'pW', 'R_n': 'ohm'},
                         'dataset': cl.metadata_fp_arr[0],
                         'allowed_rn': [cl.expected_R_n_min, cl.expected_R_n_max],
                         'cut_increasing_psat': True,
                         'thermometer_id': None,
                         'temp_list': cl.temp_list_raw,
                         'optical_bl': [bly for bly in np.sort(np.array([bl for bl in np.unique(rc.tes['bl']) \
                                                                         if bl not in cl.test_device.masked_biaslines]))],
                         'temp_offset': cl.therm_cal[1],
                         'temp_scaling': cl.therm_cal[0],
                         'temps_to_cut': np.array([None], dtype=object),
                         'p_b_%_n': rc.use_per},
              'data':{}}
    for sb_ch in np.unique(rc.crv['sb_ch']):
        ch = sb_ch % 1000
        sb = int((sb_ch-ch)/1000)
        crv_idxs = rc.find_idx_matches(ml(f"sb_ch={sb_ch}&is_iv=True&ramp_name={cl.ramp_name}"))
        # first the things in daniel's format, then what Kaiwen said she needed,
        cl_temps = [temp for temp in rc.crv['cl_temp'][crv_idxs]]
        p_b90s = [pb for pb in rc.crv[rc.p_bp][crv_idxs]]
        r_ns = [r_n for r_n in rc.crv['R_n'][crv_idxs]]
        if sb not in to_ret['data'].keys():
            to_ret['data'][sb] = {}
        to_ret['data'][sb][ch] = {'temp':cl_temps,
                                  'psat':p_b90s,
                                  'R_n':r_ns}
    #Path(rc.test_device.out_path).mkdir(parents=True, exist_ok=True)
    Path(filepath).mkdir(parents=True, exist_ok=True)
    #filename = os.path.join(rc.test_device.out_path,f"{make_filesafe(rc.dName)}_cl_temps.npy")
    np.save(filepath+filename,to_ret)
    print(f"saved to: {filepath+filename}")


# -------------- ASC graphics on single detector (clean up!!) -------------

def plot_ramp_lines(s,sb,ch,ramp_idxs=[],own_fig=True,ax_name='crv',p_bper='p_b90'):
    axis = s.axes[ax_name]
    if own_fig:
        plt.figure(figsize=default_figsize)
    if ramp_idxs==[]:
        ramp_idxs = np.array(range(len(s.ramps['ramp'])))
    idxs = [s.find_idx_matches([('is_iv','=',True),('sb','=',sb),('ch','=',ch),('ramp_from','=',ramp)],ax_name=ax_name)
            for ramp in s.ramps['ramp']]
    # making grayscale legible
    linestyles = [(0,[1,4]+[1,1]*i) for i in ramp_idxs]
    linestyle_i = 0
    for ramp_i in ramp_idxs:
        ramp = s.ramps['ramp'][ramp_i]
        linestyle = linestyles[linestyle_i]
        if ramp.ramp_type != 'bath':
            linestyle = 'dashed'
            linestyle_i -= 1
        plt.plot(axis['bath_temp'][idxs[ramp_i]],axis[p_bper][idxs[ramp_i]],label=s.ramps['ramp_name'][ramp_i],
                 linestyle=linestyle, linewidth=1.5,color=s.key_info['ramp_name']['colors'][ramp_i])
        linestyle_i+=1

def plot_together_fit_ramps(s,sb,ch,ramp_idxs=[],own_fig=True):
    if own_fig:
        plt.figure(figsize=default_figsize)
    if ramp_idxs==[]:
        ramp_idxs = np.array(range(len(s.ramps['ramp'])))
    idxs = [s.find_idx_matches([('is_iv','=',True),('sb','=',sb),('ch','=',ch),('ramp_from','=',ramp)])
            for ramp in s.ramps['ramp']]
    for ramp_i in ramp_idxs:
        ax = plt.get_gca()
        ax.plot(s.crv['bath_temp'][idxs[ramp_i]],s.crv[f'p_b{s.use_per}'][idxs[ramp_i]],label=s.ramps['ramp_name'][ramp_i],
                 linestyle='dashed',linewidth=1,color=s.key_info['ramp_name']['colors'][ramp_i],zorder=1)




def ramp_underlay_plot(s,sb,ch,ax_name='crv',p_bper='p_b90',
                       color_scale=plt.cm.CMRmap,scale_on='cl_temp'):
    plot_ramp_lines(s,sb,ch,ramp_idxs=[],ax_name=ax_name,p_bper=p_bper)
    plt.legend()
    s.plot_key_v_key_colored_by_key('bath_temp',p_bper,scale_on,
                                  match_list=[('is_iv','=',True),('sb','=',sb),('ch','=',ch)], 
                                  exclude_unknowns=True,ax_name=ax_name, 
                                  plot_args={'s':80,'edgecolors':'k','zorder':2},own_fig=False,
                                 color_scale=color_scale)

    

# -------------- superconducting offset functions (temporary) ---
    
def iva_sc_ITO_data_correction(s, py_ch_iv_analyzed,r_n=-42,set_sc_slope=False):
    # **TODO**: Import my formal slope_fix_correct_one_iv instead!!
    # nb_end_idx exists because I often saw this little downturn in the 
    # iv curve at the very highest v_bias. I'm not sure why, but it's in mv6 too.
    # takes a pysmurf data dictionary for a given (sb,ch), 
    # backcalculates the rawer values, makes corrected & expanded version.
    # Except I am not yet dealing with the responsivity calculations. 
    # And, if r_n provided, forces the norm fit line's slope to be 1/(r_n/r_sh+1)
    # dict_keys(['R' [Ohms], 'R_n', 'trans idxs', 'p_tes' [pW], 'p_trans',\
    # 'v_bias_target', 'si', 'v_bias' [V], 'si_target', 'v_tes_target', 'v_tes' [uV]])
    r_sh = s.r_sh 
    iv = {} # what I will return. iv data
    iv_py = py_ch_iv_analyzed
    # first, things I don't need to change, really:
    # Arguably should copy everything...
    for key in ['trans idxs','v_bias','R_nSM','bl']:
        iv[key] = iv_py[key]
    # I don't think I want to copy the standalone here, honestly...maybe redo.
    # as of 11/10/2021, pysmurf's v_bias_target is bugged, 
    # hence not just copying iv's value for it.

    #  Now, get fundamentals: v_bias_bin, i_bias_bin [uA], and i_tes [uA] (with an offset) = resp_bin  
    # v_tes = i_bias_bin*1/(1/R_sh + 1/R), so i_bias_bin = v_tes*(1/R_sh + 1/R)
    if 'i_bias' in iv_py.keys():
        iv['i_bias'] = iv_py['i_bias']
    else:
        iv['i_bias'] =  iv_py['v_tes']*(1/s.r_sh + 1/iv_py['R'])
    # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
    if 'i_tes' in iv_py.keys():
        resp_bin = iv_py['i_tes']
    else:
        resp_bin = iv['i_bias']/(iv_py['R']/r_sh + 1)

    # fit the superconducting branch: super simplistic version
    sc_idx,nb_idx = iv_py['trans idxs']
    #i_bias_sc = iv['i_bias'][:sc_idx]
    
    sc_start = 0
    # Dealing with negative biases, which may include a second transition!
    neg_idxs = np.where(iv['i_bias'] <0)[0]
    if len(neg_idxs) > 3:
        d_resp = np.diff(resp_bin)
        dd_resp = np.diff(d_resp)
        neg_sc_idx = np.where(min(dd_resp) == dd_resp)[0][0]
        sc_start = neg_sc_idx + int(0.25*(neg_idxs[-1] - neg_sc_idx)) # giving some wiggle room for bad sc_idx selection
        
    # need to think about this fit because it REALLY needs to be dead-on. 
    if set_sc_slope:
        set_slope=1
        if is_float(set_sc_slope):
            set_slope = set_sc_slope*1.0 # ensures it's a float for formatting
        sc_func = lambda i_bias, ITO: set_slope*i_bias + ITO
        (popt, pcov) = curve_fit(sc_func,iv['i_bias'][sc_start:sc_idx-2],\
                          resp_bin[sc_start:sc_idx-2]) # -2 is to back away from any potential messed-up transients
        sc_fit = [set_slope, popt[0]]
    else: # sc_idx often first point AFTER peak; and peak could be binned badly
        sc_fit = np.polyfit(iv['i_bias'][sc_start:sc_idx-2],\
                          resp_bin[sc_start:sc_idx-2],1)

    # we also have to do an R_n fit b/c the function I'm adapting needs one
    # simplistic nb_fit_idx
    if r_n==-42:
        nb_fit_idx=nb_idx + int(0.5*(len(iv['i_bias'])-nb_idx))
        norm_fit = np.polyfit(iv['i_bias'][nb_fit_idx:],\
                                      resp_bin[nb_fit_idx:],1)
        r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n
    iv['R_n'] = r_n

    # TODO: make residual plotter! 
    iv['i_tes'] = resp_bin - sc_fit[1] # there we go. 
    if 'i_tes_offset' in iv_py.keys():
        iv['i_tes_offset'] = iv_py['i_tes_offset'] + sc_fit[1]
    else:
        iv['i_tes_offset'] = sc_fit[1]
    iv['sc_slope'] = sc_fit[0]

    return fill_iva_from_preset_i_tes_i_bias_and_r_n(s,iv_py,iv)

    
def sc_ITO_trial_wrapper(s,sb,ch,set_sc_slope=False): # s an RC
    num_crvs = len(s.find_idx_matches([('sb','=',sb),('ch','=',ch),('is_iv','=',True)],ax_name='crv'))
    sc_ITO = {'ramp_name':np.full((num_crvs,), 'hi',dtype=object),
              'ramp_from':np.full((num_crvs,), 'hi',dtype=object),
              'temp':np.full((num_crvs,), -42,dtype=int),
              'bath_temp':np.full((num_crvs,), -42,dtype=float),
              'cl_temp':np.full((num_crvs,), -42,dtype=int),
              'iva':np.full((num_crvs,), 'hi',dtype=object),
              'sb':np.full((num_crvs,), sb,dtype=int), # necessary so I can attach sc_ITO to a ramp combination and use s.plot_key_v_key_colored_by_key
              'ch':np.full((num_crvs,), ch,dtype=int),
              'is_iv':np.full((num_crvs,), True,dtype=bool),
              'p_b90':np.full((num_crvs,), -42,dtype=float),
              'p_b80':np.full((num_crvs,), -42,dtype=float),
              'p_b70':np.full((num_crvs,), -42,dtype=float),
              'p_b60':np.full((num_crvs,), -42,dtype=float),
              'p_b50':np.full((num_crvs,), -42,dtype=float),
              'p_b40':np.full((num_crvs,), -42,dtype=float),
              'idx':np.arange(num_crvs),
              'i_bias':np.full((num_crvs,),'iva',dtype=object),
              'i_tes':np.full((num_crvs,),'iva',dtype=object),
              'p_tes':np.full((num_crvs,),'iva',dtype=object),
              'R':np.full((num_crvs,),'iva',dtype=object),
              'sc_slope':np.full((num_crvs,), -42,dtype=float),
               } # we will filter by ones that are iv curves
    j=0
    # THIS IS HACKY:
    r_n = -42
    print("\n---- pre-TANO ITO info ----")
    for i in range(len(s.ramps['ramp'])):
        ramp = s.ramps['ramp'][i]
        if ramp.ramp_type=='bath'\
           and sb in ramp.tes_dict.keys() and ch in ramp.tes_dict[sb].keys() \
           and 'R_n' in ramp.tes_dict[sb][ch].keys():
            r_n = ramp.tes_dict[sb][ch]['R_n']
            break
    for i in range(len(s.ramps['ramp'])):
        ramp = s.ramps['ramp'][i]
        ramp_name = s.ramps['ramp_name'][i]
        iva_d = ramp.find_iva_matches([('sb','=',sb),('ch','=',ch),('is_iv','=',True),('temp','=','all')])
        for temp,iva_l in iva_d.items():
            for iva in iva_l:
                sc_ITO['ramp_name'][j] = ramp_name
                sc_ITO['ramp_from'][j]=ramp
                sc_ITO['temp'][j] = temp
                if ramp.ramp_type == 'bath':
                    sc_ITO['bath_temp'][j] = temp
                    sc_ITO['cl_temp'][j] = ramp.other_temps[np.where(np.array(ramp.temp_list_raw) == temp)[0][0]]
                else:
                    sc_ITO['bath_temp'][j] = ramp.other_temps[np.where(np.array(ramp.temp_list_raw) == temp)[0][0]]
                    sc_ITO['cl_temp'][j] = temp
#                r_n = -42
#                 if sb in s.test_device.tes_dict.keys() and ch in s.test_device.tes_dict[sb].keys() \
#                    and 'R_n' in s.test_device.tes_dict[sb][ch].keys():
#                     r_n == s.test_device.tes_dict[sb][ch]['R_n']
                # this DOES NOT change the original iva
                sc_ITO['iva'][j] = iva_sc_ITO_data_correction(ramp, copy.deepcopy(iva),r_n=r_n,
                                                              set_sc_slope=set_sc_slope)
                for key in ['p_b90','p_b40','p_b50','p_b60','p_b70','p_b80','i_bias','i_tes','p_tes','R']:
                    sc_ITO[key][j] = sc_ITO['iva'][j][key]
                j+=1
                print(f"{ramp_name} {temp} iva_ITO:{iva['i_tes_offset']}")
    print("\n---- sc sets ITO results (no TANO) ---- ")
    plt.figure(figsize=default_figsize)
    for i in range(len(sc_ITO['iva'])):
        iva = sc_ITO['iva'][i]
        plt.plot(iva['i_bias'],iva['i_tes'],label=f"{sc_ITO['ramp_name'][i]}, {sc_ITO['temp'][i]}")
        print(f"{sc_ITO['ramp_name'][i]}, {sc_ITO['temp'][i]}: p_b90:{iva['p_b90']:.3}, R_n:{iva['R_n']:.3}, ITO:{iva['i_tes_offset']:.3}, sc_slope:{iva['sc_slope']:.3}")
    plt.title(f"sc sets ITO trial:\n {s.dName} sb{sb} ch{ch}, i_tes vs. i_bias ")
    plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    return sc_ITO


def sc_ITO_plus_cheat_norm_and_transition_offset_wrapper(s,sb,ch,pA_per_phi0=9000000.0,
                                                         set_ITO_plus_TANO=False,set_sc_slope=False): # s an RC
    # set_ITO_and_TANO does the average if True, or the number given if it's a float. 
    sc_ITO =sc_ITO_trial_wrapper(s,sb,ch,set_sc_slope=set_sc_slope)
    #num_crvs = len(s.find_idx_matches([('sb','=',sb),('ch','=',ch),('is_iv','=',True)],ax_name='crv'))
#     sc_ITO = {'ramp_name':np.full((num_crvs,), 'hi',dtype=object),
#               'ramp':np.full((num_crvs,), 'hi',dtype=object),
#               'temp':np.full((num_crvs,), -42,dtype=int),
#               'iva':np.full((num_crvs,), 'hi',dtype=object)}
    # resp = phase_ch * pA_per_phi0/(2.*np.pi*1e6)
    phase_jumps_in_uA = 2*np.pi * pA_per_phi0/(2.*np.pi*1e6)
    print("\n---- sc sets ITO WITH TANO ---- ")
    plt.figure()
    plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    print("ramp_name, temp, p_b90, R_n, ITO, TANO")
    for i in range(len(sc_ITO['iva'])):
        iv = sc_ITO['iva'][i]
        iv_py = copy.deepcopy(iv) # we don't really want to return this one.
        
        iv['i_bias'] =  iv_py['v_tes']*(1/sc_ITO['ramp_from'][i].r_sh + 1/iv_py['R'])
        # R = r_sh * (i_bias_bin/resp_bin - 1), so 1/((R/r_sh+1)/i_bias_bin) = resp_bin
        resp_bin = iv['i_bias']/(iv_py['R']/sc_ITO['ramp_from'][i].r_sh + 1)
        
        # We offset the non-sc_branch by units of phase_jumps_in_uA to get normal branch closer to going through 0.
        sc_idx,nb_idx = iv['trans idxs']
        nb_fit_idx= nb_idx + int(0.5*(len(iv['i_bias'])-nb_idx))
        norm_fit = np.polyfit(iv['i_bias'][nb_fit_idx:],\
                                  resp_bin[nb_fit_idx:],1)
        norm_off = norm_fit[1]
        
        add_jumps = True
        if add_jumps:
            # SEE the Doug Bennett notes on the myth of the normal curve for why this is ceiling, not rounding!
            # Actually, let's try rounding....it didn't help
            trans_and_normal_offset = (np.ceil(norm_off/phase_jumps_in_uA))*phase_jumps_in_uA
            iv['i_tes'][sc_idx:] -= trans_and_normal_offset # No!! Make it an exact jump, not add an exact jump!
            # ^ Subtract might be better...look closely
            iv['TANO'] = trans_and_normal_offset
            
        else: # note: this doesn't quite store the relevant TANO right. 
            sc_start = 0
            # Dealing with negative biases, which may include a second transition!
            neg_idxs = np.where(iv['i_bias'] <0)[0]
            # I should really be doing this WITH the sc_set so I don't have to redo the fit...
            if len(neg_idxs) > 3:
                d_resp = np.diff(resp_bin)
                dd_resp = np.diff(d_resp)
                neg_sc_idx = np.where(min(dd_resp) == dd_resp)[0][0]
                sc_start = neg_sc_idx + int(0.25*(neg_idxs[-1] - neg_sc_idx)) # giving some wiggle room for bad sc_idx selection

            sc_slope, sc_const = np.polyfit(iv['i_bias'][sc_start:sc_idx-2],\
                              resp_bin[sc_start:sc_idx-2],1) # sc_idx often first point AFTER peak; and peak could be binned badly
            peak_i = sc_slope*resp_bin[sc_idx-1]+ sc_const 
            one_jump_c_norm = norm_off + ((peak_i-phase_jumps_in_uA) -resp_bin[sc_idx]) 
            trans_and_normal_offset = (np.ceil(one_jump_c_norm/phase_jumps_in_uA))*phase_jumps_in_uA
            iv['i_tes'][sc_idx:] -= (trans_and_normal_offset-norm_off)
            iv['TANO'] = trans_and_normal_offset-norm_off
        
        if set_ITO_plus_TANO== False:
            iv = fill_iva_from_preset_i_tes_i_bias_and_r_n(sc_ITO['ramp_from'][i],iv_py, iv)
            for key in ['p_b90','p_b40','p_b50','p_b60','p_b70','p_b80','i_bias','i_tes','p_tes','R']:
                sc_ITO[key][i] = iv[key]
        sc_ITO['iva'][i] = iv
            
    # done the calculations? Ok. now:
    if set_ITO_plus_TANO==True: # average ITO + "normal" ?
        aipn = np.average(np.array([sc_ITO['iva'][i]['i_tes_offset']+sc_ITO['iva'][i]['TANO'] 
                                    for i in range(len(sc_ITO['iva']))]))
    elif is_float(set_ITO_plus_TANO):
        aipn = set_ITO_plus_TANO
    if set_ITO_plus_TANO == True or is_float(set_ITO_plus_TANO):
        for i in range(len(sc_ITO['iva'])): 
            iv = sc_ITO['iva'][i]
            diffy = aipn - (iv['i_tes_offset'] +iv['TANO'])
            iv['i_tes'][:] += diffy
            iv['i_tes_offset'] += diffy
            iv_py = copy.deepcopy(iv)
            iv = fill_iva_from_preset_i_tes_i_bias_and_r_n(sc_ITO['ramp_from'][i],iv_py, iv)
            for key in ['p_b90','p_b40','p_b50','p_b60','p_b70','p_b80','i_bias','i_tes','p_tes','R']:
                sc_ITO[key][i] = sc_ITO['iva'][i][key]
            
            
    for i in range(len(sc_ITO['iva'])):
        iv = sc_ITO['iva'][i]
        
        plt.plot(iv['i_bias'],iv['i_tes'],label=f"{sc_ITO['ramp_name'][i]}, {sc_ITO['temp'][i]}")
        
        #r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n
        print(f"{sc_ITO['ramp_name'][i]}, {sc_ITO['temp'][i]}, p_b90:{iv['p_b90']:.3}, R_n:{iv['R_n']:.3}, ITO:{iv['i_tes_offset']:.3}, TANO:{iv['TANO']:.3}, ITO+TANO:{iv['i_tes_offset']+iv['TANO']:.3}")
        #print(f"{sc_ITO['ramp_name'][i]}, {sc_ITO['temp'][i]} {iv['p_b90']:.3}, {iv['R_n']:.3}, {iv['i_tes_offset']:.3}, {iv['TANO']:.3}")
        plt.title(f"sc sets ITO plus_cheat_norm_and_transition_offset:\n {s.dName} sb{sb} ch{ch}, i_tes vs. i_bias ")
    
    return sc_ITO        



#sc_ITO_0_58 = sc_ITO_trial_wrapper(s,0,58)
#sc_ITO_0_58 = sc_ITO_trial_wrapper(s,0,474) #They have the same jump thing
 
# That makes it clear I AM seeing phase jumps. Checked, and I didn't have my jump removal turned on...also it needed some fixing up. 
def sc_ITO_and_offset_hack_check_poster_child_IV(s,sb,ch,y_lim=[0,3], 
                                                 set_ITO_plus_TANO=False,
                                                 set_sc_slope=False):
    sc_ITO = sc_ITO_plus_cheat_norm_and_transition_offset_wrapper(s,sb,ch,
                                                                  set_ITO_plus_TANO=set_ITO_plus_TANO,
                                                                  set_sc_slope=set_sc_slope)

    s.sc_ITO_recent = sc_ITO
    sc_ITO_name = f"sc_ITO_hack_{sb}_{ch}"
    s.axes[sc_ITO_name] = sc_ITO
    setattr(s,sc_ITO_name,sc_ITO)




#     s.plot_key_v_key_grouped_by_key('i_bias','i_tes','ramp_name',match_list=[('sb','=',sb),('ch','=',ch)],
#                                   ax_name=sc_ITO_name,plot_args={'alpha':0.5,'linewidth':0.5})
    r_n_used = sc_ITO['iva'][0]['R_n']
#     #r_n = r_sh*(1/norm_fit[0]-1) # our fit r_n. 
    r_n_slope = 1/(r_n_used/s.r_sh+1) # the IV slope from our r_n
#     #i_bias_max = max(sc_ITO['iva'][0]['i_bias'])
#     plt.plot(sc_ITO['iva'][0]['i_bias'], sc_ITO['iva'][0]['i_bias']*r_n_slope,
#              color='k',linestyle='dashed',linewidth=0.5,
#              label=f"if TES was always normal resistor")
#     plt.legend()
#     plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
#     plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    
#     s.plot_key_v_key_grouped_by_key('p_tes','R','ramp_name',match_list=[('sb','=',sb),('ch','=',ch)],
#                                   ax_name=sc_ITO_name,plot_args={'alpha':0.5,'linewidth':0.5})
#     plt.ylim([0,0.010])

    # s.key_info['p_b60'] = copy.deepcopy(s.key_info['p_b90'] )
    # s.key_info['p_b60']['name'] = 'p_tes_bias at R=60% R_n'
    for per in [90]:#,80,70]:##60,50,40]:
        if f'p_b{per}' not in s.key_info.keys():
            s.key_info[f'p_b{per}'] = {'name': '$P_{b90}$, the $P_{bias}$ at R=90%$R_{n}$'.replace('90',str(per)),
                                         'ax_name': 'crv',
                                         'type': 'c',
                                         'units': 'pW'}
            # not giving lim and extremes unknown, because they are unknown
        ramp_underlay_plot(s,sb,ch,ax_name=sc_ITO_name,p_bper=f'p_b{per}')
        plt.ylim(y_lim)
    
    # what is going on with the bath_ramp....
    bt0_name = s.ramps['ramp_name'][0]
    # These should really be colored by key, but that isn't setup to deal with array x/y keys yet
    s.plot_key_v_key_grouped_by_key('i_bias','i_tes','bath_temp',
                                    match_list=[('sb','=',sb),('ch','=',ch),
                                                ('ramp_name','=',bt0_name)],
                                    ax_name=sc_ITO_name,
                                    plot_args={'alpha':0.5,'linewidth':0.75}) # ,xy_overlap_offset=0
    plt.plot(sc_ITO['iva'][0]['i_bias'], sc_ITO['iva'][0]['i_bias']*r_n_slope,
             color='k',linestyle='dashed',linewidth=0.5,
             label=f"if TES was always normal resistor")
    plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    plt.legend()
    
    
#     s.plot_key_v_key_grouped_by_key('i_bias','i_tes','bath_temp',
#                                     match_list=[('sb','=',sb),('ch','=',ch),
#                                                 ('ramp_name','=',bt0_name)],
#                                   ax_name=sc_ITO_name,
#                                     plot_args={'alpha':0.5,'linewidth':0.75}) # ,xy_overlap_offset=0
#     plt.ylim([0,60])
#     plt.plot(sc_ITO['iva'][0]['i_bias'], sc_ITO['iva'][0]['i_bias']*r_n_slope,
#              color='k',linestyle='dashed',linewidth=0.5,
#              label=f"if TES was always normal resistor")
#     plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
#     plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
#     plt.legend()
    
    s.plot_key_v_key_grouped_by_key('p_tes','R','bath_temp',match_list=[('sb','=',sb),('ch','=',ch),
                                                                        ('ramp_name','=',bt0_name)],
                                  ax_name=sc_ITO_name,plot_args={'alpha':0.5,'linewidth':0.75}) # ,xy_overlap_offset=0
    plt.ylim([0.007,0.009])
    plt.hlines(r_n_used,0,max(sc_ITO['iva'][0]['p_tes']),
               color='k',linestyle='dashed',linewidth=0.75,
               label=f"R_n: {sc_ITO['iva'][0]['R_n']}")
    
    s.plot_key_v_key_grouped_by_key('p_tes','R','bath_temp',match_list=[('sb','=',sb),('ch','=',ch),
                                                                        ('ramp_name','=',bt0_name)],
                                  ax_name=sc_ITO_name,plot_args={'alpha':0.5,'linewidth':0.75}) # ,xy_overlap_offset=0
    plt.ylim([0,0.010])
    
    
    return sc_ITO_name
    

def sc_and_TANO_check(s,sb,ch):
    s.ramps['ramp'][0].plot_det_IVs(sb,ch)
    plt.axhline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    plt.axvline(0,linestyle='--',linewidth=1,color='k',alpha=0.5)
    #plt.show()
    check_sc_idx(s,sb,ch)
    print(" ================ base_TANO ================ ")
    return base_TANO(s,sb,ch,p_sat_lim= [0,10])
    
def base_TANO(s,sb,ch,p_sat_lim=[]): #,idx_below_zero='default'
    sc_ITO_name = sc_ITO_and_offset_hack_check_poster_child_IV(s,
                             sb,ch,y_lim=p_sat_lim,set_ITO_plus_TANO=False) #, set_sc_slope=1.0
    print("-------- slope-fix version ---------- ")
    s.plot_key_v_key_grouped_by_key('p_tes','R','bath_temp',
                                    match_list=[('sb_ch','=',sb*1000+ch),
                                                 ('ramp_name','=',s.ramps['ramp_name'][0])],
                                    y_lim=[0,0.010],#x_lim=[0,4],
                                    plot_args={'alpha':0.5,'linewidth':0.75})
#     print("---------- ITO and TANO ---------")
#     s.plot_key_v_key_grouped_by_key('p_tes','R','bath_temp',
#                                     match_list=[('ramp_name','=',s.ramps['ramp_name'][0])],
#                                     y_lim=[0,0.010], #x_lim=[0,4],
#                                     ax_name=sc_ITO_name, #f'sc_ITO_hack_{sb}_{ch}'
#                                     plot_args={'alpha':0.5,'linewidth':0.75})
    
def check_sc_idx(s,sb,ch,transient_idxs=[]):
    s.plot_key_v_key_grouped_by_key('i_bias','i_tes','bath_temp',
                                    match_list=ml(f"sb={sb}&ch={ch}&is_iv=True"),
                                                         x_lim=[],y_lim=[],
                                                         exclude_unknowns=['i_bias','i_tes'],
                                      ax_name='crv',plot_args={'alpha':1,
                                                               'linewidth':0.1,#'linestyle':'None',
                                                               'marker':'.','markersize':1})

    bt_colors=s.key_info['bath_temp']['colors']
    bt_options = s.key_info['bath_temp']['vals']
    x_lim=[100000,-1000000]
    y_lim=[100000,-1000000]
    
    
    my_matches = s.find_idx_matches(f"sb={sb}&ch={ch}&is_iv=True",
                                    exclude_unknowns=['i_bias','i_tes'])
    for idx in my_matches:
        color=bt_colors[bt_options == s.crv['bath_temp'][idx]]
#         p = plt.plot(s.crv['i_bias'][idx],s.crv['i_tes'][idx],
#                      linestyle='None',marker='.',markersize=0.5
#                      color=color,label=f"{s.crv['bath_temp'][idx]} mK" )
        sc_idx = s.crv['sc_idx'][idx]
        plt.plot(s.crv['i_bias'][idx][sc_idx],
                 s.crv['i_tes'][idx][sc_idx],
                 markersize=3, marker='+',alpha=1)
        
        if 'contextless_idxs' not in s.crv['iva'][idx].keys():
            continue
        sc_cii,nb_cii = s.crv['iva'][idx]['contextless_idxs']
        plt.plot(s.crv['i_bias'][idx][sc_cii],
                 s.crv['i_tes'][idx][sc_cii],
                 markersize=3, marker='x',alpha=0.5)
        if (s.crv['i_tes'][idx][sc_cii]-s.crv['i_tes'][idx][sc_cii+1]) \
           < (s.crv['i_tes'][idx][sc_cii+1]-s.crv['i_tes'][idx][sc_cii+2]):
            transient_idxs.append(idx)
        
        x_lim[0] = min(x_lim[0],s.crv['i_bias'][idx][sc_idx],s.crv['i_bias'][idx][sc_cii])
        x_lim[1] = max(x_lim[1],s.crv['i_bias'][idx][sc_idx],s.crv['i_bias'][idx][sc_cii])
        y_lim[0] = min(y_lim[0],s.crv['i_tes'][idx][sc_idx],s.crv['i_tes'][idx][sc_cii])
        y_lim[1] = max(y_lim[1],s.crv['i_tes'][idx][sc_idx],s.crv['i_tes'][idx][sc_cii])
        
    x_lim = [x_lim[0] - 0.1*(x_lim[1]-x_lim[0]), x_lim[1] + 0.1*(x_lim[1]-x_lim[0])]
    y_lim = [x_lim[0] - 0.1*(y_lim[1]-y_lim[0]), y_lim[1] + 0.1*(y_lim[1]-y_lim[0])]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    return transient_idxs


# ==============================================================================
# ----------- Generic dict-ax functions, should be in dif. file ----------------
# ==============================================================================   



def find_idx_matches(ax,match_list,dict_return=False, exclude_unknowns=False):
    '''Returns what indices of the ax's arrays match the match_list criteria.
        See Temp_Ramp.find_iva_matches() and str_to_match_list() aka ml()
        set exclude_unkowns to a list of axis keys. It will exclude
        any idxs that have a -42, np.nan, "-", or "?" as that key's value.
        See below this class for the general implementation.'''
    # like find_ivas, but returns idxs. 
    # find the matches
    return pu.dax.find_idx_matches(ax,match_list,dict_return=dict_return, exclude_unknowns=exclude_unknowns)
#     idx_list = ax['idx']
#     if type(match_list) == str:
#         match_list = str_to_match_list(match_list)
#     if exclude_unknowns:
#         if type(exclude_unknowns) == list:
#             by_keys = exclude_unknowns   
#         else:
#             by_keys = [exclude_unknowns]
#         for by_key in by_keys:
#             match_list = match_list + [(by_key,'!=',-42),(by_key,'!=',-42.0),
#                                        (by_key,'!=',-4.20),
#                                        (by_key,'!=',np.nan),(by_key,'!=','-'),
#                                        (by_key,'!=','?')] #np.nan uses special check from this
#     for match in match_list:
#         idx_list = apply_match(ax,idx_list,match)
#     # organization if necessary
#     num_levels=0
#     level_list = []
#     for match_key,match_type,match_val in match_list:
#         if match_type == "=" and (match_val in ['all','any'] or type(match_val) == list):
#             num_levels +=1
#             level_list.append((match_key,match_type,match_val))
#     if not dict_return or num_levels == 0:
#         return idx_list
#     to_return = {}
#     return add_dict_level(ax,idx_list,level_list,to_return)

        
def add_dict_level(ax,idx_list,level_list,d):
    if len(level_list) == 0: 
        return idx_list
    match_key,match_type,match_val = level_list[0]
    d_keys = np.unique(ax[match_key][idx_list])
    for key in d_keys:
        d[key] = add_dict_level(ax,idx_list[np.where(ax[match_key][idx_list] == key)[0]],
                                    level_list[1:],{})
    return d

def match_mask(ax,idx_list,match):  
    match_key,match_type,match_val = match
    if type(match_val) == list or type(match_val) == type(np.arange(0,2)):
        idx_list_mask = np.full((len(idx_list),), False,dtype=bool)
        for mv in match_val:
            idx_list_mask[single_match_mask(ax,idx_list,(match_key,match_type,mv))==True] = True
    else:
        idx_list_mask = single_match_mask(ax,idx_list,match)
    return idx_list_mask

def single_match_mask(ax,idx_list,match):
    match_key,match_type,match_val = match
    # np.isnan() crashes if given object or str inputs, you've got to be kidding me  
    # I should consider importing pandas for this. Honestly pandas in general sounds SUPER 
    # useful. 
    match_val_is_nan = False
    try:
        if np.isnan(match_val):
            match_val_is_nan = True
    except TypeError:
        pass

    if match_val_is_nan:
        try:
            return np.isnan(ax[match_key][idx_list]) 
        except TypeError: # at least some aren't np.nan. Unfortunately we need to check individually.
            return np.array([False if not type(val) == float \
                         else np.isnan(val) for val in ax[match_key][idx_list]])  

    # Ugh. now that np.nan is handled, we can dothe normal stuff. 
    idx_list_mask = np.full((len(idx_list),), False,dtype=bool)
    # numpy can't do elementwise comparisons of string to array of non-string
    # fortunately if we do detect that situation, the mask is just all False anyway
    if not ((type(match_val) == str or type(match_val) == np.str_) \
            and not (type(ax[match_key][0]) == np.str_ or
                     type(ax[match_key][0]) == str)):
        idx_list_mask[np.where(ax[match_key][idx_list] == match_val)[0]] = True        
    return idx_list_mask


def apply_match(ax,idx_list,match):
    match_key,match_type,match_val = match 
    if match_type == '=' or match_type == '!=':
        if match_val in ['all','any'] \
        or (type(match_val) == list and len(match_val)==1 and
            (match_val[0] == 'all' or match_val[0]=='any')):
            if match_type == '!=':
                return np.array([])
            return idx_list
        else: 
            idx_list_mask = match_mask(ax,idx_list,match)
            if match_type == "!=":
                return idx_list[idx_list_mask == False]
            return idx_list[idx_list_mask]
    # there should not be any np.nan match vals on these match_types
    #assert not np.isnan(match_val), f"can't use 
    if match_type == '<':
        return idx_list[np.where(ax[match_key][idx_list] < match_val)[0]]
    if match_type == '<=':
        return idx_list[np.where(ax[match_key][idx_list] <= match_val)[0]]
    if match_type == '>=':
        return idx_list[np.where(ax[match_key][idx_list] >= match_val)[0]]
    if match_type == '>':
        return idx_list[np.where(ax[match_key][idx_list] > match_val)[0]]
    
# ==============================================================================
# ----------- Loading function big enough to not be at top ---------------------
# ==============================================================================   



# copied from https://github.com/simonsobs/sodetlib/blob/master/sodetlib/analysis/det_analysis.py
# except the sc_offset argument added. 
# Oh, and I'm reducing how much it prints to terminal, because geez.
# Once contextless_is_IV is incorporated into the actual sodetlib, expect this to be unnecessary. 
def analyze_iv_info_no_sc_offset(iv_info_fp, timestamp, phase, v_bias, mask,
                    phase_excursion_min=3.0, psat_level=0.9, 
                                 sc_offset=True,bin_tod=True,save_iv_info_fp=''):
    """
    Analyzes an IV curve that was taken using original sodetlib's take_iv function,
    detailed in the iv_info dictionary. Based on pysmurf IV analysis.
    Does not require pysmurf, just the data and corresponding metadata.
    Args
    ----
    iv_info_fp: str
        Filepath to the iv_info.npy file generated when the IV
        was taken.
    timestamp: numpy.ndarray
        Array containing the timestamp associated with each unbinned point,
        in seconds (unfortunately they are taken faster than seconds...).
    phase: numpy.ndarray
        Array containing the detector response in units of radians.
    v_bias: numpy.ndarray
        Array containing the bias line information in volts.
    mask: numpy.ndarray
        Array containing the band and channel assignments for any
        active resonator channels.
    phase_excursion_min: float
        Default 3.0. In radians, the minimum response a channel must
        have to be considered a detector coupled to the bias line.
    psat_level: float
        Default 0.9. Fraction of R_n to calculate Psat.
    Returns
    -------
    iv_full_dict: dict
        Dictionary containing analyzed IV data, keyed by band
        and channel. Also contains the iv_info dict that was
        used for the analysis.
    """

    iv_info = np.load(iv_info_fp, allow_pickle=True).item()

    R_sh = iv_info['R_sh']
    pA_per_phi0 = iv_info['pA_per_phi0']
    bias_line_resistance = iv_info['bias_line_resistance']
    high_current_mode = iv_info['high_current_mode']
    high_low_current_ratio = iv_info['high_low_ratio']
    #print(f"{high_current_mode} {iv_info['high_low_ratio']}")
    bias_group = np.atleast_1d(iv_info['bias group'])

    iv_full_dict = {'metadata': {}, 'data': {}}

    iv_full_dict['metadata']['iv_info'] = iv_info
    iv_full_dict['metadata']['iv_info_fp'] = iv_info_fp
    if iv_info['wafer_id'] is not None:
        iv_full_dict['metadata']['wafer_id'] = iv_info['wafer_id']
    else:
        iv_full_dict['metadata']['wafer_id'] = None
    iv_full_dict['metadata']['version'] = 'v2'

    bands = mask[0]
    chans = mask[1]
    
    phase_exc_count = 0
    neg_normal_count = 0
    
    # switching the polarity tracker
    swapped_pol = []
    zero_v_bias_vs_ITO = np.full((len(chans),2),(-42.0,-42.0),dtype=float)
    for c in range(len(chans)):
#         if not ((bands[c] == 5 and chans[c]==380) \
#                 or (bands[c]==0 and chans[c]==58)):
#             continue

        phase_ch = phase[c]
        
        phase_exc = np.ptp(phase_ch)

        if phase_exc < phase_excursion_min:
            phase_exc_count +=1
#             print(f'Phase excursion too small.'
#                   f'Skipping band {bands[c]}, channel {chans[c]}')
            continue
        
        # SPECIAL CUT OUT 2Pi jumps --- RITA MODIFICATION!
        # not currently on
        # should really use np.diff for this, instead of for loops. 
        # **TODO**: go through Doug Bennett notes in detail and get this really working!
#         machine_epsilon = np.finfo(type(phase_ch[0])).eps # I originally had <=0, instead of machine_epsilon...
#         phase_diff = np.diff(phase_ch) #0.0005: no phase jumps
#         phase_jumps = np.where( (abs(phase_diff) > np.pi) & (abs(np.pi-(phase_diff % (2*np.pi))) > 3))[0] #2*machine_epsilon
#         jump_record = np.full((len(phase_ch),),0,dtype=float)
#         #if len(phase_jumps) > 0:
#         print(f"{bands[c]:<2} {chans[c]:<3} avg_dif{np.average(phase_diff):.5f} max{max(phase_diff):.5f} ph_jumps{len(phase_jumps)}")
#         for jump_i in phase_jumps:
#             jump_record[jump_i] = phase_diff[jump_i] # issue that I need to 'bin' these
#             #print(f"{bands[c]:<2} {chans[c]:<3} idx{jump_i:<5} phase_jump{phase_ch[jump_i+1]-phase_ch[jump_i]:.2f} mod{(phase_ch[jump_i+1]-phase_ch[jump_i]) % (2*np.pi):.5f}")
            
                
#             phase_ch[:jump_i+1] += phase_ch[jump_i+1]-phase_ch[jump_i] # **TODO**: Want really to look
#             # at slopes on either side, not just set them equal. Problem is, there might be jumps on either side.
#         # SO: do it after the initial push. 
#         for jump_i in phase_jumps: # now let's 
#             surround_idxs = [-42,jump_i,-42,-42] # trying to get two on either side that aren't jumpsthemselves.
#             l_i = jump_i-1
#             while l_i>=0 and surround_idxs.count(-42)>=3:
#                 if not l_i in phase_jumps:
#                     surround_idxs[0] = l_i
# #                     if surround_idxs.count(-42) == 4:
# #                         surround_idxs[1] = l_i
# #                     else:
# #                         surround_idxs[0] = l_i
#                 l_i -=1
#             if surround_idxs.count(-42)>=3:
#                 print(f"{bands[c]:<2} {chans[c]:<3} could not find non-jump phase_idxs below{jump_i}")
#                 continue
#             h_i = jump_i+1
#             while h_i<len(phase_ch) and surround_idxs.count(-42)>0:
#                 if not h_i in phase_jumps:
#                     if surround_idxs.count(-42) == 2:
#                         surround_idxs[2] = h_i
#                     else:
#                         surround_idxs[3] = h_i
#                 h_i +=1
#             if surround_idxs.count(-42)>0:
#                 #print(surround_idxs)
#                 print(f"{bands[c]:<2} {chans[c]:<3} could not find non-jump phase_idxs above {jump_i}")
#                 continue
            
            
                
        
#         for i in range(1,len(phase_ch)):
#             if (phase_ch[i]-phase_ch[i-1]) > (np.pi) and (phase_ch[i]-phase_ch[i-1]) % (2*np.pi) <= 0: 
                
#                 print(f"{bands[c]:<2} {chans[c]:<3} idx{i:<5} phase_jump{phase_ch[i]-phase_ch[i-1]:.2f} mod{(phase_ch[i]-phase_ch[i-1]) % (2*np.pi):.5f}")
#                 phase_ch[:i] += phase_ch[i]-phase_ch[i-1]
        
        

        # assumes biases are the same on all bias groups
        v_bias_bg = v_bias[bias_group[0]]
        v_bias_bg = np.abs(v_bias_bg)
        
        #print(pA_per_phi0)
        resp = phase_ch * pA_per_phi0/(2.*np.pi*1e6)  # convert phase to uA
        
        r_inline = bias_line_resistance
        
        if high_current_mode:
            r_inline /= high_low_current_ratio
        i_bias = 1.0E6 * v_bias_bg / r_inline
        
        zero_bias_resp_values = np.nan # there were no 0-points
        if bin_tod:
            # RS: fewer allocations to catch these cases:
            # RS addition: without the +1 here below, v_bias_bg[step_loc[2]] is the value of 
            # v_bias_bg in the 1st step (except for the 1st step's 1st point)
            vb_changes = np.where(np.diff(v_bias_bg))[0] + 1
            last_step_end = []
            if v_bias_bg[-1] == v_bias_bg[-2]:
                last_step_end = [len(v_bias_bg)-1]
            step_loc = np.concatenate( ([0],vb_changes,last_step_end) )
            
#             step_loc = step_loc+1
#             if step_loc[0] != 0: # this was basically always true even before the +1.
#                 step_loc = np.append([0], step_loc)  # starts from zero
#             if step_loc[-1] != len(v_bias_bg)-1:
#                 step_loc = np
#             # RS: Have to add the last point so the j+1 in the binning below gets it, IF the last
#             # index wasn't different from the second-to-last
            
            n_step = len(step_loc) -1 # Now that we add the last index to step_loc, this is actually correct.
            
            
            # Want to remove V_bias values with so few points they couldn't have possibly
            # finished transient. For some reason pysmurf has a lot of <5 datum V_bias steps,
            # especially 1-data-point V_bias steps
            
            step_sizes = np.diff(step_loc)
            t_mod = sum(step_sizes < 5)

            
            # 0 v_bias measurements pollutes our data with np.nans. Get rid of it. 
            # also,the last "step" is actually just an end of a step for for loop convenience.
            # don't include it. (I'm fine with losing an actual single-point change at the end,
            # single point isn't going to be good anyway).
            zzzs = np.where(v_bias_bg[step_loc][:-1] == 0.0)[0] # [:-1] because the last point in step_loc is an addendum made to make the "x+1" thing work, not an actual start of a step itself. So would overcount if that happens to be the end of the V_bias=0 step (it is if not going negative)
            z_mod = len(zzzs)
            #z_mod = 0
            #print(f"{len(v_bias_bg)} {zzzs}  {v_bias_bg[step_loc[zzzs[0]]:step_loc[zzzs[0]+1]]}")
            if len(zzzs) > 0: # I am going to assume only one 0 in this. for now.
                z_idx_unbin = zzzs[0]         
            
            # Have to check we didn't double count anything. 
            for step_loc_idx in zzzs:
                if step_loc_idx == len(step_loc)-1:
                    continue # THis shouldn't happen, but just in case.
                if step_loc[step_loc_idx+1]-step_loc[step_loc_idx] < 5:
                    z_mod -= 1
                    
                    
            num_bins = n_step-z_mod-t_mod
            # arrays for holding response, I, and V
            resp_bin = np.zeros(num_bins)
            resp_stdev_bin = np.zeros(num_bins) 
            v_bias_bin = np.zeros(num_bins)
            i_bias_bin = np.zeros(num_bins)
            timestamp_out = np.zeros(num_bins)
            
            
            # Find steps and then calculate the TES values in bins
            i=0
            for j in np.arange(n_step):
                s = step_loc[j]
                e = step_loc[j+1]
                
                st = e - s
                sb = int(s + np.floor(st/2))
                eb = int(e - np.floor(st/10)) # I think this was because used to have some of next bracket in each step?
                
                if v_bias_bg[s] == 0.0:# in v_bias_bg[s:e]: # should probably be just the start...
                    zero_v_bias_vs_ITO[c,0] = np.mean(resp[sb:eb]) # so yes, only get the last one.
                    zero_bias_resp_values = resp[s:e]
                    continue # skip. 0s are just too annoying.
                if e-s < 5:
                    continue # Too small, just a transient
                
                try: 
                    resp_bin[i] = np.mean(resp[sb:eb])
                except IndexError:
                    print(v_bias_bg[sb:])
                    print(step_loc[i:])
                    print(f"{zzzs}; {len(step_loc)} {n_step};  j:{j},i:{i} / {len(resp_bin)}; {sb} {eb} / {len(resp)} ")
                resp_stdev_bin[i] = np.std(resp[sb:eb])
                v_bias_bin[i] = v_bias_bg[sb]
                i_bias_bin[i] = i_bias[sb]
                timestamp_out[i] = np.mean(timestamp[sb:eb])
                i+=1
            
        else: # no binning, use all the raw TOD:
            # you do have to remove zeroes for the fitting unfortunately. 
            z_mask = v_bias_bg != 0
            zero_v_bias_vs_ITO[c,0] = np.mean(resp[z_mask == 0])
            zero_bias_resp_values = resp[z_mask == 0]
            #z_mask = np.array([True]*len(v_bias_bg))
            #zero_v_bias_vs_ITO[c,0] = -42
            
            resp_bin = resp[z_mask]
            v_bias_bin = v_bias_bg[z_mask]
            i_bias_bin = i_bias[z_mask]
            
            timestamp_out = timestamp[z_mask]
                        
            n_step = len(resp)  
            resp_stdev_bin = np.zeros(n_step)
            
            # I don't remember where I was going with this.
            step_dif = n_step - (len(np.where(np.diff(v_bias_bg))[0]) -1)
#             if step_dif > 0:
#                 print(f"{bands[c]:<2} {chans[c]:<3} {step_dif}")
        
        if 0.0 in v_bias_bin:
            print("0 escaped!")
        d_resp = np.diff(resp_bin)
        d_resp = d_resp[::-1]
        dd_resp = np.diff(d_resp)
        v_bias_bin = v_bias_bin[::-1]
        i_bias_bin = i_bias_bin[::-1]
        resp_bin = resp_bin[::-1]
        resp_stdev_bin = resp_stdev_bin[::-1]
        timestamp_out = timestamp_out[::-1]
        
        
        # pysmurf for some reason stores negative input volts/bias as positive. Fix that.
        # However, remember it actually stores negative volts as positive for some reason.
        pvb_idx = 0 # positive_volt_bias. Really first non-negative voltage bias index
        d_bias = np.diff(v_bias_bin)
        negs = np.where(d_bias < 0)[0]
        if len(negs) > 0:
            pvb_idx = negs[-1] + 2 # yup, +2. One for diff's contraction, second to get positive
        
        if pvb_idx > 0:
            v_bias_bin[:pvb_idx] *= -1.0
            i_bias_bin[:pvb_idx] *= -1.0

        # PROBLEMS FROM THIS FITTING SEEM TO COME FROM HOW IT FINDS
        # SC IDX AND NB IDX
        
        # index of the end of the superconducting branch
        dd_resp_abs = np.abs(dd_resp)
        # If going into negative volts, don't include in the sc_idx search! 
        # TODO: SHOULD THERE BE A +1 THERE? 
        # +1 is for the dd_ correction and to get the sc_idx on the point on the right side of things.
        sc_idx = np.ravel(np.where(dd_resp_abs[pvb_idx:] == np.max(dd_resp_abs[pvb_idx:])))[0] +1 + pvb_idx
        if sc_idx == pvb_idx: # have to have at least some superconducting branch for later
            sc_idx = pvb_idx + 1
        # not sure how this would trigger, but:
        if sc_idx == 0:
            sc_idx = 1
               
        
        
        # index of the start of the normal branch
        # default to partway from beginning of IV curve
        nb_idx_default = int(0.8*n_step)
        nb_idx = nb_idx_default
        for i in np.arange(nb_idx_default, sc_idx, -1):
            # look for minimum of IV curve outside of superconducting region
            # but get the sign right by looking at the sc branch
            if d_resp[i]*np.mean(d_resp[:sc_idx]) < 0.:
                nb_idx = i+1
                break

        nb_fit_idx = int(np.mean((n_step, nb_idx)))
        norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:],
                              resp_bin[nb_fit_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            swapped_pol.append((bands[c],chans[c]))
            #print(f"Flipped polarity of \t{bands[c]}\t{chans[c]}")
            norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:],
                                  resp_bin[nb_fit_idx:], 1)
        
        # doing w/out i_tes_offset messes up a LOT of stuff
        resp_bin -= norm_fit[1]  # now in real current units
        
        
        if not zero_v_bias_vs_ITO[c,0] == -42.0:
            zero_v_bias_vs_ITO[c,1] == norm_fit[1]
            
        # ADDING: I want to save the unbinned around sc_idx.
        if bin_tod:
            sc_step_st = np.where(v_bias_bg[step_loc] == v_bias_bin[sc_idx])[0][0]
            sc_unbin_i_bias = i_bias[step_loc[max(sc_step_st-2,0)]:
                                              step_loc[min(sc_step_st+3,len(step_loc)-1)]][::-1]
            sc_unbin_i_tes = resp[step_loc[max(sc_step_st-2,0)]:
                                              step_loc[min(sc_step_st+3,len(step_loc)-1)]][::-1] - norm_fit[1]
        
        
        
        # I guess I'd put sc_fit plus TANO here?
        # added the pvb_idx for now!
        sc_fit = np.polyfit(i_bias_bin[pvb_idx:sc_idx], resp_bin[pvb_idx:sc_idx], 1)

        # I GOT RID OF THIS, HAAAA!!!!
        # subtract off unphysical y-offset in superconducting branch; this is
        # probably due to an undetected phase wrap at the kink between the
        # superconducting branch and the transition, so it is *probably*
        # legitimate to remove it by hand. We don't use the offset of the
        # superconducting branch for anything meaningful anyway. This will just
        # make our plots look nicer.
        if sc_offset:         
            resp_bin[:sc_idx] -= sc_fit[1]
            sc_fit[1] = 0  # now change s.c. fit offset to 0 for plotting
        
        R = R_sh * (i_bias_bin/(resp_bin) - 1)
        R_n = np.mean(R[nb_fit_idx:])
        R_L = np.mean(R[1:sc_idx])

        if R_n < 0:
            neg_normal_count +=1
#             print(f'Fitted normal resistance is negative. '
#                   f'Skipping band {bands[c]}, channel {chans[c]}')
            #continue
        
        # RS: changed the below two to deal with i_bias=0 case better
        # They are identical to the originals in every other case.
        #v_tes = i_bias_bin*R_sh*R/(R+R_sh)  # voltage over TES
        v_tes = resp_bin*R 
        #i_tes = v_tes/R  # current through TES
        i_tes = resp_bin 
        p_tes = (v_tes**2)/R  # electrical power on TES

        # calculates P_sat as P_TES at 90% R_n
        # if the TES is at 90% R_n more than once, just take the first crossing
        level = psat_level
        cross_idx = np.where(np.logical_and(R/R_n - level >= 0,
                             np.roll(R/R_n - level, 1) < 0))[0]

        # Taking the first crossing
        if len(cross_idx) >= 1:
            cross_idx = cross_idx[-1]
            if cross_idx == 0:
                print(f'Error when finding {100*psat_level}% Rfrac for channel '
                      f'{(bands[c], chans[c])}. Check channel manually.')
                cross_idx = -1
                p_sat = np.nan
            else:
                p_sat = interp1d(R[cross_idx-1:cross_idx+1]/R_n,
                                p_tes[cross_idx-1:cross_idx+1])
                p_sat = p_sat(level)
        else:
            cross_idx = -1
            p_sat = np.nan

        smooth_dist = 5
        w_len = 2*smooth_dist + 1

        # Running average
        w = (1./float(w_len))*np.ones(w_len)  # window
        i_tes_smooth = np.convolve(i_tes, w, mode='same')
        v_tes_smooth = np.convolve(v_tes, w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth))*R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx]/di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        beta = 0.

        # artificially setting rL to 0 for now,
        # to avoid issues in the SC branch
        # don't expect a large change, given the
        # relative size of rL to the other terms

        rL = 0

        # Responsivity estimate, derivation done here by MSF
        # https://www.overleaf.com/project/613978cb38d9d22e8550d45c

        si = -(1./(i0*r0*(2+beta)))*(1-((r0*(1+beta)+rL)/(dv_tes/di_tes)))

        iv_dict = {}
        iv_dict['R'] = R
        iv_dict['R_n'] = R_n
        iv_dict['idxs'] = np.array([sc_idx, nb_idx, cross_idx])
        iv_dict['p_tes'] = p_tes
        iv_dict['p_sat'] = p_sat
        iv_dict['v_bias'] = v_bias_bin
        iv_dict['v_tes'] = v_tes
        iv_dict['i_tes'] = i_tes
        iv_dict['si'] = si
        # RS: added these
        iv_dict['pA_per_phi0'] = pA_per_phi0 # for looking at tod later.
        iv_dict['i_tes_offset'] = norm_fit[1]
        iv_dict['pvb_idx'] = pvb_idx # index of first non-negative bias value
        if bin_tod:
            iv_dict['i_tes_stdev'] = resp_stdev_bin # too slow.
            iv_dict['i_bias_unbin_sc'] = sc_unbin_i_bias
            iv_dict['i_tes_unbin_sc'] = sc_unbin_i_tes
        else:
            iv_dict['timestamp'] = timestamp_out
        iv_dict['zero_bias_resp_values'] = zero_bias_resp_values

        iv_full_dict['data'].setdefault(bands[c], {})
        iv_full_dict['data'][bands[c]][chans[c]] = iv_dict
        
    abs_0_dif = sum(abs(zero_v_bias_vs_ITO[:,1]-zero_v_bias_vs_ITO[:,0])) \
                /(len(chans)-len(zero_v_bias_vs_ITO[:,0][zero_v_bias_vs_ITO[:,0]==-42.0]))
    
    print(f"Skipped {phase_exc_count} chs w/ phase excursion <{phase_excursion_min}. Saw {neg_normal_count} chs w/ neg normal fit. Avg. abs(0 bias resp-c_norm) {abs_0_dif:.2}") #pA_per_phi0: {pA_per_phi0}
    #print(zero_v_bias_vs_ITO)
    if save_iv_info_fp:
        np.save(save_iv_info_fp,iv_full_dict,allow_pickle=True)
    #print(swapped_pol)
    return iv_full_dict



