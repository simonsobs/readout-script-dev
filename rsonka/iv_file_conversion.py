"""
20241109 Rita F. Sonka
'Analyzed' IV data and its accompanying metadata has been stored in three 
different formats throughout SO's history: pysmurf, original sodetlib, and new sodetlib.

This file notes the functions to take IV curves in all three, describes the files each method 
of taking iv curves creates and provides functions for converting between the three. 

Should get the original sodetlib to pysmurf in here...

FILE CONTENT SUMMARY:



80 char at |, 100 at end of line:
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
"""
import numpy as np
from sodetlib.operations.iv import * #as ivsod
#from sodetlib.operations.iv import IVAnalysis
from datetime import datetime
#import python_utilities as pu
from python_utilities.ritas_python_util_main import make_filesafe

# ======================== Reference: IV taking Methods and Output files ==========================



# ============================== Method and file descriptions ======================================
# --------------------------------------------------------------------------------------------------
# ==================================================================================================
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
# =======80 characters =========================================================

# =======80 characters =========================================================




def init_fields(ivax, nchans, nbiases):
    ivax.nchans = nchans
    ivax.nbiases = nbiases
    ivax.v_bias = np.full((ivax.nbiases, ), np.nan)
    ivax.i_bias = np.full((ivax.nbiases, ), np.nan)
    ivax.resp = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.R = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.p_tes = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.v_tes = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.i_tes = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.R_n = np.full((ivax.nchans, ), np.nan)
    ivax.R_L = np.full((ivax.nchans, ), np.nan)
    ivax.p_sat = np.full((ivax.nchans, ), np.nan)
    ivax.si = np.full((ivax.nchans, ivax.nbiases), np.nan)
    ivax.idxs = np.full((ivax.nchans, 3), -1, dtype=int)
    ivax.bgmap = np.full((ivax.nchans, ), -1, dtype=int)
    ivax.polarity = np.full((ivax.nchans, ), -1, dtype=int)
    
    
##########################################################################
# IV Translation functions: Jack lashner's, edited by me, to/from IVA
##########################################################################
def pysmurf_dict_to_ivax(d): # Needs units update
    """
    Attempts to convert an iv from the old "pysmurf dict" format into an
    IVAnalysis object. Not all of the fields will be populated, but enough
    perform certain analyses and to run through plotting functions.
    Args
    -----
    d : dict
        Dictionary containing analyzed iv data in the old format where bands
        and channels were dictionary keys.  
    """
    bands = []
    channels = []
    for b in range(8):
        if b not in d:
            continue
        cs = list(d[b].keys())
        bands.extend([b for _ in cs])
        channels.extend(cs)
    bands = np.array(bands, dtype=int)
    channels = np.array(channels, dtype=int)

    nchans = len(bands)
    nbiases = len(d[bands[0]][channels[0]]['v_bias'])
    ivax = IVAnalysis()
    ivax.bands = bands
    ivax.channels = channels
    ivax.init_fields(nchans, nbiases)

    for i in range(len(bands)):
        chan_data = d[bands[i]][channels[i]]
        for k, v in chan_data.items():
            if k == 'si':
                ivax.si[i, :-1] = v
            elif k == 'trans idxs':
                ivax.idxs[i, :2] = v
            elif k in ['v_bias']:
                ivax.v_bias = v
            elif hasattr(ivax, k):
                getattr(ivax, k)[i] = v
        ivax.i_tes[i] = chan_data['v_tes'] / chan_data['R']
    return ivax


det_aligned_atts = [
    ('bands', 'sb'), ('channels', 'ch'), ('bgmap', 'bl'),
    'polarity', 'resp', 'R', 'R_n', 'R_L', 'p_tes', 'v_tes',
    'i_tes', 'p_sat', 'si'
]


def get_chan_dict(ivax, idx): 
    """
    New IV to chan dict
    """
    u6 = 10**6
    cd = {}
    # Rita:is this making copies of lists...?
    for k in det_aligned_atts:
        if isinstance(k, (tuple, list)):
            name, new_name = k
        else:
            name, new_name = k, k
        cd[new_name] = getattr(ivax, name)[idx]
    cd.update({
        'v_bias': ivax.v_bias,
        'i_bias': ivax.i_bias
    })
    # Rita: fix idxs
    cd['trans idxs'] = getattr(ivax,'idxs')[idx][:2]
    # ...and units (though I do see the merit in pure SI)
    # to micro-units
    for att in ['i_bias','resp','v_tes','i_tes']: # 'si'? TODO: check!
        cd[att] = u6*cd[att]
    # to pico-units
    for att in ['p_tes',  'p_sat']:
        cd[att] = u6*u6*cd[att]
    # pysmurf p_trans
    cd['p_trans'] = np.median(cd['p_tes'][cd['trans idxs'][0]:cd['trans idxs'][1]])
    return cd


def ivax_to_pysmurf_files_sep_bl(ivax,met_fp_base,analysis_fp_base,
                                 bath_temp=100,PWV=-1):
    """
    Converts IVAnalysis object into csv and analysis files
    for old "pysmurf" format, 1 file per bias line
    even if they were taken together.
    Args
    -----
    ivax : IVAnalysis, str
        IVAnalysis instance or path to saved IVAnalysis file.
    met_fp_base : str
        base filepath to where the metadata csv will go. Must end in '/'
    analysis_fp_base : str
        base filepath to where the  "_iv.npy" files will go. Must end in '/'
    Optional
    bath_temp : int (default 100)
        temperature of the TES bath (UFM outside, basically) in mK 
        when the data was taken.
    """
    if isinstance(ivax, str):
        ivax = IVAnalysis.load(ivax)
    metadata_csv_header = ["bath_temp","bias_voltage","bias_line","band","data_path","type"]
    bv_str = f"{min(ivax.v_bias)} to {max(ivax.v_bias)}"
    btbv =f"{bath_temp},{bv_str}" 
    # metadata name
    timestamp = int(ivax.meta['action_timestamp'])
    date = datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d')
    metadata_csv_name = make_filesafe(f"{ivax.meta['stream_id']}_ivs_PWV{PWV}_{date}")+".csv"
    with open(met_fp_base+metadata_csv_name, 'w') as csv:
        csv.write(','.join(metadata_csv_header)+'\n')
        for bl in range(-1,12):
            py_dict = ivax_to_pysmurf_dict(ivax,bl=bl)
            afp = f"{analysis_fp_base}{timestamp}_iv_{bl}.npy"
            np.save(afp,py_dict)
            csv.write(f"{btbv},{bl},all,{afp},IV\n")
    return met_fp_base+metadata_csv_name
    
        
    
    

def ivax_to_pysmurf_dict(ivax,bl='all'): # wow, jack put a lot more work into going the other way...
    """
    Converts IVAnalysis object into old "pysmurf dict" format.
    Args
    -----
    ivax : IVAnalysis, str
        IVAnalysis instance or path to saved IVAnalysis file.
    Optional
    bl : str or int (default 'all')
        'all' to group all in one dict, else int of bias line to do.
    """
    if isinstance(ivax, str):
        ivax = IVAnalysis.load(ivax)
    if bl=='all':
        idx_list = range(len(ivax.bands))
    else: 
        idx_list = np.where(ivax.bgmap==bl)[0]
    d = {}
    for i in idx_list:
        band = ivax.bands[i]
        channel = ivax.channels[i]
        if band not in d:
            d[band] = {}
        d[band][channel] = get_chan_dict(ivax, i)
    d['high_current_mode'] = ivax.meta['high_current_mode'][0] # Ugh...different hc for different bl?!?
    return d



'''
IVAnalysis analysis file .npy format
{
    meta       :dict           {
        tunefile              :str           /data/smurf_data/tune/ufm_uv37/...
        high_low_current_ratio:float         5.73
        R_sh                  :float         0.0004
        pA_per_phi0           :float         9000000.0
        rtm_bit_to_volt       :float         1.9073486328125e-05
        bias_line_resistance  :float         16060.0
        chans_per_band        :int           512
        high_current_mode     :numpy.ndarray [True, True, True, True, True, ...
        timestamp             :float         1725735326.9524653
        stream_id             :str           ufm_uv37
        action                :str           take_iv
        action_timestamp      :str           1725734930
        bgmap_file            :str           /data/smurf_data/bias_group_map...
        iv_file               :str           /data/smurf_data/20240905/ufm_u...
        v_bias                :numpy.ndarray [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...
        pysmurf_client_version:str           7.4.0
        rogue_version         :numpy.ndarray [118, 52, 46, 49, 49, 46, 49, 48]
        smurf_core_version    :numpy.ndarray [55, 46, 52, 46, 48]
        fpga_git_hash         :str           02ed52a
        cryocard_fw_version   :tuple         (4, 0, 7)
        crate_id              :int           5
        slot                  :str           4
    } #(meta)
    bands      :numpy.ndarray [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    channels   :numpy.ndarray [0, 2, 4, 7, 11, 15, 18, 19, 20, 24, 27, 29, 3...
    sid        :int           1725734938
    start_times:numpy.ndarray [[1725735249.4097934, 1725735249.4554713, 1725...
    stop_times :numpy.ndarray [[1725735249.4199898, 1725735249.465618, 17257...
    run_kwargs :dict           {
        bias_groups      :numpy.ndarray [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        overbias_voltage :int           30
        overbias_wait    :int           10
        high_current_mode:bool          True
        cool_wait        :int           300
        cool_voltage     :NoneType      **NO PREV:NoneType**
        biases           :numpy.ndarray [35.0, 34.975, 34.95, 34.92500000000...
        bias_high        :int           35
        bias_low         :int           0
        bias_step        :float         0.025
        show_plots       :bool          True
        wait_time        :float         0.01
        run_analysis     :bool          True
        run_serially     :bool          False
        serial_wait_time :int           30
        g3_tag           :NoneType      **NO PREV:NoneType**
        analysis_kwargs  :dict          **NO PREV:dict**
    } #(run_kwargs)
    biases_cmd :numpy.ndarray [1.9895196601282805e-12, 0.0250000000019881, 0...
    bias_groups:numpy.ndarray [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    nbiases    :int           1401
    nchans     :int           1699
    bgmap      :numpy.ndarray [-1, 8, 8, 10, -1, -1, 8, 7, 8, -1, 7, 8, -1, ...
    polarity   :numpy.ndarray [0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0...
    resp       :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    v_bias     :numpy.ndarray [0.0, 0.024918365478515628, 0.0498367309570312...
    i_bias     :numpy.ndarray [0.0, 1.5515794195837875e-06, 3.10315883916757...
    R          :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    R_n        :numpy.ndarray [nan, 0.09568101228681036, 0.00834675005921205...
    R_L        :numpy.ndarray [nan, 6.6963715617145075e-06, -5.8371776176746...
    idxs       :numpy.ndarray [[-1, -1, -1], [574, 575, 1069], [334, 339, 34...
    p_tes      :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    v_tes      :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    i_tes      :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    p_sat      :numpy.ndarray [nan, 5.09031568781316e-12, 7.910378475147533e...
    si         :numpy.ndarray [[nan, nan, nan, nan, nan, nan, nan, nan, nan,...
    idx        :numpy.ndarray [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...
    sb_ch      :numpy.ndarray [0, 2, 4, 7, 11, 15, 18, 19, 20, 24, 27, 29, 3...
    st_is_iv   :numpy.ndarray [False, False, False, False, False, False, Tru...
} #()
'''



'''
Original sodetlib analysis file

{
    metadata:dict  {
        iv_info   :dict      {
            plot_dir            :str           /data/smurf_data/20230823/cra...
            output_dir          :str           /data/smurf_data/20230823/cra...
            tune_file           :str           /data/smurf_data/tune/crate1s...
            R_sh                :float         0.0004
            bias_line_resistance:float         16065.0
            high_low_ratio      :float         5.71
            pA_per_phi0         :float         9000000.0
            high_current_mode   :bool          True
            start_time          :str           1692772111
            stop_time           :str           1692772201
            basename            :str           1692772117
            session id          :int           1692772117
            datafile            :str           /data/smurf_data/20230823/cra...
            bias                :numpy.ndarray [6.129597197898424, 6.1252189...
            bias group          :numpy.ndarray [0]
            overbias_voltage    :int           19
            overbias_wait       :float         2.0
            cool_wait           :int           300
            cool_voltage        :NoneType      **NO PREV:NoneType**
            wait_time           :float         0.01
            wafer_id            :NoneType      **NO PREV:NoneType**
            version             :str           v1
        } #(iv_info)
        iv_info_fp:str      /data/smurf_data/20230823/crate1slot6/1692771745...
        wafer_id  :NoneType **NO PREV:NoneType**
        version   :str      v2
    } #(metadata)
    data    :dict  {
        1:dict  {
            390:dict  {
                R     :numpy.ndarray [-0.0004, -0.0004, -0.00037939098971077...
                R_n   :numpy.float64 0.0017655536069420574
                idxs  :numpy.ndarray [847, 1186, 847]
                p_tes :numpy.ndarray [nan, nan, -0.3414548914249187, -0.3423...
                p_sat :numpy.ndarray 111.2455740550055
                v_bias:numpy.ndarray [0.0, 0.0, 0.0043487548828125, 0.004348...
                v_tes :numpy.ndarray [nan, nan, -0.011381779702633718, -0.01...
                i_tes :numpy.ndarray [nan, nan, 30.000131819974236, 30.03882...
                si    :numpy.ndarray [nan, nan, nan, nan, nan, nan, nan, 0.5...
            } #(390)
            392:dict  {
                R     :numpy.ndarray [-0.0004, -0.0004, -0.00174819443594747...
                R_n   :numpy.float64 2.924566337218881
                idxs  :numpy.ndarray [719, 1186, 1476]
                p_tes :numpy.ndarray [nan, nan, -0.00036765905863658257, 2.8...
                p_sat :numpy.ndarray 0.19383439472997782
                v_bias:numpy.ndarray [0.0, 0.0, 0.0043487548828125, 0.004348...
                v_tes :numpy.ndarray [nan, nan, 0.0008017103720385317, 0.000...
                i_tes :numpy.ndarray [nan, nan, -0.45859336670638984, 0.0480...
                si    :numpy.ndarray [nan, nan, nan, nan, nan, nan, nan, -24...
            } #(392)
        } #(1)
        2:dict  {
            119:dict  {
                R     :numpy.ndarray [-0.0004, -0.0004, -0.00040638746922310...
                R_n   :numpy.float64 0.0005864552734622063
                idxs  :numpy.ndarray [235, 1182, 1418]
                p_tes :numpy.ndarray [nan, nan, -3.8075293978150975, -3.8289...
                p_sat :numpy.ndarray 425.464585183916
                v_bias:numpy.ndarray [0.0, 0.0, 0.0043487548828125, 0.004348...
                v_tes :numpy.ndarray [nan, nan, 0.039336144142132615, 0.0394...
                i_tes :numpy.ndarray [nan, nan, -96.7946777919416, -97.06884...
                si    :numpy.ndarray [nan, nan, nan, nan, nan, nan, nan, -30...
            } #(119)
            249:dict  {
                R     :numpy.ndarray [-0.0004, -0.0004, -0.00046613023877539...
                R_n   :numpy.float64 0.0006827129723984265
                idxs  :numpy.ndarray [742, 1186, 1319]
                p_tes :numpy.ndarray [nan, nan, -0.040744378151269044, -0.05...
                p_sat :numpy.ndarray 356.9664948225779
                v_bias:numpy.ndarray [0.0, 0.0, 0.0043487548828125, 0.004348...
                v_tes :numpy.ndarray [nan, nan, 0.004358002606287201, 0.0047...
                i_tes :numpy.ndarray [nan, nan, -9.349323952328062, -10.4530...
                si    :numpy.ndarray [nan, nan, nan, nan, nan, nan, nan, 8.5...
            } #(249)
        } #(2) (ignored 1 numpy.int64)
    } #(data) (ignored 2 numpy.int64)
} #()'''


'''
RELOADABLE analysis output from mine:

 {
    iv_analyzed_info_arr:list [[**NO PREV:dict**, **NO PREV:dict**, **NO PRE...
    options             :dict  {
        dName          :str  Uv41 P-I-004 nso ubin
        mux_map_fp     :str  /home/kaiwenz/det_map/PI04_Uv41_smurf2det.csv
        metadata_fp_arr:list [/home/rsonka/notebooks/metadata_csvs_edited/Uv...
        input_file_type:str  original_sodetlib
        fix_pysmurf_idx:bool False
        sc_offset      :bool False
        bin_tod        :bool False
    } #(options)
} #()
 {
    temp          :float 77.0
    bl            :int   7
    iv_analyzed_fp:str   /data2/smurf_data/20230922/crate1slot4/1695423856/o...
    iv_analyzed   :dict   {
        high_current_mode:bool True
        3                :dict  {
            164:dict  {
                R                    :numpy.ndarray [-0.00039897465457276873...
                R_n                  :numpy.float64 -0.00024206189715939851
                idxs                 :numpy.ndarray [5960, 13217, -1]
                p_tes                :numpy.ndarray [-153.49270594586855, -1...
                p_sat                :float         nan
                v_bias               :numpy.ndarray [0.00446319580078125, 0....
                v_tes                :numpy.ndarray [-0.2474665620527195, -0...
                i_tes                :numpy.ndarray [620.25635, 619.23584, 6...
                si                   :numpy.ndarray [726630.7520389257, 7219...
                pA_per_phi0          :float         9000000.0
                i_tes_offset         :numpy.float64 -5830.193739142216
                pvb_idx              :int           0
                timestamp            :numpy.ndarray [1695423980.0135868, 169...
                zero_bias_resp_values:numpy.ndarray [-5211.351, -5212.1885, ...
                trans idxs           :list          [-1, -1]
                p_trans              :numpy.float64 nan
                si_target            :numpy.float64 -2.9267193384840775
                v_bias_target        :numpy.float64 6.25
                v_tes_target         :numpy.float64 -1.4018760863947202
            } #(164)
        } #(3)
    } #(iv_analyzed) (ignored 4 numpy.int64)
    human_time    :str   2023-09-22 19:04:58
} #()
'''