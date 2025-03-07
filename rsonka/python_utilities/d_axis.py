"""
Rita Sonka
08/24/2023
Compilation of generic dict-axis and key_info functions
As primarily used in storing and manipulating TES, IV and 
temperature ramp data in temp_ramp_analysis_classes.py,
for Simons Observatory. 

-------------------
A dict_axis is a dictionary with string keys, where all
values are numpy arrays of the same length, but potentially
different data types (including, possibly, more numpy arrays).

It must include one key,'idx', which pairs to:
 np.arange(<length of the dict_axis's keyed numpy arrays)


This enables swift selection of data subsets from criteria on
any key(s) of the dict acis, which is then easy to plot.

--------------------
A key_info dict is a dictionary that accompanies one or more
dict-axises, and which tracks important metadata about each 
dict-axis key. It is of the form:

# {<key_name, as used in axes>:{
#      'name': <full name for display>, 
#      #'axis': an axis reference.  # Nope. Removed to make entries legible 
#      'type': <discrete 'd', continuous 'c', or continuous arrays 'ca'>
#      'units':<units of values; often '' for discrete>
#      # if discrete:
#      'vals': np.sort()ed ndarray of all unique possible associated values
#      'colors': (num_vals,3)-shape ndarray of val-associated RGB colors
#      # else, if continuous:
#      'extremes': [<lowest value>,<highest value>]
#      'lim':[<lowest val to display>,<highest val to display>]
#      # note: right now lim=extremes, but did some work on changing,
#      can manually set it.

It may include additional keys, such as a string 'ax_name':<axis name>.


GLOBALS:
FUNCTIONS:  (helpers indented)
# =============================== EDITING FUNCTIONS ============================
update_key_info(key_info,key,name,axis,dca,units,more_info={})
# =============================== SEARCHING FUNCTIONS ==========================
ml(mlstr)
    str_to_match_list(mlstr)
        str_to_num_or_bool(strv)
(fim() = alias find_idx_matches v)
find_idx_matches(dax,match_list,return_type='list') #or list,'dict',dict,'dax'
    apply_match(dax,idx_list,match)
        match_mask(dax,idx_list,match)
            single_match_mask(dax,idx_list,match)
    add_dict_level(dax,idx_list,level_list,d)
print_idx_lines(dax,idxs,keys='all',max_cell_size=110,sig_figs=-42)
    idx_lines(dax,idxs,keys='all')
# ================================ PLOTTING FUNCTIONS =======================================
# ------------------------------- Plotting Utility ----------------------------------------
enter_fig(own_fig)
restriction_title_suffix(match_list,exclude_unknowns)
label_x_y_title_and_lim(x_key,y_key,titlebase,p,own_fig,key_info='{}',restrictions='',
                            x_lim='[]',y_lim=[])
colors_for_discrete_list(unique_vals,key='')
idxs_grouped_and_colored_by_keys(dax,key_info,by_keys,match_list='[]', exclude_unknowns=False,
                                     color_by_keys='[]',return_color_dax=False)
# ------------------------------- Plot Makers ----------------------------------------
plot_key_v_key(dax,key_info,x_key,y_key,match_list='[]',exclude_unknowns=False,
                   x_lim='[]',y_lim='[]',plot_args='{}',own_fig=True,
                   prefix='',title_override='',label='')
plot_key_v_key_grouped_by_key(dax,key_info,x_key,y_key,by_keys,match_list='[]',
                                  exclude_unknowns=False, color_by_keys='[]',
                                  x_lim='[]',y_lim='[]',plot_args='{}',own_fig=True,
                                  prefix='',title_override='',legend='default')
plot_key_v_key_colored_by_key(dax,key_info,x_key,y_key,by_key,match_list='[]',
                                  exclude_unknowns=False,
                                  x_lim='[]',y_lim='[]', v_lim='[]',
                                  color_scale=plt.cm.inferno, outlier_colors=['blue','green'],
                                  xy_overlap_offset=1, plot_args={'marker':'.'},own_fig=True,
                                  prefix='',title_override = '')
plot_correlations_grouped_by(dax,key_info,key_list, by_key, 
                                 match_list='[]',exclude_unknowns=False,
                                 ax_lims='{}',prefix='',plot_args={}):
d_ax_fit_and_resid(ax,key_info,function,x_key_or_keys,y_key, p0, bounds=([],[]),
                  match_list='[]', exclude_unknowns=False,
                  own_fig=True, plot_which_x_variable=-42,  
                  label='', suptitle='',legend=False, plot_args='{}',
                  save_ys='',save_resid='',x_label='default',y_label='default')



80 char at |, 100 at end of line:
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
"""

# CANNOT have reloads in here generically, gets circular!
#from importlib import reload # needed one time to update without resetting kernel


import numpy as np
from numpy_plotting import *
import python_utilities as pu
from python_utilities.ritas_python_util_main import try_to_numerical
from python_utilities.ritas_python_util_main import is_float
from python_utilities.ritas_python_util_main import anmd
from python_utilities.dict_analyzer_pretty import pdap # imported for debugging
#reload(pu.pretty_print_table)
from python_utilities.pretty_print_table import ppt # imported for show_lines
import matplotlib as mpl



default_figsize=(5,3.3) # firefox (2.5,2.5) # chrome (5,5) # ac has this too!

t10_colors = plt.cm.tab10(np.linspace(0,1.0,9+1))
t20_colors = plt.cm.tab20(np.linspace(0, 1.0, 19+1))
t20b_colors = plt.cm.tab20b(np.linspace(0, 1.0, 19+1))


#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
# =============================== EDITING FUNCTIONS ============================

# should this exist?
# def auto_update_key_info(key_info,key,name,axis,dca,units,more_info={}):
#     '''Adds or updates key_info of an axis
#     s = Ramp_Combination; ca =continuous of arrays
#     key_info = {<key_name, as used in axes>:{
#         #      'name': <full name for display>, 
#         #      'ax_name': <name of primary axis it appears in>
#         #      'type': <discrete 'd', continuous 'c', or continuous arrays 'ca'>
#         #      'units':<units of values; often '' for discrete>
#         #      # if discrete:
#         #      'vals': np.sort()ed ndarray of all unique possible associated values
#         #      'colors': (num_vals,3)-shape ndarray of val-associated RGB colors
#         #      # else, if continuous:
#         #      'extremes': [<lowest value>,<highest value>]
#         #      'lim':[<lowest val to display>,<highest val to display>]
#         #      # note: right now lim=extremes, but did some work on changing,
#         #      can manually set it.}
#     }'''
    
def uki(dax,key_info,key,name,dca,units,more_info='{}'):
    '''update_key_info, but with arguments in an order more consistent
       with the rest of the dax functions:
       dax, key_info,'''
    return update_key_info(key_info,key,name,dax,dca,units,more_info=more_info)

def update_key_info(key_info,key,name,axis,dca,units,more_info='{}'):
    '''Adds or updates key_info of an axis
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
#         if key in s.key_info.keys(): 
#             print("already there-updating!")
    more_info=anmd(more_info)
    kd = {'name':name,#'axis':axis,
          'type':dca, 'units':units}
    for keyn,val in more_info.items(): # CANNOT BE NAMED 'key'!!!Will override argument!
        kd[keyn] = val
    if 'c' in dca:
        # now, extremes, limits
        if 'a' in dca:
            vals=[]
            v_min, v_max = np.inf,-np.inf
            try:
                arrs = axis[key][find_idx_matches(axis,[],exclude_unknowns=[key])]
                for arr in arrs:
                    if min(arr) < v_min:
                        v_min = min(arr)
                    if max(arr) > v_max:
                        v_max = max(arr)
                vals=[v_min,v_max]
            except Exception as err:#except BaseException as err: (not sure why originally base exception)
                print(f"{key} key_info err:{err}")
        else:
            try:
                vals = axis[key][find_idx_matches(axis,[],exclude_unknowns=[key])]
            except IndexError:
                print(f"ln220 Index_error {key}")
        try:
            kd['extremes'] = [min(vals),max(vals)]
            kd['lim'] = kd['extremes'] # for now....maybe use lim finder later
        except BaseException as err:
            print(f"{key} key_info err:{err}")
    else: # discrete. 
        vals = axis[key][find_idx_matches(axis,[],exclude_unknowns=[key])]
        # Dynamic allocation is crazy slow, don't do that.
        # not sure why, but my attempt at avoiding it is taking much longer, so back to that.
        #unique_vals = np.full((len(vals),),-42,dtype=object)  # Handles everything pretty well
        unique_vals = []
        j = 0
        for val in vals:
            if not val in unique_vals: # this handles 
                unique_vals.append(val)
                j += 1
        try:
            #unique_vals=np.sort(np.array(unique_vals[unique_vals != -42])) 
            unique_vals=np.sort(np.array(unique_vals)) 
        except TypeError as err:
            print(f"couldn't sort {key}: {err}")
        kd['vals'] = unique_vals
#         if len(unique_vals) > 40:
#             print(f"40+ discrete variable?!? {key}")
        kd['colors'] = colors_for_discrete_list(unique_vals,key=key)
    key_info[key] = kd
    return "added"



# =============================== SEARCHING FUNCTIONS =======================================

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


# -----------THE dax function. So much that it gets a 3-letter alias, officially.------------
def fim(dax,match_list, exclude_unknowns=False, return_type='list'):
    return find_idx_matches(dax,match_list, exclude_unknowns= exclude_unknowns, return_type= return_type)

#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
def find_idx_matches(dax,match_list, exclude_unknowns=False, return_type='list'):
    '''Returns what indices of the dax's arrays match the match_list criteria.
        See Temp_Ramp.find_iva_matches() and str_to_match_list() aka ml()
        set exclude_unkowns to a list of daxis keys. It will exclude
        any idxs that have a -42, np.nan, "-", or "?" as that key's value.
        return_type can be 'list',list, 'dict',dict, or 'dax'. 
        if it isn't list, it will return the idxs sorted into either (as directed) 
        a tiered dict or a new d_axis, split by 
        the keys listed with ='any',='all',or =[...].'''
    # like find_ivas, but returns idxs. 
    # find the matches
    idx_list = dax['idx']
#     if idx_list.dtype != np.array([],dtype=int).dtype:
#         print("PROBLEM!! idx is wrong dtype!")
    if type(match_list) == str:
        match_list = str_to_match_list(match_list)
    if exclude_unknowns:
        if type(exclude_unknowns) == list:
            by_keys = exclude_unknowns   
        else:
            by_keys = [exclude_unknowns]
        for by_key in by_keys:
            match_list = match_list + [(by_key,'!=',-42),(by_key,'!=',-42.0),
                                       (by_key,'!=',-4.20),
                                       (by_key,'!=',np.nan),(by_key,'!=','-'),
                                       (by_key,'!=','?')] #np.nan uses special check from this
    for match in match_list:
        idx_list = apply_match(dax,idx_list,match)
        
    # organization if necessary
    num_levels=0
    level_list = []
    for match_key,match_type,match_val in match_list:
        if match_type == "=" and (match_val in ['all','any'] or type(match_val) == list):
            num_levels +=1
            level_list.append((match_key,match_type,match_val))
    if return_type in ['list',list] or num_levels == 0:
        return idx_list
    dicty = {}
    ret_dict,group_count =  add_dict_level(dax,idx_list,level_list,dicty,0)
    if return_type in ['dict',dict]:
        return ret_dict
    #return_type = dax
    ndax = {'idx':np.arange(group_count),
            'group_idxs':np.full((group_count,),'',dtype=object),
            'group_name':np.full((group_count,),'',dtype=object),
            }
    for match_key,match_type,match_val in level_list:
        l_type = type(dax[match_key][0])
        # it doesn't matter if the "unknown" value is bad,
        # because guaranteed to overwrite all of them.
        ndax[match_key] = np.full((group_count,),l_type(),dtype=l_type)
    i_count = return_dax(level_list,ret_dict,ndax,0,{})
    level_keys = [level_match[0] for level_match in level_list]
    for i in range(group_count):
        ndax['group_name'][i] = ",".join([str(ndax[key][i]) for key in level_keys])
            
    return ndax

def return_dax(level_list,d,ndax,i,ndax_d): 
    '''because I decided to group the idxs for grouped_by_and_colored_by'''
    match_key,match_type,match_val = level_list[0]
    if len(level_list) == 1:
        n_entry = 0
        for key,idx_l in d.items():
            ndax[match_key][i+n_entry] = key
            ndax['group_idxs'][i+n_entry] = idx_l
            n_entry+=1
        for key,l_val in ndax_d.items():
            ndax[key][i:i+n_entry] = l_val
        return i+n_entry
    l_vals = [key for key in d.keys()]
    # I believe now they SHOULD be sorted....
    #l_vals = np.sort(l_vals) # should already be unique, but not necessarily ordered.
    for l_val in l_vals:
        ndax_d[match_key] = l_val
        i = return_dax(level_list[1:],d[l_val],ndax,i,ndax_d)
    return i

def add_dict_level(dax,idx_list,level_list,d,count):
    if len(level_list) == 0: 
        return (idx_list,count+1)
    match_key,match_type,match_val = level_list[0]
    d_keys = np.unique(dax[match_key][idx_list]) # np.unique actually sorts if it is possible!!
    for key in d_keys:
        d[key],count = add_dict_level(dax,idx_list[np.where(dax[match_key][idx_list] == key)[0]],
                                    level_list[1:],{},count)
    return (d,count)

def match_mask(dax,idx_list,match):  
    match_key,match_type,match_val = match
    if type(match_val) == list or type(match_val) == type(np.arange(0,2)):
        idx_list_mask = np.full((len(idx_list),), False,dtype=bool)
        for mv in match_val:
            idx_list_mask[single_match_mask(dax,idx_list,(match_key,match_type,mv))==True] = True
    else:
        idx_list_mask = single_match_mask(dax,idx_list,match)
    return idx_list_mask

def single_match_mask(dax,idx_list,match):
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
            return np.isnan(dax[match_key][idx_list]) 
        except TypeError: # at least some aren't np.nan. Unfortunately we need to check individually.
            return np.array([False if not type(val) == float \
                         else np.isnan(val) for val in dax[match_key][idx_list]])  
    
    # Ugh. now that np.nan is handled, we can dothe normal stuff. 
    idx_list_mask = np.full((len(idx_list),), False,dtype=bool)
    # numpy can't do elementwise comparisons of string to array of non-string
    # fortunately if we do detect that situation, the mask is just all False anyway
    if not ((type(match_val) == str or type(match_val) == np.str_) \
            and not (type(dax[match_key][0]) == np.str_ or
                     type(dax[match_key][0]) == str)):
        try:
            idx_list_mask[np.where(dax[match_key][idx_list] == match_val)[0]] = True   
        except IndexError:
            print(f"{match_key} {match_val}")
            print(idx_list)
            print(type(idx_list))
    return idx_list_mask


def apply_match(dax,idx_list,match):
    match_key,match_type,match_val = match 
    if match_type == '=' or match_type == '!=':
        if match_val in ['all','any'] \
        or (type(match_val) == list and len(match_val)==1 and
            (match_val[0] == 'all' or match_val[0]=='any')):
            if match_type == '!=':
                # numpy will break if using an array (to index) that is not explicitly boolean or int type
                return np.array([],dtype=int) 
            return idx_list
        else: 
            idx_list_mask = match_mask(dax,idx_list,match)
            if match_type == "!=":
                return idx_list[idx_list_mask == False]
            return idx_list[idx_list_mask]
    # there should not be any np.nan match vals on these match_types
    #assert not np.isnan(match_val), f"can't use 
    if match_type == '<':
        return idx_list[np.where(dax[match_key][idx_list] < match_val)[0]]
    if match_type == '<=':
        return idx_list[np.where(dax[match_key][idx_list] <= match_val)[0]]
    if match_type == '>=':
        return idx_list[np.where(dax[match_key][idx_list] >= match_val)[0]]
    if match_type == '>':
        return idx_list[np.where(dax[match_key][idx_list] > match_val)[0]]
    



def print_idx_lines(dax,idxs,keys='all',max_cell_size=110,sig_figs=-42):  
    ppt(idx_lines(dax,idxs,keys=keys),max_cell_size=max_cell_size,sig_figs=sig_figs)
    
def idx_lines(dax,idxs,keys='all'):
    if is_float(idxs):
        idxs =[idxs]
    if keys=='any' or keys=='all':
        keys=[key for key in dax.keys()]
    if type(keys) == str:
        keys=[keys]
    to_print = [keys]
    for i in idxs:
        to_print.append([dax[key][i] for key in keys])#
    return to_print 
    




# ================================ PLOTTING FUNCTIONS =======================================

# -------------------------------- Plotting Utility ----------------------------------------
def enter_fig(own_fig,figsize=default_figsize):
    '''own_fig==True -> creates new figure, returns plt
       own_fig==False -> returns plt
       own_fig== <something else> -> returns <something else>'''
    p = plt
    if own_fig == True:
        plt.figure(figsize=figsize)
    elif not own_fig == False:
        p= own_fig
    return p

def restriction_title_suffix(match_list,exclude_unknowns):
    restrictions = ''
    if exclude_unknowns or len(match_list) > 0:
        restrictions = '\n'
    if exclude_unknowns:
        restrictions = restrictions + 'no unknowns '
        if exclude_unknowns == True:
            exclude_unknowns = [x_key,y_key,by_key]
    if len(match_list) >0:
        restrictions = restrictions + f"[{', '.join([f'{mk}{mt}{mv}' for mk,mt,mv in match_list])}]"
    return restrictions

def label_x_y_title_and_lim(x_key,y_key,titlebase,p,own_fig,key_info='{}',restrictions='',
                            x_lim='[]',y_lim='[]',title_override=''):
    key_info,x_lim,y_lim = anmd(key_info,x_lim,y_lim)
    if own_fig == True:
        p.title(f"{titlebase}{restrictions}")
        if title_override:
            p.title(title_override)
        #plt.xlabel(f"{key_info[x_key]['name']} [{key_info[x_key]['units']}]") 
        x_label = f"{x_key}"
        if x_key in key_info.keys():
            x_label = f"{key_info[x_key]['name']}"
            if 'units' in key_info[x_key].keys(): # should be if continuous
                x_label = x_label + f" [{key_info[x_key]['units']}]"
        p.xlabel(x_label)
        y_label = f"{y_key}"
        if y_key in key_info.keys():
            y_label = f"{key_info[y_key]['name']}"
            if 'units' in key_info[y_key].keys(): # should be if continuous
                y_label = y_label + f" [{key_info[y_key]['units']}]"
        p.ylabel(y_label)
        if x_lim:
            p.xlim(x_lim)
        if y_lim:
            p.ylim(y_lim)
    else:
        # probably a subplot, but not guaranteed
        if x_lim:
            try:
                p.set_xlim(x_lim)
            except AttributeError as nopeThatIsFullPlot:
                p.xlim(x_lim)
        if y_lim:
            try:
                p.set_ylim(y_lim)
            except AttributeError as nopeThatIsFullPlot:
                p.xlim(y_lim)
    
def colors_for_discrete_list(unique_vals,key=''):
    if len(unique_vals) <= 10:
        return np.array([t10_colors[j,0:3] for j in range(len(unique_vals))])
    elif len(unique_vals) <=20:
        return np.array([t20_colors[j,0:3] for j in range(len(unique_vals))])
    elif len(unique_vals) <= 40:
        return np.array([t20_colors[j,0:3] for j in range(20)] \
                                + [t20b_colors[k,0:3] for k in range(len(unique_vals)-20)])
    else:
        if key=='':
            key = f"?key, 1stUniqueVal={unique_vals[0]}"
        print(f"40+ discrete variable?!? {key}") 
        color_scale = plt.cm.viridis
        v_min, v_max = 0,len(unique_vals)
        ratios = (np.arange(len(unique_vals))-v_min)/(v_max-v_min)
        colors = np.full((len(unique_vals),3), [0.0,0.0,0.0])
        for i in range(len(unique_vals)):
            with_alpha = color_scale(ratios[i]) # I don't know why I called this with_alpha.
            colors[i,:] = np.array([with_alpha[0],with_alpha[1],with_alpha[2]])
    return colors
    
def idxs_grouped_and_colored_by_keys(dax,key_info,by_keys,match_list='[]', exclude_unknowns=False,
                                     color_by_keys='[]',return_color_dax=False):
    match_list,color_by_keys=anmd(match_list,color_by_keys)
    if type(by_keys) == str: # only one key
        by_keys = [by_keys]
    if type(color_by_keys) == str:
        color_by_keys =[color_by_keys]
    # group dax
    gd = find_idx_matches(dax,[[key,'=','any'] for key in by_keys] + match_list, 
                         exclude_unknowns=exclude_unknowns, return_type='dax')
    group_names = gd['group_name']
    if color_by_keys == []: #or color_by_keys==['default'] or color_by_keys=='default':

        if len(by_keys) == 1:
            by_key = by_keys[0]
            if (not by_key in key_info.keys()) or not key_info[by_key]['type'] == 'd':
                # probably going to have other problems, but let it be for now:
                print(f"warning: {by_key} not in key_info or is not discrete!")
            else:
        
                gd['color']  = key_info[by_key]['colors'][np.in1d(key_info[by_key]['vals'],gd[by_key]).nonzero()]
                
                return gd
        gd['color'] = colors_for_discrete_list(group_names)
        return gd
    else:
        color_dax = find_idx_matches(dax,[[key,'=','any'] for key in color_by_keys] + match_list, 
                         exclude_unknowns=exclude_unknowns, return_type='dax')
        ckey = color_by_keys[0]
        if len(color_by_keys)==1 and ckey in key_info.keys() \
            and key_info[ckey]['type'] == 'd':
            color_dax['color']  = key_info[ckey]['colors'][ \
                                 np.in1d(key_info[ckey]['vals'],color_dax[ckey]).nonzero()]        
        else:
            color_dax['color'] = colors_for_discrete_list(color_dax['group_name'])        
        num = len(gd['idx'])
        for key in color_by_keys:
            k_type = type(dax[key][0])
            # it doesn't matter if the "unknown" value is bad,
            # because guaranteed to overwrite all of them.
            gd[key] = np.full((num,),k_type(),dtype=k_type)
        gd['color'] = np.full((num,3),np.nan,dtype=float)
        gd['color_group_name'] = np.full((num,),str(),dtype=str)
        for i in gd['idx']:
            dax_i = gd['group_idxs'][0] #
            mlst = []
            for j in range(len(color_by_keys)):
                key = color_by_keys[j]
                gd[key][i] = dax[key][gd['group_idxs'][i][0]]
                mlst.append((key,'=',gd[key][i]))
            # Is calling find_idx_matches on color_dax fast? Would it be better to make a dict?
            # but it's on color_dax...doesn't matter, don't be silly!
            col_i = find_idx_matches(color_dax,mlst)
            gd['color'][i] = color_dax['color'][col_i]
            gd['color_group_name'][i] = str(color_dax['group_name'][col_i])
        if return_color_dax:
            return (gd, color_dax)
        return gd
    
# ------------------------------- Plot makers ----------------------------------------
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890

def plot_key_v_key(dax,key_info,x_key,y_key,match_list='[]',exclude_unknowns=False,
                   x_lim='[]',y_lim='[]',plot_args='{}',own_fig=True,
                   prefix='',title_override='',label=''):
    match_list,x_lim,y_lim,plot_args=anmd(match_list,x_lim,y_lim,plot_args)
    # If I wanted separate plots I'd just use the ramps themselves.
    p = enter_fig(own_fig)
    axis = dax
    restrictions = restriction_title_suffix(match_list,exclude_unknowns)
    my_idxs = find_idx_matches(dax,match_list,exclude_unknowns=exclude_unknowns) 
    ret_p = p.plot(axis[x_key][my_idxs], axis[y_key][my_idxs],label=label, **plot_args)# not a good way of doing labelling
    titlebase = f"{prefix} {y_key} vs. {x_key}"
    label_x_y_title_and_lim(x_key,y_key,titlebase,p,own_fig,key_info=key_info,
                            restrictions=restrictions,x_lim=x_lim,y_lim=y_lim,
                           title_override=title_override)
    return ret_p
    
                                  
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
# legacy
def plot_key_v_key_grouped_by_key(dax,key_info,x_key,y_key,by_keys,match_list='[]',
                                  exclude_unknowns=False, colored_by_keys='[]',
                                  x_lim='[]',y_lim='[]',plot_args='{}',own_fig=True,
                                  prefix='',title_override='',labels=True,legend='default'):
    match_list,x_lim,y_lim,plot_args,colored_by_keys=anmd(match_list,x_lim,y_lim,plot_args,colored_by_keys)
    return plot_key_v_key_grouped_by_keys(dax,key_info,x_key,y_key,by_keys,match_list=match_list,
                                          exclude_unknowns= exclude_unknowns, 
                                          colored_by_keys=colored_by_keys, x_lim= x_lim,y_lim=y_lim,
                                          plot_args=plot_args,own_fig=own_fig, prefix= prefix,
                                          title_override=title_override,labels=labels,legend=legend)

def plot_key_v_key_grouped_by_keys(dax,key_info,x_key,y_key,by_keys,match_list='[]',
                                  exclude_unknowns=False, colored_by_keys='[]',
                                  x_lim='[]',y_lim='[]',plot_args='{}',own_fig=True,
                                  prefix='',title_override='',labels=True,legend='default'):
    match_list,colored_by_keys,x_lim,y_lim,plot_args=anmd(match_list,colored_by_keys,
                                                          x_lim,y_lim,plot_args)
    color_by_keys=colored_by_keys # I kept typing the latter, so it's the keyname now!
    # -------- clean up arguments -------- 
    axis = dax
    if type(by_keys) == str:
        by_keys = [by_keys]
    if type(color_by_keys)==str:
        color_by_keys = [color_by_keys]
    
    # -------- fetch and organize data --------
    idx_dax = idxs_grouped_and_colored_by_keys(dax,key_info,by_keys,match_list=match_list, 
                                               exclude_unknowns= exclude_unknowns,
                                               color_by_keys= color_by_keys,return_color_dax=True)
    if color_by_keys:
        idx_dax, color_dax = idx_dax
    group_names = idx_dax['group_name']
    colors = idx_dax['color']
    #print(idx_dax)
    color_override = False
    if 'color' in plot_args.keys():
        color_override=True

    # -------- Plot data --------
    p = enter_fig(own_fig)
    for i in range(len(group_names)):
        if not color_override:
            plot_args['color'] = colors[i]            
        g_idxs = idx_dax['group_idxs'][i] #idxs[axis[by_key][idxs] == name]
        num_points = len(g_idxs)
        
        if color_by_keys == []:
            name = group_names[i]
            if name in [-42,'?',np.nan]:
                name="?"
            # Check if these are arrays or not--nope. want to work with datetime
            label=''
            try: # check if the axis key array's values are arrays
                len(axis[x_key][g_idxs][0])
                if labels:
                    labeller = p.plot([],[],label=f"{name}", **plot_args)
                for j in range(len(g_idxs)):
                    p.plot(axis[x_key][g_idxs[j]], axis[y_key][g_idxs[j]], **plot_args)
            except Exception as e: 
                if labels:
                    label = f"{name} ({num_points} pts)"
                p.plot(axis[x_key][g_idxs], axis[y_key][g_idxs],
                       label=label, **plot_args)
                
        else:
            try:
                p.plot(axis[x_key][g_idxs], axis[y_key][g_idxs], **plot_args)
            except Exception as e: #honestly not sure what I was doing here. 
                for j in range(len(g_idxs)):
                    p.plot(axis[x_key][g_idxs[j]], axis[y_key][g_idxs[j]], **plot_args)
                    
    # -------- Plot labeling, lim setting, legend --------
    restrictions = restriction_title_suffix(match_list,exclude_unknowns)
    titlebase = f"{prefix} {y_key} v {x_key} BY {','.join(by_keys)}"
    if color_by_keys:
        titlebase = titlebase+f"\ncolored by {','.join(color_by_keys)}"
    label_x_y_title_and_lim(x_key,y_key,titlebase,p,own_fig,key_info=key_info,
                            restrictions=restrictions,x_lim=x_lim,y_lim=y_lim)
    if legend=='default':
        if color_by_keys == []:
            if len(group_names) > 20:
                legend = False
        else: # color_by_keys != []
            if len(color_dax['idx']) > 20:
                legend=False
            else:
                for i in color_dax['idx']:
                    if not color_override:
                        plot_args['color'] = color_dax['color'][i]
                    if 'alpha' in plot_args.keys() and plot_args['alpha'] < 0.5:
                        plot_args['alpha'] = 1 # for easier viewing
                    p.plot([],[],label=color_dax['group_name'][i],**plot_args)
    if legend:
        p.legend()
    #plot_args={} # mutable default now dealt with in other ways
    return p

color_converter = mpl.colors.ColorConverter()
# possibly not properly imported; well, much weaker "own_fig" understanding
#234567890123456789012345678901234567890123456789012345678901234567890123456789|12345678901234567890
def plot_key_v_key_colored_by_key(dax,key_info,x_key,y_key,by_key,match_list='[]',
                                  exclude_unknowns=False,
                                  x_lim='[]',y_lim='[]', v_lim='[]',
                                  color_scale=plt.cm.inferno, outlier_colors='[default]',
                                  xy_overlap_offset=1, plot_args='{default}',own_fig=True,
                                  prefix='',title_override = ''):
    match_list,x_lim,y_lim,v_lim=anmd(match_list,x_lim,y_lim,v_lim)
    if plot_args=='{default}':
        plot_args={'marker':'.'}
    if outlier_colors=='[default]':
        outlier_colors=['blue','green']
    axis = dax
    restrictions = restriction_title_suffix(match_list,exclude_unknowns)

    idxs = find_idx_matches(axis,match_list,exclude_unknowns=exclude_unknowns)

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

    cc = color_converter
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
    # but needs to just give plt when own_fig...
    # I dislike this way of doing this, but I have to plot a different one ...
    pt = ax.scatter([],[],c=[],vmin=v_min,vmax=v_max,cmap=color_scale, **plot_args)
    bar = plt.colorbar(pt)

    if not 's' in plot_args.keys():
        plot_args['s'] = 6
#     if not 'edgecolors' in plot_args.keys():
#         plot_args['edgecolors'] = 'k'
    ax.scatter(xs,ys,c=colors, **plot_args)
    
    titlebase = f"{prefix} {y_key} vs. {x_key} colored BY {by_key}"
    if own_fig:
        pp=plt
    else:
        pp=ax
    label_x_y_title_and_lim(x_key,y_key,titlebase,pp,own_fig,key_info=key_info,
                            restrictions=restrictions,x_lim=x_lim,y_lim=y_lim,
                            title_override=title_override)
    low_outlier, high_outlier = '',''
    if min(ratios) < 0.0:
        low_outlier = f" <{v_min}={outlier_colors[0]}"
    if max(ratios) > 1.0:
        high_outlier = f" >{v_max}={outlier_colors[1]}"
    bar.set_label(f"{key_info[by_key]['name']} [{key_info[by_key]['units']}]; {low_outlier} {high_outlier}")
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    return ax



# SHOULD GET resid_by_key in here!!!!


# Maybe (?) INCOMPLETE -- was making (or using?) in 20230730_Lp3_debugging
def d_ax_fit_and_resid(ax,key_info,function,x_key_or_keys,y_key, p0, bounds=([],[]),
                  match_list='[]', exclude_unknowns=False,
                  own_fig=True, plot_which_x_variable=-42,  
                  label='', suptitle='',legend=False, plot_args='{}',
                  save_ys='',save_resid='',x_label='default',y_label='default'):
    '''Note you should really supply a key info to save ys/resid'''
    match_list,plot_args=anmd(match_list,plot_args) 
    idxs = find_idx_matches(ax,match_list,exclude_unknowns=exclude_unknowns)
    if type(x_key_or_keys) == str:
        xs = ax[x_key_or_keys][idxs]
        if x_key_or_keys in key_info.keys():
            x_lab = f"{key_info[x_key_or_keys]['name']} [{key_info[x_key_or_keys]['units']}]"
    else:
        xs = [ax[key][idxs] for key in x_key_or_keys]
        if x_key_or_keys[plot_which_x_variable] in key_info.keys():
            x_key = x_key_or_keys[plot_which_x_variable]
            x_lab = f"{key_info[x_key]['name']} [{key_info[x_key]['units']}]"
        else:
            x_label= x_key_or_keys[plot_which_x_variable]
    ys = ax[y_key][idxs]
    if y_key in key_info.keys():
        y_lab = f"{key_info[y_key]['name']} [{key_info[y_key]['units']}]"
    else:
        y_lab = y_key
    
    if x_label:
        if not x_label == 'default':
            x_lab = x_label
    else:
        x_lab=''
    if y_label:
        if not y_label == 'default':
            y_lab = y_label
    else:
        y_lab=''    
        
    # (prm,cov,pred_y,resid,fp,rp) # this in my numpy_plotting
    to_return = fit_and_resid(function,xs,ys,p0,bounds=bounds,own_fig=own_fig, 
                         plot_which_x_variable=plot_which_x_variable,
                         label=label, plot_args=plot_args,legend=legend,
                         x_label=x_lab, y_label=y_lab, suptitle=suptitle)
    for i in range(2):
        savee = [save_ys,save_resid][i]
        ax_len = len(idxs)
        for key,val in ax.items(): # if there WAS a splice:
            ax_len = len(val)
            break
        if savee:
            if not savee in ax.keys():
                ax[savee] = np.full((ax_len,),np.nan,dtype=type(ax[y_key][0]))
            ax[savee][idxs] = to_return[2+i]
            if not key_info == {}:
                if savee in key_info.keys():
                    name ,dca,units = [key_info[savee][key] for key in ['name','type','units']]
                elif y_key in key_info.keys():
                    name ,dca,units = [key_info[y_key][key] for key in ['name','type','units']]
                    if i == 0:
                        name = name + "(fit predicted)"
                    else:
                        name = name + "(fit resid.)"
                else:
                    name, dca,units = savee, 'c','???'
                update_key_info(key_info,savee,name,ax,dca,units)
    return to_return



# INCOMPLETE---was making in 20240523_Instability_finder.ipynb
def plot_correlations_grouped_by(dax,key_info,key_list, by_key, 
                                 match_list='[]',exclude_unknowns=False,
                                 ax_lims='{}',prefix='',plot_args='{}'):
    match_list,ax_lims,plot_args=anmd(match_list,ax_lims,plot_args) 
    nvars = len(key_list)
    for key in key_list:
        if key not in ax_lims.keys():
            ax_lims[key] = []
    fig, ax = plt.subplots(nvars,nvars,figsize=(2.5*nvars,2*nvars),
                           sharex='col',sharey='row')
    fig.subplots_adjust(hspace=0)
    #fig.subplots_adjust(vspace=0)
    #ax[0][0].plot([1,2],[1,2])
    for r in range(nvars):
        for c in range(nvars):
            if c > r and not (r==0 and c==1):
                ax[r][c].axis('off')
                continue
            y_key = key_list[r]
            x_key = key_list[c]

            legend = False
            if r==0 and c==1:
                legend=True
            if c==r: # TODO: MAKE THIS 
                ax[r][c].hist(dax[x_key])
            else:
                pu.dax.plot_key_v_key_grouped_by_key(dax,key_info,x_key,y_key,by_key,
                                  match_list=match_list,exclude_unknowns=exclude_unknowns,
                                  x_lim=ax_lims[x_key],y_lim=ax_lims[y_key], plot_args=plot_args,
                                  own_fig=ax[r][c],legend=legend)
            if c==0:
                lbl = y_key
                if 'units' in key_info[y_key].keys() and key_info[y_key]['units']:
                    lbl= f"{y_key} [{key_info[y_key]['units']}]" 
                ax[r][c].set_ylabel(lbl)#key_info[y_key]['name']) # full name is too long
            if r==nvars-1:
                lbl = x_key
                if 'units' in key_info[x_key].keys() and key_info[x_key]['units']:
                    lbl= f"{x_key} [{key_info[x_key]['units']}]" 
                ax[r][c].set_xlabel(lbl)#key_info[x_key]['name'])
    plt.tight_layout()