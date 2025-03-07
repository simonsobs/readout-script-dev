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

This enables swift selection of data subsets from criteria on
any key(s) of the dict acis, which is then easy to plot.

--------------------
A key_info dict is a dictionary that accompanies one or more
dict-axises, and which tracks important metadata about each 
dict-axis key. It is of the form:

# {<key_name, as used in axes>:{
#      'name': <full name for display>, 
#      'axis': an axis reference. 
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


FUNCTIONS:  (helpers indented)
------------------------------------------------------
ml(mlstr)
    str_to_match_list(mlstr)
        str_to_num_or_bool(strv)
find_idx_matches(s,match_list,dict_return=False,ax_name='crv')
    apply_match(s,ax,idx_list,match)
        match_mask(s,ax,idx_list,match)
            single_match_mask(s,ax,idx_list,match)
    add_dict_level(s,ax,idx_list,level_list,d)
update_key_info(key_info,key,name,axis,dca,units,more_info={})

"""

import numpy as np
from numpy_plotting import *
import python_utilities as pu
from python_utilities.ritas_python_util_main import try_to_numerical
from python_utilities.ritas_python_util_main import is_float

default_figsize=(5,5) # firefox (2.5,2.5) # chrome (5,5) # ac has this too!

t10_colors = plt.cm.tab10(np.linspace(0,1.0,9+1))
t20_colors = plt.cm.tab20(np.linspace(0, 1.0, 19+1))
t20b_colors = plt.cm.tab20b(np.linspace(0, 1.0, 19+1))

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

def find_idx_matches(ax,match_list,dict_return=False, exclude_unknowns=False):
    '''Returns what indices of the ax's arrays match the match_list criteria.
        See Temp_Ramp.find_iva_matches() and str_to_match_list() aka ml()
        set exclude_unkowns to a list of axis keys. It will exclude
        any idxs that have a -42, np.nan, "-", or "?" as that key's value.
        See below this class for the general implementation.'''
    # like find_ivas, but returns idxs. 
    # find the matches
    idx_list = ax['idx']
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
        idx_list = apply_match(ax,idx_list,match)
        
    # organization if necessary
    num_levels=0
    level_list = []
    for match_key,match_type,match_val in match_list:
        if match_type == "=" and (match_val in ['all','any'] or type(match_val) == list):
            num_levels +=1
            level_list.append((match_key,match_type,match_val))
    if not dict_return or num_levels == 0:
        return idx_list
    to_return = {}
    return add_dict_level(ax,idx_list,level_list,to_return)

        
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
        try:
            idx_list_mask[np.where(ax[match_key][idx_list] == match_val)[0]] = True   
        except IndexError:
            print(f"{match_key} {match_val}")
            print(idx_list)
            print(type(idx_list))
    return idx_list_mask


def apply_match(ax,idx_list,match):
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
    




def update_key_info(key_info,key,name,axis,dca,units,more_info={}):
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
    kd = {'name':name,'axis':axis,
          'type':dca, 'units':units}
    for keyn,val in more_info.items(): # CANNOT BE NAMED 'key'!!!Will override argument!
        kd[keyn] = val
    if 'c' in dca:
        # now, extremes, limits
        if 'a' in dca:
            vals=[]
            v_min, v_max = np.inf,-np.inf
            arrs = axis[key][find_idx_matches(axis,[],exclude_unknowns=[key])]
            try:
                for arr in arrs:
                    if min(arr) < v_min:
                        v_min = min(arr)
                    if max(arr) > v_max:
                        v_max = max(arr)
                vals=[v_min,v_max]
            except BaseException as err:
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

        if len(unique_vals) <= 10:
            kd['colors'] = np.array([t10_colors[j,0:3] for j in range(len(unique_vals))])
        elif len(unique_vals) <=20:
            kd['colors'] = np.array([t20_colors[j,0:3] for j in range(len(unique_vals))])
        elif len(unique_vals) <= 40:
            kd['colors'] = np.array([t20_colors[j,0:3] for j in range(20)] \
                                    + [t20b_colors[k,0:3] for k in range(len(unique_vals)-20)])
        else:
            print(f"40+ discrete variable?!? {key}")
            # Color it like we're coloring a continuous variable....kind of.
            # Since this is discrete, it may not be a number.
            # There is most certainly a better way to do this. 
            color_scale = plt.cm.viridis
            v_min, v_max = 0,len(unique_vals)
            ratios = (np.arange(len(unique_vals))-v_min)/(v_max-v_min)
            colors = np.full((len(unique_vals),3), [0.0,0.0,0.0])
            for i in range(len(unique_vals)):
                with_alpha = color_scale(ratios[i]) # I don't know why I called this with_alpha.
                colors[i,:] = np.array([with_alpha[0],with_alpha[1],with_alpha[2]])
            kd['colors'] = colors

    key_info[key] = kd
    return "added"



# ======================== PLOTTING FUNCTIONS ==========================

def d_ax_fit_and_resid(ax,function,x_key_or_keys,y_key, p0,  key_info={},
                  match_list=[], exclude_unknowns=False,
                  own_fig=True, plot_which_x_variable=-42,  
                  label='',suptitle='',legend=False, plot_args={},
                  save_ys='',save_resid=''):
    '''Note you should really supply a key info to save ys/resid'''
    idxs = find_idx_matches(ax,match_list,exclude_unknowns=exclude_unknowns)
    if type(x_key_or_keys) == str:
        xs = ax[x_key_or_keys][idxs]
        if x_key_or_keys in key_info.keys():
            x_label = f"{key_info[x_key_or_keys]['name']} [{key_info[x_key_or_keys]['units']}]"
    else:
        xs = [ax[key][idxs] for key in x_key_or_keys]
        if x_key_or_x_keys[plot_which_x_variable] in key_info.keys():
            x_key = x_key_or_x_keys[plot_which_x_variable]
            x_label = f"{key_info[x_key]['name']} [{key_info[x_key]['units']}]"
        else:
            x_label= x_key_or_x_keys[plot_which_x_variable]
    ys = ax[y_key][idxs]
    if y_key in key_info.keys():
        y_label = f"{key_info[y_key]['name']} [{key_info[y_key]['units']}]"
    else:
        y_label = y_key
    
    # (prm,cov,pred_y,resid,fp,rp) 
    to_return = fit_and_resid(function,xs,ys,p0,own_fig=own_fig, 
                         plot_which_x_variable=plot_which_x_variable,
                         label=label, plot_args=plot_args,legend=legend,
                         x_label=x_label, y_label=y_label, suptitle=suptitle)
    for i in range(2):
        savee = [save_ys,save_resid][i]
        ax_len = len(idxs)
        for key,val in ax.items(): # if there WAS a splice:
            ax_len = len(val)
            break
        if savee:
            ax[savee] = np.full((ax_len,),np.nan,dtype=type(ax[y_key][0]))
            ax[savee][idxs] = to_return[2+i]
            if not key_info == {}:
                if savee in key_info.keys():
                    name ,dca,units = [key_info[save_ys][key] for key in ['name','type','units']]
                elif y_key in key_info.keys():
                    name ,dca,units = [key_info[y_key][key] for key in ['name','type','units']]
                    if i == 0:
                        name = name + "(fit predicted)"
                    else:
                        name = name + "(fit resid.)"
                else:
                    name, dca,units = savee, 'c','???'
                update_key_info(key_info,save_ys,name,ax,dca,units)
    return to_return


#234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
def plot_key_v_key_grouped_by_key(ax,key_info,x_key,y_key,by_key,match_list=[],exclude_unknowns=False,
                                  x_lim=[],y_lim=[], prefix='',plot_args={},
                                  own_fig=True,legend='default'):
    axis = ax
    restrictions = ''
    if exclude_unknowns or len(match_list) > 0:
        restrictions = '\n'
    if exclude_unknowns:
        restrictions = restrictions + 'no unknowns '
        if exclude_unknowns == True:
            exclude_unknowns = [x_key,y_key,by_key]
    if len(match_list) >0:
        restrictions = restrictions + f"[{', '.join([f'{mk}{mt}{mv}' for mk,mt,mv in match_list])}]"
    idxs = find_idx_matches(axis, match_list,exclude_unknowns=exclude_unknowns)
    xs,ys = axis[x_key][idxs], axis[y_key][idxs]

    group_names = np.sort(np.unique(axis[by_key][idxs]))
    if (not by_key in key_info.keys()) or not key_info[by_key]['type'] == 'd':
        # probably going to have other problems, but let it be for now:
        print(f"warning: {by_key} not in key_info or is not discrete!")
        if len(group_names) > 10:
            tab20 = plt.get_cmap('tab20')
            colors = tab20(np.linspace(0, 1.0, len(group_names)+1))
        else:
            tab10 = plt.get_cmap('tab10')
            colors = tab10(np.linspace(0, 1.0, len(group_names)+1))
    else:
        colors = key_info[by_key]['colors'][np.in1d(key_info[by_key]['vals'],group_names).nonzero()]
    p = plt
    if own_fig == True:
        plt.figure(figsize=default_figsize)
    elif not own_fig == False:
        p= own_fig

    for i in range(len(group_names)):
        name = group_names[i]

        #name_color = key_info[by_key]['colors'][np.where(key_info[by_key]['vals'])[0][0],:]
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
        p.title(f"{prefix} {y_key} v {x_key} BY {by_key}\n{restrictions}")
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
    if legend=='default':
        if len(group_names) > 20:
            legend = False
    if legend:
        p.legend()
    return p