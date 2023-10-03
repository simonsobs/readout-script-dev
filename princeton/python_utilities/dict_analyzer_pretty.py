"""
Rita Sonka 3/30/2022
Made to create reference illustrations of the structure of .npy files 
and other large dictionaries. 
"""
import sys
import numpy as np


def pdap(d, max_line=80, ignore_type='any', ignore_after=25, 
         do_padding=True, preview_val='fast', s_key="", starting_space=""):
    """ Runs dict_analyzer_pretty with given arguments and prints
    the output. """
    print(dict_analyzer_pretty(d, max_line=max_line, ignore_type=ignore_type, 
                               ignore_after=ignore_after, do_padding=do_padding,
                               preview_val=preview_val,
                               s_key=s_key, starting_space=starting_space))

  

#234567890123456789012345678901234567890123456789012345678901234567890123456789
def dict_analyzer_pretty(d, max_line=80, ignore_type='any', ignore_after=50, 
                         do_padding=True,preview_val='fast',s_key="", 
                         starting_space=""): #round_to_sf=False
    """ Trawls through a large nested dictionary and creates a string to 
    illustrate its structure, reporting each key, the type() of that key's
    value, and a preview of the string representation of that key's value;
    except, it ignores excess (> ignore_after) keys of ignore_type on
    a given dictionary level. 
    ---- Args -------
    d               : dict : the dictionary to traverse
    max_line        : int  : line character # to end value previews at
    ignore_type     : type : Ignore excess keys of this type @ one dict level
                           : NOTE: may be "numpy.int64" or similar!
                           : MAY BE string 'all', to not care about type.
    ignore_after    : int  : # of keys of ignore type to show (on a given 
                           : dict level) before ignoring
    do_padding      : bool : Pad to align colons like in this docstring?
    preview_val     : ??   : Display val preview? True, False, or fast
                           : immutable is a cautious estimate.
    s_key           : ??   : "start key," recursive parameter.
    starting_space  : str  : recursive parameter.

    ---- Returns ----
    A string that illustrates the dictionary,  of the following form,
    with all colons on a given level aligned via space padding if
    do_padding=True (assume  below that keys don't have dictionary
    values unless specified):
    
    {
        key1 : type(key1) <preview of key1>
        key2 : type(key2) <preview of key2>
        key3 : dict       {
            subkey1 : type(subkey1) <preview of subkey1>
            subkey2 : type(subkey2) <preview of subkey2>
            subkey3 : dict          {
                subsubkey1 : type(subsubkey1) <preview of subsubkey1>
            } #(subkey3)
        } #(key3) 
    } #()
    
    except if you set the ignore settings and it ignores keys, it will report
    how many were ignored in a given dictionary in parentheses after the 
    #(dictkeyname) following that dictionary preview's closing bracket.
    """
    if len(d.keys()) == 0: # only if starting dict was empty
        return "{}"
    ss = starting_space + "    "
    string = " {\n" #str(start_key) +
    max_key_len    = max([len(str(key)) for key in d.keys()])
    max_type_len   = max([len(str(type(d[key]))[8:-2]) for key in d.keys()])
    base_line_len  = len(ss) + max_key_len + 1 + max_type_len + 2
    value_space    = max(max_line - base_line_len, 3)
    seen_ignores   = 0
    key_space, type_space = " ", " " # for if do_padding turned off
    for key in d.keys():
        if ignore_type=='all' or ignore_type=='any' or type(key) == ignore_type:
            seen_ignores += 1
            if seen_ignores > ignore_after:
                continue
        if do_padding:
            key_space  = " " * (max_key_len  - len(str(key)))
            type_space = " " * (max_type_len - len(str(type(d[key]))[8:-2]))
        if type(d[key]) == dict and len(d[key].keys()) > 0:
            # f strings were being annoying about this for some reason.
            #print(f"dap(starting_space={starting_space} s_key={s_key})")
            #return "what is going on"
            string = string + ss + str(key)  + key_space  + ":"  + \
                     str(type(d[key]))[8:-2] + type_space + " "  + \
                     dict_analyzer_pretty(d[key], s_key=key, starting_space=ss,
                                          do_padding=do_padding,
                                          max_line=max_line,
                                         ignore_type=ignore_type,
                                         ignore_after=ignore_after,
                                          preview_val=preview_val) + "\n"
            
        else:
            # a sufficiently huge dereferenced d[key] has slow str(d[key])            
            # AH! python only calcs first bit if I only ask for that! ...NOPE.
            if preview_val == "fast":
                # the point is to not slow the function by 
                # having it calculate str(<something massive>)
                va = preview_fast(d[key],'',value_space=value_space)
#                 print(f"post_preview_fast {va}")
            elif preview_val == True:
                va = str(d[key])[:value_space] # used to have a +2 after value space, not sure why...
            else: #no preview
                va = ''
            if len(va) > value_space:
                va = va[:value_space-3] + "..."
            string = string + ss + str(key)  + key_space  + ":"  + \
                     str(type(d[key]))[8:-2] + type_space + " "  + \
                     va + "\n" 
    string = string + starting_space + "} #("+ str(s_key) + ")"
    if seen_ignores > ignore_after:
        string = string + " (ignored " + str(seen_ignores-ignore_after) + " " + \
                 str(ignore_type)[8:-2] + ")"
    return string

def preview_fast(val,valstr,value_space=0,count=0):
    # valstr should start off as '', even if it was a str.
#     print(f"preview_fast {count} {type(val)} {valstr} ")
#     if count >= 3:
#         return f"preview_fast called {3} times" 
    if len(valstr) > value_space:
        return valstr[:value_space-3] + "..."
    # immutable, non-iterable types, can't be too huge. Probably.
    for typy in [int, float, complex]:
        try:
            typy(val)
            va = str(val)
#             print(va)
            if len(va) > value_space:
                va = va[:value_space-3] + "..."
            return va
        except:
            pass
    # Now, the immutable iterable types.
    # can't catch similar castables here.
    if type(val) == str or type(val)==np.str_: # no calculation, it's a str already.
        if len(val) <= value_space:
            return val
        else:
            return val[:value_space-3] + "..."
    for typy in [tuple, frozenset,bytes]:
        if type(val) == typy:
            valstr1 = str(typy())[:-1]
            end = str(typy())[-1]
            return preview_special_iter(val,valstr1,end,value_space,count=count+1)
     # I'm also going to try fast-previewing lists & equivalent
    # sometimes dictionaries are values of lists.
       
    #if type(val) == dict():
    
    # Could also try something more general.
        
#     try:
    if type(val) == list or type(val) == type(np.array([])): # probably a list equivalent
        #list(val)
        # not trying to get perfect for np.ndarray
        
        valstr1 = '['
        end = ']'
        return   preview_special_iter(val,valstr1,end,value_space,count=count+1)
#         print(f"list: post preview_special_iter {checky}")
#         return checky
#     except TypeError:
#         pass
    no_prev = f"**NO PREV:{str(type(val))[8:-2]}**"
    if len(no_prev) <= value_space:
        return no_prev
    return no_prev[:value_space-3]+"..."
            
                
                

def preview_special_iter(val,valstr1, end, value_space,count=0):
#     print(f"preview_special_iter {count} {valstr1}")
#     if count > 10:
#         return f"preview_special_iter called {10} times" 
    mi = iter(val)
    finished = False
    try:
        while len(valstr1)  <= value_space:
            valstr1 = valstr1  +\
                      preview_fast(next(mi),valstr1, #-len(valstr1)
                                   value_space=value_space,count=count+1)\
                      + ", "
#             print(f"iter: {valstr1}")
            
    except StopIteration:
        finished = True
        pass
#     print(valstr1)
    valstr1 = valstr1[:-2] + end # remove last comma and space
    if len(valstr1) > value_space or not finished:
        return valstr1[:value_space-3] + "..."
    return valstr1
                
                
                

                
                
                
        


# Ultimately concluded that there was no way to do this in python, unfortunately. 
# def get_val_preview(val,space,prev=''):
#     # does not actually cut to space.
#     if not is_iter(val): 
#         return str(val)
#     if type(val) == str:
#         return val
#     # is a non-string iterable. 
#     #type(iterable)() doesn't work for numpy stuff because they require a shape parameter
#     if len(get_val_preview(next(iter(val))
        

# def is_iter(val):
#     try:
#         iter(val)
#     except:
#         return False
#     return True

