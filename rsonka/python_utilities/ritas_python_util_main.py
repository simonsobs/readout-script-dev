# Import the auxilliary utility files. Directly, because I don't want multiple .'s to calll
#from numpyPlotting import *
#from prettyPrintTable import *

# The above doesn't work if your file is in a different directory than the utility folder, apparently.  

import numpy as np
import math


def make_filesafe(filename):
    return filename.replace(" ", "_").replace(".", ",").replace(",,\\", "..\\").replace(",,//","..//").replace("(","").replace(")","")  

# ================ DEALING WITH STRING - NUMBER/BOOL CONVERSION (and vice versa)
def is_float(num_string):
    try:
        float(num_string)
        if float(num_string) == 0:
            return True
        #if float(False) == 0, give that a special exception
        if num_string == False: # wait...shouldn't that be True, for 0?
            return False
        # The below handles np.nan and other things like it, hopefully. 
        if (float(num_string) <=0) ^ (0 < float(num_string)):
            return True
        else:
            return False
    except: # ValueError
        return False

def try_to_numerical(num_string):
    if not is_float(num_string):
        return num_string # could not cast
    if float(num_string) == int(float(num_string)):
        return int(float(num_string))
    return float(num_string)
    

def round_to_sf(x, sf):
    # Sig Figs
    if not is_float(x):
        return x
    if x == 0:
        return x
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (sf - 1))

    
def round_to_sf_str(x, sf,sci_at=4):
    '''A string of x rounded to sf sig figs that actually DISPLAYS any 
    significant 0s at the end, and abides other conventions.
    Uses at most (sci_at-1) non-significant 0s (other than the 0.)
    TODO:
    add ability to make it use a certain exponent if possible, by putting
    some extra sci_notation digits before/after period.'''
    n = sf
    x = round_to_sf(x,sf)
    if not is_float(x):
        return x
    if x == 0: # I think this is right?
        return "0." + "0"*(n)
    # highest digit: 3 for 1000, 0 for 1, -1 for 0.1, etc.
    pwr = math.floor(math.log10(abs(x)))
    digits = str(round(float(x)*10.0**(n-pwr-1),0))[:-2].replace("-","") # adds the 0s!
    # 3 options: 
    # largest digit right of ".", (maybe sci notation, and maybe 0's appended right)
    # smallest digit left of ".", (maybe sci notation, and maybe 0's appended right)
    # and "." in item. (may still need 0's appended right)
    # I cannot make the f-string precision a variable!!
    sign = "-"
    if abs(x) == x:
        sign=""
    if pwr < 0: # largest digit right of "."
        if abs(pwr) < sci_at:
            return sign+"0." +"0"*(abs(pwr)-1) + digits
        else:
            dot = ""
            if n > 1:
                dot = "."
            return f"{sign}{digits[:1]}{dot}{digits[1:]}e{pwr}"
    if pwr+1-n >= 0:  # smallest digit left of "."
        if pwr+1-n == 0 and digits[-1] == "0": # special "10." case:
            return sign+digits + "."
        if digits[-1] == "0" or pwr - n >= sci_at-1: # balance vs. negs
            return f"{sign}{digits[:1]}.{digits[1:]}e+{pwr}"
        else:
            return sign+digits + "0"*(pwr+1 - n)
    
    # "." in number:
    return f"{sign}{digits[:pwr+1]}.{digits[pwr+1:]}"




#2345678901234567890123456789012345678901234567890123456789012345678901234567890
# ===================================== META ===================================

def ls_index(string,substring):
    return len(string)-1 - string[::-1].index(substring)

def make_alias(function_def_str,alias_suffix='',def_indents=0,
               max_chars_per_line=80,indent='    '):
    # TODO: make the max_chars_per_line work intelligently
    while function_def_str.count("  ") > 0:
        function_def_str = function_def_str.replace("  "," ")
    #function_def_str = ' '.join(function_def_str.split())
    function_def_str.replace(" def ","def ")
    if function_def_str.count("\ndef") > 1:
        splitty = function_def_str.split("\ndef ")
        to_return = [make_alias(splitty[0])]
        for i in range(1,len(splitty)):
            to_return.append(make_alias("def "+splitty[i],
                                        alias_suffix=alias_suffix,
                                       def_indents=def_indents,
                                       indent=indent,
                                       max_chars_per_line=max_chars_per_line))
        return to_return
    # Now only looking at one function signature.
    function_def_str.replace("\n","")
    fds = function_def_str
    end_idx = ls_index(fds,")")
    argys = fds[fds.index("(")+1:end_idx]
    req_args = []
    opt_args = []
    #print(argys)
    if "=" in argys:
        first_eq = argys.index("=")    
        first_opt_idx = 0
        if "," in argys[:first_eq]:
            first_opt_idx = ls_index(argys[:first_eq],",")
            req_argstr = argys[:first_opt_idx]
            req_args = req_argstr.split(",")
            
            
        # now the optional. optional defaults can have commas.
        opt_argstr = argys[first_opt_idx:]
        i = len(opt_argstr)
        while i > -1 and opt_argstr[:i].count("=")>0:
            eq_i = ls_index(opt_argstr[:i],"=")
            opty_start = 0
            if "," in opt_argstr[:eq_i]:
                opty_start = ls_index(opt_argstr[:eq_i],",")
            opty = opt_argstr[opty_start+1:i].split("=")
            #print(opty)
            opt_args.append(opty)
            i = opty_start
        #print(opt_args)
        opt_args = opt_args[::-1]
    else:
        req_args = argys.split(",")
    # begin construction 
    # Make it wrap by 80 LATER
    to_ret = [def_indents*indent+fds[:fds.index("(")] + alias_suffix \
             + "(" + argys + fds[end_idx:]+"\n"]
    # second line
    to_ret.append((def_indents+1)*indent+"return "+fds[4:fds.index("(")]+  "(")
    to_ret.append(",".join(req_args))
    if len(opt_args)> 0:
        to_ret.append(",")
    to_ret.append(",".join([f"{arg[0]}={arg[0]}" for arg in opt_args]) )
    to_ret.append(")")
    return "".join(to_ret)