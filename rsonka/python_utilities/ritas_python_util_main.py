""" Rita F. Sonka
This file contains relatively simple utility functions that 


phc(length,fill_char,message) #Print Heading Comment
"""


# Import the auxilliary utility files. Directly, because I don't want multiple .'s to calll
#from numpyPlotting import *
#from prettyPrintTable import *
# would add dict_analyzer_pretty in there too. 

# The above doesn't work if your file is in a different directory than the utility folder, apparently.  

import numpy as np
import math


def make_filesafe(filename):
    return filename.replace(" ", "_").replace(".", ",").replace(",,\\", "..\\").replace(",,//","..//").replace("(","").replace(")","")  

# ================ Mutable default arguments are weird.

def anmd(*args): # assign_nonMutable_defaults
    loa = [arg for arg in args]
    nonmutable_defaults = {"{}":{},"[]":[]}
    for i in range(len(loa)):
        try:
            loa[i]=nonmutable_defaults[loa[i]]
        except Exception: #(TypeError, KeyError):
            pass
    if len(loa)==1:
        return loa[0]
    return loa


# ================ DEALING WITH STRING - NUMBER/BOOL CONVERSION (and vice versa)
def is_float(num_string):
    try:
        float(num_string)
        #float(False) == 0, give that a special exception
#         if type(num_string) == bool:
#             return False
        # handling numpy.bool_
        #this is the better way, but taking str() of things takes time....
        if str(num_string) == "True" or str(num_string) == "False":
            return False
        if float(num_string) == 0:
            return True
        # The below handles np.nan and other things like it, hopefully. 
        if (float(num_string) <=0) ^ (0 < float(num_string)):
            return True
        else:
            return False
    except (TypeError,ValueError) as e: # ValueError (originally just "except:" which caused problems on later editing!
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

    
def round_to_sf_str(x, sf,sci_at=4): # **TODO**: compare pretty_print_table's roundToString(num, rTo) someday
    '''A string of x rounded to sf sig figs that actually DISPLAYS any 
    significant 0s at the end, and abides other conventions.
    Uses at most (sci_at-1) non-significant 0s (other than the 0.)
    TODO:
    add ability to make it use a certain exponent if possible, by putting
    some extra sci_notation digits before/after period.'''
    n = sf
    if not is_float(x):
        return x
    x = round_to_sf(x,sf)
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

''' ================ TODO post-thesis Auto_Docstring.ipynb !!! ==============='''


def phc(length,fill_char,message):
    """Prints a 1-line Heading Comment of <length> characters (including the #)
       of entirely <fill_char> except for <message> in the middle,+ surrounding 
       & starting spaces."""
    assert len(message) < length-2, f"ERR: {length}<'{message}'"
    f_size = (length-2-len(message)-2)-2
    flank = fill_char*int(f_size/2)
    f1, f2 = [flank, flank + (1+(f_size %2))*fill_char]
    print(f"# {f1} {message} {f2}")
    
def ls_index(string,substring): # last index, but single character only,
    """Helper function for make_alias()."""
    return len(string)-1 - string[::-1].index(substring)

def make_alias(function_def_str,alias_suffix='',def_indents=0,
               max_chars_per_line=100,indent='    '):
    """Expects a string ful of function definitions in the form ex.:
    def try_to_numerical(num_string):
    def round_to_sf(x, sf):
    def round_to_sf_str(x, sf,sci_at=4):
    Semicolon optional.
    """
    # TODO: make the max_chars_per_line work intelligently
    while function_def_str.count("  ") > 0:
        function_def_str = function_def_str.replace("  "," ")
    #function_def_str = ' '.join(function_def_str.split())
    function_def_str.replace(" def ","def ")
    if function_def_str.count("\ndef") > 1:
        splitty = function_def_str.split("\ndef ")
        to_return = []#[make_alias(splitty[0])]
        for i in range(0,len(splitty)):
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