"""Rita Sonka
1/14/2018
Ge/Ay 117 
Problem Set 2: More Parameter Estimation: Problem 2 confidence code."""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'C:\Python27\myPython\scrapeTQFRS')
import prettyPrintTable as ppt # really just want the string rounding to sig figs.

# This is another person's code.

def msum(iterable):
    "Full precision summation using multiple floats for intermediate values"
    # Rounded x+y stored in hi with the round-off stored in lo.  Together
    # hi+lo are exactly equal to x+y.  The inner loop applies hi/lo summation
    # to each partial so that the list of partial sums remains exact.
    # Depends on IEEE-754 arithmetic guarantees.  See proof of correctness at:
    # www-2.cs.cmu.edu/afs/cs/project/quake/public/papers/robust-arithmetic.ps

    partials = []               # sorted, non-overlapping partial sums
    for x in iterable:
        i = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            hi = x + y
            lo = y - (hi - x)
            if lo:
                partials[i] = lo
                i += 1
            x = hi
        partials[i:] = [x]
    return sum(partials, 0.0)



def linePlotXYwithXlim(xX, yY,  titleString, xlabel, ylabel, xStart, xStop):
    # xX and yY should be numpy arrays.
    plt.plot(xX, yY)
    plt.title(titleString)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xStart, xStop)

def normalizePDF(xs, pofxs, equallySpaced=True):
    # Normalize the distribution. I do this a lot outside of confidence() too. 
    if equallySpaced:
        spacing = (xs[1]-xs[0])
        total = spacing*msum(pofxs)
    else: # Note: using trapezoid rule for non-even normalization.
        total = sum([(xs[i+1]-xs[i])*(pofxs[i]+pofxs[i+1])/2.0 for i in range(len(xs)-1)])
    return pofxs/total 

def normalize2dPDF(xLin, yLin, z, equallySpaced=True):
    # equal spacing asusmed.
    if equallySpaced:
        area = (xLin[1]-xLin[0])*(yLin[1]-yLin[0])
        total= float(area)*np.sum(z)
    '''
    else:
        total = 0.0
        for ri in range(len(z)):
            for ci in range(len(z)):
                total += 
    '''  
    return z/total

                      
class ConfObject:
    ''' A class for representing confidence intervals. '''
    def __init__(self, peak, lower, upper, confAmount, cdf=[]):
        # SHould probably include a warning message for the edge effects! WEll, kinda worked it  into conf.
        self.pk = peak
        self.l = lower
        self.u = upper
        self.conf = confAmount  
        self.cdf = cdf
    def toString(self, r=2):
        '''myStr = "pk@ %.rf in [%.rf, %.rf] w/ %.rf conf".replace('r', str(r))
                return myStr%(round(self.pk,r),round(self.l,r),round(self.u,r),round(self.conf,r))   '''         
        myStr = "pk@ "+ppt.roundToString(self.pk, r)+" in ["+\
            ppt.roundToString(self.l,r)+", "+ppt.roundToString(self.u,r)+\
            "] w/ "+ppt.roundToString(self.conf,r)+" conf"
        return myStr

    def toArray(self):
        return [self.pk, self.l, self.u, self.cdf]

def confidence(xs, pofxs, conf, cdfofx=[], equallySpaced=True, peakIndex='default', plotCDF=False):
    '''Takes a pdf function represented by parameter and probability density 
    arrays xs and pofxs (such that pofx[i] = the probability density at xs[i].).
    Assumes xs is sorted least to greatest. Assumes SINGLE-PEAKED!
    If optional argument cdfofx is included, takes that as the empirical cdf 
    array as defined below in function; otherwise computes it itself.
    If optional argument equallySpaced is set False, uses trapezoidal rule;
    slows calcs down but is necessary if I ever want unequal spacing.
    If optional argument peakIndex != default, will make the confidence interval
    around the given INDEX instead of guessing it itself.
    If optional argument plotCDF=True, displays and plots CDF.
    Calculates the lower and upper bounds of the confidence interval conf
    represented by a float from 0 to 1.
    Returns a Conf Object with peaklocation, lowerbound, upperbound, cdfofx].'''
    assert len(xs) == len(pofxs), "len(xs):%d != len(pofxs):%d"%(len(xs), len(pofxs))
    if len(cdfofx)>0:
        assert len(xs) == len(cdfofx), "len(xs):%d != len(cdfofx):%d"%(len(xs), len(cdfofx))
    assert 0 < conf and conf < 1 , "conf %f not in (0, 1)"%(conf)
    assert len(xs) > 1, "len(xs) must be > 1!"
    pofxs = np.array(pofxs) # Expects numpy array, doesn't assume it.
    # Normalize
    pofxs = normalizePDF(xs, pofxs, equallySpaced=equallySpaced)
    if not len(cdfofx)>0: # Calculate the empirical cdf:
        # Making an array of x values such that: 
        # version where value of cdfofx[i] = cdf(t between x[i] and x[i+1])
        # so cdfofx[len(xs)-2] == ~1, can use that to check
        if equallySpaced:
            spacing = (xs[1]-xs[0])
            #cdfofx = [sum(pofxs[0:i+1])*spacing for i in range(len(xs))]
            cdfofx = [pofxs[0]*spacing]
            for i in range(len(xs)-1):
                cdfofx.append(cdfofx[i] + pofxs[i+1]*spacing)
        else:
            cdfofx = [(xs[1]-xs[0])*(pofxs[0]+pofxs[1])/2.0]
            for i in range(len(xs)-2):
                cdfofx.append(cdfofx[i] + (xs[i+2]-xs[i+1])*(pofxs[i+2]+pofxs[i+1])/2.0)
            ''' # We really want warnings if we TAKE the last pieces.
            if cdfofx[0] > 0.01:
                print "WARNING: first val of CDF=" + str(cdfofx[0])
            if abs(1. - cdfofx[-1]) > 0.01:
                print "WARNING: last val of CDF=" + str(cdfofx[-1])
            '''
            cdfofx.append(1.0)
    if plotCDF:
        plt.clf() # I might someday want to add passing limits to the graph...
        linePlotXYwithXlim(xs, cdfofx,  "cdf", "x", "cdf", xs[0], xs[len(xs)-1])       
        # I tested it and it seems to work in spite of the summing of a ton of small values.
    # Find peak location index; expects a numpy array but doesn't assume it due to earlier cast
    if peakIndex == 'default':
        pkxi = np.where(pofxs==max(pofxs))[0][0] # assumes only one peak exists...
    else:
        pkxi = peakIndex # Sometimes you want something outside to set the peak.
    # Find the confidence interval!
    (li, ui) = (pkxi, pkxi) # loweri upperi
    liValue = cdfofx[li] # Incorporate the probability in the first xval.
    warningVal = 0
    while cdfofx[ui]-liValue < conf:
        if li > 0 and ui < len(xs)-1:
            ldy = cdfofx[li]-cdfofx[li-1]
            udy = cdfofx[ui+1]-cdfofx[ui]
            if ldy > udy:
                li -= 1
                liValue = cdfofx[li]
            else:
                if ui == len(xs)-2:
                    print "WARNING! upper edge taken, val="+str(1.0-cdfofx[ui]) 
                    warningVal += 2
                ui += 1
        elif li == 0. and ui < len(xs)-1:
            if liValue == 0 or liValue < (cdfofx[ui+1]-cdfofx[ui]): # moving ui up might be better than taking the illusory 'last' point.
                ui += 1
            else: # Possibly limit these warnings to only go off if it takes a big chunk of probability?
                print "WARNING! lower edge taken, val="+str(liValue)
                warningVal += 1
                liValue = 0. # Incorporate the prob on first. 
        elif li > 0 and ui == len(xs)-1:
            li -= 1
            liValue = cdfofx[li]
        else:
            print "conf FAIL! li=%d ui=%d"%(li, ui) # Just in case
            return ConfObject(xs[pkxi], xs[li], xs[ui], -4, cdf=cdfofx)
    # Note that if cdfofx is huge, it will look weird when printed.
    if warningVal:
        conf = -(conf + warningVal)
    return ConfObject(xs[pkxi], xs[li], xs[ui], conf, cdf=cdfofx)


# ================================ Example Use 
# Testing against a gaussian.



# Testing against a gaussian =============
def gaussian(center, sigma, x):
    # Note, using the typical normalization
    return np.exp(-(x-center)**2 / (2. * sigma**2))/(np.sqrt(2*np.pi)*sigma)

def testConf():
    # For testing, only going out to 100, and using 10000 points...
    typX = np.linspace(-200, 200, num=10000) 
    gaussPDF = gaussian(5, 3, typX)
    gaussConf = confidence(typX, gaussPDF, 0.68, cdfofx=[], equallySpaced=True, plotCDF=False)
    print gaussConf.toString(r=2)
    # Gets limits [1.9801980198020033, 7.9807980798079825] (with higher r argument).
    # Accurate limits to < 1% is good enough for me!     
    
    
