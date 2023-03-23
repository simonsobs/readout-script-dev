"""Rita Sonka
2/7/2018
Ge/Ay 117 
Problem Set 5: Markov Chain Monte Carlo """

#from decimal import *

import sys, os
sys.path.insert(0, 'C:\Python27\myPython')
import numpyPlotting as pl
import Rita_Sonka_confidence as conf
sys.path.insert(0, 'C:\Python27\myPython\scrapeTQFRS')
import prettyPrintTable as ppt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def noInvalid(params, otherData):
    return True

np.set_printoptions(suppress=True)

def optimizeFromMCMC(mcmcInst, r=3):
    opt_guess = mcmcInst.myMin.x
    hv = mcmcInst.myMin.hess_inv
    error_opt_guess = np.sqrt(np.array([hv[0][0], hv[1][1], hv[2][2], hv[3][3]]))  
    p1aReportTitles = ["Param", "opt_est_value", "opt_est_1Derr"]
    p1aReport = ppt.roundTableContentsToSigFig([p1aReportTitles] + [[pNames[i], opt_guess[i], error_opt_guess[i]] for i in range(4)], r)
    ppt.prettyPrintTable(p1aReport)    

'''
# Problem 1 ==================

def get_model(params, x): ##evaluate the y-values of the model, given the current guess of parameter values
    (slope, intercept) = params
    model = x*slope+intercept ##insert equation for your model to calculate y-values
    return model

def quadratic(params, x):
    (a, b, c) = params
    model = a*x**2 + b*x+c
    return model
'''
  
 # HEY!. NOTE WHAT IT GIVES AS THE PEAK IS SET BY WHICH FUNCTION IT USES TO GET IT!
# Need to look at the function assignConfidences

class MCMC:
    ''' A class for performing and storing simple MCMC results.  
    OPTIONAL SETTINGS:
    acceptTarget         [min, max] acceptable percentages for accepts. 
    checkStepsizesEvery  number steps to wait before updating stepsizes
    confLevel            confidence to do conf intervals on
    plotTitle            plot titles = plotTitle+notes (maybe plotType for some non-obvious)
    savePath             appended to front of files saved.
    bestFitAxis          ['xaxisName', 'yaxisName']
    
    GIVEN INSTANCE VALUES: 
    startParams               starting params
    paramNames                paramNames, important for display
    stepsizes                 stepsizes currently being used.
    xX                        xData 
    yY                        yData
    yErr                      yError
    get_model(params, x)      function for calculating y vals according to model.
    errorFunc(params, yErr)   function for calculating sigmas from yErr and maybe params, or False if just yErr should be used.
    validFunc(params,other)   checks if proposed steps are valid
    
    DERIVED/UPDATED INSTANCE values:
    numSteps          total steps run on this, including init conditions.
    numP              number of parameters
    myMin             Result of scipy minimization of data. 
    bestFitParam      Current choice of best fit param. 
    bI                burnIndex; Analysis/plot functions ignore data before it.                     
    myConf            (#params) array of ConfObjects
    paramDict         Compilation of info on params.
    chain             (#params x numSteps) array of param values at steps,
    acceptChain       (#params x numSteps) array of acceptance notes: 
                          1 for proposed&accepted, -1 for proposed&rejected, 
                          0 for not-perturbed-this-step
    chi2termChain     (numSteps) array of chi2term_vals of recorded steps
    logLikelihoodChain   (numSteps) array of likelihood_vals of recorded steps
    stepsizesChain    (#params x numSteps/checkStepSizesEvery) 
    stepsizesPercents (#params x numSteps/checkStepSizesEvery) init w/ -1/-1
    myParamAcceptPer  (#params) 
    
    PLOTS:
    logLikelihoodPlot dataSetXY instance
    tracePlot    myPlot instance, (#params) plotting. paramVal vs. stepnum
    histPlot     myPlot instance, (#params) plotting. paramHist
    dens2dPlot   myPlot instance, (#params x #params) 2dDensity staircase plot.
    bestFitPlot  MySinglePlot instance.
    '''
    

    def __init__(self, params, paramNames, stepsizes, x, y, yerr, getModel, nSteps, \
                 acceptTarget=(0.2, 0.25), checkStepsizesEvery=100, confLevel=0.68, \
                 errorFunc=False, validFunc=noInvalid,\
                 plotTitle='', savePath='C:\Users\Rita\Documents\SchoolStuff\CaltechYear4Term2Winter2018\GeAy117BayesianStatistics\set6\graphics',
                 printPlotTitle='default', paramUnits='default', bestFitAxis=['xdata', 'ydata']):
        # Settings
        self.acceptTarget = acceptTarget
        self.checkStepsizesEvery = checkStepsizesEvery
        self.confLevel = confLevel
        self.plotTitle = plotTitle
        self.savePath = savePath
        self.errorFunc = errorFunc # admittedly this is a function if set
        self.validFunc = validFunc # accepts everything if not set.
        if printPlotTitle == 'default':
            if len(plotTitle) > 10:
                self.printPlotTitle=False
            else:
                self.printPlotTitle=True
        else:
            self.printPlotTitle=printPlotTitle
        self.bestFitAxis=bestFitAxis
        # DO SOMETHING WITH PARAM UNITS SO THEY DON'T have to display in titles!!!
        # Given values
        self.startParams = params
        self.paramNames = paramNames
        self.stepsizes = stepsizes
        self.xX = x
        self.yY = y
        self.yErr = yerr
        self.get_model = getModel
        # Calculated. Right now I include the init values in the chains.
        self.numSteps = 1
        self.numP = len(self.startParams)
        # myMin sometimes takes estimates from stuff, so done later
        self.bestFitParam = params
        self.bI = 0 # Until proven otherwise.
        self.myConf = [0]*self.numP # NOT numpy array, has to change vals!
        self.paramDict = {}
        self.chain = np.transpose([np.copy(params)]) 
        self.acceptChain = np.zeros((self.numP, 1), dtype=int) # Didn't really propose against anything here, so 0,0 for start.
        #self.chi2termChain = [self.get_chi2term(params)]
        self.logLikelihoodChain = [self.get_logLikelihood(params)]
        self.stepsizesChain = np.transpose([np.copy(stepsizes)])
        self.stepsizesPercents = np.transpose([[-1.0]*self.numP])
        if nSteps > 0:
            self.run_MCMC_for(nSteps)
            
        self.setPeakParams() # It's possible these shouldn't be done here.
        self.runMinimization()
        self.setMinParams()
        # I don't call analyzeResults() or makeAllPlots() here because
        # all of your MCMC data is thrown out if __init__ doesn't complete.
        # runMinimization() fails with grace.
        # NOte also that this is NOT the best! For MCMC with enough points (100000
        # typically sufficient), 
        

    # Utility ==================================================================
    # Note: calculations in the 
    # See also toLine...
    def toString(s, r=2):
        myStr = "MCMC w/ %d steps\n" %(s.numSteps)
        # For code legibility, the following does not use a list comprehension:
        # tableStats
        tS = [["Param name", "value est", "stepsize", "TotAcceptRate"]] 
        acceptS = s.myParamAcceptPer
        for pI in range(s.numP): # paramIndex
            tS.append([s.paramNames[pI], s.myConf[pI].toString(r), \
                       s.stepsizes[pI], acceptS[pI]])
        myStr +=ppt.makePrettyTableString(ppt.roundTableContentsToSigFig(tS, r))
        return myStr
    
    '''
    # Something that saves/imports the monteCarlo results could be good.
    def toArray(self): 
        return [self.pk, self.l, self.u, self.cdf]
    '''    
    
    # Plots! ===================================================================
    def makeDiagnosticPlots(s):
        #s.makeChi2TermPlot()
        s.makeLogLikelihoodPlot()
        s.makeTracePlot()
        
    
    def makeAllPlots(s, save="pdf", shape='default'):
        # 
        # shape=separate makes the plots separately. Still saves to a plot 
        # container.
        s.makeTracePlot(shape=shape, save=save)
        s.makeHistPlot(shape=shape, save=save)
        s.make2dDensityPlot(save=save)
        s.makeBestFitPlot(save=save)
        #s.makeChi2TermPlot()
        s.makeLogLikelihoodPlot(save=save)     
    
    def saveThePlot(s, fullPlotTitle, save="pdf"):
        if save:
            pl.fileExtension=save # figure needs to be up, obviously.
            pl.saveFig(plt, os.path.join(s.savePath, fullPlotTitle))  
        
    def makeChi2TermPlot(s, save="pdf"):
        # Deprecated.
        plt.clf()
        title = ''
        if s.printPlotTitle:
            title = s.plotTitle   + ", "      
        title += "Chi2term"
        ylab = "Chi2term" 
        s.chi2termPlot = pl.XYdataSet(range(s.bI, s.numSteps), 
                                      s.chi2termChain[s.bI:], 
                                      xlabel="numSteps", \
                                      ylabel=ylab,title=title)
        s.chi2termPlot.plotMeSolo(plt)
        if not s.printPlotTitle:
            title = s.plotTitle + ", "  + title 
        s.saveThePlot(s.plotTitle+ "_Chi2term", save=save)
        
    def makeLogLikelihoodPlot(s, save="pdf"):
        # Possibly make this not automatically cut the burn in? And show points, not a line?
        plt.clf()
        title = ''
        if s.printPlotTitle:
            title += s.plotTitle  + ", "      
        title += "LogLikelihood"
        
        ylab = "LogLikelihood" 
        s.logLikelihoodPlot = pl.XYdataSet(range(s.bI, s.numSteps), 
                                      s.logLikelihoodChain[s.bI:], 
                                      xlabel="numSteps", \
                                      ylabel=ylab,title=title)
        s.logLikelihoodPlot.plotMeSolo(plt)
        if not s.printPlotTitle:
            title = s.plotTitle + ", " + title        
        s.saveThePlot(s.plotTitle+ "_LogLikelihood", save=save)
        
    def makeTracePlot(s, shape='default', save="pdf"):
        # Add stuff to display burn in!
        if isinstance(shape, list):
            singlePlots = shape
        else: 
            singlePlots = [range(s.numP)]
        for rI in range(len(singlePlots)):
            for cI in range(len(singlePlots[rI])):
                pI = singlePlots[rI][cI]   
                title = ''
                if s.printPlotTitle:
                    title += s.plotTitle  + ', '
                title += s.paramNames[pI]
                ylab = s.paramNames[pI] + " value" 
                singlePlot =  pl.makeSinglePlotFromDataSetObj(\
                        pl.XYdataSet(\
                            range(s.bI, s.numSteps), s.chain[pI][s.bI:], xlabel="numSteps", \
                                 ylabel=ylab,title=title))
                if shape == 'separate':
                    plt.clf()
                    singlePlot.plotMeSolo(plt)
                    if not s.printPlotTitle:
                        title = s.plotTitle + ", " + title    
                    s.saveThePlot(s.plotTitle+"_Trace_"+s.paramNames[pI], save=save)                
                singlePlots[rI][cI] = singlePlot
        s.tracePlot = pl.myPlot(singlePlots)
        if shape != 'separate':
            s.tracePlot.plotMe(plt)
            if not s.printPlotTitle:
                title += ", " +s.plotTitle     
            s.saveThePlot(s.plotTitle+"_Trace_", save=save)  
    
    def binPick(self, start, stop):
        return max(min(150, (stop-start)/600), 5)
    
    def makeHistPlot(s, shape='default', save="pdf"):
        if isinstance(shape, list):
            singlePlots = shape
        else: 
            singlePlots = [range(s.numP)]
        for rI in range(len(singlePlots)):
            for cI in range(len(singlePlots[rI])):
                pI = singlePlots[rI][cI]
                title = ''
                if s.printPlotTitle:
                    title += s.plotTitle + ", "                
                title += s.paramNames[pI]
                (xlab, ylab) = (s.paramNames[pI], "probability density" )
                bins = s.binPick(0, s.numSteps)
                singlePlot = pl.makeSinglePlotFromDataSetObj(\
                        pl.XYdataSet(s.chain[pI][s.bI:], range(s.bI, s.numSteps), \
                                     xlabel=xlab, \
                            ylabel=ylab,title=title, plotType='hist', \
                        plotTypeArgs={'bins':bins,'normed':True},\
                        otherArgs={'vlines':[s.myConf[pI].l, s.myConf[pI].pk, s.myConf[pI].u]}))# 'density' doesn't work for some reason. I should update matplotlib?
                if shape == 'separate':
                    plt.clf()
                    singlePlot.plotMeSolo(plt)
                    if not s.printPlotTitle:
                        title = s.plotTitle + ", " + title                   
                    s.saveThePlot(s.plotTitle+"_Hist_"+s.paramNames[pI], save=save) 
                singlePlots[rI][cI] = singlePlot               
        s.histPlot = pl.myPlot(singlePlots)
        if shape != 'separate':
            s.histPlot.plotMe(plt)
            if not s.printPlotTitle:
                title = s.plotTitle + ", " + title  
            s.saveThePlot(s.plotTitle+"_Hist", save=save) 
    
    
    def make2dDensityPlot(s, save="pdf"):
        # It could still use a bit of tuning with multiple density pairs.
        #pairIndexMatrix = [[(i1, i2) for i2 in range(i1)] for i1 in range(s.numParams)] 
        singlePlotsMatrix = []
        for pI1 in range(s.numP):
            singlePlotsMatrix.append([])
            for pI2 in range(pI1):
                #title = "HistPlot, "+s.paramNames[pI]+", "+ s.plotTitle
                kwordargs = {}
                if pI1 == s.numP-1:
                    kwordargs['xlabel'] = s.paramNames[pI2]
                if pI2 == 0:
                    kwordargs['ylabel'] = s.paramNames[pI1]
                #(xlab, ylab) = (s.paramNames[pI2], s.paramNames[pI1])
                bins = [s.binPick(s.bI, s.numSteps), s.binPick(s.bI, s.numSteps)]
                singlePlotsMatrix[pI1].append( \
                    pl.makeSinglePlotFromDataSetObj(\
                        pl.XYdataSet(\
                            s.chain[pI2][s.bI:], s.chain[pI1][s.bI:], plotType='hist2d', \
                        plotTypeArgs={'bins':bins,'normed':True},\
                        **kwordargs)))
        # Leaving off the diagonal means there's one empty start row:
        singlePlotsMatrix = singlePlotsMatrix[1:]
        supTitle = ''
        if s.printPlotTitle:
            supTitle += s.plotTitle + ", "        
        supTitle = "2dDensity" + supTitle
        s.dens2dPlot = pl.myPlot(singlePlotsMatrix,title=supTitle, sharex=True, sharey=True)
        s.dens2dPlot.plotMe(plt)
        #plt.colorbar() # This might need to come out.      
        s.saveThePlot(s.plotTitle+"_2dDensity", save=save)      
    
    def makeBestFitPlot(s, save="pdf"):
        fig = plt.figure()
        plt.clf() #unlike myPlot, singlePlot does not automatically clf.
        #baseDataSet = pl.XYdataSet(s.xX, s.yY, legendName='data', plotType='errorbar', plotTypeArgs={'marker':'.','yerr':s.yErr,'linestyle':'None'})
        #modelDataSet = pl.XYdataSet(s.xX, s.get_model(s.getBestFitParams(), s.xX), legendName='bestFit', color='r', plotTypeArgs={'linestyle':'None','marker':'o'})
        baseDataSet = pl.XYdataSet(s.xX, s.yY, legendName='data', plotType='errorbar', plotTypeArgs={'linestyle':'-','yerr':s.yErr,'linestyle':'None'})
        modelDataSet = pl.XYdataSet(s.xX, s.get_model(s.getBestFitParams(), s.xX), legendName='bestFit', color='r', plotTypeArgs={'linestyle':'-'})        
        title = 'BestFit'
        if s.printPlotTitle:
            title += ", " + s.plotTitle         
        s.bestFitPlot = pl.MySinglePlot([baseDataSet, modelDataSet], title=title, xlabel=s.bestFitAxis[0], ylabel=s.bestFitAxis[1])
        s.bestFitPlot.plotMeSolo(plt)
        if not s.printPlotTitle:
            title += ", " + s.plotTitle         
        s.saveThePlot(s.plotTitle+"_bestFit", save=save)   
    
    # Stat calcs ===============================================================
    
    def analyzeModel(s, r=3):
        # Assumes you have put your best fit solution in bestFitParam already
        # ~1 reduced Chi2 is good; 1.1-1.2 ish. Easier with large data sets.
        reducedChi2 = s.get_chi2(s.bestFitParam)/(len(s.xX)-s.numP)
        # leave one out method is too much for this...
        # Akaike Information Criterion. Lowest best. Very commonly used.
        lmax = s.get_logLikelihood(s.bestFitParam)
        aic = 2.*s.numP - 2.*lmax
        # "Bayesian" information criterion: # Crossover aith AIC @ about n=8?
        # In exoplanet literature, a "B"IC difference of ~10 taken to be statistically significant
        bic = s.numP*np.log(len(s.xX))-2.*lmax
        bestFitParamsUsed = ppt.roundTableContentsToSigFig([s.bestFitParams], r)[0] 
        info = [s.plotTitle, reducedChi2, aic, bic, bestFitParamsUsed]
        tS = [["Model Measures", "reduced Chi2", "AIC", '\"B\"IC', 'best Fit Params Used'], info]
        myStr = ppt.makePrettyTableString(ppt.roundTableContentsToSigFig(tS, r))
        print myStr
        return (info, myStr)
    
    def analyzeResults(s, r=3, setPeak=False):
        print "Note: bestFitparams set via setPeakParams or setMinParams"
        print "Computing full accept Rates."
        s.myParamAcceptPer = s.myParamAcceptRates()
        print "Computing best fit and Confidence intervals."
        s.assignConfidences(s.bI, s.numSteps, setPeak=setPeak) 
        if not setPeak:
            print "Re-optimizing with distribution peak as init guess."
            s.setConfidenceParams()
            s.runMinimization()
        print s.toString(r=r)
        s.reportOptimization(r=r)
        s.analyzeModel(r=r)
        s.makeParamDict(r=r)
        line = s.toLine()
        table = [['plotTitle', 'steps used', 'optCapture', 'mcPeakLL', 'optPkLL', 'histPkLL', 'reduced Chi2', 'AIC', '\"B\"IC', 'paramDict'], line]
        ppt.prettyPrintTable(ppt.roundTableContentsToSigFig(table, 6), maxCellSize=10000)        
        
        
    
    def makeParamDict(s, r=4):
        '''This is for printing and saving results, hence the ppt use. 
        Can access all this information through appropriate vars/functions.'''
        oldBest = s.bestFitParams # Reset this at the end.
        # Get the MCMC peak results
        print "ASSUMING YOU HAVE ALREADY RUN analyzeResults() w/setPeak=False (default)"
        s.paramDict = {}
        s.setPeakParams() # Everything else is accessible somewhere else.
        for i in range(s.numP): # Compile the sub dict:
            subDict = {}
            cnf = s.myConf[i]
            subDict["conf"] = ppt.roundDictToSF({"pk":cnf.pk, "w/in":[cnf.l, cnf.u], "cnf":cnf.conf}, r)
            try:
                std = np.sqrt(s.myMin.hess_inv[i][i])
            except:
                std = "ERR"
            subDict["opt"] = ppt.roundDictToSF({"pk":s.myMin.x[i], "var": s.myMin.hess_inv[i][i], "std":std}, r)
            subDict["chain"] = ppt.roundDictToSF({"maxLpk":s.bestFitParams[i], "stpsz":s.stepsizes[i], "TotAccept":s.myParamAcceptPer[i]}, r)
            s.paramDict[s.paramNames[i]] = subDict    
        s.bestFitParams = oldBest
        
    def toLine(s, r=4):
        # A big one for saving things. Assumes you've run makeDict! 
        # ['plotTitle', 'steps used', 'optCapture', 'mcPeakLL', 'optPkLL', 'histPkLL', 'educed Chi2', 'AIC', '\"B\"IC', 'paramDict']
        # optCapture: (% of params that, without using setPeak, is the opt minimizer result within the MCMC error range?)
        # The 3 model strength values are calculated for the highest LL of mcPeak, optPk, histPk
        # Remember: LOG likelihood can be negative! Will be if likelihood less than 1...
        # [plotTitle, steps used, optCapture, mcPeakLL, optPkLL, histPkLL, reduced Chi2, AIC, "B"IC, paramDict]
        line = [s.plotTitle, s.numSteps]
        num = 0.0
        s.assignConfidences(s.bI, s.numSteps, setPeak=False)  # Just in case
        for i in range(s.numP):
            #confy = s.paramDict[s.paramNames[i]]["conf"]
            #[l, u] = confy["w/in"]
            (l, u) = (s.myConf[i].l, s.myConf[i].u)
            if  l < s.myMin.x[i] and s.myMin.x[i] < u:
                num += 1.
        line += [num/float(s.numP), max(s.logLikelihoodChain), -s.myMin.fun, s.get_logLikelihood([s.myConf[i].pk for i in range(s.numP)])]
        (info, myStr) = s.analyzeModel(r=r)
        line += info[1:-1] +[s.paramDict]
        return line    
            
    def runMinimization(self):
        '''THIS IS NOT FULLY IMPLEMENTED, TESTED OR USED.'''
        # Make sure to change this if I switch to LL!
        guessParam = self.bestFitParam
        #self.myMin = minimize(self.get_chi2, guessParam) #, args=(TIME, OBLIQ, OBLIQ_ERR)
        self.myMin = minimize(self.get_negLogLikelihood, guessParam) 
   
    def reportOptimization(s, r=3):
        opt_guess = s.myMin.x
        hv = s.myMin.hess_inv
        error_opt_guess = np.sqrt(np.array([hv[i][i] for i in range(s.numP)]))  
        repTitles = ["Param", "opt_est_value", "opt_est_1Derr"]
        report = ppt.roundTableContentsToSigFig([repTitles] + [[s.paramNames[i], opt_guess[i], error_opt_guess[i]] for i in range(s.numP)], r)                 
        ppt.prettyPrintTable(report) 
        return report   
    
    def setPeakParams(self):
        # Make sure to change this if I switch to LL!
        # Returns an np array, by the way. The first [0] just gets it out of the
        # tuple, the second [0] strips the excess layer no longer relevant
        # after the transpose.
        #self.bestFitParam = np.transpose(self.chain[:, np.where(self.chi2termChain[self.bI:] == max(self.chi2termChain[self.bI:]))[0]])[0]
        #print "maxChi2term: " + str(max(self.chi2termChain[self.bI:]))
        #print "minChi2term: " + str(min(self.chi2termChain[self.bI:]))   
        self.bestFitParam = np.transpose(self.chain[:, np.where(self.logLikelihoodChain[self.bI:] == max(self.logLikelihoodChain[self.bI:]))[0]])[0]
        #print "PeakChiParams assigned: " + str(self.bestFitParam)
        print "Peak LogLikelihood assigned: " + str(self.bestFitParam)
        
    def setConfidenceParams(self):
        self.bestFitParam = np.array([confy.pk for confy in self.myConf])
    
    def setMinParams(self):
        self.bestFitParams = self.myMin.x
    def setOptParams(self):
        self.bestFitParams = self.myMin.x    
        
    def getBestFitParams(s):
        # Only still here for backwards compatibility.
        return s.bestFitParam
            
        
    def assignConfidencesCounts(s, start, stop, setPeak=False):
        # start, stop are indices.
        #minimizeParam = s.getBestFitParams() # TURN THIS BACK ON when done for class?
        #mcmcPeakParam = s.getPeakChiParams() # this is fine with a million points but not so much with low values.
        #print "mcmcPeak param:" + str(mcmcPeakParam)
        #print "minimizeParam, if it completed, else mcmcPeak:" +str(minimizeParam)
        guessParam = s.bestFitParam
        print "Using:" + str(guessParam)
        for pI in range(s.numP): # param Index
            chainVals = np.sort(s.chain[pI][start:stop])
            xs, counts = np.unique(chainVals, return_counts=True)
            # get index closest to peak.
            if setPeak:
                diff = np.absolute(xs - guessParam[pI])
                myPeakIndex = np.where(diff == min(diff))[0][0]
                print s.paramNames[pI] + " chain val " + str(xs[myPeakIndex]) + \
                      " closest to best fit val " + str(guessParam[pI])
                s.myConf[pI] = conf.confidence(xs, counts, s.confLevel, \
                                               peakIndex=myPeakIndex, \
                                               equallySpaced=False)
                # Hard set the best fit value?
                #s.myConf[pI].pk = guessParam[pI]
            else:
                s.myConf[pI] = conf.confidence(xs, counts, s.confLevel, \
                                               equallySpaced=False)    
                
    def assignBurnedConfidences(s, setPeak=False):
        assignConfidences(s.bI, s.numSteps, setPeak=setPeak)
        
    def assignConfidences(s, start, stop, setPeak=False):
        #start, stop are indices
        if setPeak:
            print "Assuming bestFitParam values for peak."
        else:
            print "Detecting best fit from distributions"
        for pI in range(s.numP):
            pofxs, xs = np.histogram(s.chain[pI][start:stop], bins=s.binPick(start, stop), density=True)
            xs = xs[:-1] # They have an extra.
            numData = len(xs)
            #plt.clf()
            #pl.linePlotXY(plt, xs, pofxs, "pofxs v. xs", "xs", "pofxs")            
            if setPeak:
                diff = np.absolute(xs - s.bestFitParam[pI])
                myPeakIndex = np.where(diff == min(diff))[0][0]  
                print s.paramNames[pI] + " hist val " + str(xs[myPeakIndex]) + \
                                  " closest to best fit val " + str(s.bestFitParam[pI])     
                s.myConf[pI] = conf.confidence(xs, pofxs, s.confLevel, \
                                               equallySpaced=False, \
                                               peakIndex=myPeakIndex) 
                #s.myConf[pI].pk = s.bestFitParam[pI]
            else:
                s.myConf[pI] = conf.confidence(xs, pofxs, s.confLevel, \
                                                equallySpaced=False)                 
       
    
    def myParamAcceptRates(self):
        return self.paramAcceptRates(self.bI, len(self.chain[1]))
    
    
    def paramAcceptRates(self, start, stop):
        #print [float(sum((self.acceptChain[parNum] == 1))) for parNum in range(len(self.startParams))]
        #print [float(sum(np.abs(self.acceptChain[parNum]))) for parNum in range(len(self.startParams))]
        return [float(sum((self.acceptChain[parNum][start:stop] == 1))) \
                / float(sum(np.abs(self.acceptChain[parNum][start:stop]))) \
                for parNum in range(self.numP)]
           
    def setBurnIndex(s):
        #med = np.median(s.chi2termChain)
        med = np.median(s.logLikelihoodChain)
        #s.bI = np.where(((s.chi2termChain - med)>= 0))[0][0]
        s.bI = np.where(((s.logLikelihoodChain - med)>= 0))[0][0]
        print "burnIndex set:" + str(s.bI)
        

        
    # running MCMC =============================================================
    
    # THe one that does the business
    def run_MCMC_for(self, n_steps): 
        # ADD n_steps to the chain currently self.numSteps long
        # Possibly could add some logic for adjusting step_size.
        s = self
        oriLen = s.numSteps
        params = np.copy(s.chain[:,-1]) 
        numReports = 5
        divider = n_steps/numReports # Note the progress every this many steps.
        noteProgress = [divider*(num+1) for num in range(numReports)]
        print "MCMC adding %d n_steps to chain of len %d"%(n_steps, s.numSteps)
        print "curr. start:" + str(params)+ " orig. start: " + str(s.startParams) 
        print "orig. stepsizes: " + str(s.stepsizes)
        # Go! Wait, should I include the starting point? I think I will, though it'll probably be axed by burn-in if far out.
        
        # Pre-allocate for speed. 
        # When Adding things in, make sure to make value-copies not references; see # testing: arr1 = [[1, 2, 3],[-1, -2, -3]]
        # https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy
        # Need B[:] = A, B[:] = A[:] <-Most efficient for overwriting B values. Or B = np.copy(A) <-making new array.
        # np.append(arr1, arr2, axis=?) makes new array, must be reassigned.
        # Note B = A[:] does referencing.
        self.chain = np.append(self.chain, np.zeros((len(params), n_steps)), axis=1)
        self.acceptChain = np.append(self.acceptChain, np.zeros((len(params), n_steps), dtype=int), axis=1) 
        #self.chi2termChain = np.append(self.chi2termChain, np.zeros(n_steps))
        self.logLikelihoodChain = np.append(self.logLikelihoodChain, np.zeros(n_steps))
        for n in range(n_steps):
            # Choose param to perturb.
            perturb_index = self.perturb_pick(params)
            #chi2term_old = self.chi2termChain[n+oriLen-1]
            logLikelihood_old = self.logLikelihoodChain[n+oriLen-1]
            #params, acceptvalues, chi2term_accept = self.step_eval(params, perturb_index, chi2term_old) # could do this stuff in step_eval....
            params, acceptvalues, logLikelihood_accept = self.step_eval(params, perturb_index, logLikelihood_old)
            #print str(params) + str(acceptvalues) + str(chi2term_accept)
            s.chain[:, n+oriLen] = params
            s.acceptChain[:, n+oriLen] = acceptvalues
            #s.chi2termChain[n+oriLen] = chi2term_accept
            s.logLikelihoodChain[n+oriLen] = logLikelihood_accept
            s.numSteps += 1
            if n in noteProgress:
                print str(n) + " steps complete."
            if s.numSteps % self.checkStepsizesEvery == 0:
                self.update_step_size()
        self.setBurnIndex()
        print "MCMC and burn cutting complete. Maybe call analyzeResults() or makePlots?"
        
    def update_step_size(self):       
        s=self
        percents = s.paramAcceptRates( s.numSteps - s.checkStepsizesEvery, s.numSteps)
        if s.stepsizesPercents[0,-1] == -1: #FIx init.
            s.stepsizesPercents[:,-1] = percents
        else:
            s.stepsizesChain = np.append(s.stepsizesChain, np.transpose([np.copy(s.stepsizes)]), axis=1)
            s.stepsizesPercents = np.append(s.stepsizesPercents, np.transpose([np.copy(percents)]), axis=1)
        for paramNum in range(len(s.startParams)):
            if percents[paramNum] < s.acceptTarget[0]: # acceptance too low, stepsize too large
                if percents[paramNum] > 0:
                    # Decreases step size more if the percent acceptance is smaller
                    s.stepsizes[paramNum] = s.stepsizes[paramNum]*(percents[paramNum]/s.acceptTarget[0])
                else: 
                    s.stepsizes[paramNum] = s.stepsizes[paramNum]/10.0
            elif percents[paramNum] > s.acceptTarget[1]: # acceptance too high, stepsize too small
                # Increases stepsize more if the percent acceptance is higher.
                s.stepsizes[paramNum] = s.stepsizes[paramNum]*(percents[paramNum]/s.acceptTarget[1])
        
                
    def get_err(self, params):
        # A default function 
        if not self.errorFunc:
            return self.yErr
        else:
            return self.errorFunc(params, self.yErr)
    
    def get_chi2(self, params): ##obtain the chi2 value of the model y-values given current parameters vs. the measured y-values
        ##calculate chi2
        # We assume that the errors are correct and gaussian. Thus just chi2 TERM is sufficient for log-likelihood. Note that the 'const' should be added in some cases!
        # Note, actually need to put the -1 . or the model won't converge properly. I have currently 
        modelYvals = self.get_model(params, self.xX)
        # Chi2 = Sum[((Mk-Dk)/sigmak)^2], Sivia 3.66
        # Full = Log[ Prod [ 1/(sigmak*sqrt(2*Pi))*exp(-0.5 *((Mk-Dk)/sigmak)^2 ) ]
        diff = modelYvals - self.yY
        chi2 = conf.msum((diff)**2. / self.get_err(params)**2.)
        return chi2
        
    
    def get_chi2term(self, params):
        chi2term = -0.5*self.get_chi2(params)
        #nonConstLL = np.exp(chi2term)
        #const = conf.msum(-1
        return chi2term        
        
    def get_logLikelihood(self, params):
        chi2term = self.get_chi2term(params)
        return -np.sum(np.log(np.sqrt(2.*np.pi)*self.get_err(params))) + chi2term
    
    def get_negLogLikelihood(self, params):
        return -self.get_logLikelihood(params)
        
        
    def perturb_pick(self, params): ##select a model parameter to perturb
        ##this function randomly selects which model parameter to perturb based on how many parameters are in the model
        # Returns an index into param.
        return np.random.choice(range(len(params)))
    
    
    def propose_param(self, active_params, perturb_index): 
        ##obtain a trial model parameter for the current step
        # active_params: the params, currently. 
        # stepsizes: the gaussian widths associated with each param.
        # I truly have no idea what perturb_value is supposed to be.
        # I'm making it into perturb_index
        nTries = 1
        try_params = np.copy(active_params) # Gotta make a true copy! List slicing doesn't automatically copy numpy arrays
        try_params[perturb_index] = np.random.normal(active_params[perturb_index], self.stepsizes[perturb_index])
        while not self.validFunc(try_params, []): # It's possible this while loop should also cover whatever considers perturb_pick.
            if nTries % 100 == 0:
                print str(nTries) +  " nTries changing " + str(perturb_index) +"from " +str(active_params)     
            # It should do something. 
            try_params[perturb_index] = np.random.normal(active_params[perturb_index], self.stepsizes[perturb_index])
            nTries +=1
        return try_params
        
        
    def step_eval(self, params, perturb_index, logLikelihood_old): 
        ##evaluate whether to step to the new trial value
        #chi2_old = get_chi2term(params, x, y, error) ##the chi2 value of the parameters from the previous step
        # Don't need to repeat this calculation, just return the old one, we want to note it anyway.
        try_params = self.propose_param(params, perturb_index) ## read in the trial model parameters for the current step
        #chi2term_try = self.get_chi2term(try_params) ## the chi2 value of the trial model parameters for the current step
        logLikelihood_try = self.get_logLikelihood(try_params) 
        #insert some if/else statements here to determine whether a step should be taken, and document the resulta.
        acceptvalues = np.zeros(len(params), dtype=int)
        #prob = np.exp(chi2term_try-chi2term_old)
        prob = np.exp(logLikelihood_try-logLikelihood_old)
        if (prob > 1) or (np.random.uniform(0., 1.) < prob):
            new_params = try_params
            acceptvalues[perturb_index] = 1 # Proposed and accepted!
            #chi2term_accept = chi2term_try
            logLikelihood_accept = logLikelihood_try
        else:
            new_params = params
            acceptvalues[perturb_index] = -1 # Proposed and rejected.    
            #chi2term_accept = chi2term_old
            logLikelihood_accept = logLikelihood_old
        return new_params, acceptvalues, logLikelihood_accept    
    
    
'''
# BE CAREFUL WITH GLOBALS. They should have long names.

rawfile = np.genfromtxt('ps5_line_data.txt') ##Import data from file location
X = rawfile[:,0] ##pressure data
Y = rawfile[:,1] ##density data
Yerr = rawfile[:,2] ##error data

params0 = [2.5, 10.] ## insert initial guess for [slope, intercept]

stepsize = [1., .25] ## insert reasonable values for the step size in each parameter

n_steps = 10 ## insert number of steps to run the MCMC
#1000000 #<-kinda the largest plausible
# You can add one more order of magnitude if you're willing to wait a while to get data-though you WON'T get plots due to MemoryError. Beyond that, immediate MemoryError.
#myMCMC = MCMC(params0, ["slope", "yIntrcpt"], stepsize, X, Y, Yerr, get_model, 1000000, plotTitle="GoodStart") ##run the MCMC and output the chains

qMCMC = MCMC([0., 2.5, 10.], ["quad", "slope", "yIntrcpt"], [.2, .2, .2], X, Y, Yerr, quadratic, 1000000, plotTitle="GoodStartQuad")

#badGuessMCMC = MCMC([1.0, 4.0], ["slope", "yIntrcpt"], [.5, 1.0], X, Y, Yerr, get_model, 10000, plotTitle="BurnRemoved", checkStepsizesEvery=1000000)

#bGMCMC = MCMC([1.0, 4.0], ["slope", "yIntrcpt"], [.5, 1.0], X, Y, Yerr, get_model, 10000, plotTitle="checkStep100", checkStepsizesEvery=100)

#badGuessMCMC = MCMC([1.0, 4.0], ["slope", "yIntrcpt"], [.5, 1.0], X, Y, Yerr, get_model, 10000, plotTitle="BurnRemoved", checkStepsizesEvery=1000000)
'''





''''
# original no-class MCMC
def get_model(params,x): ##evaluate the y-values of the model, given the current guess of parameter values
    (slope, intercept) = params
    model = x*slope+intercept ##insert equation for your model to calculate y-values
    return model
    
def get_log_likelihood(params, x, y, error): ##obtain the chi2 value of the model y-values given current parameters vs. the measured y-values
    ##calculate chi2
    # We assume that the errors are correct and gaussian. Thus just chi2 TERM is sufficient for log-likelihood. Note that the 'const' should be added in some cases!
    # Note, actually need to put the -1 . or the model won't converge properly. I have currently 
    modelYvals = get_model(params, x)
    # Chi2 = Sum[((Mk-Dk)/sigmak)^2], Sivia 3.66
    # Full = Log[ Prod [ 1/(sigmak*sqrt(2*Pi))*exp(-0.5 *((Mk-Dk)/sigmak)^2 ) ]
    chi2 = conf.msum((modelYvals - y)**2. / error**2.)
    chi2term = -0.5*chi2
    #nonConstLL = np.exp(chi2term)
    #const = conf.msum(-1
    return chi2term
    
def perturb_pick(params): ##select a model parameter to perturb
    ##this function randomly selects which model parameter to perturb based on how many parameters are in the model
    # Returns an index into param.
    return np.random.choice(range(len(params)))


def propose_param(active_params, stepsizes, perturb_index): 
    ##obtain a trial model parameter for the current step
    # active_params: the params, currently. 
    # stepsizes: the gaussian widths associated with each param.
    # I truly have no idea what perturb_value is supposed to be.
    # I'm making it into perturb_index
    try_params = np.copy(active_params) # Gotta make a true copy! List slicing doesn't automatically copy numpy arrays
    try_params[perturb_index] = np.random.normal(active_params[perturb_index], stepsizes[perturb_index])
    return try_params
    
    
def step_eval(params, stepsizes, x, y, error, perturb_index, L_old): 
    ##evaluate whether to step to the new trial value
    #chi2_old = get_log_likelihood(params, x, y, error) ##the chi2 value of the parameters from the previous step
    # Don't need to repeat this calculation, just return the old one, we want to note it anyway.
    try_params = propose_param(params, stepsizes, perturb_index) ## read in the trial model parameters for the current step
    chi2term_try = get_log_likelihood(try_params, x, y, error) ## the chi2 value of the trial model parameters for the current step
    #insert some if/else statements here to determine whether a step should be taken, and document the resulta.
    acceptvalues = np.zeros(len(params), dtype=int)
    prob = np.exp(chi2term_try-chi2term_old)
    if (prob > 1) or (np.random.uniform(0., 1.) < prob):
        new_params = try_params
        acceptvalues[perturb_index] = 1 # Proposed and accepted!
        chi2term_accept = chi2term_try
    else:
        new_params = params
        acceptvalues[perturb_index] = -1 # Proposed and rejected.    
        chi2term_accept = chi2term_old
    return new_params, acceptvalues, chi2term_accept

def runMCMC(params, stepsizes, x, y, error, n_steps): 
    # Possibly could add some logic for adjusting step_size.
    ##run the whole MCMC routine, calling the subroutines written above
    # returns:
    # chain         (#params x num_steps) array of param values at steps,
    # accept_chain  (#params x num_steps) array of acceptance notes: 
    #                   1 for proposed&accepted, -1 for proposed&rejected, 
    #                   0 for not-perturbed-this-step
    # chi2_chain    (num_step) array of chi2_vals of recorded step
    print "MCMC num_steps: %d"%(n_steps)
    print "start: " + str(params)
    print "stepsizes: " + str(stepsizes)
    
    # Go! Wait, should I include the starting point? I think I will, though it'll probably be axed by burn-in if far out.
    
    # Pre-allocate for speed. 
    # When Adding things in, make sure to make value-copies not references; see # testing: arr1 = [[1, 2, 3],[-1, -2, -3]]
    # https://stackoverflow.com/questions/19676538/numpy-array-assignment-with-copy
    # Need B[:] = A, B[:] = A[:] <-Most efficient for overwriting B values. Or B = np.copy(A) <-making new array.
    # np.append(arr1, arr2, axis=?) makes new array, must be reassigned.
    # Note B = A[:] does referencing.
    chain = np.append(np.transpose([np.copy(params)]), np.zeros((len(params), n_steps)), axis=1)
    accept_chain = np.zeros((len(params), n_steps+1), dtype=int) # Didn't really propose against anything here, so 0,0 for start.
    chi2term_chain = np.append([get_log_likelihood(params, x, y, error)], np.zeros(n_steps))
    #print chain
    #print accept_chain
    #print chi2term_chain
    for n in range(n_steps):
        # Choose param to perturb.
        perturb_index = perturb_pick(params)
        chi2term_old = chi2term_chain[n]
        params, acceptvalues, chi2term_accept = step_eval(params, stepsizes, x, y, error, perturb_index, chi2term_old)
        print str(params) + str(acceptvalues) + str(chi2term_accept)
        #chain = np.append(chain, np.transpose([params]), axis=1)
        chain[:, n+1] = params
        accept_chain[:, n+1] = acceptvalues
        chi2term_chain[n+1] = chi2term_accept
    return chain, accept_chain, chi2term_chain
    
chain, accept_chain, chi2term_chain = doMCMC(params0, stepsize, X, Y, Yerr, n_steps) ##run the MCMC and output the chains

'''