"""Rita Sonka
1/24/2017
numpy plotting
"""

import numpy as np
import matplotlib.pyplot as plt
#from ritas_python_util_main import make_filesafe 

def make_filesafe(filename):
    return filename.replace(" ", "_").replace(".", ",").replace(",,\\", "..\\").replace(",,//","..//").replace("(","").replace(")","")  

def makeFilesafe(filename):
    return make_filesafe(filename)  

fileExtension = "png"
# THings to add in general: SAVING! Does below count?
def saveFig(plt, filename):
    plt.savefig(makeFilesafe(filename)+"." +fileExtension, bbox_inches='tight')
    


def fit_and_resid(function,xs,ys,p0,own_fig=True, plot_which_x_variable=-42,
                  label='',plot_args={},legend=False,x_label='',y_label='',suptitle=''):
    '''to put on other, own_fig= (fit_plot ax, resid_plot ax); 
    can make one or both false to not do that plot.
    plot_args should NOT contain label; also not contain a marker or a linestyle I think?
    Returns: (prm,cov,pred_y,resid,fp,rp)'''
    if own_fig == True:
        fig, ax = plt.subplots(nrows=2, ncols=1) #figsize=(8,9) # remember it's x,y size
        fp,rp = ax[0],ax[1]
    elif own_fig == False: # not sure that proper behavior...should probably put on the thing.
        fp,rp = False,False
    else:
        fp,rp = own_fig
        if (fp==True and rp==False) or (fp==False and rp==True):
            plt.figure()
            if fp==True:
                fp = plt
            if rp ==True:
                rp = plt
    # the fit.
    (prm,cov) = curve_fit(function, xs, ys, p0)
    
    # the y the fit curve predicts
    pred_y = function(xs,*prm)
    resid = pred_y - ys
    
    if not plot_which_x_variable == -42: # what to plot if function of 2+variables
        xs = xs[plot_which_x_variable]    
    
    if fp:
        # the original data
        p = fp.plot(xs,ys,alpha=0.5,marker='.',label=label,**plot_args)
        # the fit line
        fp.plot(xs,pred_y,linestyle='dashed',linewidth=0.5,color=p[-1].get_color(),**plot_args)
        try: # I assume it's usually going to be a subplot
            fp.set_xlabel(x_label+ " dashed=Fit predict") 
            fp.set_ylabel(y_label)
        except AttributeError: # it's a full plot
            plt.xlabel(x_label+ " dashed=Fit predict")
            plt.ylabel(y_label)
        if legend:
            fp.legend()
    if rp:
        # the residual
        rp.plot(xs,resid,marker='.',label=label,**plot_args)
        if own_fig==True or rp==plt:
            rp.axhline(0,linestyle='dashed',color='k')
        try: # I assume it's usually going to be a subplot
            rp.set_xlabel(x_label) 
            rp.set_ylabel(y_label+ " resid.")
        except AttributeError: # it's a full plot
            plt.xlabel(x_label)
            plt.ylabel(y_label+ " resid.")
        if legend:
            rp.legend()
            
    if fp or rp:
        try: # I assume it's usually going to be a subplot
            plt.suptitle(suptitle)
        except AttributeError: # it's one full plot
            plt.title(suptitle)
        plt.tight_layout()
    
    return (prm,cov,pred_y,resid,fp,rp)   
    
    
    
    
# Old code for trying to make the matplotlib subplots the same as plots, plus other stuff, below. 


# Specific common types of graphs.
def linePlotXY(plt, xX, yY,  titleString, xlabel, ylabel):
    # xX and yY should be numpy arrays. No real need to go through shenanigans.
    plt.plot(xX, yY)
    plt.title(titleString)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)





# I should change how it handles default xlim/ylim on all the instantiation and  plotting functions!

# This is for x vs. y plots.

# This doesn't work if you import this file as something. MySinglePlot now has it's own method to do this.
def makeSinglePlotFromDataSetObj(xydataset):
    d = xydataset
    return MySinglePlot([d], xlabel=d.xlabel, ylabel=d.ylabel,title=d.title,xlim=d.xlim, ylim=d.ylim)




class XYdataSet:
    ''' A class for representing data sets I intend to plot. 
    UNIVERSAL vars:
    xX
    yY
    legendName
    labelHandle ; set when it is plotted. # Not really anymore. errorbar ruined it.   
    
    Optional:
    color ;  colorOptions = bgrcmykw, should expand to include color scales sometime # possibly move into plotType args?
    plotType= 'xy' (default), 'hist', 'hist2d', 'errorbar'
    plotTypeArgs dictionary of arguments to pass to plotting functions.
    Notes on that:
    xy --    marker ; markerOptions = https://matplotlib.org/api/markers_api.html;
             some quick examples: '.,ov^<>8spP*hH+xXDd|_, NOT '-','--','-.',':'
             linestyle ; '-','--','-.',':',or'None' ; last one a scatterplot.
    hist --  bins, normed important (density won't work for me, possible I should update matplotlib)
    errorbar -- yerr, xerr
    otherArgs: vlines:[xpos] text:[(toWrite, x%, y%)]
    
    Special class variables set by plotTypes:
    Should really put the binning stuff in here.
    
    INFERIOR ones: (if included on a plot with multiple datasets, mySinglePlot should have these)
    xlabel
    ylabel
    title
    xlim = [xStart, xEnd]
    ylim = [yStart, yEnd]
    filename
    
    Not currently included: other matplotlib options:
    markersize
    '''
    def __init__(self, xX, yY, legendName="",xlabel="", ylabel="",title="",\
                 xlim=['default'], ylim=['default'], filename="", color='b',\
                 plotType='xy', plotTypeArgs={'linestyle':'-'}, otherArgs={'default':None}):
        assert len(xX) == len(yY), "len(xX)=%d != len(yY)=%d"%(len(xX), len(yY))
        assert len(xX) > 0, "len(xX)=%d notOkay!"%(len(xX))
        assert len(yY) > 0, "len(yY)=%d notOkay!"%(len(yY))
        self.xX = xX
        self.yY = yY
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.legendName = legendName
        self.xlim = xlim
        self.ylim = ylim
        '''
        if xlim[0] == 'default':
            #self.xlim = [xX[0], xX[len(xX)-1]]
        if ylim[0] == 'default':
            #self.ylim = [min(yY), max(yY)]
        '''
        self.color = color
        self.plotType = plotType
        self.plotTypeArgs = plotTypeArgs            
        if not filename == "":
            self.filename = makeFilesafe(filename)
        else:
            if not self.title == '':
                self.filename = makeFilesafe(title)
            elif not self.legendName == '':
                self.filename = makeFilesafe(legendName)
            elif not xlabel == '' and not ylabel == '':
                self.filename = makeFilesafe(ylabel + "_vs_" + xlabel)
            else:
                filename = makeFilesafe("genericFile")
        self.otherArgs=otherArgs
                
    def plotMeSolo(self, plt):
        s=self
        self.plotMe(plt)
        if not self.xlim[0] == 'default':
            plt.xlim(s.xlim) 
        if not self.ylim[0] == 'default':
            plt.ylim(s.ylim)
        self.setLabels(plt)
    
    def soloPlotAndSave(self, plt):
        plt.clf()
        self.plotMeSolo(plt)
        plt.savefig(self.filename+"." +fileExtension, bbox_inches='tight')
        #plt.savefig(self.filename, format=fileExtension)
        
    
    def plotMe(self,plt):
        # For use by other classes here, mostly.
        s=self
        if self.plotType == 'xy':
            #colorMarker = s.color + s.marker
            #s.labelHandle = plt.plot(s.xX, s.yY, color=s.color, marker=s.marker, label='legendName') # THis doesn't work, it doesn't recognize marker '-'
            #s.labelHandle, = plt.plot(s.xX, s.yY, colorMarker, label=self.legendName, **s.plotTypeArgs)
            s.labelHandle, = plt.plot(s.xX, s.yY, color=self.color, label=self.legendName, **s.plotTypeArgs)
        elif self.plotType == "errorbar":
            check =  plt.errorbar(s.xX, s.yY, color=s.color, label=self.legendName, **s.plotTypeArgs)
            #print check
        elif self.plotType == 'hist':
            n, bins, patches = plt.hist(self.xX, **s.plotTypeArgs)
            s.labelHandle = patches # NO idea if this works iin any reasonable manner.
        elif self.plotType == 'hist2d':
            plt.hist2d(self.xX, self.yY, **s.plotTypeArgs)
            plt.colorbar()
            s.labelHandle = -1
        if 'vlines' in self.otherArgs:
            for x in self.otherArgs['vlines']:
                plt.axvline(x=x, color='r')
        if 'text' in self.otherArgs: # text:[(toWrite, x%, y%)]
            xmin, xmax = plt.xlim() # doesn't set if you don't include anything! Note, relies on being after the plotting...
            ymin, ymax = plt.ylim()
            for texty in self.otherArgs['text']:
                plt.text(float(texty[0])*(xmax-xmin)+xmin, float(texty[1])*(ymax-ymin)+ymin,  texty[2])
            
        #return self.labelHandle # errorbar has forced me to obsolete this.
        
    def setLabels(self, plt):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)   
    

class MySinglePlot:
    '''A class representing a single plot that may have multiple data sets
    plotted on it.
    myDataSets = [XYdataSet]
    xlabel
    ylabel
    title
    xlim = [xStart, xEnd]
    ylim = [yStart, yEnd]
    
    otherArgs: vlines:[xpos] text:[(toWrite, x%, y%)]
    
    Not currently included: 
    axis types: semilog plot, whatever. That would go here. Actually, histogram should go in single data set
    '''
    def __init__(self, myDataSets, xlabel="", ylabel="",title="",xlim=['default'], ylim=['default'],otherArgs={'default':None}):
        self.myDataSets = myDataSets
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        '''
        if xlim[0] == 'default':
            xX = myDataSets[0].xX
            xMin = min(xX)
            xMax = max(xX)
            for dataSet in myDataSets:
                if min(dataSet.xX) < xMin:
                    xMin = min(dataSet.xX)
                if max(dataSet.xX) > xMax:
                    xMax = max(dataSet.xX)
            self.xlim = [xMin, xMax]
        else:
            self.xlim = xlim'''
        self.xlim = xlim 
        '''
        if ylim[0] == 'default':
            yY = myDataSets[0].yY
            yMin = min(yY)
            yMax = max(yY)
            for dataSet in myDataSets:
                if min(dataSet.yY) < yMin:
                    yMin = min(dataSet.yY)
                if max(dataSet.yY) > yMax:
                    yMax = max(dataSet.yY)
            self.ylim = [yMin, yMax]
        else:
            self.ylim = ylim'''
        self.ylim = ylim
        self.otherArgs = otherArgs
    
    def plotMeSolo(self, plt):
        for dSet in self.myDataSets:
            dSet.plotMe(plt)
        if len(self.myDataSets)>1:
            plt.legend()
        ''''
        myHandles = []
        for dSet in self.myDataSets:
            myHandles.append(dSet.plotMe(plt))
        if len(myHandles) > 1:
            plt.legend(handles=myHandles)'''
        self.setPlotLimits(plt)
        self.setLabels(plt)  
        
        if 'vlines' in self.otherArgs:
            for x in self.otherArgs['vlines']:
                plt.axvline(x=x, color='r')
        if 'text' in self.otherArgs: # text:[(toWrite, x%, y%)]
            xmin, xmax = plt.xlim() # doesn't set if you don't include anything! Note, relies on being after the plotting...
            ymin, ymax = plt.ylim()
            for texty in self.otherArgs['text']:
                plt.text(float(texty[0])*(xmax-xmin)+xmin, float(texty[1])*(ymax-ymin)+ymin,  texty[2])        
        
    def setLabels(self, plt):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel) 
        
    def setSubplotLabels(self, ax):
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)    
        
    def setPlotLimits(self, plt):
        if not self.xlim[0] == 'default':
            plt.xlim(self.xlim)
        if not self.ylim[0] == 'default':
            plt.ylim(self.ylim) 
    def setSubplotLimits(self, ax):
        if not self.xlim[0] == 'default':
            ax.set_xlim(self.xlim) 
        if not self.ylim[0] == 'default':
            ax.set_ylim(self.ylim)        
    
    def subplotMeOnAxes(self, ax):
        myHandles = []
        for dSet in self.myDataSets:
            myHandles.append(dSet.plotMe(ax))
        if len(myHandles) > 1:
            ax.legend(handles=myHandles)
        self.setSubplotLimits(ax)
        self.setSubplotLabels(ax) 
    
    
    
    
        
        
        

class myPlot:
    '''A class representing a figure that references multiple data sets.
    May be a single plot or use subplotting.
    myPlots = [[ mySinglePlot  ]]. May not be square! # Must be square--put 0 or False in it to skip spots.
    myPlots[row][col] = the mySinglePlot subplot in row, col. If there is only one subplot, does not use subplotting.
    rows = # of rows
    cols = # of cols. Should always be updated whenever myPlots adds stuff.
    title = Super title above all the subplots.
    sharex, sharey : shares as logically as it can.
    self.fig
    
    derived:
    myAxes = [[ axes corresponding to myPlots ]]  # Assigned when plotted.
    
    RIght now:
    
    Note: myPlot is supposed to be able to handle all XYplot variants. It is what I work with.
    https://matplotlib.org/users/pyplot_tutorial.html
    '''
    def __init__(self, myPlots, title='', sharex='', sharey=''):
        self.myPlots = myPlots
        try: 
            myPlots[0][0]
        except:
            self.myPlots = self.autoArrange(myPlots)
        self.setRowCols()
        self.sharex=sharex
        self.sharey=sharey
        self.title=title
        self.fig= None # the supertitle, if one exists.
        #Note it doesn't check anything. That's so that I can make some other quickInit functions
        
    def setRowCols(self):
        self.rows = len(self.myPlots)
        cols = 1
        for r in self.myPlots:
            if len(r) > cols:
                cols = len(r)
        self.cols = cols
            
    
    def plotMe(self, plt):
        # the big one. Once you've set things up, this plots things up.
        # https://matplotlib.org/examples/pylab_examples/subplots_demo.html
        s=self
        #plt.clf()
        if self.fig == None:
            fig = plt.figure()
            self.fig = fig
        else:
            fig = self.fig
            plt.figure(self.fig.number)
        ref = plt
        
        if self.rows == 1 and self.cols == 1:
            self.myPlots[0][0].plotMeSolo(ref)  
            return 1
        #f, axarr = plt.subplots(self.rows, self.cols)
        self.myAxes = []
        for r in range(self.rows):
            self.myAxes.append([])
            for c in range(len(self.myPlots[r])):
                if self.myPlots[r][c]:
                    #subplot = fig.add_subplot(r, c, 
                    #print str(r*c + c+ 1)
                    kwordArgs = {}
                    if self.sharex:
                        for r2 in range(r):
                            if len(self.myAxes[r2]) > c and self.myAxes[r2][c]:
                                kwordArgs['sharex'] = self.myAxes[r2][c]
                                break
                    if self.sharey:
                        for c2 in range(c):
                            if self.myAxes[r][c2]:
                                kwordArgs['sharey'] = self.myAxes[r][c2]
                                break
                    axy = ref.subplot(self.rows, self.cols, r*self.cols + c + 1, **kwordArgs)
                    self.myAxes[r].append(axy)
                    self.myPlots[r][c].plotMeSolo(ref)  
                    #self.myPlots[r][c].plotMeSolo(axarr[r, c])  
                    #self.myPlots[r][c].subplotMeOnAxes(axarr[r, c]) 
                    ref.tight_layout(rect=[0, 0.03, 1, 0.95])
        if self.title: # useful comments: https://stackoverflow.com/questions/7066121/how-to-set-a-single-main-title-above-all-the-subplots-with-pyplot
            #plt.suptitle(self.title, size=16)   # didn't work?
            fig.tight_layout() #THis Also didn't work...
            fig.subplots_adjust(top=0.88) 
            fig.suptitle(self.title, size=16)  
            # Maybe something for just 1?
        #return fig
        
    def autoArrange(self, plots):
        # plots a 1-D list:
        squares = np.array([1, 4, 9, 16, 25])
        squareSize = squares[ np.where(len(plots)<=squares)[0][-1] ]
        plotMatrix = []
        indexUsed = 0
        for rowNum in range(int(np.sqrt(squareSize))):
            if indexUsed == len(plots):
                break
            plotMatrix.append([])
            for colNum in range(int(np.sqrt(squareSize))): 
                plotMatrix[rowNum].append(plots[indexUsed])
                indexUsed += 1
                if indexUsed == len(plots):
                    break
        return plotMatrix    
            
    
    
    
    
# Testing  should include the "if name == __main__" or whatever.
'''
x1 = np.linspace(1,10, num=10)
y1 = np.linspace(1,10,num=10)

p1Data = XYdataSet(x1,y1,xlabel='x1label',ylabel='y1label',title='title1',legendName='legendName1',xlim=[2,8],ylim=[2,8],color='r',marker='*')
#p1Data.plotMeSolo(plt)

x2 = np.linspace(1,10, num=10)
y2 = np.linspace(10,1,num=10)

p2Data = XYdataSet(x2,y2,xlabel='x2label',ylabel='y2label',title='title2',legendName='legendName2',xlim=[1,8],ylim=[1,8],color='b',marker='s')

x1x2 = mySinglePlot([p1Data, p2Data], xlabel='x1&2label',ylabel='y1&2label',title='title1&2',xlim=[0,11],ylim=[0,11])
#x1x2.plotMeSolo(plt)

x3 = np.linspace(1,10, num=10)
y3 = np.array([5,5,5,5,5,5,5,5,5,5])

p3Data = XYdataSet(x3,y3,xlabel='x3label',ylabel='y3label',title='title3',legendName='legendName3',xlim=[1,8],ylim=[1,8],color='g',marker='-')

x1x3 = mySinglePlot([p1Data, p3Data], xlabel='x1&3label',ylabel='y1&3label',title='title1&3',xlim=[0,11],ylim=[0,11])
#x1x3.plotMeSolo(plt)


#bigPlot = myPlot([[x1x2]])
bigPlot = myPlot([[x1x2], [x1x3]])

#bigPlot.plotMe(plt)
'''





