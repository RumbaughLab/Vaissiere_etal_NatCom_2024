import os
os.chdir('Y:/2020-09-Paper-ReactiveTouch/__code__/')
from ALLmethods import *

### change the title of the console
import ctypes
ctypes.windll.kernel32.SetConsoleTitleW("analysis")
import colorsys
import concurrent.futures
import copy
import cv2
from cycler import cycler
from datetime import date
import glob
import itertools
import logging
import numpy as np
import numpy
import os
import pandas as pd
import platform
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from matplotlib.transforms import Bbox
import random
import re
import seaborn as sns
from scipy.signal import find_peaks
from scipy import signal
import scipy
from scipy.interpolate import griddata
import scipy.io
from scipy.io import loadmat
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import time 
import tqdm
from typing import List, Optional, Sequence, Tuple

if sys.platform != 'linux':
    import msvcrt
plt.ioff()
## ***************************************************************************
## * CUSTOM FUNCTION                                                         *
## ***************************************************************************

def paramsForCustomPlot(data, variableLabel='genotype', valueLabel='value', sort=True, **kwargs):
    """Function to create the parametters for ploting the variable, value, subject setup manually create a dictionary for the parameter to be reused for ploting see
    
    Parameters:
        data (DataFrame): dataframe with the data
        myPal (list): list of hexadecimal RGB value should be at least the length of the variableLable // this was removed due to change in default settings
        variableLabel (str): name of the variable of interest, header of the variable column
        subjectLabel (str): name of the subject of interest, header of the subject column
        valueLabel (str): name of the subject of interest, 
    """

    subjectLabel = kwargs.get('subjectLabel', None)
    if subjectLabel is None:
        subjectLabel = 'tmpSub'
        data.index = data.index.set_names(['tmpSub'])
        data = data.reset_index()
    dfSummary=data.groupby([variableLabel,subjectLabel]).mean()
    dfSummary.reset_index(inplace=True)
    if sort == True:
        dfSummary =  dfSummary.sort_values(by=[variableLabel],  ascending=False)
        data = data.sort_values(by=[variableLabel],  ascending=False)

    params = dict(  data=dfSummary,
                    x=str(variableLabel),
                    y=str(valueLabel),
                    hue=str(variableLabel),
                    )

    paramsNest = dict( data=data,
                    x=str(variableLabel),
                    y=str(valueLabel),
                    hue=str(variableLabel),
                    )

    ## calculate the number of observation 
    tmpObs = data[[variableLabel, valueLabel]]
    tmpObs = tmpObs.dropna()
    nobs = tmpObs.groupby(variableLabel).count()
    # nobs = list(itertools.chain.from_iterable(nobs.values))
    # nobs = [str(x) for x in nobs]
    nobs = nobs.reset_index()
    nobs = nobs.sort_values(by=[variableLabel],  ascending=False)
    nobs = list(nobs[valueLabel])
    nmax = tmpObs.max()[-1]*1.1

    return params, paramsNest, nobs, nmax

def fileOfInterest(filename):
    filesAnal = filename
    saveName = filesAnal.split(os.path.sep)[-1].split('.')[0]

    dt = pd.read_csv(filesAnal)
    dt = dt.T # need to transpose the table to switch to long format
    dt = dt.reset_index(col_fill='geno')
    dt.columns = ['geno'] + list(dt.loc[0][1:])
    dt = dt.drop([0])
    colgeno = dt['geno'].str.split('.', n=1, expand=True)
    dt['geno'] = colgeno[0]
    dt[list(dt.columns[1:])] = dt[list(dt.columns[1:])].apply(pd.to_numeric, errors='coerce') # convert to NaN if number present with *
    return dt, saveName

def customPlotingMulti(data, myx, myy, myu,myorder,figSet=[],nval=0, myponitszie=6):

    if figSet == []:    
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figSet)

    ax = sns.barplot(
    data, 
    x=myx, 
    y=myy, 
    hue=myu, 
    errorbar=('ci',68),
    capsize=0, # error bar width
    order= myorder,
    edgecolor='k', linewidth=0.4, dodge=True, zorder=0, alpha = 1,)

    # to change the width of the error bar
    for line in ax.lines:
        line.set_linewidth(1)

    # Get the legend from just the bar chart
    handles, labels = ax.get_legend_handles_labels()

    # Draw the stripplot
    sns.stripplot(
        data, 
        x=myx, 
        y=myy, 
        hue=myu, 
        order=myorder,
        edgecolor = 'k', linewidth = 0.4,  size=myponitszie, zorder=3, ax = ax, dodge = True, jitter = 0.2, alpha=0.5,
    )
    # Remove the old legend
    ax.legend_.remove()
    # Add just the bar chart legend back
    ax.legend(
        handles,
        labels,
        loc=7,
        bbox_to_anchor=(1.25, .5),
    )



    # Annotating the count at the bottom of each bar
    bars = ax.patches
    for bar in bars:
        height = bar.get_height()
        width = bar.get_width()
        # Adjust the position of the annotation based on the bar location and width
        ax.annotate(
            '{}'.format(int(nval)),
            (bar.get_x() + width / 2, 0),
            ha='center', va='center', fontsize=6, color='white',
            xytext=(0, 2), textcoords='offset points'
        )

    ax.legend_.remove()
    ax.tick_params(labelbottom=False)
    plt.xlabel('')

def customPloting(data, val ,genoOfInt = 'geno', plottick = 4, custylim=[0,300], figSet = [1,2], baroff = False, sort=True):
    data = data[[genoOfInt, val]]
    if sort == True:
        data = data.sort_values(by=[genoOfInt],  ascending=False)
    for i,j in enumerate([genoOfInt]):
        print(i,j)
        params, paramsNest, nobs, nmax = paramsForCustomPlot(data = data,  variableLabel = genoOfInt, valueLabel = val, sort = sort)
        # plt.close('all')
        if figSet == []:    
            fig, ax = plt.subplots()
        else:
            fig, ax = plt.subplots(figsize=figSet)
        # fig = plt.figure()
        ## VIOLIN pPLOT
        # axv = sns.violinplot(split = False, width=0.6, cut=1, **paramsNest, inner='quartile', zorder=2, linewidth = 0.5, dodge=False, ax = ax)
        ## change the line stile 
        # for l in axv.lines:
        #     l.set_linestyle('-')
        #     l.set_color('white')
        #     l.set_alpha(0.3)
        # sns.pointplot(ax= ax, ci=95, scale=0.1, errwidth=0.6, **params, palette = ['black'])
        ## ensure the point plot are above everything else
        ## to troublehoot order of appearanc ##https://stackoverflow.com/questions/32281580/using-seaborn-how-do-i-get-all-the-elements-from-a-pointplot-to-appear-above-th
        plt.setp(ax.lines, zorder=100)
        plt.setp(ax.collections, zorder=100, label="")

        if baroff == False:
            ax = sns.pointplot(errorbar=('ci', 68), scale=0.1, errwidth=1.2, **params, palette = ['k'])
            sns.barplot(**paramsNest, edgecolor='k',  linewidth=0.4 , errorbar=None, dodge=False, zorder=0, alpha = 1)
            ## plot the observation on the graph
            ## need to work on proportional positionning and not fix to one now 4% of max value
            pos = range(len(nobs))
            for tick,label in zip(pos,ax.get_xticklabels()):
               ax.text(pos[tick], nmax*0.04, nobs[tick], horizontalalignment='center', size='x-small', color='w', weight='semibold')
        elif baroff == True:
            ax = sns.pointplot(errorbar=('ci', 68), scale=0.4, errwidth=1.2, **params, palette = ['k'])
            ## plot the observation on the graph
            ## need to work on proportional positionning and not fix to one now 4% of max value
            pos = range(len(nobs))
            for tick,label in zip(pos,ax.get_xticklabels()):
               ax.text(pos[tick], nmax*0.04, nobs[tick], horizontalalignment='center', size='x-small', color='grey', weight='semibold')
        ## plot the individual data point

        sns.stripplot(**paramsNest, edgecolor = 'k', linewidth = 0.4,  size=2, zorder=3, ax = ax, dodge = False, jitter = 0.2, alpha=0.5)
        plt.xlabel(None)
        plt.ylabel(val)
        ax.legend_.remove()


        plt.locator_params(axis = 'y', nbins = plottick) #change the amount of ticks in y
        if not custylim == []:
            plt.ylim(custylim)  
        plt.tight_layout()
    return plt.show()

def customPlot(params, paramsNest, dirName='C:/Users/Windows/Desktop/MAINDATA_OUTPUT', figName = ' ', tradMean = True, viewPlot=False, **kwargs):
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#e26770'])
    """Function to create save the plot to determine directior
    
    Parameters:
        params (dict): set of parameters for plotting main data
        paramsNest (dict): set of parameters for plotting main data (subNesting level)
        dirName(str): string to determine the directory
        tradMean(bool): True, determine if mean std and confidence interval are to be plotted

    kwargs:
        myfsize: list with fig dim 
        myylim: list with y axis limits
        myy: name of the y axis
    """

    # create the frame for the figure
    os.makedirs(dirName,exist_ok=True)

    # the figure size correspond to the size of a plot in inches
    myfsize = kwargs.get('myfsize', None)
    if myfsize is not None:
        f, ax = plt.subplots(figsize=myfsize)
    else:
        f, ax = plt.subplots(figsize=(7, 7))

    ## add if the study was longitudinal / repeated measures
    # castdf=pd.pivot_table(df, values='value', index=['subject'], columns=['genotype'])
    # for i in castdf.index:
    #     ax.plot(['wt','het'], castdf.loc[i,['wt','het']], linestyle='-', color = 'gray', alpha = .3)

    # fill the figure with appropiate seaborn plot
    # sns.boxplot(dodge = 10, width = 0.2, fliersize = 2, **params)
    sns.violinplot(split = True, inner = 'quartile', width=0.6, cut=1, **paramsNest)

    custSpike = kwargs.get('custSpike', None)
    if custSpike is not None:
        plt.setp(ax.collections, alpha=.2)
        sns.stripplot(jitter=0.2, dodge=True, edgecolor='white', size=4, linewidth=0, alpha=0.5, **paramsNest) # change this linewidth = 1 by default
    else:
        sns.stripplot(jitter=0.08, dodge=True, edgecolor='white', size=4, linewidth=0, alpha=1, **paramsNest) # change this linewidth = 1 by default
        plt.setp(ax.collections, alpha=.2)

    # control the figure parameter with matplotlib control
    # this order enable to have transparency of the distribution violin plot

    if tradMean == True:
        # the point plot enable to plot the mean and the standard error 
        # to have the "sd" or 95 percent confidence interval 
        # for sem ci=68
        sns.pointplot(errorbar=('ci', 68), scale=1.2, dodge= -0.1, errwidth=4, **params)
        sns.pointplot(errorbar=('ci', 95), dodge= -0.1, errwidth=2, **params)
        # plot the median could be done with the commented line below however this would be redundant 
        # since the median is already ploted in the violin plot
        # sns.pointplot(ci=None, dodge= -0.2, markers='X',estimator=np.median, **params)
    
    custSpike = kwargs.get('custSpike', None)
    if custSpike is not None:
        sns.stripplot(jitter=0.00, dodge=False, edgecolor='black', alpha=0.5, size=7, linewidth=0.4, **params)
    else:
        sns.stripplot(jitter=0.08, dodge=True, edgecolor='white', size=8, linewidth=1, **params)


    # label plot legend and properties
    ax.legend_.remove()
    sns.despine() 

    myylim = kwargs.get('myylim', None)
    if myylim is not None:
        ax.set_ylim(myylim)

    ax.set_ylabel(params.get('y'), fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    # ax.set_ylim([-5,5])

    ax.set_xlabel(params.get('x'), fontsize=30)
    ax.tick_params(axis="x", labelsize=20, pad=10) # could also use: ax.xaxis.labelpad = 10 // plt.xlabel("", labelpad=10) // or change globally rcParams['xtick.major.pad']='8'


    ### to obligate viewing of the plot
    if viewPlot == True:
        plt.show(block=False)

    myy = kwargs.get('myy', None)
    if myy is not None:
        ax.set_ylabel(myy)
    
    myxlabel = kwargs.get('myxlabel', None)
    if myy is not None:
        ax.set_xlabel(myxlabel)

    plt.tight_layout()
    # property to export the plot 
    # best output and import into illustrator are svg 
    return plt.savefig(dirName+os.sep+figName+".pdf")#,     plt.show(block=False)

def disp():
    for i,j in enumerate(files):
        print(i,j)

def getGenotype(file):
    """
    file: string 
        raw file name of the genotype
    return:
        pandas data frame trim and workable for downstream processing
    """
    geno = pd.read_csv(file)
    geno = geno[['Animal_id', 'Animal_sex', 'Animal_geno', 'Record_folder']]
    geno['Record_folder'] = geno['Record_folder'].str.split('_', expand = True)[0]
    geno['Animal_sex'] = geno['Animal_sex'].str.replace(' ', '')
    geno = geno.drop_duplicates()
    geno = geno.rename(columns={'Record_folder': 'sID'})
    geno.reset_index(inplace = True, drop = True)

    return geno

def figparam(expType = 'KO', restoreDefault = False):
    '''
    Parameters:
    type (str): describe the type of genotype / color to be used
                KO for conventional wt
                CR for conditional rescue in cON line
                CD for conditional deficit in the cOFF line
    '''

    if restoreDefault == True:
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['pdf.fonttype'] = 42 # to make sure it is recognize as true font in illustrator
        # line above may be equivalent to 
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#e26770'])
        plt.ion()
    else:

        ## think of using dict
        # genoDict = {'KO': {'wt': '#6e9bbd', 'het': '#e26770', 'abrev': 'KO', 'desc': 'global defect'}, 
        #  'Rum2': {'wt': '#92c5de', 'het': '#e26770', 'abrev': 'noCR', 'desc': 'global defect'}, 
        #  'EmxRum2': {'wt': '#92c5de', 'het': '#0571b0',  'abrev': 'CR', 'desc': 'emx1+ rescue'}}

        mpl.rcParams['pdf.fonttype'] = 42 # to make sure it is recognize as true font in illustrator
        # line above may be equivalent to 
        mpl.rcParams['svg.fonttype'] = 'none'
        mpl.rcParams['font.sans-serif'] = 'Arial'
        mpl.rcParams['font.family'] = 'sans-serif'
        if expType == 'KO':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#e26770']) ## (['#6e9bbd', '#e26770'] regular line) (['#6e9bbd', '#f4a582'] cortical deficit set) (['#92c5de', '#0571b0'] cortical rescue set - wt always before het)
        if expType == 'CD':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#f4a582'])
        if expType == 'CR':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#92c5de', '#0571b0'])
        if expType == 'noCR':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#92c5de', '#e26770'])
        if expType == 'SPE':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#e26770'])
        if expType == 'SPE1':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#92c5de', '#e26770'])
        if expType == 'FMR1':
            mpl.rcParams['axes.prop_cycle'] = cycler(color=['#d9d9d9', '#66c2a4'])

        mpl.rcParams['figure.figsize'] = [0.77, 1.2]
        mpl.rcParams['figure.frameon'] = False
        mpl.rcParams['figure.autolayout'] = False
        mpl.rcParams['font.size'] = 6
        mpl.rcParams['figure.dpi'] = 300
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 0.5
        mpl.rcParams['ytick.major.width'] = 0.5
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'out'
        mpl.rcParams['xtick.major.bottom'] = True
        mpl.rcParams['ytick.major.size'] =  2
        mpl.rcParams['xtick.major.size'] =  0
        mpl.rcParams['legend.frameon'] = False
        plt.ion()

figparam('FMR1')
figparam(restoreDefault = True)
figparam(restoreDefault = True)

##################################
## GD average methods
##################################

def getCategoricalData(allDat, subdataset):

    pdFirstOnsetMAIN = []
    pdwithProtRetMAIN = []
    errorStoreMAIN = []
    for j in allDat['aid'].unique():
        print(j)
        tst = allDat[allDat['aid'] == j]
        pdFirstOnset, pdwithProtRet,  errorStore= addonAnalysisPump(tst)
        pdFirstOnsetMAIN.append(pdFirstOnset)
        pdwithProtRetMAIN.append(pdwithProtRet)
        errorStoreMAIN.append(errorStore)
    pdFirstOnsetMAIN = pd.concat(pdFirstOnsetMAIN)
    pdwithProtRetMAIN = pd.concat(pdwithProtRetMAIN)
    errorStoreMAIN = pd.concat(errorStoreMAIN)


    touchOccurence = allDat.groupby(['geno', 'aid', 'touchGrp', 'peakCatAmp']).agg({'peakCatAmp': [np.ma.count], 'frame': ['first']})
    touchOccurence = quickConversion(touchOccurence)
    touchOccurence = touchOccurence[touchOccurence['peakCatAmp'] == 'prot']
    touchOccurence.columns = ['geno', 'aid', 'touchGrp', 'peakCatAmp', 'peakCatAmpcount', 'framefirst']
    touchOccurence = touchOccurence[['geno', 'aid', 'touchGrp', 'peakCatAmpcount']]

    subdataset = pd.merge(pdFirstOnsetMAIN, touchOccurence, on = ['aid', 'geno', 'touchGrp'], how = 'left')
    # subdataset = subdataset[(subdataset['accelPeak']>=2) & (subdataset['peakCatAmpcount']==2)]
    subdataset['cat'] = 0
    subdataset.loc[(subdataset['accelPeak']<2) & (subdataset['peakCatAmpcount']==1), 'cat'] = 1
    subdataset.loc[(subdataset['accelPeak']>=2) & (subdataset['peakCatAmpcount']==1), 'cat'] = 2
    subdataset.loc[(subdataset['accelPeak']>=2) & (subdataset['peakCatAmpcount']==2), 'cat'] = 3
    subdataset.loc[(subdataset['accelPeak']>=2) & (subdataset['peakCatAmpcount']>2), 'cat'] = 4

    allDat = createUniqueID(allDat)
    subdataset = createUniqueID(subdataset)
    subdatasetArchive = copy.copy(subdataset)
    subdataset = subdataset[['uniqueTchId', 'cat']]

    allDat = pd.merge(allDat, subdataset, on = ['uniqueTchId'], how = 'left')

    return allDat, subdatasetArchive

def grandAveragePlot(mainsubst, mainsub, winSpanMSm = 40, winSpanMSp= 40, label = 'Curvature (um-1*s)'):
    fig, ax = plt.subplots(1,2, figsize=([10.  ,  4.84]))
    sns.lineplot(data=mainsubst, x='frameNormtoMS', y='valNorm', hue='geno', ax=ax[0])
    ax[0].set_xlim([-winSpanMSm,winSpanMSp])
    ax[0].set_ylim([-30,30])
    ax[0].set_ylabel(label + ' norm. to \n touch onset')
    ax[0].set_xlabel('Time (ms)')
    ax[0].vlines(0, -winSpanMSm,winSpanMSp, linestyles='dashed', color='grey')
    ax[0].hlines(0, -winSpanMSm,winSpanMSp, linestyles='dashed', color='grey')
    ax[0].set_title(label)

    sns.lineplot(data=mainsub, x='frameNormtoMS', y='valNorm', hue='geno', ax=ax[1])
    ax[1].set_ylabel(label + ' norm. to \n touch onset')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_xlim([-winSpanMSm,winSpanMSp])
    ax[1].vlines(0, -60,2, linestyles='dashed', color='grey')
    ax[1].hlines(0, -2,45, linestyles='dashed', color='grey')
    ax[1].set_title(label + ' during touch only')

    plt.tight_layout()

    ax[0].set_ylim([-1,1])
    ax[1].set_ylim([-3,3])

# def getDataFor(myVariable = 'angfilt', mainData = allDat, refData = errorStoreMAIN):
    mainsubst = []
    mainsub = []
    # mainData = allDat
    # refData = errorStoreMAIN #pdwithProtRetMAIN #pdFirstOnsetMAIN ## can be replaced by pdFirstOnsetMAIN and then subsequent category
    refData = createUniqueID(refData)
    mainData = createUniqueID(mainData)
    winSpan = 20
    for i in list(refData['uniqueTchId']):
        # print(i)
        ## work on all the aspect related to the actual touch time 
        sub = mainData[(mainData['uniqueTchId'] == i)]
        firstFrame = sub['frame'].values[0]
        whiskNorm = sub[myVariable].values[0]
        
        ## enlarage the window to look at the trace surrounding the it
        tmpIndex = mainData.index[(mainData['uniqueTchId'] == i)].tolist()

        subst = mainData.loc[(tmpIndex[0]-winSpan):(tmpIndex[-1]+winSpan)]


        subst['frameNorm'] = subst['frame'] - firstFrame
        subst['valNorm'] = subst[myVariable] - whiskNorm
        sub['frameNorm'] = sub['frame'] - firstFrame
        sub['valNorm'] = sub[myVariable] - whiskNorm

        mainsubst.append(subst)
        mainsub.append(sub)

    mainsubst =  pd.concat(mainsubst)
    mainsub = pd.concat(mainsub)

    mainsubst['frameNormtoMS'] = mainsubst['frameNorm']/500*1000
    mainsub['frameNormtoMS'] = mainsub['frameNorm']/500*1000
    winSpanMS = winSpan/500*1000

    return mainsub, mainsubst


def getFinalFigForCat(myVariable = 'angfilt', myCat=1):

    mainData = allDat
    # the refData define the category to fetch
    refData = refDatAll[refDatAll['cat'].isin([myCat])]


    mainsubst = []
    mainsub = []
    winSpanM = 20
    winSpanP = 140

    for i in list(refData['uniqueTchId']):
        # print(i)
        ## work on all the aspect related to the actual touch time 
        sub = mainData[(mainData['uniqueTchId'] == i)]
        firstFrame = sub['frame'].values[0]
        whiskNorm = sub[myVariable].values[0]
        
        ## enlarage the window to look at the trace surrounding the it
        tmpIndex = mainData.index[(mainData['uniqueTchId'] == i)].tolist()
        subst = mainData.loc[(tmpIndex[0]-winSpanM):(tmpIndex[-1]+winSpanP)]

        subst['frameNorm'] = subst['frame'] - firstFrame
        subst['valNorm'] = subst[myVariable] - whiskNorm
        sub['frameNorm'] = sub['frame'] - firstFrame
        sub['valNorm'] = sub[myVariable] - whiskNorm

        mainsubst.append(subst)
        mainsub.append(sub)

    mainsubst =  pd.concat(mainsubst)
    mainsub = pd.concat(mainsub)

    mainsubst['frameNormtoMS'] = mainsubst['frameNorm']/500*1000
    mainsub['frameNormtoMS'] = mainsub['frameNorm']/500*1000
    winSpanMSm = winSpanM/500*1000
    winSpanMSp = winSpanP/500*1000


    ##****************************************************
    ## final structure for figure with 
    ##****************************************************

    byGenoM = mainsub.groupby(['geno', 'aid', 'frameNormtoMS']).agg({'valNorm':np.mean})
    byGenoM = quickConversion(byGenoM)
    byGenoST = mainsubst.groupby(['geno', 'aid', 'frameNormtoMS']).agg({'valNorm':np.mean})
    byGenoST = quickConversion(byGenoST)

    winSpanMSp = 40
    mainsubst = mainsubst.sort_values(by = 'geno', ascending=False)
    mainsub = mainsub.sort_values(by = 'geno', ascending=False) 
    byGenoM = byGenoM.sort_values(by = 'geno', ascending=False)
    byGenoST = byGenoST.sort_values(by = 'geno', ascending=False)   
    grandAveragePlot(mainsubst, mainsub, winSpanMSm = winSpanMSm, winSpanMSp= winSpanMSp)
    f.suptitle('Gd avg by touch')
    plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'KOgdAvgbyTchCURV_'+str(myCat)+'.svg')
    # plt.close('all')
    grandAveragePlot(byGenoST, byGenoM, winSpanMSm = winSpanMSm, winSpanMSp= winSpanMSp)
    f.suptitle('Gd avg by animal')
    plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'KOgdAvgbyIDCURV_'+str(myCat)+'.svg')
    # plt.close('all')

##################################
## pump methods
##################################

def convertStartStopTotimeserie(touchEventFilename, filterIntveral=False):
    """ convert a data frame with start and stop column to a timeserie
    this function can be used to be able to easily compare / extract information from manually anotated data set
    for which the frame of begining of the pole touch (start) and end (stop) are registered
    start   stop
    1520    1526 (#1520 correspond to frame number of touch start and 1526 of touch stop
    Arguments:
        touchEventFilename (str): string of touchEventFilename that contain start and stop column for pole start pole stop that mark
                        begining and end of a touch
        noTouchVal (int): integer value to categorize the touch category (eg. no touch =0)
        touchVal (int): integer value to categorize the touch category (eg. no touch =1)
    Return:
        timeserie data frame
    Usage:
        touchEventFilename = '/home/rum/Desktop/img/2019-11-21_poleTouchManualGT.csv'
        manualDat = convertStartStopTotimeserie(touchEventFilename)
    """

    # extract the data 
    if '.xlsx' in touchEventFilename:
        manual = pd.read_excel(touchEventFilename)
    else:
        manual = pd.read_csv(touchEventFilename)

    manual = manual[manual['Behavior']  == 'bouts duration']
    manual = manual[['Start (s)', 'Stop (s)']]
    manual = manual.reset_index(drop=True)
    manual = manual*500
    manual.index = manual.index.rename('touchEventGrp')
    manual = manual.reset_index()
    
    # get the first time of interval to get the proper category
    if filterIntveral == True:
        manual['Interval'] = manual['Start (s)'].shift(-1) - manual['Stop (s)']
        manual['Duration'] = manual['Stop (s)'] - manual['Start (s)']
        manual['Interval'] = manual['Interval']/500
        for i,j in enumerate(manual['touchEventGrp']):
            if manual['Interval'][i] < 0.05:
                manual['touchEventGrp'][i+1] = manual['touchEventGrp'][i]

        # summarize the new interval
        summary = pd.DataFrame({'LastwhiskFrame' : dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].last(),
                            'FirstwhiskFrame': dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].first()})
        summary.reset_index(inplace=True)
        summary['whiskDuration'] = summary['LastwhiskFrame'] -  summary['FirstwhiskFrame']
        summary['noWhiskbefore'] = [summary['FirstwhiskFrame'][0]] + (summary['FirstwhiskFrame'].shift(-1) - summary['LastwhiskFrame']).dropna().tolist()
        summary = summary[summary['whiskDuration'] > whiskLengthlim]
        summary.reset_index(inplace = True, drop = True)
        summary['whiskGrp'] = summary.index
    else:
        summary = manual
        summary['Interval'] = summary['Start (s)'].shift(-1) - summary['Stop (s)']
        summary['Interval'] = summary['Stop (s)'] - summary['Start (s)']

    # expand the time series based on the new interval 
    listalldfs = []
    for i in range(len(summary)):
        # print(i)
        list1 = pd.DataFrame({'frame': list(range(int(summary.loc[i]['Start (s)']), int(summary.loc[i]['Stop (s)']) + 1)),
                              'touchCat': 1})
        # print(list1)
        listalldfs.append(list1)
    
    listalldfs = pd.concat(listalldfs)
    listalldfs.index = listalldfs.frame
    listalldfs = listalldfs.drop_duplicates()
    mock = pd.DataFrame({'frame': list(range(0, max(listalldfs.frame) + 1)),
                         'default': 0})
    res = pd.concat([listalldfs, mock], axis=1, sort=False)
    res = res.iloc[:, [1, 2]]
    res['touchCat'] = res['touchCat'].fillna(0)
    # res.loc[res['touchCat'] == 0, 'touchCat'] = noTouchVal
    # res.loc[res['touchCat'] == 1, 'touchCat'] = touchVal
    res['touchGrp'] = (res['touchCat'].diff(1) != 0).astype('int').cumsum()
    # plt.plot(res.touch)
    # plt.plot(res.touchGrp)
    res['frame'] = res.index
    
    return summary, res

def getCleanPoleTimeSerie(poleFileName, timeserieReg = False):
    poleDat = pd.read_csv(poleFileName)
    poleDat = poleDat.rename(columns={'fid': 'frame'})
    # this section is to deal with the timeseries duplicates and gaps
    # first deal with duplicates
    ## poleDat[poleDat['fid'].duplicated()] ## check for duplicates
    ## this method here is not ideal ideal would be important to export data with follicles and remove duplicate 
    ## by keeping the value which are closer to the follicle may be over kill with the current type of error
    if timeserieReg == True:
        poleDat['fid'] = poleDat.index # to be REMOVED
        poleDat = poleDat.drop_duplicates(subset=['fid'])
        # first deal with duplicates
        poleDat = poleDat.rename(columns={'fid': 'frame'})
        # second deal with missing
        # find_missing(list(poleDat.fid))
        poleDat.index = poleDat['frame'] 
        poleDat = poleDat.reindex(range(poleDat.index[-1]+1)).bfill()
        poleDat['frame'] = poleDat.index # this is necessary as operation above just filled missing frame
        poleDat.index.name = None
    
    return poleDat

def roughPlotFct(pole, touchEventFilename, idVar,  geno = 'wt', filterIntveral=False, val = 2):
    if geno == 'wt':
        pCol = 'blue'
    else:
        pCol = 'red'

    summary, res = convertStartStopTotimeserie(touchEventFilename[0])
    pole = getCleanPoleTimeSerie(pole[0])

    # summary, res = convertStartStopTotimeserie(eventFile[0])
    # pole = getCleanPoleTimeSerie(poleFile[0])

    combineData = pd.merge(res, pole, on = 'frame')

    if filterIntveral == True:
        interval = np.array([val, val+2])*500
        sub = pole[(pole['frame']>=interval[0]) & (pole['frame']<=interval[1])]
        res = res[(res['frame']>=interval[0]) & (res['frame']<=interval[1])]
        summary = summary[(summary['Start (s)']>=interval[0]) & (summary['Start (s)']<=interval[1])]
    else:
        sub = pole



    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True)
    ax1.plot(res['frame'], res['touchCat'], color = pCol)
    ax2.plot(sub['frame'], sub['filt_curvature'], color = pCol)
    ax3.plot(sub['frame'], sub['delta_curv'], color = pCol)
    ax4.plot(sub['frame'], sub['angfilt'], color = pCol)
    ax5.plot(sub['frame'], sub['ang_accel_filt_'], color = pCol)

    ax2.set_ylim([-0.009, 0.004])
    ax3.set_ylim([-0.009, 0.004])


    ax5.set_xlabel('frame (500fps)')

    ax1.set_ylabel('Touch')
    ax2.set_ylabel('filt_curvature')
    ax3.set_ylabel('delta_curv')
    ax4.set_ylabel('angfilt')
    ax5.set_ylabel('ang_accel_filt_')

    for i,j in summary.iterrows():
        rect = matplotlib.patches.Rectangle((j['Start (s)'], -0.009), j['Interval'], 0.13, color ='grey', alpha = 0.3)
        ax2.add_patch(rect)

    for i,j in summary.iterrows():
        rect = matplotlib.patches.Rectangle((j['Start (s)'], -35), j['Interval'], 70, color ='grey', alpha = 0.3)
        ax3.add_patch(rect)

    for i,j in summary.iterrows():
        rect = matplotlib.patches.Rectangle((j['Start (s)'], -3.2), j['Interval'], 6.4, color ='grey', alpha = 0.3)
        ax4.add_patch(rect)

    for i,j in summary.iterrows():
        rect = matplotlib.patches.Rectangle((j['Start (s)'], -1.5), j['Interval'], 3, color ='grey', alpha = 0.3)
        ax5.add_patch(rect)


    f.suptitle(str(idVar))
    plt.tight_layout()

    plt.show()

    return combineData, summary, f, (ax1, ax2, ax3, ax4, ax5)

def whiskGetKeyOnset(df, sampleRate = 500, whiskCycleLim = 0.25, whiskLengthlim = 0):
    '''
    df: pd.DataFrame on which other whisking measure are appended
    sampleRate: data acquisition rate of the video fps usually 500 for highspeed
    whiskCycleLim: default is 250 ms expressed in seconds 0.25 every whisk with high amplitude during this phase will be concatenated in the same category thus whisking onset separated by 250 ms or less are define as the same whisk
    whiskLengthlim: whiskk lasting 150 ms or more are included otherwise discareded
    '''

    whiskCycleLim = whiskCycleLim * sampleRate #conversion to frame number # to determine the whisk cycle or lasting whisk events only events lasting more than 250 ms are retained
    whiskLengthlim = whiskLengthlim * sampleRate #conversion to frame number

    resSubset = res[['frame', 'touchCat', 'touchGrp']]
    resSubset = resSubset[resSubset['touchCat'] == 1]
    resSubset['Interval'] = resSubset['frame'].shift(-1)-resSubset['frame']
    resSubset.reset_index(inplace=True, drop=True)
    # toTry = toTry[toTry['Interval']>=whiskCycleLim]
    resSubset = resSubset.dropna()

    # s = time.time()
    # this part need to be optmized should use apply and the use of a function
    # resSubset['trueWhisk'] = np.nan
    for i,j in enumerate(res['touchGrp']):
        print(i,j)
        if resSubset['Interval'][i] < whiskCycleLim:
            print(i,j)
            resSubset['touchGrp'][i+1] = resSubset['touchGrp'][i]

    # e = time.time()  -s


    summary = pd.DataFrame({'LastwhiskFrame' : dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].last(),
                        'FirstwhiskFrame': dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].first()})
    summary.reset_index(inplace=True)
    summary['whiskDuration'] = summary['LastwhiskFrame'] -  summary['FirstwhiskFrame']
    summary['noWhiskbefore'] = [summary['FirstwhiskFrame'][0]] + (summary['FirstwhiskFrame'].shift(-1) - summary['LastwhiskFrame']).dropna().tolist()
    summary = summary[summary['whiskDuration'] > whiskLengthlim]
    summary.reset_index(inplace = True, drop = True)
    summary['whiskGrp'] = summary.index

    # this section is enabling to get read of the slight delay as it is originally determined by 2.5 changes in degree and at phase 0
    summary['FirstwhiskFrameFromZero'] = np.nan
    for SumVal in summary['FirstwhiskFrame']:
        if SumVal<500:
            pass
        else:
            xMax = fromZero(df, SumVal)
            tmpVal = SumVal-xMax
            summary.loc[summary['FirstwhiskFrame'] == SumVal, 'FirstwhiskFrameFromZero'] = tmpVal

    return summary

def find_missing(lst): 
    return [x for x in range(lst[0], lst[-1]+1)  
                               if x not in lst] 

def getTotalFrames(vidPath):
    video = cv2.VideoCapture(vidPath)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total

def closestVal(lst, K): 
     lst = np.asarray(lst) 
     idx = (np.abs(lst - K)).argmin() 
     return lst[idx] 

def getAllpeak(data, graph = True):
    upPic, _ = find_peaks(data)
    downPic, _ = find_peaks(-data)
    allPeak = np.concatenate((upPic, downPic), axis = 0)
    allPeak = data.index[allPeak]
    if graph == True:
        plt.figure()
        plt.plot(data)
        plt.plot(allPeak, data[allPeak],'.', color = 'red')
        plt.show()
    return allPeak

def getPeakAmpVel(idVar, eventsPath, whiskPath):

    eventFile, poleFile, peakFile, aid = dataExtraction(idVar, eventsPath, whiskPath)
    summary, res = convertStartStopTotimeserie(eventFile[0])
    pole = getCleanPoleTimeSerie(poleFile[0])

    ## look at the data of 944 between frame 600 and 800
    # retraction during touch are corresponding to pumps
    # work on find peak within touch 
    tst = pd.merge(pole, res, on = 'frame', how = 'right')

    peakAmp = []
    peakVel = []
    peakCurv = []

    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        # create a subset data frame for the specific touch category of interest
        sub = tst[tst['touchGrp'] == i]

        # this section enable the interval as peak curvature is reached
        # to expand the interval
        tmpIndex = tst.index[tst['touchGrp'] == i].tolist()
        winSpan = 40
        if tmpIndex[0]<winSpan:
            subts = tst.iloc[(0):tmpIndex[-1]+winSpan] # if the index is at the very begining of the time serie
        elif tmpIndex[-1]+winSpan>tst.index[-1]:
             subts = tst.iloc[(tmpIndex[0]-winSpan):tmpIndex[-1]]
        else:
            subts = tst.iloc[(tmpIndex[0]-winSpan):tmpIndex[-1]+winSpan]

        ## peakCurvature amplitude ##############################
        ## **********************************************
        # need to be added for peak curvature and look at the data this way
        # go and check Sheldon criteria for this or may be can just be merged 

        peaksCurv, _ = find_peaks(-sub['delta_curv'].values, height=sub['delta_curv'].min(),  width = 5, distance=10)

        if peaksCurv.size == 0:
            peaksCurv, _ = find_peaks(-sub['delta_curv'].values, height=sub['delta_curv'].min())

        subtsPeakCurv = pd.DataFrame({
        'frame': sub.index[peaksCurv].tolist(),
        'peakCatCurv': len(sub.index[peaksCurv].tolist())*['peakCurv']
        })

        peakCurv.append(subtsPeakCurv)

        ## angle amplitude ##############################
        ## **********************************************
        # need to find protraction and retraction peaks
        peaksPro, _ = find_peaks(subts['angfilt'].values, height=subts['angfilt'].min())
        peaksRet, _ = find_peaks(-subts['angfilt'].values, height=subts['angfilt'].min())

        subtsPeak = pd.DataFrame({
            'frame': subts.index[peaksPro].tolist() + subts.index[peaksRet].tolist(),
            # 'peakY': subts[subts.index[peaksPro]].tolist() + subts[subts.index[peaksRet]].tolist(),
            'peakCatAmp': len(subts.index[peaksPro].tolist())*['prot'] + len(subts.index[peaksRet].tolist())*['ret']
            })
        subtsPeak['touchGrpUpdtAmp'] = i

        subtsPeak = subtsPeak.sort_values('frame').reset_index(drop=True)

        # this here enable to narrow back down the interval 
        subtsPeakAmp = subtsPeak[subtsPeak['frame'].isin(sub['frame'])]

        # condition present for extremely short events that have no peak
        if subtsPeakAmp.empty:
            tmpValsubstitute = closestVal(subtsPeak['frame'].values,tmpIndex[0])
            subtsPeakAmp = subtsPeak[subtsPeak['frame'] == tmpValsubstitute]

        #every peak angle at touch should start with protraction and and with retracction
        #condition below to make sure the last retraction peak is include even if outside touch

        if subtsPeakAmp.iloc[-1]['peakCatAmp'] == 'prot':
            subtsPeakAmp = subtsPeak[subtsPeakAmp.index[0]:subtsPeakAmp.index[-1]+2]
        
        peakAmp.append(subtsPeakAmp)


        ## angle amplitude ##############################
        ## **********************************************
        #### in this seciton work on the angle velocity
        #### instead of an approach finding + or - position 
        #### use an approach to find 

        subtsPeakInterval = subtsPeak.loc[subtsPeakAmp.index[0]-1:subtsPeakAmp.index[-1]+1, 'frame'].values
        subts = tst[subtsPeakInterval[0]:subtsPeakInterval[-1]]

        peaksPro, _ = find_peaks(subts['ang_vel_filt_'].values, height=subts['ang_vel_filt_'].min())
        peaksRet, _ = find_peaks(-subts['ang_vel_filt_'].values, height=subts['ang_vel_filt_'].min())

        subtsPeakVel = pd.DataFrame({
            'frame': subts.index[peaksPro].tolist() + subts.index[peaksRet].tolist(),
            # 'peakY': subts[subts.index[peaksPro]].tolist() + subts[subts.index[peaksRet]].tolist(),
            'peakCatVel': len(subts.index[peaksPro].tolist())*['prot'] + len(subts.index[peaksRet].tolist())*['ret'],
            })
        subtsPeakVel = subtsPeakVel.sort_values('frame').reset_index(drop=True)

        # narrow down the overlap trim rightward overhang
        subtsPeakVel = subtsPeakVel[subtsPeakVel['frame']<subtsPeakAmp['frame'].max()]


        ## following section below to prevent both overlap and curve flatness altering 
        ## velocities
        bestPeakVel = []
        for ijidx, ij in subtsPeakAmp.iterrows():
            # print(ijidx, ij) 
            if subtsPeak.loc[subtsPeak['frame']<ij['frame'], 'frame'].empty:
                continue
            else:  
                ijLow = subtsPeak.loc[subtsPeak['frame']<ij['frame'], 'frame'].values[-1]
                ijHigh = ij['frame']
                # print(ijLow, ijHigh)
                tmpSubVelInterval = subtsPeakVel[(subtsPeakVel['frame']>=ijLow) & (subtsPeakVel['frame']<=ijHigh) & (subtsPeakVel['peakCatVel']==ij['peakCatAmp'])]
                if tmpSubVelInterval.empty:
                    continue
                tmpBestFrame = tst.loc[tst['frame'].isin(tmpSubVelInterval['frame']),'ang_vel_filt_'].idxmax()
                tmpBestDat = pd.DataFrame({
                    'frame': [tmpBestFrame],
                    'peakCatVel': [ij['peakCatAmp']]
                    })
                tmpBestDat['touchGrpUpdtVel'] = i
                bestPeakVel.append(tmpBestDat)
        if bestPeakVel == []:
            bestPeakVel = [subtsPeakVel]
        subtsPeakVel = pd.concat(bestPeakVel)
       
        peakVel.append(subtsPeakVel)
        # print(subtsPeak)

    peakAmp = pd.concat(peakAmp)
    peakVel = pd.concat(peakVel)
    peakCurv = pd.concat(peakCurv)
    tst = pd.merge(tst, peakAmp, on = 'frame', how = 'left')
    tst = pd.merge(tst, peakVel, on = 'frame', how = 'left')
    tst = pd.merge(tst, peakCurv, on = 'frame', how = 'left')

    return tst

def dataExtraction(idVar, eventsPath, whiskPath):
    ''' very rough not super functional function should be substitued with class'''
    
    idVar = str(idVar)
    eventFile = glob.glob(eventsPath + os.sep + '*' + idVar + '*events*.xlsx')
    if eventFile == []:
        eventFile = glob.glob(eventsPath + os.sep + '*' + idVar + '*events*.csv')
    poleFile = glob.glob(whiskPath + os.sep + '*' + idVar + '_p.csv')
    peakFile = glob.glob(whiskPath + os.sep +  '*' + idVar + '*peaks.csv')

    return eventFile, poleFile, peakFile, idVar

def checkGeno(allDat, geno, experiment = ['HS009', 'WDIL003']):
    aidDat = list(allDat['aid'].unique().astype(int))
    aidGen = list(geno.loc[geno['cohort'].isin(experiment), 'aid'].values)

    return [x for x in aidDat+aidGen if x not in aidGen]
    # [x for x in aidDat+aidGen if x not in aidDat or x not in aidGen] # could be use for other purpose
    # [x for x in aidDat+aidGen if x in aidDat or x in aidGen] # could be use for other purpose

def quickConversion(tmp, myCol=None, option=1):
    ''' convert groupby pandas table into a simpler version
    '''
    if option ==1:
        tmp.columns = ['value']
        tmp = tmp.reset_index()
        if tmp.columns.nlevels > 1:
            tmp.columns = ['_'.join(col) for col in tmp.columns] 
        tmp.columns = tmp.columns.str.replace('[_]','', regex=True)
        if myCol:
            tmp = tmp.rename(columns={np.nan: myCol})
        return tmp
    elif option ==2:
        tmp.columns = ['_'.join(col) for col in tmp.columns] 
        tmp = tmp.reset_index(drop=True)
        tmp.columns = tmp.columns.str.replace('[_]','', regex=True)
        return tmp

def figHelp():
    print('''
    ### FOR FIGURE SIZE #################################
    
    import matplotlib.plt
    fig = plt.figure()
    size = fig.get_size_inches()*fig.dpi # size in pixels
    
    fig = plt.gcf()
    size = fig.get_size_inches()*fig.dpi # size in pixels
    ''')

    print('''
    ### FOR FIGURE POSITION #################################
    
    fig, ax = plt.subplots()
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    mngr.window.setGeometry(238,1180,1494,855)

    # get the QTCore PyRect object
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    print(x,y,dx,dy)

    mngr.window.setGeometry(newX, newY, dx, dy)
    ''')

    print('''
    ### PANDAS HELP #################################
        pd.set_option('display.max_rows', 500)
        ''')


    print('''
    ### TO ARRANGE PLOTS $$############################
    0) sortplot(nrow=2)
    1) collapseplot(position = 'tl', size=200)

        ''')

def pf(monitor = 3):
    plt.figure()
    mngr = plt.get_current_fig_manager()
    if monitor == 3:
        mngr.window.setGeometry(-1577,1111,1494,855)
    else:
        mngr.window.setGeometry(238,1180,1494,855)

def whiskAngleFilter(data, bandPassCutOffsInHz = [6, 60]):

    # fs = 500; #sampling frequency in Hz
    # lowcut = 4; #unit in Hz
    # highcut = 30; #unit in Hz
    # [b,a]=butter(2, [lowcut highcut]/(fs/2)); # open ephys
    # angfilt = filtfilt(b,a,data);

    '''

    process the signal angle and band-pass filter between 6-60Hz with a 4th order Butterworth filter

    see Svoboda publication: 10.1016/j.neuron.2019.07.027

    '''
    if not isinstance(data, (np.ndarray)):
        data = np.array(data)

    data = np.nan_to_num(data)

    # bandPassCutOffsInHz = [4,30] #Sheldon whisk parameter [4,30] Svoboda [6,60]
    sampleRate = 500
    W1 = bandPassCutOffsInHz[0] / (sampleRate/2)
    W2 = bandPassCutOffsInHz[1] / (sampleRate/2)
    passband = [W1, W2]
    b, a = butter(4, passband, 'bandpass')
    anglefilt = filtfilt(b, a, data)


    return anglefilt

def datLoadTmp(val):
    file = [r"C:\Users\Windows\Desktop\### WISKER ###"+ os.sep+ 'WDIL002_0005-SyngapKO.csv', r"C:\Users\Windows\Desktop\### WISKER ###"+ os.sep+ 'WDIL003_HS0009-SyngapRUM1.csv', r"C:\Users\Windows\Desktop\### WISKER ###"+ os.sep+ 'WDIL004_0008-SyngapRescueEMXRUM2.csv']
    allDat = pd.read_csv(r"C:\Users\Windows\Desktop\### WISKER ###\WDIL002_0005-SyngapKOupdated.csv")
    geno = pd.read_csv(r"Y:\Sheldon\Highspeed\analyzed\NewExport20210209\geno.csv")
    allDat = pd.merge(allDat, geno, on = ['aid', 'cohort'], how = 'left')
    tst = allDat[allDat['aid'] == val]
    return tst

def qcPlotting(tst):
    ########## overall ##########
    # to manipulate and save data stream
    tstori = copy.copy(tst)
    tst = copy.copy(tstori)

    ########## overall ##########
    f, (ax3, ax1, ax0, ax2, ax4) = plt.subplots(5, 1, sharex=True)

    # to plot all the traces for th entire timeserie for the angle amplitude
    mainTrans = 0.3
    subTrans = 1
    ax3.plot(tst['frame'], tst['delta_curv'], alpha = mainTrans) # plot the entire timeseries
    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        ax3.plot(sub['frame'], sub['delta_curv'], alpha = subTrans)
        ax3.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')


    ax0.plot(tst['frame'], tst['phase'], alpha = mainTrans) # plot the entire timeseries
    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        ax0.plot(sub['frame'], sub['phase'], alpha = subTrans)
        ax0.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')


    ax4.plot(tst['frame'], tst['ang_accel_filt_'], alpha = mainTrans) # plot the entire timeseries
    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        ax4.plot(sub['frame'], sub['ang_accel_filt_'], alpha = subTrans)
        ax4.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')
    # tstSet = tst[tst['peakCatCurv'].isin(['peakCurv'])]
    # ax3.plot(tstSet['frame'], tstSet['delta_curv'], 'x', color = 'black')

    # to plot all the traces for th entire timeserie for the angle amplitude
    ax1.plot(tst['frame'], tst['angfilt'], alpha = mainTrans) # plot the entire timeseries
    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        ax1.plot(sub['frame'], sub['angfilt'], alpha = subTrans)
        ax1.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')

    tstSet = tst[tst['peakCatAmp'].isin(['ret'])]
    ax1.plot(tstSet['frame'], tstSet['angfilt'], 'x', color = 'red')
    ax0.vlines(tstSet['frame'], -2, 2, 'orange')
    ax0.hlines(0, 0, 25000, 'orange')
    tstSet = tst[tst['peakCatAmp'].isin(['prot'])]
    ax1.plot(tstSet['frame'], tstSet['angfilt'], 'x', color = 'green')


    # to plot all the traces for th entire timeserie for the angle amplitude
    ax2.plot(tst['frame'], tst['ang_vel_filt_'], alpha = mainTrans)
    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        ax2.plot(sub['frame'], sub['ang_vel_filt_'], alpha = subTrans)
        ax2.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')

    tstSet = tst[tst['peakCatVel'].isin(['ret'])]
    ax2.plot(tstSet['frame'], tstSet['ang_vel_filt_'], 'x', color = 'red')
    tstSet = tst[tst['peakCatVel'].isin(['prot'])]
    ax2.plot(tstSet['frame'], tstSet['ang_vel_filt_'], 'x', color = 'green')
    tstSetCurv = tst[tst['peakCatCurv'].isin(['peakCurv'])]
    ax3.plot(tstSetCurv['frame'], tstSetCurv['delta_curv'], 'x', color = 'green')

    ax0.set_ylabel('angle')
    ax3.set_ylabel('curvature')
    ax2.set_ylabel('angular velocity')
    ax1.set_ylabel('angle')
    ax2.set_xlabel('Time (frame @500fps)')
    ax4.set_xlabel('ang_accel_filt_')
    # plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\figdatPNG.svg')


    ########## for given touch  ##########
    # to plot all the traces for th entire timeserie for the angle amplitude
    ########## for given touch  ##########

    # tstori = copy.copy(tst)

    # tst = copy.copy(tstori)
    # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # TouchEventToInspect = 4

    # indxtmp = tst.loc[tst['touchGrp']==TouchEventToInspect,'frame'].values # input the touch cluster to verify
    # tst = tst[indxtmp[0]-20:indxtmp[-1]+20]
    # ax1.plot(tst['angfilt'], alpha = mainTrans) # plot the entire timeseries
    # for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
    #     # print(i)
    #     sub = tst[tst['touchGrp'] == i]
    #     ax1.plot(sub['frame'], sub['angfilt'], alpha = subTrans)
    #     ax1.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')

    # tstSet = tst[tst['peakCatAmp'].isin(['ret'])]
    # ax1.plot(tstSet['frame'], tstSet['angfilt'], 'x', color = 'red')
    # tstSet = tst[tst['peakCatAmp'].isin(['prot'])]
    # ax1.plot(tstSet['frame'], tstSet['angfilt'], 'x', color = 'green')

    # # to plot all the traces for th entire timeserie for the angle amplitude
    # ax2.plot(tst['ang_vel_filt_'], alpha = mainTrans)
    # for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
    #     # print(i)
    #     sub = tst[tst['touchGrp'] == i]
    #     ax2.plot(sub['frame'], sub['ang_vel_filt_'], alpha = subTrans)
    #     ax2.text(sub['frame'].values[0], 0, i, horizontalalignment='center', verticalalignment='center')

    # tstSet = tst[tst['peakCatVel'].isin(['ret'])]
    # ax2.plot(tstSet['frame'], tstSet['ang_vel_filt_'], 'x', color = 'red')
    # tstSet = tst[tst['peakCatVel'].isin(['prot'])]
    # ax2.plot(tstSet['frame'], tstSet['ang_vel_filt_'], 'x', color = 'green')


##################################
## pumpDefwitVel methods
##################################

def p(i, var = 'ang_accel_filt_', frame = False): # ang_accel_filt_ angfilt
    plt.clf()
    sub = tst[tst['touchGrp'] == i]
    tmpIndex = tst.index[tst['touchGrp'] == i].tolist()
    winSpan = 20
    subts = tst.loc[(tmpIndex[0]-winSpan):(tmpIndex[-1]+winSpan)]

    if frame == False:
        plt.plot(subts.index, subts[var], lw=1, color='grey')
        plt.plot(sub.index, sub[var], lw=2, color='red')
    else:
        plt.plot(subts['frame']/500*1000, subts[var], lw=1, color='grey')
        plt.plot(sub['frame']/500*1000, sub[var], lw=2, color='red')
# plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'tmp'+'.svg')

def addonAnalysisPump(tst):
    pdFirstOnset = []
    pdwithProtRet = []
    errorStore = []

    # here the unique is present because some duplicate frame seems to appear
    # the presence of duplicate is not normal since duplicate removal based on frame
    # was carried out in earlier code
    tst = tst.drop_duplicates('frame')
    tst = tst.reset_index(drop = True)

    for i in tst.loc[tst['touchCat'] == 1, 'touchGrp'].unique():
        # print(i)
        sub = tst[tst['touchGrp'] == i]
        # index of the data that correspond to the touch 
        tmpIndex = tst.index[tst['touchGrp'] == i].tolist()
        winSpan = 60
        subts = tst.loc[(tmpIndex[0]-winSpan):(tmpIndex[-1]+winSpan)]

        # to be able to find the peak prior to the touch
        allP = getAllpeak(subts['angfilt'], False)
        valtmp = np.sort(allP)

        ## 2 conditions statement to be able to deal 
        ## with the length of the series
        if tmpIndex[0]-winSpan<tst.index[0]:
            retbeforeTouch = tst.index[0]
        else:
            # get the retraction point closest to actu
            retbeforeTouch = valtmp[valtmp<tmpIndex[0]][-1]
        if tmpIndex[-1]+winSpan>tst.index[-1]:
            retafterTouch = tst.index[-1]
        elif valtmp[valtmp>tmpIndex[-1]].size == 0:
            retafterTouch = sub.index[-1]
        else:
            retafterTouch = valtmp[valtmp>tmpIndex[-1]][0]

        if sub[sub['peakCatAmp'] == 'prot'].empty:
            


            # to catch the error for the n
            errortmp = pd.DataFrame({
            'touchGrp': i,
            'aid': tst.aid.unique(),
            'cohort': tst.cohort.unique(),
            'geno': tst.geno.unique(),
            'noPeak': 1})
            errorStore.append(errortmp)
            continue


        # get the time from onset of touch to peak protraction
        Ret2Peakprot = sub[sub['peakCatAmp'] == 'prot'].index[0] - retbeforeTouch
        # get the time from start of protraction to touch onset
        Ret2TouchOnset = tmpIndex[0] - retbeforeTouch
        # get the time from touch onset 
        touchOnset2Peakprot = sub[sub['peakCatAmp'] == 'prot'].index[0] - tmpIndex[0]

        ###################################
        # number of peaks for calssification // category
        ###################################
        allPCat = len(getAllpeak(sub['ang_accel_filt_'], False))

        FOtmp = pd.DataFrame({
            'touchGrp': i,
            'aid': tst.aid.unique(),
            'cohort': tst.cohort.unique(),
            'geno': tst.geno.unique(),
            # potentially include the retbeforeTouch here for downstream analysis
            # thus if the retbeforeTouch frame belong to another touch then merge
            'ret2Touch':Ret2TouchOnset, 
            'Ret2Peakprot':Ret2Peakprot,
            'touchOnset2Peakprot': touchOnset2Peakprot,
            'accelPeak': allPCat
            })
        pdFirstOnset.append(FOtmp)


        ###################################
        # get all the Ret2PeakWhitin touch
        ###################################
        retbeforeTouch = np.array([subts['frame'][retbeforeTouch]])
        retafterTouch  = np.array([subts['frame'][retafterTouch]])
        touchOnsetPre = sub['frame'].iloc[0] - retbeforeTouch
        touchOffsetPost = retafterTouch - sub['frame'].iloc[-1]
        protRetFComb = sub.loc[sub['peakCatAmp'].notna(), 'frame'].values
        protRetFComb = np.concatenate((retbeforeTouch, protRetFComb, retafterTouch), axis = 0)
        nprot = len(protRetFComb)
        touchProtractionPeak = len(sub.loc[sub['peakCatAmp'] == 'prot', 'frame'].values)

        if subts.loc[subts['frame'] ==protRetFComb[0], 'angfilt'].item()<0:
            valP1 = protRetFComb[1:][::2][:touchProtractionPeak]
            valR1 = protRetFComb[2:][::2][:touchProtractionPeak]

            protTime = protRetFComb[1:][::2][:touchProtractionPeak] - protRetFComb[::2][:touchProtractionPeak]
            retTime = protRetFComb[2:][::2][:touchProtractionPeak] -  protRetFComb[1:][::2][:touchProtractionPeak]
            protAmpt = subts.loc[subts['frame'].isin(list(valP1)), 'angfilt'].values
            retAmpt = subts.loc[subts['frame'].isin(list(valR1)), 'angfilt'].values
            protAmp = protAmpt - retAmpt
            retAmp  = retAmpt - protAmpt
        else:
            # here could refine if this is positive could be due to same wisk cycle
            # need to count nan
            protTime = [np.nan]
            retTime = [np.nan]
            protAmp = [np.nan]
            retAmp = [np.nan]

        # store the information for all the peak done
        trPro = pd.DataFrame({'indProtDuration':protTime})
        trRet = pd.DataFrame({'indRetDuration': retTime})
        trProAmp = pd.DataFrame({'protAmp':protAmp})
        trRetAmp = pd.DataFrame({'retAmp': retAmp})
        trOnsetpre = pd.DataFrame({'touchOnsetPre': touchOnsetPre})
        trOffsetpost = pd.DataFrame({'touchOffsetPost': touchOffsetPost})
        WPRtmp = pd.concat([trPro, trRet, trOnsetpre, trOffsetpost], axis=1)
        WPRtmp['touchGrp']= i
        WPRtmp['aid'] = tst.aid.unique()[0]
        WPRtmp['cohort'] = tst.cohort.unique()[0]
        WPRtmp['geno'] = tst.geno.unique()[0]
            # potentially include the retbeforeTouch here for downstream analysis
            # thus if the retbeforeTouch frame belong to another touch then merge
        pdwithProtRet.append(WPRtmp)

        # do this after the loop
    pdFirstOnset = pd.concat(pdFirstOnset)
    pdwithProtRet = pd.concat(pdwithProtRet)

    if errorStore == []:
        errortmp = pd.DataFrame({
        'touchGrp': [np.nan],
        'aid': [np.nan],
        'cohort':[np.nan],
        'geno': [np.nan],
        'noPeak': [np.nan]})
        errorStore.append(errortmp)

    errorStore = pd.concat(errorStore)
        # try to work on charactherization based 

    return pdFirstOnset, pdwithProtRet, errorStore

def plotTraceError(aid, subdataset, myVariable, timeBase, col=['blue', 'red'], Myax=None):

    '''
    myVaraiable (str): name of the column of interest ['angfilt']
    timeBase (str): name of the column to be used for time base either frame or timeMS
    '''

    for i in list(subdataset.loc[(subdataset['aid'] == aid), 'touchGrp']):
        # print(i)
        sub = allDat[(allDat['touchGrp'] == i) & (allDat['aid']==aid)]
        tmpIndex = allDat.index[(allDat['touchGrp'] == i) & (allDat['aid']==aid)].tolist()
        winSpan = 20
        tmpDat = allDat[(allDat['aid'] == aid)]
        subst = tmpDat.loc[(tmpIndex[0]-winSpan):(tmpIndex[-1]+winSpan)]
        firstFrame = sub[timeBase].values[0]
        whiskNorm = sub[myVariable].values[0]
        subst['timeBaseNorm'] = subst[timeBase] - firstFrame
        myVariableNorm = myVariable+'Norm'
        subst[myVariableNorm] = subst[myVariable] - whiskNorm
        # plt.plot(subst['timeBaseNorm'], subst[myVariableNorm], color='red')

        if Myax is not None:
            if sub['geno'].unique()[0] == 'wt':
                ax[Myax].plot(subst[timeBase] - firstFrame, subst[myVariable] - whiskNorm, lw=1, color='grey', alpha = 0.1)
                ax[Myax].plot(sub[timeBase] - firstFrame, sub[myVariable] - whiskNorm, lw=2, color=col[0], alpha = 0.1)
            else:
                ax[Myax].plot(subst[timeBase] - firstFrame, subst[myVariable] - whiskNorm, lw=1, color='grey', alpha = 0.1)
                ax[Myax].plot(sub[timeBase] - firstFrame, sub[myVariable] - whiskNorm, lw=2, color=col[1], alpha = 0.1)

        else:
            if sub['geno'].unique()[0] == 'wt':
                ax[0].plot(subst[timeBase] - firstFrame, subst[myVariable] - whiskNorm, lw=1, color='grey', alpha = 0.1)
                ax[0].plot(sub[timeBase] - firstFrame, sub[myVariable] - whiskNorm, lw=2, color=col[0], alpha = 0.1)
            else:
                ax[1].plot(subst[timeBase] - firstFrame, subst[myVariable] - whiskNorm, lw=1, color='grey', alpha = 0.1)
                ax[1].plot(sub[timeBase] - firstFrame, sub[myVariable] - whiskNorm, lw=2, color=col[1], alpha = 0.1)
        # plt.show()

def getStat(data, measure, groupVar = 'geno', group = ['wt', 'het'], param = True):
    import pingouin as pg
    x = data.loc[data[groupVar] == group[0], measure].astype(float)
    y = data.loc[data[groupVar] == group[1], measure].astype(float) 
    if param == True:
        stat = pg.ttest(x, y)
    else:
        stat = pg.mwu(x, y, alternative='two-sided')
    print(stat)
    if stat['p-val'].item() < 0.001:
        sigMark = '***'
    elif stat['p-val'].item() < 0.01:
        sigMark = '**'
    elif stat['p-val'].item() < 0.05:
        sigMark = '*'
    elif stat['p-val'].item() >= 0.05:
        sigMark = 'na'
    elif np.isnan(stat['p-val'].item()):
        sigMark = 'na'
    stat['sigMark'] = sigMark
    x = stat['p-val'][0]
    x = "{:.4f}".format(x)
    stat['p-val'] = x

    return stat

def createUniqueID(data, namePart1 = 'aid', namePart2 = 'touchGrp'):
    data = data.dropna(subset=[namePart1])
    data['uniqueTchId'] = data[namePart1].astype(int).astype(str) + data[namePart2].astype(int).astype(str)
    return data


##################################
## touche methods ttouchMethods
##################################

def convertFirstFile(i):
    mydir = os.path.dirname(i)
    myfile = os.path.basename(i)
    aid = int(re.findall('[0-9]+', myfile)[0])
    startTime = refFile.loc[refFile['sID']==aid,'stablePole'].item()
    tmpdat = np.load(i)
    tmpdat = tmpdat[0:2, startTime:-1]
    nf = myfile.split('_')
    newfilename = mydir+os.sep+nf[0]+'_p_1_'+nf[-1]
    np.save(newfilename, tmpdat)

def detectArea(j, fileVid):
    # fileVid = [x for x in glob.glob(mainDir + '/*.avi') if j['animalID'] in x][0]
    # print(fileVid)
    outDir = os.path.dirname(os.path.dirname(os.path.dirname(fileVid)))+os.sep+'out_iter3'
    vcap = cv2.VideoCapture(fileVid)
    aid = j['sID'].item()
    aidSave = os.path.basename(fileVid).split('.')[0]
    # print(aid)

    # get the coordinates of the area to measure
    startXp, endXp, startYp, endYp = getCoordsForAnalysis(j['dimensionH x y w h'].item())
    startXs, endXs, startYs, endYs = getCoordsForAnalysis(j['dimensionL x y w h'].item())
    startXerr, endXerr, startYerr, endYerr = getCoordsForAnalysis(j['dimensionOtherSide x y w h'].item())

    meanFramePole=[]
    meanFrameSpace = []
    menFrameErr = []
    vidFMain=[]
    # for vidF in range(int(j['stablePole']), int(vcap.get(7))):
    # for vidF in range(0, 2):
    for vidF in range(0, int(vcap.get(7))):
        # print(vidF)
        vcap.set(1, vidF)
        ret, frame = vcap.read()
        framePole = frame[startYp:endYp, startXp:endXp]
        frameSpace = frame[startYs:endYs, startXs:endXs]
        frameErr = frame[startYerr:endYerr, startXerr:endXerr]
        meanFramePole.append(np.mean(framePole))
        meanFrameSpace.append(np.mean(frameSpace))
        menFrameErr.append(np.mean(frameErr))
        vidFMain.append(vidF)
    vcap.release()
    toSavePole = np.array([vidFMain, meanFramePole])
    toSaveSpace = np.array([vidFMain, meanFrameSpace])
    toSaveErr = np.array([vidFMain, menFrameErr])
    np.save(outDir+os.sep+'pole'+'_'+aidSave+'.npy', toSavePole)
    np.save(outDir+os.sep+ 'space' + '_' +aidSave+ '.npy', toSaveSpace)
    np.save(outDir+os.sep+ 'err' + '_' +aidSave+ '.npy', toSaveErr)

def getCoordsForAnalysis(inputCoord):
    '''
    input: a string of number with coordinate X Y width height '364 141 7 2'
    '''
    areaPole = [float(s) for s in inputCoord.split(' ')]
    startXp, endXp, startYp, endYp = int(areaPole[0]), int(areaPole[0] + areaPole[2]), int(areaPole[1]),  int(areaPole[1] + areaPole[3])
    return startXp, endXp, startYp, endYp

def modify_outlier_value(arr):
    """
    This function modifies the value in an array if it is the only one that is different from its surrounding values.
    
    Arguments:
    arr -- the input array
    
    Returns:
    arr -- the modified array
    """
    for i in range(1, len(arr) - 1):
        if arr[i] != arr[i - 1] and arr[i] != arr[i + 1]:
            arr[i] = arr[i - 1]
    return arr

def thresholdDataSimple(array, threshold , noTouchVal=-1000, touchVal=-1001): #
    """ threshold data return an array of filtered data based on threshold this convert data to binary on/off
    like state. Need to plot the graph first
    Arguments:
        array (list): list of data to be converted and threshold
        stdDevVal (int, optional): integer value that will set the cutoff based
        noTouchVal (int): integer value to categorize the touch category (eg. no touch =0)
        touchVal (int): integer value to categorize the touch category (eg. no touch =1)
        refFile: a reference file with information about the animal to get animal id
        customThreshold: array of 2 values the extra values and the regular values ref.loc[ref['sID'] == aid, ['cutomThreshold', 'errorDetectThreshold']].values[0]

    Return:
        list with the filtered data
    Usage:
        dat['filt1'] = thresholdData(dat['raw1'], 190, 200)
    """

    meanFrame = copy.deepcopy(array[1]) # add to multiply here since the thing as been normalized
    # meanFrame = meanFrame - np.mean(meanFrame)
    meanFrame = np.where(meanFrame <= threshold, touchVal, meanFrame)
    meanFrame = np.where(meanFrame > threshold, noTouchVal, meanFrame)
    meanFrame = modify_outlier_value(meanFrame)

    dat = pd.DataFrame({'frame': array[0], 'filt': meanFrame})
    dat.loc[dat['filt']==-1001, 'filt']=0
    dat.loc[dat['filt']==-1000, 'filt']=1

    return dat


def thresholdData(array, noTouchVal=0, touchVal=1, customThreshold=np.array([-1])):
    """ threshold data return an array of filtered data based on threshold this convert data to binary on/off
    like state. Need to plot the graph first
    Arguments:
        array (list): list of data to be converted and threshold
        stdDevVal (int, optional): integer value that will set the cutoff based
        noTouchVal (int): integer value to categorize the touch category (eg. no touch =0)
        touchVal (int): integer value to categorize the touch category (eg. no touch =1)
        refFile: a reference file with information about the animal to get animal id
        customThreshold: array of 2 values the extra values and the regular values ref.loc[ref['sID'] == aid, ['cutomThreshold', 'errorDetectThreshold']].values[0]

    Return:
        list with the filtered data
    Usage:
        dat['filt1'] = thresholdData(dat['raw1'], 190, 200)
    """
    if customThreshold[0] == -1:
        # print('o1')
        stdDevVal = 2
        threshold = np.mean(array) - np.std(array) * stdDevVal
        meanFrame = copy.deepcopy(array)
        meanFrame = np.where(meanFrame <= threshold, touchVal, meanFrame)
        meanFrame = np.where(meanFrame > threshold, noTouchVal, meanFrame)

    elif customThreshold.size == 2:
        # print('o2')
        threshold = customThreshold
        meanFrame = copy.deepcopy(array)
        meanFrame = np.where(meanFrame < customThreshold[-1], -5, meanFrame)
        meanFrame = np.where((meanFrame > 0) & (meanFrame <= customThreshold[0]), touchVal, meanFrame)
        meanFrame = np.where(meanFrame > customThreshold[0], noTouchVal, meanFrame)

    else:
        # print('o3')
        threshold = customThreshold
        meanFrame = copy.deepcopy(array)
        meanFrame = np.where(meanFrame <= threshold, touchVal, meanFrame)
        meanFrame = np.where(meanFrame > threshold, noTouchVal, meanFrame)

    return list(meanFrame), threshold

def getThreshold(aid, ref, files, window = 1000, stdDevVal = 10):
    """ function to get the threshold the process is the following
    * use the thresholdData()
    * from there use teh QCmultivideoCheckThreshold()
    * visually inspect the trace and get 


    Arguments:
        aid (int): animal id
        ref (DataFame): reference pandas dataFrmae
        refFile: a reference name for export file with information about the animal to get animal id
        window (int): window size in frames around which the baseline can be taken
        files (list): list of all the npy array exported from the detectarea function

    Return:
        save the threshold folder output
    
    Usage:

    refFile = r"Y:\Sheldon\Highspeed\not_analyzed\WDIL007_SyngapKO 12-16-19 cohort high Stim\autoDetectTouches\animals.csv"
    ref = pd.read_csv(refFile)
    files = glob.glob(mainDir+os.sep+'space*_p_[0-9]*_*.npy')
    for i in ref['sID'].unique():
        ref = getThreshold(i, ref, files, stdDevVal = 10)
        # print(ref)
    ref.to_csv(r'Y:\\Sheldon\\Highspeed\\not_analyzed\\WDIL007_SyngapKO 12-16-19 cohort high Stim\\autoDetectTouches\\animals.csv')

    """    


    aidLoc = str(ref.loc[ref['sID'] == aid, 'thresholdpointVideo'].item())+'_'+str(aid)
    workfile = [x for x in files if aidLoc in x][0]
    workfile = np.load(workfile)[1]

    refFrame = ref.loc[ref['sID'] == aid, 'thresholdpointFrame'].item()
    array = workfile[(refFrame-window) : (refFrame+window)]
    threshold = np.mean(array) - np.std(array) * stdDevVal
    errorDetectThreshold = np.mean(array) - 20

    ref.loc[ref['sID'] == aid, 'cutomThresholdMean'] = np.mean(array)
    ref.loc[ref['sID'] == aid, 'cutomThresholdSTD'] = np.std(array)
    ref.loc[ref['sID'] == aid, 'cutomThresholdSTDval'] = stdDevVal

    ref.loc[ref['sID'] == aid, 'cutomThreshold'] = threshold
    ref.loc[ref['sID'] == aid, 'errorDetectThreshold'] = errorDetectThreshold

    return ref

def convertRawtoSummaryTouch(dat, rawDat='filt'):
    """ create group create based on touch status to be able to generate a summary of the data
    to quantify type and length of events
    Arguments:
        dat (pd.DataFrame): pandas data frames that containes frame and type touch not touch binary on/off formant
        see thresholdData
        rawDat (str): string, that contains the column name of interest with pole/no pole touch d
    Return:
        dat (pd.DataFrame): updated pandas dataFrame
        summary (pd.DataFrame): summary data frame based on group
    Usage:
        dat, t = grpSummary(dat=dat, rawDat='dlc')
    Usage (batch):
        coltoChange=['dlc','filt1','filt2','manual']
        for i,j in enumerate(coltoChange):
            dat, t=grpSummary(dat, j)
    """
    dat = dat
    grpDat = rawDat + '_grp'
    if any(x in list(dat.columns) for x in ['frame']) == False:
        dat['frame'] = dat.index
    dat[grpDat] = (dat[rawDat].diff(1) != 0).astype('int').cumsum()  # detect switch to no consecutive values

    dat.loc[dat[rawDat] == min(dat[rawDat]), rawDat] = 0
    dat.loc[dat[rawDat] == max(dat[rawDat]), rawDat] = 1
    summary = pd.DataFrame({'touchCount': dat.groupby([rawDat, grpDat])[rawDat].count(),
                            'FirstTouchFrame': dat.groupby([rawDat, grpDat])['frame'].first()})
    summary = summary.assign(interEventinter=summary['FirstTouchFrame'].shift(-1) - (summary['FirstTouchFrame'] +
                                                                                     summary['touchCount']))

    summary.reset_index(inplace=True)
    return dat, summary
    
def convertRawtoSummary(dat, rawDat='dlc'):
    """ create group create based on touch status to be able to generate a summary of the data
    to quantify type and length of events
    Arguments:
        dat (pd.DataFrame): pandas data frames that containes frame and type touch not touch binary on/off formant
        see thresholdData
        rawDat (str): string, that contains the column name of interest with pole/no pole touch d
    Return:
        dat (pd.DataFrame): updated pandas dataFrame
        summary (pd.DataFrame): summary data frame based on group
    Usage:
        dat, t = grpSummary(dat=dat, rawDat='dlc')
    Usage (batch):
        coltoChange=['dlc','filt1','filt2','manual']
        for i,j in enumerate(coltoChange):
            dat, t=grpSummary(dat, j)
    """
    dat = dat
    grpDat = rawDat + '_grp'
    if any(x in list(dat.columns) for x in ['frame']) == False:
        dat['frame'] = dat.index
    dat[grpDat] = (dat[rawDat].diff(1) != 0).astype('int').cumsum()  # detect switch to no consecutive values

    dat.loc[dat[rawDat] == min(dat[rawDat]), rawDat] = 0
    dat.loc[dat[rawDat] == max(dat[rawDat]), rawDat] = 1
    summary = pd.DataFrame({'touchCount': dat.groupby([rawDat, grpDat])[rawDat].count(),
                            'FirstTouchFrame': dat.groupby([rawDat, grpDat])['frame'].first()})
    summary = summary.assign(interEventinter=summary['FirstTouchFrame'].shift(-1) - (summary['FirstTouchFrame'] +
                                                                                     summary['touchCount']))

    summary.reset_index(inplace=True)
    return dat, summary

def QCmultivideoCheckThreshold(sID, refFile, files, expDir, nvideo = 6, customT = True, customPlot = True):
    """ function to plot QC of touch detection to per animal and evaluate threshold

    Arguments:
        refFile: a reference file with information about the animal to get animal id
        files (list): list of all the npy array exported from the detectarea function
        noTouchVal (int): integer value to categorize the touch category (eg. no touch =0)
        n video (int): the number of video that were acquired per animal

    Return:
        save the threshold folder output
    
    Usage:
        expDir = 'Y:\\Sheldon\\Highspeed\\not_analyzed\\WDIL007_SyngapKO 12-16-19 cohort high Stim\\High Speed Video pole whisking 1-20-20\\out_iter2\\tmpGraph2'
        files = glob.glob(mainDir+os.sep+'*_p_[0-9]*_*.npy')
        QCmultivideoCheckThreshold(refFile, files, expDir, nvideo = 6)


    """
    tmpfiles = [x for x in files if str(sID) in x]
    print(sID)
    tmpfiles.sort()
    if customPlot == True:
        fig, ax = plt.subplots(2,nvideo, sharex='col', sharey='row', figsize=[38, 4])

    for idx in np.arange(1,7):
        # print(idx)
        datStr = 'space_p_'+str(idx)
        errStr = 'err_p_'+str(idx)
        tmpDat = np.load([x for x in tmpfiles if datStr in x][0])
        errtmpDat = np.load([x for x in tmpfiles if errStr in x][0])
        aid = [x for x in tmpfiles if datStr in x][0].split('.')[0].split(os.sep)[-1]

        if customT == True:
            # get the threshold and touch for the data actual touch data
            customThreshold = refFile.loc[refFile['sID'] == sID, ['cutomThresholdMeanSpaceL']].values[0]
            customThreshold = np.append(customThreshold, customThreshold-20)
            thrshDat, thrsh = thresholdData(tmpDat[1], customThreshold=customThreshold)
            # get the threshold and touch for the actual error
            errThreshold = refFile.loc[refFile['sID'] == sID, ['cutomThresholdMeanOut']].values[0]
            errDat, errthrsh = thresholdData(errtmpDat[1], noTouchVal=0, touchVal=-10, customThreshold=errThreshold)
        else:
            thrshDat, thrsh = thresholdData(tmpDat[1])

        dat = pd.DataFrame({'frame': tmpDat[0], 'filt': thrshDat, 'error': errDat})

        #### strart - section to work on the error
        ##########################################
        '''
        this is for when the whisker goes on the other side of the pole
        or when the pole is grabbed by the paw
        '''

        dat['error'] = dat['error']+dat['filt']
        dat.loc[dat['error']>=0, 'error'] = 0
        dat.loc[dat['error']<0, 'error'] = -1
        dat['errorGrp'] = (dat['error'].diff(1) != 0).astype('int').cumsum()

        tstDat = dat.groupby(['errorGrp','error']).agg({'errorGrp':[np.ma.count]})
        tstDat = quickConversion(tstDat)
        # this block out the error in case grabbing or opposite pole touching last more than 500ms aka 250 frames
        tstDat.loc[(tstDat['error']==0) & (tstDat['errorGrpcount']< 250), 'error'] =-1

        newDat = dat[['frame', 'filt', 'errorGrp']]
        tstDat = tstDat[['errorGrp', 'error']].drop_duplicates()
        newDat = pd.merge(tstDat, newDat, on='errorGrp', how='right')
        newDat.loc[newDat['error'] == -1, 'filt'] = -1

        #### end - section to work on the error
        ##########################################

        dat, summary = convertRawtoSummary(newDat, rawDat='filt')
        summary = summary[summary['filt']==1]
        print(expDir+os.sep+aid+'_'+str(idx)+'.csv')
        print(expDir+os.sep+aid+'_'+str(idx)+'touchDat.csv')
        summary.to_csv(expDir+os.sep+aid+'_'+str(idx)+'touchDat.csv')
        dat.to_csv(expDir+os.sep+aid+'_'+str(idx)+'.csv')
        idf = idx-1
        if customPlot == True:
            ax[0][idf].plot(tmpDat[0], tmpDat[1])
            ax[0][idf].plot([0, len(thrshDat)], [thrsh, thrsh])
            ax[1][idf].plot(dat.frame, dat.filt)

    if customPlot == True:
        ax[0][0].set_ylim([244,255])
        plt.suptitle(i)
        plt.tight_layout()
        fig.savefig(expDir+os.sep+aid+'.jpg')

        plt.close('all')

def touchCorrectionShortInterval(filename):
    #### start - to deal with short interval between touches
    #######################################################
    '''
    this section is to group events which are smiliar

    '''
        # for i,j in enumerate(manual['touchEventGrp']):
    #     if manual['Interval'][i] < 0.05:
    #         manual['touchEventGrp'][i+1] = manual['touchEventGrp'][i]
    #### end - to deal with short interval between touches
    #######################################################
    tt = pd.read_csv(filename)
    print(filename)
    # section to deal with consecutive miss
    groupingInt = 2 # grouping interval value 
    #### criteria to decide what intervals should be grouped together
    tt.loc[tt['interEventinter']<=groupingInt, 'interEventinter'] = 1

    #### match the value of the preceeding events for classification
    #### careful with the startegy above it should only apply to value that are consecutive and within the groupingInt
    #### see if statement in the for loop
    tt['consecutiveMiss'] = (tt['interEventinter'].diff(1) != 0).astype('int').cumsum()

    for i in tt['consecutiveMiss'].unique():
        # print(i)
        if tt.loc[(tt['consecutiveMiss'] == i), 'interEventinter'].values[0]<=groupingInt:
            newVal = tt.loc[tt['consecutiveMiss'] == i, 'filt_grp'].iloc[0]
            tt.loc[tt['consecutiveMiss'] == i, 'filt_grp'] = newVal
        #### this retrieve the index and will fetch the next index of the series

            toChange = tt.loc[(tt['consecutiveMiss'] == i) & (tt['interEventinter']==1)]
            if not toChange.empty:
                a = toChange.iloc[-1].name
                tt.loc[a+1, 'filt_grp'] = tt.loc[a, 'filt_grp']

    #### recreate a summary of the data
    alpha = tt.groupby(['filt_grp']).agg({'filt': ['first'], 'touchCount': [np.sum], 'FirstTouchFrame': ['first'], 'interEventinter': ['last']})
    alpha = quickConversion(alpha)
    alpha = alpha.rename(columns={'filtfirst': 'filt', 'touchCountsum': 'touchCount', 'FirstTouchFramefirst': 'FirstTouchFrame', 'interEventinterlast': 'interEventinter'})

    saveName = filename.split('.')[0]+'corrected.csv'
    alpha.to_csv(saveName)

    return alpha

def loadTheFiles():
    if platform.system() == 'Linux':
        prefixPath = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab' 
    else:
        prefixPath = 'Y:/'

    touchFiles = os.path.join(prefixPath, r'Sheldon\Highspeed\not_analyzed\WDIL007_SyngapKO 12-16-19 cohort high Stim\High Speed Video pole whisking 1-20-20\out_iter3\tmpGraph')
    touchFiles = glob.glob(touchFiles+os.sep+'*touchDat.csv')
    expDir = os.path.join(prefixPath, 'Sheldon\\Highspeed\\not_analyzed\\WDIL007_SyngapKO 12-16-19 cohort high Stim\\High Speed Video pole whisking 1-20-20\\out_iter3\\tmpGraph')
    # os.makedirs(expDir)
    mainDir = os.path.join(prefixPath, r'Sheldon\Highspeed\not_analyzed\WDIL007_SyngapKO 12-16-19 cohort high Stim\High Speed Video pole whisking 1-20-20\out_iter3')
    mainVidDir = os.path.join(prefixPath, 'Sheldon/Highspeed/not_analyzed/WDIL007_SyngapKO 12-16-19 cohort high Stim/High Speed Video pole whisking 1-20-20')
    files = glob.glob(mainDir+os.sep+'*_p_[0-9]*_*.npy')
    ref = pd.read_csv(os.path.join(prefixPath,'Sheldon/Highspeed/not_analyzed/WDIL007_SyngapKO 12-16-19 cohort high Stim/autoDetectTouches/animals_iter3.csv'))
    idlist = ref.sID.unique()

    vidFiles = glob.glob(mainVidDir + '/**/*p*.avi', recursive=True)
    vidFiles = [x for x in vidFiles if 'raw' not in x]

    return expDir, mainDir, files, ref, ref, idlist, touchFiles, vidFiles

def parallel_QCmultivideoCheckThreshold(sID):
    QCmultivideoCheckThreshold(sID, ref, files, expDir, customT = True, customPlot = False)


def parallel_DetectArea(filename):

    # mainDir = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/Sheldon/Highspeed/FMR1 High Speed Whisker'
    # refFile = mainDir+'/FMR1toolsForDetect/FMR1.csv'

    refFile = pd.read_csv('/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/Sheldon/Highspeed/not_analyzed/WDIL007_SyngapKO 12-16-19 cohort high Stim/autoDetectTouches/animals_iter3.csv')
    aid = os.path.basename(filename)[-8:-4]
    # aid = filename.split(os.sep)[-4]
    j = refFile[refFile['sID']==int(aid)]
    print(j)
    print(filename)
    detectArea(j, filename)

def convertSolomon(filename):
    '''
    this function enable the conversion of csv file generated with Solomon coder to touch file that can be converted with the function raw to summary
    '''

    dat = pd.read_csv(filename)
    # clean up the data 
    dat.index.name = 'frame'
    dat = dat.reset_index()
    # there is a drift in solomonn coder of 1 frame which is corrected here:
    dat = dat[1:]
    dat['frame'] = dat['frame']-1
    dat = dat.rename(columns={'Whisking duration': 'touch'})
    dat = dat[['frame', 'touch']]

    # correct the values
    dat['touch'] = dat['touch'].fillna(0)
    dat['touch'] = dat['touch'].replace(2,1)

    return dat

def combineData(animal, files):

    datAll = []
    for aid in animal['sID'].unique():
        print(aid)
        seri = [x for x in files if str(aid) in x]
        seri.sort()
        for idx, i in enumerate(seri):
            # print(idx, i)
            bidx = idx+1
            dat = pd.read_csv(i)
            dat['sID'] = aid
            dat['order'] = bidx
            startTime = refFile.loc[refFile['sID']==aid,'stablePole'].item()
            if bidx > 1:
                constant = 24999 - startTime
                # print(constant, constant*idx)
                dat['FirstTouchFrame'] = dat['FirstTouchFrame'] + constant * idx
            datAll.append(dat)

    datAll = pd.concat(datAll)
    return datAll

##################################
## NORT/NORx
##################################

def discrimIdx(dat):
    '''
    calculate the discrimination index for NORT/NOR
    dat: pandas data frame to add column with discrimination index
    '''
    dat['discrimIdx'] = (dat['T2'] - dat['T1'])/(dat['T2'] + dat['T1'])
    return dat

def appFig(pdStat, panelABS, datName, datFilter, cat1, cat2, package = 'pingouin'):
    '''
    function to create master tabel for figure stat with 
    pdStat: pandas data frame coming from pingouin packages for stat or custom function for stat
    panelABS: correspond to the absolute panel number for figure 

    cat2: 'Discrimination', 'Exploration'

    '''

    mainF = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\mainFig.csv")
    mainF = mainF.loc[:, mainF.columns != 'Unnamed: 0']

    pdStat['figPannelABS'] = panelABS
    pdStat['dataFile'] = datName
    pdStat['dataFilter'] = datFilter
    pdStat['package'] = package
    pdStat['cat1'] = cat1
    pdStat['cat2'] = cat2

    df = pd.concat([mainF,pdStat])
    df.to_csv(r"Y:\2020-09-Paper-ReactiveTouch\mainFig.csv", index=False)


    return df

def discrimStat(dat, alt = False):
    '''
    for discrimnation index output the 3 statistics
    '''

    statall = []
    if alt == True:
        varA = ['whisker', 'noWhisker']
        for idx, varAn in enumerate(varA):
            print(varAn)
            a = pg.ttest(dat.loc[(dat['expSubset'] == varAn), 'discrimIdx'].values, 0)
            a['statType'] = 'one-sample'
            a['geno'] =  varAn
            statall.append(a)
        a = pg.ttest(dat.loc[(dat['expSubset'] == varA[0]), 'discrimIdx'].values,
                     dat.loc[(dat['expSubset'] == varA[1]), 'discrimIdx'].values)
        statall.append(a)
        statall = pd.concat(statall)

    else:
        varA = ['wt', 'het']
        for idx, varAn in enumerate(varA):
            print(varAn)
            a = pg.ttest(dat.loc[(dat['geno'] == varAn), 'discrimIdx'].values, 0)
            a['statType'] = 'one-sample'
            a['geno'] =  varAn
            statall.append(a)
        a = pg.ttest(dat.loc[(dat['geno'] == varA[0]), 'discrimIdx'].values,
                     dat.loc[(dat['geno'] == varA[1]), 'discrimIdx'].values)
        statall.append(a)
        statall = pd.concat(statall)

    return statall

def expltimeStat(data, alt = False):
    statall = []
    if alt == True:
        varA = ['whisker', 'noWhisker']
        for idx, varA in enumerate(varA):
            a = pg.ttest(data.loc[(data['expSubset'] == varA) & (data['variable'] == 'T1'), 'value'].values,
                     data.loc[(data['expSubset'] == varA) & (data['variable'] == 'T2'), 'value'].values)
            a['statType'] = 'T1vsT2'
            a['geno'] =  varA
            statall.append(a)
        statall = pd.concat(statall)

    else:
        varA = ['wt', 'het']
        for idx, varA in enumerate(varA):
            a = pg.ttest(data.loc[(data['geno'] == varA) & (data['variable'] == 'T1'), 'value'].values,
                     data.loc[(data['geno'] == varA) & (data['variable'] == 'T2'), 'value'].values)
            a['statType'] = 'T1vsT2'
            a['geno'] =  varA
            statall.append(a)
        statall = pd.concat(statall)

    return statall

def aovPH_2way(data, dv='discrimIdx', within='object', between='geno', subject='sID'):
    # https://raphaelvallat.com/pingouin.html
    aov = pg.mixed_anova(dv=dv, within=within, between=between, subject=subject, data=data)

    aov = pg.mixed_anova(dv=dv, within=within, between=between, data=data)
    aov['statType'] = '2way-aov'
    posthocs = pg.pairwise_ttests(dv=dv, within=within, between=between, subject=subject, data=data, padjust='holm')
    posthocs['statType'] = '2way-aov-posthocs'
    # pg.print_table(posthocs)

    df = pd.concat([aov,posthocs])
    return df


## ***************************************************************************
## * CUSTOM FUNCTION TO CHECK FILE USAGE                                                      *
## ***************************************************************************

def outputPresent(x):
    import sys
    if sys.platform == 'win32':
        fdrive = 'Y:/'
    elif sys.platform =='linux':
        fdrive = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/'
    
    mainPath = fdrive+r'Jessie\e3 - Data Analysis\e3 Data'
    # confirm if the path exist
    path=glob.glob(mainPath+os.sep+x+os.sep+'output')
    if path == []:
        pout = 'empty'
    else:
        pout = 'output'
        plen = str(len(glob.glob(path[0]+os.sep+'*')))

        pout = pout + ' ' + plen

    return pout

def videoPresent(x):
    import sys
    if sys.platform == 'win32':
        fdrive = 'Y:/'
    elif sys.platform =='linux':
        fdrive = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/'

    x = x.split(' ')[0]
    mainPath = fdrive+r'Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline'
    path = glob.glob(mainPath+'/**/*'+x+'*.mp4', recursive=True)
    if path == []:
        pout = 'empty'
    else:
        pout = os.sep.join(path[0].split(os.sep)[-2:])
        # plen = str(len(glob.glob(path[0]+os.sep+'*')))

        # pout = pout + ' ' + plen


    return pout

## ***************************************************************************
## * CUSTOM FOR POLE TOUCH EXTRACTION                                           *
## ***************************************************************************


######## process1
class correspTouchDatStream:
    def __init__(self, aid, polePres=1):
        self.aid = aid
        self.polePres = polePres
        vidPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline'
        self.vidFile = glob.glob(vidPath+'/**/*'+aid+'*.mp4', recursive=True)[0]
        datPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'
        if glob.glob(datPath+os.sep+aid+os.sep+'output/ReferenceTableTransition.csv') == []:
            self.timeFile = []
        else:
            self.timeFile = glob.glob(datPath+os.sep+aid+os.sep+'output/ReferenceTableTransition.csv')[0]
        refFile = pd.read_csv(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p"+str(self.polePres)+".csv")
        self.cropAreas = refFile[refFile['Record_folder'] == aid]

    def timeForVid(self):
        srE = 25000 # sampling rate of the ephys
        srHS = 500 # sampling rate of the Highspeed camera
        if self.timeFile == []:
            times = np.array([588000, 948000])
            return times

        else:
            dat = pd.read_csv(self.timeFile)
            times = (dat.loc[dat['Row'].isin(['LEDon2', 'LEDoff2']),'timeSlot'].values/srE).astype(int) # presented in seconds
            # those times are the center point on the pole presentation
            # centered during the pole presentation which is lasting 10 min 
            padInterval = 6*60 # 6 minutes x 60 seconds to have the unit in seconds
            times = [[times[0]-padInterval,times[0]+padInterval], [times[1]-padInterval,times[1]+padInterval]]
            times = np.array(times)*srHS # convert back to frame

            if self.polePres == 1:
                times = times[0]
            elif self.polePres == 2:
                times = times[1]

            return times

    def detectArk1eaDf(self):
        # fileVid = [x for x in glob.glob(mainDir + '/*.avi') if j['animalID'] in x][0]
        # print(fileVid)
        mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2'
        # polePosition = fileVid.split(os.sep)[-2]
        outDir = mainDir+os.sep+'autoDetectTouches-TEST'
        os.makedirs(outDir, exist_ok=True)
        vcap = cv2.VideoCapture(self.vidFile)

        # aid = j['sID'].item()
        aidSave = os.path.basename(self.vidFile).split('.')[0]
        # print(aid)

        # get the coordinates of the area to measure
        startXp, endXp, startYp, endYp = getCoordsForAnalysis(self.cropAreas['dimensionH x y w h'].item()) # dimension for the horizontal
        startXs, endXs, startYs, endYs = getCoordsForAnalysis(self.cropAreas['dimensionL x y w h'].item()) # dimension for the vertical
        startXerr, endXerr, startYerr, endYerr = getCoordsForAnalysis(self.cropAreas['dimensionOtherSide x y w h'].item()) # dimension for the error area vertical
        startXLED, endXLED, startYLED, endYLED = getCoordsForAnalysis(self.cropAreas['dimensionLED x y w h'].item()) # dimension for the LED area

        meanFramePole=[]
        meanFrameSpace = []
        meanFrameErr = []
        meanFrameLED = []
        vidFMain=[]

        # reference frame
        # here the reference frame is established at 0 but need to confirm for each video that the refrence frame is ok 


        vidrefImage = self.cropAreas['refImage']
        vcap.set(1, vidrefImage)
        ret, refframe = vcap.read()
        refframe = cv2.cvtColor(refframe, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
        refframe = refframe.astype(np.int64)

        times = self.timeForVid()
        frames = [*range(times[0], times[1])]
        # for vidF in range(int(j['stablePole']), int(vcap.get(7))):
        # for vidF in range(882387-500, 882387+500):
        for vidF in frames:
        # for vidF in range(0, 3):
        # for vidF in range(0, int(vcap.get(7))):
            # print(vidF)
            
            # rolling frames
            vcap.set(1, vidF)
            ret, frame = vcap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
            frame = frame.astype(np.int64)

            # substract the signal to the reference frame and extract the mean value per row or column depending on the relevance
            # when the detection is horizontal signal per row (axis=0) otherwise per column (axis = 1)
            framePole = abs(frame[startYp:endYp, startXp:endXp] - refframe[startYp:endYp, startXp:endXp]).mean(axis = 1)
            frameSpace = abs(frame[startYs:endYs, startXs:endXs] - refframe[startYs:endYs, startXs:endXs]).mean(axis = 0)
            frameErr = abs(frame[startYerr:endYerr, startXerr:endXerr] - refframe[startYerr:endYerr, startXerr:endXerr]).mean(axis = 0)
            frameLED = abs(frame[startYLED:endYLED, startXLED:endXLED] - refframe[startYLED:endYLED, startXLED:endXLED]).mean(axis = 0)

            # get the heigest pixel value
            # this is specialy relevent in the case of the framePole with vertical detection where the whisker crossing the beam
            # will occupy may be around 6 pixels
            def nelem(elem):
                nelem = int(len(elem)*0.15)
                return nelem
            framePole = framePole[(-framePole).argsort()[:nelem(framePole)]].mean()
            frameSpace = frameSpace[(-frameSpace).argsort()[:nelem(frameSpace)]].mean()
            frameErr = frameErr[(-frameErr).argsort()[:nelem(frameErr)]].mean()
            frameLED = frameLED[(-frameLED).argsort()[:nelem(frameLED)]].mean()

            meanFramePole.append(framePole)
            meanFrameSpace.append(frameSpace)
            meanFrameErr.append(frameErr)
            meanFrameLED.append(frameLED)
            vidFMain.append(vidF)
        vcap.release()
        toSavePole = np.array([vidFMain, meanFramePole])
        toSaveSpace = np.array([vidFMain, meanFrameSpace])
        toSaveErr = np.array([vidFMain, meanFrameErr])
        toSaveLED = np.array([vidFMain, meanFrameLED])
        np.save(outDir+os.sep+'pole'+'_'+aidSave+'_pres'+str(self.polePres)+'.npy', toSavePole)
        np.save(outDir+os.sep+ 'space' + '_' +aidSave+'_pres'+str(self.polePres)+ '.npy', toSaveSpace)
        np.save(outDir+os.sep+ 'err' + '_' +aidSave+'_pres'+str(self.polePres)+ '.npy', toSaveErr)
        np.save(outDir+os.sep+ 'LED' + '_' +aidSave+'_pres'+str(self.polePres)+ '.npy', toSaveLED)

######## process3
def getID(allfiles):
   '''
   get the id number from a file list
   '''
   # get the base name first
   filesbasename = [x.split(os.sep)[-1] for x in allfiles]
   uniqueid = np.unique([x.split('_')[1] for x in filesbasename])

   return uniqueid

def getDataFromGraph_batch(keyfiles, myref, param1 , param2):
    '''
    param1: correpsond to the detection type either 'detection' or 'error'
    param2: correspond to the pole presentation epoch either the first or second one input : 1 or 2
    '''

    datAll = []
    
    for i in keyfiles:
        try:
            # change input here if pole pres and detection different
            # t = graphManualThreshold(filename, ref =  r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p1.csv', dataType = 'detection', e3dat = True, polePres=1)
            t = graphManualThreshold(i, ref = myref, dataType = param1, e3dat = True, polePres=param2)
            dat, tmp = t.getDataFromGraph()
            print(t.getInfo())
            datAll.append(dat)
        except:
            logging.info('fail: ', i)

    datAll = pd.concat(datAll)
    saveName = os.path.dirname(os.path.dirname(keyfiles[0]))+os.sep+'manualThreshold_'+param1+'_'+str(param2)+'.csv'
    datAll.to_csv(saveName)
    print('DONE !! Data saved here: ', saveName)

    return datAll

class Cursor:
    """
    A cross hair cursor. to register clicked point to data
    see doc 
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            # update the line positions
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()
    
    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        if event.dblclick:
            plt.close()
        global coords
        coords = []
        coords.append(event.ydata)

###########
class graphManualThreshold:
    '''
    The process here is: 
    1. get all the initial parameters
    2. find the limit of the area of interest to be analyzed downstream
    3. plot the area of interest and establish the threshold
    4. save the threshold and detrended area of interest for analysis downstream
    '''

    def __init__(self, filename, ref = [], dataType = 'detection', e3dat = True, polePres=1):
        '''
         need to define the 
         e3Dat: are for the data from the e3
         polePres (1 or 2): correspond to 1 or 2nd presentation of pole during the task when ePhys is recorded options are either 1 or 2
        '''
        self.ref = ref # file path for selef ref r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\captrue_forTouch\animals_close_iter_wThresh.csv"
        self.e3dat = e3dat
        self.mainDir =  os.path.dirname(filename)
        self.dataType = dataType
        self.polePres = polePres
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.position  = filename.split(os.sep)[-2]
        self.sID = self.filename.split(os.sep)[-1].split('_')[1]#'_'.join(self.filename.split(os.sep)[-1].split('_')[1:4])

        posList = ['close', 'middle', 'far']
        if any(self.position in x for x in posList) == False:
            self.position = ''
            self.err = np.load(glob.glob(self.mainDir+os.sep+'*err*'+self.sID+'*.npy')[0])
            self.led = np.load(glob.glob(self.mainDir+os.sep+'*LED*'+self.sID+'*.npy')[0])
            self.horizDetect = np.load(glob.glob(self.mainDir+os.sep+'*pole*'+self.sID+'*.npy')[0])
            self.vertDetect = -np.load(glob.glob(self.mainDir+os.sep+'*space*'+self.sID+'*.npy')[0])

            self.trim_vertDetect = np.load(glob.glob(self.mainDir+os.sep+'*Update-w-offset_detection*'+self.sID+'*poleP*'+str(self.polePres)+'*.npy')[0])
            self.trim_err = np.load(glob.glob(self.mainDir+os.sep+'*Update-w-offset_err*'+self.sID+'*poleP*'+str(self.polePres)+'*.npy')[0])

        else:
            ''' this section is to link all the other npy array
            which are associated to the main numpy are being by default the 
            space vertical detect out put '''
            self.err = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*err*'+self.sID+'*.npy')[0])
            self.led = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*LED*'+self.sID+'*.npy')[0])
            self.horizDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*space*'+self.sID+'*.npy')[0])
            # note most of the detection is performed on the pole space
            self.vertDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*pole*'+self.sID+'*.npy')[0])

            self.trim_vertDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*Update-w-offset_detection*'+self.sID+'*.npy')[0])
            self.trim_err = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*Update-w-offset_err*'+self.sID+'*.npy')[0])

    def getmanThreshold(self):
        if self.ref != []:
            ref = pd.read_csv(self.ref)
            if self.position == '':
                ref['position'] = ''
            ref = ref.loc[(ref['sID']==self.sID) & (ref['position']==self.position), ['cutomThresholdMeanSpaceL', 'cutomThresholdMeanOut']].reset_index(drop=True)
        else:
            print('input a reference files containing threshold')    

        return ref


    def getInfo(self):
        ''' get info from file name
        '''
        dat = pd.DataFrame({'position':[self.position], 'sID':[self.sID]})

        return dat

    def getThelimit(self):

        ''' function where the limit of the array should be kept for analysis downstream filtering etc'''
        # for pole touches subsample the data
        # get the limit for pole in and pole out
        try:
            tmpErr = np.where(np.diff(self.err[1][::10])<-5)[0]*10
            limcutoffx = len(self.err[1])/2
            lowlimErr = tmpErr[tmpErr<limcutoffx][-1]+250
            highlimErr = tmpErr[tmpErr>limcutoffx][0]-250
            errArray = [lowlimErr, highlimErr]
            # plt.plot(err[1])
            # plt.axvline(lowlimErr, color='orange')
            # plt.axvline(highlimErr, color='orange')

            # get the limit from the led file
            tmpLED = self.led[1]
            limcutoffx = len(tmpLED)/2
            limcutoffy = min(tmpLED)+(max(tmpLED)-min(tmpLED))/2
            tmp = np.where(tmpLED>limcutoffy)[0]
            lowlimLED = tmp[tmp<limcutoffx][-1]+250
            highlimLED = tmp[tmp<len(tmpLED)][-1]-250
            ledArray = [lowlimLED, highlimLED]
            # plt.plot(tmpLED)
            # plt.axvline(lowlimErr, color='orange')
            # plt.axvline(highlimErr, color='orange')

            if ledArray[-1]-ledArray[0] < errArray[-1]-errArray[0]:
                limitKept = errArray
            else:
                limitKept = ledArray
        except:
            limitKept = [0,len(self.err[1])]

        return limitKept

    def nonLinearDetrend(self):
        """ function to perform polynomial detrending (default order is 20 - can make it more flex)
        this function has been replaced by scipy.signal.detrend and is not use in the class
        """
        limit, trim_vertDetect = self.getThelimit()
        t = range(0, len(trim_vertDetect))
        # polynomial fit (returns coefficients)
        p = scipy.polyfit(t,trim_vertDetect,50) #  20 could run Bayes information criterion to get best polynomial order
        # predicted data is evaluation of polynomial
        yHat = scipy.polyval(p,t)
        # compute residual (the cleaned signal)
        residual = trim_vertDetect - yHat
        return residual

    def e3Limit(self):
        a = self.led
        ididx = np.where(a[1]>160)[0] # 160 seems to be a general threshold for the image coming on and of
        ididx = [ididx[0], ididx[-1]]
        idtime = a[0][ididx]

        minInter = 4 # minute interval surrounding the led signal 
        fs = 60*500 # acquisition rate of the camer
        finter = fs * minInter

        interval = [[idtime[0]-finter, idtime[0]+finter], [idtime[1]-finter, idtime[1]+finter]]
        interval = np.where(np.isin(a[0], interval))[0]

        if self.polePres == 1:
            return interval[:2]# limit index
        else:
            return interval[2:]


    def getDataFromGraph(self):
        plt.ioff()
        # vertDetect = self.nonLinearDetrend() ## this is off to perform this without linear detrending
        

        if self.e3dat == False:
            limit = self.getThelimit()

            if self.dataType == 'detection':
                vertDetect = self.vertDetect[1][limit[0]:limit[1]]
                vertDetect = vertDetect-vertDetect.mean()

            else:
                vertDetect = self.err[1][limit[0]:limit[1]]
                vertDetect = vertDetect-vertDetect.mean()
        else:
            limit = self.e3Limit()

            if self.dataType == 'detection':
                vertDetect = self.vertDetect[1][limit[0]:limit[1]]
                vertDetect = vertDetect-vertDetect.mean()

            else:
                vertDetect = self.err[1][limit[0]:limit[1]]
                vertDetect = vertDetect-vertDetect.mean()
        
        ## this section is to choose if detrending should be applied or not
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(vertDetect, alpha=0.5, label='raw')
        ax.plot(signal.detrend(vertDetect)-5, alpha=0.5, label='detrend')
        admed = np.median(vertDetect)
        ax.set_ylim([admed-15,admed+5])
        # ax.set_xlim([10000, 50000])
        ax.set_title(self.position+'//'+self.sID)
        ax.legend()
        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show()

        ## change from input to msvrt to avoid having to press enter
        # yesDetrend = input('Should detrending be applied (y/n): ')

        print('Should detrending be applied (y/n): ')
        yesDetrend = msvcrt.getch()

        if yesDetrend == b'y':
            print('detrending')
            vertDetect = signal.detrend(vertDetect)
        else:
            print('NO detrending applied')
            vertDetect = vertDetect-vertDetect.mean()

        ## this section is to get the data from the graph
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(vertDetect)
        admed = np.median(vertDetect)
        ax.set_ylim([admed-5,admed+5])
        # ax.set_xlim([10000, 50000])
        ax.set_title(self.position+'//'+self.sID)
        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)

        plt.show()
        dat = self.getInfo()
        if self.dataType == 'detection':
            dat['cutomThresholdMeanSpaceL'] = coords # coords comes from the cursor object
        else:
            dat['cutomThresholdMeanOut'] = coords # coords comes from the cursor object
        dat['polePres'] = self.polePres

        updateSaveName = self.dirname+os.sep+'Update-w-offset_'+self.dataType+'_'+self.sID+'_'+str(limit[0])+'_poleP_'+str(self.polePres)+'.npy'
        print(updateSaveName)
        np.save(updateSaveName, vertDetect)
        return dat, vertDetect

    def correctionForError(self):      

        if self.e3dat == True:
            limit = self.e3Limit()
            datDetect = thresholdData(array = self.trim_vertDetect, customThreshold = self.getmanThreshold()['cutomThresholdMeanSpaceL'].item(), timeFrame = self.vertDetect[0][limit[0]:limit[1]])
            datErr = thresholdData(array = self.trim_err, customThreshold = self.getmanThreshold()['cutomThresholdMeanOut'].item(), timeFrame = self.vertDetect[0][limit[0]:limit[1]])
        else:
            limit = self.getThelimit()
            datDetect = thresholdData(array = self.trim_vertDetect, customThreshold = self.getmanThreshold()['cutomThresholdMeanSpaceL'].item(), timeFrame = self.vertDetect[0][limit[0]:limit[1]])
            datErr = thresholdData(array = self.trim_err, customThreshold = self.getmanThreshold()['cutomThresholdMeanOut'].item(), timeFrame = self.vertDetect[0][limit[0]:limit[1]])

        datErr = datErr.rename(columns={'filt': 'error'})
        dat = datDetect.merge(datErr, on='frame')

        #### strart - section to work on the error
        ##########################################
        '''
        this is for when the whisker goes on the other side of the pole
        or when the pole is grabbed by the paw
        '''

        dat['error'] = dat['error']+dat['filt']
        dat.loc[dat['error']>=0, 'error'] = 0
        dat.loc[dat['error']<0, 'error'] = -1
        dat['errorGrp'] = (dat['error'].diff(1) != 0).astype('int').cumsum()

        tstDat = dat.groupby(['errorGrp','error']).agg({'errorGrp':[np.ma.count]})
        tstDat = quickConversion(tstDat)
        # this block out the error in case grabbing or opposite pole touching last more than 500ms aka 250 frames
        tstDat.loc[(tstDat['error']==0) & (tstDat['errorGrpcount']< 250), 'error'] =-1

        newDat = dat[['frame', 'filt', 'errorGrp']]
        tstDat = tstDat[['errorGrp', 'error']].drop_duplicates()
        newDat = pd.merge(tstDat, newDat, on='errorGrp', how='right')
        newDat.loc[newDat['error'] == -1, 'filt'] = -1

        #### end - section to work on the error
        ##########################################

        dat, summary = convertRawtoSummary(newDat, rawDat='filt')
        summary = summary[summary['filt']==1]

        touchSaveName = self.dirname+os.sep+'touchDAT_'+self.dataType+'_'+self.sID+'.csv'
        print(touchSaveName)
        summary.to_csv(touchSaveName)

        return dat, summary

        # return summary

def comboREF(polePres):
    refName = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p'+str(polePres)+'.csv'
    refNameUpdate = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p'+str(polePres)+'_update.csv'
    ref = pd.read_csv(refName)
    tmp = ref['Record_folder'].str.split('_', expand=True)
    ref['sID'] = tmp[0]

    refout = glob.glob(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2'+os.sep+'*'+str(polePres)+'*.csv')
    ref1 = pd.read_csv(refout[0]).merge(pd.read_csv(refout[1]), on=['sID', 'polePres', 'position', 'Unnamed: 0'])
    ref1 = ref1.drop('Unnamed: 0', 1)
    ref = ref1.merge(ref, on=['sID', 'polePres', 'position'])
    ref.to_csv(refNameUpdate)
    return print(refNameUpdate)

## ***************************************************************************
## * WDIL PLOT                                                               *
## ***************************************************************************


def pullbackGraph(tmpDat, saveDest = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'):

    flist = glob.glob(saveDest+os.sep+'*.pdf')
    flistn = [int(x.split(os.sep)[-1].split('_')[0].split('#TV')[-1]) for x in flist]
    flistind = '{:0>4}'.format(max(flistn)+1)
    description = 'wdil_pullback'
    saveName = saveDest+os.sep+'#TV'+str(flistind)+'_'+description+'.pdf'

    fig, ax = plt.subplots()

    fig.get_size_inches()
    mpl.rcParams['figure.figsize'] = [1.77, 1.52]
    mpl.rcParams['xtick.major.size'] =  2
    # sns.lineplot(
    #     data=tmpDat,
    #     x="deflection_um", y="value", hue="geno", style = 'geno', marker='o', dashes=False, units = 'id', estimator=None, lw=1, alpha = 0.3, markersize=3)

    sns.lineplot(
        data=tmpDat,
        x="deflection_um", y="value", hue="geno", style = 'geno', ci = 68,
         marker='.', markersize=8, markeredgewidth = 0.4, dashes=False)

    plt.title(tmpDat['experiment'].values[0])   
    plt.xlim(tmpDat['deflection_um'].max()*1, tmpDat['deflection_um'].min()*0)

    if tmpDat['measure'].values[0] == "d'":
        plt.ylabel("d'")
        plt.ylim([0,2.5])
    else:
        plt.ylim([0,1])
        plt.ylabel(tmpDat['measure'].values[0]+' rate')

    plt.locator_params(axis = 'y', nbins = 4) 
    plt.locator_params(axis = 'x', nbins = 5) 
    plt.tight_layout()
    ax.legend_.remove()
    # plt.xlabel('Deflection amp. (\u03BCm)') #unicode
    plt.xlabel('Angular velocity (°/s)') #unicode

    description = tmpDat['experiment'].values[0] + '_' + tmpDat['step'].values[0] + '_' +  tmpDat['measure'].values[0] 
    saveName = saveDest+os.sep+'#TV'+str(flistind)+'_'+description+'.pdf'

    plt.savefig(saveName)
    plt.close('all')

def wdilBarGraph(tmpDat, saveDest = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'):

    flist = glob.glob(saveDest+os.sep+'*.pdf')
    flistn = [int(x.split(os.sep)[-1].split('_')[0].split('#TV')[-1]) for x in flist]
    flistind = '{:0>4}'.format(max(flistn)+1)
    description = 'wdil_bar'
    saveName = saveDest+os.sep+'#TV'+str(flistind)+'_'+description+'.pdf'

    fig, ax = plt.subplots()

    fig.get_size_inches()
    mpl.rcParams['figure.figsize'] = [1.77, 1.52]
    mpl.rcParams['xtick.major.size'] =  2
    # sns.lineplot(
    #     data=tmpDat,
    #     x="deflection_um", y="value", hue="geno", style = 'geno', marker='o', dashes=False, units = 'id', estimator=None, lw=1, alpha = 0.3, markersize=3)

    sns.lineplot(
        data=tmpDat,
        x="deflection_um", y="value", hue="geno", style = 'geno', ci = 68,
         marker='.', markersize=8, markeredgewidth = 0.4, dashes=False)

    plt.title(tmpDat['experiment'].values[0])
    plt.xlim(data['deflection_um'].max()*1.1, data['deflection_um'].min()*0)

    if tmpDat['measure'].values[0] == "d'":
        plt.ylabel("d'")
        plt.ylim([0,2.5])
    else:
        plt.ylim([0,1])
        plt.ylabel(tmpDat['measure'].values[0]+' rate')

    plt.locator_params(axis = 'y', nbins = 4) 
    plt.locator_params(axis = 'x', nbins = 5) 
    plt.tight_layout()
    ax.legend_.remove()
    plt.xlabel('Deflection amp. (\u03BCm)') #unicode

    description = tmpDat['experiment'].values[0] + '_' + tmpDat['step'].values[0] + '_' +  tmpDat['measure'].values[0] 
    saveName = saveDest+os.sep+'#TV'+str(flistind)+'_'+description+'.pdf'

    plt.savefig(saveName)
    plt.close('all')

def makemetalist(start, stop):
    stop = stop+1
    llist = list(np.arange(start,stop))
    return print(['#TV'+'{:0>4}'.format(x) for x in llist])


## ***************************************************************************
## * Cluster Analysis Functions                                              *
## ***************************************************************************
def tpath(mypath, shareDrive = 'Y'):
    '''
    path conversion to switch form linux to windows platform with define drive
    Args:
    mypath (str): path of the file of interest
    shareDrive (str): windows letter of the shared folder
    '''
    # if ('google.colab' in str(get_ipython())) or sys.platform == 'win32':
    if sys.platform == 'win32':
         myRoot = shareDrive+':'      
    else:
        myRoot = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab'


    newpath = myRoot+os.sep+mypath
    newpath = os.path.normpath(newpath)

    return newpath

def getAllZETA(zetaType = 'ZETA'):
    '''
    zetaType (str): correpsond to the type of the zeta if extracted directly from the spikes
                    'ZETA' (default): directly extracted from the spikes with KS 
                    'ZETA_thresh': extracted from the threshold crossing per channel
    '''

    allZETA = glob.glob(tpath('Jessie/e3 - Data Analysis/e3 Data/**/output/**/'+zetaType+'.csv'))

    zetaCombo = []
    for i in allZETA:
        print(i)
        tmp = pd.read_csv(i)
        zetaCombo.append(tmp)
    zetaCombo = pd.concat(zetaCombo)



    geno = tpath('Vaissiere/fileDescriptor2.csv')
    geno = getGenotype(geno)

    zetaCombo = pd.merge(zetaCombo, geno, on= 'sID')

    return zetaCombo

def clusterSummary(zetaCombo):
    # get the cluster and across how many behavioral modality they are present 
    a = zetaCombo.groupby(['Animal_geno','sID','brainArea','KSLabel','cluster']).agg({'KSLabel':['count']})
    a = quickConversion(a)

    # get the cluster present 
    a = a.groupby(['Animalgeno','sID','brainArea', 'KSLabel']).agg({'KSLabel':['count']})
    a = quickConversion(a)
    b = a.groupby(['Animalgeno','sID','brainArea']).agg({'KSLabelcount':['sum']})
    b = quickConversion(b)

    b['KSLabel'] = 'all'
    b = b.rename(columns={'KSLabelcountsum':'KSLabelcount'})
    a = pd.concat([a,b])
    a = a.reset_index(drop=True)
    a = a.rename(columns={'KSLabelcount': 'Clusters (n)'})

    return a

def thresholdSummary(zetaCombo):
    # get the cluster and across how many behavioral modality they are present 
    a = zetaCombo.groupby(['Animal_geno','sID','brainArea','behaviorType']).agg({'sID':['count']})
    a = quickConversion(a)

    # get the cluster present 
    a = a.groupby(['Animalgeno','sID','brainArea', 'KSLabel']).agg({'KSLabel':['count']})
    a = quickConversion(a)
    b = a.groupby(['Animalgeno','sID','brainArea']).agg({'KSLabelcount':['sum']})
    b = quickConversion(b)

    b['KSLabel'] = 'all'
    b = b.rename(columns={'KSLabelcountsum':'KSLabelcount'})
    a = pd.concat([a,b])
    a = a.reset_index(drop=True)
    a = a.rename(columns={'KSLabelcount': 'Clusters (n)'})

    return a

def clusterGraph(a, figIter, myx = 'KSLabel', myy = {'Clusters (n)':'test'}, title = True, myYlim = None):
    a = a.sort_values(by=['Animalgeno','sID'], ascending=False)
    
    if type(myy) == dict:
        oriCol = list(myy.items())[0][0]
        newCol = list(myy.items())[0][1]

        a = a.rename(columns={oriCol:newCol})
        myy = newCol


    defaultDodgeVal = 0.8
    mycategory = ['good','mua','all']
    

    graph = sns.FacetGrid(a, col ='brainArea')
    graph.map(sns.stripplot, myx, myy, data=a, hue='Animalgeno', order=mycategory, dodge=True, zorder=0, alpha =0.4, palette=['#6e9bbd', '#e26770']) # best way to get scatter plot with the Facet grid 
    graph.map(sns.pointplot, myx, myy, data=a, hue='Animalgeno', order=mycategory, join=False, dodge=defaultDodgeVal - defaultDodgeVal/len(mycategory), palette=['#6e9bbd', '#e26770'], zorder=0, alpha =0.4,  ci=68, scale=0.8) # best way to get the mean with the plot // scale is for the marker size 
    graph.add_legend()
    if myYlim == None:
        graph.set(ylim=([0,np.max(a[myy])*1.1])) # set the limit from 0 to 10% of the maximum value
    else:
        graph.set(ylim = (myYlim))
    plt.show()

    if title == True:
        behavType = a['behaviorType'].unique()[0]
        plt.suptitle(behavType)
        plt.tight_layout()
    
        plt.savefig(tpath(r'2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+figIter+behavType+'.pdf'))
    else:
        plt.savefig(tpath(r'2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+figIter+'.pdf'))


## ***************************************************************************
## * ePhys                                                                   *
## ***************************************************************************

# class rasterPSTH:
#     ''' class to access and load the raster data WORK IN PROGRESS'''

#     def __init__(self, myKsDir, eventTimes, depthBinSize=20, window=[-0.300, 1.00], baseline=[-0.3,-0.05], binsize=0.001):
#         self.myKsDir = myKsDir
#         self.eventTimes = eventTimes
#         self.depthBinSize = depthBinSize
#         self.window = window
#         self.baseline = baseline
#         self.binsize = binsize
#         self.spTimes, self.spClu, self.df = spikeload(myKsDir)
#         self.eventTimes =  np.load(eventTimes)

#     def rasterOut(self, cluster=[], normM = True):
#         ########## to do 
#         ########## to do include in class with rasterPsth function


#         ''' function to be able to plot raster directly adapted from spikes from Nick 
#         in Matlab https://github.com/cortex-lab/spike
        
#         Correspondance note matlab to python:
#             pythonToMatlabDict = {'xOut':'rasterX', 'yOut':'rasterY', 'raster':'binnedArray', 'psthMEAN':'psth', 'allP':'allP'}

#         relevant matlabScripts from https://github.com/cortex-lab/spike are:
#             -loadKSDir.m
#             -psthAandB.m
#             -timestampsToBinned.m
#             -psthByDepth.m




#         Parameters:
#             df(pd.DataFrame): contains 'times' and 'cluster'

#             eventTimes: times of relevant behavioral onset in seconds

#             cluster (list): optional can plot give cluster of intrest

#             window(list, float): time of interest of window for analysis in seconds by defautl -0.3 seconds to 1 seconds

#             binsize(float): size of the bins in seconds

#             rasterOnly(logical): True

#         Returns:
#             key parameters for ploting raster and performing psth analysis 

#         '''
#         # create the template for the psth vizualization
#         binTemplate = np.arange(self.window[0], self.window[1] + self.binsize, self.binsize)

#         # condtions to be able to plot only individual clusters or set of clusters
#         # if cluster == []:
#         #     cluster = df['cluster'].unique()

#         # for singleCluster in np.unique(cluster):
#             # get the spike times for a given cluter

#         if type(cluster) == int or type(cluster) == numpy.int64:
#             # print('test')
#             spikeTimes = np.array(self.df.loc[self.df['cluster'].isin([cluster]), 'times'])
        
#         elif cluster == []:
#             # print('test1')
#             cluster = self.df.cluster.unique()
#             spikeTimes = np.array(self.df.loc[self.df['cluster'].isin(cluster), 'times'])
#         else:
#             # print('test2')
#             spikeTimes = np.array(self.df.loc[self.df['cluster'].isin(cluster), 'times'])

#         raster = []
#         rasterBaseline = []
#         for evTime in self.eventTimes:
#             bins = evTime + binTemplate

#             # consider only spikes within window
#             spikeIdx = np.squeeze(np.searchsorted(spikeTimes,
#                                           [evTime + self.window[0],
#                                            evTime + self.window[1]]))
            
#             if self.baseline != []:
#                 baselineIdx = np.squeeze(np.searchsorted(spikeTimes,
#                                               [evTime + self.window[0],
#                                                evTime + self.window[1]]))

#                 spikeCountsBaseline = np.histogram(
#                 spikeTimes[spikeIdx[0]:spikeIdx[1]],
#                 bins)[0]

#                 rasterBaseline.append(spikeCountsBaseline)
#                 psthBaseline = np.mean(raster, axis=0) / self.binsize
#                 normMn = np.mean(psthBaseline)
#                 normStd = np.std(psthBaseline)

#             spikeCounts = np.histogram(
#             spikeTimes[spikeIdx[0]:spikeIdx[1]],
#             bins)[0]

#             raster.append(spikeCounts)

#         raster = np.array(raster)
#         spikeCounts = np.sum(raster, axis=1)

#         tr, b = np.nonzero(raster) #np.where

#         #sort the array to have proper shape
#         toSort = pd.DataFrame({'tr':tr, 'b':b})
#         toSort = toSort.sort_values(by=['b', 'tr'])

#         b = np.array(toSort.b)
#         tr = np.array(toSort.tr)

#         timeStamps = binTemplate[b] # may be error here


#         ## create teh final raster
#         minVal = 0
#         maxVal = 1

#         xOut = np.repeat(np.NaN, len(timeStamps)*3)
#         xOut[0::3] = timeStamps
#         xOut[1::3] = timeStamps

#         yOut = minVal + np.repeat(0, len(timeStamps)*3)
#         yOut[1::3] = maxVal
#         yOut = yOut+np.repeat(tr, 3)



#         #### get the data relevant to the psth

#         psthMEAN = np.mean(raster, axis=0) / self.binsize
        
#         if normM == True and np.std(psthMEAN)>0:
#             # condition above avoid error when normalizing if std is 0
#             psthNorm = (psthMEAN-normMn)/normStd
#         else:
#             psthNorm = psthMEAN


#         return xOut, yOut, raster, binTemplate, psthNorm

#     def psthByDepth(self):
#         '''
#         variable which are identified and taken care of
#         - spikeTimes (derive from df)
#         - spikeDepths (derive from df)
#         - depthBinSize (input pre-established) limited value in this context
#         - timeBinSize (input resoluiton on the analysis windiow)
#         - eventTimes (input based on data)
#         - win (input size of the total analysis window)
#         - bslWin (size of the window)
#         '''
#         ## to obtain the bins and their size bins based on the spacing of the channels in um on the probe 

class myAnimalsRef:
    '''class of file to associate all the propereties related to one specifc id

    '''

    def __init__(self, aid, touchMainDir, archive=False):
        if sys.platform != 'linux':
            myroot = 'Y:/'
        else:
            myroot = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/'

        self.aid = aid
        refFile = pd.read_csv(myroot+'Vaissiere/fileDescriptor2.csv')
        self.animalInfo = refFile.loc[refFile['Record_folder'].str.contains(aid), ['Animal_sex', 'Animal_geno']].values[0]

        # mainDirectories
        ePhysMainDir = myroot + 'Jessie/e3 - Data Analysis/e3 Data/'
        if archive == True:
            ePhysMainDir = myroot + 'Jessie/e3 - Data Analysis/e3 Data/binArchive/'
        touchMainDir = touchMainDir+os.sep
        # r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST\touchTimes'+os.sep## updated on 2/14/2023
        #touchMainDir = 'Y:/Vaissiere/__UbuntuLamda/DLC/cut_pole1and2_threshMeth/touches/evTime'+os.sep
        # touchMainDir = r'Y:\Vaissiere\__UbuntuLamda\DLC\cut_pole_treshMeth_evTime\segAll_tall'+os.sep#'/home/rum/Desktop/DLC/cut_pole_treshMeth_evTime/segAll_tall/'
        whiskMainDir = 'Y:/2020-09-Paper-ReactiveTouch/eventTimes/whiskTime'+os.sep
        # whiskMainDir = '/home/rum/Desktop/DLC/videooutput_angle/evTime/'

        # specific dir
        self.aidShort = aid.split('_')[0]
        self.ePhysDir = glob.glob(ePhysMainDir+'*'+aid+'*/output/*__ap')
        self.runEv = glob.glob(ePhysMainDir+'*'+self.aidShort+'*/outputrunningOnset.txt')
        self.touchEv = glob.glob(touchMainDir+'*'+self.aidShort+'*')
        self.whiskEv = glob.glob(whiskMainDir+'*'+self.aidShort+'*')
        self.ePhysThreshold = glob.glob(ePhysMainDir+'*'+aid+'*/output/**/*threshold_4.5.txt')

    def eventTimes(self):

        if self.runEv == []:
            runEvTimes = [0]
        else:
            runEvTimes = pd.read_csv(self.runEv[0])
            runEvTimes = np.array(runEvTimes['runningOnset']/25000)
            if runEvTimes.size == 0:
                runEvTimes = [0]

        if self.whiskEv == []:
            whiskEvTimes = [0]
        else:
            whiskEvTimes = np.load(self.whiskEv[0])

        if self.touchEv == []:
            touchEvTimes = [0]
        else:
            touchEvTimes = np.load(self.touchEv[0])

        arrayId = ['touch', 'whisk', 'run']
        allEvTimes = [touchEvTimes, whiskEvTimes, runEvTimes]

        return arrayId, allEvTimes

def psthByDepth(df,
                eventTimes, 
                psthType = 'depth',
                depthBinSize = 20,  
                doPlot = False, 
                p_window=[-0.025, 0.050], 
                p_baseline =[-0.025,-0.005], 
                p_binsize = 0.001):
    '''
    psthType (string) = 'depth', 'all', 'cluster',
                        psth type correpsond to the 3 types of psths that can be done
     psthByDepth(dfMasterSave[(dfMasterSave['sID']==animali)], depthBinSize = 20, eventTimes=allTimes[0])
    '''
    window = p_window
    spikeTimes = df['times']
    # filter out the spiketimes as done in the original matlab script see psthAandBA.m :26
    spikeTimes = spikeTimes[(spikeTimes>min(eventTimes+window[0])) & (spikeTimes<max(eventTimes+window[1]))]

    spikeDepths = df['spikeDepth']
    spikeDepths=spikeDepths*depthBinSize
    depthBins = np.arange(min(spikeDepths),max(spikeDepths),depthBinSize)
    nD = len(depthBins)-1
    # bsl event times
    bslEventTimes = eventTimes
    # bsl form evenWindow
    normVals = np.zeros([nD, 2])

    ### add-on for the cluster option
    nCluster = np.unique(df['cluster'])


    if psthType == 'depth':
        allP = []
        actualDepth = []
        print('depth setting')
        for d in range(nD):
            # print(d)
            # get the specific spikes which are located within a depth range
            # this is w
            theseSp = (spikeDepths>depthBins[d]) & (spikeDepths<=depthBins[d+1])
            subsetspTimes = df[theseSp]
            # use python function to get the raster out
            xOut, yOut, raster, binTemplate, psth = rasterOut(df = subsetspTimes, eventTimes = eventTimes, cluster=[], window=p_window, baseline =p_baseline, binsize = p_binsize)
            allP.append(psth)
            actualDepth.append(depthBins[d])
        actualDepth = np.asarray(actualDepth)
        allP = np.squeeze(allP)

        return allP, actualDepth

    elif psthType == 'cluster':
        print('cluster setting')
        allP = []
        actualCluster = []
        for d in nCluster:
            # print(d)
            # get the specific spikes which are located within a depth range
            # this is w
            subsetspTimes = df[df['cluster']==d]
            # use python function to get the raster out
            xOut, yOut, raster, binTemplate, psth = rasterOut(df = subsetspTimes, eventTimes = eventTimes, cluster=[], window=p_window, baseline =p_baseline, binsize = p_binsize)
            allP.append(psth)
            actualCluster.append(d)
        actualCluster = np.asarray(actualCluster)
        allP = np.squeeze(allP)

        return allP, actualCluster

    

def statusCheck(mynpArray, window=[-0.025, 0.05], binsize=0.001, respWin=[0, 0.03], baselineWin=-0.000):
    '''
    function to be used on numpy array to filter the data of interest based on the response amplitude
    '''

    binTemplate = np.arange(window[0], window[1] + binsize, binsize)
    respWin = np.where((binTemplate>=respWin[0]) & (binTemplate<respWin[1]))
    baselineWin = np.where(binTemplate<baselineWin)
    # since we are already using a z-score denotating std above baseline straight up valuel can be used
    tmpVal = np.max(mynpArray[respWin])>2.5
    # tmpVal = np.max(mynpArray[respWin])>np.std(mynpArray[baselineWin])*10

    return tmpVal

def getZeta(arrSpikeTimes, arrEventTimes, dblUseMaxDur=None, intResampNum=100, intPlot=0,
            intLatencyPeaks=2, tplRestrictRange=(-np.inf,np.inf),
            boolReturnRate=False, boolReturnZETA=False, boolVerbose=False):
    """
    Calculates neuronal responsiveness index ZETA.

    Montijn, J.S., Seignette, K., Howlett, M.H., Cazemier, J.L., Kamermans, M., Levelt, C.N.,
    and Heimel, J.A. (2021). A parameter-free statistical test for neuronal responsiveness.
    eLife 10, e71969.

    Parameters
    ----------
    arrSpikeTimes : 1D array
        spike times (in seconds).
    arrEventTimes : 1D or 2D array
        event on times (s), or [T x 2] including event off times to calculate mean-rate difference.
    dblUseMaxDur : float
        window length for calculating ZETA: ignore all spikes beyond this duration after event onset
        (default: median of event onset to event onset)
    intResampNum : integer
        number of resamplings (default: 100)
    intPlot : integer
        plotting switch (0=none, 1=inst. rate only, 2=traces only, 3=raster plot as well,
                         4=adds latencies in raster plot) (default: 0)
    intLatencyPeaks : integer
        maximum number of latency peaks to return (1-4) (default: 2)
    tplRestrictRange : 2 element tuple
        temporal range within which to restrict onset/peak latencies (default: [-inf inf])
    boolReturnRate : boolean
        switch to return dictionary with spiking rate features
    boolReturnZETA : boolean
        switch to return dictionary with additional ZETA parameters
    boolVerbose : boolean
        switch to print progress messages (default: false)

    Returns
    -------
    dblZetaP : float
        p-value based on Zenith of Event-based Time-locked Anomalies
    arrLatencies : 1D array
        different latency estimates, number determined by intLatencyPeaks.
        If no peaks are detected, it returns NaNs
            1) Latency of ZETA
            2) Latency of largest z-score with inverse sign to ZETA
            3) Peak time of instantaneous firing rate
            4) Onset time of above peak, defined as the first crossing of peak half-height
    dZETA : dict (optional)
        additional parameters of ZETA test, return when using boolReturnZETA
            dblZETA; FDR-corrected responsiveness z-score (i.e., >2 is significant)
            dblD; temporal deviation value underlying ZETA
            dblP; p-value corresponding to ZETA
            dblPeakT; time corresponding to ZETA
            intPeakIdx; entry corresponding to ZETA
            dblMeanD; Cohen's D based on mean-rate stim/base difference
            dblMeanP; p-value based on mean-rate stim/base difference
            vecSpikeT: timestamps of spike times (corresponding to vecD)
            vecRealFrac; cumulative distribution of spike times
            vecRealFracLinear; linear baseline of cumulative distribution
            vecD; temporal deviation vector of data
            vecNoNormD; temporal deviation which is not mean subtracted
            matRandD; baseline temporal deviation matrix of jittered data
            dblD_InvSign; largest peak of inverse sign to ZETA (i.e., -ZETA)
            dblPeakT_InvSign; time corresponding to -ZETA
            intPeakIdx_InvSign; entry corresponding to -ZETA
            dblUseMaxDur; window length used to calculate ZETA
    dRate : dict (optional)
        additional parameters of the firing rate, return with boolReturnRate
            vecRate; instantaneous spiking rates (like a PSTH)
            vecT; time-points corresponding to vecRate (same as dZETA.vecSpikeT)
            vecM; Mean of multi-scale derivatives
            vecScale; timescales used to calculate derivatives
            matMSD; multi-scale derivatives matrix
            vecV; values on which vecRate is calculated (same as dZETA.vecZ)
            Data on the peak:
            dblPeakTime; time of peak (in seconds)
            dblPeakWidth; duration of peak (in seconds)
            vecPeakStartStop; start and stop time of peak (in seconds)
            intPeakLoc; spike index of peak (corresponding to dZETA.vecSpikeT)
            vecPeakStartStopIdx; spike indices of peak start/stop (corresponding to dZETA.vecSpikeT)
            Additionally, it will return peak onset latency (first crossing of peak half-height)
            dblOnset: latency for peak onset

    Original code by Jorrit Montijn, ported to python by Alexander Heimel & Guido Meijer

    Version history:
    2.5 - 17 June 2020 Jorrit Montijn, translated to python by Alexander Heimel
    2.5.1 - 18 February 2022 Bugfix by Guido Meijer of 1D arrEventTimes
    2.6 - 20 February 2022 Refactoring of python code by Guido Meijer
    """

    # ensure arrEventTimes is a N x 2 array
    if len(arrEventTimes.shape) > 1:
        boolStopSupplied = True
        if np.shape(arrEventTimes)[1] > 2:
            arrEventTimes = np.transpose(arrEventTimes)
    else:
        boolStopSupplied = False
        arrEventTimes = np.vstack((arrEventTimes, np.zeros(arrEventTimes.shape))).T

    # trial dur
    if dblUseMaxDur is None:
        dblUseMaxDur = np.median(np.diff(arrEventTimes[:,0]))

    ## build onset/offset vectors
    arrEventStarts = arrEventTimes[:,0]

    ## prepare interpolation points
    intMaxRep = np.shape(arrEventTimes)[0]
    cellSpikeTimesPerTrial = [None] * intMaxRep

    # go through trials to build spike time vector
    for intEvent in range(intMaxRep):
        # get times
        dblStartT = arrEventStarts[intEvent]
        dblStopT = dblStartT + dblUseMaxDur

        # build trial assignment
        cellSpikeTimesPerTrial[intEvent] = arrSpikeTimes[(arrSpikeTimes < dblStopT)
                                                         & (arrSpikeTimes > dblStartT)] - dblStartT

    # get spikes in fold
    vecSpikeT = np.array(sorted(flatten([0,cellSpikeTimesPerTrial, dblUseMaxDur])))
    intSpikes = vecSpikeT.shape[0]

    ## run normal
    vecOrigDiff, vecRealFrac, vecRealFracLinear = getTempOffset(vecSpikeT, arrSpikeTimes,
                                                                arrEventStarts, dblUseMaxDur)

    # mean subtract difference
    vecRealDiff = vecOrigDiff - np.mean(vecOrigDiff)

    ## run bootstraps
    hTic = time.time()
    matRandDiff = np.empty((intSpikes, intResampNum))
    matRandDiff[:] = np.nan
    for intResampling in range(intResampNum):
        ## msg
        if boolVerbose and ((time.time()-hTic) > 5):
            ### print('Now at resampling %d/%d' % (intResampling,intResampNum))
            hTic = time.time()

        ## get random subsample
        vecStimUseOnTime = (arrEventStarts + 2 * dblUseMaxDur
                            * ((np.random.rand(arrEventStarts.shape[0]) - 0.5) * 2))

        # get temp offset
        vecRandDiff, vecRandFrac, vecRandFracLinear = getTempOffset(vecSpikeT, arrSpikeTimes,
                                                                    vecStimUseOnTime, dblUseMaxDur)

        # assign data
        matRandDiff[:,intResampling] = vecRandDiff - np.mean(vecRandDiff)

    ## calculate measure of effect size (for equal n, d' equals Cohen's d)
    if (len(vecRealDiff) < 3) | (arrSpikeTimes.shape[0] < 10):
        if boolVerbose:
            logging.warning('Insufficient samples to calculate zeta')
        dblZetaP = 1
        arrLatencies = np.array([np.nan] * intLatencyPeaks)
        dZETA = dict()
        dRate = dict()
        if (boolReturnZETA & boolReturnRate):
            return dblZetaP, arrLatencies, dZETA, dRate
        elif boolReturnZETA:
            return dblZetaP, arrLatencies, dZETA
        elif boolReturnRate:
            return dblZetaP, arrLatencies, dRate
        else:
            return dblZetaP, arrLatencies

    # find highest peak and retrieve value
    vecMaxRandD = np.max(np.abs(matRandDiff), 0)
    dblRandMu = np.mean(vecMaxRandD)
    dblRandVar = np.var(vecMaxRandD, ddof=1)
    intZETALoc = np.argmax(np.abs(vecRealDiff))
    dblPosD = np.max(np.abs(vecRealDiff)) # Can be combined with line above

    # get location
    dblMaxDTime = vecSpikeT[intZETALoc]
    dblD = vecRealDiff[intZETALoc]

    # calculate statistical significance using Gumbel distribution
    # if boolVerbose:
        ### printoff ### print('Python: Gumbel %0.7f, %0.7f, %0.7f' % (dblRandMu, dblRandVar, dblPosD))
    dblZetaP, dblZETA = getGumbel(dblRandMu, dblRandVar, dblPosD)

    # find peak of inverse sign
    intPeakLocInvSign = np.argmax(-np.sign(dblD)*vecRealDiff)
    dblMaxDTimeInvSign = vecSpikeT[intPeakLocInvSign]
    dblD_InvSign = vecRealDiff[intPeakLocInvSign]

    if boolStopSupplied:
        ## calculate mean-rate difference
        vecEventStops = arrEventTimes[:,1]
        vecStimHz = np.zeros(intMaxRep)
        vecBaseHz = np.zeros(intMaxRep)
        dblMedianBaseDur = np.median(arrEventStarts[1:] - vecEventStops[0:-1])

        # go through trials to build spike time vector
        for intEvent in range(intMaxRep):
            # get times
            dblStartT = arrEventStarts[intEvent]
            dblStopT = dblStartT + dblUseMaxDur
            dblPreT = dblStartT - dblMedianBaseDur

            # build trial assignment
            vecStimHz[intEvent] = (np.sum((arrSpikeTimes < dblStopT) & (arrSpikeTimes > dblStartT))
                                   / (dblStopT - dblStartT))
            vecBaseHz[intEvent] = (np.sum((arrSpikeTimes < dblStartT) & (arrSpikeTimes > dblPreT))
                                   / dblMedianBaseDur)

        # get metrics
        dblMeanD = np.mean(vecStimHz - vecBaseHz) / ((np.std(vecStimHz) + np.std(vecBaseHz)) / 2)
        dblMeanP = stats.ttest_rel(vecStimHz, vecBaseHz)

    ## plot
    if intPlot > 1:
        logging.warning('Plotting is not translated to python yet')
        """
        %plot maximally 50 traces
        intPlotIters = min([size(matRandDiff,2) 50]);

        %make maximized figure
        figure
        drawnow;
        jFig = get(handle(gcf), 'JavaFrame');
        jFig.setMaximized(true);
        figure(gcf);
        drawnow;

        if intPlot > 2
            subplot(2,3,1)
            plotRaster(arrSpikeTimes,arrEventStarts(:,1),dblUseMaxDur,10000);
            xlabel('Time from event (s)');
            ylabel('Trial #');
            title('Spike raster plot');
            fixfig;
            grid off;
        end

        %plot
        subplot(2,3,2)
        sOpt = struct;
        sOpt.handleFig =-1;
        [vecMean,vecSEM,vecWindowBinCenters] = doPEP(arrSpikeTimes,0:0.025:dblUseMaxDur,arrEventStarts(:,1),sOpt);
        errorbar(vecWindowBinCenters,vecMean,vecSEM);
        ylim([0 max(get(gca,'ylim'))]);
        title(sprintf('Mean spiking over trials'));
        xlabel('Time from event (s)');
        ylabel('Mean spiking rate (Hz)');
        fixfig

        subplot(2,3,3)
        plot(vecSpikeT,vecRealFrac)
        hold on
        plot(vecSpikeT,vecRealFracLinear,'color',[0.5 0.5 0.5]);
        title(sprintf('Real data'));
        xlabel('Time from event (s)');
        ylabel('Fractional position of spike in trial');
        fixfig

        subplot(2,3,4)
        cla;
        hold all
        for intOffset=1:intPlotIters
            plot(vecSpikeT,matRandDiff(:,intOffset),'Color',[0.5 0.5 0.5]);
        end
        plot(vecSpikeT,vecRealDiff,'Color',lines(1));
        scatter(dblMaxDTime,vecRealDiff(intZETALoc),'bx');
        scatter(dblMaxDTimeInvSign,vecRealDiff(intPeakLocInvSign),'b*');
        hold off
        xlabel('Time from event (s)');
        ylabel('Offset of data from linear (s)');
        if boolStopSupplied
            title(sprintf('ZETA=%.3f (p=%.3f), d(Hz)=%.3f (p=%.3f)',dblZETA,dblZetaP,dblMeanD,dblMeanP));
        else
            title(sprintf('ZETA=%.3f (p=%.3f)',dblZETA,dblZetaP));
        end
        fixfig
    """

    ## calculate MSD if significant
    if intLatencyPeaks > 0:
        # get average of multi-scale derivatives, and rescaled to instantaneous spiking rate
        dblMeanRate =  intSpikes / (dblUseMaxDur * intMaxRep)
        vecRate, dRate = msd.getMultiScaleDeriv(vecSpikeT, vecRealDiff, intPlot=intPlot,
                                                dblMeanRate=dblMeanRate, dblUseMaxDur=dblUseMaxDur)
    else:
        dRate = None

    ## calculate MSD statistics
    if dRate is not None and intLatencyPeaks > 0:
        # get MSD peak
        (dblPeakRate, dblPeakTime, dblPeakWidth, vecPeakStartStop,
         intPeakLoc, vecPeakStartStopIdx) = getPeak(vecRate, vecSpikeT, tplRestrictRange)

        dRate['dblPeakRate'] = dblPeakRate
        dRate['dblPeakTime'] = dblPeakTime
        dRate['dblPeakWidth'] = dblPeakWidth
        dRate['vecPeakStartStop'] = vecPeakStartStop
        dRate['intPeakLoc'] = intPeakLoc
        dRate['vecPeakStartStopIdx'] = vecPeakStartStopIdx

        if not math.isnan(dblPeakTime):
            # assign array data
            if intLatencyPeaks > 3:
                # get onset
                dblOnset, dblOnsetVal = getOnset(vecRate, vecSpikeT, dblPeakTime, tplRestrictRange)[:2]
                dRate['dblOnset'] = dblOnset
                arrLatencies = np.array([dblMaxDTime, dblMaxDTimeInvSign, dblPeakTime, dblOnset])
                vecLatencyVals = np.array([vecRate[intZETALoc], vecRate[intPeakLocInvSign],
                                           vecRate[intPeakLoc], dblOnsetVal], dtype=object)
            else:
                dRate['dblOnset'] = np.nan
                arrLatencies = np.array([dblMaxDTime, dblMaxDTimeInvSign, dblPeakTime])
                vecLatencyVals = np.array([vecRate[intZETALoc], vecRate[intPeakLocInvSign],
                                           vecRate[intPeakLoc]], dtype=object)
            arrLatencies = arrLatencies[0:intLatencyPeaks]
            vecLatencyVals = vecLatencyVals[0:intLatencyPeaks]
            if intPlot > 0:
                logging.warning('Plot not translated to python yet')
                """
                hold on
                scatter(dblPeakTime,vecRate(intPeakLoc),'gx');
                scatter(dblMaxDTime,vecRate(intZETALoc),'bx');
                scatter(dblMaxDTimeInvSign,vecRate(intPeakLocInvSign),'b*');
                if intLatencyPeaks > 3
                    scatter(dblOnset,dblOnsetVal,'rx');
                    title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms,On=%.2fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000,dblOnset*1000));
                else
                    title(sprintf('ZETA=%.0fms,-ZETA=%.0fms,Pk=%.0fms',dblMaxDTime*1000,dblMaxDTimeInvSign*1000,dblPeakTime*1000));
                end
                hold off
                fixfig;

                if intPlot > 3
                    vecHandles = get(gcf,'children');
                    ptrFirstSubplot = vecHandles(find(contains(get(vecHandles,'type'),'axes'),1,'last'));
                    axes(ptrFirstSubplot);
                    vecY = get(gca,'ylim');
                    hold on;
                    if intLatencyPeaks > 3,plot(dblOnset*[1 1],vecY,'r--');end
                    plot(dblPeakTime*[1 1],vecY,'g--');
                    plot(dblMaxDTime*[1 1],vecY,'b--');
                    plot(dblMaxDTimeInvSign*[1 1],vecY,'b-.');
                    hold off
                end
                """
        else:
            #placeholder peak data
            dRate['dblOnset'] = np.nan
            arrLatencies = np.array([np.nan] * intLatencyPeaks)
            vecLatencyVals = np.array([np.nan] * intLatencyPeaks)
    else:
        arrLatencies = []
        vecLatencyVals = []

    ## build optional output dictionary
    dZETA = dict()
    dZETA['dblZeta'] = dblZETA
    dZETA['dblD'] = dblD
    dZETA['dblP'] = dblZetaP
    dZETA['dblPeakT'] = dblMaxDTime
    dZETA['intPeakIdx'] = intZETALoc
    if boolStopSupplied:
        dZETA['dblMeanD'] = dblMeanD
        dZETA['dblMeanP'] = dblMeanP
    dZETA['vecSpikeT'] = vecSpikeT
    dZETA['vecRealFrac'] = vecRealFrac
    dZETA['vecRealFracLinear'] = vecRealFracLinear
    dZETA['vecD'] = vecRealDiff
    dZETA['vecNoNormD'] = vecOrigDiff
    dZETA['matRandD'] = matRandDiff
    dZETA['dblD_InvSign'] = dblD_InvSign
    dZETA['dblPeakT_InvSign'] = dblMaxDTimeInvSign
    dZETA['intPeakIdx_InvSign'] = intPeakLocInvSign
    dZETA['dblUseMaxDur'] = dblUseMaxDur

    if (boolReturnZETA & boolReturnRate):
        return dblZetaP, arrLatencies, dZETA, dRate
    elif boolReturnZETA:
        return dblZetaP, arrLatencies, dZETA
    elif boolReturnRate:
        return dblZetaP, arrLatencies, dRate
    else:
        return dblZetaP, arrLatencies

def tpath(mypath, shareDrive = 'Y'):
    '''
    path conversion to switch form linux to windows platform with define drive
    Args:
    mypath (str): path of the file of interest
    shareDrive (str): windows letter of the shared folder
    '''
    # if ('google.colab' in str(get_ipython())) or sys.platform == 'win32':
    if sys.platform == 'win32':
         myRoot = shareDrive+':'      
    else:
        myRoot = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab'


    newpath = myRoot+os.sep+mypath
    newpath = os.path.normpath(newpath)

    return newpath

def gaussianSmooth(spikets, k = 5): 
    ''' this function enable the smoothing of the spike trains after histogram has been computed 
    k(int) default is 5: time in ms
    '''
    # normalized time vector in ms
    gtime = np.arange(-k,k)

    # create Gaussian window
    fwhm = k
    gauswin = np.exp( -(4*np.lib.scimath.log(2)*gtime**2) / fwhm**2 )
    gauswin = gauswin / np.sum(gauswin)

    # initialize filtered signal vector
    filtsigG = np.zeros(len(spikets))

    # implement the weighted running mean filter
    for i in range(k+1,len(spikets)-k-1):
        filtsigG[i] = np.sum( spikets[i-k:i+k]*gauswin)
    return filtsigG

def spikeload(path, threshold = False):
    ''' from the file path load the cluster (=id of the units) and the spike times4'''
    sampRate = 25000
    
    if threshold == True:
        file = glob.glob(path+'/*threshold_4.5.txt')
        tmpDF = pd.read_csv(file[0])
        spClu = np.array(tmpDF['channel'])
        spTimes = np.array(tmpDF['peak_loc']/sampRate)
        df=pd.DataFrame({'cluster':spClu, 'times':spTimes})
        # dt['peak_loc']
        return spTimes, spClu, df

    else:
        '''
        to retrive spike depths and site depths and other spiking inforation
        this is a python port from the matlab version see here https://github.com/cortex-lab/spikes
        '''
        spClu = np.squeeze(np.load(path+'/spike_clusters.npy'))
        pcFeatInd = np.load(path+'/pc_feature_ind.npy')
        spAmps = np.squeeze(np.load(path+'/amplitudes.npy'))

        pcFeat = np.load(path+'/pc_features.npy')
        pcFeat = pcFeat[:,0,:] # take only the first components
        pcFeat[pcFeat<0] = 0 

        ycoords = np.load(path+"/channel_positions.npy")
        ycoords = ycoords[:,1:]

        ## output of the spike absolute depths
        #### which channels for each spike?
        spikeFeatInd = pcFeatInd[spClu,:]
        spikeFeatInd = np.squeeze(spikeFeatInd) # to reduce one dimension of the array
        spikeFeatYcoords = ycoords[spikeFeatInd]
        spikeFeatYcoords = np.squeeze(spikeFeatYcoords) # to reduce one dimension of the array

        spikeDepths = np.squeeze(np.array([np.sum(spikeFeatYcoords*pcFeat**2, 1)]).T/np.array([np.sum(pcFeat**2,1)]).T)

        ## output of the spike depths per sites
        #### matrix operation ported from matlab script see above ref
        winv = np.load(path + "/whitening_mat_inv.npy")
        temps = np.load(path+"/templates.npy")
        tempsUnW = np.zeros(np.shape(temps))
        for t in np.arange(0,np.shape(temps)[0]):
            # print(t)
            tempsUnW[t,:,:] = np.matmul(temps[t,:,:],winv)
        max_site = np.array([np.argmax(np.max(abs(tempsUnW),1),1)]).T
        spikeSites = np.squeeze(max_site[spClu])

        spTimes = np.squeeze(np.load(path+'/spike_times.npy')/sampRate)
        ksStatusAmp = pd.read_csv(path+"/cluster_Amplitude.tsv", sep='\t') # get the cluster id 
        ksStatusCont = pd.read_csv(path+"/cluster_ContamPct.tsv", sep='\t') # get the cluster id 
        ksStatustmp =  pd.read_csv(path+"/cluster_KSLabel.tsv", sep='\t') 
        ksStatus = pd.merge(ksStatusAmp, ksStatustmp, on='cluster_id')
        ksStatus = pd.merge(ksStatus, ksStatusCont, on='cluster_id')
        ksStatus = ksStatus.rename(columns={'cluster_id':'cluster', 'Amplitude':'clust_amp'})

        df=pd.DataFrame({'cluster':spClu, 'times':spTimes, 'spAmps':spAmps, 'spikeDepth': spikeDepths, 'spikeSites': spikeSites})

        df = pd.merge(ksStatus, df, on='cluster')
        ## reformat the data frame so that the values order correspond
        ## to the order of the spike times plt
        df = df.sort_values('times').reset_index(drop=True)

        return spTimes, spClu, df

def getGenotype(file):
    """
    file: string 
        raw file name of the genotype
    return:
        pandas data frame trim and workable for downstream processing
    """
    geno = pd.read_csv(file)
    geno = geno[['Animal_id', 'Animal_sex', 'Animal_geno', 'Record_folder']]
    geno['Record_folder'] = geno['Record_folder'].str.split('_', expand = True)[0]
    geno['Animal_sex'] = geno['Animal_sex'].str.replace(' ', '')
    geno = geno.drop_duplicates()
    geno = geno.rename(columns={'Record_folder': 'sID'})
    geno.reset_index(inplace = True, drop = True)

    return geno

def subselectNonProcessed():
    '''Function to select and reprocess only the file that have not be already exported'''

    animalsRecFolder = getIDofAnalyzedAnimals()
    processedAnimals = glob.glob('/home/rum/Desktop/testScatterOut/*')
    processedAnimals = pd.DataFrame({'procLst': processedAnimals})
    processedAnimals = processedAnimals['procLst'].str.split('/', expand = True)[5].str[0:19].unique()# map(lambda x: x[0:19])
    subselectNonProcessed = np.setdiff1d(animalsRecFolder,processedAnimals)

    return subselectNonProcessed

def rasterOut(df, eventTimes, cluster=[], window=[-0.300, 1.00], baseline =[-0.3,-0.05],binsize = 0.001, normM = True):
    ########## to do 
    ########## to do include in class with rasterPsth function


    ''' function to be able to plot raster directly adapted from spikes from Nick 
    in Matlab https://github.com/cortex-lab/spike
    
    Correspondance note matlab to python:
        pythonToMatlabDict = {'xOut':'rasterX', 'yOut':'rasterY', 'raster':'binnedArray', 'psthMEAN':'psth', 'allP':'allP'}

    relevant matlabScripts from https://github.com/cortex-lab/spike are:
        -loadKSDir.m
        -psthAandB.m
        -timestampsToBinned.m
        -psthByDepth.m




    Parameters:
        df(pd.DataFrame): contains 'times' and 'cluster'

        eventTimes: times of relevant behavioral onset in seconds

        cluster (list): optional can plot give cluster of intrest

        window(list, float): time of interest of window for analysis in seconds by defautl -0.3 seconds to 1 seconds

        binsize(float): size of the bins in seconds

        rasterOnly(logical): True

    Returns:
        key parameters for ploting raster and performing psth analysis 

    '''
    # create the template for the psth vizualization
    binTemplate = np.arange(window[0], window[1], binsize)
    basebinTemplate = np.arange(baseline[0], baseline[1]+binsize, binsize) # the plus baseline[1]+binsize really important here to match matlab output
    # condtions to be able to plot only individual clusters or set of clusters
    # if cluster == []:
    #     cluster = df['cluster'].unique()

    # for singleCluster in np.unique(cluster):
        # get the spike times for a given cluter

    if type(cluster) == int or type(cluster) == numpy.int64:
        # print('test')
        spikeTimes = np.array(df.loc[df['cluster'].isin([cluster]), 'times'])
    
    elif cluster == []:
        # print('test1')
        cluster = df.cluster.unique()
        spikeTimes = np.array(df.loc[df['cluster'].isin(cluster), 'times'])
    else:
        # print('test2')
        spikeTimes = np.array(df.loc[df['cluster'].isin(cluster), 'times'])

    raster = []
    rasterBaseline = []
    for evTime in eventTimes:
        bins = evTime + binTemplate

        # consider only spikes within window
        spikeIdx = np.squeeze(np.searchsorted(spikeTimes,
                                      [evTime + window[0],
                                       evTime + window[1]]))
        
        if baseline != []:
            binsBase = evTime + basebinTemplate
            baselineIdx = np.squeeze(np.searchsorted(spikeTimes,
                                          [evTime + baseline[0],
                                           evTime + baseline[1]]))

            spikeCountsBaseline = np.histogram(
            spikeTimes[baselineIdx[0]:baselineIdx[1]],
            binsBase)[0]

            rasterBaseline.append(spikeCountsBaseline)

        spikeCounts = np.histogram(
        spikeTimes[spikeIdx[0]:spikeIdx[1]],
        bins)[0]

        raster.append(spikeCounts)

    rasterBaseline = np.array(rasterBaseline)
    psthBaseline = np.mean(rasterBaseline, axis=0) / binsize
    normMn = np.mean(psthBaseline)
    normStd = np.std(psthBaseline)


    raster = np.array(raster)
    ## TODO improve on the dealing with this
    ## there are some issue with floating binaries
    ## rounding so in order to fix this
    ## just drop the last column of the array
    raster = raster[:, :-1]
    spikeCounts = np.sum(raster, axis=1)

    tr, b = np.nonzero(raster) #np.where

    #sort the array to have proper shape
    toSort = pd.DataFrame({'tr':tr, 'b':b})
    toSort = toSort.sort_values(by=['b', 'tr'])

    b = np.array(toSort.b)
    tr = np.array(toSort.tr)

    timeStamps = binTemplate[b] # may be error here


    ## create teh final raster
    minVal = 0
    maxVal = 1

    xOut = np.repeat(np.NaN, len(timeStamps)*3)
    xOut[0::3] = timeStamps
    xOut[1::3] = timeStamps

    yOut = minVal + np.repeat(0, len(timeStamps)*3)
    yOut[1::3] = maxVal
    yOut = yOut+np.repeat(tr, 3)



    #### get the data relevant to the psth

    psthMEAN = np.mean(raster, axis=0) / binsize
    
    if normM == True and normStd>0:
        # condition above avoid error when normalizing if std is 0
        psthNorm = (psthMEAN-normMn)/normStd
    else:
        psthNorm = psthMEAN

    return xOut, yOut, raster, binTemplate, psthNorm

def rasterPsthPlot(raster, ax, binTemplate, k=10, labelcust='Touch', window=[-0.300, 1.00], binsize = 0.001, normM = True, **kwargs):
    ########## to do 
    ########## to do include in class with rasterOut function

    ######## kwargs ########
    mycol = kwargs.get('mycol', None)
    if mycol is not None:
        mycol = mycol
        lineOri = 'grey'
    else:
        mycol = 'k'
        lineOri = 'red'
    ######## kwargs ########

    psthMEAN = np.mean(raster, axis=0) / binsize
    psthSTD = np.sqrt(np.var(raster, axis=0)) / binsize
    psthSEM = psthSTD / np.sqrt(float(np.shape(raster)[0]))

    ### this part was commented out in some cases 
    binTemplate = binTemplate[:np.shape(raster)[-1]] ### this part was commented out in some cases 

    if normM == True and np.std(psthMEAN)>0:
        # condition above avoid error when normalizing if std is 0
        normMn = np.mean(psthMEAN)
        normStd = np.std(psthMEAN)
        psthNorm = (psthMEAN-normMn)/normStd
    else:
        psthNorm = psthMEAN

    # if yOut.size == 0:
    #     ax.text(0.5, 0.5, 'no data')
    # else:
    ax.plot(binTemplate, gaussianSmooth(psthMEAN, k), color=mycol)
    ax.fill_between(binTemplate, gaussianSmooth(psthMEAN+psthSEM, k), gaussianSmooth(psthMEAN-psthSEM, k), color=mycol, alpha=0.2)
    ax.vlines(0, 0, max(gaussianSmooth(psthMEAN+psthSEM, k))*1.1, linestyles='dashed' , color=lineOri, linewidth=1)
    ax.set_xlim(window[0], window[1])
    ax.set_ylim(0, max(gaussianSmooth(psthMEAN+psthSEM, k))*1.1)
    ax.set_ylabel('spikes/s')
    ax.set_xlabel('Time (sec.) from '+labelcust+' onset')

    return psthMEAN

def rasterPlot(xOut, yOut, ax, labelcust='Touch', **kwargs):
    # xOut, yOut, raster, binTemplate = rasterOut(df, eventTimes, cluster=cluster, window=[-0.300, 1.00], binsize = binsize) 
    
    ######## kwargs ########
    mycol = kwargs.get('mycol', None)
    if mycol is not None:
        mycol = mycol
        lineOri = 'grey'
    else:
        mycol = 'k'
        lineOri = 'red'
    ######## kwargs ########

    if yOut.size == 0:
        ax.text(0.5, 0.5, 'no data')
    else:    
        ax.plot(xOut, yOut, color=mycol, linewidth=0.4)  #linewidth=0.2 default is 0.4
        # plt.xlim(window[0], window[1])
        # plt.set_ylim(0, max(yOut))
        ax.vlines(0, min(yOut), max(yOut)*0.1, linestyles='dashed', color=lineOri, linewidth=1)
        ax.vlines(0, max(yOut)*0.9, max(yOut), linestyles='dashed', color=lineOri, linewidth=1)
        ax.set_xlabel('Time (sec.) from '+labelcust+' onset')
        ax.set_ylabel(labelcust+' (n)')
        # ax.show()
        # f.savefig('/home/rum/Desktop/testScatterOut'+'/'+str(cluster)+'.png')
        # plt.close('all')

def openTreadmillData(data):
    ''' Function to open treadmill data
    data(str): string name of a dat file acquired and saved in matlab
    '''
    with open(data, 'rb') as fid:
        data_array = np.fromfile(fid, np.single)
    return data_array

def getIDofAnalyzedAnimals(path=tpath('Jessie/e3 - Data Analysis/e3 Data')):
    allFolder = glob.glob(path+'/*/output/*')# all folder with an output
    allFolder = pd.DataFrame({'folder':allFolder})
    allFolder = allFolder['folder'].str.split(os.sep, expand=True)
    if sys.platform == 'win32':
        allFolder = allFolder[4].unique()
    else:
        allFolder = allFolder[9].unique()
    allFolder = np.array([s for s in allFolder if 'opto' not in s])

    return allFolder

def getSummaryZETAstat(path, df, behaviorType = None, threshold = False, eventTimesPath = tpath('2020-09-Paper-ReactiveTouch\\eventTimes\\touchTime')):
    '''
    function to output the ZETA files and optain summary for it
    args:
        path (str): file path containing kilosort output
        df (pd.DataFrame): assoicated with the file path above and output from the spikeload function
        eventTimes (str): main path of events eg.  'Y:\\2020-09-Paper-ReactiveTouch\\eventTimes\\touchTime\\'

    return:
        allZETA (pd.DataFrame): pandas data frame 
    '''
    
    ### get information from file path
    sID = path.split(os.sep)[-1].split('_')[0]
    brainArea = path.split(os.sep)[-1].split('_')[3]

    ### get the event times based on folder
    
    try:
        # this deal with the potential missing file for some of the experiments
        arrEventTimes = np.load(eventTimesPath+os.sep+sID+'.npy', allow_pickle=True)
    except:
        arrEventTimes = []
    
    if behaviorType == None:
        behaviorType = eventTimesPath.split(os.sep)[-1]# get the type of behavior 

    ### trim the df to appropriate spikes and other info
    allZETA=[]
    for kk in df['cluster'].unique():
        # print(kk)
        arrSpikeTimes = np.array(df.loc[df['cluster'] == kk, 'times'])

        ### get the information from ZETA
        ### import to go back to the example where begining and end of stimulation can be defined
        try:
            np.random.seed(1)
            dblZetaP, arrLatencies, dZETA, dRate= getZeta(arrSpikeTimes, arrEventTimes, dblUseMaxDur=None, intResampNum=100, intPlot=0,
                        intLatencyPeaks=2, tplRestrictRange=(-np.inf,np.inf),
                        boolReturnRate=True, boolReturnZETA=True, boolVerbose=True)

            if threshold == True:
                ##iiiii summary purposes table reconstruction
                tmpZETA = pd.DataFrame({
                ## all the ZETA related information
                'ZETA':dZETA['dblPeakT'],
                'ZETA':dZETA['dblPeakT_InvSign'],
                'stat_ZETA_FDR>2':dZETA['dblZeta'],# FDR-corrected responsiveness z-score (i.e., >2 is significant)
                'stat_ZETA_pValue':dZETA['dblP'],
                # peak useful information 
                'spRate@Peak':dRate['dblPeakRate'], # spike per seconds at peak time
                'peakTime': dRate['dblPeakTime'], # peak time
                'peakWidth': dRate['dblPeakWidth'],
                'peakStart': [dRate['vecPeakStartStop'][0]],
                'peakStop': [dRate['vecPeakStartStop'][-1]],
                # dRate['vecRate'] #instantaneous spiking rates (like a PSTH)

                ## all the sample related information
                'cluster': kk,
                'sID': sID,
                'brainArea': brainArea,
                'behaviorType': behaviorType
                  })

            else:
                ##iiiii summary purposes table reconstruction
                tmpZETA = pd.DataFrame({
                ## all the ZETA related information
                'ZETA':dZETA['dblPeakT'],
                'ZETA':dZETA['dblPeakT_InvSign'],
                'stat_ZETA_FDR>2':dZETA['dblZeta'],# FDR-corrected responsiveness z-score (i.e., >2 is significant)
                'stat_ZETA_pValue':dZETA['dblP'],
                # peak useful information 
                'spRate@Peak':dRate['dblPeakRate'], # spike per seconds at peak time
                'peakTime': dRate['dblPeakTime'], # peak time
                'peakWidth': dRate['dblPeakWidth'],
                'peakStart': [dRate['vecPeakStartStop'][0]],
                'peakStop': [dRate['vecPeakStartStop'][-1]],
                # dRate['vecRate'] #instantaneous spiking rates (like a PSTH)

                ## all the sample related information
                'spikeSites': df.loc[df['cluster'] == kk, 'spikeSites'].values[0],
                'spikeDepth': df.loc[df['cluster'] == kk, 'spikeDepth'].values[0],
                'KSLabel': df.loc[df['cluster'] == kk, 'KSLabel'].values[0],
                'cluster': kk,
                'sID': sID,
                'brainArea': brainArea,
                'behaviorType': behaviorType
                  })

            allZETA.append(tmpZETA)

        except:
            continue
            # print('see WARNING above')
    try:
        allZETA = pd.concat(allZETA)
    except:
        allZETA=pd.DataFrame()

    return allZETA

def edgeDetection(array, getPlot = False):
    ''' usage for optogenetic analysis of analog pulse extraction
    '''
    ## clean up the the signal with rounding approximate but will help 
    ## with this specific case 
    array = np.round(array,1) # round the value with decimal under one 
    array[array>=1] = np.round(array[array>=1],0)
    array[array<=0] = 0

    ## get the onset of the event rising edge
    filt = np.append(True, np.diff(array)>0) #need to append as diff will result in n-1 
    ## as the edges are not perfect in the time series with points between low and high
    ## need to get rid of the consecutive points idenified with the diff
    idxRise = np.where(filt)[0]
    filt = np.append(np.diff(idxRise)!=1, True) #need to append as diff will result in n-1 
    idxRise = idxRise[filt]

    filt = np.append(True, np.diff(array)<0) #need to append as diff will result in n-1 
    ## as the edges are not perfect in the time series with points between low and high
    ## need to get rid of the consecutive points idenified with the diff
    idxFall = np.where(filt)[0]
    filt = np.append(np.diff(idxFall)!=1, True) #need to append as diff will result in n-1 
    idxFall = idxFall[filt]

    idxFall = idxFall[1:]
    idxRise = idxRise[1:]
    ### create arbitrary category based on the succession of amplitude
    amplitude = array[idxRise]
    switchIdx = np.where(np.diff(amplitude)!=0)[0][::-1]
    stimCategories = np.zeros(len(amplitude))+len(switchIdx)
    for i, j in enumerate(switchIdx):
        j = j+1
        catOrder = np.array(range(len(switchIdx)))[::-1]
        stimCategories[:j] = catOrder[i]
    everyPulse = pd.DataFrame({'start': idxRise, 'stop': idxFall, 'amplitude': amplitude, 'stimCat': stimCategories})

    ### get a SUPcategorical order for highFrequency pulses 
    ### create major category based on the succession of amplitude
    ### summarize the data 
    supFall = idxFall[np.append(np.diff(idxFall)>600, True)]
    supRise = idxRise[np.append(True, np.diff(idxRise)>600)]
    
    amplitude = array[supRise]
    switchIdx = np.where(np.diff(amplitude)!=0)[0][::-1]
    stimCategories = np.zeros(len(amplitude))+len(switchIdx)
    for i, j in enumerate(switchIdx):
        j = j+1
        catOrder = np.array(range(len(switchIdx)))[::-1]
        stimCategories[:j] = catOrder[i]

    mainEvents = pd.DataFrame({'start': supRise, 'stop': supFall, 'amplitude': amplitude, 'stimCat': stimCategories})
    mainEvents['trial'] = mainEvents.groupby(['stimCat']).cumcount()

    #### check with plot
    if getPlot == True:
        plt.figure()
        plt.plot(array)
        plt.plot(idxRise, array[idxRise], '.', color='green')
        plt.plot(idxFall, array[idxFall], '.', color='red')

        plt.figure()
        plt.plot(array)
        plt.plot(supRise, array[supRise], '.', color='green')
        plt.plot(supFall, array[supFall], '.', color='red')
        ## get the onset 

    return mainEvents, everyPulse


def reshapThresholdDat(dat):
    ''' function to reshape the threshold data frame equivalent to df from threshold 4.5 
    so that it can be read by the method rasterOut
    '''
    fs = 25000 # default acquisition rate
    dat['times'] = dat['peak_loc']/fs
    dat['spikeSites'] = dat['channel']

    return dat

####### windows plot manipulation 
### how to get a list of the windows 
import sys
from tkinter import *
from tkinter.ttk import *
import matplotlib.pyplot as plt
plt.ion()



def collapseplot(position = 'tl', size=200):
    '''
    This function will collapse all the plot present to a specific location with a default size of 200px
    position option tl : for top left
    position option tr : for top right
    position option bl : for bottom left
    postion option br: for bottom right
    '''

    ## get the screen size 
    root = Tk()
    h = root.winfo_screenheight()
    w = root.winfo_screenwidth()

    for i in plt.get_fignums():
        plt.figure(i)
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:

        if position == 'tl':
            mngr.window.setGeometry(0,0,size,size)
        elif position == 'bl':
            mngr.window.setGeometry(0,h-size,size,size)
        elif position == 'tr':
            mngr.window.setGeometry(w-size,0,size,size)
        elif position == 'br':
            mngr.window.setGeometry(w-size,h-size,size,size)

def sortplot(nrow=2):
    '''
    function to sort all the plots on display on the window

    example usage:
    # for i in range(16):
    #     print(i)
    #     plt.figure()
    #     plt.plot(range(i), range(i), '.')


    # sortplot(3)


    '''
    #nrow = 2 # number of rows for all the plot 
    nplot = len(plt.get_fignums()) ### get the number of plot
    ncol = -(-nplot//nrow) # integer division to round up hence the inverse

    # get the column configuration
    ncolt = []
    for i in range(nrow):
        ncolt.append(ncol)
    print(ncolt)

    ## get the screen size 
    root = Tk()
    h = root.winfo_screenheight()
    w = root.winfo_screenwidth()

    ## get the individual size of the figure
    w_ind = w//ncolt[0]
    h_ind = h//len(ncolt)

    init = 0
    count = 0
    for k in range(len(ncolt)):
        # print(k)
        for i in range(ncolt[k]):
            count += 1
            print(count)
            plt.figure(count)
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            mngr.window.setGeometry(init+i*w_ind,init+k*h_ind,w_ind,h_ind)

def search_sequence_numpy(arr,seq):
    # https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found


def readHdf(file):
    ''' function to flatten the hdf files output from DLC
    '''

    vname = file.split(os.sep)[-1].split('DLC')[0]
    scorer = file.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]

    if '_filtered' in scorer:
        scorer = scorer.split('_filtered')[0]

    df = pd.read_hdf(file, "df_with_missing")
    df = df[scorer]
    # drop the multi index and change the column names
    df.columns = [''.join(col) for col in df.columns]
    # reset row index
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'frame'}, inplace=True)

    return df

##################################################################################
## curvature methods
##################################################################################

def polynomial_fit(x, a, b, c):
        return a*x**2+b*x+c

def curvaturenew(x, a, b, c):
    return (2*a)/pow((1+(2*a*x+b)*(2*a*x+b)),(3/2)) #signed polynomila
    # return abs(2*a)/pow((1+(2*a*x+b)*(2*a*x+b)),(3/2)) #to have unsigned

def global_curvature(yval, xval, getPlot=False):
    
    '''
    Usage example:
    df = readHdf(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2019-03-25_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5")
    df['frame'] = df['frame']+1

    start = time.time()
    df['curvature'] = df.apply(lambda x: global_curvature([x['w0y'], x['w1y'], x['w2y'], x['w3y'], x['w4y']]
                                        ,[x['w0x'], x['w1x'], x['w2x'], x['w3x'], x['w4x']]), axis=1)
    end = time.time()-start;print(end)

    df = df[['frame','curvature']]
        '''

    X=np.array(xval)
    Y=np.array(yval)  

    a,b,c = np.polyfit(X, Y, 2) 
    x = np.linspace(np.min(X), np.max(X)) 
    y = np.mean(curvaturenew(x, a, b, c))

    if getPlot == True:
        plt.plot()
        plt.scatter(X,Y, s=5, alpha=0.6) 
        plt.plot(x, polynomial_fit(x, a, b, c), alpha=0.06, color='grey')
    return y

def extractThePoint(df):

    # select the column with the following criteria
    dfw=df.filter(regex=r'w', axis=1)
    dfwx=dfw.filter(regex=r'x', axis=1)
    dfwy=dfw.filter(regex=r'y', axis=1)

    ## for some of the data this requires 
    dfwx=dfwx.filter(regex='^((?!cc).)*$', axis=1) # this is the regex syntax for ommision
    dfwy=dfwy.filter(regex='^((?!cc).)*$', axis=1)

    return dfwx, dfwy

class comboBehavior:
    ''' class to combine behavior for one sample'''

    def __init__(self, sID):
        self.sID = sID
        self.whiskDetails = pd.read_csv(glob.glob('Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122/'+'*'+sID+'*'+'_annoted.csv')[0])
        self.whiskDetails = self.whiskDetails[['Unnamed: 0','FirstwhiskTime','anal_status','polePresent','whiskDurationTime']]
        self.whiskDetails = self.whiskDetails.rename(columns={'anal_status':'anal_status_whisk'})
        self.whiskDetails = self.whiskDetails.rename(columns={'Unnamed: 0':'whiskIdx'})
        self.whiskDetails['whiskIdx'] = self.whiskDetails['whiskIdx']+1
        self.whiskDetails = self.whiskDetails.rename(columns={'FirstwhiskTime':'onset_time_whisk', 'whiskDurationTime':'duration_time_whisk'})

        self.newTouch = glob.glob(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST"+os.sep+self.sID+"*.csv")


        ## the section below is depracated section above should be prefered
        # self.touchDetails = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_501_touchDetails.csv")
        self.touchDetails = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_501_touchDetails_CORRECTED_2023.csv")
        self.touchDetails = self.touchDetails[self.touchDetails['sID']==sID]
        self.touchDetails = self.touchDetails.rename(columns={'Unnamed: 0':'touchIdx'})
        self.touchDetails['touchIdx'] = self.touchDetails['touchIdx']+1
        self.touchDetails = self.touchDetails.rename(columns={'FirstTouchTime':'onset_time_touch', 'touchCountTime':'duration_time_touch'})
        
        ### TODO need to add the auc of the whisking curvature 
        self.curvature = pd.read_csv(glob.glob('Y:/2020-09-Paper-ReactiveTouch/__interimDat__/curvature/'+'*'+sID+'*'+'.csv')[0])
        
        ## see "Y:\2020-09-Paper-ReactiveTouch\__code__\codeForRun.py"
        fpath = 'Y:\\Jessie\\e3 - Data Analysis\\e3 Data'
        self.runDetails = pd.read_csv(glob.glob(fpath+'/*/output/*'+sID+"*_treadmill_summary.csv")[0])
        self.runDetails = self.runDetails.reset_index()
        self.runDetails = self.runDetails.rename(columns={'index':'runIdx'})
        self.runDetails = self.runDetails[['runIdx', 'length', 'idx_start', 'auc']]
        self.runDetails = self.runDetails.rename(columns={'length':'duration_time_run', 'idx_start':'onset_time_run','auc':'auc_run'})

    def temporalCombination(self, option):
        '''
        array1: pandas data frame with all the details for the TOUCH analysis
        array2: pandas data frame with all the details for WHISK per animals
        option (str): 
        return: give the indices for the combined datasets 

        in the use case of combining this touch anc whisk data first start by combining the touch is the array1, array2 is whiks
        in the use case of combining this run and whisk data first arra1 correspond to the whisk data array2 is run 

        TODO: those can be merged and probably extended to use the function independently of the array being used 
        '''
        if option == 1:
            array1 = self.touchDetails
            array2 = self.whiskDetails
            array2 = arrya2.dropna() ### this was introduce on 2-15-2023 the id of the touches might have been shifted

            print('touch_to_whisk')
            timeLabel_arr1 = 'onset_time_touch'
            durLabel_arr1 = 'duration_time_touch'
            idxLabel_arr1 = 'touchIdx'

            timeLabel_arr2 = 'onset_time_whisk'
            durLabel_arr2 = 'duration_time_whisk'
            idxLabel_arr2 = 'whiskIdx'

        elif option == 2:
            array1 = self.whiskDetails
            array1 = array1.dropna() ### this was introduce on 2-15-2023 the id of the touches might have been shifted
            array2 = self.runDetails
            
            print('whisk_to_run')
            timeLabel_arr1 = 'onset_time_whisk'
            durLabel_arr1 = 'duration_time_whisk'
            idxLabel_arr1 = 'whiskIdx'

            timeLabel_arr2 = 'onset_time_run'
            durLabel_arr2 = 'duration_time_run'
            idxLabel_arr2 = 'runIdx'


        tmp = []
        for kdy, y in tqdm.tqdm(array1.iterrows()):
            
            lbd_t = y[timeLabel_arr1]
            hbd_t = lbd_t + y[durLabel_arr1]
            idxVal = y[idxLabel_arr1]

            ## create a subset of array2 that need to be assessed
            ## because every touch is detected after the onset of whisking can limit the analysis this way 
            filterLow = array2.loc[(array2[timeLabel_arr2]<lbd_t), timeLabel_arr2].values[-1]
            tmpArray = array2[array2[timeLabel_arr2]>=filterLow]
            idxWhi = tmpArray[idxLabel_arr2].values[0]

            if option == 1:
                tmpSmp = pd.DataFrame({'touchIdx':[idxVal], 'whiskIdx':[idxWhi]})
            elif option == 2:
                tmpSmp = pd.DataFrame({'whiskIdx':[idxVal], 'runIdx':[idxWhi]})
            tmp.append(tmpSmp)

        tmp = pd.concat(tmp)

        return tmp

    def getcombinedIdx(self):
        touch_to_whisk = self.temporalCombination(option=1)
        whisk_to_run = self.temporalCombination(option=2)

        # do an outer merge to combine the data 
        # this will results in nan but no loos 
        mergedDat = pd.merge(whisk_to_run, touch_to_whisk, on = 'whiskIdx', how='outer')

        return mergedDat
    
    def getCurvature(self, method='local'):

        ''' calculate the curvature difference based on the 
        method ('global'): default is global takes the curvature average of the first 
        'local': takes the curvature average of the previous 10 frames
        '''
        print('AUC curvature -----')
        if method == 'global':
            baselineAUC = self.curvature.loc[:10000,'curvature'].mean()

        AUCall = []
        peakCurv = []
        for i, j in self.touchDetails.iterrows():
            # print(i, j)
            baselineAUC = self.curvature.loc[int(j['FirstTouchFrame']-10):int(j['FirstTouchFrame']-1), 'curvature'].mean()
            tmpauc = np.trapz(self.curvature.loc[int(j['FirstTouchFrame']):int(j['FirstTouchFrame']+j['touchCountFrame']),'curvature']-baselineAUC)
            tmpPeak = np.max(self.curvature.loc[int(j['FirstTouchFrame']):int(j['FirstTouchFrame']+j['touchCountFrame']),'curvature']-baselineAUC)
            AUCall.append(tmpauc)
            peakCurv.append(tmpPeak)

        self.touchDetails['aucCurv_touch'] = np.asarray(AUCall)
        self.touchDetails['peakCurv_touch'] = np.asarray(peakCurv)

        return self.touchDetails
    
    def getcombinedAnalogMetrics(self):
        temp = self.getcombinedIdx()
        self.touchDetails = self.getCurvature()
        
        temp = pd.merge(temp, self.whiskDetails, on = 'whiskIdx', how='outer')
        temp = pd.merge(temp, self.touchDetails, on = 'touchIdx', how='outer')
        temp = pd.merge(temp, self.runDetails, on = 'runIdx', how='outer')
        ## this will expend the range of the the data since not all the the run events includes whisking


        return temp



def extractThePoint(x):

    # select the column with the following criteria
    dfw=df.filter(regex=r'w', axis=1)
    dfwx=dfw.filter(regex=r'x', axis=1)
    dfwy=dfw.filter(regex=r'y', axis=1)

    ## for some of the data this requires 
    dfwx=dfwx.filter(regex='^((?!cc).)*$', axis=1) # this is the regex syntax for ommision
    dfwy=dfwy.filter(regex='^((?!cc).)*$', axis=1)

    return dfwx, dfwy

def extractTheFrame(i):
    '''
    function to be able to extract frame and add DLC 
    label of curvature
    '''
    i = int(i)
    exportPath = r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\curvature\video"
    aid = "2019-03-25"
    
    ### define video properties
    ###----------------------------------------------
    cap = cv2.VideoCapture(vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) 
    custdpi = 300
    xdimIm = cap.get(3)
    ydimIm = cap.get(4)



    ### get the frame with the proper orientation
    ###---------------------------------------------------
    figure= plt.figure(frameon=False, figsize=(xdimIm/custdpi, ydimIm/custdpi))
    # i = random.randint(0,length)
    # i = 589195
    # print(i)

    # ax=plt.Axes(figure, [0., 0., 1., 1.])
    # ax.set_axis_off()
    Index=i
    cap.set(1, Index)
    ret, frame1= cap.read()
    plt.imshow(frame1)
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    figure.savefig(exportPath+'/original/'+aid+'_'+f"{i:08d}"+'.png', bbox_inches=Bbox([[0.0, 0.0], [xdimIm/custdpi, ydimIm/custdpi]]), pad_inches=0, dpi=custdpi)

    X=np.array(dfwx.iloc[i]) 
    Y=np.array(dfwy.iloc[i])  

    # X = np.array([X[0],X[3],X[2],X[4]])
    # Y = np.array([Y[0],Y[3],Y[2],Y[4]])

    x = np.linspace(np.min(X), np.max(X))
    a,b,c = np.polyfit(X, Y, 2) 
    y = np.mean(curvaturenew(x, a, b, c))

    plt.plot()
    plt.scatter(X,Y, color='red', s=1, alpha=0.5) 
    plt.plot(x, polynomial_fit(x, a, b, c), alpha=0.5, color='red', linewidth=0.5)

    figure.savefig(exportPath+'/label/'+aid+'_'+f"{i:08d}"+'_label.png', bbox_inches=Bbox([[0.0, 0.0], [xdimIm/custdpi, ydimIm/custdpi]]), pad_inches=0, dpi=custdpi)

    plt.close('all')

def getRange(begin, duration):
    a = begin-10
    b = begin+duration+10
    return np.arange(a,b,1)

def getFigureNumbers():
    '''
    function to get the number used for the figure panels

    '''
    f = glob.glob(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+'/**/*.*', recursive = True)

    alln = []   
    for a in f:
        print(a)
        a = a.split(os.sep)[-1].split('.')[0]
        if '_' in a:
            a = a.split('_')[0]
        tmp = ''.join(filter(str.isdigit, a))
        if tmp=='':
            continue
        tmp = int(tmp)
        alln.append(tmp)
    alln = np.unique([int(x) for x in alln])
    nmax = np.max(alln)

def conversionTouchTimingperAnimal(aid):
    '''
    
    '''
    aid = aid.split('_')[0]
    print(aid)
    npyDir = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST"

    ### 
    ### get the actual time from the experiments
    datPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'
    timeFile = glob.glob(datPath+os.sep+'*'+aid+'*'+os.sep+'output/ReferenceTableTransition.csv')[0] ## self.timeFile
    srE = 25000 # sampling rate of the ephys
    srHS = 500 # sampling rate of the Highspeed camera
    dat = pd.read_csv(timeFile) ## self.timeFile
    times = (dat.loc[dat['Row'].isin(['LEDon2', 'LEDoff2']),'timeSlot'].values/srE) # those are the e3 times in seconds

    ###
    ### get the actual led time from the videos previously extracted with 
    ### "Y:\2020-09-Paper-ReactiveTouch\__code__\process1_dataExtraction_touchNew_ULTIMATE(with p1).py"
    frameFile = glob.glob(npyDir + '/**/*'+'LEDframes_*'+aid+'*.npy', recursive=True)[0]
    frames = np.load(frameFile)
    framesTime = frames/srHS

    ### obtained finale frame correction
    frameCorrection =  times - framesTime

    ###
    ### get the actual data from the frames
    combined = []
    for idx, i in enumerate([1,2]):
        print(idx, i)
        tmp = pd.read_csv(glob.glob(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST'+os.sep+aid+'*'+str(i)+'.csv')[0])
        tmp = tmp['FirstTouchFrame'].values
        tmp = tmp/srHS+frameCorrection[idx]
        combined.append(tmp)
    combined = [item for sublist in combined for item in sublist]
    folder = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST\touchTimes"
    np.save(folder+os.sep+aid+'.npy', combined)



#############################################
### VIDEO TOOLS
#############################################

def create_videoTrace(dataFrame, touchType):

    # Create a VideoWriter object
    output_path = 'output_video'+str(touchType)+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 3
    frame_size = (640, 480)
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Create a Matplotlib figure
    # fig, ax = plt.subplots(frameon=False, figsize=(xdimIm / custdpi, ydimIm / custdpi))
    # Create a Matplotlib figure and axis
    # fig, ax = plt.subplots(figsize=(frame_size[0] / 100, frame_size[1] / 100), dpi=100)  # Adjust the figure size
    fig, ax = plt.subplots()
    ax.set_xlim(0, frame_size[0])
    ax.set_ylim(0, frame_size[1])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Remove subplot margins

    # Generate frames and write them to the video
    x = np.array(dataFrame.index.to_list())
    xscale = scale_array(x, 10, 100)

    ### for ploting the accel
    yacc = dataFrame['ang_accel_filt_'].values
    yacc_scale = 480-scale_array(yacc, 240, 320)

    ### for ploting the curvature
    ycurv = dataFrame['filt_curvature'].values
    ycurv_scale = 480-scale_array(ycurv, 120, 200)

    ### for ploting the amplitude
    yamp = dataFrame['inst_amplitude_filt'].values
    yamp_scale = 480-scale_array(yamp, 360, 420)

    for idx, frameOfInterest in enumerate(x):
        ax.clear() # clear the previous axis

        ## get frame from the video part
        cap.set(1, frameOfInterest)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
        frame = frame.astype(np.int64)
        ax.imshow(frame, cmap='gray')
        ax.axis('off')

        ax.text(10, 20, 'Touch cat:  '+touchType, color='white', weight='bold', fontsize=14) # color purple
        ax.text(10, 140, 'amplitude', color='#762a83', weight='bold') # color purple
        ax.plot(xscale[:idx], yamp_scale[:idx], '#762a83', lw=2, label='Line')
        ax.text(10, 260, 'acceleration', color='#1b7837', weight='bold') # color green
        ax.plot(xscale[:idx], yacc_scale[:idx], '#1b7837', lw=2, label='Line')
        ax.text(10, 380, 'curvature', color='#053061', weight='bold') # color blue
        ax.plot(xscale[:idx], ycurv_scale[:idx], '#053061', lw=2, label='Line')

        ## set the axis so that the oriation is proper
        ax.set_xlim([0,640])
        ax.set_ylim([480,0])

        ### draw the canvas and maniputate the image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image = image[:, :, ::-1] ## convert the image back from bgr to rgb
        
        out.write(image)

    # Release the VideoWriter and close the Matplotlib window
    out.release()
    plt.close('all')


from scipy import ndimage as nd
import skimage
import math 

def on_click(event):
    '''
    example usage see below 
    select_points = []
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    '''
    if event.button == 1 and event.inaxes == ax:  # Left mouse button clicked
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

        select_points.append(np.array([x, y]))

        color = colormap[len(select_points) - 1]
        color = tuple(np.array(color) / 255.0)
        ax.plot(x, y, 'o', color=color, markersize=5)
        plt.draw()

def extract_numeric_part(file):
    '''
    function to extract the nummeric part of the string 
    in this case designe for corefiles which have poor indexing values
    '''
    baseName = file.split(os.sep)[-1].split('.')[0].split('_')
    coreIdx = baseName[1]
    return int(coreIdx)

def bbox_scaling(bbox, scaleFactor = 0.2):
    '''
    bbox expension by a factor of 20 percent 
    this is non uniform factor
    '''
    scaleFactor = scaleFactor*max(bbox)


    ymin, xmin, ymax, xmax = bbox
    ymax = int(ymax + scaleFactor)
    xmin = int(xmin - scaleFactor)
    xmax = int(xmax + scaleFactor)

    return  ymin, xmin, ymax, xmax

def filter_mask(mask, min_area_threshold=100):
    '''
    Filter the mask to get rid of small areas
    and filter to make binary to close gaps 
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    filtered_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area_threshold:
            filtered_mask[labels == label] = 255

    closed_mask = nd.binary_closing(filtered_mask, structure=np.ones((10,10)))
    return closed_mask

# class poleLocInit():
#     def __init__(self, file):
#         ## label properties
#         self.file = file
#         self.sID = self.file.split(os.sep)[-2]
#         self.baseName = file.split(os.sep)[-1].split('.')[0].split('_')
#         self.f_baseName = file.split('.')[0]
#         self.coreIdx = self.baseName[1]
#         self.lenCoreIdx = self.coreIdx
#         self.coreDate = self.baseName[-2]
#         self.coreTime = self.baseName[-1]

#         ## video propeties
#         self.cap = cv2.VideoCapture(self.file)
#         self.custdpi = 300
#         self.ydimIm = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.xdimIm = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS)

#         ## 
#         # self.reference

#     def getFrame(self, frameOfInterest = int(3.5*self.fps), plotme=False):
#         '''
#         enable to get the frame of interest
#         '''
#         self.cap.set(1, frameOfInterest)
#         ret, frame = self.cap.read()
#         ## convert to grey scale
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
#         frame = frame.astype(np.int64)
#         frames.append(frame)

#         if plotme == True:
#             figure = plt.figure(frameon=False, figsize=(self.xdimIm / self.custdpi, self.ydimIm / self.custdpi))
#             plt.imshow(frame, cmap='gray')
#             plt.axis('off')
#             plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
#             figure.savefig( self.f_baseName + '_'+ str(frameOfInterest).zfill(len(str(self.length))) + '.png',
#                             bbox_inches=Bbox([[0.0, 0.0], [self.xdimIm / self.custdpi, self.ydimIm / self.custdpi]]), pad_inches=0,
#                             dpi=self.custdpi)
#             plt.close('all')

#         return frame
    
#     def getFrame_avg(self):
#         '''
#         enables to get an average value for the frame 
#         '''
#         lrange = int(3.5*475)
#         hrange = 2500
#         allFrames = []
#         for i in range(lrange,hrange):
#             try:
#                 frame = self.getFrame(frameOfInterest=i)
#                 allFrames.append(frame)
#             except:
#                 continue
#         allFrames_mean = np.mean(allFrames, axis=0).astype(int)

#         return allFrames_mean

#     def getPoleInfo(self, refFrame=False):
#         '''
#         Function that work partially on the segmentation the creation of the masks etc.
#         ### first get the postion of the pole 

#         ###  create a mask for segementation for some resources on that see
#         # https://www.youtube.com/watch?v=4hPl7GMnz5I&t=2s
#         # https://github.com/bnsreenu/python_for_microscopists
#         '''
#         if refFrame==True:
#             frame = self.getFrame(frameOfInterest=0)
#         else:
#             frame = self.getFrame()
#         # figure = plt.figure(frameon=False, figsize=(self.xdimIm / self.custdpi, self.ydimIm / self.custdpi))
#         # plt.imshow(frame)
#         low_val = np.min(frame)
#         mask = cv2.inRange(frame, 9, 100)
#         closed_mask = nd.binary_closing(mask, structure=np.ones((10,10))) # the code below enables to close the segmented objects if there are holes present
#         label_image = skimage.measure.label(closed_mask)
#         props = skimage.measure.regionprops_table(label_image, subtracted_image,
#         properties=['label',
#                     'centroid',
#                     'area', 'equivalent_diameter',
#                     'mean_intensity', 'solidity'])
#         # centroid-0 correspond to the y axis
#         # centroid-1 correspond to the x axis
#         df = pd.DataFrame(props)
#         df = df[df['area']>500] ### <<<-- this is to get rid of meaningless segmentation
        

#         ########### section to get the specific pole postion
#         ## note that there is a cv2 alternative to this  # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
#         p_label = df.loc[df['area']==np.min(df['area']),'label'].item()
#         target_mask = (label_image == p_label)
#         contours = skimage.measure.find_contours(target_mask, 0.5)
#         # plt.figure(figsize=(8, 8))
#         # plt.imshow(label_image, cmap='rainbow')
#         # for contour in contours:
#         #     plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
#         contours = contours[0] # in this case that works since all contours are adjacent
#         # plt.plot(contour[:, 1], contour[:, 0], '.', color='red')
#         # plt.xlim([0,self.xdimIm])
#         # plt.ylim([self.ydimIm,0])

    
#         ##### find the bottom of the pole
#         # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
#         target_mask = np.uint8(target_mask)*255
#         regions = skimage.measure.regionprops(target_mask)[0]
#         # y0, x0 = regions.centroid
#         # orientation = regions.orientation

#         ## get the bounding box around the pole
#         minr, minc, maxr, maxc = regions.bbox ## equivalent to miny, minx, maxy, maxx
#         bx = (minc, maxc, maxc, minc, minc)
#         by = (minr, minr, maxr, maxr, minr)
#         # plt.plot(bx, by, '-b', linewidth=2.5)

#         bottomPole = contours[np.where(contours[:, 0] == np.max(contours[:, 0]))[0]]
#         # plt.plot(bottomPole[:, 1], bottomPole[:, 0], '.', color='blue')


#         return df, contours, regions

#     def getGonoGo(self):
#         '''
#         On Go trials, the pole was positioned in a posterior position (3 mm from home) and lifted into the whisker range by a pneumatic linear slide (SLS-10-15-P-A, Festo) attached to the linear actuator. On NoGo trials, the pole was moved to an anterior position (3 mm from home) and lifted. Therefore, the offset of the Go and NoGo was 6 mm,
#         '''
#         ref_df, _, _ = self.getPoleInfo(refFrame=True)
#         curr_df, contours, regions = self.getPoleInfo()
#         curr_dfout = copy.copy(curr_df.loc[curr_df['area']==np.min(curr_df['area'])])
#         # print(curr_df)

#         if len(curr_df)!=2:
#             g_value = np.nan
#         else:
#             curr_ypos = curr_df.loc[curr_df['area']==np.min(curr_df['area']),'centroid-0'].item()
#             ref_ypos = ref_df.loc[curr_df['area']==np.min(curr_df['area']),'centroid-0'].item()
#             if curr_ypos<ref_ypos:
#                 g_value = 'NoGo'
#                 g_bin = 0
#             elif curr_ypos>ref_ypos:
#                 g_value = 'Go'
#                 g_bin = 1
#         curr_dfout['goNoGo_bin'] = g_bin
#         curr_dfout['goNoGo'] = g_value
#         curr_dfout['bbox'] = [regions.bbox]
#         curr_dfout['contours'] = [contours]
#         curr_dfout['trial'] = self.coreIdx
        
#         return curr_dfout, contours

def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """
  ## usage colormap = viz_utils.get_colors(20)
  borrowed from https://github.com/deepmind/tapnet/blob/main/utils/viz_utils.py
  Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors

def maskFrame(my_image, select_points):
    """
    This function returns a masked image that can be used and be more practicle
    than the current image 
    my_image: np.array of an image
    select_points: np.array of polygon 
    """
    ### create a mask based on the polygon 
    mask = np.zeros_like(my_image)
    cv2.fillPoly(mask, [select_points], (255, 255, 255))
    masked_image = cv2.bitwise_and(my_image, mask) # Apply the mask to the original image

    return masked_image


################################################################################################################################################################################################################
################################################################################################################################################################################################################
######  EPM ANALYSIS ################################################################################################################################################################################################
################################################################################################################################################################################################################
################################################################################################################################################################################################################

'''
Notes and metrics based on the camera position for S1WDIL
Mouse full length is roughly 70 px 
The EPM maze as a width of 68mm which correspond to 78 pixels
convFactor to mm is
convFactor_mm = 68/78
convFactor_m = convFactor_mm/1000
'''

import keyboard
from shapely.geometry import Point, Polygon

vidFiles = glob.glob(r'C:\Users\Windows\Desktop\S1WDIL_EPM\*.mp4')

class poleLocInit():
    def __init__(self, file):
        ## label properties
        self.file = file
        self.sID = self.file.split(os.sep)[-1].split('_')[0]
        self.baseName = self.file.split(os.sep)[-1].split('.')[0].split('_')
        self.dirName = os.path.dirname(self.file)
        self.f_baseName = self.file.split('.')[0]
        self.coreIdx = self.baseName[1]
        self.lenCoreIdx = self.coreIdx
        self.coreDate = self.baseName[-2]
        self.coreTime = self.baseName[-1]
        self.timeofAssay = 5*60 # 5 min in seconds
        self.h5file = glob.glob(os.path.dirname(self.file)+os.sep+'*'+self.sID+'*1000.h5')[0]

        ## video propeties
        self.cap = cv2.VideoCapture(self.file)
        self.custdpi = 300
        self.ydimIm = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.xdimIm = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        ## 
        # self.reference
        self.arena = np.load(os.path.dirname(self.file)+os.sep+self.sID+'_empPolygon.npy')
        self.arena_rois = {
        # based on the location of the epm for the viewer looking at the video footage
            'OA_up': [self.arena[0], self.arena[1], self.arena[2], self.arena[11]],
            'CA_right': [self.arena[2], self.arena[3], self.arena[4], self.arena[5]],
            'OA_down': [self.arena[5], self.arena[6], self.arena[7], self.arena[8]],
            'CA_left':[self.arena[8], self.arena[9], self.arena[10], self.arena[11]],
            'center': [self.arena[11], self.arena[2], self.arena[5], self.arena[8]]
        }
        self.arena_rois_poly = {name: Polygon(coords) for name, coords in self.arena_rois.items()}

    def frameRef(self, frameOfInterest=1):
        self.cap.set(1, frameOfInterest)
        ret, frame = self.cap.read()
        ## convert to grey scale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
        frame = frame.astype(np.int64)

        return frame

    def getFrame(self, frameOfInterest=1, plotme=False):
        '''
        enable to get the frame of interest
        '''
        centerPoints = self.arena

        self.cap.set(1, frameOfInterest)
        ret, frame = self.cap.read()
        ## convert to grey scale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # added conversion to grayscale
        frame = frame.astype(np.int64)

        if plotme == True:
            figure = plt.figure(frameon=False, figsize=(self.xdimIm / self.custdpi, self.ydimIm / self.custdpi))
            plt.imshow(frame, cmap='gray')
            # plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.plot(centerPoints[:,0], centerPoints[:,1], '.', color = 'red')
            figure.savefig( self.f_baseName + '_'+ str(frameOfInterest).zfill(len(str(self.length))) + '_zones.png',
                            bbox_inches=Bbox([[0.0, 0.0], [self.xdimIm / self.custdpi, self.ydimIm / self.custdpi]]), pad_inches=0,
                            dpi=self.custdpi)
            # plt.close('all')
            # plt.show()

        return frame
    
    def getFrame_avg(self, numberofFrames=50):
        '''
        enables to get an average value for the frame 
        '''
        hbd = self.length
        lbd = hbd-self.timeofAssay*self.fps
        rdmFrames = np.random.randint(lbd, hbd, 50)

        allFrames = []
        for i in rdmFrames:
            try:
                frame = self.getFrame(frameOfInterest=i)
                allFrames.append(frame)
            except:
                continue
        allFrames_mean = np.mean(allFrames, axis=0).astype(int)

        return allFrames_mean

    def getPoleInfo(self, refFrame=False):
        '''
        Function that work partially on the segmentation the creation of the masks etc.
        ### first get the postion of the pole 

        ###  create a mask for segementation for some resources on that see
        # https://www.youtube.com/watch?v=4hPl7GMnz5I&t=2s
        # https://github.com/bnsreenu/python_for_microscopists
        '''
        if refFrame==True:
            frame = self.getFrame(frameOfInterest=0)
        else:
            frame = self.getFrame()
        # figure = plt.figure(frameon=False, figsize=(self.xdimIm / self.custdpi, self.ydimIm / self.custdpi))
        # plt.imshow(frame)
        low_val = np.min(frame)
        mask = cv2.inRange(frame, 9, 100)
        closed_mask = nd.binary_closing(mask, structure=np.ones((10,10))) # the code below enables to close the segmented objects if there are holes present
        label_image = skimage.measure.label(closed_mask)
        props = skimage.measure.regionprops_table(label_image, subtracted_image,
        properties=['label',
                    'centroid',
                    'area', 'equivalent_diameter',
                    'mean_intensity', 'solidity'])
        # centroid-0 correspond to the y axis
        # centroid-1 correspond to the x axis
        df = pd.DataFrame(props)
        df = df[df['area']>500] ### <<<-- this is to get rid of meaningless segmentation
        

        ########### section to get the specific pole postion
        ## note that there is a cv2 alternative to this  # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        p_label = df.loc[df['area']==np.min(df['area']),'label'].item()
        target_mask = (label_image == p_label)
        contours = skimage.measure.find_contours(target_mask, 0.5)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(label_image, cmap='rainbow')
        # for contour in contours:
        #     plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
        contours = contours[0] # in this case that works since all contours are adjacent
        # plt.plot(contour[:, 1], contour[:, 0], '.', color='red')
        # plt.xlim([0,self.xdimIm])
        # plt.ylim([self.ydimIm,0])

    
        ##### find the bottom of the pole
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
        target_mask = np.uint8(target_mask)*255
        regions = skimage.measure.regionprops(target_mask)[0]
        # y0, x0 = regions.centroid
        # orientation = regions.orientation

        ## get the bounding box around the pole
        minr, minc, maxr, maxc = regions.bbox ## equivalent to miny, minx, maxy, maxx
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        # plt.plot(bx, by, '-b', linewidth=2.5)

        bottomPole = contours[np.where(contours[:, 0] == np.max(contours[:, 0]))[0]]
        # plt.plot(bottomPole[:, 1], bottomPole[:, 0], '.', color='blue')


        return df, contours, regions

    def getGonoGo(self):
        '''
        On Go trials, the pole was positioned in a posterior position (3 mm from home) and lifted into the whisker range by a pneumatic linear slide (SLS-10-15-P-A, Festo) attached to the linear actuator. On NoGo trials, the pole was moved to an anterior position (3 mm from home) and lifted. Therefore, the offset of the Go and NoGo was 6 mm,
        '''
        ref_df, _, _ = self.getPoleInfo(refFrame=True)
        curr_df, contours, regions = self.getPoleInfo()
        curr_dfout = copy.copy(curr_df.loc[curr_df['area']==np.min(curr_df['area'])])
        # print(curr_df)

        if len(curr_df)!=2:
            g_value = np.nan
        else:
            curr_ypos = curr_df.loc[curr_df['area']==np.min(curr_df['area']),'centroid-0'].item()
            ref_ypos = ref_df.loc[curr_df['area']==np.min(curr_df['area']),'centroid-0'].item()
            if curr_ypos<ref_ypos:
                g_value = 'NoGo'
                g_bin = 0
            elif curr_ypos>ref_ypos:
                g_value = 'Go'
                g_bin = 1
        curr_dfout['goNoGo_bin'] = g_bin
        curr_dfout['goNoGo'] = g_value
        curr_dfout['bbox'] = [regions.bbox]
        curr_dfout['contours'] = [contours]
        curr_dfout['trial'] = self.coreIdx
        
        return curr_dfout, contours

def polygons_containing_point(df, bodypart, arena_rois):
    '''
    Args:
    -----------
        df (str): pd.DataFrame
        bodypart: bodypart of interest

    Retruns:
    -----------
    
    
    Examples:
    -----------
        df['polygons_containing_point'] = df.apply(polygons_containing_point, axis=1)

    '''
    point = Point(df[bodypart]['x'], df[bodypart]['y'])
    containing_polygons = [name for name, poly in arena_rois.items() if poly.contains(point)]
    return containing_polygons

def read_dlcHdf(filePath, flatten=False):
    ''' 
    function to flatten the hdf files output from DLC
    
    Args:
    -----------
        filePath (str): file path

    Retruns:
    -----------
        df (pd.DataFrame): data frame flat or mulitindex
        unique_levels (list): return the level of the multiindex
    
    Examples:
    -----------
        filePath = "C:/Users/Windows/Desktop/DLCattempt2/camDLC_snapshot-1000.h5"
        dat = read_dlcHdf(filePath, flatten=True)
        dat, unique_levels, scorer = read_dlcHdf(filePath, flatten=False)
    '''

    vname = filePath.split(os.sep)[-1].split('DLC')[0]
    scorer = filePath.split(os.sep)[-1].split('.h5')[0].split(vname)[-1]
    if '_filtered' in scorer:
        scorer = scorer.split('_filtered')[0]
    df = pd.read_hdf(filePath, "df_with_missing")
    df = df[scorer]
    unique_levels = list(df.columns.get_level_values(0).unique()) # get the unique l

    return df, unique_levels, scorer

    if flatten == True:
        # drop the multi index and change the column names
        df.columns = [''.join(col) for col in df.columns]
        # reset row index
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'frame'}, inplace=True)

        return df

def on_key(event):
    if event.key == 'l':
        # Save the data_array to a file (e.g., CSV or NumPy binary file)
        np.save(r'C:\Users\Windows\Desktop'+os.sep+a.sID+'_empPolygon.npy', select_points)
        plt.close()
        print("Data saved to 'data.npy'")

def labeltheArena(vidFiles):
    plt.ioff()
    for i in vidFiles:
        print(i)
        a = poleLocInit(i)
        my_image = a.getFrame()

        fig, ax = plt.subplots()
        ax.imshow(my_image)
        select_points = []
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1920, 23, 1920, 1017)
        plt.show()






    select_points = np.array([[x, y] for x, y in select_points]) ## format the array properly 
    np.save(r'C:\Users\Windows\Desktop\ezTrack-test\empPolygon.npy', select_points)

def addColortoColormap(colormap='jet', colorToAdd=[256,256,256], categorical=False):
    '''
    colorToAdd: RGB value of the color to be added
    color map manip to add a color to an existingn colormap
    ## colormap manipulation 
    # https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.colors as mcolors

    if categorical == True:
        viridis = mpl.colormaps['jet'].resampled(256) ## get the list of colors 
        viridis.colors
    else:
        mcmap = plt.get_cmap('jet')
        gradient = np.linspace(0, 1, 256)
        rgba_values = mcmap(gradient)
        # rgb_values = (rgba_values[:, :3] * 255).astype(int) # Extract the RGB values from the RGBA values
        newcmap = copy.copy(rgba_values)
        newcmap[:1, :] = np.array([colorToAdd[0]/256, colorToAdd[1]/256, colorToAdd[2]/256, 1])
        newcmap = ListedColormap(newcmap)

    return newcmap



    hm.set_clim(1, 100)
    plt.colorbar(label='Value')
    
    map_i = hv.Image((np.arange(heatmap.shape[1]), np.arange(heatmap.shape[0]), heatmap))
    map_i.opts(width=int(heatmap.shape[1]*video_dict['stretch']['width']),
           height=int(heatmap.shape[0]*video_dict['stretch']['height']),
           invert_yaxis=True, cmap='jet', alpha=1,
           colorbar=False, toolbar='below', title="Heatmap")
    
    return map_i

def heatmap(df, ax, bodypart='mouse_center', sigma=10, plotme = False):
 

    heatmap = np.zeros([self.ydimIm, self.xdimIm])+255
    for idx, i in df.iterrows():
        Y,X = int(i[bodypart]['y']), int(i[bodypart]['x'])
        heatmap[Y,X]+=1
    
    sigma = np.mean(heatmap.shape)*0.05 if sigma == None else sigma
    heatmap1 = cv2.GaussianBlur(heatmap,(0,0),sigma)
    heatmap2 = (heatmap1 / heatmap1.max())

    if plotme == True:
        palette = copy.copy(plt.get_cmap('jet'))
        palette.set_under('white', 0.0)  # 1.0 represents not transparent
        newcmap = addColortoColormap('jet')
        ax.imshow(heatmap2, cmap=newcmap, origin='lower', aspect='auto')

    return heatmap2
    ## alternative 
    # plt.figure()
    # plt.imshow(heatmap1, cmap=newcmap, interpolation = 'nearest', aspect='auto')

def outputArena(vidFiles):
    plt.ion()
    for i in vidFiles:
        print(i)
        a = poleLocInit(i)
        a.getFrame(1500, True)

def getThegeno():
    tmp = pd.DataFrame({'sID':[os.path.basename(x).split('_')[0] for x in vidFiles]})
    # geno = pd.read_csv(r"C:\Users\Windows\Desktop\S1WDIL_EPM\geno.csv")
    geno['sID'] = geno['sID'].astype(str)
    tmp = pd.merge(tmp, geno, on='sID')
    tmp = tmp.sort_values('geno', ascending=False)
    return tmp

def quickConversion(tmp, myCol=None, option=1):
    ''' convert groupby pandas table into a simpler version
    '''
    if option ==1:
        tmp.columns = ['value']
        tmp = tmp.reset_index()
        if tmp.columns.nlevels > 1:
            tmp.columns = ['_'.join(col) for col in tmp.columns] 
        tmp.columns = tmp.columns.str.replace('[_]','',regex=True)
        if myCol:
            tmp = tmp.rename(columns={np.nan: myCol})
        return tmp
    elif option ==2:
        tmp.columns = ['_'.join(col) for col in tmp.columns] 
        tmp = tmp.reset_index()
        return tmp

def distanceScale(val):
    convFactor_mm = 68/78
    convFactor_m = convFactor_mm/1000
    scaledVal = val*convFactor_m

    return scaledVal

###------------------------------------------------------------------------
### THIS IS TO GENERATE HEATMAPS
###------------------------------------------------------------------------
# tmp = getThegeno()
# for k in ['mouse_center', 'nose', 'mid_back', 'tail_base']:
#     fig, ax = plt.subplots(4,4)
#     for i,j in tqdm.tqdm(zip(ax.flatten(), tmp.iterrows())):
#         print(i, j)
#         vidFile = glob.glob(r'C:\Users\Windows\Desktop\S1WDIL_EPM'+os.sep+j[1]['sID']+'*.mp4')[0]
#         tmpDat = poleLocInit(vidFile)
#         dfAll = read_dlcHdf(tmpDat.h5file)
#         df = dfAll[0]
#         startPoint = len(df) - 300*60
#         df = df[startPoint:]
#         heatmap(df=df, ax=i, bodypart=k, plotme = True)
#         i.axis('off')
#         i.text(0,0,j[1]['geno']+' - '+j[1]['sID'])

#     plt.suptitle(k)
#     plt.savefig(r'C:\Users\Windows\Desktop\S1WDIL_EPM\heatmap_'+k+'.png')

def getEPMoutput():
    ###------------------------------------------------------------------------
    ### THIS IS TO GENERATE THE OUTPUTS
    ###------------------------------------------------------------------------
    tmp = getThegeno()
    bodypart = 'mouse_center'
    globalSummary = []
    for j in tqdm.tqdm(tmp.iterrows()):
        # print(j)
        vidFile = glob.glob(r'C:\Users\Windows\Desktop\S1WDIL_EPM'+os.sep+j[1]['sID']+'*.mp4')[0]
        tmpDat = poleLocInit(vidFile)
        dfAll = read_dlcHdf(tmpDat.h5file)
        df = dfAll[0]
        startPoint = len(df) - 300*60
        df = df[startPoint:]

        ### key to define where the mouse is
        df = copy.copy(df)
        df['mouse_in'] = df.apply(polygons_containing_point, args=(bodypart, tmpDat.arena_rois_poly), axis=1)

        ### get the distance traveled 
        temp = pd.DataFrame(list(((df[bodypart]['x'] - df[bodypart]['x'].shift()) ** 2 + (
                    df[bodypart]['y'] - df[bodypart]['y'].shift()) ** 2) ** 0.5),
                    columns=[bodypart])
        temp[temp>50] = 0 ## need to improve on that just to refine 
        df['distance'] = temp.values
        df['cumulative_distance'] = df['distance'].cumsum()

        ### get the transition zone
        df['transition'] = df['mouse_in'].shift() != df['mouse_in']

        ### need to write the data
        dfouput = df[['mouse_in', 'distance', 'cumulative_distance', 'transition']]
        dfouput = dfouput.reset_index()
        dfouput.columns = [''.join(col) for col in dfouput.columns] ## flatten the multi index columns
        dfouput = dfouput.explode('mouse_in') ## polygon are stored as a list so need to be exploded to perform summary
        dfouput['mouse_in'].fillna('missTrack', inplace=True)
        fileNameOutput = tmpDat.dirName+os.sep+tmpDat.sID+'_ouput_full.csv'
        dfouput.to_csv(fileNameOutput)
        # df.to_hdf(fileName, key="df_with_missing", mode="w") 


    ###------------------------------------------------------------------------
    ### THIS IS TO GENERATE THE SUMMARY
    ###------------------------------------------------------------------------

        ## calculate the time spent
        timeSpent = quickConversion(dfouput.groupby(['mouse_in']).agg({'mouse_in': [np.ma.count]}))
        timeSpent.rename(columns={'value':'time_frame'}, inplace=True)
        timeSpent['time_s'] = timeSpent['time_frame']/60
        timeSpent['percent_time'] = timeSpent['time_s']/300

        ## calculate the distance moved 
        distMoved = quickConversion(dfouput.groupby(['mouse_in']).agg({'distance': [np.sum]}))
        distMoved.rename(columns={'value':'dist_px'}, inplace=True)
        distMoved['dist_m'] = distMoved['dist_px'].apply(distanceScale)

        ## calculate the transition area
        dfoutputTransition = dfouput[dfouput['transition']==True]
        transition = quickConversion(dfoutputTransition.groupby(['mouse_in']).agg({'transition': [np.ma.count]}))
        transition.rename(columns={'value':'transition_n'}, inplace=True)

        summary = pd.merge(timeSpent, distMoved, on = 'mousein')
        summary = pd.merge(summary, transition, on='mousein')
        summary['mainCat'] = summary['mousein'].str.split('_').str[0]
        summary['sID'] = tmpDat.sID
        melted_summary = summary.melt(id_vars=['mousein','sID', 'mainCat'], var_name='variable', value_name='value')
        # pivot_df = melted_df.pivot(index='variable', columns='mousein', values='value')
        fileNameOutputSummary = tmpDat.dirName+os.sep+tmpDat.sID+'_ouput_summary.csv'
        melted_df.to_csv(fileNameOutputSummary)

        globalSummary.append(melted_summary)

    globalSummary = pd.concat(globalSummary)
    fileNameGlobalSummary = tmpDat.dirName+os.sep+tmpDat.sID+'_globalSummary.csv'
    globalSummary.to_csv(fileNameGlobalSummary)
    return globalSummary 
