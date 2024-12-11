# work on whiskers 
## see original script also from Clack in python 
## https://github.com/nclack/whisk/blob/d938b9d8dc332d0494cbced6ae844c0ff8250b46/python/features.py

## ***************************************************************************
## * LIBRARIES                                                               *
## ***************************************************************************

import copy
import matplotlib.pyplot as plt
import matplotlib
import glob
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert
import time
import cv2
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import tqdm

plt.ion()


## ***************************************************************************
## * CUSTOM FCT                                                              *
## ***************************************************************************

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

def groupSequence(lst): 

    '''

    function to enable to find the maximum number of consecutive frame present

    this function can be used to be able to set interval during Median Filter

    '''

    res = [[lst[0]]] 

    for i in range(1, len(lst)): 

        if lst[i-1]+1 == lst[i]: 

            res[-1].append(lst[i]) 

        else: 

            res.append([lst[i]]) 

    res = len(max(res))

    return res 
    
def custMedianFilterSpikeNoise(angleData):

    '''

    create a median filter that that remove spike noise

    '''

    spikeNoise=np.where((angleData<=0) | (angleData>=170))[0] # filter based on DLC likelihood statistics

    signalOri=list(angleData)

    if len(spikeNoise) == 0:

       return signalOri

    else:

        filtsig=list(angleData) # create a duplicate of the signal that will be filter

        n=len(filtsig)

        k= groupSequence(list(spikeNoise))+50

        for i in range(0, len(spikeNoise)):

            lowbnd= np.max((1, spikeNoise[i]-k))

            uppbnd= np.min((spikeNoise[i]+k, n))

            filtsig[spikeNoise[i]]=np.median(signalOri[lowbnd:uppbnd])

        return filtsig

def whiskingEpoch(medianFilterSignal):

    '''

    process the signal angle and band-pass filter between 6-60Hz with a 4th order Butterworth filter

    see Svoboda publication: 10.1016/j.neuron.2019.07.027

    '''
    if not isinstance(medianFilterSignal, (np.ndarray)):
        medianFilterSignal = np.array(medianFilterSignal)

    medianFilterSignal = np.nan_to_num(medianFilterSignal)

    bandPassCutOffsInHz = [6,60]
    sampleRate = 500
    W1 = bandPassCutOffsInHz[0] / (sampleRate/2)
    W2 = bandPassCutOffsInHz[1] / (sampleRate/2)
    passband = [W1, W2]
    b, a = butter(4, passband, 'bandpass')
    anglefilt = filtfilt(b, a, medianFilterSignal)

    ## obtaining the amplitude derive from hilbert transform of the filtered signal
    # z=abs(hilbert(anglefilt[0:3500000])) ## unknow error in some arrays if not split transfromation stall
    # zz=abs(hilbert(anglefilt[3500000:]))
    # z=list(z)+list(zz)

    ## AMPLITUDE 
    ## ****************************

    hilb = hilbert(anglefilt)
    inst_amplitude = abs(hilb)
    dfAmp=pd.DataFrame({'amp':inst_amplitude})
    inst_amplitude[np.where(inst_amplitude<=2)]=0
    inst_amplitude[np.where(inst_amplitude>=2)]=1
    dfAmp['cat']=inst_amplitude
    dfAmp['grp']=(dfAmp.cat.diff(1) != 0).astype('int').cumsum()
    dfAmp['frame']=dfAmp.index

    ## PHASE 
    ## ****************************




    summary=pd.DataFrame({'whiskCount' : dfAmp.groupby(['cat','grp'])['cat'].count(),
                      'FirstwhiskFrame' : dfAmp.groupby(['cat','grp'])['frame'].first()})
    summary.reset_index(inplace=True)
    summary=summary.sort_values(by=['grp'])
    summary.reset_index(inplace=True)
    summary=summary.assign(interEventinter=summary.FirstwhiskFrame.shift(-1)-(summary.FirstwhiskFrame+summary.whiskCount))

    whiskCycleLim=0.25*sampleRate #0.5 correspond to 500 ms between whisk 500 is the frame rate at which video are acquired
    whiskCycleStopTmp=list(summary.loc[(summary['whiskCount']>=whiskCycleLim) & (summary['cat']==0),'grp'])
    whiskCycleStartTmp=[i + 1 for i in whiskCycleStopTmp][0:-1]
    whiskCycleStop=list(summary.loc[summary['grp'].isin(whiskCycleStopTmp),'FirstwhiskFrame'][1:])
    whiskCycleStart=list(summary.loc[summary['grp'].isin(whiskCycleStartTmp),'FirstwhiskFrame'])
    whiskCycleOutput=pd.DataFrame({'start': whiskCycleStart,
                                    'stop': whiskCycleStop})
    whiskCycleOutput['whiskOnFrame']=whiskCycleOutput['stop']-whiskCycleOutput['start']
    whiskCycleOutput['whiskOffFrame']=whiskCycleOutput['start'].shift(-1)-whiskCycleOutput['stop']

    return anglefilt, hilb, whiskCycleOutput, dfAmp

def curvaturenew(x):
    return (2*a)/pow((1+(2*a*x+b)*(2*a*x+b)),(3/2)) #signed polynomila
    # return abs(2*a)/pow((1+(2*a*x+b)*(2*a*x+b)),(3/2)) #to have unsigned

def extractWhisckerXY(data):
    '''
    Functions to extract the the specific columns require to perform curvature analysis extraction of x and y columns
    '''
    # determine the column of interest to keep 
    dfwy = []
    dfwx = []
    for j in range(0,5):
        tmp = 'w'+str(j)+'y'
        # print(tmp)
        dfwy.append(tmp)

    for j in range(0,5):
        tmp = 'w'+str(j)+'x'
        # print(tmp)
        dfwx.append(tmp)

    dfwx = data[dfwx]
    dfwy = data[dfwy]

    return dfwx, dfwy

    ## previous strategy to filter out columns not working anymore in the new labled
    # dfw=dt.filter(regex=r'w', axis=1)
    # dfwx=dfw.filter(regex=r'x', axis=1)
    # dfwy=dfw.filter(regex=r'y', axis=1)

def determineAngle(data):
    '''
    Function to determint the angle of the whiskers. Pretty static given standard output of the DLC file can be chnaged
    '''

    data=data.assign(w3xO=data.w3x-data.w0x, w3yO=data.w3y-data.w0y)
    data=data.assign(angle= (np.arctan2(data.w3yO.shift(),data.w3xO.shift())-np.arctan2(data.w3yO,data.w3xO)))
    data=data.assign(angleWhiskSoft= (np.arctan2(480,0)-np.arctan2(data.w3yO,data.w3xO)))
    data['angle'] = data['angle']* 360/(2*np.pi)
    data['angleWhiskSoft'] = data['angleWhiskSoft']* 360/(2*np.pi)

    return data

# new updated methods

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

def whiskAmplitudePhase(anglefilt):
    '''
    data from angle filter
    '''
    hilb = hilbert(anglefilt)
    inst_amplitude = abs(hilb)
    phase = np.angle(hilb)

    return inst_amplitude, phase
    
def whiskPhaseZeroDetect(df, phaseCol, instAmpCol, degreeThreshold = 2.5):
    
    '''
    function to determine amplitude and phase based on Svoboda description
    at phase 0 increase of 2.5 degree can mark the onset of whisking amplitude
    df: pd.DataFrame from DLC
    phaseCol: pd.Series in df that contains the computed phase from hilbert transform
    instAmpCol: pd.Series corresponding to instantaneous amplitdue not filter
    degreeThreshold = 2.5 set to 2..5 degree see neuron paper ref above could be change also rather than pure 2.5 cut off

    '''


    # determine at which level of the phase the whisker phase is 0
    mysignal = np.array(df[phaseCol])
    phaseZero = mysignal *0
    phaseZero[np.argwhere(np.diff(np.sign(phaseZero - mysignal))).flatten()+1] = 1 # where phase signal cross 0
    df['phaseZero'] = phaseZero

    # determine at which level of the phase whisker phase is 0 and the amplitude of whisking is more than 2.5 degree
    df['phaseAmpCat'] = 0
    df.loc[(df['phaseZero'] == 1) & (df['inst_amplitude'] >= degreeThreshold), 'phaseAmpCat'] = 1
    df['phaseAmpGrp'] = (df['phaseAmpCat'].diff(1) != 0).astype('int').cumsum()

    return df

def fromZero(df, val, usepeak = True):
    '''
    function to be used to correct for the shift that occurs with defining whisking onset at 2.5
    '''
    x = (np.array(df.loc[df['frame'] < val, 'inst_amplitude'].rolling(window=20).mean())*-1)[::-1][0:500]
    if usepeak == True:
        peaks, _ = find_peaks(x, distance=2)
        # plt.figure()
        # plt.plot(x)
        # plt.plot(peaks, x[peaks], "x")
        xMax = peaks[0]

    else:
        xMax = (np.array(df.loc[df['frame'] < val, 'inst_amplitude'].rolling(window=20).mean())*-1)[::-1][0:100].max()
        xMax = list(np.where(x == xMax))[0][0]

    return xMax

def whiskGetKeyOnset(df, sampleRate = 500, whiskCycleLim = 0.25, whiskLengthlim = 0):
    '''
    df: pd.DataFrame on which other whisking measure are appended
    sampleRate: data acquisition rate of the video fps usually 500 for highspeed
    whiskCycleLim: default is 250 ms expressed in seconds 0.25 every whisk with high amplitude during this phase will be concatenated in the same category thus whisking onset separated by 250 ms or less are define as the same whisk
    whiskLengthlim: whiskk lasting 150 ms or more are included otherwise discareded
    '''

    whiskCycleLim = whiskCycleLim * sampleRate #conversion to frame number # to determine the whisk cycle or lasting whisk events only events lasting more than 250 ms are retained
    whiskLengthlim = whiskLengthlim * sampleRate #conversion to frame number

    dfSubset = df[['frame', 'phaseAmpCat', 'phaseAmpGrp']]
    dfSubset = dfSubset[dfSubset['phaseAmpCat'] == 1]
    dfSubset['Interval'] = dfSubset['frame'].shift(-1)-dfSubset['frame']
    dfSubset.reset_index(inplace=True, drop=True)
    # toTry = toTry[toTry['Interval']>=whiskCycleLim]


    # s = time.time()
    # this part need to be optmized should use apply and the use of a function
    # dfSubset['trueWhisk'] = np.nan
    for i,j in enumerate(dfSubset['phaseAmpGrp']):
        if dfSubset['Interval'][i] < whiskCycleLim:
            dfSubset['phaseAmpGrp'][i+1] = dfSubset['phaseAmpGrp'][i]

    # e = time.time()  -s


    summary = pd.DataFrame({'LastwhiskFrame' : dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].last(),
                        'FirstwhiskFrame': dfSubset.groupby(['phaseAmpGrp','phaseAmpCat'])['frame'].first()})
    summary.reset_index(inplace=True)
    summary['whiskDuration'] = summary['LastwhiskFrame'] -  summary['FirstwhiskFrame']
    summary['noWhiskbefore'] = [summary['FirstwhiskFrame'][0]] + (summary['FirstwhiskFrame'].shift(-1) - summary['LastwhiskFrame']).dropna().tolist()
    summary = summary[summary['whiskDuration'] > whiskLengthlim]
    summary.reset_index(inplace = True, drop = True)
    summary['whiskGrp'] = summary.index
    # is section is enabling to get read of the slight delay as it is originally determined by 2.5 changes in degree and at phase 0
    summary['FirstwhiskFrameFromZero'] = np.nan
    for SumVal in summary['FirstwhiskFrame']:
        if SumVal<500:
            pass
        else:
            xMax = fromZero(df, SumVal)
            tmpVal = SumVal-xMax
            summary.loc[summary['FirstwhiskFrame'] == SumVal, 'FirstwhiskFrameFromZero'] = tmpVal

    return summary

def whiskSummaryFunction(h5File):

    ## read the file from dlc
    ## r"C:\Users\Windows\Desktop\### WISKER ###\WhiskForTom\whiskTest\Test20201015DLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5"
    if type(h5File) == str:
        df = readHdf(h5File)
    else:
        df = h5File

    ## step0 computer angle and add to data frame and filter
    df = determineAngle(df)
    df['angleWhiskSoftFilt'] = custMedianFilterSpikeNoise(df['angleWhiskSoft'])
    df['angleWhiskSoftFilt'] = whiskAngleFilter(df['angleWhiskSoft'])

    ## step1 compute phase and amplitdue
    df['inst_amplitude'], df['phase'] = whiskAmplitudePhase(df['angleWhiskSoftFilt'])

    ## get intersection the angle at phase 0
    df = whiskPhaseZeroDetect(df, phaseCol = 'phase', instAmpCol = 'inst_amplitude', degreeThreshold = 2.5)

    ## get the key information on the whisk 
    summary = whiskGetKeyOnset(df, sampleRate = 500, whiskCycleLim = 0.25, whiskLengthlim = 0)

    return df, summary


#### GRAPHING SUBSET
def whiskPlottingDetectCheck(df, summary, whiskOnset = 'FirstwhiskFrame'):

    f, (ax1) = plt.subplots(1, 1, sharex=True, constrained_layout=True, figsize=([20.74, 4.8]))
    import matplotlib
    ax1.plot(df['inst_amplitude'])
    for i in summary.index:
        rect = matplotlib.patches.Rectangle((summary[whiskOnset][i], 0), summary['whiskDuration'][i], max(df['inst_amplitude']), color ='red', alpha = 0.2)
        ax1.add_patch(rect)
        ax1.axvline(x=summary[whiskOnset][i], color ='red')
    ax1.set_xlabel('Frames (500fps // total time 30 s.)')
    ax1.set_ylabel('Inst. amplitude')
    # plt.xlim([0,15000])
    plt.tight_layout()

    return plt.show()

def dataForRelplot(df, summary,  whiskOnset = 'FirstwhiskFrameFromZero', filterSummary = False, lowLim = -0.2, highLim = 0.4, acqRateFPS = 500):
    ### create relplot
    ### assign new gropu with the desired interval 
    ### if error try to remove both first and last row of the file
    ### something like this summary = summary[1:-1]
    summary = summary[1:-1]
    df['whiskGrp'] = np.nan
    df['timeNorm'] = 0
    for ix, i in enumerate(summary[whiskOnset]):
        # print(ix,i)
        window = (np.array([lowLim, highLim])*acqRateFPS)+i
        df.loc[(df['frame']>=window[0]) & (df['frame']<=window[1]), 'whiskGrp'] = ix
        df.loc[(df['frame']>=window[0]) & (df['frame']<=window[1]), 'timeNorm'] = np.arange(lowLim, highLim, 1/acqRateFPS)
    ### subset the value only for group of interest
    dfplot = df[~df['whiskGrp'].isin([np.nan])]

    ### also important feature to include the mean amplitude of event could help filter the whisk as well
    summary['meanAmp'] = list(dfplot.groupby(['whiskGrp'])['inst_amplitude'].mean())
    ### example of added filtering
    # summaryUpdated = summary[(summary['meanAmp']>2.5) & (summary['whiskDuration']>500*0.4) & (summary['noWhiskbefore']>500*0.2) ]
    if filterSummary == True: 
        summaryUpdated = summary[(summary['whiskDuration']>500*0.4) & (summary['noWhiskbefore']>500*0.2)]
    else:
        summaryUpdated = summary

    dfplot = dfplot[dfplot['whiskGrp'].isin(summaryUpdated['whiskGrp'])]

    return dfplot, summaryUpdated

def makeGraphfor(dfplot, label, withLine = False):
    sns.relplot(data=dfplot, x='timeNorm', y=label, kind='line')
    if withLine == True:
        sns.lineplot(data=dfplot, x='timeNorm', y=label, units='whiskGrp', estimator=None, alpha=0.3)
    plt.axvline( x=0,linestyle='--',color='grey')
    plt.tight_layout()
    plt.ylabel(label)
    plt.xlabel('Time (seconds)')

    return plt.show()

def someStat(summaryUpdated):
    print('n: ', len(summaryUpdated))
    print('mean (frame): ', int(summaryUpdated['whiskDuration'].mean()))
    print('median (frame): ', int(summaryUpdated['whiskDuration'].median()))

def p(var):
    plt.figure()
    plt.plot(var[1:10000])
    plt.show()
## ***************************************************************************
## * PIPELINE                                                                *
## ***************************************************************************

df, summary = whiskSummaryFunction(r"C:\Users\Windows\Desktop\### WISKER ###\WhiskForTom\whiskTest\Test20201015DLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5")

df, summary = whiskSummaryFunction(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2019-04-08_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5")

df, summary = whiskSummaryFunction(r"C:\Users\Windows\Desktop\2019-04-08_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5")

files = glob.glob('/home/rum/Desktop/vide3HighSpeed/*.h5')
files.pop(9) # get rid of the opto file

filesToRedo = []
for i in tqdm.tqdm(files):
    try:
        print(i)
        aid = i.split(os.sep)[-1][0:10]
        df, summary = whiskSummaryFunction(i)
        outputFolder = '/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122'
        summary.to_csv(outputFolder+'/'+aid+'.csv')
    except: 
        filesToRedo.append(i)
        print(' ')
        print('###########################################')
        print('BAD FILE' + i)
        print('###########################################')
        print(' ')       
        pass


filesToRedo = ['/home/rum/Desktop/vide3HighSpeed/2019-11-14_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5',
 '/home/rum/Desktop/vide3HighSpeed/2019-11-07_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5',
 '/home/rum/Desktop/vide3HighSpeed/2019-12-13_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5',
 '/home/rum/Desktop/vide3HighSpeed/2019-12-18_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5',
 '/home/rum/Desktop/vide3HighSpeed/2019-11-12_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5',
 '/home/rum/Desktop/vide3HighSpeed/2019-12-12_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5']

h5File = filesToRedo[0]



df = determineAngle(df)
whiskPlottingDetectCheck(df, summary)

fig, ax = plt.subplots(nrows=4, ncols=1, sharex = True)
ax[0].plot(df['angleWhiskSoft'])
ax[1].plot(df['w0likelihood'])
ax[2].plot(df['w3likelihood'])
ax[3].plot(df['angleWhiskSoftFilt'])
ax[3].plot(filtsig)
tst = custMedianFilterSpikeNoise(df['angleWhiskSoft'])
plt.plot(tst)


dfplot, summaryUpdated = dataForRelplot(df, summary, whiskOnset = 'FirstwhiskFrameFromZero')
sns.displot(summaryUpdated, x="whiskDuration")
someStat(summaryUpdated)


makeGraphfor(dfplot, 'phase')
makeGraphfor(dfplot, 'inst_amplitude')

dfplotTest = dfplot[['timeNorm', 'inst_amplitude']
plt.plot(dfplotTest.groupby(['timeNorm']).mean())

# ## ***************************************************************************
# ## * TEST WHISKER OUTPUT AND FUNCTION                                        *
# ## ***************************************************************************

# #### python side
# tst = pd.read_csv('/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/Vaissiere/2021-01 -  Whisking/test.csv')
# tst2 = pd.read_csv(r"C:\Users\Windows\Desktop\### WISKER ###\WhiskForTom\comp\Test20201015_baseline.csv")

# ## ***************************************************************************
# ## * TO IMPROVE SIGNAL                                                       *
# ## ***************************************************************************

# # can do 
# #* square of abs
# #* cumulative 
# #* look at deconvolution and/or FFT
# #* contact angle

# # from frame
# # tst2['ang'][6409]
# # tst2['ang'][9816]
# # tst2['ang'][7457]
# # from frame


# # determine the angle for the data


# f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, constrained_layout=True, figsize=([20.74, 15]))
# start, end = 0, 15000
# # start, end = 0, len(al)
# ax1.plot(np.arange(start,end), tst2['ang'][start:end], label='whisk', color='#2e3192', alpha = 0.5)
# ax1.plot(np.arange(start,end), df['angleWhiskSoft'][start:end], label='dlc', color='#00a651', alpha = 0.5)

# ax2.plot(np.arange(start,end), whiskAngleFilter(df['angleWhiskSoft'][start:end]), label='dlc', color='#00a651', alpha = 0.5)
# ax2.plot(np.arange(start,end), tst2['angfilt'][start:end], label='whisk', color='#2e3192', alpha = 0.5)

# ax3.plot(np.arange(start,end), df['inst_amplitude'][start:end], label='dlc', color='#00a651', alpha = 0.5)
# ax3.plot(np.arange(start,end), tst2['inst_amplitude'][start:end], label='whisk', color='#2e3192', alpha = 0.5)

# ax4.plot(np.arange(start,end), df['phase'][start:end], label='dlc', color='#00a651', alpha = 0.5)
# ax4.plot(np.arange(start,end), tst2['phase'][start:end], label='whisk', color='#2e3192', alpha = 0.5)

# for i in summary.index:
#     rect = matplotlib.patches.Rectangle((summary['FirstwhiskFrame'][i], 0), summary['whiskDuration'][i], max(df['inst_amplitude']), color ='red', alpha = 0.2)
#     ax3.add_patch(rect)
#     ax3.axvline(x=summary['FirstwhiskFrame'][i], color ='red')

# ax1.title.set_text('angle')
# ax2.title.set_text('anglefilt')
# ax4.title.set_text('phase')
# ax3.title.set_text('instantaneous amplitude and whisk detection')
# ax4.set_xlabel('Frames (500fps // total time 30 s.)')
# plt.xlim([0,15000])
# plt.tight_layout()

# ## ***************************************************************************
# ## * DRAFT                                       *
# ## ***************************************************************************

# ####### extract past data #######
# ####### for comparison 

# # extract the whisking and whisker
# al = np.load('/home/rum/Desktop/DLC/videooutput_angle/evTime/2019-04-08.npy')
# df = readHdf('/home/rum/Desktop/vide3HighSpeed/2019-04-08_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5')
# start, end = 510000, 540000

# df = df[start:end]


# df, summary = whiskSummaryFunction(df)
# whiskPlottingDetectCheck(df, summary)




f, (ax3, ax2, ax1) = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=([20.74, 4.8]))

ax3.plot(df['angleWhiskSoft'])
ax2.plot(df['angleWhiskSoftFilt'])
ax1.plot(df['inst_amplitude'])
for i in summary.index:
    rect = matplotlib.patches.Rectangle((summary['FirstwhiskFrame'][i], 0), summary['whiskDuration'][i], max(df['inst_amplitude']), color ='red', alpha = 0.2)
    ax1.add_patch(rect)
    ax1.axvline(x=summary['FirstwhiskFrame'][i], color ='red')
ax1.set_xlabel('Frames (500fps // total time 30 s.)')
ax1.set_ylabel('Inst. amplitude')
# plt.xlim([0,15000])
ax2.title.set_text('angle')
ax2.title.set_text('anglefilt')
ax1.title.set_text('amplitude')
plt.tight_layout()









# f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, constrained_layout=True)
# start, end = 510000, 540000
# # start, end = 0, len(al)
# ax1.plot(np.arange(start,end), al[start:end])
# ax2.plot(np.arange(start,end), anglefilt[start:end])
# ax3.plot(np.arange(start,end), phase[start:end])

# subdf = whiskCycleOutput[(whiskCycleOutput['start'] >= start) &  (whiskCycleOutput['start'] <= end)]
# for i in subdf.index:
#     print(i)
#     print(whiskCycleOutput['start'][i])
#     ax2.axvline(whiskCycleOutput['start'][i], 0, 1, color='g')
#     # ax2.axvline(whiskCycleOutput['stop'][i], 0, 2, color='r')
# ax1.title.set_text('angle')
# ax2.title.set_text('anglefilt')
# ax3.title.set_text('phase')




# f, (ax1) = plt.subplots(1, 1, sharex=True, constrained_layout=True, figsize=([20.74, 4.8]))
# import matplotlib
# ax1.plot(df['inst_amplitude'])
# for i in summary.index:
#     rect = matplotlib.patches.Rectangle((summary['FirstwhiskFrame'][i], 0), summary['whiskDuration'][i], max(df['inst_amplitude']), color ='red', alpha = 0.2)
#     ax1.add_patch(rect)
#     ax1.axvline(x=summary['FirstwhiskFrame'][i], color ='red')
# ax1.set_xlabel('Frames (500fps // total time 30 s.)')
# ax1.set_ylabel('Inst. amplitude')
# ax1.plot(np.arange(start,end), al[start:end], color = 'green')
# # plt.xlim([0,15000])
# plt.tight_layout()