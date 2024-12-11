# ################################################
# ## Get the ref files 
# ################################################

## METHODS TO BE IMPROVED

# # group specific files
# f1 = pd.read_csv(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\autoDetectTouches\manualThreshold_detection.csv")
# f1 = f1.iloc[:,1:]
# f2 =  pd.read_csv(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\autoDetectTouches\manualThreshold_error.csv")
# f2 = f2.iloc[:,1:]
# f3 =  pd.read_csv(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\captrue_forTouch\animals_close_iter.csv")
# f3 = f3.dropna(axis=1)
# f1 = f1.merge(f2, on=['position', 'sID'])
# f1 = f1.merge(f3, on=['position', 'sID'])

# f1.to_csv(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\captrue_forTouch\animals_close_iter_wThresh.csv")

def thresholdData(array, customThreshold , noTouchVal=-1000, touchVal=-1001): #
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

    threshold = customThreshold
    meanFrame = copy.deepcopy(array[1]) # add to multiply here since the thing as been normalized
    meanFrame = meanFrame - np.mean(meanFrame)
    meanFrame = np.where(meanFrame <= threshold, touchVal, meanFrame)
    meanFrame = np.where(meanFrame > threshold, noTouchVal, meanFrame)

    dat = pd.DataFrame({'frame': array[0], 'filt': meanFrame})
    dat.loc[dat['filt']==-1001, 'filt']=1
    dat.loc[dat['filt']==-1000, 'filt']=0


    return dat

def convertRawtoSummary(dat, rawDat='filt'):
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

def quickConversion(tmp):
    tmp = tmp.reset_index()
    if tmp.columns.nlevels > 1:
        tmp.columns = ['_'.join(col) for col in tmp.columns] 
    tmp.columns = tmp.columns.str.replace('[_]','')
    return tmp

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

    saveName = filename.split('.')[0]+'_corrected.csv'
    alpha.to_csv(saveName)

    return alpha

def filterTiming(a, filterVal = 40):
    '''
    This function retruns a truncated version of the file
    along with graphic and 
    
    Args:
        a = csvFile
        filterVal = 40 ## default: 40 (frames) # everything more than 40 frames 80ms gets discounted
    '''

    # shift the interval between the 2
    a['interEventinter'] = a['interEventinter'].shift(1)
    a['interEventinter'] = a['interEventinter'].fillna(filterVal+1)

    af = a[a['interEventinter']>filterVal] 

    return af

def conversionTouchTimingperAnimal(aid, filterT = None):
    datPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'
    timeFile = glob.glob(datPath+os.sep+'*'+aid+'*'+os.sep+'output/ReferenceTableTransition.csv')[0] ## self.timeFile
    srE = 25000 # sampling rate of the ephys
    srHS = 500 # sampling rate of the Highspeed camera
    dat = pd.read_csv(timeFile) ## self.timeFile
    times = (dat.loc[dat['Row'].isin(['LEDon2', 'LEDoff2']),'timeSlot'].values/srE) # those are the e3 times in seconds

    ## frame path 
    framePath = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline"
    frameFile = glob.glob(framePath + '/**/*'+'LEDframes_*'+aid+'*.npy', recursive=True)[0]
    frames = np.load(frameFile)
    framesTime = frames/srHS

    ## quick check
    check = times[0]-framesTime[0]
    if check >= 0:
        print('looks good:', times - framesTime)
    else:
        print('STOP - verify timing')
    frameCorrection =  times - framesTime

    if filterT == True:
        # conversion of file
        event1 = pd.read_csv(glob.glob(framePath + '/**/*' + 'touchDAT_detection*' +aid+ '*1_corrected.csv', recursive=True)[0])
        event2 = pd.read_csv(glob.glob(framePath + '/**/*' + 'touchDAT_detection*' +aid+ '*2_corrected.csv', recursive=True)[0])

        raw1 = np.load(glob.glob(framePath + '/**/*' + 'pole_*' +aid+ '*_pres1.npy', recursive=True)[0])
        raw2 = np.load(glob.glob(framePath + '/**/*' + 'pole_*' +aid+ '*_pres2.npy', recursive=True)[0])


        f1 = filterTiming(event1)
        f2 = filterTiming(event2)


        infoNevents = pd.DataFrame({'aid':[aid, aid], 'order': [1,2], 'nEvents':[len(event1), len(event2)],'nEventsFilter':[len(f1), len(f2)]})

        # eventTimes
        eventTimes1 = f1['FirstTouchFrame'].values/srHS+frameCorrection[0]
        eventTimes2 = f2['FirstTouchFrame'].values/srHS+frameCorrection[1]
        allevents = np.concatenate([eventTimes1,eventTimes2])


        dirSavePath = os.sep.join(frameFile.split(os.sep)[:-2])+os.sep+'touchTimeFilter'
        os.makedirs(dirSavePath, exist_ok=True)

        saveName = dirSavePath+os.sep+aid+'filter.npy'
        np.save(saveName, allevents)


        rawl = [raw1, raw2]; eventl = [event1, event2]; fl = [f1, f2]
        for i in [0,1]:
            plt.figure(figsize=[38,19])
            plt.plot(rawl[i][0], rawl[i][1])
            plt.vlines(eventl[i].FirstTouchFrame.values, 0 ,5, color='red')
            plt.vlines(fl[i].FirstTouchFrame.values, -10 ,0, color='purple')
            plt.savefig(dirSavePath+os.sep+aid+'_'+str(i)+'_graphfilt.jpg')
            plt.close('all')


    else:
        # conversion of file
        event1 = pd.read_csv(glob.glob(framePath + '/**/*' + 'touchDAT_detection*' +aid+ '*1_corrected.csv', recursive=True)[0])
        event2 = pd.read_csv(glob.glob(framePath + '/**/*' + 'touchDAT_detection*' +aid+ '*2_corrected.csv', recursive=True)[0])

        # eventTimes
        eventTimes1 = event1['FirstTouchFrame'].values/srHS+frameCorrection[0]
        eventTimes2 = event2['FirstTouchFrame'].values/srHS+frameCorrection[1]
        allevents = np.concatenate([eventTimes1,eventTimes2])


        dirSavePath = os.sep.join(frameFile.split(os.sep)[:-2])+os.sep+'touchTime'
        os.makedirs(dirSavePath, exist_ok=True)

        saveName = dirSavePath+os.sep+aid+'.npy'
        np.save(saveName, allevents)

    return print('data saved here: ', saveName), infoNevents


################################################
## Get the touch data
################################################



mainDir1 = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches'
mainDir2 = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2\autoDetectTouches'

mainDir = [mainDir2, mainDir1]

for k in mainDir:
    for j in [1,2]:
        param2 = j
        keyfiles = glob.glob(k+'/**/pole*pres'+str(param2)+'.npy', recursive=True)
        for i in keyfiles:
            try:
                # parameters
                param1 = 'detection'
                # myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p'+str(param2)+'_update.csv'
                myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p'+str(param2)+'_update.csv'
                selft = graphManualThreshold(i, ref = myref, dataType = param1, e3dat = True, polePres=param2, binTouch = True) # see class in 
                dat, summary = selft.correctionForError()
                # f, ax = plt.subplots(2, sharex=True)
                # ax[0].plot(selft.vertDetect[0], selft.vertDetect[1])
                # ax[1].plot(dat.frame, dat.filt)
            except:
                print('!!!!!ERROR!!!!!!')
                print(selft.sID)
                print('!!!!!ERROR!!!!!!')




################################################
## apply the correction for touch files 
################################################

for k in mainDir:
    print(k)
    touchFiles = glob.glob(mainDir1+'/**/*touchDAT_detection*.csv', recursive=True)
    for i in touchFiles:
        print(i)
        touchCorrectionShortInterval(i)



### QUICK CHECK
###------------------------------------------------------------
# a = pd.read_csv(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches\touchDAT_detection_2020-06-10_poleP_1.csv")
# b = pd.read_csv(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches\touchDAT_detection_2020-06-10_poleP_1_corrected.csv")
# c = np.load(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches\pole_2020-06-10_17-12-21 - pole noise__2020-06-10_HS_pres1.npy")
# plt.plot(a.FirstTouchFrame, np.repeat(1, len(a.FirstTouchFrame)),'.')
# plt.plot(a.FirstTouchFrame, np.repeat(1.2, len(a.FirstTouchFrame)),'.')
# plt.plot(c[0], c[1])
# plt.vlines(a.FirstTouchFrame.values, 0 ,5, color='red')
# plt.vlines(b.FirstTouchFrame.values, -5 ,0, color='green')
# plt.vlines(af.FirstTouchFrame.values, -10 ,0, color='purple')


#################################################
## apply timing corrections 
################################################

## RUM2
files = glob.glob(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches'+os.sep+'*corrected.csv')

idlist = np.unique([x.split('_')[2] for x in files])

# a = idlist[1]
# lista = [x for x in files if a in x]

for i in idlist:
    print(i)
    conversionTouchTimingperAnimal(i)



## EMXRUM2
files = glob.glob(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2\autoDetectTouches'+os.sep+'*corrected.csv')

idlist = np.unique([x.split('_')[2] for x in files])


for i in idlist:
    print(i)
    try:
        conversionTouchTimingperAnimal(i)
    except:
        print('ERROR ERROR with: ', i)


## with the filter 
############-----------------

allInfo = []
for i in idlist:
    print(i)
    a = conversionTouchTimingperAnimal(i, filterT = True)
    allInfo.append(a)
allInfo = pd.concat([x[1] for x in allInfo])
allInfo['ratio'] = allInfo['nEventsFilter']/allInfo['nEvents']

## RUM2
allInfo.to_csv(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\touchTimeFilter\eventInfo.csv")

## EMXRUM2
allInfo.to_csv(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2\touchTimeFilter\eventInfo.csv')
############-----------------



################################################
## get all the files
################################################

mainDir = r'Y:\Sheldon\Highspeed\not_analyzed\WDIL009\autoDetectTouches'
touchFiles = glob.glob(mainDir+'/**/*corrected.csv', recursive=True)

datAll = []
for i in touchFiles:
    position  = i.split(os.sep)[-2]
    sID = '_'.join(i.split(os.sep)[-1].split('_')[2:5])
    ssID = sID.split('_')[0]
    print(position, sID, ssID)


    dat = pd.read_csv(i)
    dat['position'] = position
    dat['sID'] = ssID

    datAll.append(dat)

datAll = pd.concat(datAll)


################################################
## get the genotype info 
################################################

geno = pd.read_excel(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\Copy of WDIL of Emx1CrexRUM2 mice 1-27-21.xls", header = 1)
geno['sID'] = geno['Ped#'].str.strip('[^a-zA-Z]')
geno['geno'] = geno['Syngap1-3 EX']
geno.loc[geno['geno']=='+', 'geno']='wt'
geno.loc[geno['geno']=='-', 'geno']='het'

geno = geno[['sID','geno']]


datAll = datAll.merge(geno, on='sID')

datAll.to_csv(mainDir+os.sep+'alltheData.csv')