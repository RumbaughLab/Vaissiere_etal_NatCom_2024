## ***************************************************************************
## * CUSTOM FCT                                                              *
## ***************************************************************************
from matplotlib.transforms import Bbox
import glob 

def conversionTiming(markerHS,markerHiSpV, filesNPY,  markerID='HSLEDoff2', outputFolder = '/home/rum/Desktop/DLC/cut_pole1/touches', vidTrans='Transition'):
    '''
    merge 2 pandas data frame and convert eventsTime into the timing corresponding to the timing of the 
    markerHS: dataframe
        pandas dataframe containing the reference time for the events of interest synchrony is perfromed on reperesenting the dataframe for e3 neural data
    markerHiSpV: dataframe
        pandas dataframe containing the reference time for the events of interest synchrony is perfromed on reperesenting the dataframe for the LED marker of interest on the video
    filesNPY: array
        array of files that contains the touch timing
    vidTrans (str): name of the video transition column

    '''
    ####### constant #######
    fs=25000 # rate of acquisition 
    fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500

    ####### constant #######
    markerConv=pd.merge(markerHS, markerHiSpV, on='animalID')
    markerConv['vidE3Delay']= markerConv[markerID]/fs - markerConv[vidTrans]/fsVid # vconversion to the same unit in seconds ## the camera is behind the timing of the actual spikes

    markerConv=markerConv[markerConv['vidE3Delay']>=0]
    
    os.makedirs(outputFolder+'/evTime', exist_ok=True)
    for idxfiles, evtime in enumerate(filesNPY):
        print(idxfiles, evtime)
        aid=os.path.splitext(evtime)[0].split(os.sep)[-1]

        aid=aid.split('_')[-1]
        tmp=np.load(evtime)

        if markerConv[markerConv['animalID']==aid].empty == False :
            tmp=tmp/fsVid+markerConv[markerConv['animalID']==aid]['vidE3Delay'].item()
            np.save(outputFolder+'/evTime/'+aid+'.npy',tmp)

def fillInPole(aid, markerHS):
    fs = 25000 # rate of acquisition of the data
    poleList = ['HSpoleIN1', 'HSpoleOUT1', 'HSpoleIN2', 'HSpoleOUT2']
    markerList = ['HSLEDon2', 'HSLEDon2', 'HSLEDoff2', 'HSLEDoff2']
    timeCorr = np.array([-5,5,-5,5])*60*fs

    for i in zip(poleList, markerList, timeCorr):
        if np.isnan(markerHS.loc[markerHS['animalID']==aid, i[0]].values[0]) == True:
            markerHS.loc[markerHS['animalID']==aid, i[0]] = markerHS.loc[markerHS['animalID']==aid, i[1]]+i[2]

    return markerHS[markerHS['animalID']==aid]

def findLEDmarker(filesRef, markerID='LEDoff2'):
    '''
    Enable to find the marker (trigger) extracted from the reference files created during the initial analysis
    filesRef: array
        array of files based on pre-established criteria
    marekerID: string
        string that can be either one of the following ['LEDon1', 'camTrigger', 'LEDoff1', 'LEDon2', 'LEDoff2', 'LEDon3', 'soundOff1']
    '''
    listAll=[]
    for idx, fRef in enumerate(filesRef):
        # print(idx,fRef)
        aid=fRef.split(os.sep)[-3]
        aid=aid.split('_')[0]
        tmp=pd.read_csv(fRef)
        if markerID == False:
            tmp = pd.pivot_table(tmp, values='timeSlot', columns=['Row']).reset_index(drop=True)
            tmp.columns = ['HS']+tmp.columns
            tmp['animalID'] = aid
            sumArry = tmp

        else:
            tmp=tmp[tmp['Row']==markerID]['timeSlot'].item()
            sumArry=pd.DataFrame({'animalID':[aid], 'HS'+markerID:[tmp]})

        listAll.append(sumArry)

    listAll=pd.concat(listAll)
    listAll.reset_index(inplace=True, drop=True)
    return listAll # or LEDoff2 LEDon2

def getInfoFiles():
        # path='/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab/Jessie/e3 - Data Analysis/e3 Data/**/output/ReferenceTableTransition.csv'
        path = 'Y:/Jessie/e3 - Data Analysis/e3 Data/**/output/ReferenceTableTransition.csv'
        filesRef=glob.glob(path, recursive=True)
        filesRef = [x for x in filesRef if not 'opto' in x] # remove the opto file
        markerHS=findLEDmarker(filesRef, markerID=False)

        # path='/home/rum/Desktop/DLC/cut_pole2/timing/HSLEDtransitionFramesN.csv'
        path = r"Y:\Vaissiere\__UbuntuLamda\DLC\cut_pole2\timing\HSLEDtransitionFramesN.csv"
        markerHiSpV=pd.read_csv(path)

        markerHiSpV['animalID']=markerHiSpV['animalID'].str.split('_', expand=True)[1]

        return markerHS, markerHiSpV

class associationForTimeConversion:
    def __init__(self, aid, markerHS, markerHiSpV, filetoRunConversionTiming = 'Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122'):
        # recover markerHS
        '''
        filetoRunConversionTiming = '/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122'
        filetoRunConversionTiming = 'Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122'
        '''
        self.aid = aid
        self.filetoRunConversionTiming = filetoRunConversionTiming
        self.markerHS = fillInPole(markerHS, animalID=aid)
        self.markerHiSpV = markerHiSpV[markerHiSpV['animalID'].str.contains(aid)]
    
        self.fs=25000 # rate of acquisition 
        self.fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500 
        marker = 'HSLEDon2'   
        self.vidE3Delay = self.markerHS[marker].values/self.fs - self.markerHiSpV['fullVidTrans_'+marker].values/self.fsVid

    def conversionUpdatedforWhisk(self):
        whiskBehavior = glob.glob(self.filetoRunConversionTiming+'/*'+self.aid+'*') # glob.glob(self.filetoRunConversionTiming+'/*'+self.aid+'*')
        whiskBehavior = pd.read_csv(whiskBehavior[0])
        whiskBehavior['FirstwhiskFrameFromZeroConvTime'] = whiskBehavior['FirstwhiskFrameFromZero'] / self.fsVid + self.vidE3Delay

        ### limiting factor for the type of whisking with duration more than 400 ms and no whisking pror with a 200 ms window 
        whiskBehavior['anal_status'] = 'excluded'
        whiskBehavior.loc[(whiskBehavior['whiskDuration']>500*0.4) & (whiskBehavior['noWhiskbefore']>500*0.2),'anal_status'] = 'included' # also see dataForRelplot
        whiskBehaviorFull = copy.copy(whiskBehavior)

        limP1low = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleIN1'].item()/self.fs
        limP1high = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleOUT1'].item()/self.fs
        limP2low = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleIN2'].item()/self.fs
        limP2high = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleOUT2'].item()/self.fs

        ### limiting factor based on the presence of a pole
        whiskBehavior = whiskBehavior[(whiskBehavior['FirstwhiskFrameFromZeroConvTime']<limP1low) | (whiskBehavior['FirstwhiskFrameFromZeroConvTime']>limP1high) & (whiskBehavior['FirstwhiskFrameFromZeroConvTime']<limP2low) | (whiskBehavior['FirstwhiskFrameFromZeroConvTime']>limP2high)]

        ### no filtering but with annotation
        whiskBehaviorFull['polePresent'] = 1
        whiskBehaviorFull.loc[(whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']<limP1low) | (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']>limP1high) & (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']<limP2low) | (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']>limP2high), 'polePresent'] = 0

        return whiskBehavior, whiskBehaviorFull, limP1low, limP1high, limP2low, limP2high

class associationForTimeConversionFORTOUCH:
    def __init__(self, aid, markerHS, markerHiSpV, filetoRunConversionTiming = 'Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122'):
        # recover markerHS
        '''
        filetoRunConversionTiming = '/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122'
        filetoRunConversionTiming = 'Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122'
        '''
        self.aid = aid
        self.filetoRunConversionTiming = filetoRunConversionTiming
        self.markerHS = fillInPole(markerHS, animalID=aid)
        self.markerHiSpV = markerHiSpV[markerHiSpV['animalID'].str.contains(aid)]
    
        self.fs=25000 # rate of acquisition 
        self.fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500 
        marker = 'HSLEDon2'   
        self.vidE3Delay = self.markerHS[marker].values/self.fs - self.markerHiSpV['fullVidTrans_'+marker].values/self.fsVid

    def conversionUpdatedforWhisk(self):
        whiskBehavior = glob.glob(self.filetoRunConversionTiming+'/*'+self.aid+'*') # glob.glob(self.filetoRunConversionTiming+'/*'+self.aid+'*')
        whiskBehavior = pd.read_csv(whiskBehavior[0])
        whiskBehavior['FirstwhiskFrameFromZeroConvTime'] = whiskBehavior['FirstTouchFrame'] / self.fsVid + self.vidE3Delay

        ### limiting factor for the type of whisking with duration more than 400 ms and no whisking pror with a 200 ms window 
        whiskBehavior['anal_status'] = 'excluded'
        whiskBehavior.loc[(whiskBehavior['whiskDuration']>500*0.4) & (whiskBehavior['noWhiskbefore']>500*0.2),'anal_status'] = 'included' # also see dataForRelplot
        whiskBehaviorFull = copy.copy(whiskBehavior)

        limP1low = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleIN1'].item()/self.fs
        limP1high = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleOUT1'].item()/self.fs
        limP2low = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleIN2'].item()/self.fs
        limP2high = self.markerHS.loc[self.markerHS['animalID']==aid, 'HSpoleOUT2'].item()/self.fs

        ### limiting factor based on the presence of a pole
        whiskBehavior = whiskBehavior[(whiskBehavior['FirstwhiskFrameFromZeroConvTime']<limP1low) | (whiskBehavior['FirstwhiskFrameFromZeroConvTime']>limP1high) & (whiskBehavior['FirstwhiskFrameFromZeroConvTime']<limP2low) | (whiskBehavior['FirstwhiskFrameFromZeroConvTime']>limP2high)]

        ### no filtering but with annotation
        whiskBehaviorFull['polePresent'] = 1
        whiskBehaviorFull.loc[(whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']<limP1low) | (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']>limP1high) & (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']<limP2low) | (whiskBehaviorFull['FirstwhiskFrameFromZeroConvTime']>limP2high), 'polePresent'] = 0

        return whiskBehavior, whiskBehaviorFull, limP1low, limP1high, limP2low, limP2high


## ***************************************************************************
## * GATHERING DATA                                                          *
## ***************************************************************************


markerHS, markerHiSpV = getInfoFiles()
aid = '2019-04-08'
files = glob.glob('/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122/*.csv')
files = glob.glob('Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122/*.csv')
aidList = list(map(lambda x: x.split(os.sep)[-1][0:10], files))

exportInfo = {} # initiate a dictionary to store some info
filesToRedo = []
for idx, aid in enumerate(aidList):
    try:
        print(idx, aid)
        t = associationForTimeConversion(aid, markerHS = markerHS, markerHiSpV = markerHiSpV)
        whiskBehavior, whiskBehaviorFull, limP1low, limP1high, limP2low, limP2high = t.conversionUpdatedforWhisk()


        toSave = np.array(whiskBehavior.FirstwhiskFrameFromZeroConvTime)
        exportInfo[aid] = len(toSave)

        np.save('/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122/evTime/'+aid+'.npy', toSave)
    except:
        filesToRedo.append(aid)
        pass



markerHS, markerHiSpV = getInfoFiles()
aid = '2019-04-08'
files = glob.glob('/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122/*.csv')
files = glob.glob('Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122/*.csv')
aidList = list(map(lambda x: x.split(os.sep)[-1][0:10], files))

exportInfo = {} # initiate a dictionary to store some info
filesToRedo = []
for idx, aid in enumerate(aidList):
    try:
        print(idx, aid)
        t = associationForTimeConversion(aid, markerHS = markerHS, markerHiSpV = markerHiSpV)
        whiskBehavior, whiskBehaviorFull, limP1low, limP1high, limP2low, limP2high = t.conversionUpdatedforWhisk()
        

        ### section to annotate and saved the annotate 
        whiskBehaviorFull = whiskBehaviorFull[['FirstwhiskFrameFromZero','FirstwhiskFrameFromZeroConvTime','whiskDuration','anal_status','polePresent']]
        whiskBehaviorFull.columns = ['FirstwhiskFrame','FirstwhiskTime','whiskDurationFrame','anal_status','polePresent']
        whiskBehaviorFull['whiskDurationTime'] = whiskBehaviorFull['whiskDurationFrame']/500
        whiskBehaviorFull.to_csv('Y:/Vaissiere/__UbuntuLamda/DLC/videooutput_angle/updatedWhisk20210122/'+aid+'_annoted.csv')



        # toSave = np.array(whiskBehavior.FirstwhiskFrameFromZeroConvTime)
        # exportInfo[aid] = len(toSave)

        # np.save('/home/rum/Desktop/DLC/videooutput_angle/updatedWhisk20210122/evTime/'+aid+'.npy', toSave)
    except:
        filesToRedo.append(aid)
        pass



# ## ***************************************************************************
# ## * WHISK REPRESENTATION                                                    *
# ## ***************************************************************************

# plt.plot(whiskBehaviorFull.FirstwhiskFrameFromZeroConvTime, np.repeat(1.5, len(whiskBehaviorFull)), '.', label='all whisk', alpha=0.1)
# plt.plot(whiskBehavior.FirstwhiskFrameFromZeroConvTime, np.repeat(1, len(whiskBehavior)), '.', label='filltered whisk',  alpha=0.1)
# plt.ylim([0.5,2])


# for i in [limP1low, limP1high, limP2low, limP2high]:
#     plt.axvline(x=i, color='red')
# plt.show()

# plt.xlabel('Time (s)')
# plt.ylabel('Wisk \n (events)')
# # filter whiskBehavior
# # first filter base on the the timing 

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()