import os
os.chdir(r'Y:\2020-09-Paper-ReactiveTouch')
from ALLmethods import *

mainPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'

file = r"Y:\Vaissiere\fileDescriptor2.csv"
f = pd.read_csv(file)
# mylst = f.loc[f['genotype']=='Rum2', ['Record_folder', 'Animal_geno']].reset_index(drop=True)
mylst = f.loc[f['genotype']=='EMX1-Rum2', ['Record_folder', 'Animal_geno']].reset_index(drop=True)
# EMX1-Rum2
mylst['outputStatus'] = mylst['Record_folder'].apply(outputPresent) # see outputPresent in main function
mylst['vidStatus'] = mylst['Record_folder'].apply(videoPresent) # see outputPresent in main function
mylst = mylst[mylst['vidStatus']!='empty']

# for i in mylst['Record_folder']:
#     print(i)
#     file = glob.glob(mainPath+os.sep+i+os.sep+'output/ReferenceTableTransition.csv')[0]
#     dat = pd.read_csv(file)


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

    def detectAreaDf(self):
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

def tempFunction(indName):
    print(indName)
    t = correspTouchDatStream(indName[0], indName[1])
    t.detectAreaDf()

indName =  mylst['Record_folder'].values
a = pd.DataFrame({'a': indName})
a['b'] = 1
b = pd.DataFrame({'a': indName})
b['b'] = 2
a = a.append(b)
files = []
for x,i in a.iterrows():
    # print(x,i)
    files.append([i['a'], i['b']])
print(files)


with concurrent.futures.ProcessPoolExecutor() as executor:
    if __name__ == '__main__':
        executor.map(tempFunction, files)

