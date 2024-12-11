import os
os.chdir(r'Y:\2020-09-Paper-ReactiveTouch')
from ALLmethods import *

mainPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'

file = r"Y:\Vaissiere\fileDescriptor2.csv"
f = pd.read_csv(file)
mylst = f.loc[f['genotype']=='Rum2', ['Record_folder', 'Animal_geno']].reset_index(drop=True)
mylst['outputStatus'] = mylst['Record_folder'].apply(outputPresent) # see outputPresent in main function
mylst['vidStatus'] = mylst['Record_folder'].apply(videoPresent) # see outputPresent in main function


class refTiler:
    def __init__(self, aid, polePres=1):
        self.aid = aid
        self.polePres = polePres
        vidPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline'
        self.vidFile = glob.glob(vidPath+'/**/*'+aid+'*.mp4', recursive=True)[0]
        datPath = r'Y:\Jessie\e3 - Data Analysis\e3 Data'
        self.timeFile = glob.glob(datPath+os.sep+aid+os.sep+'output/ReferenceTableTransition.csv')[0]
        refFile = pd.read_csv(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p"+str(self.polePres)+".csv")
        self.cropAreas = refFile[refFile['Record_folder'] == aid]

    def timeForVid(self):
        srE = 25000 # sampling rate of the ephys
        srHS = 500 # sampling rate of the Highspeed camera
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

    def refview(self):
        mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2'
        # polePosition = fileVid.split(os.sep)[-2]
        outDir = mainDir+os.sep+'autoDetectTouches-TESTPERF'
        os.makedirs(outDir, exist_ok=True)
        vcap = cv2.VideoCapture(self.vidFile)

        # aid = j['sID'].item()
        aidSave = os.path.basename(self.vidFile).split('.')[0]
        # print(aid)


        vcap.set(1, 0)
        ret, refframe = vcap.read()

        plt.figure()
        plt.imshow(refframe)
        plt.title(self.aid)
        plt.show()


for i in files:
	try:
		t = refTiler(i[0], i[1])
		t.refview()
	except:
		print(i)
