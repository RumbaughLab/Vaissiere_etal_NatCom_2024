df, summary = whiskSummaryFunction(r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2019-04-08_HSDLC_resnet50_e3_highspeedWhiskerOct30shuffle1_1030000.h5")


meas = ['angle', 'angleWhiskSoft', 'angleWhiskSoftFilt', 'inst_amplitude',
       'phase', 'phaseZero', 'phaseAmpCat', 'phaseAmpGrp']



## ***************************************************************************
## * AUC whisker for instant amplitude                                       *
## ***************************************************************************


f_frame = lo['FirstwhiskFrame'][0]
l_frame = lo['FirstwhiskFrame'][0]+lo['whiskDurationFrame'][0]

fig, ax = plt.subplots(8,1, sharex=True)
for i,j in enumerate(meas):
        print(i,j)
        a = df.loc[(df['frame']>f_frame) & (df['frame']<l_frame), [j,'frame']]
        ax[i].plot(a['frame'], a[j])

        if j == 'inst_amplitude':
            b = df.loc[(df['frame']>f_frame) & (df['frame']<l_frame), j].values
            np.trapz(b.values)


## ***************************************************************************
## * AUC whisker for instant amplitude                                       *
## ***************************************************************************

touchMainDir = 'Y:/Vaissiere/__UbuntuLamda/DLC/cut_pole1and2_threshMeth/touches/evTime'+os.sep
tmpEV = glob.glob(touchMainDir+'*'+'2019-04-08'+'*')
tmpEV = np.load(tmpEV[0])
tmpEV = pd.DataFrame({'FirstTouchTime':tmpEV})
tmpEV['used'] = 1


## ***************************************************************************
## * Assessment of touch                                       *
## ***************************************************************************

### check code here Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline ==> final files 
### check code here for conversion "Y:\2020-09-Paper-ReactiveTouch\__code__\whisking\pythonWhiskerCLEAN_lastTimingConversion.py"
### check here Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails

a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg1toProcess\1_space_2019-04-08.csv"
b = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg2toProcess\2_pole_2019-04-08.csv"

alld = []
for i in [a,b]:
    tmp = pd.read_csv(i)
    alld.append(tmp)

alld = pd.concat(alld)
tmp = alld[alld['newMtouch']==1]
tmp[tmp['touchCount']>2]



###########################################################################
### Try out for a with the first section 
###########################################################################


mainpath = 




markerHS, markerHiSpV = getInfoFiles()
aid = '2019-04-08'
a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg1toProcess\1_space_2019-04-08.csv"
a = pd.read_csv(a)
a = a.loc[a['newMtouch']==1, ['FirstTouchFrame']].reset_index(drop=True)
a['csv']= 1


fs=25000 # rate of acquisition 
fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500 
marker = 'HSLEDon2'   
markerHS = fillInPole(markerHS, animalID=aid)
markerHiSpV = markerHiSpV[markerHiSpV['animalID'].str.contains(aid)]
vidE3Delay = markerHS[marker].values/fs - markerHiSpV['fullVidTrans_'+marker].values/fsVid
a['FirstTouchTime'] = a['FirstTouchFrame'] / fsVid + vidE3Delay

b = pd.merge(a,tmpEV)

plt.figure()
plt.plot(a['FirstTouchTime'],'.', alpha=.3, color = 'green')
plt.plot(tmpEV['FirstTouchTime'],'.', alpha=.3, color='blue')
plt.plot(b['FirstTouchTime'],'.', alpha=.3, color='red')

###########################################################################
### Try out for a with the first section 
###########################################################################

markerHS, markerHiSpV = getInfoFiles()
aid = '2019-04-08'
a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg2toProcess\2_pole_2019-04-08.csv"
a = pd.read_csv(a)
a = a.loc[a['newMtouch']==1, ['FirstTouchFrame']].reset_index(drop=True)
a['csv']= 1


fs=25000 # rate of acquisition 
fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500 
marker = 'HSLEDoff2'   
markerHS = fillInPole(markerHS, animalID=aid)
markerHiSpV = markerHiSpV[markerHiSpV['animalID'].str.contains(aid)]
vidE3Delay = markerHS[marker].values/fs - markerHiSpV['fullVidTrans_'+marker].values/fsVid
a['FirstTouchTime'] = a['FirstTouchFrame'] / fsVid + vidE3Delay

b = pd.merge(a,tmpEV)

plt.figure()
plt.plot(a['FirstTouchTime'],'.', alpha=.3, color = 'green')
plt.plot(tmpEV['FirstTouchTime'],'.', alpha=.3, color='blue')
plt.plot(b['FirstTouchTime'],'.', alpha=.3, color='red')






markerHS[markerHS['animalID']==aid]