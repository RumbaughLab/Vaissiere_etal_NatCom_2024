import os
os.chdir(r'Y:\2020-09-Paper-ReactiveTouch')
from ALLmethods import *

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
            # # change input here if pole pres and detection different
            # ### baseic in put for troubleshooting
            myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p1.csv'
            param1 = 'error'
            param2=1
            i = keyfiles[19]
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
        global xlimits
        # coords = []
        # xlimits = []
        coords.append(event.ydata)
        xlimits.append(event.xdata)

###########
class graphManualThreshold:
    '''
    The process here is:
    1. get all the initial parameters
    2. find the limit of the area of interest to be analyzed downstream
    3. plot the area of interest and establish the threshold
    4. save the threshold and detrended area of interest for analysis downstream
    '''

    def __init__(self, filename, ref = [], dataType = 'detection', e3dat = True, polePres=1, binTouch = False, detrend=False):
        plt.ioff()
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
        self.detrend = detrend
        self.binTouch = binTouch
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.position  = filename.split(os.sep)[-2]
        self.sID = self.filename.split(os.sep)[-1].split('_')[1]#'_'.join(self.filename.split(os.sep)[-1].split('_')[1:4])

        posList = ['close', 'middle', 'far']
        if any(self.position in x for x in posList) == False:
            self.position = ''
            self.err = np.load(glob.glob(self.mainDir+os.sep+'*err_*'+self.sID+'*pres'+str(self.polePres)+'*.npy')[0])
            self.err[1] = -self.err[1]
            self.led = np.load(glob.glob(self.mainDir+os.sep+'*LED_*'+self.sID+'*pres'+str(self.polePres)+'*.npy')[0])
            self.horizDetect = np.load(glob.glob(self.mainDir+os.sep+'*space_*'+self.sID+'*pres'+str(self.polePres)+'*.npy')[0])
            self.vertDetect = np.load(glob.glob(self.mainDir+os.sep+'*pole_*'+self.sID+'*pres'+str(self.polePres)+'*.npy')[0])
            self.vertDetect[1] = -self.vertDetect[1]

            if binTouch == True:
                self.trim_vertDetect = np.load(glob.glob(self.mainDir+os.sep+'*Update-w-offset_detection*'+self.sID+'*poleP_'+str(self.polePres)+'*.npy')[0])
                self.trim_err = np.load(glob.glob(self.mainDir+os.sep+'*Update-w-offset_err*'+self.sID+'*poleP_'+str(self.polePres)+'*.npy')[0])

        else:
            ''' this section is to link all the other npy array
            which are associated to the main numpy are being by default the 
            space vertical detect out put '''
            self.err = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*err_*'+self.sID+'*.npy')[0])
            self.err[1] = -self.err[1]
            self.led = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*LED_*'+self.sID+'*.npy')[0])
            self.horizDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*space*'+self.sID+'*.npy')[0])
            # note most of the detection is performed on the pole space
            self.vertDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*pole_*'+self.sID+'*.npy')[0])
            self.vertDetect[1] = -self.vertDetect[1]

            if binTouch == True:
                self.trim_vertDetect = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*Update-w-offset_detection*'+self.sID+'*.npy')[0])
                self.trim_err = np.load(glob.glob(self.mainDir+os.sep+self.position+os.sep+'*Update-w-offset_err*'+self.sID+'*.npy')[0])

    def getLEDtime(self):
        led1 = np.load(glob.glob(self.mainDir+os.sep+'*LED_*'+self.sID+'*pres'+str(1)+'*.npy')[0])
        led2 = np.load(glob.glob(self.mainDir+os.sep+'*LED_*'+self.sID+'*pres'+str(2)+'*.npy')[0])

        ###############
        ## this is for led1
        ###############
        global coords
        global xlimits
        coords =  []
        xlimits = []


        # fig = plt.figure()
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(-5760, 23, 1920, 1017) # for neuropixel
        # ax = fig.add_subplot(111)
        # ax.plot(led1[0], led1[1])
        # ax.set_xlim([led1[0][0], led1[0][-1]])
        # plt.show(block = True)
        # # ax.plot(comled[0][:-1], np.diff(comled[1]))


        # print('Is light up in first phase (y/n): ')
        # # example yes -----_____
        # # example no  ____------
        # yesDetrend = msvcrt.getch()

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(-5760, 23, 1920, 1017) # for neuropixel
        ax = fig.add_subplot(111)
        ax.plot(led1[0], led1[1])
        ax.set_xlim([led1[0][0], led1[0][-1]])
        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show(block = True)

        Lval = coords[-1]
        led1s = int(round(xlimits[-1]))
        # ledthresh = [-Lval,Lval] # to find the part that decend
        # led1 = int(comled[0][np.where(np.diff(comled[1])<ledthresh[0])[0][0]])
        # led2 = int(comled[0][np.where(np.diff(comled[1])>ledthresh[1])[0][-1]])

        # if yesDetrend == b'y':
        #     led1s = int(led1[0][np.where(led1[1]>Lval)[0][-1]])
        # else:
        #     led1s = int(led1[0][np.where(led1[1]>Lval)[0][0]])

        ## verification plot
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(-5760, 23, 1920, 1017) # for neuropixel
        ax = fig.add_subplot(111)
        ax.plot(led1[0], led1[1])
        ax.set_xlim(led1s-1000, led1s+1000)
        ax.axvline(led1s, color = 'green')
        plt.show(block = True)

        ###############
        ## this is for led2
        ###############

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(-5760, 23, 1920, 1017) # for neuropixel
        ax = fig.add_subplot(111)
        ax.plot(led2[0], led2[1])
        ax.set_xlim([led2[0][0], led2[0][-1]])
        # ax.plot(comled[0][:-1], np.diff(comled[1]))

        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show(block = True)

        Lval = coords[-1]
        led2s = int(round(xlimits[-1]))
        # ledthresh = [-Lval,Lval] # to find the part that decend
        # led2 = int(comled[0][np.where(np.diff(comled[1])<ledthresh[0])[0][0]])
        # led2 = int(comled[0][np.where(np.diff(comled[1])>ledthresh[1])[0][-1]])

        # if yesDetrend == b'y':
        #     led2s = int(led2[0][np.where(led2[1]>Lval)[0][0]])
        # else:
        #     led2s = int(led2[0][np.where(led2[1]>Lval)[0][-1]])

        ## verification plot
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(-5760, 23, 1920, 1017) # for neuropixel
        ax = fig.add_subplot(111)
        ax.plot(led2[0], led2[1])
        ax.set_xlim(led2s-1000, led2s+1000)
        ax.axvline(led2s, color = 'green')
        plt.show(block = True)


        tmp = np.array([led1s, led2s])    
        plt.close('all')

        print(tmp)
        updateSaveName = self.dirname+os.sep+'LEDframes_'+'_'+self.sID+'.npy'
        print(updateSaveName)
        np.save(updateSaveName, tmp)

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
        plt.ioff()
        # SECTION TO GET the limit and the error
        ##------------------------------------------------------------------------
        vertDetect = self.err[1]
        vertDetect = vertDetect-vertDetect.mean()
        vertFrame = self.err[0]


        global coords
        global xlimits
        coords =  []
        xlimits = []

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 23, 2560, 1377) # for neuropixel
        # mngr.window.setGeometry(0, 56, 3840, 2004) # for laptop
        ax = fig.add_subplot(111)
        ax.plot(vertFrame, vertDetect)
        admed = np.median(vertDetect)
        # ax.set_ylim([admed-40,admed+5])
        ax.set_xlim([vertFrame[0], vertFrame[-1]])
        ax.set_title(self.position+'//'+self.sID+'// error')
        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show(block=True)

        mylimit = [xlimits[0],xlimits[-1]]
        mylimit = np.array(mylimit).astype(int)

        ## correction of the error for threshold detection
        newVertDetect = np.vstack((vertFrame, vertDetect))
        curridx = np.where((vertFrame>mylimit[0]) & (vertFrame<mylimit[-1]))[0][[0,-1]]
        newVertDetect = newVertDetect[:, curridx[0]:curridx[-1]]
        newVertDetect[1] = -abs(newVertDetect[1]-newVertDetect[1].mean())

        # get threshold for error with coords
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 23, 2560, 1377) # for neuropixel
        # mngr.window.setGeometry(0, 56, 3840, 2004) # for laptop
        ax = fig.add_subplot(111)
        cursor = Cursor(ax)
        ax.set_title(self.position+'//'+self.sID+'// errorZOOM')
        ax.plot(newVertDetect[0], newVertDetect[1])
        ax.set_xlim([newVertDetect[0][0], newVertDetect[0][-1]])
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show(block=True)

        updateSaveName = self.dirname+os.sep+'Update-w-offset_'+'error'+'_'+self.sID+'_poleP_'+str(self.polePres)+'.npy'
        print(updateSaveName)
        np.save(updateSaveName, newVertDetect)


        ## TODO work on this more accurate 
        ## TODO REALLY NEED TO IMPROVE THIS
        # a = self.led
        # ididx = np.where(np.diff(a[1])<-70)[0] # 160 seems to be a general threshold for the image coming on and of
        # ididx = [ididx[0], ididx[-1]]
        # idtime = a[0][ididx]

        # minInter = 4 # minute interval surrounding the led signal 
        # fs = 60*500 # acquisition rate of the camer
        # finter = fs * minInter

        # interval = [[idtime[0]-finter, idtime[0]+finter], [idtime[1]-finter, idtime[1]+finter]]
        # interval = np.where(np.isin(a[0], interval))[0]


        # interval = np.array([0,len(a[0])])

        # interval = abs(np.subtract(t.led[1][:-100], t.led[1][100:])) # similar idea as np.diff on longer window and performing only one derivative
        # interval = np.where(interval>20)[0][[0,-1]]

        # # directly with the value
        # interval = [[615048,903958],[]]
        # if polePres == 1:
        #     interval = interval[0]
        # else:
        #     interval = interval[1]

        
        return mylimit, coords

        # if self.polePres == 1:
        #     return interval[:2]# limit index
        # else:
        #     return interval[2:]

    def getDataFromGraph(self):
        plt.ioff()
        # vertDetect = self.nonLinearDetrend() ## this is off to perform this without linear detrending
        
        dat = self.getInfo()
        mylimit, coords = self.e3Limit()

        # ## saving the reference
        dat['cutomThresholdMeanOut'] = coords[-1] # coords comes from the cursor object

        ## this section is to get the data from the graph
        ##------------------------------------------------------------------------

        vertDetect = self.vertDetect[1]
        vertDetect = vertDetect-vertDetect.mean()
        vertFrame = self.vertDetect[0]

        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 23, 2560, 1377) # for neuropixel
        # mngr.window.setGeometry(0, 56, 3840, 2004) # for laptop
        ax = fig.add_subplot(111)
        ax.plot(vertFrame, vertDetect)
        admed = np.median(vertDetect)
        ax.set_ylim([admed-40,admed+5])
        ax.set_xlim([mylimit[0], mylimit[-1]])
        ax.set_title(self.position+'//'+self.sID+'//detection')
        # usage of the Cursor class
        cursor = Cursor(ax)
        fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        plt.show(block=True)

        ## saving the reference
        dat['polePres'] = self.polePres
        dat['cutomThresholdMeanSpaceL'] = coords[-1] # coords comes from the cursor object

        newVertDetect = np.vstack((vertFrame, vertDetect))
        curridx = np.where((vertFrame>mylimit[0]) & (vertFrame<mylimit[-1]))[0][[0,-1]]
        newVertDetect = newVertDetect[:, curridx[0]:curridx[-1]]

        updateSaveName = self.dirname+os.sep+'Update-w-offset_'+'detection'+'_'+self.sID+'_poleP_'+str(self.polePres)+'.npy'
        print(updateSaveName)
        np.save(updateSaveName, newVertDetect)

        return dat, vertDetect


        # if self.e3dat == False:
        #     limit = self.getThelimit()

        #     if self.dataType == 'detection':
        #         vertDetect = self.vertDetect[1][limit[0]:limit[1]]
        #         vertDetect = vertDetect-vertDetect.mean()
        #         vertFrame = self.vertDetect[0][limit[0]:limit[1]]

        #     else: # to detect the error
        #         vertDetect = self.err[1][limit[0]:limit[1]]
        #         vertDetect = vertDetect-vertDetect.mean()
        #         vertFrame = self.err[0][limit[0]:limit[1]]
        # else:
        #     limit = self.e3Limit()

        #     if self.dataType == 'detection':
        #         vertDetect = self.vertDetect[1][limit[0]:limit[1]]
        #         vertDetect = vertDetect-vertDetect.mean()
        #         vertFrame = self.vertDetect[0][limit[0]:limit[1]]

        #     else: # to detect the error
        #         vertDetect = self.err[1][limit[0]:limit[1]]
        #         vertDetect = vertDetect-vertDetect.mean()
        #         vertFrame = self.err[0][limit[0]:limit[1]]
        
        # if self.detrend == True:
        #     ## this section is to choose if detrending should be applied or not
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.plot(vertFrame, vertDetect, alpha=0.5, label='raw')
        #     ax.plot(vertFrame, signal.detrend(vertDetect)-5, alpha=0.5, label='detrend')
        #     admed = np.median(vertDetect)
        #     ax.set_ylim([admed-40,admed+5])
        #     ax.set_xlim([vertFrame[0], vertFrame[-1]])
        #     ax.set_title(self.position+'//'+self.sID)
        #     ax.legend()
        #     # usage of the Cursor class
        #     cursor = Cursor(ax)
        #     fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        #     fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        #     plt.show()

        #     ## change from input to msvrt to avoid having to press enter
        #     # yesDetrend = input('Should detrending be applied (y/n): ')

        #     print('Should detrending be applied (y/n): ')
        #     yesDetrend = msvcrt.getch()

        #     if yesDetrend == b'y':
        #         print('detrending')
        #         vertDetect = signal.detrend(vertDetect)
        #     else:
        #         print('NO detrending applied')
        #         vertDetect = vertDetect-vertDetect.mean()

    def correctionForError(self):      

        if self.e3dat == True:
            datDetect = thresholdData(array = self.trim_vertDetect, customThreshold = self.getmanThreshold()['cutomThresholdMeanSpaceL'].item())
            datErr = thresholdData(array = self.trim_err, customThreshold = self.getmanThreshold()['cutomThresholdMeanOut'].item())
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

        touchSaveName = self.dirname+os.sep+'touchDAT_'+self.dataType+'_'+self.sID+'_poleP_'+str(self.polePres)+'.csv'
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
    ref1 = pd.read_csv(refout[0])
    ref1 = ref1.drop('Unnamed: 0', 1)
    ref = ref1.merge(ref, on=['sID', 'polePres', 'position'])
    ref.to_csv(refNameUpdate)


    refName = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p'+str(polePres)+'.csv'
    refNameUpdate = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p'+str(polePres)+'_update.csv'
    ref = pd.read_csv(refName)
    tmp = ref['Record_folder'].str.split('_', expand=True)
    ref['sID'] = tmp[0]

    refout = glob.glob(r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2'+os.sep+'*'+str(polePres)+'*.csv')
    ref1 = pd.read_csv(refout[0])
    ref1 = ref1.drop('Unnamed: 0', 1)
    ref = ref1.merge(ref, on=['sID', 'polePres', 'position'])
    ref.to_csv(refNameUpdate)

    return print(refNameUpdate)

#############################################
## run the function to define the threshold
########################################################

##----- RUM2 --------
mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches'
logFile = mainDir+os.sep+'log.log'
logging.basicConfig(filename= logFile, filemode='w', level=logging.INFO)

# run this on the side that has the detection with vertical bar
keyfiles = glob.glob(mainDir+'/*pole_*pres1.npy')
dt = getDataFromGraph_batch(keyfiles, myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p1.csv',  param1 = 'detection', param2=1)

keyfiles = glob.glob(mainDir+'/*pole_*pres2.npy')
dt = getDataFromGraph_batch(keyfiles, myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p2.csv', param1 = 'detection', param2=2)



##----- EMX1-RUM2 --------
mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2\autoDetectTouches'
logFile = mainDir+os.sep+'log.log'
logging.basicConfig(filename= logFile, filemode='w', level=logging.INFO)

keyfiles = glob.glob(mainDir+'/*pole_*pres1.npy')
dt = getDataFromGraph_batch(keyfiles, myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p1.csv',  param1 = 'detection', param2=1)

keyfiles = glob.glob(mainDir+'/*pole_*pres2.npy')
dt = getDataFromGraph_batch(keyfiles, myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_EMX_p2.csv', param1 = 'detection', param2=2)

##################################################################################
##################################################################################
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed for neural data
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
#############  WARNING ON EMX2 SECOND POLE PRESENTATION for samples after 2020-11-10 as output where not computed
##################################################################################
##################################################################################


#############################################
# to get the LED timing
#############################################

# example yes -----_____
# example no  ____------


mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches'
myref = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\touchRefMain_Rum2_p1.csv'

keyfiles = glob.glob(mainDir+'/*pole_*pres1.npy')
param1 = 'detection'
param2 = 2
for i in keyfiles:
    print(i)
    t = graphManualThreshold(i, ref = myref, dataType = param1, e3dat = True, polePres=param2)
    t.getLEDtime()





mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\EMX1-Rum2\autoDetectTouches'
keyfiles = glob.glob(mainDir+'/*pole_*pres1.npy')
param1 = 'detection'
param2 = 2
for i in keyfiles:
    print(i)
    t = graphManualThreshold(i, ref = myref, dataType = param1, e3dat = True, polePres=param2)
    t.getLEDtime()

############ 
## REF combine
############ 
# combine and update the reference files do this by pole presentation 
comboREF(1) # for the first pole presentation
comboREF(2) # for the second pole presentation


########################################################
## notes about button for other application
########################################################

# # import keyboard
# while True:  # this loop wait one second or slightly more
#     if keyboard.is_pressed('p'):
#         pause_keyboard = Trueve=True)
## keyfiles = np.array(keyfiles)[[2,50]] # this is to test different type of conditions

#         print('lo')
#         plt.plot(np.random.rand(10))
#         plt.show()

#     elif keyboard.is_pressed('s'):
#         pause_keyboard = False
#         print('la')
#         plt.plot(np.sin(np.arange(0, 10, 0.1)))
#         plt.show()
#     elif keyboard.is_pressed('q'):
#         break

#         if pause_keyboard:
#             continue



# # use the LED file to narrow down the representation of the trace
# LEDfiles = glob.glob(mainDir+'/*LED_*.npy')
# a = np.load(j)
# ididx = np.where(a[1]>160)[0] # 160 seems to be a general threshold for the image coming on and of
# ididx = [ididx[0], ididx[-1]]
# idtime = a[0][ididx]

# minInter = 4 # minute interval surrounding the led signal 
# fs = 60*500 # acquisition rate of the camer
# finter = fs * minInter

# interval = [[idtime[0]-finter, idtime[0]+finter], [idtime[1]-finter, idtime[1]+finter]]
# interval = np.where(np.isin(a[0], interval))[0]


# plt.plot(a[0][interval[0]:interval[1]], a[1][interval[0]:interval[1]])
# plt.plot(a[0][interval[2]:interval[3]], a[1][interval[2]:interval[3]])
# plt.axvline(idtime[0], color='r')
# plt.axvline(idtime[1], color='r')
# # run this to detect the LED
