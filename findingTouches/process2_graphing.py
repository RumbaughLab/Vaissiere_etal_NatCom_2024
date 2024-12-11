import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy import signal
import os

def getFigurePoleSpace(idi, allfiles, polePres = 1):
    """
    get the figure for the actual output of the mean pixel intensity for the 2 areas which have been determined
    pole and space
    Args:
        i: list of files with identifier
        allfiles: list of all the files

    Returns:
        plot

    Example usage:
        path = 'C:/meanFrame202007'
        allfiles = glob.glob(path + '/*.npy')
        filesPole = glob.glob(path + '/*pole*.npy')
        for i in filesPole:
            getFigurePole(i)

    """
    print(idi)
    files = [x for x in allfiles if str(idi) in x]
    poleDesc = 'pres'+str(polePres)
    files = [x for x in files if poleDesc in x]
    err = np.load([x for x in files if 'err_' in x][0])
    led = np.load([x for x in files if 'LED_' in x][0])
    horizDetect = np.load([x for x in files if 'pole_' in x][0])
    vertDetect = np.load([x for x in files if 'space_' in x][0])


    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)
    ax1.plot(horizDetect[0], horizDetect[1])
    ax2.plot(vertDetect[0], vertDetect[1])
    ax3.plot(err[0], err[1])
    ax4.plot(led[0], led[1])

    ax1.title.set_text('Plot1: pole horizontal detect output')
    ax2.title.set_text('Plot2: space vertical detect output')
    ax3.title.set_text('Plot3: other side output')
    ax4.title.set_text('Plot4: LED output')

    path = os.path.dirname(allfiles[0])
    f.savefig(path+'/fig_'+idi+'pres'+str(polePres)+'.jpg')
    plt.close('all')

def getID(allfiles):
   '''
   get the id number from a file list
   '''
   # get the base name first
   filesbasename = [x.split(os.sep)[-1] for x in allfiles]
   uniqueid = np.unique([x.split('_')[1] for x in filesbasename])

   return uniqueid

########################################################
## Threshold data get touches
########################################################

mainDir = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches'
# get the subdirectory of the directory of interest


allfiles =  glob.glob(mainDir+os.sep+'/*pres2.npy', recursive=True)
uniqueidS = getID(allfiles)
for j in uniqueidS:
  print(j)
  getFigurePoleSpace(j, allfiles, polePres = 2)


