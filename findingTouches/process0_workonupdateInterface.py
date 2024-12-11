from matplotlib.widgets import RectangleSelector, Cursor, Button
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import msvcrt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
import glob
plt.ion()

def getCoordsForAnalysis(inputCoord):
    '''
    input: a string of number with coordinate X Y width height '364 141 7 2'
    '''
    areaPole = [float(s) for s in inputCoord.split(' ')]
    startXp, endXp, startYp, endYp = int(areaPole[0]), int(areaPole[0] + areaPole[2]), int(areaPole[1]),  int(areaPole[1] + areaPole[3])
    return startXp, endXp, startYp, endYp

def shapeConversion(startXp, endXp, startYp, endYp):
    '''
    input: a string of number with coordinate X Y width height '364 141 7 2'
    '''
    coord = (startXp, startYp)
    x1, y1, whidth1, height1
    return coord, whidth1, height1

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    startXp, startYp = eclick.xdata, eclick.ydata
    endXp, endYp = erelease.xdata, erelease.ydata
    # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print('Should detrending be applied (y/n): ')
    yesDetrend = msvcrt.getch() 
    if yesDetrend == b'y':
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (startXp, startYp, endXp, endYp))
        global coords
        coords = []
        coords = startXp, startYp, endXp, endYp
        coords = [int(x) for x in coords]
        widthx=endXp-startXp
        heighty=endYp-startYp
        rect = patches.Rectangle((startXp, startYp), widthx, heighty, linewidth=1,
                         edgecolor='r', facecolor="none")
 
 
        ax0.add_patch(rect)
        print(coords)

    else:
        print('no')

    # print(" The button you used were: %s %s" % (eclick.button, erelease.button))

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)



################ DRAFT USE
## 1 get the image
## the image is initially captured in premiere pro
img = r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\captrue_forTouch\close\25_d_p1_57000-119500.20599.Still001.png"

class imgInfo:

    def __init__(self, img, coords):
        self.imgName = img
        self.coords = coords
        self.keyFrame = img.split(os.sep)[-1].split('.')[1]
        self.imgDat = mpimg.imread(img)
        self.position = img.split(os.sep)[-2]
        self.vid = glob.glob(os.sep.join(img.split(os.sep)[:-3])+'/*'+self.position+'*/*'+img.split(os.sep)[-1].split('.')[0]+'*.mp4')[0]
        self.interval = range(int(self.keyFrame)-100, int(self.keyFrame)+200)


    def detectArea(self):
        vcap = cv2.VideoCapture(self.vid)
        vidFMain=[]
        meanFramePole=[]
        startXp, startYp, endXp, endYp = self.coords
        # for vidF in range(int(j['stablePole']), int(vcap.get(7))):
        # for vidF in range(0, 2):
        for vidF in self.interval:
            # print(vidF)
            vcap.set(1, vidF)
            ret, frame = vcap.read()
            framePole = frame[startYp:endYp, startXp:endXp]
            meanFramePole.append(np.mean(framePole))
            vidFMain.append(vidF)
        vcap.release()
        toSavePole = np.array([vidFMain, meanFramePole])
        return toSavePole






t = imgInfo(img, coords)
dat = t.detectArea()
plt.plot(dat[0],dat[1])




fig = plt.figure(figsize=(12, 6))
gs = GridSpec(nrows=2, ncols=2)
## https://towardsdatascience.com/plot-organization-in-matplotlib-your-one-stop-guide-if-you-are-reading-this-it-is-probably-f79c2dcbc801
img = mpimg.imread(r"C:\Users\Windows\Downloads\touchRef_close-02.jpg")
gs = GridSpec(nrows=4, ncols=4)

## plt the main figure
ax0 = fig.add_subplot(gs[:3, :3])
ax0.imshow(img)

## plt the subfigure of the zoom afte the area has been drawn
# ax1 = fig.add_subplot(gs[:3, -1])
# ax1.imshow(img[...,:3][startYp:endYp, startXp:endXp])


## plt the data surronding the frame of interest


# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(ax0, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
plt.connect('key_press_event', toggle_selector)
# cursor = Cursor(ax0)
# fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
# fig.canvas.mpl_connect('button_press_event', cursor.onclick)
plt.show()


startXp, startYp, endXp, endYp = coords
plt.figure()
plt.imshow(img[startYp:endYp, startXp:endXp])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
plt.figure()
plt.imshow(color.rgb2gray(img))

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)