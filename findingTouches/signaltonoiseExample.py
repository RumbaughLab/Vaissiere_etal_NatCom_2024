
a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches-TEST\pole_2020-06-10_17-12-21 - pole noise__2020-06-10_HS_pres1.npy"
b = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches\pole_2020-06-10_17-12-21 - pole noise__2020-06-10_HS.npy"


a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches\pole_2020-06-18_16-13-30__2020-06-18_HS.npy"
b = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\Rum2\autoDetectTouches-TEST\pole_2020-06-18_16-13-30__2020-06-18_HS_pres1.npy"

a= np.load(a)
b= np.load(b)

ran = np.arange(32000,325000)
a = a[:,ran]
b = -b[:,ran]


norma = a[1]-a[1].mean()
normb = b[1]-b[1].mean()

f, ax = plt.subplots(2,2, sharex=False)
ax[0][0].plot(np.arange(len(a[1])), norma)
ax[1][0].plot(np.arange(len(a[1])), normb)

ax[0][1].plot(np.arange(len(a[1][90000:114000])), norma[90000:114000])
ax[1][1].plot(np.arange(len(a[1][90000:114000])), normb[90000:114000])



f, ax = plt.subplots(2,3, sharex=False, sharey=True)
ax[0][0].plot(np.arange(len(a[1]))/500, norma)
ax[1][0].plot(np.arange(len(b[1]))/500, normb)

ax[0][1].plot(np.arange(len(a[1][90000:114000]))/500, norma[90000:114000])
ax[1][1].plot(np.arange(len(b[1][90000:114000]))/500, normb[90000:114000])
ax[0][2].plot(np.arange(len(a[1][98000:103000]))/500, norma[98000:103000])
ax[1][2].plot(np.arange(len(b[1][98000:103000]))/500, normb[98000:103000])



f, ax = plt.subplots(2,3, sharex=False, sharey=False)
ax[0][0].plot(np.arange(len(a[1])), norma)
ax[1][0].plot(np.arange(len(b[1])), normb)

ax[0][1].plot(np.arange(len(a[1][90000:114000])), norma[90000:114000])
ax[1][1].plot(np.arange(len(b[1][90000:114000])), normb[90000:114000])
ax[0][2].plot(np.arange(len(a[1][98000:103000])), norma[98000:103000])
ax[1][2].plot(np.arange(len(b[1][98000:103000])), normb[98000:103000])




plt.xlim([770000, 780000])

a[0:200,0:200]

a[:,ran]


plt.close('all')

f, ax = plt.subplots(2,2, sharex=True)
ax[0][0].plot(np.arange(len(a[1])), norma)
ax[1][0].plot(np.arange(len(a[1])), normb)

def rmsn(inputval):
    rmsn = np.sqrt(np.mean(inputval**2))
    return rmsn

def p2pn(inputval):
    p2p = max(inputval)-min(inputval)
    namp = min(inputval)
    return p2p, namp


noisea = norma[np.r_[94500:98337,98610:99937,100094:101850,102865:105558,111677:113500]]
noiseb = normb[np.r_[94500:98337,98610:99937,100094:101850,102865:105558,111677:113500]]

plt.figure()
plt.plot()

