mPath = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST"
files = glob.glob(mPath+os.sep+'*.csv')
ids = np.unique([x.split(os.sep)[-1].split('_')[0] for x in files])

allf = []
for i in files:
    tmp = pd.read_csv(i)
    allf.append(tmp)
allf = pd.concat(allf)
allf['duration_time_touch'] = allf['touchCount']/500
allf['duration_time_interEvent'] = allf['interEventinter']/500

tmp = allf.groupby(['sID']).agg({'duration_time_touch':[np.mean, 'count'],'duration_time_interEvent':np.mean})
c = quickConversion(tmp, option=2)
c = c.rename(columns = {'value':'touch_new(mean_s)'})
geno = tpath('Vaissiere/fileDescriptor2.csv')
geno = getGenotype(geno)

c = pd.merge(c, geno, on='sID') ## narrow the data to the data for which relevant animals have been included in the ephys study 

## data from the old data
sID = ids[0]
myDat = comboBehavior(sID) 
b = myDat.getcombinedAnalogMetrics()
oldTouchDat = copy.copy(b)
b = pd.read_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__\curvature\curvatureAll.csv')
b = b[['onset_time_touch','duration_time_touch','sID','geno']].dropna().reset_index()
b = quickConversion(b.groupby(['sID','geno']).agg({'duration_time_touch':[np.mean]}))
b = b.rename(columns = {'value':'touch_old(mean_s)'})

### not important the 

c = pd.merge(b,c, how='inner')
c['delta'] = c['touch_new(mean_s)']-c['touch_old(mean_s)']
c['touch_new(mean_s)'] = c['touch_new(mean_s)']*1000
c = c.rename(columns = {'touch_new(mean_s)':'Touch Duration (avg in ms)'})

mergedDat = pd.merge(allf, b, on = 'sID', how='outer')






### for the behaviro 2019-04-30 need to be excluded as most of the 
### behavior consist of a lot of retraction
c = quickConversion(allf.groupby(['sID']).agg({'duration_time_touch':[np.mean]}))
c = c.rename(columns = {'value':'Touch Duration (avg in ms)'})
c = pd.merge(b,c, how='inner')
c = c[c['sID']!='2019-04-30'] ### need to remove this sample
figparam('KO')
customPloting(c, 'Touch Duration (avg in ms)', 'Animal_geno', custylim=[0,50])
getStat(c, 'Touch Duration (avg in ms)', groupVar = 'Animal_geno', group = ['wt', 'het'], param = True)


outpath = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'
plt.savefig(outpath+os.sep+'#TV0800.pdf')

### to get the data for the count
##########################################################################
c = quickConversion(allf.groupby(['sID']).agg({'duration_time_touch':['count']}))
c = c.rename(columns = {'value':'Touches (n)'})
c = pd.merge(b,c, how='inner')
c = c[c['sID']!='2019-04-30'] ### need to remove this sample due to retraction whisker only animal with this extensive behavior
customPloting(c, 'Touches (n)', 'geno', custylim=[0,4000])
getStat(c, 'Touches (n)', groupVar = 'geno', group = ['wt', 'het'], param = True)
outpath = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'
plt.savefig(outpath+os.sep+'#TV0801.pdf')

### to get the data for inter events
##########################################################################
c = quickConversion(allf.groupby(['sID']).agg({'interEventinter':[np.mean]}))
c = c.rename(columns = {'value':'Interevent duration (avg in ms)'})
c = pd.merge(b,c, how='inner')
c = c[c['sID']!='2019-04-30'] ### need to remove this sample due to retraction whisker only animal with this extensive behavior
customPloting(c, 'Interevent duration (avg in ms)', 'geno', custylim=[0,1000])
getStat(c, 'Interevent duration (avg in ms)', groupVar = 'geno', group = ['wt', 'het'], param = False)
outpath = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'
plt.savefig(outpath+os.sep+'#TV0802.pdf')


### cumulative plot
############################################################################
allf_filt = allf[~allf['sID'].isin(['2019-04-30','2019-04-02', '2019-04-04', '2019-11-14'])]
allf_filt = pd.merge(allf_filt,geno, on='sID')
allf_filt = allf_filt.sort_values('Animal_geno', ascending=False)
sns.kdeplot(data=allf_filt, x="duration_time_touch", hue="Animal_geno",cumulative=True, common_norm=False, common_grid=True,levels=100)
sns.ecdfplot(data=allf_filt, x='duration_time_touch',  hue='Animal_geno', stat='proportion', linewidth=0.4 )
plt.ylim([0,1])
plt.xlim([0,0.2])
plt.ylabel('Proportion')
plt.xlabel('Touch Duration (s)')
plt.savefig(outpath+os.sep+'#TV0803.pdf')



getStat(allf_filt, 'duration_time_touch', 'Animal_geno', param=False)

from scipy import stats
wtVal = allf_filt.loc[allf_filt['Animal_geno']=='wt',"duration_time_touch"].values
hetVal = allf_filt.loc[allf_filt['Animal_geno']=='het',"duration_time_touch"].values
statistic, pvalue = scipy.stats.ks_2samp(wtVal, hetVal)



## ************************************
## plot the distribution
## ************************************
params, paramsNest, nobs, nmax = paramsForCustomPlot(data=allf_filt, variableLabel="Animal_geno", valueLabel='duration_time_touch', sort=True)

mpl.rcParams['figure.figsize'] = [1.5, 2]
fig, ax = plt.subplots()
# fig = plt.figure()

## VIOLIN pPLOT
ax = sns.stripplot(**paramsNest, edgecolor = 'k', linewidth = 0.4,  size=2, zorder=3, ax = ax, dodge = True, jitter = 0.15, alpha=0.5)
axv = sns.violinplot(split = False, width=0.6, cut=1, **paramsNest, inner='quartile', zorder=2, linewidth = 0.5, dodge=False, ax = ax)

ax.legend_.remove()
plt.tight_layout()
outpath = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'
plt.savefig(outpath+os.sep+'#TV0804.png')




#### check out potential correlations
####################################################
allf_filt = allf[~allf['sID'].isin(['2019-04-30','2019-04-02', '2019-04-04', '2019-11-14'])]
tmp = allf_filt.groupby(['sID']).agg({'duration_time_touch':[np.mean, 'count'],'duration_time_interEvent':np.mean})
c = quickConversion(tmp, option=2)
c = pd.merge(c,geno, on='sID')
c = c.sort_values('Animal_geno', ascending=False)

facL = ['duration_time_touch_mean','duration_time_touch_count','duration_time_interEvent_mean']
sns.lmplot(x=facL[0], y=facL[1], hue="Animal_geno", data=c)
sns.lmplot(x=facL[0], y=facL[2], hue="Animal_geno", data=c)
sns.lmplot(x=facL[1], y=facL[2], hue="Animal_geno", data=c)
