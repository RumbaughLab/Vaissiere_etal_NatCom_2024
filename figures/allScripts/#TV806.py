
def quickConversion(tmp):
    tmp = tmp.reset_index()
    if tmp.columns.nlevels > 1:
        tmp.columns = ['_'.join(col) for col in tmp.columns] 
    # tmp.columns = tmp.columns.str.replace('[_]','')
    return tmp



# ####################################
## GET correlation pump occurence
####################################
## retrive all the orginal data
files = glob.glob(r"Y:\Sheldon\Highspeed\analyzed\NewExport20210209\*.csv")[1:]
# allDat = pd.read_csv(r"Y:\Sheldon\Highspeed\analyzed\NewExport20210209\WDIL002_0005-SyngapKO.csv")
# allDat = pd.read_csv(r'Y:\\Sheldon\\Highspeed\\analyzed\\NewExport20210209\\WDIL003_HS0009-SyngapRUM1.csv')
# allDat = pd.read_csv(r'Y:\\Sheldon\\Highspeed\\analyzed\\NewExport20210209\\WDIL004_0008-SyngapRescueEMXRUM2.csv')

genoType = ['KO', 'CD', 'CR']
mainName = r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+"TV806"

for i, k in zip(files, genoType):
    print(i)
    figparam(expType = k, restoreDefault = False)
    allDat = pd.read_csv(i)
    geno = pd.read_csv(r"Y:\Sheldon\Highspeed\analyzed\NewExport20210209\geno.csv")
    allDat = pd.merge(allDat, geno, on = ['aid', 'cohort'], how = 'left')
    allDat['timeMS'] = allDat['frame']/500*1000 # convert to  ms```
    touchOccurence = allDat.groupby(['geno', 'aid', 'touchGrp', 'peakCatAmp']).agg({'peakCatAmp': [np.ma.count], 'frame': ['first']})
    touchOccurence = quickConversion(touchOccurence, option=2)
    touchOccurence = copy.copy(touchOccurence)
    touchOccurence = touchOccurence[touchOccurence['peakCatAmp'] == 'prot']
    touchOccurence['touchGrp'] = touchOccurence['touchGrp']/2 # devide by 2 to get actual order of occurence otherwise multiplied by 2 due to set up
    touchOccurence['frame_first'] = touchOccurence['frame_first']/500
    # color management during iteration

    ### Peak protraction per touch number above threshold
    ratio = touchOccurence[touchOccurence['peakCatAmp_count'] >2.5] # 2.5 threshold 
    ratio = ratio.sort_values(by=['geno_'], ascending=False)
    customPloting(ratio, 'peakCatAmp_count' ,genoOfInt = 'geno_', plottick = 4, custylim=[])
    print(i)
    dtype = 'abovThresh'
    print(dtype)
    getStat(ratio, 'peakCatAmp_count', groupVar = 'geno_', param=False)
    plt.tick_params(axis='y', labelsize=16)
    plt.ylabel('Peak prot. thresh. (n)', fontsize=18)
    fig = plt.gcf()
    fig.set_size_inches([1.3, 2])
    plt.tight_layout()
    plt.savefig(mainName+'_'+k+'_'+dtype+'.pdf')
    
    ### Peak protraction per touch number
    ratio = touchOccurence # 2.5 threshold 
    ratio = ratio.sort_values(by=['geno'], ascending=False)
    customPloting(ratio, 'peakCatAmp_count' ,genoOfInt = 'geno', plottick = 4, custylim=[])
    dtype = 'all'
    print(dtype)
    getStat(ratio, 'peakCatAmp_count', groupVar = 'geno', param=False)
    plt.tick_params(axis='y', labelsize=16)
    plt.ylabel('Peak prot. (n)', fontsize=18)
    if k == 'CR':
        plt.ylim([0,23])
    fig = plt.gcf()
    fig.set_size_inches([1.3, 2])
    plt.tight_layout()
    plt.savefig(mainName+'_'+k+'_'+dtype+'.pdf')

    ### Average Peak protraction per touch number
    ratio = touchOccurence 
    ratio = ratio.groupby(['geno', 'aid']).agg({'peakCatAmp_count':[np.mean]})
    ratio = quickConversion(ratio, option=2)
    customPloting(ratio, 'peakCatAmp_count_mean' ,genoOfInt = 'geno', plottick = 4, custylim=[])
    dtype = 'byAnimal'
    print(dtype)
    getStat(ratio, 'peakCatAmp_count_mean',  groupVar = 'geno', param=False)
    plt.tick_params(axis='y', labelsize=16)
    plt.ylabel('Peak prot. (n)', fontsize=18)
    # plt.ylabel('')
    if k == 'CR':
        plt.ylim([0,3.3])
    fig = plt.gcf()
    fig.set_size_inches([1.1, 2])
    plt.tight_layout()
    plt.savefig(mainName+'_'+k+'_'+dtype+'.pdf')














setx = 'framefirst'
sety = 'peakCatAmpcount'

f, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = [12,5])
for i in touchOccurence['aid'].unique():
    # print(i)
    tmp = touchOccurence[touchOccurence['aid'] == i]
    if tmp['geno'].iloc[0] == 'wt':
        p = ax[0].plot(tmp[setx], tmp[sety], alpha = 0.3, label = i)
        ax[0].plot(tmp[setx], tmp[sety], '.', color = p[0].get_color())
    else:      
        q = ax[1].plot(tmp[setx], tmp[sety], alpha = 0.3, label = i)
        ax[1].plot(tmp[setx], tmp[sety], '.', color = q[0].get_color())

ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'animalID')
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'animalID')
plt.tight_layout()

# https://stackoverflow.com/questions/20288842/matplotlib-iterate-subplot-axis-array-through-single-list/20289530
for axs in ax.reshape(-1):
    axs.set_xlabel('tst')
    axs.hlines([2.5], 0, touchOccurence[setx].max(), color = 'grey', linestyle = '--')

for axs in zip(ax.reshape(-1), ['wt','het']):
    print(axs)
    axs[0].set_xlabel('Timestamp (s)')
    axs[0].set_title(axs[1])
ax[0].set_ylabel('"Pumps" per touch (n)')
f.suptitle('data for CD line')
plt.tight_layout()
plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'CDTimeStamp'+'.svg')


# ####################################
## GET ratio per animal
####################################
# set limit at 60 frames 0.12 seconds

ratio = touchOccurence[touchOccurence['peakCatAmpcount'] >2.5] # 2.5 threshold 
ratio = ratio.sort_values(by=['geno'], ascending=False)

sns.displot(ratio, x="framefirst", hue="geno", kind = 'kde', fill = True, height=1.5, aspect=1.5/1.4)
# plt.xlim([0, ratio['frame_first'].max()])
plt.xlabel('Timestamp (s)')
plt.ylabel('Density (KDE)')
plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'CDKDEall'+'.svg')
# sns.displot(ratio, x="frame_first", hue="geno_", kde = True, fill = True, bins=20)
# comment: good ref see last point with other article https://stackoverflow.com/questions/51666784/what-is-y-axis-in-seaborn-distplot https://towardsdatascience.com/histograms-and-density-plots-in-python-f6bda88f5ac0

ratio = ratio.groupby(['geno', 'aid']).agg({'aid':[np.ma.count]})
ratio = quickConversion(ratio)


allID = quickConversion(touchOccurence.groupby(['geno'])['aid'].nunique())
subID = quickConversion(ratio.groupby(['geno'])['aid'].nunique())

figparam()
customPloting(ratio, 'aid__count' ,genoOfInt = 'geno__', plottick = 4, custylim=[])
plt.ylabel('"Pumps" (n) per animal')
plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'CDGrp'+'.svg')
figparam(True)




ratio = touchOccurence.groupby(['geno_', 'aid_']).agg({'touchGrp_': [np.ma.count]})
ratio = quickConversion(ratio)
customPloting(ratio, 'touchGrp__count' ,genoOfInt = 'geno__', plottick = 4, custylim=[])
plt.ylabel('Touch (n)')
plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'CDtouchN'+'.svg')


# ####################################
## Looking at none parametric way
######################################
# set limit at 60 frames 0.12 seconds
touchOccurence = allDat.groupby(['geno', 'aid', 'touchGrp', 'peakCatAmp']).agg({'peakCatAmp': [np.ma.count], 'frame': ['first']})
touchOccurence = quickConversion(touchOccurence)
touchOccurence = touchOccurence[touchOccurence['peakCatAmp_'] == 'prot']
touchOccurence['touchGrp_'] = touchOccurence['touchGrp_']/2 # devide by 2 to get actual order of occurence otherwise multiplied by 2 due to set up
touchOccurence['frame_first'] = touchOccurence['frame_first']/500
# color management during iteration
ratio = touchOccurence[touchOccurence['peakCatAmp_count'] >2.5] 


customPloting(ratio, 'peakCatAmp_count' ,genoOfInt = 'geno_', plottick = 4, custylim=[])
params, paramsNest, nobs, nmax = paramsForCustomPlot(data = ratio,  variableLabel = 'geno_', valueLabel = 'peakCatAmp_count')

figparam('CR')
fig, ax = plt.subplots(figsize=[1,2])
sns.stripplot(**paramsNest, edgecolor = 'k', linewidth = 0.4,  size=2, ax = ax, dodge = False, jitter = 0.1, zorder = 2, alpha  = 0.3)
sns.violinplot(split = False, width=0.6, cut=1, **paramsNest, inner='quartile', linewidth = 0.5, dodge=False, ax = ax, zorder = 1)
ax.set_alpha(0.1)
ax.legend_.remove()
plt.tight_layout()
plt.ylabel('"Pumps" (n) per touch')
plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'CRperTouch'+'.svg')


def getStat(data, measure, groupVar = 'geno_', group = ['wt', 'het']):
    x = data.loc[data[groupVar] == group[0], measure] 
    y = data.loc[data[groupVar] == group[1], measure] 
    stat = pg.mwu(x, y, tail='one-sided')
    print(stat)
    if stat['p-val'].item() < 0.001:
        print('***')
    elif stat['p-val'].item() < 0.01:
        print('**')
    elif stat['p-val'].item() < 0.05:
        print('***')
    
    

stat = getStat(ratio, 'peakCatAmp_count')



model1 = sm.MixedLM.from_formula("peakCatAmp_count ~ 1", re_formula="1", vc_formula={"geno_": "0 + C(geno_)"},
                groups="aid_", data=ratio)
result1 = model1.fit()
print(result1.summary())
# ####################################
## GET correlation pump occurence
####################################
## not the best representation thus dropped

# touchOccurence = allDat.groupby(['geno', 'aid', 'touchGrp', 'touchCat']).agg({'touchCat': [np.ma.count]})
# touchOccurence = quickConversion(touchOccurence)
# touchOccurence = touchOccurence[touchOccurence['touchCat_'] ==1]
# touchOccurence['touchGrp_'] = touchOccurence['touchGrp_']/2 # devide by 2 to get actual order of occurence otherwise multiplied by 2 due to set up
# touchOccurence['touchCat_count'] = touchOccurence['touchCat_count']/500
# # color management during iteration

# setx='touchGrp_'
# sety = 'touchCat_count'

# f, ax = plt.subplots(1,2, sharex = True, sharey = True, figsize = [12,5])
# for i in touchOccurence['aid_'].unique():
#     # print(i)
#     tmp = touchOccurence[touchOccurence['aid_'] == i]
#     if tmp['geno_'].iloc[0] == 'wt':
#         p = ax[0].plot(tmp[setx], tmp[sety], alpha = 0.3, label = i)
#         ax[0].plot(tmp[setx], tmp[sety], '.', color = p[0].get_color())
#     else:      
#         q = ax[1].plot(tmp[setx], tmp[sety], alpha = 0.3, label = i)
#         ax[1].plot(tmp[setx], tmp[sety], '.', color = q[0].get_color())

# ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'animalID')
# ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title = 'animalID')
# plt.tight_layout()

# # https://stackoverflow.com/questions/20288842/matplotlib-iterate-subplot-axis-array-through-single-list/20289530
# for axs in ax.reshape(-1):
#     axs.set_xlabel('tst')
#     axs.hlines([0.12], 0, touchOccurence[setx].max(), color = 'grey', linestyle = '--')

# for axs in zip(ax.reshape(-1), ['wt','het']):
#     print(axs)
#     axs[0].set_xlabel('occurence')
#     axs[0].set_title(axs[1])
# ax[0].set_ylabel('Touch duration (s)')
# f.suptitle('data for KO line')
# plt.tight_layout()
# plt.savefig(r'C:\Users\Windows\Desktop\### WISKER ###\touchOnly'+os.sep+'KOoccurence'+'.svg')
