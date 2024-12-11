##### 
behaviorCombo = pd.read_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__\curvature\curvatureAll.csv')
## narrow down to the 
# https://medium.com/@sk.shravan00/k-means-for-3-variables-260d20849730


tmpDat = behaviorCombo[~behaviorCombo['touchIdx'].isnull()]

test = tmpDat[['duration_time_whisk','aucCurv_touch']]
test = test.iloc[:,:].values

## using the elbow methods to determine the number of clusters
from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
    k_means.fit(test)
    wcss.append(k_means.inertia_)


plt.plot(np.arange(1,11),wcss)
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.show()


k_means_optimum = KMeans(n_clusters = 2, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(test)
tmpDat['kmeansCat'] = y

####################################################
## cat kmeans based on the touch time
####################################################
def getCategoricalplot(tmpDat, variable, custName = 'MS600_2_'):
	'''
	summary is the data frame of interest
	in the custName used: 'MS600_2_'

	'''
	summary = quickConversion(tmpDat.groupby(['sID', 'geno', 'cat_percentile_aucCurv_touch', variable]).agg({variable:['count']}))
	summaryAll = quickConversion(tmpDat.groupby(['sID', 'geno']).agg({variable:['count']}))
	summaryAll = summaryAll.rename(columns={'value': 'total'})
	summary = pd.merge(summary, summaryAll, on = ['sID','geno'])
	summary['ratio'] = summary['value']/summary['total']

	variable = variable.replace('_','')
	for i in np.unique(summary[variable]):
		tmpSum = summary[summary[variable]==i]

		try:
			params, paramsNest, nobs, nmax = paramsForCustomPlot(tmpSum , variableLabel='geno', subjectLabel='sID', valueLabel='ratio')
			customPlot(params,paramsNest, dirName=r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels', 
										  figName = 'MS600_2', 
										  # myylim = [0,5], 
										  myfsize = [3,6],
										  myy='Cat' +str(i) + 'kmeans. \n(AUC curvature)')
			plt.title('cat '+str(i))
			plt.tight_layout()
			plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+custName+str(i)+'.pdf') #'MS600_2_'
		except:
			print('error : ' + str(i))

getCategoricalplot(tmpDat, 'cat_kmeans')
####################################################
## cat kmeans based on the touch time
####################################################
## see Y:\2020-09-Paper-ReactiveTouch\__code__\whisking\pythonTouchRETRIVAL.py

getCategoricalplot(tmpDat=tmpDat, variable='cat_percentile')


####################################################
## cat kmeans based on the auc_curvature
####################################################
## see Y:\2020-09-Paper-ReactiveTouch\__code__\whisking\pythonTouchRETRIVAL.py
percentile_list = [100, 75,50,25]
# percentile_list = [100, 66, 33]

for i in percentile_list:
    tmpreturn = np.percentile(tmpDat['aucCurv_touch'], i)
    print(tmpreturn)
    tmpDat.loc[tmpDat['aucCurv_touch']<tmpreturn,'cat_percentile_aucCurv_touch'] = i


tmpDat['cat_percentile_auc_touch'] = 75
getCategoricalplot(tmpDat=tmpDat, variable='cat_percentile_aucCurv_touch')

getCategoricalplot(tmpDat=tmpDat, variable='kmeansCat')

tmpDat.groupby(['cat_percentile_aucCurv_touch']).agg({'aucCurv_touch':[np.mean]})

percentile_list = [100, 75,50,25]
mainFolder = r'Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\ThyChR2\autoDetectTouches-TEST\touchTimes'
for i in percentile_list:
	destFold = mainFolder + os.sep + str(i)
	os.makedirs(destFold, exist_ok=True)
	for j in np.unique(tmpDat.sID):
		tmp = tmpDat.loc[(tmpDat['sID']==j)&(tmpDat['cat_percentile_aucCurv_touch']==i),'onset_time_touch'].values
		np.save(destFold+os.sep+j+'.npy', tmp)




####################################################
## check the curvature cumulative probability for every touch
####################################################
tmpDat = tmpDat.sort_values(by='geno', ascending=False)
sns.ecdfplot(data=tmpDat, x='auc_run', hue='geno')

tmpDat = tmpDat.sort_values(by='geno', ascending=False)
sns.ecdfplot(data=tmpDat, x='duration_time_run', hue='geno')


##########################################################
## establishing the measure per sub category and as a whole  with correlation as well
##########################################################
def customPlot(params, paramsNest, dirName='C:/Users/Windows/Desktop/MAINDATA_OUTPUT', figName = ' ', tradMean = True, viewPlot=False, showSample=True, **kwargs):
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['#6e9bbd', '#e26770'])
    """Function to create save the plot to determine directior
    
    Parameters:
        params (dict): set of parameters for plotting main data
        paramsNest (dict): set of parameters for plotting main data (subNesting level)
        dirName(str): string to determine the directory
        tradMean(bool): True, determine if mean std and confidence interval are to be plotted

    kwargs:
        myfsize: list with fig dim 
        myylim: list with y axis limits
        myy: name of the y axis
    """

    # create the frame for the figure
    os.makedirs(dirName,exist_ok=True)

    # the figure size correspond to the size of a plot in inches
    myfsize = kwargs.get('myfsize', None)
    if myfsize is not None:
        f, ax = plt.subplots(figsize=myfsize)
    else:
        f, ax = plt.subplots(figsize=(7, 7))

    ## add if the study was longitudinal / repeated measures
    # castdf=pd.pivot_table(df, values='value', index=['subject'], columns=['genotype'])
    # for i in castdf.index:
    #     ax.plot(['wt','het'], castdf.loc[i,['wt','het']], linestyle='-', color = 'gray', alpha = .3)

    # fill the figure with appropiate seaborn plot
    # sns.boxplot(dodge = 10, width = 0.2, fliersize = 2, **params)
    sns.violinplot(split = True, inner = 'quartile', width=0.6, cut=1, **paramsNest)

    if showSample == True:
	    custSpike = kwargs.get('custSpike', None)
	    if custSpike is not None:
	        plt.setp(ax.collections, alpha=.2)
	        sns.stripplot(jitter=0.2, dodge=True, edgecolor='white', size=4, linewidth=0, alpha=0.5, **paramsNest) # change this linewidth = 1 by default
	    else:
	        sns.stripplot(jitter=0.08, dodge=True, edgecolor='white', size=4, linewidth=0, alpha=1, **paramsNest) # change this linewidth = 1 by default
	        plt.setp(ax.collections, alpha=.2)

    # control the figure parameter with matplotlib control
    # this order enable to have transparency of the distribution violin plot

    if tradMean == True:
        # the point plot enable to plot the mean and the standard error 
        # to have the "sd" or 95 percent confidence interval 
        # for sem ci=68
        sns.pointplot(errorbar=('ci', 68), scale=1.2, dodge= -0.1, errwidth=4, **params)
        sns.pointplot(errorbar=('ci', 95), dodge= -0.1, errwidth=2, **params)
        # plot the median could be done with the commented line below however this would be redundant 
        # since the median is already ploted in the violin plot
        # sns.pointplot(ci=None, dodge= -0.2, markers='X',estimator=np.median, **params)
    
    custSpike = kwargs.get('custSpike', None)
    if custSpike is not None:
        sns.stripplot(jitter=0.00, dodge=False, edgecolor='black', alpha=0.5, size=7, linewidth=0.4, **params)
    else:
        sns.stripplot(jitter=0.08, dodge=True, edgecolor='white', size=8, linewidth=1, **params)


    # label plot legend and properties
    ax.legend_.remove()
    sns.despine() 

    myylim = kwargs.get('myylim', None)
    if myylim is not None:
        ax.set_ylim(myylim)

    mytitle = kwargs.get('mytitle', None)
    if mytitle is not None:
        ax.set_title(mytitle)

    ax.set_ylabel(params.get('y'), fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    # ax.set_ylim([-5,5])

    ax.set_xlabel(params.get('x'), fontsize=30)
    ax.tick_params(axis="x", labelsize=20, pad=10) # could also use: ax.xaxis.labelpad = 10 // plt.xlabel("", labelpad=10) // or change globally rcParams['xtick.major.pad']='8'


    ### to obligate viewing of the plot
    if viewPlot == True:
        plt.show(block=False)

    myy = kwargs.get('myy', None)
    if myy is not None:
        ax.set_ylabel(myy)
    
    plt.tight_layout()
    # property to export the plot 
    # best output and import into illustrator are svg 
    return plt.savefig(dirName+os.sep+figName+".pdf")#,     plt.show(block=False)


def getCategoricalplot(tmpDat, variable, custName = 'MS600_2_'):
	'''
	summary is the data frame of interest
	in the custName used: 'MS600_2_'

	'''
	summary = quickConversion(tmpDat.groupby(['sID', 'geno', 'cat_percentile_aucCurv_touch']).agg({variable:[np.mean]}))
	summaryAll = quickConversion(tmpDat.groupby(['sID', 'geno']).agg({variable:['count']}))
	summaryAll = summaryAll.rename(columns={'value': 'total'})
	summary = pd.merge(summary, summaryAll, on = ['sID','geno'])
	summary['ratio'] = summary['value']/summary['total']

	variable = variable.replace('_','')
	for i in np.unique(summary[variable]):
		tmpSum = summary[summary[variable]==i]

		try:
			params, paramsNest, nobs, nmax = paramsForCustomPlot(tmpSum , variableLabel='geno', subjectLabel='sID', valueLabel='ratio')
			customPlot(params,paramsNest, dirName=r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels', 
										  figName = 'MS600_2', 
										  # myylim = [0,5], 
										  myfsize = [3,6],
										  myy='Cat' +str(i) + 'kmeans. \n(AUC curvature)')
			plt.title('cat '+str(i))
			plt.tight_layout()
			plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+custName+str(i)+'.pdf') #'MS600_2_'
		except:
			print('error : ' + str(i))


variable = ['duration_time_touch','aucCurv_touch', 'peakCurv_touch']

for ij in variable:
	percentile_list = [100, 75,50,25]
	for jj in percentile_list:
		tmpSum = tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj]
		params, paramsNest, nobs, nmax = paramsForCustomPlot(tmpSum , variableLabel='geno', subjectLabel='sID', valueLabel=ij)
		customPlot(params,paramsNest, dirName=r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels', 
									  figName = 'MS602_2'+ij+'_'+str(jj), 
									  # myylim = [0,5], 
									  myfsize = [3,6],
									  myy='Cat_' +str(jj) + ij,
									  mytitle = jj)

		customPlot(params,paramsNest, dirName=r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels', 
								  figName = 'MS602_2', 
								  # myylim = [0,5], 
								  myfsize = [3,6],
								  myy='Cat_' +str(jj) + variable)

for jj in percentile_list:
	plt.figure()
	sns.lmplot(x=variable[0], y=variable[1], hue='geno', data=tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj],scatter_kws={"s": 10, "alpha":0.3})
	plt.title(str(jj))
	plt.savefig('corr_'+str(jj)+variable[0]+'___'+variable[1]+'.pdf')

for jj in percentile_list:
	plt.figure()
	sns.lmplot(x=variable[0], y=variable[2], hue='geno', data=tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj],scatter_kws={"s": 10, "alpha":0.3})
	plt.title(str(jj))
	plt.savefig('corr_'+str(jj)+variable[0]+'___'+variable[2]+'.pdf')

for jj in percentile_list:
	plt.figure()
	sns.lmplot(x=variable[1], y=variable[2], hue='geno', data=tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj],scatter_kws={"s": 10, "alpha":0.3})
	plt.title(str(jj))
	plt.savefig('corr_'+str(jj)+variable[1]+'___'+variable[2]+'.pdf')




variable = ['duration_time_touch','aucCurv_touch', 'peakCurv_touch']
for ij in variable:
	percentile_list = [100, 75,50,25]
	for jj in percentile_list:
		tmpSum = tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj]
		
		tmp = pd.DataFrame({'percentile category': [jj],'variable':[ij]})

		formula = ij + " ~ geno + sID"
		model = sm.MixedLM.from_formula(formula, groups="geno", data=tmpSum)
		result = model.fit().summary()
		print(result)
		pvaltmp = result.tables[1]

		tmp['p_value'] = pvaltmp['P>|z|'][1]



		## table format for prism
		tmpSum = tmpDat[tmpDat['cat_percentile_aucCurv_touch']==jj]
		tmpSum = tmpSum[['geno',ij,'sID']]
		tmpSum = tmpSum.sort_values('geno',ascending=False)
		tmpSum = tmpSum.pivot(columns=['geno','sID'])
		tmpSum = tmpSum.transform(np.sort)
		tmpSum.reset_index(inplace=True,drop=True)
		tmpSum = tmpSum.dropna(axis=0, how='all')
		tmpSum.to_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__\forPrism'+os.sep+'TV0600_1_'+str(jj)+'_'+ij+'.csv')


    # for i in castdf.index:

#### get the stat 
#### see this file "Y:\2020-09-Paper-ReactiveTouch\stats.R"
#### might be worth comparing and contrasting this with the prism files 
