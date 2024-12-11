files = glob.glob(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__"+os.sep+"*percentile*.npy")

KEYTHRESHVAL = 1.5 # this is the key threshold to determin if unit is responsive or not

def valueExtraction(myarray, iter, window =[-0.025, 0.050], plot=False, binsize = 0.001):

    binTemplate = np.arange(window[0], window[1] + binsize, binsize)
    respWin = np.where((binTemplate>=0) & (binTemplate<0.03))
    baselineWin = np.where(binTemplate<0)

    ### test val 
    ### consider using this
    ### https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html
    trueBaseline = baselineWin[0][6:] # this is to limit due to gaussian smoothing artifacts

    ###################################### This was changed to match the responder definition

    ####$$$$$$$$$$$$$$$$
    #### PEAKS
    ####$$$$$$$$$$$$$$$$

    ### narrowing down the peaks to after the pass of stim
    ### change from the original which was this to accomodate for het dat which 
    ### appears to be earlier
    # stim = np.where((binTemplate>-0.001) & (binTemplate<0.001))[0][0]
    stim = np.where((binTemplate>-0.011) & (binTemplate<-0.009))[0][0]
    x = myarray[stim:] # if looking for peak at the stim 

    #### find the peaks
    #### filter to the first peak above the keythreshold value
    peaks, _ = find_peaks(x, height=KEYTHRESHVAL)
    peaks = peaks + stim # place the index back where it belong for timing

    timePeak = binTemplate[peaks[0]]
    valPeak = myarray[peaks[0]]

    ####$$$$$$$$$$$$$$$$
    #### ONSET
    ####$$$$$$$$$$$$$$$$

    # plt.plot(np.diff(np.flip(myarray[:peaks[0]]))) ## reverse the array prior to the peak 

    ### for the method below to work we need to UPSAMPLE and INTERPOLATE the signal
    ## upsampling would not change
    # N = len(binTemplate[:-3])*10 #upsampling factor
    # upsampled_binTemplate = np.linspace(binTemplate[0], binTemplate[-1],N)
    # upsampled_myarray = griddata(binTemplate[:-3], myarray, upsampled_binTemplate, method='nearest')

    # plt.figure()
    # plt.plot(binTemplate[:-3],myarray,'.')
    # plt.plot(upsampled_binTemplate,upsampled_myarray,'.', color='red')

    ### find the index which have 
    ### modify my array to have upward trend
    myarray[:6] = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.])
    tmpDiff = np.where(np.diff(np.flip(myarray[:peaks[0]]))>0) # find the drive in the same direction
    tst = np.diff(tmpDiff) ## find where are consecutive number 
    tst = search_sequence_numpy(tst[0], np.array([1,1,1]))[0] # this is sequence matching for value 
    onset = tmpDiff[0][tst]

    # np.flip(myarray[:peaks[0]])
    # onset = np.where(np.diff(np.flip(myarray[:peaks[0]]))>0)[0][0] ## this return the onset from the peak
    onset = peaks[0]-onset+1 #true index of onset

    # testVal = ONSETTHRES #np.std(resp[0][trueBaseline])*2 

    # ### examples for the onset 
    # tmp = np.where(myarray>testVal)[0][0]
    timeOnset = binTemplate[onset] # return onset time
    valOnset = myarray[onset] # return the actual value 
    
    ####$$$$$$$$$$$$$$$$

    tmpDF = pd.DataFrame({'onsetTime':[timeOnset], 'onsetVal':[valOnset], 'peakTime':[timePeak], 'peakVal':[valPeak], 'respOrder':[iter]})

    if plot == True:
        f = plt.figure(str(iter))
        plt.plot(binTemplate[:-3], myarray)
        plt.plot(binTemplate[peaks[0]], valPeak, 'x', color='red')
        plt.plot(binTemplate[onset], valOnset, 'x', color='green')
        plt.show()
        return tmpDF, f

    return tmpDF

def temporalParameters(window =[-0.25, 0.50], binsize = 0.001):
    binTemplate = np.arange(window[0], window[1] + binsize, binsize)
    if window == [-0.025, 0.050]:
        print('precise temporal scaling')
        respWin = np.where((binTemplate>=0) & (binTemplate<0.03))
        baselineWin = np.where(binTemplate<-0.000)
    elif window == [-0.25, 0.50]:
        print('coarse temporal scaling')
        respWin = np.where((binTemplate>=0) & (binTemplate<0.2))
        baselineWin = np.where(binTemplate<-0.0)

    respWin = [binTemplate[respWin[0][0]], binTemplate[respWin[0][-1]]]
    return binTemplate, respWin, baselineWin

def statusCheck(mynpArray, window=[-0.025, 0.05], binsize=0.001, respWin=[0, 0.03], baselineWin=-0.000):
    '''
    function to be used on numpy array to filter the data of interest based on the response amplitude
    '''

    binTemplate = np.arange(window[0], window[1] + binsize, binsize)
    respWin = np.where((binTemplate>=respWin[0]) & (binTemplate<respWin[1]))
    baselineWin = np.where(binTemplate<baselineWin)
    # since we are already using a z-score denotating std above baseline straight up valuel can be used
    tmpVal = np.max(mynpArray[respWin])>KEYTHRESHVAL
    # tmpVal = np.max(mynpArray[respWin])>np.std(mynpArray[baselineWin])*10

    return tmpVal

alltmp_forAllCat = []

for i in files:
	print(i)
	brainRegion = i.split(os.sep)[-1].split('-')[1].split('_')[0]
	percentile = i.split(os.sep)[-1].split('_')[-1].split('.')[0]
	print(brainRegion, percentile)
	binTemplate, respWin, baselineWin = temporalParameters(window =[-0.025, 0.050])
	myColor = ['#6e9bbd', '#e26770']
	# myColor = ['blue', 'red']
	myArr = np.load(i, allow_pickle=True)

	mygeno =['wt','het']
	resp_wt = []
	resp_het = []

	# figparam(expType = 'KO', restoreDefault=True)
	alltmp = []
	### this first loop iterate over the different genotypes
	for kcol, karr, kgeno in zip(myColor, myArr, mygeno):
	    # print(kcol, kgeno)
	    # plt.figure()

	    ### iterate over the filtered arrays
	    for idx, i in enumerate(range(len(karr))):
	        # print(idx, i)
	        # gaussian smooth the array
	        tmpP = karr[i]

	        # get all the responder and the average
	        resp = tmpP[np.apply_along_axis(statusCheck, 1, tmpP)]

	        #######################################################################
	        ## get the table for the responder
	        ## this is just based on order can be tracked back to actual identity
	        
	        for i,j in enumerate(resp):
	            # print(i,j)
	            tmp = valueExtraction(myarray = j, iter = i, plot=False)
	            tmp['sIDorder'] = idx
	            tmp['geno'] = kgeno
	            alltmp.append(tmp)      


	        #######################################################################


	        if kgeno == 'wt':
	            resp_wt.append(resp)
	        else:
	            resp_het.append(resp)
	alltmp = pd.concat(alltmp)
	alltmp['percentile'] = percentile
	alltmp['brainRegion'] = brainRegion
	alltmp_forAllCat.append(alltmp)


	reformAll = []
	for idx, i in enumerate([resp_wt, resp_het]):
	    for j in range(len(i)):
	        for k in range(len(i[j])):
	            print(k)
	            reform = pd.DataFrame({'time':range(len(i[j][k])), 'value':i[j][k]})
	            if idx == 0:
	                reform['geno'] = 'wt'
	                reform['subject'] = 'wt_'+str(j)+'_'+str(k)
	            else:
	                reform['geno'] = 'het'
	                reform['subject'] = 'het_'+str(j)+'_'+str(k)
	            reformAll.append(reform)
	reformAll = pd.concat(reformAll)
	reformAll['area']=brainRegion
	reformAll['area']=percentile
	reformAll.to_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__'+os.sep+'#TV600_brain_stat_'+brainRegion+'_perc_'+str(percentile)+'.csv')



	savePath = 'Y:/2020-09-Paper-ReactiveTouch/_Figures/allPanels/#TV0370'
	figparam(expType = 'KO')
	mpl.rcParams['xtick.direction'] = 'out'
	mpl.rcParams['ytick.direction'] = 'out'
	mpl.rcParams['xtick.major.bottom'] = True
	mpl.rcParams['ytick.major.size'] =  2
	mpl.rcParams['xtick.major.size'] =  2

	mycol = ['#6e9bbd', '#f4a582']

	mpl.rcParams['font.size'] = 14
	fig, ax = plt.subplots(figsize=(4,4))
	ax.spines.right.set_visible(False)
	ax.spines.top.set_visible(False)
	for i,j in zip(['wt','het'], [resp_wt, resp_het]):
	    # print(i)
	    
	    custMean = np.mean(np.vstack(j), axis=0)
	    custSEM = np.std(np.vstack(j), axis=0)/np.sqrt(np.shape(np.vstack(j))[0])

	    if i == 'wt':
	        myColor = '#6e9bbd'#'blue' 
	    elif i == 'het':
	        myColor = '#e26770' #'red'#

	    ax.plot(binTemplate[:-3], custMean, color=myColor)
	    ax.fill_between(binTemplate[:-3], custMean+custSEM, custMean-custSEM, color=myColor, alpha=0.3)
	    # ax.set_title('Responsive popolutions (all bins)') # need to calculate the proportion
	    ax.set_ylabel('Spikes/s (z-score)')
	    # ax.set_yticks([])
	    ax.axvline(0, color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
	    ax.axhline(0, color='grey', linestyle='dashed', linewidth=0.5, alpha=1)
	    ax.set_xlabel('Time (s)')
	    ax.set_ylim([-0.5,4])
	    ax.set_xlim([-0.02,0.04])
	    plt.tight_layout()
	fig.savefig(savePath+os.sep+'#TV600_brain_'+brainRegion+'_perc_'+str(percentile)+'.pdf')
	plt.close()




##################
## to make a linear plot 
#####################################################################
alltmp_forAllCat = pd.concat(alltmp_forAllCat)
alltmp_forAllCat['percentile'] = pd.to_numeric(alltmp_forAllCat['percentile'])
## https://seaborn.pydata.org/examples/strip_regplot.html
alltmp_forAllCat.to_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV600_peakAmpbyCat.csv')

for i in ['M1','S1','TH']:
	# i = 'TH'
	a = alltmp_forAllCat[alltmp_forAllCat['brainRegion']==i]

	# sns.catplot(
	#     data=a, x="percentile", y="peakVal", hue="geno",
	#     native_scale=True, zorder=1
	# )


	# fig, ax = plt.subplots(figsize=[4,4])
	# ## plot all the point
	# # sns.stripplot(data=a, x="percentile", y="peakVal", hue="geno", edgecolor = 'k', linewidth = 0.4,  size=2, zorder=3, ax = ax, dodge = True, jitter = 0.2, alpha=0.5)
	# sns.pointplot(data=a, x="percentile", y="peakVal", hue="geno", errorbar=('ci', 68), scale=0.4, errwidth=1.2, dodge=True,ax = ax)
	fig, ax = plt.subplots(figsize=[4,4])
	sns.lineplot(data=a, x="percentile", y="peakVal", hue="geno",errorbar=('ci', 68),ax = ax, marker='o')
	ax.legend_.remove()
	plt.ylim([1.5,4.7])
	plt.tight_layout()
	ax = plt.gca()
	new_x_ticks  = [25,50,75,100]
	ax.set_xticks(new_x_ticks)
	plt.ylabel('Peak amp. (z-score)')
	plt.xlabel('Touch cat. (percentile)')

	# fig.savefig(savePath+os.sep+'#TV600_brain_summaryPercentile_'+i+'.pdf')
	a['subj'] = a['geno']+a['sIDorder'].astype(str)
	print(a.groupby('geno').count())
	print(quickConversion(a.groupby(['geno','subj']).agg({'subj':np.ma.count})).groupby('geno').count())


	aov = pg.mixed_anova(dv='peakVal', between='geno',
	                  within='percentile', subject='subj', data=a, correction = True)
	    
	for i,j in aov.iterrows():
	    if j['p-unc'] < 0.001:
	        sigMark = '***'
	    elif j['p-unc'] < 0.01:
	        sigMark = '**'
	    elif j['p-unc'] < 0.05:
	        sigMark = '*'
	    elif j['p-unc'] >= 0.05:
	        sigMark = 'na'
	    aov['sigMark'] = sigMark
	print(i)
	print(aov)
