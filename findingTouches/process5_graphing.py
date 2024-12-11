### IMPORTANT SEE this file
# "Y:\Sheldon\Highspeed\not_analyzed\WDIL007_SyngapKO 12-16-19 cohort high Stim\autoDetectTouches\package\figureForCohort.py"

mainDir = r'Y:\Sheldon\Highspeed\not_analyzed\WDIL009\autoDetectTouches'

datAll = pd.read_csv(r"Y:\Sheldon\Highspeed\not_analyzed\WDIL009\autoDetectTouches\alltheData.csv")


# convert to seconds
datAll['touchCount_ms'] = datAll['touchCount']*2
# datAll['touchCount'] = datAll['touchCount']*10**-3
datAll = datAll[datAll['touchCount']<800]
datAll['FirstTouchFrame'] = datAll['FirstTouchFrame']/500

# normalized to 0 at oneset of first touch
firstTouch = quickConversion(datAll.groupby(['position', 'sID'])['FirstTouchFrame'].first())
firstTouch = firstTouch.rename(columns={'FirstTouchFrame':'touchStart'})
datAll = pd.merge(datAll, firstTouch, on=['sID','position'])
datAll['normFirstTouchFrame'] = datAll['FirstTouchFrame'] - datAll['touchStart']



# work only with the first 30 seconds after the first touch 

p_type = ['close_position', 'middle_position', 'far_position'] # type of position
datAll = datAll.sort_values(by='geno',  ascending=False)
datAll = datAll[datAll['normFirstTouchFrame']<30]




fig, ax = plt.subplots(1,3, sharey=True, figsize=(14,4))

for idx, i in enumerate(p_type):
	dat = datAll[datAll['position'] == i]
	sns.scatterplot(data = dat, x='normFirstTouchFrame', y='touchCount', hue='geno', ax=ax[idx], alpha= 0.2)
	ax[idx].set_ylim([0,600])
	ax[idx].title.set_text(i)
	ax[0].set_ylabel('Touch duration (ms)')
	ax[idx].set_xlabel('Touch onset (s)')




touchDur = quickConversion(datAll.groupby(['position', 'sID', 'geno'])['touchCount_ms'].mean())



for idx, i in enumerate(p_type):
	dat = datAll[datAll['position'] == i]
	customPloting(touchDur[touchDur['position'] == i], val='touchCountms', plottick = 4, custylim=[0,150], figSet = [1.5,3])
	plt.title(i)
	plt.ylabel('Touch duration (ms)')
	plt.tight_layout()
	plt.show()

