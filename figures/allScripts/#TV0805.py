
sID = ['2019-03-25', '2019-04-04', '2019-04-08', '2019-04-15', '2019-04-23', '2019-04-30', '2019-05-15', '2019-05-20', '2019-11-07', '2019-11-12', '2019-11-15', '2019-11-18', '2019-11-21', '2019-11-22', '2019-12-11', '2019-12-12', '2019-12-13', '2019-12-18', '2020-01-08', '2020-01-09']
#'2019-04-04'  no curvature passed 15min for 2019-04-04  need to traceback why
collapseTouches = []
for k in sID:
	print(k)
	myDat = comboBehavior(k) #sID = '2019-03-25'
	touchesSub = myDat.touchDetails
	# f, ax = plt.subplots()

	for i,j in tqdm.tqdm(touchesSub.iterrows()):
		# try:
			leadingValues = 10
			trailingValues = 0
			# print(i,j)
		    startT = j.FirstTouchFrame
		    finishT = startT + j.touchCountFrame
			fullVal = myDat.curvature.loc[myDat.curvature['frame'].isin(np.arange(startT-leadingValues,finishT+trailingValues)),'curvature'].values
			baseline = fullVal[:leadingValues]
			fullVal1 = fullVal-np.mean(baseline)
			touchVal = myDat.curvature.loc[myDat.curvature['frame'].isin(np.arange(startT,finishT)),'curvature'].values
			collapseX = (np.arange(startT-leadingValues, finishT+trailingValues)-startT)/500
			collapseXtouch = (np.arange(startT, finishT)-startT)/500
			touchStatus = np.concatenate((np.repeat(0, leadingValues), np.repeat(1, len(touchVal)), np.repeat(0, trailingValues)))

			# ax.plot(collapseX, fullVal, lw=1, color='grey')
			# ax.plot(collapseXtouch, touchVal, lw=2, color='orange')

			tmpFrame = pd.DataFrame({'time':collapseX,'curvature':fullVal,'touch':touchStatus})
			tmpFrame['touchIdx'] = j.touchIdx
			tmpFrame['sID'] = j.sID
			collapseTouches__LO.append(tmpFrame)
		except:
			print('ERROR ERROR: '+k)

collapseTouches = pd.concat(collapseTouches__LO)
# collapseTouches.to_csv(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__'+os.sep+'#TV0805_collapsedCurvature.csv')

geno = 'Y:/Vaissiere/fileDescriptor2.csv'
geno = getGenotype(geno)


mDat = pd.merge(collapseTouches, geno, on='sID')
mDat = mDat.sort_values(by='Animal_geno', ascending=False)

sns.relplot(data=collapseTouches, x="time", y="curvature", kind="line")
sns.lineplot(data=collapseTouches, x="time", y="curvature", units="touchIdx", estimator=None, lw=0.2)


figparam('KO')
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.major.bottom'] = True
mpl.rcParams['ytick.major.size'] =  2
mpl.rcParams['xtick.major.size'] =  2

mycol = ['#6e9bbd', '#f4a582']

mpl.rcParams['font.size'] = 14
fig, ax = plt.subplots(figsize=(6,4))

sns.lineplot(data=mDat, x="time", y="curvature", hue='Animal_geno')
plt.axvline(0,linestyle='--',color='grey')
plt.axhline(0,linestyle='--',color='grey')
plt.xlim([-0.01,1])
plt.ylim([-0.002,0.006])
plt.xlabel('Time (s)')
plt.ylabel('Curvature')
plt.tight_layout()
plt.savefig(r"Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels\#TV0805_1s.pdf")