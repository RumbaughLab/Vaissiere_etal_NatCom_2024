def getGlobStat(filePath, fromFile=True):
	if fromFile == True:
		reformAll = pd.read_csv(filePath)
	else:
		reformAll = filePath
	aov = pg.mixed_anova(dv='value', between='geno',
	                  within='time', subject='subject', data=reformAll, correction = True)

	## get the n
	count = pd.DataFrame({'sID':reformAll.subject.unique()})
	count[['c1', 'c2']] = count.sID.str.rsplit('_', n=1, expand=True)
	count[['c1b', 'c2b', 'c3']] = count.sID.str.split('_', expand=True)
	print(count.groupby('c1b').count())
	atmp = pd.DataFrame({'c1':count['c1'].unique()})
	atmp[['c1', 'c2']] = atmp.c1.str.split('_', expand=True)
	print(atmp.groupby('c1').count())

	# for i,j in aov.iterrows():
	#     if j['p-unc'] < 0.001:
	#         sigMark = '***'
	#     elif j['p-unc'] < 0.01:
	#         sigMark = '**'
	#     elif j['p-unc'] < 0.05:
	#         sigMark = '*'
	#     elif j['p-unc'] >= 0.05:
	#         sigMark = 'na'
	#     aov['sigMark'] = sigMark

	print(aov)

getGlobStat(r'Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0413_plot_M1whisk.csv')
getGlobStat(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_plot_S1whisk.csv")
getGlobStat(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_plot_THwhisk.csv")

getGlobStat(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0394_plot_M1Touch.csv")
getGlobStat(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0385_plot_S1Touch.csv")
getGlobStat(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0400_plot_THTouch.csv")



files = glob.glob(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__"+os.sep+'*stat*perc*.csv')
for i in files:
	print(i)
	getGlobStat(i, True)




