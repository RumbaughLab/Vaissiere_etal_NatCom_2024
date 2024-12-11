files = [
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0413_MS0073a_M1Whisk.csv",
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_MS0073b_S1Whisk.csv",
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0419_MS0073c_THWhisk.csv",
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0394_MS0074a_M1Touch_02-2023.csv",
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0385_MS0074b_S1Touch_02-2023.csv",
r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0400_MS0074c_THTouch_02-2023.csv"
]


for i in files:
	alltmp = pd.read_csv(i)
	plt.close('all')
	tmp = quickConversion(alltmp.groupby(['sIDorder','geno']).agg({'sIDorder':['count']}),option=2)
	print(i)
	getStat(tmp, 'sIDorder_count')


customPloting(tmp, 'sIDorder_count', 'geno', custylim=[0,100], baroff=True)
ax = plt.figure(1).get_axes()[0] # get the axis from the default plot useful
tmp = quickConversion(alltmp.groupby(['geno']).agg({'sIDorder':['count']}),option=2)
tmp = tmp.sort_values(['geno'], ascending=False)
tmp.reset_index(inplace=True, drop=True)
sns.stripplot(x=tmp['geno'],y=tmp['sIDorder_count'], hue=['#6e9bbd', '#e26770'], marker="D", s=3, alpha =1) # add the sum 
plt.legend([],[], frameon=False)


pos = range(2)
for tick,label in zip(pos,ax.get_xticklabels()):
    print(tick,label)
    ax.text(pos[tick], tmp['sIDordercount'][tick]*0.9, tmp['sIDordercount'][tick], horizontalalignment='center', size='x-small', color='black', weight='semibold') #, weight='semibold'
plt.ylabel('Samples (n) \n Diamond:sum')
plt.ylim([0,max(tmp['sIDordercount'])*1.1])
plt.tight_layout()
plt.savefig('Y:/2020-09-Paper-ReactiveTouch/_Figures/allPanels/#TV0370'+os.sep+'#TV0419_bis_N.pdf')
