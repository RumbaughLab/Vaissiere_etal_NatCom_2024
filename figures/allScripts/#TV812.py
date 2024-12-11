dat = pd.read_csv(r"Y:\TEMP\datamo1_476.csv")
brainRegionlist = ['SS','AUD','PTLp','VIS','MO','TH','CP','RSP','ACA']

for i in brainRegionlist:
	tmp = dat[(dat['variable']=='ipsi') & (dat['parent.struc']==i)]
	print(i)
	getStat(tmp, 'value.norm', groupVar='geno', group=['WT','HET'])