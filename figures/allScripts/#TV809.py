# MS0045
# MS0050
# MS0052

file = r"Y:\2020-09-Paper-ReactiveTouch\MS_Cre_Data.xlsx"
dat = pd.read_excel(file)
dat = dat.loc[dat['Figure']=='MS0045', ['+/+','+/ls']]
dat = dat.melt()
getStat(dat, 'value', groupVar = 'variable', group=['+/+','+/ls'], param=True)


dat = pd.read_excel(file)
dat = dat.loc[dat['Figure']=='MS0052', ['+/+','+/ls']]
dat = dat.melt()
getStat(dat, 'value', groupVar = 'variable', group=['+/+','+/ls'], param=True)