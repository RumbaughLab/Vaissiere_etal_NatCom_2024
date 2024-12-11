dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\Figure4_LFP-power.csv")
a = dat[dat['panel']=='i']
getStat(a, 'value')

a = dat[dat['panel']=='j']
getStat(a, 'value')
dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\Figure4_outwhiskall.csv")
aov = pg.mixed_anova(dv='lfp', between='geno',
                  within='whisker', subject='subject', data=dat, correction = True)