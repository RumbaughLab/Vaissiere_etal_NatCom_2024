dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\Figure4_outwhiskall.csv")


# dat = dat[dat['variable']=='ipsi']
dat=dat.sort_values(by='geno', ascending=False)
myorder = ['a', 'A1', 'b', 'B1', 'B2', 'C1', 'C2', 'D1', 'g']
plt.close(); customPlotingMulti(dat, myy='lfp', myx='whisker', myu='geno', nval='', myorder=myorder, figSet = [4,2.5], myponitszie=3); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1100.pdf')
