dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\_archivedTools\figurePanels\paper_rabies_datapand.csv")


dat = dat[dat['variable']=='ipsi']
dat=dat.sort_values(by='geno', ascending=False)
myorder = ['SS','AUD','PTLp','VIS','MO','TH','CP','RSP','ACA']
plt.close(); customPlotingMulti(dat, myy='value.norm', myx='parent.struc', myu='geno', nval='', myorder=myorder, figSet = [4,2.5], myponitszie=3); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1003.pdf')


dat = dat[dat['parent.struc'].isin(['MO', 'TH'])]
dat=dat.sort_values(by='geno', ascending=False)
myorder = ['MO', 'TH']
plt.close(); customPlotingMulti(dat, myy='value.norm', myx='parent.struc', myu='geno', nval='', myorder=myorder, figSet = [3.5,2.1], myponitszie=6); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1004.pdf')