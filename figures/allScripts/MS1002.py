dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\___REVISIONS\whisk_1NMSE_Score_same_area_data.csv")

dat=dat.sort_values(by='Genotype', ascending=False)
myorder = ["('S1_source', 'S1_target')", "('M1_source', 'M1_target')", "('TH_source', 'TH_target')"]
plt.close(); customPlotingMulti(dat, myy='Rank', myx='Match', myu='Genotype', nval=10, myorder=myorder,figSet = [2.9,2.1], myponitszie=6); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1001.pdf')


plt.close(); customPlotingMulti(dat, myy='1-NMSE_Score', myx='Match', myu='Genotype', nval=10, myorder=myorder,figSet = [2.9,2.1], myponitszie=6); plt.tight_layout()
plt.yticks([0,0.4,0.8])
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1002.pdf')



dat = pd.read_csv(r"Y:\2020-09-Paper-ReactiveTouch\___REVISIONS\whisk_1NMSE_Score_cross_area_data.csv")
dat=dat.sort_values(by='Genotype', ascending=False)
dat['Match'].unique()
myorder = [ "('S1_source', 'M1_target')", "('S1_source', 'TH_target')",  
			"('M1_source', 'S1_target')", "('M1_source', 'TH_target')",       
            "('TH_source', 'S1_target')", "('TH_source', 'M1_target')",]
plt.close(); customPlotingMulti(dat, myy='Rank', myx='Match', myu='Genotype', nval=10, myorder=myorder,figSet = [3.4,1.6], myponitszie=4); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1003.pdf')


plt.close(); customPlotingMulti(dat, myy='1-NMSE_Score', myx='Match', myu='Genotype', nval=10, myorder=myorder,figSet = [3.4,1.6], myponitszie=4); plt.tight_layout()
plt.savefig(r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels'+os.sep+'MS1004.pdf')