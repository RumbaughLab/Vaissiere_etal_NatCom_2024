from "Y:\Vaissiere\ARCHIVE\00-ephys temp\peak.csv"
also see "Y:\Vaissiere\ARCHIVE\00-ephys temp\F0021a.pdf"
and see "Y:\Vaissiere\ARCHIVE\00-ephys temp\F0021.R"

a = pd.read_csv(r"Y:\Vaissiere\ARCHIVE\00-ephys temp\peak.csv")
getStat(a[a['wf']=='wf1'], 'minPeak', groupVar='sID', group=[539,546])
getStat(a[a['wf']=='wf2'], 'minPeak', groupVar='sID', group=[539,546])