'''
get the stat for some of the figure 
intially from the paper folder python.py
'''

path = r'Y:\2020-09-Paper-ReactiveTouch\_archivedTools\exportAttempt'
files = glob.glob(path+'/*.csv')

for i,j in enumerate(files):
    print(i,j)

# for setpoints EMX1RUM1
dt = fileOfInterest(files[13])[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/ls'], param=True)
dt = fileOfInterest(files[13])[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/ls'], param=True)

# for prot am EMX1RUM1
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Pro Amp.csv')[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Pro Amp.csv')[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)

# for ret am EMX1RUM1
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Ret Amp.csv',)[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Ret Amp.csv',)[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)

# for prot vel EMX1RUM1
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Pro Vel.csv',)[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Pro Vel.csv',)[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())

# for ret vel EMX1RUM1
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Ret Vel.csv',)[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Ret Vel.csv',)[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())

# for AUC curv EMX1RUM1
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\AUC Curvature.csv',)[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\AUC Curvature.csv',)[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())


## for touch duration
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Mean Touch Durations.csv')[0]
dt = dt[['geno','Emx1xRUM1']].dropna()
getStat(dt, 'Emx1xRUM1', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())

## for touch duration
dt = fileOfInterest('Y:\\2020-09-Paper-ReactiveTouch\\_archivedTools\\exportAttempt\\Mean Touch Durations.csv')[0]
dt = dt[['geno','Emx1xRUM2']].dropna()
getStat(dt, 'Emx1xRUM2', groupVar = 'geno', group = ['+/+ ', '+/-'], param=True)
print(dt.groupby('geno').count())
