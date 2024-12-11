'''
IMPORT FUNCTION FROM 
Y:\2020-09-Paper-ReactiveTouch\__code__\whisking
r"Y:\2020-09-Paper-ReactiveTouch\__code__\whisking\pythonWhiskerCLEAN_lastTimingConversion.py"
'''
import warnings
warnings.simplefilter("ignore")

class TouchRetrieval():
    '''
    Class to retieve the actual touch times and touch data for the touch which have been 
    used during the analysis of the touch themselves 
    '''
    def __init__(self, 
                    sID, 
                    markerHS, 
                    markerHiSpV, 
                    summaryPath =  r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe",
                    evTimePath = 'Y:/Vaissiere/__UbuntuLamda/DLC/cut_pole1and2_threshMeth/touches/evTime'):
        
        self.markerHS = fillInPole(aid=sID, markerHS=markerHS)
        self.markerHiSpV = markerHiSpV[markerHiSpV['animalID'].str.contains(sID)]
        self.summaryPath = np.asarray(glob.glob(summaryPath+os.sep+'*'+os.sep+'*'+sID+'*.csv'))
        self.evTimePath = glob.glob(evTimePath+os.sep+sID+'*')[0]

    def timeConversion(self, df, marker):
        fs=25000 # rate of acquisition 
        fsVid=500 # 500 correspond to the acquisition rate of the high speed video note true acquisition rate may be different 500.25 vs 500 
        vidE3Delay = self.markerHS[marker].values/fs - self.markerHiSpV['fullVidTrans_'+marker].values/fsVid
        df['FirstTouchTime'] = df['FirstTouchFrame'] / fsVid + vidE3Delay
        df = df[['FirstTouchFrame','FirstTouchTime','touchCount','interEventinter']]# simplify the file
        df.columns = ['FirstTouchFrame','FirstTouchTime','touchCountFrame','interEventinterFrame']
        df['touchCountTime'] = df['touchCountFrame']/fsVid
        df['interEventinterTime'] = df['interEventinterFrame']/fsVid

        return df

    def retrieval(self):
        ## Definition of the constants 

        polePresType = [1,2] # define the pole presentation type
        markerType = ['HSLEDon2', 'HSLEDoff2'] # these directly correspond to the pole transition and the timing is define on the closest portion of time 

        evTimeRetrieved = []
        for ipole, imark in zip(polePresType, markerType):
            print(ipole, imark)


            tmpl = np.asarray(['seg'+str(ipole)+'toProcess' in x for x in self.summaryPath])
            csvSummaryFile = self.summaryPath[tmpl]
            csvSummaryFile = pd.read_csv(csvSummaryFile[0])
            ## filter out only the time where the pole is present
            csvSummaryFile = csvSummaryFile[csvSummaryFile['newMtouch']==1] # only subset the one with touch 
            csvSummaryFile = self.timeConversion(df=csvSummaryFile, marker = imark) # get the time conversion from the csv file 

            ## get the full file from the actual touch that were used for analysis
            npEV = np.load(self.evTimePath)
            npEV = pd.DataFrame({'FirstTouchTime':npEV})

            ## filter down the touches from the csvSummary file to match the event time by merging the 2 pandas data frame
            evTR = pd.merge(csvSummaryFile, npEV)
            evTimeRetrieved.append(evTR)

        evTimeRetrieved = pd.concat(evTimeRetrieved)
        lenDat = [len(evTimeRetrieved), len(npEV)]
        print(len(evTimeRetrieved), len(npEV))

        return evTimeRetrieved, lenDat



## ***************************************************************************
## * Assessment of touch                                       *
## ***************************************************************************

### check code here Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline ==> final files 
### check code here for conversion "Y:\2020-09-Paper-ReactiveTouch\__code__\whisking\pythonWhiskerCLEAN_lastTimingConversion.py"
### check here Y:\2020-09-Paper-ReactiveTouch\_behavior\whiskingPumpDetails

a = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg1toProcess\1_space_2019-04-08.csv"
b = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe\seg2toProcess\2_pole_2019-04-08.csv"

alld = []
for i in [a,b]:
    tmp = pd.read_csv(i)
    alld.append(tmp)

alld = pd.concat(alld)
tmp = alld[alld['newMtouch']==1]
tmp[tmp['touchCount']>2]



###########################################################################
### Try out for a with the first section 
###########################################################################


markerHS, markerHiSpV = getInfoFiles()
mainpath = r"Y:\Jessie\e3 - Data Analysis\e3 Data\allVideos\inDLCpipeline\constitutive_Syngap\2020-07-01 - poleTouch ReRun\timeToUSe"
files = glob.glob(mainpath+os.sep+'*'+os.sep+'*.csv')

aidList = np.unique(list(map(lambda x: x.split(os.sep)[-1].split('_')[-1].split('.')[0], files)))

alld = []
for k in aidList:
    # print(k)
    try:
        tmp = TouchRetrieval(sID=k, markerHS=markerHS, markerHiSpV=markerHiSpV)
        dat, lenDat = tmp.retrieval()
        dat['sID'] = k
        alld.append(dat)
    except:
        print('nogo for this sampele: ', k)
alld = pd.concat(alld)


#########
###get the relevant genotype information
genoRef = []
for i in aidList:
    tmp = myAnimalsRef(i)
    sex, geno = tmp.animalInfo
    tmp = pd.DataFrame({'sID':[i], 'geno':[geno], 'sex':[sex]})
    genoRef.append(tmp)
genoRef = pd.concat(genoRef)

alld = pd.merge(alld,genoRef, on='sID')







#####################################
### clustering with kmeans 1D
#####################################
data = alld['touchCountTime'].values
kmeans = KMeans(n_clusters=5).fit(data.reshape(-1,1))
a = kmeans.predict(data.reshape(-1,1))
kmeans.cluster_centers_
alld['cat_kmeans'] = a

#####################################
### clustering percentile
#####################################
## get the value correspondng to touch with a given percentile
percentile_list = [75,50,25]

alld['cat_percentile'] = 75
for i in percentile_list:
    tmpreturn = np.percentile(alld['touchCountTime'], i)
    print(tmpreturn)
    alld.loc[alld['touchCountTime']<tmpreturn,'cat_percentile'] = i
# alld.to_csv(r"Y:\2020-09-Paper-ReactiveTouch\__interimDat__\#TV0407_501_touchDetails.csv")
#######################################
### get some stat on the clustering
#######################################

total = alld.groupby(['sID','geno']).agg({'cat_percentile':'count'})
byCat = alld.groupby(['sID','geno','cat_percentile']).agg({'cat_percentile':'count'})
byCat = quickConversion(byCat, myCol='ts')
byCat = pd.merge(byCat, total, on = 'sID')
byCat['ratio'] = byCat['value']/byCat['cat_percentile']

for i in np.unique(byCat['catpercentile']):
    try:
        subdf = byCat[byCat['catpercentile']==i]
        print(i)
        print(subdf)

        params, paramsNest, nobs, nmax = paramsForCustomPlot(subdf , variableLabel='geno', subjectLabel='sID', valueLabel='ratio')

        customPlot(params,paramsNest, dirName=r'Y:\2020-09-Paper-ReactiveTouch\_Figures\allPanels', 
                                      figName = 'MS0100_'+'ratio'+str(i), 
                                      myylim = [0,1],  
                                      myfsize = [3,6],
                                      myy='ratio',
                                      tradMean=False,
                                      custSpike = True)
        plt.suptitle(str(i))
    except:
        print('skip cat: ', str(i))

byCat = byCat.sort_values(by='geno', ascending=False)
plt.figure()
sns.relplot(data=byCat, x='catpercentile', y='ratio', hue='geno', kind='line')
plt.figure()
sns.lineplot(data=byCat, x='catpercentile', y='ratio', hue='geno', estimator=None, units='sID')


aov = pg.mixed_anova(dv='ratio', between='geno',
                  within='catkmeans', subject='sID', data=byCat, correction = False)
for i,j in aov.iterrows():
    if j['p-unc'] < 0.001:
        sigMark = '***'
    elif j['p-unc'] < 0.01:
        sigMark = '**'
    elif j['p-unc'] < 0.05:
        sigMark = '*'
    elif j['p-unc'] >= 0.05:
        sigMark = 'na'
    aov['sigMark'] = sigMark

print(aov)   
###########################################################################
### run kmeans to define the number of cluster and what are those clusters
###########################################################################
# https://towardsdatascience.com/k-means-clustering-how-it-works-finding-the-optimum-number-of-clusters-in-the-data-13d18739255c
# https://medium.com/grabngoinfo/5-ways-for-deciding-number-of-clusters-in-a-clustering-model-5db993ea5e09
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_samples, silhouette_score

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np

# range_n_clusters = [3, 4, 5]
# for n_clusters in range_n_clusters:
#     # Create a subplot with 1 row and 2 columns
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.set_size_inches(18, 7)

#     # The 1st subplot is the silhouette plot
#     # The silhouette coefficient can range from -1, 1 but in this example all
#     # lie within [-0.1, 1]
#     ax1.set_xlim([-0.1, 1])
#     # The (n_clusters+1)*10 is for inserting blank space between silhouette
#     # plots of individual clusters, to demarcate them clearly.
#     ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

#     # Initialize the clusterer with n_clusters value and a random generator
#     # seed of 10 for reproducibility.
#     clusterer = KMeans(n_clusters=n_clusters, random_state=10)
#     cluster_labels = clusterer.fit_predict(X)

#     # The silhouette_score gives the average value for all the samples.
#     # This gives a perspective into the density and separation of the formed
#     # clusters
#     silhouette_avg = silhouette_score(X, cluster_labels)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score is :", silhouette_avg)

#     # Compute the silhouette scores for each sample
#     sample_silhouette_values = silhouette_samples(X, cluster_labels)

#     y_lower = 10
#     for i in range(n_clusters):
#         # Aggregate the silhouette scores for samples belonging to
#         # cluster i, and sort them
#         ith_cluster_silhouette_values = \
#             sample_silhouette_values[cluster_labels == i]

#         ith_cluster_silhouette_values.sort()

#         size_cluster_i = ith_cluster_silhouette_values.shape[0]
#         y_upper = y_lower + size_cluster_i

#         color = cm.nipy_spectral(float(i) / n_clusters)
#         ax1.fill_betweenx(np.arange(y_lower, y_upper),
#                           0, ith_cluster_silhouette_values,
#                           facecolor=color, edgecolor=color, alpha=0.7)

#         # Label the silhouette plots with their cluster numbers at the middle
#         ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

#         # Compute the new y_lower for next plot
#         y_lower = y_upper + 10  # 10 for the 0 samples

#     ax1.set_title("The silhouette plot for the various clusters.")
#     ax1.set_xlabel("The silhouette coefficient values")
#     ax1.set_ylabel("Cluster label")

#     # The vertical line for average silhouette score of all the values
#     ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

#     ax1.set_yticks([])  # Clear the yaxis labels / ticks
#     ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

#     # 2nd Plot showing the actual clusters formed
#     colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
#     ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
#                 c=colors, edgecolor='k')

#     # Labeling the clusters
#     centers = clusterer.cluster_centers_
#     # Draw white circles at cluster centers
#     ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
#                 c="white", alpha=1, s=200, edgecolor='k')

#     for i, c in enumerate(centers):
#         ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
#                     s=50, edgecolor='k')

#     ax2.set_title("The visualization of the clustered data.")
#     ax2.set_xlabel("Feature space for the 1st feature")
#     ax2.set_ylabel("Feature space for the 2nd feature")

#     plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
#                   "with n_clusters = %d" % n_clusters),
#                  fontsize=14, fontweight='bold')

# plt.show()