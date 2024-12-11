In the whisking csv output columns with the following description

`anal_status (included/excluded):` initial inclusion or exclusion of whisking behavior for psth analysis based on the following criteria: whisking epoch last longer than 0.4 second and whisking interval greater than 0.2 second.

`polePresent (1/0):` indicate the phase of the behavior epoch during which the pole was present (1) or absent (0). When the pole is present (1) this lead to potentially more complex behavior that could include touch 

# Note
filtering the data from csv on 
	- anal_status == 'included' and
	- polePresent == 0
whiskBehaviorFull[(whiskBehaviorFull['anal_status']=='included') & (whiskBehaviorFull['polePresent']==0)]

would lead to the actual outcome of evTime for whisking analysis