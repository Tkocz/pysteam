import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
import pandas as pd
cf = CF.CollaborativFiltering()
cbf = CBF.ContentBasedFiltering()

#recommenders = {'cf': cf, 'cbf': cbf}

file_size = 1000
dataset = cf.spark.read.csv('Resources/formateddataset{0}.csv'.format(file_size), header=True, inferSchema=True)
appnames = cf.spark.read.csv('Resources/allgames.csv', header=True, inferSchema=True)
(training, validation) = dataset.randomSplit([0.9, 0.1])
(train, test) = training.randomSplit([0.8, 0.2])

users = cf.takeSamples(test, 1)
cbftest = test.subtract(users)

#10 fold
#foreach fold do everithing below

#apps = dataset.toPandas()
#cbf.generateGameGenreMatrix(apps, save=True, file_size=file_size)
#cbf.generateSimMatrix(cbf.gm, save=True, file_size=file_size)
cbf.readsimilaritymatrix(file_size)
cbf_df = cbf.predict(cbftest, 20)

#cf.paramOpt(validation, 2, 10)
cf.fit(train)
cf_df = cf.predict(test)

#Show predictions
#cbf_df.join(appnames.intersect(cbf_df), on=['appid'], how='left').show()
#new = cf_df.subtract(users).show()
#cbf_df.join(cf_df, on=['steamid', 'appid'], how='outer').show()
cbf_df.join(appnames, on=['appid'], how='left').show()
cf_df.join(appnames, on=['appid'], how='left').show()



cf.spark.stop()

#ranking
#CF ta bort 1:or som inte är spelet, hämta index för spelet som vi letar efter.
#CBF hämta spele hämta index för spelet som vi letar efter.