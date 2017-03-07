from pyspark.sql.types import *
from sklearn.model_selection import KFold
from pyspark.sql.functions import rand
import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
import time
import random as rand
import pandas as pd
from tqdm import *
cf = CF.CollaborativFiltering()
cbf = CBF.ContentBasedFiltering()

#recommenders = {'cf': cf, 'cbf': cbf}

schema = StructType([
    StructField("steamid", IntegerType()),
    StructField("appid", IntegerType()),
    StructField("rating", DoubleType())
])

# set envparam PYSPARK_PYTHON = python3

FILE_SIZE = 100

dataset = cf.spark.read.csv('Resources/formateddataset{0}.csv.gz'.format(FILE_SIZE), header=True, schema=schema)

appnames = cf.spark.read.csv('Resources/allgames.csv.gz', header=True, inferSchema=True)
cbf.readsimilaritymatrix(FILE_SIZE)
cf.setOptParams()
#apps = dataset.toPandas()
#cbf.generateGameGenreMatrix(apps, save=True, file_size=FILE_SIZE)
#cbf.generateSimMatrix(cbf.gm, save=True, file_size=FILE_SIZE)
print('Read Data - Done!')


"""ParamOpt"""
#(training, validation) = dataset.randomSplit([0.9, 0.1])
#(train, test) = training.randomSplit([0.8, 0.2])
#cf.paramOpt(validation, 2, 10)

#10 fold
#foreach fold do everithing below
result = pd.DataFrame()
splits = dataset.randomSplit([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
for fold, split in enumerate(tqdm(splits)):
    nUsers = split.select(split.steamid).where(split.rating == 1).distinct().count()
    users = cf.takeSamples(split, nUsers)
    users.show()
    cbftest = split.subtract(users)
    cbf_df = cbf.predict(cbftest, 0)
    cbf_df.show()
    train = dataset.subtract(split)
    cf.fit(train)
    cf_df = cf.predict(split)

    for user in users.collect():
        cf_sel = cf_df.where((cf_df.steamid == user.steamid) & (cf_df.appid == user.appid)).collect()[0]
        cf_count = cf_df.where((cf_df.steamid == cf_sel.steamid) & (cf_df.rating != 1) & (cf_df.prediction > cf_sel.prediction)).count()
        cbf_sel = cbf_df.where((cbf_df.steamid == user.steamid) & (cbf_df.appid == user.appid)).collect()[0]
        cbf_count = cbf_df.where(
            (cbf_df.steamid == cbf_sel.steamid) & (cbf_df.prediction > cbf_sel.prediction)).count()
        result = result.append(
            pd.DataFrame([[fold, 'CF', cf_sel.steamid, cf_sel.appid, cf_sel.rating, cf_sel.prediction, cf_count + 1]]))
        result = result.append(
            pd.DataFrame([[fold, 'CBF', cbf_sel.steamid, cbf_sel.appid, cbf_sel.rating, cbf_sel.prediction, cbf_count + 1]]))
    result.columns = ['fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank']
    print(result)
    break


    #test.select(test.steamid == 2).show()



#TDOD check user split

# save = iter, fold, steamid, appid model_type, score

#user = dataset[dataset.steamid == 11]

#cbf.predict(user, 20).join(appnames, on=['appid'], how='left').show()
#cf.predict(user).join(appnames, on=['appid'], how='left').show()

#Show predictions
#cbf_df.join(appnames, on=['appid'], how='left').show()
#cf_df.join(appnames, on=['appid'], how='left').show()

cf.spark.stop()

#ranking
#CF ta bort 1:or som inte är spelet, hämta index för spelet som vi letar efter.
#CBF hämta spele hämta index för spelet som vi letar efter.