import datetime

from pyspark.sql import Window
from pyspark.sql.types import *
import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
from pyspark.sql.functions import broadcast
import pandas as pd
from tqdm import *
import numpy as np
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
ITER = 2
NFOLDS = 10

dataset = cf.spark.read.csv('Resources/formateddataset{0}.csv.gz'.format(FILE_SIZE), header=True, schema=schema)

appnames = cf.spark.read.csv('Resources/allgames.csv.gz', header=True, inferSchema=True)
cbf.readsimilaritymatrix(FILE_SIZE)
cf.setOptParams()
#apps = dataset.toPandas()
#cbf.generateGameGenreMatrix(apps, save=True, file_size=FILE_SIZE)
#cbf.generateSimMatrix(cbf.gm, save=True, file_size=FILE_SIZE)

"""ParamOpt"""
#(training, validation) = dataset.randomSplit([0.9, 0.1])
#(train, test) = training.randomSplit([0.8, 0.2])
#cf.paramOpt(validation, 2, 10)

#10 fold
#foreach fold do everithing below
result = pd.DataFrame()
folds = [(1.0/NFOLDS)] * NFOLDS
for i in tqdm(range(ITER)):
    splits = dataset.randomSplit(folds)
    for fold, split in enumerate(tqdm(splits)):
        bSplit = broadcast(split)
        nUsers = bSplit.select(bSplit.steamid).where(bSplit.rating == 1).distinct().count()
        users = cf.takeSamples(bSplit, nUsers)
        busers = broadcast(users)
        #users.show()
        cbftest = bSplit.subtract(users)
        bcbftest = broadcast(cbftest)
        preds = cbf.predict(bcbftest.toPandas(), 0)
        cbf_df = cf.spark.createDataFrame(preds)
        bCbf_df = broadcast(cbf_df)
        train = dataset.subtract(bSplit)
        cf.fit(train)
        cf_df = cf.predict(bSplit)
        bCf_df = broadcast(cf_df)

        for user in busers.collect():
            prediction = -1.0
            cf_sel = bCf_df.where((bCf_df.steamid == user.steamid) & (bCf_df.appid == user.appid))#.collect()[0]
            cf_count = bCf_df.where((bCf_df.steamid == cf_sel.first().steamid) & (bCf_df.rating != 1) & (bCf_df.prediction > cf_sel.first().prediction)).count()
            cbf_sel = bCbf_df.where((bCbf_df.steamid == user.steamid) & (bCbf_df.appid == user.appid))#.collect()
            if cbf_sel.first() is not None:
                cbf_count = bCbf_df.where(
                    (bCbf_df.steamid == cbf_sel.first().steamid) & (bCbf_df.prediction > cbf_sel.first().prediction)).count()
                prediction = cbf_sel.first().prediction
            else:
                cbf_count = bCbf_df.where(bCbf_df.steamid == user.steamid).count()
            result = result.append(
                pd.DataFrame([[int(i + 1), int(fold + 1), 'CF', int(user.steamid), int(user.appid), int(user.rating), float(cf_sel.first().prediction), int(cf_count + 1)]]))
            result = result.append(
                pd.DataFrame([[int(i + 1), int(fold + 1), 'CBF', int(user.steamid), int(user.appid), int(user.rating), float(prediction), int(cbf_count + 1)]]))
            result.columns = ['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank']
            print(result)
        break
result.columns = ['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank']
print(result)
result.to_csv('ExperimentData/E1-{0}-{1}-{2}r2.csv.gz'.format(FILE_SIZE, ITER, NFOLDS ), compression='gzip')

#user = dataset[dataset.steamid == 11]
#cbf.predict(user, 20).join(appnames, on=['appid'], how='left').show()
#cf.predict(user).join(appnames, on=['appid'], how='left').show()
#Show predictions
#cbf_df.join(appnames, on=['appid'], how='left').show()
#cf_df.join(appnames, on=['appid'], how='left').show()

cf.spark.stop()

#E1-100-2-10    Time
#iter- 1        100%|██████████| 10/10 [38:32<00:00, 240.82s/it]
#iter- 2        100%|██████████| 10/10 [42:30<00:00, 254.83s/it]
#total          100%|██████████| 2/2 [1:21:02<00:00, 2384.01s/it]
#E1-250-2-10
#iter- 1
#iter- 2
#total