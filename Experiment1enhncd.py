from pyspark.sql.types import *
import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
import pandas as pd
from tqdm import *
import numpy as np
from time import localtime, strftime
import CheckCSV

cf = CF.CollaborativFiltering()
cbf = CBF.ContentBasedFiltering()
csv = CheckCSV.CheckCSV()
schema = StructType([
    StructField("steamid", IntegerType()),
    StructField("appid", IntegerType()),
    StructField("rating", DoubleType())
])

# set envparam PYSPARK_PYTHON = python3
FILE_SIZE = 100
ITER = 1
NFOLDS = 10
MIN_GAMES = 2

dataset = cf.spark.read.csv('Resources/formateddataset{0}.csv.gz'.format(FILE_SIZE), header=True, schema=schema)
print("OnUsers: ", dataset.select('steamid').distinct().count())
nGames = dataset[dataset.rating == 1.0].groupBy('steamid').count().filter('count>=' + str(MIN_GAMES))
dataset = dataset.join(nGames, 'steamid').select('steamid', 'appid', 'rating')
dataset.cache()
cbf.readsimilaritymatrix(FILE_SIZE)
print("nUsers: ", dataset.select('steamid').distinct().count())
print("nApps: ", dataset.select('appid').distinct().count())
cf.setOptParams()
"""ParamOpt"""
# (training, validation) = dataset.randomSplit([0.9, 0.1])
# (train, test) = training.randomSplit([0.8, 0.2])
# cf.paramOpt(validation, 2, 10)

result = pd.DataFrame()
folds = [(1.0 / NFOLDS)] * NFOLDS

for i in tqdm(range(ITER)):

    splits = dataset.randomSplit(folds)

    #TODO: fix stratefied fold with even distribution

    for fold, test in enumerate(tqdm(splits)):

        nUsers = test.select(test.steamid).where(test.rating == 1).distinct().count()
        sampledtest = cf.takeSamples(test)
        train = dataset.subtract(test)
        ones = train.toPandas()
        ones = ones.where(ones.rating == 1)
        cbf_pred = cbf.predict(ones)
        print(cbf_pred.info())
        print(cbf_pred)
        cf.fit(train)
        cf_df = cf.predict(test)
        pd_users = sampledtest.toPandas()
        del pd_users['rating']
        cf_pred = cf_df.toPandas()
        cf_pred[['steamid', 'appid']] = cf_pred[['steamid', 'appid']].astype(int)
        iterators = {'cbf': cbf_pred, 'cf': cf_pred}

        for type, data in iterators.items():

            subset = data.where((data.rating == 0.0) | (
                (data.steamid.isin(pd_users.steamid)) & (data.appid.isin(pd_users.appid)))).dropna()
            subset['rank'] = subset.groupby('steamid').cumcount() + 1
            targets = subset.merge(pd_users, how='inner', on=('steamid', 'appid'))
            targets[['steamid', 'appid']] = targets[['steamid', 'appid']].astype(int)
            targets.insert(0, 'iter', i + 1)
            targets.insert(1, 'fold', fold + 1)
            targets.insert(2, 'type', type)
            #targets = targets.merge(nGames, how='inner', on='steamid')
            result = result.append(targets)
        break
result = result.sort_values(by=['iter', 'fold', 'steamid', 'rating'], ascending=[True, True, True, False])
print(result)
result.to_csv('ExperimentData/E1-{0}-{1}-{2}-{3}-{4}.csv.gz'.format(FILE_SIZE, ITER, NFOLDS, MIN_GAMES, strftime("%Y%m%d%H%M", localtime())),
              compression='gzip')
