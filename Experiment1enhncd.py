from pyspark.sql.types import *
from sklearn import cross_validation

import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
from pyspark.sql.functions import broadcast
import pandas as pd
from tqdm import *
from sklearn.model_selection import KFold
import time
from CheckCSV import *

cf = CF.CollaborativFiltering()
cbf = CBF.ContentBasedFiltering()

schema = StructType([
    StructField("steamid", IntegerType()),
    StructField("appid", IntegerType()),
    StructField("rating", DoubleType())
])

# set envparam PYSPARK_PYTHON = python3

FILE_SIZE = 10000
ITER = 2
NFOLDS = 10
MIN_GAMES = 5

data = cf.spark.read.csv('Resources/formateddataset{0}.csv.gz'.format(FILE_SIZE), header=True, schema=schema)
data = broadcast(data)
pandasset = data.toPandas()
pdataset = CheckCSV.remove_min_games(pandasset, minGames=MIN_GAMES)
dataset = cf.spark.createDataFrame(pdataset)
appnames = cf.spark.read.csv('Resources/allgames.csv.gz', header=True, inferSchema=True)
cbf.readsimilaritymatrix(FILE_SIZE)
cf.setOptParams()

"""ParamOpt"""
# (training, validation) = dataset.randomSplit([0.9, 0.1])
# (train, test) = training.randomSplit([0.8, 0.2])
# cf.paramOpt(validation, 2, 10)

result = pd.DataFrame()
folds = [(1.0 / NFOLDS)] * NFOLDS
kf = KFold(n_splits=10)
for i in tqdm(range(ITER)):

    splits = dataset.randomSplit(folds)

    for fold, test in enumerate(tqdm(splits)):
        datasetc = dataset
        nUsers = test.select(test.steamid).where(test.rating == 1).distinct().count()
        users = cf.takeSamples(test, nUsers)
        cbf_test = test.subtract(users)
        cbf_pred = cbf.predict(cbf_test.toPandas())
        train = datasetc.subtract(test)
        cf.fit(train)
        cf_df = cf.predict(test)
        pd_users = users.toPandas()
        del pd_users['rating']
        cf_pred = cf_df.toPandas()
        cf_pred[['steamid', 'appid']] = cf_pred[['steamid', 'appid']].astype(int)
        iterators = {'cf': cf_pred, 'cbf': cbf_pred}

        for type, data in iterators.items():

            subset = data.where((data.rating == 0.0) | (
                (data.steamid.isin(pd_users.steamid)) & (data.appid.isin(pd_users.appid)))).dropna()
            subset['rank'] = subset.groupby('steamid').cumcount() + 1
            targets = subset.merge(pd_users, how='inner', on=('steamid', 'appid'))
            targets[['steamid', 'appid']] = targets[['steamid', 'appid']].astype(int)
            targets.insert(0, 'iter', i + 1)
            targets.insert(1, 'fold', fold + 1)
            targets.insert(2, 'type', type)
            result = result.append(targets)

result = result.sort_values(by=['iter', 'fold', 'steamid', 'rating'], ascending=[True, True, True, False])
print(result)
result.to_csv('ExperimentData/E1-{0}-{1}-{2}-{3}-{4}.csv.gz'.format(FILE_SIZE, ITER, NFOLDS, MIN_GAMES, time.strftime("%Y%m%d%H%m")),
              compression='gzip')
