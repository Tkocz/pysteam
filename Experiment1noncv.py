from pyspark.sql.functions import explode, udf
from pyspark.sql.types import *
import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
import pandas as pd
import sklearn
from tqdm import *
import numpy as np
from time import localtime, strftime
import CheckCSV
from Ranking import *
from multiprocessing import Pool
from pyspark.sql import functions as F

#http://127.0.0.1:4040/jobs/

cf = CF.CollaborativFiltering()
cbf = CBF.ContentBasedFiltering()
csv = CheckCSV.CheckCSV()
rank = Rank()
schema = StructType([
    StructField("steamid", IntegerType()),
    StructField("appid", IntegerType()),
    StructField("rating", DoubleType())
])

# set envparam PYSPARK_PYTHON = python3
FILE_SIZE = 10000
ITER = 30
MIN_GAMES = 2
NUM_PARTITIONS = 10
NUM_CORES = 8


dataset = cf.spark.read.csv('Resources/formateddataset{0}.csv.gz'.format('PTime10000'), header=True, schema=schema)
print("OnUsers: ", dataset.select('steamid').distinct().count())
nGames = dataset[dataset.rating == 1.0].groupBy('steamid').count().filter('count>=' + str(MIN_GAMES))
dataset = dataset.join(nGames, 'steamid').select('steamid', 'appid', 'rating')
dataset.cache()
cbf.readsimilaritymatrix(FILE_SIZE)
print("nUsers: ", dataset.select('steamid').distinct().count())
print("nApps: ", dataset.select('appid').distinct().count())


cf.setOptParams()
"""ParamOpt"""
#(dataset, validation) = dataset.randomSplit([0.9, 0.1])
#cf.paramOpt(validation, 2, 10)

result = pd.DataFrame()
for i in tqdm(range(ITER), leave=True):
    nUsers = dataset.select(dataset.steamid).where(dataset.rating == 1).distinct().count()
    test = cf.takeSamples(dataset)
    #print(test.show())
    #print(dataset.where("steamid=1" and "rating=1.0").show())
    train = dataset.subtract(test)
    ones = train.where(dataset.rating == 1)
    pdones = ones.toPandas()
    gb = pdones.groupby(by=['steamid'], as_index=False)
    dataframe = pd.DataFrame([i for i in gb])
    del dataframe[0]
    cbftest = dataframe.values.flatten()
    split = np.array_split(cbftest, NUM_PARTITIONS)
    cbf_pred = pd.DataFrame(columns=['steamid', 'appid', 'rating', 'prediction'])
    for r in tqdm(split):
        pool = Pool(NUM_CORES)
        cbf_pred = cbf_pred.append(pool.map(cbf.predict, r))
        pool.close()
        pool.join()
    cbf_pred[['steamid', 'appid']] = cbf_pred[['steamid', 'appid']].astype(int)
    cf_shit = dataset.subtract(ones)
    cf.fit(train)
    #userrecs = cf.model.recommendForAllUsers(dataset.select('appid').distinct().count())
    #cf_df = userrecs.withColumn("recommendations", explode('recommendations')).selectExpr("steamid", "recommendations.*")
    #cf_df.sort(['steamid', 'prediction'], ascending=[True, False])
    cf_df = cf.predict(cf_shit)
    pd_users = test.toPandas()
    del pd_users['rating']
    cf_pred = cf_df.toPandas()
    cf_pred[['steamid', 'appid']] = cf_pred[['steamid', 'appid']].astype(int)
    predictions = {'cbf': cbf_pred, 'cf': cf_pred}
    targets = rank.rank(predictions, pd_users, i)
    result = result.append(targets)

result = result.sort_values(by=['iter', 'steamid', 'rating'], ascending=[True, True, False])
print(result)
result.to_csv('ExperimentData/E1-{0}-{1}-{2}-{3}.csv.gz'.format(FILE_SIZE, ITER, MIN_GAMES, strftime("%Y%m%d%H%M", localtime())),
              compression='gzip')
