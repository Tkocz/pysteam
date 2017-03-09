import datetime

from pyspark.sql.types import *
import CollaborativeFiltering as CF
import ContentBasedFiltering as CBF
from pyspark.sql.functions import broadcast
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
ITER = 2
NFOLDS = 10

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
        cbf_df = cbf.predict(cbftest, 0)
        bCbf_df = broadcast(cbf_df)
        train = dataset.subtract(bSplit)
        cf.fit(train)
        cf_df = cf.predict(bSplit)
        bCf_df = broadcast(cf_df)
        for user in busers.collect():
            prediction = -1
            cf_sel = bCf_df.where((bCf_df.steamid == user.steamid) & (bCf_df.appid == user.appid)).collect()[0]
            cf_count = bCf_df.where((bCf_df.steamid == cf_sel.steamid) & (bCf_df.rating != 1) & (bCf_df.prediction > cf_sel.prediction)).count()
            cbf_sel = bCbf_df.where((bCbf_df.steamid == user.steamid) & (bCbf_df.appid == user.appid)).collect()
            if len(cbf_sel) > 0:
                cbf_count = bCbf_df.where(
                    (bCbf_df.steamid == cbf_sel[0].steamid) & (bCbf_df.prediction > cbf_sel[0].prediction)).count()
                prediction = cbf_sel[0].prediction
            else:
                cbf_count = bCbf_df.where(bCbf_df.steamid == user.steamid).count()
            result = result.append(
                pd.DataFrame([[int(i + 1), int(fold + 1), int(0), int(user.steamid), int(user.appid), int(user.rating), float(cf_sel.prediction), int(cf_count + 1)]]))
            result = result.append(
                pd.DataFrame([[int(i + 1), int(fold + 1), int(1), int(user.steamid), int(user.appid), int(user.rating), float(prediction), int(cbf_count + 1)]]))

result.columns = ['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank']
print(result)
result.to_csv('ExperimentData/E1-{0}-{1}-{2}.csv.gz'.format(FILE_SIZE, ITER, NFOLDS ), compression='gzip')
#result.groupby(by=['type'], axis=0).mean()
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

#E1-100-2-10    Time
#iter- 1        100%|██████████| 10/10 [38:32<00:00, 240.82s/it]
#iter- 2        100%|██████████| 10/10 [42:30<00:00, 254.83s/it]
#total          100%|██████████| 2/2 [1:21:02<00:00, 2384.01s/it]
#E1-250-2-10
#iter- 1
#iter- 2
#total