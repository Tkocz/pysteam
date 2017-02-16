from __future__ import print_function

import sys

if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from sklearn.metrics import auc, roc_curve
import pandas as pd
import numpy as np
import itertools

spark = SparkSession \
    .builder \
    .appName("pysteam") \
    .getOrCreate()

spark.sparkContext.setLogLevel('OFF')

# run1 : The best model was trained with rank = 12, lambda = 0.05, alpha = 10and numIter = 12, and its RMSE on the test set is 0.257741. mean-square error = 0.009006494757883858 mean absolute error = 0.06807511706369994 lmbda 0.01, 0.02, 0.05
# run2 : The best model was trained with rank = 12, lambda = 0.15, alpha = 10and numIter = 12, and its RMSE on the test set is 0.259563. mean-square error = 0.008499430241066145 mean absolute error = 0.0668242950350116  lambdas = [0.05, 0.1, 0.15]

# params
ranks = np.arange(8, 20, 2)
lambdas = np.linspace(0.01, 0.5, 10)
numIters = np.arange(8, 20, 2)
alpha = np.arange(8, 40, 2)
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1
bestAlpha = 0

# process data
dataset = spark.read.csv('Resources/formateddataset1000.csv', header=True, inferSchema=True)

(training, validation) = dataset.randomSplit([0.9, 0.1])

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
bevaluator = BinaryClassificationEvaluator(labelCol="rating")

pm = [i for i in itertools.product(ranks, lambdas, numIters, alpha)]
indexes = np.random.permutation(len(pm))
indexes = [pm[i] for i in indexes[:2]]
count = 0
for rank, lmbda, numIter, alf in indexes:

    for i in range(10):
        (opttrain, optval) = validation.randomSplit([0.8, 0.2])
        model = ALS(implicitPrefs=True, rank=rank, regParam=lmbda, maxIter=numIter, alpha=alf, userCol="steamid",
                    itemCol="appid", ratingCol="rating").fit(opttrain)
        predictions = model.transform(optval)
        validationRmse = evaluator.evaluate(predictions)
        print("\n")
        print("RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.2f, and numIter = %d. alpha = %d" % (rank, lmbda, numIter, alf))

        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter
            bestAlpha = alf
    count += 1
    print(round((count / len(indexes)) * 100, 0), '%')
print("The best model was trained on evalData with rank = %d, lambda = %.2f, alpha = %d, " % (bestRank, bestLambda, bestAlpha) \
      + "numIter = %d and RMSE %f." % (bestNumIter, bestValidationRmse))

bestTrainModel = None
valtrainrmse = float("inf")

(testtrain, testval) = training.randomSplit([0.6, 0.4])
for i in range(10):
    (opttrain, optval) = testtrain.randomSplit([0.8, 0.2])
    model = ALS(implicitPrefs=True, rank=bestRank, regParam=bestLambda, maxIter=bestNumIter, alpha=bestAlpha, userCol="steamid",
                itemCol="appid", ratingCol="rating").fit(opttrain)
    predictions = model.transform(optval)
    valrmse = evaluator.evaluate(predictions)

    if (valrmse < valtrainrmse):
        bestTrainModel = model

    count += 1
    print(round((i / 10) * 100, 0), '%')

# brier score
# AUC

predictions = bestTrainModel.transform(testval)

setvalues = ['all', 'zeroes', 'ones']

em = pd.DataFrame(columns=['rmse', 'mse', 'mae'])
em.index.names = ["set values"]

ones = predictions.where("rating=1")
zeroes = predictions.where("rating=0")
predictors = {'all': predictions, 'zeroes': zeroes, 'ones': ones}


for s, p in predictors.items():
    em.set_value(s, "rmse", evaluator.setParams(metricName="rmse").evaluate(p))
    em.set_value(s, "mse", evaluator.setParams(metricName="mse").evaluate(p))
    em.set_value(s, "mae", evaluator.setParams(metricName="mae").evaluate(p))

print(em)

spark.stop()
