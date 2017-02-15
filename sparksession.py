from __future__ import print_function

import sys

if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import param
import numpy as np
import itertools

spark = SparkSession \
    .builder \
    .appName("pysteam") \
    .getOrCreate()

spark.sparkContext.setLogLevel('OFF')
# params

#run1 : The best model was trained with rank = 12, lambda = 0.05, alpha = 10and numIter = 12, and its RMSE on the test set is 0.257741. mean-square error = 0.009006494757883858 mean absolute error = 0.06807511706369994 lmbda 0.01, 0.02, 0.05
#run2 : The best model was trained with rank = 12, lambda = 0.15, alpha = 10and numIter = 12, and its RMSE on the test set is 0.259563. mean-square error = 0.008499430241066145 mean absolute error = 0.0668242950350116  lambdas = [0.05, 0.1, 0.15]

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

# $example on$
dataset = spark.read.csv('Resources/formateddataset100.csv', header=True, inferSchema=True)

(training, validation, test) = dataset.randomSplit([0.6, 0.2, 0.2])

# Build the recommendation model using ALS on the training data

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

pm = [i for i in itertools.product(ranks, lambdas, numIters, alpha)]
indexes = np.random.permutation(len(pm))
indexes = indexes[:1]


for rank, lmbda, numIter, alf in [pm[i] for i in indexes]:
    model = ALS(implicitPrefs=True, rank=rank, regParam=lmbda, maxIter=numIter, alpha=alf, userCol="steamid", itemCol="appid", ratingCol="rating").fit(dataset)
    predictions = model.transform(validation)
    validationRmse = evaluator.evaluate(predictions)
    print("RMSE (validation) = %f for the model trained with " % validationRmse + \
          "rank = %d, lambda = %.2f, and numIter = %d. alpha = %d" % (rank, lmbda, numIter, alf))
    if (validationRmse < bestValidationRmse):
        bestModel = model
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lmbda
        bestNumIter = numIter
        bestAlpha = alf

predictions = bestModel.transform(test)

print("The best model was trained with rank = %d, lambda = %.2f, alpha = %d" % (bestRank, bestLambda, bestAlpha) \
        + "and numIter = %d." % (bestNumIter))

ones = predictions.where("rating=1")
zeroes = predictions.where("rating=0")

print(zeroes.collect())
print(ones.collect())

rmseall = evaluator.evaluate(predictions)
rmseones = evaluator.evaluate(ones)
rmsezeroes = evaluator.evaluate(zeroes)

print("total root mean-square error = %f, root ones mean-square error = %f, root ones mean-square error = %f. " %(rmseall, rmseones, rmsezeroes))

evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")

mseall = evaluator.evaluate(predictions)
mseeones = evaluator.evaluate(ones)
msezeroes = evaluator.evaluate(zeroes)
print("total mean-square error = %f, ones mean-square error = %f, ones mean-square error = %f. " %(mseall, mseeones, msezeroes))

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

maeall = evaluator.evaluate(predictions)
maeeones = evaluator.evaluate(ones)
maezeroes = evaluator.evaluate(zeroes)
print("total mean absolute error = %f, ones mean absolute error = %f, zeroes mean absolute error = %f." % (maeall, maeeones, maezeroes))

spark.stop()