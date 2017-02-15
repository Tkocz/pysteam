from __future__ import print_function

import sys

if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

spark = SparkSession \
    .builder \
    .appName("pysteam") \
    .getOrCreate()

spark.sparkContext.setLogLevel('OFF')
# params
rank = 10
nIter = 10
alpha = 40.0
lamb = 0.01

# $example on$
dataset = spark.read.csv('Resources/formateddataset100.csv', header=True, inferSchema=True)

(training, test) = dataset.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(rank= 10, maxIter=nIter, regParam=lamb, alpha=alpha, userCol="steamid", itemCol="appid", ratingCol="rating")
model = als.fit(dataset)

predictions = model.transform(test)
print(predictions.collect())
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")
mse = evaluator.evaluate(predictions)
print("mean-square error = " + str(mse))

evaluator = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")
mae = evaluator.evaluate(predictions)
print("mean absolute error = " + str(mae))

spark.stop()