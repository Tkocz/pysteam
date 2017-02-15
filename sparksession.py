from __future__ import print_function

import sys

if sys.version >= '3':
    long = int

from pyspark.sql import SparkSession

# $example on$
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
# $example off$


spark = SparkSession \
    .builder \
    .appName("ALSExample") \
    .getOrCreate()

# params
rank = 10
nIter = 10
alpha = 40.0
lamb = 0.01

# $example on$
ratings = spark.read.csv('Resources/formateddataset100.csv', header=True, inferSchema=True)
print(ratings.collect())
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(rank= 10, maxIter=nIter, regParam=lamb, alpha=alpha, userCol="steamid", itemCol="appid", ratingCol="rating")
model = als.fit(ratings)

predictions = model.transform(test)

print(predictions.collect())

spark.stop()
