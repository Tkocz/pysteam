from __future__ import print_function
import pandas as pd
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS


def flipBit(df, nUser):
    ones = df[df.rating == 1.0].toPandas().values
    zeroes = df[df.rating == 0.0]
    id = np.array(np.unique(ones[:, 0]), dtype=int)
    index = np.random.choice(id, nUser, replace=False)
    print(index)
    ones[index, 2] = 0.0
    newpdf = pd.DataFrame(ones, columns=["user", "item", "rating"])
    newpdf[["user", "item"]] = newpdf[["user", "item"]].astype(int)
    newdf = spark.createDataFrame(newpdf)
    newdf = newdf.union(zeroes)
    target = df.subtract(newdf)
    return newdf, target

spark = SparkSession \
    .builder \
    .appName("pysteam") \
    .getOrCreate()

spark.sparkContext.setLogLevel('OFF')

dataset = spark.read.csv('Resources/ptFormateddataset1000.csv', header=True, inferSchema=True)

df = spark.createDataFrame(
    [(0, 0, 1.0), (0, 1, 1.0),  (1, 0, 1.0), (1, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0), (2, 2, 1.0)],
    ["user", "item", "rating"])

df.show()
#print(df[random.randin(0, df.groupBy(df.user).count())])
newdf, user = flipBit(df, 3)
user.show()
newdf.show()

als = ALS(rank=12, maxIter=16, regParam=5.0, alpha=25.0,  implicitPrefs=True, userCol="user", itemCol="item", ratingCol="rating")

model = als.fit(newdf)

predictions = model.transform(user)
print(predictions.collect())

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

#0.109186649323