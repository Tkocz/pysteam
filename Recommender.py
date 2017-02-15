from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import numpy as np
import time


ss = spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

sc = ss.sparkContext
sc.setLogLevel('OFF')

# $example on$d
# Load and parse the data

df = spark.read.csv('Resources/formateddataset.csv', header=True, inferSchema=True)
print(df.select('steamid').show())


ratings = df.rdd
training, test = ratings.randomSplit([0.8, 0.2])


# Build the recommendation model using Alternating Least Squares
rank = 10
numIterations = 10
alpha = 40.0
lamb = 0.01

model = ALS.trainImplicit(ratings, rank, numIterations, lambda_=lamb, alpha=alpha)

testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
jim = model.recommendProducts(2, 10)
print(jim)
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
print("Mean Squared Error = " + str(MSE))
# Save and load model
#model.save(sc, "target/tmp/myCollaborativeFilter")
#sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
# $example off$
print(ratesAndPreds.collect())