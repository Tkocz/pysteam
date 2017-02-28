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


class CollaborativFiltering():
    """Content-based Filtering based on content similarities with Top-N recommendations."""
    __author__ = "Jim Glansk, Martin Bergqvist"
    __copyright__ = "Copyright 2017, OpenHorse"

    def __init__(self):
        self.spark = SparkSession \
            .builder \
            .appName("pysteam") \
            .getOrCreate()

        self.spark.sparkContext.setLogLevel('OFF')
        self.bestModel = None
        self.bestValidationRmse = None
        self.bestRank = None
        self.bestLambda = None
        self.bestNumIter = None
        self.bestAlpha = None
        self.model = None

    def fit(self, X, rank=None, nIter=None, lmbda=None, alpha=None):
        """Fit traning data to model."""

        rank = 12 if rank is None else self.rank
        nIter = 10 if nIter is None else self.bestNumIter
        lmbda = 0.01 if lmbda is None else self.bestLambda
        alpha = 40.0 if alpha is None else self.bestAlpha

        self.train(X, rank, nIter, lmbda, alpha)

    def train(self, X, rank, nIter, lmbda, alpha):
        """Train model with traning data and generate a similarity matrix."""

        self.model = ALS(implicitPrefs=True,
                         rank=rank,
                         maxIter=nIter,
                         regParam=lmbda,
                         alpha=alpha,
                         userCol="steamid",
                         itemCol="appid",
                         ratingCol="rating").fit(X)

        return self.model

    def predict(self, users):
        """Predict similar games from user-owned games based on game genre tags"""
        predictions = self.model.transform(users)

        return predictions

    def evalModel(self, X, numTrain):
        """Evaluate model from training"""
        pdf = pd.DataFrame()
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        count = 0
        for i in range(numTrain):
            (train, test) = X.randomSplit([0.8, 0.2])
            model = ALS(implicitPrefs=True,
                        rank=self.bestRank,
                        maxIter=self.bestNumIter,
                        regParam=self.bestLambda,
                        alpha=self.bestAlpha,
                        userCol="steamid",
                        itemCol="appid", ratingCol="rating").fit(train)

            predictions = model.transform(test)
            ones = predictions.where("rating=1")
            zeroes = predictions.where("rating=0")
            predictors = {'all': predictions, 'zeroes': zeroes, 'ones': ones}

            for s, p in predictors.items():
                pdf = pdf.append(pd.DataFrame([[i, s,
                                                evaluator.setParams(metricName="rmse").evaluate(p),
                                                evaluator.setParams(metricName="mse").evaluate(p),
                                                evaluator.setParams(metricName="mae").evaluate(p)]]))
            count += 1
            print(round((i / 10) * 100, 0), '%')
        pdf.columns = ['iteration', 'type', 'rmse', 'mse', 'mae']
        print(pdf)
        print(pdf.groupby(by=['type'], axis=0).mean())

    def paramOpt(self, X, numVal, numParam):
        """Optimize parameters to find best model"""

        ranks = np.arange(8, 20, 2)
        lambdas = np.linspace(0.01, 0.5, 10)
        numIters = np.arange(8, 20, 2)
        alpha = np.arange(8, 40, 2)
        bestValidationRmse = float("inf")
        bestRank = 0
        bestLambda = -1.0
        bestNumIter = -1
        bestAlpha = 0
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        pm = [i for i in itertools.product(ranks, lambdas, numIters, alpha)]
        indexes = np.random.permutation(len(pm))
        indexes = [pm[i] for i in indexes[:numParam]]
        count = 0
        for rank, lmbda, numIter, alf in indexes:
            for i in range(numVal):
                (opttrain, optval) = validation.randomSplit([0.8, 0.2])
                model = ALS(implicitPrefs=True, rank=rank, regParam=lmbda, maxIter=numIter, alpha=alf,
                            userCol="steamid",
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
        print("The best model was trained on evalData with rank = %d, lambda = %.2f, alpha = %d, " % (
                            bestRank, bestLambda, bestAlpha) \
                            + "numIter = %d and RMSE %f." % (bestNumIter, bestValidationRmse))
        self.bestRank, self.bestNumIter, self.bestLambda, self.bestAlpha = bestRank, bestNumIter, bestLambda, bestAlpha
        self.bestModel = bestModel
        return bestModel

    def flipBit(self, df, nUser):
        ones = df[df.rating == 1.0].toPandas().values
        zeroes = df[df.rating == 0.0]
        id = np.array(np.unique(ones[:, 0]), dtype=int)
        index = np.random.choice(id, nUser, replace=False)
        ones[index, 2] = 0.0
        newpdf = pd.DataFrame(ones, columns=["steamid", "appid", "rating"])
        newpdf[["steamid", "appid"]] = newpdf[["steamid", "appid"]].astype(int)
        newdf = self.spark.createDataFrame(newpdf)
        newdf = newdf.union(zeroes)
        target = df.subtract(newdf)
        return newdf, target

    def takeSamples(self, df):
        pass

# test CF
CF = CollaborativFiltering()
dataset = CF.spark.read.csv('Resources/formateddataset1000.csv', header=True, inferSchema=True)
(training, validation) = dataset.randomSplit([0.9, 0.1])
CF.paramOpt(validation, 1, 1)
#CF.evalModel(training, 1)
(train, test) = training.randomSplit([0.8, 0.2])
CF.fit(train)
predictions = CF.predict(test)
#print(predictions.collect())