from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import numpy as np

sc = SparkContext(appName="PythonCollaborativeFilteringSteam")
# $example on$
# Load and parse the data

data = sc.textFile('Resources/formateddataset.csv')
ratings = data.map(lambda l: l.split(',')) \
    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

# Build the recommendation model using Alternating Least Squares
rank = 8
numIterations = 10
alpha = 40.0
lamb = 10.0

model = ALS.trainImplicit(ratings, rank, numIterations, lambda_=lamb, alpha=alpha)

# Evaluate the model on training data
testdata = ratings.map(lambda p: (p[0], p[1]))
predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("Mean Squared Error = " + str(MSE))

# Save and load model
#model.save(sc, "target/tmp/myCollaborativeFilter")
sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")
# $example off$
print(ratesAndPreds.collect())

#[(180821795, (Rating(user=180821795, product=437900, rating=0.9973569628617586), Rating(user=180821795, product=262060, rating=0.9879263204708468), Rating(user=180821795, product=244160, rating=0.9879263204708468))),
# (8911412, (Rating(user=8911412, product=57690, rating=0.9916282379038353), Rating(user=8911412, product=72850, rating=0.9916282379038353), Rating(user=8911412, product=223530, rating=0.9916282379038353))),
# (48730871, (Rating(user=48730871, product=57690, rating=0.997629751405977), Rating(user=48730871, product=72850, rating=0.997629751405977), Rating(user=48730871, product=223530, rating=0.997629751405977)))]