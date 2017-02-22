import itertools
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import steamfront
client = steamfront.Client()

def getApp(appid):

    currentGame = client.getApp(appid=appid)
    currentGenres = (list(currentGame.genres))
    currentGenres.extend(list(currentGame.categories))

    return currentGenres

def generateGameMatrix(appids):
    uniqids = np.unique(appids)

    gm = pd.DataFrame()
    for id in uniqids:
        for genre in getApp(id):
            gm.set_value(id, genre, 1)
    gm = gm.fillna(value=0)
    return gm

def generateSimMatrix(dataset):
    tdataset = dataset.T
    appids = tdataset.columns
    simMatrix = pd.DataFrame()
    for id1, id2 in itertools.product(appids, appids):
        simMatrix.set_value(id1, id2, 1 - cosine(tdataset[id1], tdataset[id2]))
    return simMatrix

def recommend(dataset, appid, n = 1):

    df = dataset.drop(appid, axis=0)
    return df.nlargest(n, appid)

result = generateGameMatrix([13200, 13210, 13230, 13240, 13250])
matrix = generateSimMatrix(result)
print(matrix)
print(recommend(matrix, [13250]))
print(result.T)