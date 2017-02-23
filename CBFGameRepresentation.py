import itertools
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import steamfront
import requests
client = steamfront.Client()

class CBF():

    def __init__(self):
        self.Model = None

    def getApps(self, appid):
        currentGenres = []
        try:
            currentGame = client.getApp(appid=appid)
            currentGenres = (list(currentGame.genres))
            currentGenres.extend(list(currentGame.categories))
        except:
            pass
        return currentGenres

    def generateGameMatrix(self):
        steamAppList = 'http://api.steampowered.com/ISteamApps/GetAppList/v0001/'
        dictGames = requests.get(steamAppList)
        jsonGames = dictGames.json()
        gameList = [i['appid'] for i in jsonGames['applist']['apps']['app']]
        uniqids = np.unique(gameList)
        uniqids = uniqids[3:100]
        gm = pd.DataFrame()
        count = 0;
        for id in uniqids:
            for genre in self.getApps(id):
                if genre != None:
                    gm.set_value(id, genre, 1)
            count += 1
            print('\r{0}%'.format(round((count / uniqids.shape[0]) * 100)), end="", flush=True)
        gm = gm.fillna(value=0)
        return gm

    def generateSimMatrix(self, dataset):
        tdataset = dataset.T
        appids = tdataset.columns
        simMatrix = pd.DataFrame()
        for id1, id2 in itertools.product(appids, appids):
            simMatrix.set_value(id1, id2, 1 - cosine(tdataset[id1], tdataset[id2]))
        return simMatrix

    def recommend(self, dataset, appid, n = 1):

        df = dataset.drop(appid, axis=0)
        return df.nlargest(n, appid)
test = CBF()
result = test.generateGameMatrix()
result.to_csv('Resources/gamematrix.csv', mode='w+')
matrix = test.generateSimMatrix(result)
matrix.to_csv('Resources/cbfsimmatrix.csv', mode='w+')
print(matrix)
print('recommend', test.recommend(matrix, [730], 10))
print(result.T)