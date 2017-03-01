from __future__ import print_function
import itertools
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import steamfront
import requests
client = steamfront.Client()

from pyspark.sql import SparkSession

class ContentBasedFiltering():
    """Content-based Filtering based on content similarities with Top-N recommendations."""

    __author__ = "Jim Glansk, Martin Bergqvist"

    def __init__(self):
        self.Model = None
        self.gm = None
        self.sm = None

    def fit(self, X=None):
        pass

    def fitt(self, X=None, nGames = None):
        """Fit traning data to model."""

        X = X.toPandas()
        X = X['appid'].unique()
        self.generateGameGenreMatrix(X, nGames)
        self.train()

    def train(self):
        """Train model with traning data and generate a similarity matrix."""

        sm = self.generateSimMatrix(self.gm)

        self.sm = sm

    def _getApps(self, appid):
        """Get app genres from api""" #http://store.steampowered.com/api/appdetails/?appids=<APPID>&filters=genre

        currentGenres = []
        try:
            currentGame = client.getApp(appid=appid)
            currentGenres = (list(currentGame.genres))
            currentGenres.extend(list(currentGame.categories))
        except:
            pass
        return currentGenres

    def generateGameGenreMatrix(self, appids=None, nGames = 10, save=None):
        """Generate game-genre matrix (app * genre)"""

        if appids is None:
            steamAppList = 'http://api.steampowered.com/ISteamApps/GetAppList/v0001/'
            dictGames = requests.get(steamAppList)
            jsonGames = dictGames.json()
            gameList = [i['appid'] for i in jsonGames['applist']['apps']['app']]
            uniqids = np.unique(gameList)
            appids = uniqids[2: nGames]

        gm = pd.DataFrame()
        gm.index.names = ["appid"]
        for i, id in enumerate(appids):
            for genre in self._getApps(id):
                if genre is not None:
                    gm.set_value(id, genre, int(1))
            print('\rGenrate gm:{0}%'.format(round(i / appids.size * 100)), end="", flush=True)
        gm = gm.fillna(value=0)
        print('\n')
        self.gm = gm
        return(gm)
        if save:
            gm.to_csv('Resources/gamematrix.csv', mode='w+')

    def generateSimMatrix(self, dataset=None, save=None):
        """Generate similarity matrix (app * app)"""

        if dataset is None:
            dataset = self.gm
        tdataset = dataset.T
        appids = tdataset.columns
        simMatrix = pd.DataFrame()
        simMatrix.index.names = ["appid"]
        count = 0
        for id1, id2 in itertools.product(appids, appids):
            simMatrix.set_value(id1, id2, 1 - cosine(tdataset[id1], tdataset[id2]))
            count += 1
            print('\rGenerat sm: {0}%'.format(round(count / (appids.shape[0] ** 2) * 100)), end="", flush=True)
        self.sm = simMatrix
        return(simMatrix)
        if save:
            simMatrix.to_csv('Resources/cbfsimmatrix.csv', mode='w+')

    def predict(self, df, nRec=10):
        """Predict similar games from user-owned games based on game genre tags"""

        ones = df[df.rating == 1.0].toPandas()
        sim = pd.DataFrame(None, columns=ones.columns)
        users = ones.steamid.unique()
        for i in users:
            user = ones[(ones.steamid == i)]
            user = user[user.appid.isin(self.sm.index)]
            result = self.sm.drop(user.appid, axis=0)
            result = result[user.appid]
            result['TopN'] = result.max(axis=1)
            appids = result.sort_values(['TopN'], ascending=False).head(nRec)
        sim = sim.append(appids)
        return sim

    def readsimilaritymatrix(self):
        """Read similarity and Game-genre matrix from csv file"""

        if self.sm is None:
            sm = pd.read_csv('Resources/cbfsimmatrix.csv', index_col=['appid'], delimiter=',')
            sm.columns = sm.columns.astype('Int64')
            self.sm = sm
            gm = pd.read_csv('Resources/gamematrix.csv', index_col=['appid'], delimiter=',')
            self.gm = gm
        else:
            return('model already created')

    def show(self):
        """Show similarity and game-genre matrix if created"""

        if self.gm is not None:
            print('GameMatrix')
            with pd.option_context('display.max_rows', self.gm.shape[0], 'display.max_columns', self.gm.shape[1]):
                print(self.gm)
        if self.sm is not None:
            print('SimilarityMatrix')
            with pd.option_context('display.max_rows', self.sm.shape[0], 'display.max_columns', self.sm.shape[1]):
                print(self.sm)

#test CBF
# from pyspark.sql import SparkSession
#
# spark = SparkSession \
#             .builder \
#             .appName("pysteam") \
#             .getOrCreate()
# cbf = ContentBasedFiltering()
# #cbf.readsimilaritymatrix()
# sm = spark.read.csv('Resources/test.csv', header=True, inferSchema=True)
# print(sm.collect())
# print(sm[sm.steamid == 0])
# cbf.fit(sm)
# print(cbf.sm)
# prediction = cbf.predict(sm[sm.steamid == 0], 7)
#
# print(prediction)
