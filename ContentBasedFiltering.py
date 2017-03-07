from __future__ import print_function
import itertools
from scipy.spatial.distance import cosine
import pandas as pd
import steamfront
import requests
from tqdm import *

from pyspark.sql import SparkSession

class ContentBasedFiltering():
    """Content-based Filtering based on content similarities with Top-N recommendations."""

    __author__ = "Jim Glansk, Martin Bergqvist"

    def __init__(self):
        self.Model = None
        self.gm = None
        self.sm = None
        self.client = steamfront.Client()

        self.spark = SparkSession \
            .builder \
            .appName("pysteam") \
            .getOrCreate()

    def fit(self, X=None, nGames = None):
        """Fit traning data to model."""

        X = X.toPandas()
        self.generateGameGenreMatrix(X, nGames)
        self.train()

    def train(self):
        """Train model with traning data and generate a similarity matrix."""

        sm = self.generateSimMatrix(self.gm)

        self.sm = sm

    def getApps(self, appid):
        """Get app genres from api"""

        currentGenres = []
        try:
            currentGame = self.client.getApp(appid=appid)
            currentGenres = (list(currentGame.genres))
            currentGenres.extend(list(currentGame.categories))
        except:
            pass
        return currentGenres

    def generateGameGenreMatrix(self, appids=None, nGames=10, save=None, file_size=''):
        """Generate game-genre matrix (app * genre)"""

        if appids is None:
            steamAppList = 'http://api.steampowered.com/ISteamApps/GetAppList/v2/'
            dictGames = requests.get(steamAppList)
            jsonGames = dictGames.json()
            gameList = [i['appid'] for i in jsonGames['applist']['apps']['app']]
            appids = pd.DataFram(gameList, columns=['appid'])

        appids = appids['appid'].unique()
        gm = pd.DataFrame()
        gm.index.names = ["appid"]
        for id in tqdm(appids):
            for genre in self.getApps(id):
                if genre is not None:
                    gm.set_value(id, genre, int(1))
            #print('\rGenerate gm:{0}%'.format(round(i / appids.size * 100)), end="", flush=True)
        gm = gm.fillna(value=0)
        print('\n')
        self.gm = gm
        if save is not None:
            gm.to_csv('Resources/gamematrix{0}.csv.gz'.format(file_size), compression='gzip', mode='w+')

        return (gm)

    def generateSimMatrix(self, dataset=None, save=None, file_size=''):
        """Generate similarity matrix (app * app)"""

        if dataset is None:
            dataset = self.gm
        tdataset = dataset.T
        appids = tdataset.columns
        simMatrix = pd.DataFrame()
        simMatrix.index.names = ["appid"]
        for id1, id2 in tqdm(itertools.product(appids, appids)):
            simMatrix.set_value(id1, id2, 1 - cosine(tdataset[id1], tdataset[id2]))
        self.sm = simMatrix
        if save:
            simMatrix.to_csv('Resources/simmatrix{0}.csv.gz'.format(file_size), compression='gzip', mode='w+')

        return (simMatrix)

    def predict(self, df, nRec=10):
        """Predict similar games from user-owned games based on game genre tags"""

        ones = df[df.rating == 1.0].toPandas()
        preds = pd.DataFrame()
        users = ones.steamid.unique()
        for i in users:
            #focus user
            user = ones[(ones.steamid == i)]
            #drop NA-apps
            user = user[user.appid.isin(self.sm.index)]
            #drop user-owned games from axis 0
            result = self.sm.drop(user.appid, axis=0)
            #focus axis 1 on owned games
            result = result[user.appid]
            #create new column with max similarities from row
            result['prediction'] = result.max(axis=1)
            #sort all columns in decending order and take Top-N apps
            if nRec > 0:
                appids = result.sort_values(['prediction'], ascending=False).head(nRec)
            else:
                appids = result.sort_values(['prediction'], ascending=False)
            #arrange (steamid, appid, rating, predictions)
            newpred = appids.prediction
            newpred = newpred.reset_index()
            newpred.insert(0,'steamid',i)
            newpred.insert(2, 'rating', 0)
            #append result
            preds = preds.append(newpred)
        #formate to spark df
        predictions = self.spark.createDataFrame(preds)
        predictions = predictions.sort(['steamid', 'prediction'], ascending=[True, False])
        return predictions

    def readsimilaritymatrix(self, file_size):
        """Read similarity and Game-genre matrix from csv file"""

        if self.sm is None:
            sm = pd.read_csv('Resources/simmatrix{0}.csv.gz'.format(file_size), compression='gzip', index_col=['appid'], delimiter=',')
            sm.columns = sm.columns.astype('Int64')
            self.sm = sm
            gm = pd.read_csv('Resources/gamematrix{0}.csv.gz'.format(file_size), compression='gzip', index_col=['appid'], delimiter=',')
            self.gm = gm
        else:
            return('model already created')

    def showMatrix(self):
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

#cbf.readsimilaritymatrix(100)
#apps = pd.read_csv('Resources/formateddataset10000.csv.gz', compression='gzip')
#cbf.generateGameGenreMatrix(apps, save=True, file_size=10000)
#cbf.generateSimMatrix(cbf.gm, save=True, file_size=10000)
# sm = cbf.spark.read.csv('Resources/test.csv', header=True, inferSchema=True)
# print(sm.collect())
# print(sm[sm.steamid == 0])
# cbf.fit(sm)
# print(cbf.sm)
# prediction = cbf.predict(sm[sm.steamid == 0], 7)
#
# print(prediction)
# sm = sm.toPandas()
# matrix = cbf.generateGameGenreMatrix(sm['appid'])
# simmatrix = cbf.generateSimMatrix(matrix)
#
# print(matrix)
# print(simmatrix)