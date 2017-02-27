import itertools
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
import steamfront
import requests
client = steamfront.Client()

class ContentBasedFiltering():
    """Content-based Filtering based on content similarities with Top-N recommendations."""
    __author__ = "Jim Glansk, Martin Bergqvist"
    __copyright__ = "Copyright 2017, OpenHorse"

    def __init__(self):
        self.Model = None
        self.gm = None
        self.sm = None

    def fit(self, X=None, nGames = None):
        """Fit traning data to model."""

        self.generateGameGenreMatrix(X['appid'], nGames)

        self.train(X)

    def train(self):
        """Train model with traning data and generate a similarity matrix."""

        sm = self.generateSimMatrix(self.gm)

        self.sm = sm

    def _getApps(self, appid):
        """Get app genres from api"""

        currentGenres = []
        try:
            currentGame = client.getApp(appid=appid)
            currentGenres = (list(currentGame.genres))
            currentGenres.extend(list(currentGame.categories))
        except:
            pass
        return currentGenres

    def generateGameGenreMatrix(self, appids=None, nGames = 10, save=False):
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
        count = 0;
        for id in appids:
            for genre in self._getApps(id):
                if genre != None:
                    gm.set_value(id, genre, 1)
            count += 1
        gm = gm.fillna(value=0)
        print('\n')
        self.gm = gm
        if save:
            gm.to_csv('Resources/gamematrix.csv')

    def generateSimMatrix(self, dataset, save=False):
        """Generate similarity matrix (app * app)"""

        tdataset = dataset.T
        appids = tdataset.columns
        simMatrix = pd.DataFrame()
        simMatrix.index.names = ["appid"]
        count = 0
        for id1, id2 in itertools.product(appids, appids):
            simMatrix.set_value(id1, id2, 1 - cosine(tdataset[id1], tdataset[id2]))
            count += 1
        self.sm = simMatrix
        if save:
            simMatrix.to_csv('Resources/gamematrix.csv')

    def predict(self, user, nRec=10):
        """Predict similar games from user-owned games based on game genre tags"""

        df = self.sm.drop(user, axis=0)
        df = df[(user)]
        df['TopN'] = df.max(axis=1)
        #print(df.sort_values(['TopN'], ascending=False))
        appids = df.sort_values(['TopN'], ascending=False).head(nRec).index.values
        return appids

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
cbf = ContentBasedFiltering()
cbf.readsimilaritymatrix()

prediction = cbf.predict([907], 10)

print(prediction)
