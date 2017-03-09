import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from time import sleep
from tqdm import *

class CheckCSV():

    def removeLegacy(self, path=None):
        df = pd.read_csv(path, compression='gzip')
        print(df.shape)
        gamelist = pd.read_csv('Resources/validgames.csv', index_col=[0])
        filter_df = pd.merge(df, gamelist, on='appid', how='inner')
        filter_df = filter_df.dropna()
        filter_df = filter_df.sort_values(['steamid', 'appid'], ascending=[True, True])
        print(filter_df.shape)
        filter_df.to_csv(path, compression='gzip', columns=['steamid', 'appid', 'rating'], mode='w+', index=None)


    def checkapp(self, app):

        data = requests.get('http://store.steampowered.com/api/appdetails?appids={0}&format=json'.format(app)).json()

        if data[str(app)]["success"]:
            type = data[str(app)]["data"]['type']
            if (type != 'game'):
                return False
        return data[str(app)]["success"]

    def getAllValidGames(self):

        gamelist = pd.read_csv('Resources/allgames.csv.gz', compression='gzip')
        games = []
        appids = gamelist['appid'].unique()
        i = 0
        pbar = tqdm(total=appids.size)
        pbar.set_description('Processing  ')
        appsize = appids.shape[0] - 1
        while (i <= appsize):
            try:
                if self.checkapp(appids[i]):
                    games.append(appids[i])
                i += 1
                pbar.update(1)
            except:
                pbar.set_description('{(-_-)}Zzz..')
                sleep(5)
                pbar.set_description('Processing  ')
                continue
        pbar.close()
        validgames = pd.DataFrame(games)
        validgames.to_csv('Resources/validgames.csv')

    def check_size(self, path):
        dataset = pd.read_csv(path)
        dataset['steamid'].unique().size()

    def getValidGamesList(self):
        data = requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v2/').json()
        df = json_normalize(data['applist'], 'apps')
        df.to_csv('Resources/allgames.csv.gz', compression='gzip', index=False)


data = pd.read_csv('Resources/formateddataset100.csv.gz', compression='gzip')
#size = data.groupby(by=['steamid', 'rating'])
#size = pd.DataFrame({'count': data.groupby(by=['steamid', 'rating']).count()}).reset_index()
#print(size)
apps = data[(data['rating'])].groupby(by=['steamid']).count()
print(apps)

#ax2 = data.groupby(by=['rating']).count().hist(bins=20, normed=True)
#ax1 = data['steamid'].hist(by=data['rating'], bins=20, normed=True) #Figure1
plt.show()
#test = data.groupby(by=['steamid', 'rating'], axis=0).count()
#hist = data['rating'].hist(by=data['steamid'])
#print(test)
#print(hist)
#csv = CheckCSV()
#csv.removeLegacy('Resources/formateddataset10000.csv.gz')