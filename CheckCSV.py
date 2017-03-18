import requests
import pandas as pd
from pandas.io.json import json_normalize
from time import sleep
from tqdm import *

class CheckCSV:

    def removeLegacy(self, path=None):
        """Remove obsolete games from chosen dataset"""

        df = pd.read_csv(path, compression='gzip')
        print(df.shape)
        gamelist = pd.read_csv('Resources/validgames.csv', index_col=[0])
        filter_df = pd.merge(df, gamelist, on='appid', how='inner')
        filter_df = filter_df.dropna()
        filter_df = filter_df.sort_values(['steamid', 'appid'], ascending=[True, True])
        print(filter_df.shape)
        filter_df.to_csv(path, compression='gzip', columns=['steamid', 'appid', 'rating'], mode='w+', index=None)

    @staticmethod
    def remove_min_games(df, minGames=0):
        data = df.copy()
        users = data[(data.rating == 1.0)].groupby(by=['steamid']).rating.count().reset_index()
        users = users[((users.rating >= minGames))]
        datafilt = data.where((data.steamid.isin(users.steamid))).dropna()
        #print(df.steamid.nunique(), df.appid.nunique())
        #print(datafilt.steamid.nunique(), datafilt.appid.nunique())
        #with pd.option_context('display.max_rows', df.shape[0], 'display.max_columns', 6):
            #print(pd.concat([df, datafilt], axis=1))
        datafilt[['steamid', 'appid']] = datafilt[['steamid', 'appid']].astype(int)
        return datafilt

    def checkapp(self, app):
        """Check if game is applies for Content-based filtering"""

        data = requests.get('http://store.steampowered.com/api/appdetails?appids={0}&format=json'.format(app)).json()

        if data[str(app)]["success"]:
            type = data[str(app)]["data"]['type']
            if (type != 'game'):
                return False
        return data[str(app)]["success"]

    def getAllValidGames(self):
        """Check all games not sutied for Content-based filtering"""

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
        """Get list of all games available on steam as of knowledge"""

        data = requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v2/').json()
        df = json_normalize(data['applist'], 'apps')
        df.to_csv('Resources/allgames.csv.gz', compression='gzip', index=False)

    def getGamePrices(self):
        """Get the pricing for each game in the chosen file, export into price-file"""
        result = pd.DataFrame()
        gamelist = pd.read_csv('Resources/validgames.csv')
        for appid in tqdm(gamelist['appid']):
            pricedata = requests.get('http://store.steampowered.com/api/appdetails?appids={0}&filters=price_overview&format=json'.format(appid)).json()
            try:
                df = json_normalize(pricedata)
                df.columns = ['currency', 'discount', 'initial', 'price', 'success']
                del df['discount'], df['initial'], df['success']
                df.set_value(0, 'appid', int(appid))
                df['price'] /= 100
                result = result.append(df)
                if appid == 359070:
                    sleep(30)
            except:
                continue

        result.to_csv('Resources/gameprices.csv.gz', compression='gzip', index=False)

csv = CheckCSV()
#csv.getGamePrices()
#csv.removeLegacy('Resources/formateddatasetMJL.csv.gz')
#df = pd.read_csv('Resources/formateddatasetMJL.csv.gz', compression='gzip')
#csv.removeMinGames(df, minGames=5)