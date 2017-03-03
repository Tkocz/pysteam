import requests
import pandas as pd
from pandas.io.json import json_normalize
from time import sleep
from tqdm import *

class CheckCSV():
#37156
    def removeLegacy(self, path):
        #df = pd.read_csv(path)

        gamelist = pd.read_csv('Resources/allgames.csv')
        games = []
        gamelist = gamelist['appid'].unique()
        for appid in tqdm(gamelist):
            sleep(1)
            if self.checkapp(appid):
                games.append(appid)

        validgames = pd.DataFrame(games)
        validgames.to_csv('Resources/validgamesall.csv')
        # newdf = df.copy()
        # games = df.appid.unique()
        # for i, app in enumerate(games):
        #     if app not in gamelist:
        #         df = df[df.appid != app]
        #     print('\r{0}%'.format(round(i / games.shape[0] * 100)), end="", flush=True)
        # print(df != newdf)
        #newdataset.to_csv(path, mode='w+')

    def checkapp(self, app):
        data = requests.get('http://store.steampowered.com/api/appdetails?appids={0}&format=json'.format(app)).json()
        # if data[str(app)]["success"]:
        #     type = data[str(app)]["data"]['type']
        #     if(type != 'game'):
        #         return False
        return data[str(app)]["success"]

    def getAllGames(self):
        data = requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v2/').json()
        df = json_normalize(data['applist'], 'apps')
        df.to_csv('Resources/allgames.csv', index=False)

    def check_size(self, path):
        dataset = pd.read_csv(path)
        dataset['steamid'].unique().size()


csv = CheckCSV()
csv.removeLegacy( path=None)