import json
import pandas as pd
from steamwebapi.api import IPlayerService, ISteamUserStats, ISteamWebAPIUtil
from tqdm import *

playerserviceinfo = IPlayerService()
steamuserstats = ISteamUserStats()

features = [
    'steamid',
    'appid',
    'playtime_forever'
]


def achievementprocentage(ach):
    achieved = [i for i in ach if i['achieved'] == 1]
    return len(achieved) / len(ach)

AMOUNT = 1000

iddict = dict()

#json_data = [76561198048730871, 76561198180821795, 76561198008911412]
json_file = open('Resources/steamkey{0}.json'.format(AMOUNT), 'r')
json_data = json.loads(json_file.read())
json_file.close()

df = pd.DataFrame()
df.index.names = ['steamID/appID']
id = 0
for steamid in tqdm(json_data):
    response = playerserviceinfo.get_owned_games(steamid)['response']
    if len(response) > 1:
        games = response['games']
        #iddict[id] = steamid
        for game in games:
            jointid = str(steamid) + "/" + str(game['appid'])
            df = df.append(pd.DataFrame([[jointid, int(id), int(game['appid']), int(game['playtime_forever'])]]))
            # df.set_value(jointid, 'playtime_forever', game['playtime_forever'])
            # df.set_value(jointid, 'steamid', id)
            # df.set_value(jointid, 'appid', game['appid'])
        id += 1
df.columns = ['steamID/appID', 'steamid', 'appid', 'playtime_forever']
df = df.sort_values(by=['steamid', 'appid'], ascending=[True, False])
df.to_csv('Resources/dataset{0}.csv.gz'.format(AMOUNT), mode="w+", compression='gzip', columns=df.columns, index=None)

            # try:
            #     currentGame = client.getApp(name=game['name'])
            #     currentGenres = (list(currentGame.genres))
            #     currentGenres.extend(list(currentGame.categories))
            #     df.set_value(jointid, 'genres', currentGenres)
            # except:
            #     continue

            # try:
            #    achievements = steamuserstats.get_player_achievements(steamid, game['appid'])['playerstats'][
            #        'achievements']
            #   df.set_value(jointid, 'achievements', achievementprocentage(achievements))
            # except:
            #   df.set_value(jointid, 'achievements', None)