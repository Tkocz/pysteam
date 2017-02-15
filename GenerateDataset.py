import json
import pandas as pd
import numpy as np
from steamwebapi.api import IPlayerService, ISteamUserStats
import uuid
playerserviceinfo = IPlayerService()
steamuserstats = ISteamUserStats()

features = [
    'steamid',
    'appid',
    'playtime_forever',
    'achievements'
]


def achievementprocentage(ach):
    achieved = [i for i in ach if i['achieved'] == 1]
    return len(achieved) / len(ach)

iddict = dict()


#json_data = [76561198008911412, 76561198048730871, 76561198180821795]
json_file = open('Resources/steamkey1000.json', 'r')
json_data = json.loads(json_file.read())
json_file.close()

df = pd.DataFrame(None, columns=features)
df.index.names = ['steamID/appID']
id = -1;
for index, steamid in enumerate(json_data):
    response = playerserviceinfo.get_owned_games(steamid)['response']
    if len(response) > 1:
        games = response['games']
        id = id + 1
        print(round((index/len(json_data)) * 100, 0), '%')
        iddict[id] = steamid
        for game in games:
            jointid = str(steamid) + "/" + str(game['appid'])
            df.set_value(jointid, 'playtime_forever', game['playtime_forever'])
            df.set_value(jointid, 'steamid', int(id))
            df.set_value(jointid, 'appid', game['appid'])
            #try:
            #    achievements = steamuserstats.get_player_achievements(steamid, game['appid'])['playerstats'][
            #        'achievements']
             #   df.set_value(jointid, 'achievements', achievementprocentage(achievements))
            #except:
             #   df.set_value(jointid, 'achievements', None)
print(iddict)
df.to_csv('Resources/dataset1000.csv', mode='w+')

