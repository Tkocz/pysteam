
import json
import pandas as pd
import numpy as np
from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats

playerserviceinfo = IPlayerService()
steamuserstats = ISteamUserStats()

features = [
    'playtime_forever',
    'achievements'
]

json_file = open('steamkey.json', 'r')
json_data = json.loads(json_file.read())
json_file.close()

df = pd.DataFrame(None, index=np.arange(len(json_data)), columns=features)
df.index.names = ['steamID/appID']

for index, id in enumerate(json_data):

    games = playerserviceinfo.get_owned_games(id)['response']['games']
    print(games)
    for game in games:
        war = game['appid']
        df.set_value(id, 'playtime_forever', game['playtime_forever'])
        achievements = steamuserstats.get_player_achievements(id, game['appid'])['playerstats']['achievements']
        achieved = [i for i in achievements if i['achieved'] is 1]
        df.set_value(int(str(id) + game['appid'], 'achievements', len(achieved) / len(achievements)))
        print(game['appid'])
    print(index, id)


