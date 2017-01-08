
import json
import pandas as pd
import numpy as np
from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats

playerserviceinfo = IPlayerService()
steamuserstats = ISteamUserStats()

features = [
    'playtime_forever',
    'achievements',
    'number_of_games'
]

json_file = open('steamkey.json', 'r')
json_data = json.loads(json_file.read())
json_file.close()

df = pd.DataFrame(None, columns=features)
df.index.names = ['steamID/appID']

for index, id in enumerate(json_data):
    response = playerserviceinfo.get_owned_games(id)['response']
    if(len(response) > 1):
        games = response['games']
        for game in games:
            sa = str(id) + "," + str(game['appid'])
            df.set_value(sa, 'playtime_forever', game['playtime_forever'])
            achievements = steamuserstats.get_player_achievements(id, game['appid'])['playerstats']['achievements']
            print(achievements)
            achieved = [i for i in achievements if i['achieved'] == 1]
            df.set_value(sa, 'achievements', len(achieved) / len(achievements))
            df.set_value(sa, 'number_of_games', response['game_count'])
    print(df)


