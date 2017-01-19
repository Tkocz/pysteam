import json
import pandas as pd
from steamwebapi.api import IPlayerService, ISteamUserStats

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


json_data = [76561198048730871, 76561198180821795, 76561198008911412]
#json_file = open('steamkey.json', 'r')
#json_data = json.loads(json_file.read())
#json_file.close()

df = pd.DataFrame(None, columns=features)
df.index.names = ['steamID/appID']

for index, steamid in enumerate(json_data):
    response = playerserviceinfo.get_owned_games(steamid)['response']
    if len(response) > 1:
        games = response['games']
        for game in games:
            jointid = str(steamid) + "/" + str(game['appid'])
            df.set_value(jointid, 'playtime_forever', game['playtime_forever'])
            df.set_value(jointid, 'steamid', steamid)
            df.set_value(jointid, 'appid', game['appid'])
            try:
                achievements = steamuserstats.get_player_achievements(steamid, game['appid'])['playerstats'][
                    'achievements']
                df.set_value(jointid, 'achievements', achievementprocentage(achievements))
            except:
                df.set_value(jointid, 'achievements', None)

df.to_csv('dataset.csv', mode='w+')
