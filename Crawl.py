from __future__ import print_function
from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats
import json

steamID = [76561198048730871, 76561198180821795, 76561198008911412]
steamuserinfo = ISteamUser()
playerserviceinfo = IPlayerService()
count = 0;
while len(steamID) < 10000:
    state = steamuserinfo.get_player_summaries(steamID[count])['response']['players'][0]['communityvisibilitystate']
    if state == 3:
        friendslist = steamuserinfo.get_friends_list(steamID[count])['friendslist']['friends']
        for i in friendslist:
            if i['steamid'] not in steamID:
                steamID.append(int(i['steamid']))
    print(count)
    count += 1
print(len(steamID))
json_file = open('steamkey.json', 'w')
json.dump(steamID, json_file)
json_file.close()
