from steamwebapi.api import ISteamUser, IPlayerService, ISteamUserStats

APIkey = "B24B7331A0D05E5528F268C6C912B273"

steamID = 76561198048730871

steamuserinfo = ISteamUser(steam_api_key='B24B7331A0D05E5528F268C6C912B273')
playerserviceinfo = IPlayerService(steam_api_key='B24B7331A0D05E5528F268C6C912B273')

usersummary = steamuserinfo.get_player_summaries(steamID)['response']['players'][0]

owned_games = playerserviceinfo.get_owned_games(steamID=steamID)


print(usersummary)

print(owned_games)