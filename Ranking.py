import pandas as pd

class Rank:
    def rank(self, predictions, users, i):
        results = pd.DataFrame()

        for type, data in predictions.items():
            # gör ett test av kod snuten
            subset = data.where((data.rating == 0.0) | (
                (data.steamid.isin(users.steamid)) & (data.appid.isin(users.appid)))).dropna()
            subset['rank'] = subset.groupby('steamid').cumcount() + 1
            targets = subset.merge(users, how='inner', on=('steamid', 'appid'))
            targets[['steamid', 'appid']] = targets[['steamid', 'appid']].astype(int)
            targets.insert(0, 'iter', i + 1)
            targets.insert(1, 'type', type)
            results = results.append(targets)

        return results
