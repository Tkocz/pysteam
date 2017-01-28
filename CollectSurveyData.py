import numpy as np
import pandas as pd

games = ['72850', '252950', '730', '271590', '294100', '245620', '292030', '482730', '289070', '47890', '570', '268500',
         '359550', '346110', '8500', '102600', '379720', '251470', '70600', '620']
stars = {"Don't own this game on steam": 0, "Don't own this game": 0, "Disliked strongly": 1, "Disliked": 2,
         "Liked slightly": 3, "Liked": 4,
         "Liked strongly": 5, "Haven't played / No opinion": 0}

convertfunc = lambda x: stars[x]
dataset = pd.read_csv('Resources/Steam Game Ranking.csv', usecols=range(1, 22), index_col=[0], header=None,
                      skiprows=[0], )
dataset.columns = games
dataset.index.names = ['steamid']
dataset = dataset.applymap(convertfunc)
dataset = dataset.applymap(np.int64)
print(dataset)

dataset.to_csv('Resources/userratings.csv')
