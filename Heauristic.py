
import pandas as df
import numpy as np

dataset = df.read_csv('dataset.csv')


print(dataset)

steamIDs = np.unique(dataset['steamid'])
games = np.unique(dataset['appid'])

print(games)
print(steamIDs)