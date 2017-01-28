
import pandas as pd
import numpy as np

dataset = pd.read_csv('Resources/dataset.csv')

games = np.unique(dataset['appid'])
steamlist = []
matrix = pd.DataFrame(columns=games)
for i, row in enumerate(dataset.values):
    steamlist.append((row[1], row[2], 1))

matrix = pd.DataFrame().from_records(steamlist)
print(steamlist)
print(matrix)
print('nGames: ' + str(len(games)))



matrix.to_csv('Resources/formateddataset.csv', mode='w+', header=None, index=None, sep=',')