
import pandas as pd
import numpy as np

dataset = pd.read_csv('Resources/dataset100.csv')

games = np.unique(dataset['appid'])
steamlist = []
matrix = pd.DataFrame(columns=games)
matrix.index.names = ["steamid"]
for i, row in enumerate(dataset.values):
    matrix.set_value(row[1], row[2], 1)
    #steamlist.append((row[1], row[2], 1))

matrix = matrix.fillna(value=0)

steamlist = list()
for i in matrix.index:
    for j in matrix.columns:
        steamlist.append((i, j, matrix.ix[i,j]))
print(matrix)
matrix = pd.DataFrame().from_records(steamlist)
matrix.to_csv('Resources/formateddataset100.csv', header=["steamid", "appid", "rating"], mode='w+', index=None, sep=',')