
import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv')


games = np.unique(dataset['appid'])

matrix = pd.DataFrame(columns=games)

for i, row in enumerate(dataset.values):
    print(row)
    matrix.set_value(row[1], row[2], 1)
matrix = matrix.fillna(value=0)
print(matrix.values)
print(len(games))

matrix.to_csv('formateddataset.csv', mode='w+')
