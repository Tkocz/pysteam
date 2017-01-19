
import pandas as df
import numpy as np

dataset = df.read_csv('dataset.csv')


games = np.unique(dataset['appid'])

matrix = df.DataFrame(columns=games)

for i, row in enumerate(dataset.values):
    print(row)
    matrix.set_value(row[1], row[2], 1)
matrix = matrix.fillna(value=0)
print(matrix.values)
print(len(games))