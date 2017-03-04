from __future__ import print_function
import pandas as pd
import numpy as np
from tqdm import *

def evaluatetime(time):

    time = time / 60
    # if time == 0:
    #     return 0
    if time >= 0 and time <= 1:
        return 0
    else:
        return 1
    # if time > 1 and time <= 5:
    #     return 2
    # if time > 5 and time <= 20:
    #     return 3
    # if time > 20 and time < 50:
    #     return 4
    # else:
    #     return 5

AMOUNT = 100

dataset = pd.read_csv('Resources/dataset{0}.csv'.format(AMOUNT))

games = np.unique(dataset['appid'])
steamlist = []
matrix = pd.DataFrame(columns=games)
matrix.index.names = ["steamid"]
for row in tqdm(dataset.values):
    matrix.set_value(int(row[1]), int(row[2]), 1)

print('Wait..')
matrix = matrix.fillna(value=0)
sdf = matrix.to_sparse(fill_value=0)
print('nUsers:', len(dataset.index), 'Sparsity:', 1 - sdf.density, 'Density:', sdf.density)

steamlist = list()

for i in tqdm(matrix.index):
    for j in matrix.columns:
        steamlist.append((i, j, matrix.ix[i, j]))

matrix = pd.DataFrame().from_records(steamlist)
matrix.to_csv('Resources/formateddataset{0}.csv.gz'.format(AMOUNT), mode='w+', compression='gzip', header=["steamid", "appid", "rating"], index=None, sep=',')