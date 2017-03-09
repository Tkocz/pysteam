import pandas as pd

data = pd.read_csv('ExperimentData/E1-100-2-10.csv.gz', compression='gzip', usecols=['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank'])
print(data)
print(data.groupby(by=['type'], axis=0).mean())