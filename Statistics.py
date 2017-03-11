import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class Statistics():
    def evaulateExperiment(self, path):
        data = pd.read_csv(path, compression='gzip',
                           usecols=['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank'])
        print(data)
        exp = data.groupby(by=['type'], axis=0).mean()
        print(exp)

    def evaulateuser(self, path, minGames=0, maxGames=float('inf')):
        data = pd.read_csv(path, compression='gzip')
        comp = data[data.rating == 1]
        apps = data[(data.rating == 1.0)].groupby(by=['steamid'])['rating'].count().reset_index()
        apps = apps[((apps.rating < maxGames) & (apps.rating > minGames))]
        datafilt = data.where((data.steamid.isin(apps.steamid)) & (data.rating == 1.0)).dropna()
        apps['rating'].hist(bins=500)
        plt.title("Game Distribution Histogram")
        plt.xlabel("Games")
        plt.ylabel("Users")
        plt.figtext(.82, .02, "nGames: {0}".format(datafilt.appid.nunique()))
        plt.figtext(.02, .02, "nUsers: {0}".format(apps.steamid.nunique()))
        plt.show()


stat = Statistics()

stat.evaulateuser('Resources/formateddataset10000.csv.gz', minGames=5)

#stat.evaulateExperiment('ExperimentData/E1-100-2-10.csv.gz')