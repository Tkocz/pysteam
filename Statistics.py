import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Statistics():


    def evaulateExperiment(self):
        data = pd.read_csv('ExperimentData/E1-100-2-10.csv.gz', compression='gzip', usecols=['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank'])
        print(data)
        exp = data.groupby(by=['type'], axis=0).mean()
        exp

    def evaulateuser(self):
        data = pd.read_csv('Resources/formateddataset10000copy.csv.gz', compression='gzip')
        apps = data[(data['rating'] == 1)].groupby(by=['steamid']).count()
        apps[apps['rating'] < 1000]['rating'].hist(bins=49)
        plt.title("Game Distribution Histogram")
        plt.xlabel("Games")
        plt.ylabel("Users")
        plt.figtext(.82, .02, "nGames: {0}".format(data.appid.nunique()))
        plt.figtext(.02, .02, "nUsers: {0}".format(data.steamid.nunique()))
        plt.show()


stat = Statistics()

stat.evaulateuser()

stat.evaulateExperiment()