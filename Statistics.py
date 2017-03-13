import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


class Statistics:

    def evaluateExperiment(self, path):
        """Evaluate binary experiment data"""

        data = pd.read_csv(path, compression='gzip',
                           usecols=['iter', 'fold', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank'])
        print(data.groupby(by=['type'], axis=0).mean()['rank'])
        axe = data.groupby(by=['type', 'steamid'], axis=0).mean()['rank'].reset_index()
        fig = plt.figure()
        axe[axe.type == 'cbf'].plot(fig=fig)
        axe[axe.type == 'cf'].plot(fig=fig)
        #axe.plot()
        plt.show()

    def evaluateUser(self, path, minGames=float('-inf'), maxGames=float('inf')):
        """Evaluate distribution of games and users"""

        data = pd.read_csv(path, compression='gzip')
        apps = data[(data.rating == 1.0)].groupby(by=['steamid']).rating.count().reset_index()
        apps = apps[((apps.rating <= maxGames) & (apps.rating >= minGames))]
        datafilt = data.where((data.steamid.isin(apps.steamid)) & (data.rating == 1.0)).dropna()
        nGames = datafilt.appid.nunique()
        nUsers = apps.steamid.nunique()
        apps.rating.hist(bins=maxGames - minGames)
        plt.title("Game Distribution Histogram")
        plt.xlabel("Games")
        plt.ylabel("Users")
        plt.figtext(.82, .02, "nGames: {0}".format(nGames))
        plt.figtext(.02, .02, "nUsers: {0}".format(nUsers))
        plt.show()


stat = Statistics()

stat.evaluateUser('Resources/formateddataset10000.csv.gz', minGames=0, maxGames=500)

stat.evaluateExperiment('ExperimentData/E1-100-2-10-20170311.csv.gz')