import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import CheckCSV as CC

plt.style.use('ggplot')


class Statistics:

    def evaluateExperiment(self, path):
        """Evaluate binary experiment data"""
        csv = CC.CheckCSV()
        ngames = csv.get_n_owned_games(10000)

        data = pd.read_csv(path, compression='gzip',
                           usecols=['iter', 'type', 'steamid', 'appid', 'rating', 'prediction', 'rank'])
        data = pd.merge(data, ngames, on='steamid')
        with pd.option_context('display.max_rows', data.shape[0], 'display.max_columns', data.shape[1]):
            print(data.groupby(by=['type'], axis=0).mean()['rank'])
            #print(data.groupby(by=['type', 'steamid', 'appid'], axis=0).mean()['rank'])
            #print(data.groupby(by=['type', 'steamid'], axis=0).mean()['rank'])

        CBF = data[data.type == 'cbf']
        CF = data[data.type == 'cf']

        if(np.array_equal(CBF[['steamid', 'appid']], CF[['steamid', 'appid']]) and CBF.values.shape == CF.values.shape):
            print('PASS - Equal')
        else:
            print('FAIL - Not Equal')

        axe = data[(data.nGames <= 1000)].groupby(by=['type', 'nGames'], axis=0).mean()['rank'].reset_index()
        g = sns.lmplot(y='rank', x='nGames', data=axe, hue='type', fit_reg=False)
        axe1 = data[(data.nGames <= 1000)].groupby(by=['type', 'appid'], axis=0).mean()['rank'].reset_index()
        g1 = sns.lmplot(y='rank', x='appid', data=axe1, hue='type', fit_reg=False)
        cfaxe = data[(data.type == 'cf') & (data.nGames <= 100)]
        cbfaxe = data[(data.type == 'cbf') & (data.nGames <= 100)]
        sns.set()

        cfg = sns.jointplot(x="nGames", y="rank", data=cfaxe, kind='kde', color="b")
        cfg.set_axis_labels("Number of Games", "Rank")
        cfg.fig.suptitle('CF')
        cbfg = sns.jointplot(x="nGames", y="rank", data=cbfaxe, kind='kde', color="r")
        cbfg.set_axis_labels("Number of Games", "Rank")
        cbfg.fig.suptitle('CBF')
        plt.show()

    def evaluateUser(self, path, minGames=0, maxGames=float('inf')):
        """Evaluate distribution of games and users"""

        data = pd.read_csv(path, compression='gzip')
        apps = data[(data.rating == 1.0)].groupby(by=['steamid']).rating.count().reset_index()
        apps = apps[((apps.rating <= maxGames) & (apps.rating >= minGames))]
        datafilt = data.where((data.steamid.isin(apps.steamid)) & (data.rating == 1.0)).dropna()
        nGames = datafilt.appid.nunique()
        nUsers = apps.steamid.nunique()
        if maxGames == float('inf'):
            maxGames = apps.rating.max()
        apps.rating.hist(bins=maxGames - minGames)
        plt.title("Game Distribution Histogram")
        plt.xlabel("Games")
        plt.ylabel("Users")
        plt.figtext(.82, .02, "nGames: {0}".format(nGames))
        plt.figtext(.02, .02, "nUsers: {0}".format(nUsers))
        plt.show()


stat = Statistics()

#stat.evaluateUser('Resources/formateddataset10000.csv.gz', minGames=0, maxGames=1000)

stat.evaluateExperiment('ExperimentData/E1-10000-30-2-201708172154.csv.gz')

#Titta på variationen mellan spel spelar äger (usertags) / entropy / consinesimilarity och antal spel
#Kolla om det skilelr mellan usertags och latenta predictioner genom att mätta medelvärdet för ranken för grupperade spel.
#Hur vanligt förekommande är spel
#variance impoprtence med ranfomforest med variabler som variance, genere, frekvence med.
# Vad kan vi se utifrån steam tjänst användarens preferencer.
# speltid som heaurustic.
# I vilket lägge ska jag använda repektive algorith beroendes av en unik spelare.
# Finns det nått intressant med att titta på att en spealr faktiskt köper ett spel och sedan spelar det.