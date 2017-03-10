import json

from bs4 import BeautifulSoup
import mechanize
import pandas as pd
from tqdm import *

class GameTags():

    def __init__(self):
        self.mb = mechanize.Browser()


    def getGameTags(self, appids):
        """Get tags of games thats available in steam store"""

        mb = self.mb
        mb.open("http://store.steampowered.com/agecheck/app/{0}/".format(10))
        mb.select_form(nr=1)
        mb.form['ageYear'] = ["1900"]
        mb.submit()

        tag_df = pd.DataFrame()

        for appid in tqdm(appids):
            tag_list = 0
            mb.open('http://store.steampowered.com/agecheck/app/{0}/'.format(appid))
            soup = BeautifulSoup(mb.response().read(), "html5lib")
            tags = soup.find('div', 'glance_tags popular_tags')
            if tags != None:
                tag_list = [t.text.strip().encode('utf8')
                            for t in tags.findAll("a", {"href": True})]
            tag_df = tag_df.append(pd.DataFrame([[appid, tag_list]]))
        tag_df.columns = ['appid', 'tags']
        tag_df = tag_df[tag_df.tags != 0]
        tag_df.to_csv('Resources/gamegenres.csv.gz', compression='gzip')
        return(tag_df)
    def converting(self):
        """Convert genres from unicode to string format"""

        data = pd.read_csv('Resources/gamegenres.csv.gz', compression='gzip')

        tags = []
        for i, id in enumerate(tqdm(data['appid'])):
            new = list(data['tags'][i].split(','))
            for i in new:
                tags.append((id, i.strip("[]''"" ")))

        newdata = pd.DataFrame(tags)
        newdata.to_csv('Resources/Genres.csv')

gt = GameTags()
data = pd.read_csv('Resources/validgameswmods.csv')
data2 = pd.read_csv('Resources/validgames.csv')
conut = data['appid'].unique()
print(conut.shape)
print(data2.shape)

# apps = pd.read_csv('Resources/allgames.csv.gz', compression='gzip')
# gt.getGameTags(apps['appid'])