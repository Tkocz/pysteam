from bs4 import BeautifulSoup
import mechanize
import pandas as pd
from tqdm import *

class GameTags():

    def __init__(self):
        self.mb = mechanize.Browser()


    def getGameTags(self, appids):
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
                tag_list = [str(t.text.strip())
                            for t in tags.findAll("a", {"href": True})]
            tag_df = tag_df.append(pd.DataFrame([[appid, tag_list]]))
        tag_df.columns = ['appid', 'tags']
        tag_df = tag_df[tag_df.tags != 0]
        print(tag_df)
        tag_df.to_csv('Resources/gamegenres.csv.gz', compression='gzip')

gt = GameTags()

apps = pd.read_csv('Resources/gamegenres.csv.gz', compression='gzip', usecols=['appid', 'tags'])
print(apps)
value = apps[(apps.appid == 20)]
print(value['tags'])
