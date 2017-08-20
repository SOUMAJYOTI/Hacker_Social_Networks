import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator


def getTopicIds(forumsList, start_date, end_date):
    keyword_List = ['windows', 'microsoft', 'vista', 'windows xp', 'win', 'windows 8']
    for f in forumsList:
        # for kw in keyword_List:
        results = ldap.getHackingPosts_Content(fId=f, fromDate=start_date, toDate=end_date, limNum=5000)

if __name__ == "__main__":
    event_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Jan15-Mar16.csv', encoding='ISO-8859-1')
    forumsList = list(set(event_df['forumsId'].tolist()))


