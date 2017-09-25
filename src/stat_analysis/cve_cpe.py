import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator
import pickle

from sqlalchemy import create_engine

def dateToString(date):
    yearNum = str(date.year)
    monthNum = str(date.month)
    dayNum =str(date.day)
    if len(monthNum)<2:
        monthNum = "0"+monthNum
    if len(dayNum)<2:
        dayNum= "0"+dayNum
    return yearNum+"-"+monthNum+"-"+dayNum

def cveUsers(data):
    cves = data['cve']
    print(len(list(set(cves))))

if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"

    df = pd.read_sql_query(query, con=engine)
    cpe_groups = df['cluster_tag']
    cveUsers(df)

    # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')