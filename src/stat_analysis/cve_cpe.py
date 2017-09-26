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


def user_CVE_groups(cve_cpe_data, vul_data):
    usersCVE_map = {}
    CVE_usersMap = {}
    for cve in cve_cpe_data['cve']:
        vulnItems = vul_data[vul_data['vulnId'] == cve]
        users = vulnItems['users'].tolist()
        if cve not in CVE_usersMap:
            CVE_usersMap[cve] = []
        CVE_usersMap[cve].extend(users)

        if len(users) == 0:
            continue

        usersList = users[0]
        for u in usersList:
            # print(u)
            if u not in usersCVE_map:
                usersCVE_map[u] = []

            usersCVE_map[u].append(cve)

    return usersCVE_map, CVE_usersMap


if __name__ == "__main__":
    engine = create_engine('postgresql://postgres:Impossible2@10.218.109.4:5432/cyber_events_pred')
    query = "select vendor, product, cluster_tag, cve from  cve_cpegroups"
    posts_df = pickle.load(open('../../data/DW_data/09_15/train/data/DW_data_selected_forums_Oct15-Mar16.pickle', 'rb'))
    vulData = pickle.load(open('../../data/DW_data/08_29/Vulnerabilities-sample_v2+.pickle', 'rb'))

    # df = pd.read_sql_query(query, con=engine)
    # df.to_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    df = pd.read_csv('../../data/DW_data/09_15/CVE_CPE_groups.csv')
    print("Number of cluster tags: ", len(list(set(df['cluster_tag']))))

    users_CVE_map, CVE_users_map = user_CVE_groups(df, vulData)
    print(len(users_CVE_map))
    # cpe_groups = df['cluster_tag']
    # print(cpe_groups)
    # cveUsers(df)

    # results_df.to_csv('../../data/DW_data/08_20/DW_data_selected_forums_Jul16.csv')