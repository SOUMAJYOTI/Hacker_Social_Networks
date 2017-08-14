import sys
sys.path.insert(0, '../load_data/')

import load_data_api as ldap
import pandas as pd
import networkx as nx
import datetime as dt
import operator


def getCPEDesc_MentionsInDW(eventDesc, start_date, end_date):
    results_df = pd.DataFrame()

    postsId_seen = []
    for ed in eventDesc:
        print(ed)
        results = ldap.getHackingPosts_Content(searchContent=ed, fromDate=start_date, toDate=end_date)
        count_wrong = 0
        for r_idx in range(len(results)):
            # try:
            item = results[r_idx]
            if item['postsId'] in postsId_seen:
                continue
            postsId_seen.append(item['postsId'])
            if 'postedDate' not in item:
                item['postedDate'] = ''
            if 'language' not in item:
                item['language'] = ''
            item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                         'topicsId': item['topicId'], 'topicsName': item['topicsName'], 'language': item['language'],
                         'uid': item['uid']}

            item_df = pd.DataFrame(item_dict, index=[r_idx])
            results_df = results_df.append(item_df)
            # except:
            #     count_wrong += 1
            #     print(count_wrong)

    return results_df


def getImpForums(dataDf):
    dataDf['count'] = dataDf.groupby(['forumsId'])['forumsId'].transform('count')

    # forumsSet = df_new[]
    series_count = dataDf.groupby(['forumsId']).agg('count')['uid']

    count_dict = {}
    for ind, val in series_count.iteritems():
        count_dict[ind] = val

    topCount = 10
    sortedForums_count = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
    for f, c in sortedForums_count:
        print(f, c)


def getArmstrongEvent_MentionsInDW(filename, start_date, end_date):
    results_df = pd.DataFrame()
    results = ldap.getHackingPosts_Content(searchContent=filename, fromDate=start_date, toDate=date_event)

    postsId_seen = []
    for r_idx in range(len(results)):
        # try:
        item = results[r_idx]
        if item['postsId'] in postsId_seen:
            continue
        postsId_seen.append(item['postsId'])
        if 'postedDate' not in item:
            item['postedDate'] = ''
        if 'language' not in item:
            item['language'] = ''
        item_dict = {'postsId': item['postsId'], "forumsId": item['forumsId'], 'postedDate': item['postedDate'],
                     'topicsId': item['topicId'], 'topicsName': item['topicsName'],
                     'language': item['language'],
                     'uid': item['uid']}

        item_df = pd.DataFrame(item_dict, index=[r_idx])
        results_df = results_df.append(item_df)

    return  results_df

if __name__ == "__main__":
    # 0. Different event descriptions
    # eventDesc = ['windows 7', 'internet explorer', 'windows vista', 'windows server', 'windows 8']
    # eventDesc = ['linux', 'linux kernel', 'canonical', 'ubuntu', 'ubuntu os']
    # eventDesc = ['apple', 'mac', 'mackintosh', 'mac_os', 'mac operating system']
    # eventDesc = ['google', 'google chrome', 'chrome', 'chrome OS']

    # amEvenets = ['port 80',	'endpoint-malware', 'exploit_kit', 'fireeye_web_mps', 'fireeye_Hx', 'malicious-destination',
    #              'malicious-email',	'malware', 'mcafee_vse', 'pup', 'ransomware', 'trojan', 'virus', 'windows 7']


    # 1. Get the data with CPE desc in Dark Web within a time period
    # start_date = dt.datetime.strptime('2016-03-01', '%Y-%m-%d')
    # end_date = dt.datetime.strptime('2016-03-30', '%Y-%m-%d')
    # results_df = getCPEDesc_MentionsInDW(start_date, end_date)
    # results_df.to_csv('../../data/Armstrong_data/Apple_IE_DW_Mar2015-sample.csv')

    # 2. Get the top k forums in Mar 16 that has these keywords
    # results_df = pd.read_csv('../../data/Armstrong_data/Windows_IE_DW_Mar2015-sample.csv', encoding='ISO-8859-1')
    # getImpForums(results_df)

    # 3. Find the data with Armstrong event description in Darkweb data
    df_path = '../../data/Armstrong_data/eventsDF_v1.0-demo.csv'
    amEvents_df = pd.read_csv(df_path)
    amEvents_df['date'] = pd.to_datetime(amEvents_df['date'])
    amEvents_df = amEvents_df.fillna(value='')

    results_df = []
    filenames_seen = []
    start_date = pd.to_datetime('2016-04-01')
    end_date = pd.to_datetime('2016-06-01')
    amEvents_df_slice = amEvents_df[amEvents_df['date'] < end_date]
    for i, r in amEvents_df_slice.iterrows():
        date_event = r['date']
        filename = r['filename']

        if filename in filenames_seen and filename != '':
            continue

        filenames_seen.append(filename)
        print(date_event, filename)

        r = getArmstrongEvent_MentionsInDW(filename, start_date, date_event)
        results_df.append(r)

    print(results_df)
